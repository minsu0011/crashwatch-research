from __future__ import annotations

import gc
import logging
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .folds import FoldSlice, inner_tuning_slices
from .metrics import scoped_metric_rows
from .utils import atomic_json, canonical_json_hash, ensure_thread_env, hash_strings, read_json

LOGGER = logging.getLogger(__name__)
MODEL_CACHE_VERSION = "cw3h_lgb_v3"


@dataclass(frozen=True)
class TuneTask:
    cache_root: str
    profile: str
    fold: dict[str, Any]
    dataset_signature: str
    lightgbm_config: dict[str, Any]
    threads: int
    inner_validation_days: int
    inner_purge_days: int
    min_train_days: int
    seed: int = 17


@dataclass(frozen=True)
class BlockTask:
    cache_root: str
    profile: str
    condition: str
    phase: str
    fold: dict[str, Any]
    seeds: list[int]
    best_iteration: int
    dataset_signature: str
    lightgbm_config: dict[str, Any]
    threads: int
    deadline_epoch: float
    save_predictions: bool
    prediction_compression: str


METRIC_BASE_FIELDS = [
    "task_id", "family", "backend", "phase", "profile", "condition", "outer_fold", "seed",
    "scope_type", "scope_value", "feature_count", "feature_hash", "best_iteration", "elapsed_seconds",
    "train_rows", "validation_rows", "train_date_min", "train_date_max", "validation_date_min", "validation_date_max",
    "rows", "positives", "positive_rate", "raw_pr_auc", "raw_roc_auc", "raw_brier", "raw_logloss",
    "raw_mean_prediction", "balanced_accuracy", "raw_pr_auc_lift", "top_1pct_precision", "top_1pct_recall",
    "top_3pct_precision", "top_3pct_recall", "top_5pct_precision", "top_5pct_recall", "prediction_path",
]


def _lgb_params(config: dict[str, Any], y_train: np.ndarray, seed: int, threads: int) -> dict[str, Any]:
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    scale_pos_weight = max(1.0, negatives / max(1, positives))
    params = {
        "objective": config.get("objective", "binary"),
        "metric": "average_precision",
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "max_depth": int(config["max_depth"]),
        "min_data_in_leaf": int(config["min_data_in_leaf"]),
        "feature_fraction": float(config["feature_fraction"]),
        "bagging_fraction": float(config.get("bagging_fraction", 1.0)),
        "bagging_freq": int(config.get("bagging_freq", 0)),
        "lambda_l1": float(config["lambda_l1"]),
        "lambda_l2": float(config["lambda_l2"]),
        "max_bin": int(config["max_bin"]),
        "scale_pos_weight": scale_pos_weight,
        "verbosity": -1,
        "num_threads": int(threads),
        "device_type": "cpu",
        "deterministic": bool(config.get("deterministic", True)),
        "force_col_wise": bool(config.get("force_col_wise", True)),
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        # Dataset is deliberately reused across seeds. Keep bin sampling deterministic.
        "data_random_seed": 17,
        "feature_pre_filter": False,
    }
    return params


def tune_best_iteration(task: TuneTask) -> dict[str, Any]:
    ensure_thread_env(task.threads)
    import lightgbm as lgb

    cache_root = Path(task.cache_root)
    profile_dir = cache_root / "profiles" / task.profile
    profile_manifest = read_json(profile_dir / "profile_manifest.json")
    feature_hash = profile_manifest["conditions"]["B0"]["feature_hash"]
    fold = FoldSlice(**task.fold)
    key_payload = {
        "kind": "lightgbm_tune",
        "runner_version": MODEL_CACHE_VERSION,
        "dataset_signature": task.dataset_signature,
        "profile": task.profile,
        "feature_hash": feature_hash,
        "fold": task.fold,
        "model": task.lightgbm_config,
        "seed": task.seed,
        "inner_validation_days": task.inner_validation_days,
        "inner_purge_days": task.inner_purge_days,
        "min_train_days": task.min_train_days,
        "threads_semantics_excluded": True,
        "backend": "cpu",
    }
    task_id = canonical_json_hash(key_payload)
    result_dir = cache_root / "tuning_cache"
    result_path = result_dir / f"{task_id}.json"
    if result_path.exists():
        cached = read_json(result_path)
        if cached.get("status") == "completed":
            cached["cache_status"] = "hit"
            return cached

    started = time.perf_counter()
    try:
        dates_ns = np.load(profile_dir / "dates_ns.npy", mmap_mode="r")
        y = np.load(profile_dir / "target.npy", mmap_mode="r")
        x = np.load(profile_dir / "matrices" / "B0.npy", mmap_mode="r")
        inner = inner_tuning_slices(
            dates_ns,
            fold,
            task.inner_validation_days,
            task.inner_purge_days,
            task.min_train_days,
        )
        if inner is None:
            best_iteration = min(600, int(task.lightgbm_config["max_rounds"]))
            result = {
                "status": "completed",
                "task_id": task_id,
                "profile": task.profile,
                "outer_fold": fold.fold_id,
                "best_iteration": best_iteration,
                "fallback": True,
                "fallback_reason": "insufficient_inner_dates",
                "elapsed_seconds": time.perf_counter() - started,
                "cache_status": "miss",
            }
            atomic_json(result, result_path)
            return result
        train_slice, valid_slice = inner
        x_train = x[train_slice]
        y_train = np.asarray(y[train_slice], dtype=np.uint8)
        x_valid = x[valid_slice]
        y_valid = np.asarray(y[valid_slice], dtype=np.uint8)
        train_set = lgb.Dataset(x_train, label=y_train, free_raw_data=False, params={"max_bin": int(task.lightgbm_config["max_bin"])})
        valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, free_raw_data=False)
        params = _lgb_params(task.lightgbm_config, y_train, task.seed, task.threads)
        model = lgb.train(
            params,
            train_set,
            num_boost_round=int(task.lightgbm_config["max_rounds"]),
            valid_sets=[valid_set],
            valid_names=["inner_valid"],
            callbacks=[
                lgb.early_stopping(int(task.lightgbm_config["early_stopping_rounds"]), first_metric_only=True, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        best_iteration = int(model.best_iteration or task.lightgbm_config["max_rounds"])
        result = {
            "status": "completed",
            "task_id": task_id,
            "profile": task.profile,
            "outer_fold": fold.fold_id,
            "best_iteration": best_iteration,
            "fallback": False,
            "inner_train_rows": int(len(y_train)),
            "inner_validation_rows": int(len(y_valid)),
            "elapsed_seconds": time.perf_counter() - started,
            "cache_status": "miss",
        }
        atomic_json(result, result_path)
        del model, train_set, valid_set, x_train, x_valid, y_train, y_valid
        gc.collect()
        return result
    except Exception as exc:
        result = {
            "status": "failed",
            "task_id": task_id,
            "profile": task.profile,
            "outer_fold": fold.fold_id,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
            "cache_status": "miss",
        }
        atomic_json(result, result_path)
        return result


def _task_identity(task: BlockTask, seed: int, feature_hash: str) -> tuple[str, dict[str, Any]]:
    payload = {
        "kind": "lightgbm_outer_prediction",
        "runner_version": MODEL_CACHE_VERSION,
        "dataset_signature": task.dataset_signature,
        "family": "lightgbm",
        "backend": "cpu",
        "profile": task.profile,
        "condition": task.condition,
        "feature_hash": feature_hash,
        "fold": task.fold,
        "seed": int(seed),
        "best_iteration": int(task.best_iteration),
        "model": task.lightgbm_config,
        # Thread count intentionally excluded: deterministic CPU predictions must be equivalent.
        "cache_contract": "dataset,target,fold,purge,feature_hash,seed,family,config,best_iteration,backend",
    }
    return canonical_json_hash(payload), payload


def run_lightgbm_block(task: BlockTask) -> dict[str, Any]:
    ensure_thread_env(task.threads)
    import lightgbm as lgb

    cache_root = Path(task.cache_root)
    profile_dir = cache_root / "profiles" / task.profile
    profile_manifest = read_json(profile_dir / "profile_manifest.json")
    condition_meta = profile_manifest["conditions"][task.condition]
    feature_hash = condition_meta["feature_hash"]
    feature_count = len(condition_meta["features"])
    fold = FoldSlice(**task.fold)
    result_root = cache_root / "task_results" / "lightgbm"
    metric_dir = result_root / "metrics"
    prediction_dir = result_root / "predictions"
    metric_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)

    completed: list[dict[str, Any]] = []
    missing: list[tuple[int, str, Path, Path, dict[str, Any]]] = []
    for seed in task.seeds:
        task_id, identity = _task_identity(task, seed, feature_hash)
        metric_path = metric_dir / f"{task_id}.json"
        pred_path = prediction_dir / f"{task_id}.parquet"
        if metric_path.exists():
            cached = read_json(metric_path)
            prediction_ok = (not task.save_predictions) or (pred_path.exists() and pred_path.stat().st_size > 0)
            if cached.get("status") == "completed" and prediction_ok:
                cached["cache_status"] = "hit"
                completed.append(cached)
                continue
        missing.append((int(seed), task_id, metric_path, pred_path, identity))

    if not missing:
        return {"status": "completed", "cache_only": True, "block": asdict(task), "tasks": completed}
    if time.time() >= task.deadline_epoch:
        return {"status": "deadline_skipped", "cache_only": False, "block": asdict(task), "tasks": completed}
    if not fold.eligible:
        failures = []
        for seed, task_id, metric_path, _, _ in missing:
            _, identity = _task_identity(task, seed, feature_hash)
            failure = {"status": "ineligible", "task_id": task_id, "identity": identity, "profile": task.profile, "condition": task.condition, "outer_fold": fold.fold_id, "seed": seed, "error": fold.reason}
            atomic_json(failure, metric_path)
            failures.append(failure)
        return {"status": "ineligible", "cache_only": False, "block": asdict(task), "tasks": completed + failures}

    x = np.load(profile_dir / "matrices" / f"{task.condition}.npy", mmap_mode="r")
    y = np.load(profile_dir / "target.npy", mmap_mode="r")
    dates_ns = np.load(profile_dir / "dates_ns.npy", mmap_mode="r")
    tickers = np.load(profile_dir / "tickers.npy", mmap_mode="r")
    buckets = np.load(profile_dir / "buckets.npy", mmap_mode="r")
    row_ids = np.load(profile_dir / "original_row_id.npy", mmap_mode="r")

    train_slice = slice(fold.train_start, fold.train_stop)
    valid_slice = slice(fold.validation_start, fold.validation_stop)
    x_train = x[train_slice]
    y_train = np.asarray(y[train_slice], dtype=np.uint8)
    x_valid = x[valid_slice]
    y_valid = np.asarray(y[valid_slice], dtype=np.uint8)
    valid_dates = np.asarray(dates_ns[valid_slice], dtype=np.int64)
    valid_tickers = np.asarray(tickers[valid_slice]).astype(str)
    valid_buckets = np.asarray(buckets[valid_slice]).astype(str)
    valid_row_ids = np.asarray(row_ids[valid_slice], dtype=np.int64)

    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        free_raw_data=False,
        params={"max_bin": int(task.lightgbm_config["max_bin"]), "feature_pre_filter": False, "data_random_seed": 17},
    )
    # The expensive bin construction is performed once and reused for every seed in this block.
    train_set.construct()

    for seed, task_id, metric_path, pred_path, identity in missing:
        if time.time() >= task.deadline_epoch:
            skipped = {"status": "deadline_skipped", "task_id": task_id, "identity": identity, "profile": task.profile, "condition": task.condition, "outer_fold": fold.fold_id, "seed": seed, "error": "hard deadline reached before seed start"}
            atomic_json(skipped, metric_path)
            completed.append(skipped)
            continue
        started = time.perf_counter()
        try:
            params = _lgb_params(task.lightgbm_config, y_train, seed, task.threads)
            model = lgb.train(params, train_set, num_boost_round=int(task.best_iteration), callbacks=[lgb.log_evaluation(period=0)])
            pred = np.asarray(model.predict(x_valid, num_iteration=int(task.best_iteration)), dtype=np.float32)
            include_battery = task.profile == "common_period" and task.condition in {"B0", "S1", "S2", "S3"}
            scope_rows = scoped_metric_rows(y_valid, pred, valid_dates, valid_buckets, include_battery)
            elapsed = time.perf_counter() - started
            if task.save_predictions:
                prediction_frame = pd.DataFrame({
                    "original_row_id": valid_row_ids,
                    "date": pd.to_datetime(valid_dates),
                    "ticker": valid_tickers,
                    "bucket": valid_buckets,
                    "target": y_valid,
                    "prediction": pred,
                })
                temp_path = pred_path.with_suffix(".tmp.parquet")
                prediction_frame.to_parquet(temp_path, index=False, compression=task.prediction_compression, engine="pyarrow")
                os.replace(temp_path, pred_path)
            else:
                pred_path = Path("")
            metric_rows = []
            for scope in scope_rows:
                metric_rows.append({
                    "task_id": task_id,
                    "family": "lightgbm",
                    "backend": "cpu",
                    "phase": task.phase,
                    "profile": task.profile,
                    "condition": task.condition,
                    "outer_fold": fold.fold_id,
                    "seed": seed,
                    "feature_count": feature_count,
                    "feature_hash": feature_hash,
                    "best_iteration": int(task.best_iteration),
                    "elapsed_seconds": elapsed,
                    "train_rows": int(len(y_train)),
                    "validation_rows": int(len(y_valid)),
                    "train_date_min": fold.train_date_min,
                    "train_date_max": fold.train_date_max,
                    "validation_date_min": fold.validation_date_min,
                    "validation_date_max": fold.validation_date_max,
                    "prediction_path": str(pred_path) if task.save_predictions else "",
                    **scope,
                })
            result = {
                "status": "completed",
                "task_id": task_id,
                "identity": identity,
                "cache_status": "miss",
                "metrics": metric_rows,
                "prediction_path": str(pred_path) if task.save_predictions else "",
                "elapsed_seconds": elapsed,
            }
            atomic_json(result, metric_path)
            completed.append(result)
            del model, pred
            gc.collect()
        except Exception as exc:
            pred_path.with_suffix(".tmp.parquet").unlink(missing_ok=True)
            failure = {
                "status": "failed",
                "task_id": task_id,
                "identity": identity,
                "profile": task.profile,
                "condition": task.condition,
                "outer_fold": fold.fold_id,
                "seed": seed,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }
            atomic_json(failure, metric_path)
            completed.append(failure)

    del train_set, x_train, x_valid, y_train, y_valid, x, y, dates_ns, tickers, buckets, row_ids
    gc.collect()
    failed_count = sum(item.get("status") == "failed" for item in completed)
    return {"status": "failed" if failed_count else "completed", "cache_only": False, "block": asdict(task), "tasks": completed}
