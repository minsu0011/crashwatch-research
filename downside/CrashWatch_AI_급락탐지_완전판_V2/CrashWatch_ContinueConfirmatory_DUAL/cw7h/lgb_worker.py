from __future__ import annotations

import gc
import logging
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .folds import FoldSlice, rolling_inner_slices
from .metrics import compute_metrics
from .utils import atomic_json, canonical_hash, ensure_thread_env, hash_strings, read_json

LOGGER = logging.getLogger(__name__)
MODEL_CACHE_VERSION = "cw7h_lgb_feature_contri_v1"


@dataclass(frozen=True)
class ModelCondition:
    condition_id: str
    test_type: str
    feature: str
    feature_index: int
    cluster_id: int | None
    enabled_indices: tuple[int, ...] | None
    save_prediction: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "test_type": self.test_type,
            "feature": self.feature,
            "feature_index": self.feature_index,
            "cluster_id": self.cluster_id,
            "enabled_indices": list(self.enabled_indices) if self.enabled_indices is not None else None,
            "save_prediction": self.save_prediction,
        }


@dataclass(frozen=True)
class LgbBlock:
    cache_root: str
    output_dir: str
    dataset_signature: str
    run_signature: str
    fold: dict[str, Any]
    conditions: tuple[ModelCondition, ...]
    best_iteration: int
    model_config: dict[str, Any]
    threads: int
    deadline_epoch: float
    feature_names: tuple[str, ...]
    stage: str


def _base_params(config: dict[str, Any], y_train: np.ndarray, threads: int) -> dict[str, Any]:
    positives = max(1, int(np.sum(y_train == 1)))
    negatives = max(1, int(np.sum(y_train == 0)))
    return {
        "objective": "binary",
        "metric": "None",
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "max_depth": int(config.get("max_depth", -1)),
        "min_data_in_leaf": int(config["min_data_in_leaf"]),
        "feature_fraction": 1.0,
        "feature_fraction_bynode": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": float(config.get("lambda_l1", 0.0)),
        "lambda_l2": float(config.get("lambda_l2", 0.0)),
        "max_bin": int(config.get("max_bin", 255)),
        "min_gain_to_split": float(config.get("min_gain_to_split", 1e-12)),
        "scale_pos_weight": float(negatives / positives),
        "num_threads": int(threads),
        "deterministic": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "verbosity": -1,
        "seed": 17,
        "feature_fraction_seed": 17,
        "bagging_seed": 17,
        "data_random_seed": 17,
        "drop_seed": 17,
    }


def tune_fold_iteration(task: dict[str, Any]) -> dict[str, Any]:
    ensure_thread_env(int(task["threads"]))
    import lightgbm as lgb
    from sklearn.metrics import average_precision_score

    cache_root = Path(task["cache_root"])
    output_dir = Path(task["output_dir"])
    fold = FoldSlice(**task["fold"])
    result_path = output_dir / "tuning" / f"fold_{fold.fold_id}.json"
    identity = {
        "version": MODEL_CACHE_VERSION,
        "dataset_signature": task["dataset_signature"],
        "fold": task["fold"],
        "model_config": task["model_config"],
        "rolling_windows": task["rolling_windows"],
        "rolling_step_days": task["rolling_step_days"],
        "inner_validation_days": task["inner_validation_days"],
        "inner_purge_days": task["inner_purge_days"],
        "min_train_days": task["min_train_days"],
    }
    identity_hash = canonical_hash(identity)
    if result_path.exists():
        cached = read_json(result_path)
        if cached.get("status") == "completed" and cached.get("identity_hash") == identity_hash:
            cached["cache_status"] = "hit"
            return cached

    started = time.perf_counter()
    try:
        X = np.load(cache_root / "X_all_valid.npy", mmap_mode="r")
        y = np.load(cache_root / "target.npy", mmap_mode="r")
        dates = np.load(cache_root / "dates_ns.npy", mmap_mode="r")
        inner_slices = rolling_inner_slices(
            dates,
            fold,
            validation_days=int(task["inner_validation_days"]),
            purge_days=int(task["inner_purge_days"]),
            min_train_days=int(task["min_train_days"]),
            windows=int(task["rolling_windows"]),
            step_days=int(task["rolling_step_days"]),
        )
        rounds: list[int] = []
        window_rows: list[dict[str, Any]] = []
        for window_id, (train_slice, valid_slice) in enumerate(inner_slices):
            x_train = np.asarray(X[train_slice], dtype=np.float32)
            y_train = np.asarray(y[train_slice], dtype=np.uint8)
            x_valid = np.asarray(X[valid_slice], dtype=np.float32)
            y_valid = np.asarray(y[valid_slice], dtype=np.uint8)
            train_set = lgb.Dataset(
                x_train,
                label=y_train,
                free_raw_data=False,
                params={"max_bin": int(task["model_config"].get("max_bin", 255)), "feature_pre_filter": False, "data_random_seed": 17},
            )
            valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, free_raw_data=False)
            params = _base_params(task["model_config"], y_train, int(task["threads"]))
            model = lgb.train(
                params,
                train_set,
                num_boost_round=int(task["model_config"]["max_rounds"]),
                valid_sets=[valid_set],
                valid_names=["inner_valid"],
                callbacks=[
                    lgb.early_stopping(int(task["model_config"]["early_stopping_rounds"]), first_metric_only=True, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
                feval=lambda pred, data: ("aucpr", average_precision_score(data.get_label(), pred), True),
            )
            best = int(model.best_iteration or task["model_config"]["max_rounds"])
            rounds.append(best)
            window_rows.append({
                "window_id": window_id,
                "best_iteration": best,
                "train_rows": int(len(y_train)),
                "validation_rows": int(len(y_valid)),
            })
            del model, train_set, valid_set, x_train, y_train, x_valid, y_valid
            gc.collect()
        if rounds:
            chosen = int(np.median(rounds))
            chosen = max(int(task["model_config"].get("min_effective_rounds", 40)), chosen)
            chosen = min(int(task["model_config"]["max_rounds"]), chosen)
            fallback = False
        else:
            chosen = int(task["model_config"].get("fallback_rounds", 240))
            fallback = True
        result = {
            "status": "completed",
            "identity_hash": identity_hash,
            "cache_status": "miss",
            "fold_id": fold.fold_id,
            "best_iteration": chosen,
            "window_best_iterations": rounds,
            "windows": window_rows,
            "fallback": fallback,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(result, result_path)
        return result
    except Exception as exc:
        result = {
            "status": "failed",
            "identity_hash": identity_hash,
            "fold_id": fold.fold_id,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(result, result_path)
        return result


def _feature_contri(feature_count: int, condition: ModelCondition) -> list[float] | None:
    if condition.test_type == "baseline_all_valid":
        return None
    values = np.zeros(feature_count, dtype=np.float64)
    if condition.enabled_indices is None:
        values.fill(1.0)
        if 0 <= condition.feature_index < feature_count:
            values[condition.feature_index] = 0.0
    else:
        if len(condition.enabled_indices):
            values[np.asarray(condition.enabled_indices, dtype=np.int32)] = 1.0
    return values.tolist()


def _condition_identity(block: LgbBlock, condition: ModelCondition) -> tuple[str, dict[str, Any]]:
    enabled_hash = "all_except_one" if condition.enabled_indices is None else hash_strings(map(str, condition.enabled_indices))
    payload = {
        "version": MODEL_CACHE_VERSION,
        "dataset_signature": block.dataset_signature,
        "run_signature": block.run_signature,
        "backend": "lightgbm_cpu",
        "fold": block.fold,
        "stage": block.stage,
        "condition": condition.to_dict() | {"enabled_indices": enabled_hash},
        "best_iteration": block.best_iteration,
        "model_config": block.model_config,
    }
    return canonical_hash(payload), payload


def run_lgb_block(block: LgbBlock) -> dict[str, Any]:
    ensure_thread_env(block.threads)
    import lightgbm as lgb

    cache_root = Path(block.cache_root)
    output_dir = Path(block.output_dir)
    fold = FoldSlice(**block.fold)
    result_dir = output_dir / "task_results" / "lightgbm_cpu"
    pred_dir = output_dir / "predictions" / "lightgbm_cpu" / f"fold_{fold.fold_id}"
    result_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[ModelCondition, str, Path, dict[str, Any]]] = []
    cached_count = 0
    for condition in block.conditions:
        task_id, identity = _condition_identity(block, condition)
        result_path = result_dir / f"{task_id}.json"
        if result_path.exists():
            cached = read_json(result_path)
            if cached.get("status") == "completed" and cached.get("identity") == identity:
                cached_count += 1
                continue
        pending.append((condition, task_id, result_path, identity))
    if not pending:
        return {"status": "completed", "cached": cached_count, "trained": 0, "failed": 0, "stage": block.stage, "fold_id": fold.fold_id}
    if time.time() >= block.deadline_epoch:
        return {"status": "deadline_skipped", "cached": cached_count, "trained": 0, "failed": 0, "stage": block.stage, "fold_id": fold.fold_id}

    X = np.load(cache_root / "X_all_valid.npy", mmap_mode="r")
    y = np.load(cache_root / "target.npy", mmap_mode="r")
    dates = np.load(cache_root / "dates_ns.npy", mmap_mode="r")
    row_ids = np.load(cache_root / "original_row_id.npy", mmap_mode="r")
    train_slice = slice(fold.train_start, fold.train_stop)
    valid_slice = slice(fold.validation_start, fold.validation_stop)
    x_train = np.asarray(X[train_slice], dtype=np.float32)
    y_train = np.asarray(y[train_slice], dtype=np.uint8)
    x_valid = np.asarray(X[valid_slice], dtype=np.float32)
    y_valid = np.asarray(y[valid_slice], dtype=np.uint8)
    valid_dates = np.asarray(dates[valid_slice], dtype=np.int64)
    valid_row_ids = np.asarray(row_ids[valid_slice], dtype=np.int64)
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        feature_name=list(block.feature_names),
        free_raw_data=False,
        params={"max_bin": int(block.model_config.get("max_bin", 255)), "feature_pre_filter": False, "data_random_seed": 17},
    )
    train_set.construct()
    base_params = _base_params(block.model_config, y_train, block.threads)
    trained = 0
    failed = 0
    deadline_skipped = 0
    elapsed_values: list[float] = []
    for condition, task_id, result_path, identity in pending:
        if time.time() >= block.deadline_epoch:
            deadline_skipped += 1
            continue
        started = time.perf_counter()
        try:
            params = dict(base_params)
            contrib = _feature_contri(len(block.feature_names), condition)
            if contrib is not None:
                params["feature_contri"] = contrib
            model = lgb.train(
                params,
                train_set,
                num_boost_round=int(block.best_iteration),
                callbacks=[lgb.log_evaluation(period=0)],
            )
            pred = np.asarray(model.predict(x_valid, num_iteration=int(block.best_iteration)), dtype=np.float32)
            elapsed = time.perf_counter() - started
            metrics = compute_metrics(y_valid, pred, valid_dates)
            prediction_path = ""
            if condition.save_prediction:
                prediction_path_obj = pred_dir / f"{task_id}.npz"
                temp_path = prediction_path_obj.with_suffix(".tmp.npz")
                np.savez_compressed(temp_path, row_id=valid_row_ids, target=y_valid, prediction=pred, date_ns=valid_dates)
                os.replace(temp_path, prediction_path_obj)
                prediction_path = str(prediction_path_obj)
            result = {
                "status": "completed",
                "task_id": task_id,
                "identity": identity,
                "backend": "lightgbm_cpu",
                "dataset_signature": block.dataset_signature,
                "run_signature": block.run_signature,
                "stage": block.stage,
                "test_type": condition.test_type,
                "condition_id": condition.condition_id,
                "feature": condition.feature,
                "feature_index": condition.feature_index,
                "cluster_id": condition.cluster_id,
                "outer_fold": fold.fold_id,
                "best_iteration": int(block.best_iteration),
                "feature_count_total": len(block.feature_names),
                "enabled_feature_count": len(condition.enabled_indices) if condition.enabled_indices is not None else len(block.feature_names) - (0 if condition.test_type == "baseline_all_valid" else 1),
                "train_rows": int(len(y_train)),
                "validation_rows": int(len(y_valid)),
                "train_date_min": fold.train_date_min,
                "train_date_max": fold.train_date_max,
                "validation_date_min": fold.validation_date_min,
                "validation_date_max": fold.validation_date_max,
                "elapsed_seconds": elapsed,
                "prediction_path": prediction_path,
                **metrics,
            }
            atomic_json(result, result_path)
            trained += 1
            elapsed_values.append(elapsed)
            del model, pred
        except Exception as exc:
            failed += 1
            atomic_json({
                "status": "failed",
                "task_id": task_id,
                "identity": identity,
                "dataset_signature": block.dataset_signature,
                "run_signature": block.run_signature,
                "stage": block.stage,
                "test_type": condition.test_type,
                "condition_id": condition.condition_id,
                "feature": condition.feature,
                "outer_fold": fold.fold_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.perf_counter() - started,
            }, result_path)
        if (trained + failed) % 12 == 0:
            gc.collect()
    del train_set, x_train, y_train, x_valid, y_valid, X, y, dates, row_ids
    gc.collect()
    return {
        "status": "failed" if failed else ("partial" if deadline_skipped else "completed"),
        "cached": cached_count,
        "trained": trained,
        "failed": failed,
        "deadline_skipped": deadline_skipped,
        "mean_model_seconds": float(np.mean(elapsed_values)) if elapsed_values else float("nan"),
        "stage": block.stage,
        "fold_id": fold.fold_id,
    }
