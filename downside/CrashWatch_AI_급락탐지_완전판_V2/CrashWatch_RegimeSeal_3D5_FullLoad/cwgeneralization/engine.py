from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from cw7h.utils import atomic_json, read_json
from cwfull.common import set_worker_mode
from .core import canonical_hash


def _checkpoint_paths(output: Path, stage: str, model: str, plan_id: str) -> tuple[Path, Path]:
    root = output / "checkpoints" / stage / model
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{plan_id}.json", root / f"{plan_id}.npz"


def make_plan(
    output: Path,
    stage: str,
    candidate: dict[str, Any],
    fold: dict[str, Any],
    model: str,
    seeds: list[int],
    meta: dict[str, Any],
    config: dict[str, Any],
    *,
    exclude_train_regime: int | None = None,
    validation_regime: int | None = None,
    exclude_train_sector: int | None = None,
    validation_sector: int | None = None,
    block_permutation: int = 0,
) -> dict[str, Any]:
    profile_key = config["models"][candidate["profile"]]["profile"]
    model_config = config["models"][candidate["profile"]][model]
    identity = {
        "schema": "crashwatch_generalization_bundle_v1",
        "runtime_hash": meta["runtime_hash"], "stage": stage,
        "candidate": candidate, "fold_id": int(fold["fold_id"]), "model": model,
        "model_config": model_config, "seeds": list(map(int, seeds)),
        "exclude_train_regime": exclude_train_regime, "validation_regime": validation_regime,
        "exclude_train_sector": exclude_train_sector, "validation_sector": validation_sector,
        "block_permutation": int(block_permutation),
    }
    plan_id = canonical_hash(identity, 28)
    json_path, npz_path = _checkpoint_paths(output, stage, model, plan_id)
    return {
        **identity, "plan_id": plan_id, "json_path": str(json_path), "npz_path": str(npz_path),
        "profile_key": profile_key, "matrix_path": meta["profile_paths"][profile_key],
        "target_path": meta["target_path"], "valid_path": meta["valid_path"],
        "first_hit_path": meta["first_hit_path"], "dates_path": meta["dates_path"],
        "tickers_path": meta["tickers_path"], "date_index_path": meta["date_index_path"],
        "regime_code_path": meta["regime_code_path"], "sector_code_path": meta["sector_code_path"],
        "train_start_ns": int(fold["train_start_ns"]), "train_end_ns": int(fold["train_end_ns"]),
        "validation_start_ns": int(fold["validation_start_ns"]), "validation_end_ns": int(fold["validation_end_ns"]),
        "threads": int(config["resources"][f"{model}_threads_per_worker"]),
        "priority": str(config["resources"].get("priority", "normal")),
        "gpu_duty_cycle": float(config["resources"].get("xgboost_gpu_duty_cycle", 1.0)),
        "random_control_features": int(config["robustness"]["random_control_features"]),
    }


def _indices_and_weights(plan: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = np.load(plan["dates_path"], mmap_mode="r")
    valid = np.load(plan["valid_path"], mmap_mode="r").astype(bool)
    date_index = np.load(plan["date_index_path"], mmap_mode="r")
    regimes = np.load(plan["regime_code_path"], mmap_mode="r")
    sectors = np.load(plan["sector_code_path"], mmap_mode="r")
    train_mask = valid & (dates >= int(plan["train_start_ns"])) & (dates <= int(plan["train_end_ns"]))
    validation_mask = valid & (dates >= int(plan["validation_start_ns"])) & (dates <= int(plan["validation_end_ns"]))
    candidate = plan["candidate"]
    window = int(candidate.get("train_window_days", 0))
    if window > 0:
        train_date_values = np.unique(dates[train_mask])
        if len(train_date_values) > window:
            train_mask &= dates >= train_date_values[-window]
    if plan.get("exclude_train_regime") is not None:
        train_mask &= regimes != int(plan["exclude_train_regime"])
    if plan.get("validation_regime") is not None:
        validation_mask &= regimes == int(plan["validation_regime"])
    if plan.get("exclude_train_sector") is not None:
        train_mask &= sectors != int(plan["exclude_train_sector"])
    if plan.get("validation_sector") is not None:
        validation_mask &= sectors == int(plan["validation_sector"])
    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(validation_mask)
    if len(train_idx) < 1000 or len(val_idx) < 40:
        raise RuntimeError(f"insufficient rows train={len(train_idx)} validation={len(val_idx)}")
    weighting = str(candidate.get("weighting", "none"))
    weights = np.ones(len(train_idx), dtype=np.float32)
    age = date_index[train_idx].max() - date_index[train_idx]
    if weighting == "recent_hl250":
        weights *= np.power(0.5, age / 250.0).astype(np.float32)
    elif weighting in {"recent_hl500", "positive_recent_hybrid"}:
        weights *= np.power(0.5, age / 500.0).astype(np.float32)
    elif weighting == "rare_regime":
        counts = np.bincount(regimes[train_idx].astype(np.int64), minlength=8)
        scale = np.sqrt(max(1, counts.max()) / np.maximum(1, counts))
        weights *= scale[regimes[train_idx]].astype(np.float32)
    elif weighting != "none":
        raise ValueError(f"unknown weighting: {weighting}")
    if weighting == "positive_recent_hybrid":
        y = np.load(plan["target_path"], mmap_mode="r")
        weights *= np.where(y[train_idx] == 1, 1.25, 1.0).astype(np.float32)
    weights = np.clip(weights / max(1e-6, float(weights.mean())), 0.10, 5.0).astype(np.float32)
    return train_idx, val_idx, weights


def _control_features(indices: np.ndarray, count: int) -> np.ndarray:
    row = np.asarray(indices, dtype=np.uint64)[:, None]
    col = np.arange(1, count + 1, dtype=np.uint64)[None, :]
    value = (row * np.uint64(6364136223846793005) + col * np.uint64(1442695040888963407))
    value ^= value >> np.uint64(33)
    return ((value & np.uint64(0xFFFFFF)).astype(np.float32) / float(0xFFFFFF) - 0.5).astype(np.float32)


def _matrix(plan: dict[str, Any], indices: np.ndarray) -> np.ndarray:
    base = np.load(plan["matrix_path"], mmap_mode="r")
    result = np.asarray(base[indices], dtype=np.float32)
    extras = list(plan["candidate"].get("extras", []))
    additions = []
    if "regime" in extras:
        code = np.load(plan["regime_code_path"], mmap_mode="r")[indices].astype(np.int64)
        additions.append(np.eye(8, dtype=np.float32)[code])
    if "sector" in extras:
        code = np.load(plan["sector_code_path"], mmap_mode="r")[indices].astype(np.int64)
        additions.append(np.eye(8, dtype=np.float32)[code])
    if "random_control" in extras:
        additions.append(_control_features(indices, int(plan["random_control_features"])))
    if additions:
        result = np.column_stack([result, *additions]).astype(np.float32, copy=False)
    result[~np.isfinite(result)] = np.nan
    return result


def _permuted_labels(plan: dict[str, Any], train_idx: np.ndarray, y: np.ndarray) -> np.ndarray:
    permutation = int(plan.get("block_permutation", 0))
    if permutation <= 0:
        return y
    tickers = np.load(plan["tickers_path"], mmap_mode="r")[train_idx].astype(str)
    dates = np.load(plan["dates_path"], mmap_mode="r")[train_idx]
    result = y.copy()
    for ticker in np.unique(tickers):
        positions = np.flatnonzero(tickers == ticker)
        positions = positions[np.argsort(dates[positions], kind="mergesort")]
        if len(positions):
            shift = (20 * permutation) % len(positions)
            result[positions] = np.roll(y[positions], shift)
    return result


def _load_arrays(plan: dict[str, Any]):
    train_idx, val_idx, weights = _indices_and_weights(plan)
    target = np.load(plan["target_path"], mmap_mode="r")
    first_hit = np.load(plan["first_hit_path"], mmap_mode="r")
    y_train = np.asarray(target[train_idx], dtype=np.uint8)
    y_train = _permuted_labels(plan, train_idx, y_train)
    y_val = np.asarray(target[val_idx], dtype=np.uint8)
    return (
        _matrix(plan, train_idx), y_train, weights,
        _matrix(plan, val_idx), y_val,
        np.asarray(first_hit[val_idx], dtype=np.int8), val_idx,
    )


def _cached(plan: dict[str, Any]) -> dict[str, Any] | None:
    cached = read_json(Path(plan["json_path"]), {})
    npz = Path(plan["npz_path"])
    if cached.get("status") == "complete" and cached.get("plan_id") == plan["plan_id"] and npz.exists() and npz.stat().st_size > 0:
        return cached
    return None


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    temp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def lightgbm_worker(plan: dict[str, Any]) -> dict[str, Any]:
    cached = _cached(plan)
    if cached:
        return cached
    started = time.time()
    try:
        set_worker_mode(int(plan["threads"]), str(plan.get("priority", "normal")))
        import lightgbm as lgb
        x_train, y_train, weights, x_val, y_val, hit, val_idx = _load_arrays(plan)
        cfg = dict(plan["model_config"])
        rounds = int(cfg.pop("rounds"))
        class_multiplier = float(cfg.pop("class_weight_multiplier", 1.0))
        pos = max(1, int(y_train.sum())); neg = max(1, int(len(y_train) - pos))
        base = {
            "objective": "binary", "metric": "None", "verbosity": -1,
            "deterministic": True, "force_col_wise": True, "feature_pre_filter": False,
            "num_threads": int(plan["threads"]), "scale_pos_weight": neg / pos * class_multiplier,
        }
        base.update(cfg)
        dataset = lgb.Dataset(x_train, label=y_train, weight=weights, free_raw_data=False,
                              params={"max_bin": int(base.get("max_bin", 255)), "feature_pre_filter": False})
        predictions = []; importances = []
        for seed in plan["seeds"]:
            params = {**base, "seed": int(seed), "feature_fraction_seed": int(seed), "bagging_seed": int(seed), "drop_seed": int(seed)}
            model = lgb.train(params, dataset, num_boost_round=rounds, callbacks=[lgb.log_evaluation(0)])
            predictions.append(np.asarray(model.predict(x_val, num_threads=int(plan["threads"])), dtype=np.float32))
            importances.append(np.asarray(model.feature_importance(importance_type="gain"), dtype=np.float32))
        _save_npz(Path(plan["npz_path"]), val_idx=val_idx, y=y_val, first_hit=hit,
                  predictions=np.vstack(predictions), importance_gain=np.vstack(importances),
                  seeds=np.asarray(plan["seeds"], dtype=np.int32))
        result = {
            "status": "complete", "plan_id": plan["plan_id"], "stage": plan["stage"], "model": "lightgbm",
            "candidate_id": plan["candidate"]["id"], "fold_id": int(plan["fold_id"]),
            "train_rows": int(len(y_train)), "validation_rows": int(len(y_val)),
            "train_positive_rate": float(y_train.mean()), "validation_positive_rate": float(y_val.mean()),
            "feature_count": int(x_train.shape[1]), "elapsed_seconds": time.time() - started,
            "npz_path": plan["npz_path"], "seeds": plan["seeds"],
        }
        atomic_json(result, Path(plan["json_path"]))
        return result
    except Exception as exc:
        result = {"status": "failed", "plan_id": plan["plan_id"], "candidate_id": plan["candidate"]["id"],
                  "fold_id": int(plan["fold_id"]), "model": "lightgbm", "error": repr(exc), "traceback": traceback.format_exc()}
        atomic_json(result, Path(plan["json_path"]))
        return result


def xgboost_worker(plan: dict[str, Any]) -> dict[str, Any]:
    cached = _cached(plan)
    if cached:
        return cached
    started = time.time()
    try:
        set_worker_mode(int(plan["threads"]), str(plan.get("priority", "normal")))
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        import xgboost as xgb
        x_train, y_train, weights, x_val, y_val, hit, val_idx = _load_arrays(plan)
        cfg = dict(plan["model_config"])
        rounds = int(cfg.pop("rounds"))
        class_multiplier = float(cfg.pop("class_weight_multiplier", 1.0))
        max_bin = int(cfg.get("max_bin", 256))
        dtrain = xgb.QuantileDMatrix(x_train, label=y_train, weight=weights, max_bin=max_bin, nthread=int(plan["threads"]))
        dval = xgb.QuantileDMatrix(x_val, ref=dtrain, max_bin=max_bin, nthread=int(plan["threads"]))
        pos = max(1, int(y_train.sum())); neg = max(1, int(len(y_train) - pos))
        predictions = []; importances = []
        for seed in plan["seeds"]:
            params = {
                "objective": "binary:logistic", "eval_metric": "aucpr", "device": "cuda", "tree_method": "hist",
                "scale_pos_weight": neg / pos * class_multiplier, "seed": int(seed), "verbosity": 0,
                "nthread": int(plan["threads"]), **cfg,
            }
            callbacks = []
            duty_cycle = min(1.0, max(0.01, float(plan.get("gpu_duty_cycle", 1.0))))
            if duty_cycle < 1.0:
                class DutyCycleCallback(xgb.callback.TrainingCallback):
                    def __init__(self, duty: float):
                        self.duty = duty
                        self.last = 0.0

                    def before_training(self, model):
                        self.last = time.perf_counter()
                        return model

                    def after_iteration(self, model, epoch, evals_log):
                        now = time.perf_counter()
                        active = max(0.0, now - self.last)
                        delay = min(0.25, active * (1.0 / self.duty - 1.0))
                        if delay > 0.0:
                            time.sleep(delay)
                        self.last = time.perf_counter()
                        return False

                callbacks.append(DutyCycleCallback(duty_cycle))
            model = xgb.train(params, dtrain, num_boost_round=rounds, verbose_eval=False, callbacks=callbacks)
            predictions.append(np.asarray(model.predict(dval), dtype=np.float32))
            score = model.get_score(importance_type="gain")
            gain = np.zeros(x_train.shape[1], dtype=np.float32)
            for key, value in score.items():
                if key.startswith("f") and key[1:].isdigit():
                    index = int(key[1:])
                    if index < len(gain):
                        gain[index] = float(value)
            importances.append(gain)
        _save_npz(Path(plan["npz_path"]), val_idx=val_idx, y=y_val, first_hit=hit,
                  predictions=np.vstack(predictions), importance_gain=np.vstack(importances),
                  seeds=np.asarray(plan["seeds"], dtype=np.int32))
        result = {
            "status": "complete", "plan_id": plan["plan_id"], "stage": plan["stage"], "model": "xgboost",
            "candidate_id": plan["candidate"]["id"], "fold_id": int(plan["fold_id"]),
            "train_rows": int(len(y_train)), "validation_rows": int(len(y_val)),
            "train_positive_rate": float(y_train.mean()), "validation_positive_rate": float(y_val.mean()),
            "feature_count": int(x_train.shape[1]), "elapsed_seconds": time.time() - started,
            "npz_path": plan["npz_path"], "seeds": plan["seeds"],
        }
        atomic_json(result, Path(plan["json_path"]))
        return result
    except Exception as exc:
        result = {"status": "failed", "plan_id": plan["plan_id"], "candidate_id": plan["candidate"]["id"],
                  "fold_id": int(plan["fold_id"]), "model": "xgboost", "error": repr(exc), "traceback": traceback.format_exc()}
        atomic_json(result, Path(plan["json_path"]))
        return result


def _xgb_pool(plans: list[dict[str, Any]], workers: int, result_path: str) -> None:
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=int(workers), mp_context=context) as executor:
        results = list(executor.map(xgboost_worker, plans))
    atomic_json(results, Path(result_path))


def run_parallel(
    lgb_plans: list[dict[str, Any]],
    xgb_plans: list[dict[str, Any]],
    *,
    lgb_workers: int,
    xgb_workers: int,
    output: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scratch = output / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    result_path = scratch / f"xgb_{canonical_hash([p['plan_id'] for p in xgb_plans], 16)}.json"
    context = mp.get_context("spawn")
    gpu_process = None
    lgb_results: list[dict[str, Any]] = []
    xgb_results: list[dict[str, Any]] = []
    try:
        if xgb_plans:
            gpu_process = context.Process(target=_xgb_pool, args=(xgb_plans, xgb_workers, str(result_path)), name="cw-generalization-xgb-pool")
            gpu_process.start()
        if lgb_plans:
            with cf.ProcessPoolExecutor(max_workers=int(lgb_workers), mp_context=context) as executor:
                lgb_results = list(executor.map(lightgbm_worker, lgb_plans))
        if gpu_process is not None:
            gpu_process.join()
            if gpu_process.exitcode != 0:
                raise RuntimeError(f"XGBoost pool exited with {gpu_process.exitcode}")
            xgb_results = read_json(result_path, [])
        failed = [row for row in lgb_results + xgb_results if row.get("status") != "complete"]
        gpu_oom = [row for row in failed if "out of memory" in (str(row.get("error")) + str(row.get("traceback"))).lower()]
        if gpu_oom and xgb_workers > 1:
            retry_ids = {row["plan_id"] for row in gpu_oom}
            retry_plans = [plan for plan in xgb_plans if plan["plan_id"] in retry_ids]
            retry_path = scratch / f"xgb_retry_{canonical_hash(sorted(retry_ids), 16)}.json"
            _xgb_pool(retry_plans, 1, str(retry_path))
            retry = {row["plan_id"]: row for row in read_json(retry_path, [])}
            xgb_results = [retry.get(row.get("plan_id"), row) for row in xgb_results]
            failed = [row for row in lgb_results + xgb_results if row.get("status") != "complete"]
        if failed:
            raise RuntimeError(f"bundle failures: {failed[:3]}")
        return lgb_results, xgb_results
    finally:
        if gpu_process is not None and gpu_process.is_alive():
            gpu_process.terminate(); gpu_process.join(timeout=10)
            if gpu_process.is_alive():
                gpu_process.kill()


def load_checkpoint_predictions(results: list[dict[str, Any]], candidate_id: str, fold_id: int, model: str) -> dict[str, np.ndarray]:
    matches = [row for row in results if row.get("candidate_id") == candidate_id and int(row.get("fold_id", -1)) == int(fold_id) and row.get("model") == model]
    if len(matches) != 1:
        raise RuntimeError(f"checkpoint match {candidate_id}/{fold_id}/{model} count={len(matches)}")
    with np.load(matches[0]["npz_path"]) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}
