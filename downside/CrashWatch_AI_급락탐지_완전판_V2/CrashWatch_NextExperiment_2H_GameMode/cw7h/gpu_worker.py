from __future__ import annotations

import gc
import logging
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from .folds import FoldSlice
from .metrics import compute_metrics
from .utils import atomic_json, canonical_hash, ensure_thread_env, read_json

LOGGER = logging.getLogger(__name__)
GPU_CACHE_VERSION = "cw7h_xgb_cuda_exact_loo_v1"


def detect_cuda() -> dict[str, Any]:
    result: dict[str, Any] = {"available": False, "name": "", "reason": ""}
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            result["reason"] = proc.stderr.strip() or "nvidia-smi unavailable"
            return result
        parts = [x.strip() for x in proc.stdout.splitlines()[0].split(",")]
        result.update({
            "available": True,
            "name": parts[0] if parts else "NVIDIA GPU",
            "memory_total_mb": float(parts[1]) if len(parts) > 1 else None,
            "driver_version": parts[2] if len(parts) > 2 else "",
        })
        return result
    except Exception as exc:
        result["reason"] = repr(exc)
        return result


def _params(config: dict[str, Any], y_train: np.ndarray) -> dict[str, Any]:
    pos = max(1, int(np.sum(y_train == 1)))
    neg = max(1, int(np.sum(y_train == 0)))
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "device": "cuda",
        "tree_method": "hist",
        "learning_rate": float(config.get("learning_rate", 0.035)),
        "max_depth": int(config.get("max_depth", 0)),
        "max_leaves": int(config.get("max_leaves", 96)),
        "grow_policy": str(config.get("grow_policy", "lossguide")),
        "min_child_weight": float(config.get("min_child_weight", 7.0)),
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": float(config.get("reg_alpha", 0.2)),
        "reg_lambda": float(config.get("reg_lambda", 1.8)),
        "max_bin": int(config.get("max_bin", 256)),
        "scale_pos_weight": float(neg / pos),
        "seed": 17,
        "nthread": int(config.get("nthread", 1)),
        "verbosity": 0,
    }


def _task_identity(dataset_signature: str, run_signature: str, fold: FoldSlice, feature: str, feature_index: int, config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    identity = {
        "version": GPU_CACHE_VERSION,
        "dataset_signature": dataset_signature,
        "run_signature": run_signature,
        "backend": "xgboost_cuda",
        "fold": fold.to_dict(),
        "test_type": "single_feature_loo",
        "feature": feature,
        "feature_index": feature_index,
        "config": config,
    }
    return canonical_hash(identity), identity


def _train_predict(xgb: Any, X_train: np.ndarray, y_train: np.ndarray, X_valid: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    max_bin = int(config.get("max_bin", 256))
    try:
        dtrain = xgb.QuantileDMatrix(X_train, label=y_train, max_bin=max_bin, nthread=int(config.get("nthread", 1)))
        dvalid = xgb.QuantileDMatrix(X_valid, ref=dtrain, max_bin=max_bin, nthread=int(config.get("nthread", 1)))
    except Exception:
        dtrain = xgb.DMatrix(X_train, label=y_train, nthread=int(config.get("nthread", 1)))
        dvalid = xgb.DMatrix(X_valid, nthread=int(config.get("nthread", 1)))
    params = _params(config, y_train)
    try:
        model = xgb.train(params, dtrain, num_boost_round=int(config.get("rounds", 260)), verbose_eval=False)
    except xgb.core.XGBoostError as first_exc:
        # Compatibility fallback for older XGBoost builds.
        fallback = dict(params)
        fallback.pop("device", None)
        fallback["tree_method"] = "gpu_hist"
        fallback["predictor"] = "gpu_predictor"
        try:
            model = xgb.train(fallback, dtrain, num_boost_round=int(config.get("rounds", 260)), verbose_eval=False)
        except Exception:
            raise first_exc
    pred = np.asarray(model.predict(dvalid), dtype=np.float32)
    del model, dtrain, dvalid
    return pred


def run_gpu_feature_audit(plan: dict[str, Any]) -> None:
    ensure_thread_env(int(plan["gpu_config"].get("nthread", 1)))
    output_dir = Path(plan["output_dir"])
    status_path = output_dir / "gpu_audit_status.json"
    gpu = detect_cuda()
    if not gpu.get("available"):
        atomic_json({"status": "skipped", "gpu": gpu, "reason": "CUDA GPU not detected"}, status_path)
        return
    try:
        import xgboost as xgb
    except Exception as exc:
        atomic_json({"status": "skipped", "gpu": gpu, "reason": f"xgboost import failed: {exc!r}"}, status_path)
        return

    cache_root = Path(plan["cache_root"])
    result_dir = output_dir / "task_results" / "xgboost_cuda"
    result_dir.mkdir(parents=True, exist_ok=True)
    X = np.load(cache_root / "X_all_valid.npy", mmap_mode="r")
    y = np.load(cache_root / "target.npy", mmap_mode="r")
    dates = np.load(cache_root / "dates_ns.npy", mmap_mode="r")
    feature_names = list(plan["feature_names"])
    folds = [FoldSlice(**item) for item in plan["folds"] if item.get("eligible", True)]
    priority_path = output_dir / "correlation" / "feature_priority.csv"
    if priority_path.exists():
        priority = pd.read_csv(priority_path)["feature"].astype(str).tolist()
        ordered = [f for f in priority if f in feature_names]
        ordered.extend(f for f in feature_names if f not in set(ordered))
    else:
        ordered = feature_names
    max_features = int(plan["gpu_config"].get("max_features", len(feature_names)))
    ordered = ordered[:max_features]
    fold_order = list(plan["gpu_config"].get("fold_order", [7, 6, 5, 4, 3, 2, 1, 0]))
    fold_map = {f.fold_id: f for f in folds}
    folds = [fold_map[i] for i in fold_order if i in fold_map]
    started = time.time()
    completed = 0
    failed = 0
    skipped = 0
    atomic_json({
        "status": "running",
        "gpu": gpu,
        "dataset_signature": plan["dataset_signature"],
        "run_signature": plan["run_signature"],
        "started_epoch": started,
        "planned_features": len(ordered),
        "planned_folds": len(folds),
    }, status_path)
    try:
        for fold in folds:
            if time.time() >= float(plan["soft_deadline_epoch"]):
                break
            train_slice = slice(fold.train_start, fold.train_stop)
            valid_slice = slice(fold.validation_start, fold.validation_stop)
            x_train_full = np.asarray(X[train_slice], dtype=np.float32)
            y_train = np.asarray(y[train_slice], dtype=np.uint8)
            x_valid_full = np.asarray(X[valid_slice], dtype=np.float32)
            y_valid = np.asarray(y[valid_slice], dtype=np.uint8)
            valid_dates = np.asarray(dates[valid_slice], dtype=np.int64)
            baseline_path = result_dir / f"baseline_{plan['run_signature']}_fold_{fold.fold_id}.json"
            baseline = read_json(baseline_path, {}) if baseline_path.exists() else {}
            baseline_valid = (
                baseline.get("status") == "completed"
                and baseline.get("dataset_signature") == plan["dataset_signature"]
                and baseline.get("run_signature") == plan["run_signature"]
                and int(baseline.get("outer_fold", -1)) == fold.fold_id
            )
            if not baseline_valid:
                bstart = time.perf_counter()
                pred = _train_predict(xgb, x_train_full, y_train, x_valid_full, plan["gpu_config"])
                baseline = {
                    "status": "completed",
                    "backend": "xgboost_cuda",
                    "dataset_signature": plan["dataset_signature"],
                    "run_signature": plan["run_signature"],
                    "test_type": "baseline_all_valid",
                    "condition_id": "GPU_B0",
                    "feature": "",
                    "feature_index": -1,
                    "outer_fold": fold.fold_id,
                    "elapsed_seconds": time.perf_counter() - bstart,
                    **compute_metrics(y_valid, pred, valid_dates),
                }
                atomic_json(baseline, baseline_path)
                del pred
            for feature in ordered:
                if time.time() >= float(plan["soft_deadline_epoch"]):
                    break
                if psutil.virtual_memory().available / 1024**3 < float(plan["gpu_config"].get("minimum_free_ram_gb", 4.0)):
                    time.sleep(5)
                    if psutil.virtual_memory().available / 1024**3 < float(plan["gpu_config"].get("minimum_free_ram_gb", 4.0)):
                        skipped += 1
                        continue
                feature_index = feature_names.index(feature)
                task_id, identity = _task_identity(plan["dataset_signature"], plan["run_signature"], fold, feature, feature_index, plan["gpu_config"])
                result_path = result_dir / f"{task_id}.json"
                if result_path.exists():
                    cached = read_json(result_path)
                    if cached.get("status") == "completed" and cached.get("identity") == identity:
                        completed += 1
                        continue
                start = time.perf_counter()
                try:
                    mask = np.ones(len(feature_names), dtype=bool)
                    mask[feature_index] = False
                    x_train = np.ascontiguousarray(x_train_full[:, mask], dtype=np.float32)
                    x_valid = np.ascontiguousarray(x_valid_full[:, mask], dtype=np.float32)
                    pred = _train_predict(xgb, x_train, y_train, x_valid, plan["gpu_config"])
                    result = {
                        "status": "completed",
                        "identity": identity,
                        "task_id": task_id,
                        "backend": "xgboost_cuda",
                        "dataset_signature": plan["dataset_signature"],
                        "run_signature": plan["run_signature"],
                        "stage": "gpu_audit",
                        "test_type": "single_feature_loo",
                        "condition_id": f"GPU_LOO::{feature}",
                        "feature": feature,
                        "feature_index": feature_index,
                        "outer_fold": fold.fold_id,
                        "enabled_feature_count": len(feature_names) - 1,
                        "elapsed_seconds": time.perf_counter() - start,
                        **compute_metrics(y_valid, pred, valid_dates),
                    }
                    atomic_json(result, result_path)
                    completed += 1
                    del x_train, x_valid, pred
                except Exception as exc:
                    failed += 1
                    atomic_json({
                        "status": "failed",
                        "identity": identity,
                        "task_id": task_id,
                        "backend": "xgboost_cuda",
                        "dataset_signature": plan["dataset_signature"],
                        "run_signature": plan["run_signature"],
                        "stage": "gpu_audit",
                        "test_type": "single_feature_loo",
                        "condition_id": f"GPU_LOO::{feature}",
                        "feature": feature,
                        "outer_fold": fold.fold_id,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                        "elapsed_seconds": time.perf_counter() - start,
                    }, result_path)
                gc.collect()
            del x_train_full, x_valid_full, y_train, y_valid, valid_dates
            gc.collect()
    except Exception as exc:
        atomic_json({
            "status": "failed",
            "gpu": gpu,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "completed": completed,
            "failed": failed,
            "elapsed_seconds": time.time() - started,
        }, status_path)
        return
    final_status = "completed" if completed >= len(ordered) * len(folds) else "partial"
    atomic_json({
        "status": final_status,
        "gpu": gpu,
        "dataset_signature": plan["dataset_signature"],
        "run_signature": plan["run_signature"],
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
        "planned": len(ordered) * len(folds),
        "elapsed_seconds": time.time() - started,
        "soft_deadline_reached": time.time() >= float(plan["soft_deadline_epoch"]),
    }, status_path)
