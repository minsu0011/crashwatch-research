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

from .folds import FoldSlice
from .metrics import scoped_metric_rows
from .utils import atomic_json, canonical_json_hash, ensure_thread_env, read_json

LOGGER = logging.getLogger(__name__)
GPU_CACHE_VERSION = "cw3h_xgb_cuda_v3"


def detect_nvidia_gpu() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=8,
            check=True,
        )
        first = result.stdout.strip().splitlines()[0]
        name, total, free = [part.strip() for part in first.split(",", 2)]
        return {"available": True, "name": name, "memory_total_mb": float(total), "memory_free_mb": float(free)}
    except Exception as exc:
        return {"available": False, "error": repr(exc)}


def run_gpu_audit_sequence(payload: dict[str, Any]) -> dict[str, Any]:
    threads = int(payload["threads"])
    ensure_thread_env(threads)
    status_path = Path(payload["output_dir"]) / "gpu_audit_status.json"
    gpu_info = detect_nvidia_gpu()
    if not gpu_info.get("available"):
        result = {"status": "unavailable", "gpu": gpu_info}
        atomic_json(result, status_path)
        return result
    try:
        import xgboost as xgb
    except Exception as exc:
        result = {"status": "unavailable", "gpu": gpu_info, "error": f"xgboost import failed: {exc!r}"}
        atomic_json(result, status_path)
        return result

    cache_root = Path(payload["cache_root"])
    deadline_epoch = float(payload["deadline_epoch"])
    tasks_complete = 0
    tasks_failed = 0
    cache_hits = 0
    started_all = time.perf_counter()
    try:
        for profile in payload["profiles"]:
            profile_dir = cache_root / "profiles" / profile
            profile_manifest = read_json(profile_dir / "profile_manifest.json")
            dates_ns = np.load(profile_dir / "dates_ns.npy", mmap_mode="r")
            y = np.load(profile_dir / "target.npy", mmap_mode="r")
            tickers = np.load(profile_dir / "tickers.npy", mmap_mode="r")
            buckets = np.load(profile_dir / "buckets.npy", mmap_mode="r")
            row_ids = np.load(profile_dir / "original_row_id.npy", mmap_mode="r")
            fold_map = {int(item["fold_id"]): FoldSlice(**item) for item in payload["folds_by_profile"][profile]}
            for fold_id in payload["fold_ids"]:
                fold = fold_map[int(fold_id)]
                if not fold.eligible:
                    continue
                for condition in payload["conditions"]:
                    if time.time() >= deadline_epoch:
                        raise TimeoutError("gpu audit deadline reached")
                    condition_meta = profile_manifest["conditions"][condition]
                    feature_hash = condition_meta["feature_hash"]
                    feature_count = len(condition_meta["features"])
                    x = np.load(profile_dir / "matrices" / f"{condition}.npy", mmap_mode="r")
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

                    result_root = cache_root / "task_results" / "xgboost_cuda"
                    metric_dir = result_root / "metrics"
                    pred_dir = result_root / "predictions"
                    metric_dir.mkdir(parents=True, exist_ok=True)
                    pred_dir.mkdir(parents=True, exist_ok=True)
                    pending: list[tuple[int, str, Path, Path, dict[str, Any]]] = []
                    for seed in payload["seeds"]:
                        identity = {
                            "kind": "xgboost_cuda_audit",
                            "runner_version": GPU_CACHE_VERSION,
                            "family": "xgboost",
                            "backend": "cuda",
                            "dataset_signature": payload["dataset_signature"],
                            "profile": profile,
                            "condition": condition,
                            "feature_hash": feature_hash,
                            "fold": fold.to_dict(),
                            "seed": int(seed),
                            "model": payload["xgboost_config"],
                        }
                        task_id = canonical_json_hash(identity)
                        metric_path = metric_dir / f"{task_id}.json"
                        pred_path = pred_dir / f"{task_id}.parquet"
                        if metric_path.exists() and pred_path.exists():
                            cached = read_json(metric_path)
                            if cached.get("status") == "completed":
                                cache_hits += 1
                                tasks_complete += 1
                                continue
                        pending.append((int(seed), task_id, metric_path, pred_path, identity))
                    if not pending:
                        del x
                        continue

                    config = payload["xgboost_config"]
                    max_bin = int(config["max_bin"])
                    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, max_bin=max_bin)
                    dvalid = xgb.QuantileDMatrix(x_valid, label=y_valid, max_bin=max_bin, ref=dtrain)
                    positives = int(np.sum(y_train == 1))
                    negatives = int(np.sum(y_train == 0))
                    scale_pos_weight = max(1.0, negatives / max(1, positives))
                    for seed, task_id, metric_path, pred_path, identity in pending:
                        if time.time() >= deadline_epoch:
                            break
                        started = time.perf_counter()
                        try:
                            params = {
                                key: value for key, value in config.items() if key != "num_boost_round"
                            }
                            params.update({
                                "tree_method": "hist",
                                "device": "cuda",
                                "seed": seed,
                                "nthread": threads,
                                "scale_pos_weight": scale_pos_weight,
                                "verbosity": 0,
                            })
                            model = xgb.train(params, dtrain, num_boost_round=int(config["num_boost_round"]), verbose_eval=False)
                            pred = np.asarray(model.predict(dvalid), dtype=np.float32)
                            scope_rows = scoped_metric_rows(y_valid, pred, valid_dates, valid_buckets, False)
                            elapsed = time.perf_counter() - started
                            prediction_frame = pd.DataFrame({
                                "original_row_id": valid_row_ids,
                                "date": pd.to_datetime(valid_dates),
                                "ticker": valid_tickers,
                                "bucket": valid_buckets,
                                "target": y_valid,
                                "prediction": pred,
                            })
                            temp_path = pred_path.with_suffix(".tmp.parquet")
                            prediction_frame.to_parquet(temp_path, index=False, compression=payload["prediction_compression"], engine="pyarrow")
                            os.replace(temp_path, pred_path)
                            metric_rows = [{
                                "task_id": task_id,
                                "family": "xgboost",
                                "backend": "cuda",
                                "phase": "gpu_sensitivity",
                                "profile": profile,
                                "condition": condition,
                                "outer_fold": fold.fold_id,
                                "seed": seed,
                                "feature_count": feature_count,
                                "feature_hash": feature_hash,
                                "best_iteration": int(config["num_boost_round"]),
                                "elapsed_seconds": elapsed,
                                "train_rows": int(len(y_train)),
                                "validation_rows": int(len(y_valid)),
                                "train_date_min": fold.train_date_min,
                                "train_date_max": fold.train_date_max,
                                "validation_date_min": fold.validation_date_min,
                                "validation_date_max": fold.validation_date_max,
                                "prediction_path": str(pred_path),
                                **scope,
                            } for scope in scope_rows]
                            atomic_json({"status": "completed", "task_id": task_id, "identity": identity, "metrics": metric_rows, "prediction_path": str(pred_path), "elapsed_seconds": elapsed}, metric_path)
                            tasks_complete += 1
                            del model, pred
                            gc.collect()
                        except Exception as exc:
                            pred_path.with_suffix(".tmp.parquet").unlink(missing_ok=True)
                            atomic_json({"status": "failed", "task_id": task_id, "identity": identity, "error": repr(exc), "traceback": traceback.format_exc()}, metric_path)
                            tasks_failed += 1
                    del dtrain, dvalid, x, x_train, x_valid
                    gc.collect()
        result = {
            "status": "completed" if tasks_failed == 0 else "partial",
            "gpu": gpu_info,
            "tasks_completed": tasks_complete,
            "tasks_failed": tasks_failed,
            "cache_hits": cache_hits,
            "elapsed_seconds": time.perf_counter() - started_all,
        }
        atomic_json(result, status_path)
        return result
    except TimeoutError as exc:
        result = {
            "status": "deadline_stopped",
            "gpu": gpu_info,
            "tasks_completed": tasks_complete,
            "tasks_failed": tasks_failed,
            "cache_hits": cache_hits,
            "error": str(exc),
            "elapsed_seconds": time.perf_counter() - started_all,
        }
        atomic_json(result, status_path)
        return result
    except Exception as exc:
        result = {
            "status": "failed",
            "gpu": gpu_info,
            "tasks_completed": tasks_complete,
            "tasks_failed": tasks_failed + 1,
            "cache_hits": cache_hits,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.perf_counter() - started_all,
        }
        atomic_json(result, status_path)
        return result
