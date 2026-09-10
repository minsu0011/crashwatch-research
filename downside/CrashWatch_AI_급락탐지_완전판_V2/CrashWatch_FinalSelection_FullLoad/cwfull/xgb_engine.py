from __future__ import annotations

import gc
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from cw7h.gpu_worker import detect_cuda
from cw7h.metrics import compute_metrics
from cw7h.utils import atomic_json, canonical_hash, hash_strings, read_json
from .common import nvml_snapshot, set_worker_mode

ENGINE_VERSION = "cw_final_selection_xgb_profiles_v1"


def _params(config: dict[str, Any], y_train: np.ndarray, seed: int) -> dict[str, Any]:
    positives = max(1, int(np.sum(y_train == 1)))
    negatives = max(1, int(np.sum(y_train == 0)))
    return {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "device": "cuda",
        "tree_method": "hist",
        "learning_rate": float(config["learning_rate"]),
        "max_depth": int(config["max_depth"]),
        "max_leaves": int(config["max_leaves"]),
        "grow_policy": str(config["grow_policy"]),
        "min_child_weight": float(config["min_child_weight"]),
        "subsample": float(config["subsample"]),
        "colsample_bytree": float(config["colsample_bytree"]),
        "reg_alpha": float(config["reg_alpha"]),
        "reg_lambda": float(config["reg_lambda"]),
        "max_bin": int(config["max_bin"]),
        "scale_pos_weight": float(negatives / positives),
        "seed": int(seed),
        "nthread": int(config["nthread"]),
        "verbosity": 0,
    }


def _wait_for_gpu(config: dict[str, Any]) -> float:
    waited = 0.0
    max_memory_fraction = float(config.get("max_memory_fraction_before_launch", 0.90))
    max_wait = float(config.get("launch_guard_max_wait_seconds", 300.0))
    poll = float(config.get("launch_guard_poll_seconds", 2.0))
    while waited < max_wait:
        snap = nvml_snapshot()
        used = snap.get("gpu_memory_used_gb")
        total = snap.get("gpu_memory_total_gb")
        if used is None or total in (None, 0):
            return waited
        if float(used) / float(total) <= max_memory_fraction:
            return waited
        time.sleep(poll)
        waited += poll
    return waited


def _make_matrices(xgb: Any, x_train: np.ndarray, y_train: np.ndarray, x_valid: np.ndarray, config: dict[str, Any]):
    max_bin = int(config["max_bin"])
    nthread = int(config["nthread"])
    try:
        dtrain = xgb.QuantileDMatrix(x_train, label=y_train, max_bin=max_bin, nthread=nthread)
        dvalid = xgb.QuantileDMatrix(x_valid, ref=dtrain, max_bin=max_bin, nthread=nthread)
    except Exception:
        dtrain = xgb.DMatrix(x_train, label=y_train, nthread=nthread)
        dvalid = xgb.DMatrix(x_valid, nthread=nthread)
    return dtrain, dvalid


def _assignment_worker(plan: dict[str, Any], queue: mp.Queue) -> None:
    set_worker_mode(int(plan["config"]["nthread"]), "above_normal")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "2"
    if int(plan.get("worker_index", 0)):
        time.sleep(float(plan.get("worker_start_stagger_seconds", 3.0)) * int(plan["worker_index"]))
    gpu = detect_cuda()
    if not gpu.get("available"):
        queue.put({"ok": False, "error": "CUDA GPU not detected", "gpu": gpu})
        return
    try:
        import xgboost as xgb

        cache_root = Path(plan["cache_root"])
        X = np.load(cache_root / "X_all_valid.npy", mmap_mode="r")
        y = np.load(cache_root / "target.npy", mmap_mode="r")
        dates = np.load(cache_root / "dates_ns.npy", mmap_mode="r")
        feature_index = {feature: index for index, feature in enumerate(plan["feature_names"])}
        task_dir = Path(plan["task_dir"])
        task_dir.mkdir(parents=True, exist_ok=True)
        completed = cached = failed = 0
        elapsed_values: list[float] = []
        for bundle in plan["bundles"]:
            _wait_for_gpu(plan["config"])
            fold = bundle["fold"]
            features = bundle["features"]
            profile = bundle["profile"]
            profile_hash = hash_strings(features)
            indices = np.asarray([feature_index[feature] for feature in features], dtype=np.int32)
            train_slice = slice(int(fold["train_start"]), int(fold["train_stop"]))
            valid_slice = slice(int(fold["validation_start"]), int(fold["validation_stop"]))
            x_train = np.ascontiguousarray(X[train_slice][:, indices], dtype=np.float32)
            x_valid = np.ascontiguousarray(X[valid_slice][:, indices], dtype=np.float32)
            y_train = np.asarray(y[train_slice], dtype=np.uint8)
            y_valid = np.asarray(y[valid_slice], dtype=np.uint8)
            valid_dates = np.asarray(dates[valid_slice], dtype=np.int64)
            dtrain, dvalid = _make_matrices(xgb, x_train, y_train, x_valid, plan["config"])
            for seed in plan["seeds"]:
                identity = {
                    "engine_version": ENGINE_VERSION,
                    "dataset_signature": plan["dataset_signature"],
                    "profile": profile,
                    "profile_hash": profile_hash,
                    "feature_count": len(features),
                    "fold": fold,
                    "seed": int(seed),
                    "config": plan["config"],
                }
                task_id = canonical_hash(identity)
                task_path = task_dir / f"{task_id}.json"
                old = read_json(task_path, {})
                if old.get("status") == "completed" and old.get("identity") == identity:
                    cached += 1
                    continue
                started = time.perf_counter()
                before = nvml_snapshot()
                try:
                    params = _params(plan["config"], y_train, int(seed))
                    try:
                        model = xgb.train(
                            params,
                            dtrain,
                            num_boost_round=int(plan["config"]["rounds"]),
                            verbose_eval=False,
                        )
                    except xgb.core.XGBoostError as first_error:
                        fallback = dict(params)
                        fallback.pop("device", None)
                        fallback["tree_method"] = "gpu_hist"
                        fallback["predictor"] = "gpu_predictor"
                        try:
                            model = xgb.train(
                                fallback,
                                dtrain,
                                num_boost_round=int(plan["config"]["rounds"]),
                                verbose_eval=False,
                            )
                        except Exception:
                            raise first_error
                    prediction = np.asarray(model.predict(dvalid), dtype=np.float32)
                    elapsed = time.perf_counter() - started
                    elapsed_values.append(elapsed)
                    after = nvml_snapshot()
                    result = {
                        "status": "completed",
                        "identity": identity,
                        "task_id": task_id,
                        "backend": "xgboost_cuda",
                        "profile": profile,
                        "profile_hash": profile_hash,
                        "feature_count": len(features),
                        "seed": int(seed),
                        "outer_fold": int(fold["fold_id"]),
                        "rounds": int(plan["config"]["rounds"]),
                        "elapsed_seconds": elapsed,
                        "gpu_before": before,
                        "gpu_after": after,
                        **compute_metrics(y_valid, prediction, valid_dates),
                    }
                    atomic_json(result, task_path)
                    completed += 1
                    del model, prediction
                except Exception as exc:
                    failed += 1
                    atomic_json(
                        {
                            "status": "failed",
                            "identity": identity,
                            "task_id": task_id,
                            "backend": "xgboost_cuda",
                            "profile": profile,
                            "profile_hash": profile_hash,
                            "seed": int(seed),
                            "outer_fold": int(fold["fold_id"]),
                            "error": repr(exc),
                            "traceback": traceback.format_exc(),
                        },
                        task_path,
                    )
                gc.collect()
            del dtrain, dvalid, x_train, x_valid, y_train, y_valid, valid_dates
            gc.collect()
        queue.put(
            {
                "ok": True,
                "result": {
                    "worker_index": int(plan["worker_index"]),
                    "gpu": gpu,
                    "completed": completed,
                    "cached": cached,
                    "failed": failed,
                    "median_model_seconds": float(np.median(elapsed_values)) if elapsed_values else None,
                },
            }
        )
    except Exception as exc:
        queue.put({"ok": False, "error": repr(exc), "traceback": traceback.format_exc(), "gpu": gpu})


def run_profiles(
    *,
    prepared,
    folds,
    profiles: dict[str, list[str]],
    seeds: list[int],
    config: dict[str, Any],
    task_dir: Path,
    workers: int,
    summary_path: Path,
) -> dict[str, Any]:
    gpu = detect_cuda()
    if not gpu.get("available"):
        summary = {"status": "skipped", "gpu": gpu, "reason": "CUDA GPU not detected"}
        atomic_json(summary, summary_path)
        return summary
    bundles = []
    fold_order = {fold_id: position for position, fold_id in enumerate([7, 6, 5, 4, 3, 2, 1, 0])}
    for fold in sorted(folds, key=lambda item: fold_order[item.fold_id]):
        for profile, features in profiles.items():
            bundles.append({"fold": fold.to_dict(), "profile": profile, "features": features})
    assignments = [[] for _ in range(max(1, int(workers)))]
    for index, bundle in enumerate(bundles):
        assignments[index % len(assignments)].append(bundle)

    context = mp.get_context("spawn")
    queue: mp.Queue = context.Queue()
    processes: list[mp.Process] = []
    started = time.time()
    for worker_index, assignment in enumerate(assignments):
        plan = {
            "worker_index": worker_index,
            "worker_start_stagger_seconds": float(config.get("worker_start_stagger_seconds", 3.0)),
            "cache_root": str(prepared.root),
            "feature_names": prepared.feature_names,
            "dataset_signature": prepared.signature,
            "bundles": assignment,
            "seeds": [int(seed) for seed in seeds],
            "config": config,
            "task_dir": str(task_dir),
        }
        process = context.Process(
            target=_assignment_worker,
            args=(plan, queue),
            name=f"cw-final-xgb-{worker_index}",
            daemon=False,
        )
        process.start()
        processes.append(process)

    messages = []
    for process in processes:
        process.join()
    for _ in processes:
        try:
            messages.append(queue.get(timeout=10))
        except Exception:
            messages.append({"ok": False, "error": "worker exited without queue result"})
    failures = [message for message in messages if not message.get("ok")]
    worker_results = [message["result"] for message in messages if message.get("ok")]
    expected = len(profiles) * len(folds) * len(seeds)
    summary = {
        "status": "completed" if not failures and sum(item["failed"] for item in worker_results) == 0 else "partial",
        "gpu": gpu,
        "profiles": list(profiles),
        "profile_count": len(profiles),
        "fold_count": len(folds),
        "seed_count": len(seeds),
        "expected_model_count": expected,
        "completed": sum(item["completed"] for item in worker_results),
        "cached": sum(item["cached"] for item in worker_results),
        "failed": sum(item["failed"] for item in worker_results),
        "worker_failures": failures,
        "workers": int(workers),
        "elapsed_seconds": time.time() - started,
        "worker_results": worker_results,
    }
    atomic_json(summary, summary_path)
    return summary
