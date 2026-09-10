from __future__ import annotations

import concurrent.futures as cf
import gc
import logging
import multiprocessing as mp
import os
import subprocess
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from cw7h.folds import FoldSlice
from cw7h.gpu_worker import _task_identity, _train_predict, detect_cuda
from cw7h.metrics import compute_metrics
from cw7h.utils import atomic_json, ensure_thread_env, read_json

from .common import write_eta

LOGGER = logging.getLogger(__name__)


def _gpu_state() -> tuple[float, float, float]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4, check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            return 0.0, 0.0, 0.0
        parts = [float(x.strip()) for x in proc.stdout.splitlines()[0].split(",")]
        return parts[0], parts[1], parts[2]
    except Exception:
        return 0.0, 0.0, 0.0


def _game_running() -> bool:
    targets = {"tslgame.exe", "tslgame", "pubg.exe", "pubg"}
    try:
        return any((p.info.get("name") or "").lower() in targets for p in psutil.process_iter(["name"]))
    except Exception:
        return False


def _set_worker_priority(mode: str, cpu_affinity: list[int] | None) -> None:
    proc = psutil.Process(os.getpid())
    try:
        if cpu_affinity:
            proc.cpu_affinity(cpu_affinity)
    except Exception:
        pass
    try:
        if os.name == "nt":
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if mode == "game" else psutil.NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


class UtilSampler:
    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.values: list[float] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.wait(self.interval):
                util, _, _ = _gpu_state()
                self.values.append(util)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return float(np.mean(self.values)) if self.values else _gpu_state()[0]


def _wait_launch_guard(config: dict[str, Any], mode: str) -> None:
    max_memory_fraction = float(config.get("max_memory_fraction_before_launch", 0.78 if mode == "game" else 0.95))
    max_wait = float(config.get("launch_guard_max_wait_seconds", 120.0))
    started = time.time()
    while True:
        util, used, total = _gpu_state()
        memory_ok = total <= 0 or used / total <= max_memory_fraction
        # Do not block forever on game utilization. Memory pressure is the more reliable stutter/OOM guard.
        if memory_ok:
            return
        if time.time() - started >= max_wait:
            LOGGER.warning("GPU launch guard timeout; used=%.0fMB total=%.0fMB util=%.1f%%", used, total, util)
            return
        time.sleep(3 if _game_running() else 1)


def _cooldown(active_seconds: float, sampled_util: float, config: dict[str, Any], mode: str) -> float:
    if mode != "game":
        return 0.0
    target = max(1.0, float(config.get("target_average_gpu_percent", 10.0)))
    # sampled_util includes the game. This intentionally errs on the safe side.
    sleep_seconds = active_seconds * max(0.0, sampled_util / target - 1.0)
    minimum = float(config.get("minimum_cooldown_seconds", 2.0))
    maximum = float(config.get("maximum_cooldown_seconds", 180.0))
    if _game_running():
        minimum = max(minimum, float(config.get("pubg_minimum_cooldown_seconds", 8.0)))
    sleep_seconds = min(maximum, max(minimum, sleep_seconds))
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return sleep_seconds


def _gpu_fold_worker(plan: dict[str, Any]) -> dict[str, Any]:
    mode = str(plan["mode"])
    config = dict(plan["gpu_config"])
    ensure_thread_env(int(config.get("nthread", 1)))
    _set_worker_priority(mode, plan.get("cpu_affinity"))
    try:
        import xgboost as xgb
    except Exception as exc:
        return {"status": "failed", "error": f"xgboost import failed: {exc!r}"}

    cache_root = Path(plan["cache_root"])
    output_dir = Path(plan["output_dir"])
    result_dir = output_dir / "task_results" / "xgboost_cuda"
    result_dir.mkdir(parents=True, exist_ok=True)
    X = np.load(cache_root / "X_all_valid.npy", mmap_mode="r")
    y = np.load(cache_root / "target.npy", mmap_mode="r")
    dates = np.load(cache_root / "dates_ns.npy", mmap_mode="r")
    feature_names = list(plan["feature_names"])
    index = {name: i for i, name in enumerate(feature_names)}
    completed = failed = cached = 0
    elapsed_active = elapsed_sleep = 0.0
    task_times: list[float] = []
    progress_path = output_dir / f"gpu_resume_progress_worker_{int(plan['worker_id'])}.json"

    def write_progress(status: str = "running") -> None:
        atomic_json({
            "status": status,
            "worker_id": int(plan["worker_id"]),
            "planned": int(plan["planned"]),
            "completed": completed,
            "cached": cached,
            "failed": failed,
            "active_seconds": elapsed_active,
            "cooldown_seconds": elapsed_sleep,
            "updated_epoch": time.time(),
        }, progress_path)

    write_progress()
    for fold_dict in plan["folds"]:
        fold = FoldSlice(**fold_dict)
        train_slice = slice(fold.train_start, fold.train_stop)
        valid_slice = slice(fold.validation_start, fold.validation_stop)
        x_train_full = np.asarray(X[train_slice], dtype=np.float32)
        y_train = np.asarray(y[train_slice], dtype=np.uint8)
        x_valid_full = np.asarray(X[valid_slice], dtype=np.float32)
        y_valid = np.asarray(y[valid_slice], dtype=np.uint8)
        valid_dates = np.asarray(dates[valid_slice], dtype=np.int64)
        baseline_path = result_dir / f"baseline_{plan['run_signature']}_fold_{fold.fold_id}.json"
        baseline = read_json(baseline_path, {})
        baseline_valid = (
            baseline.get("status") == "completed"
            and baseline.get("dataset_signature") == plan["dataset_signature"]
            and baseline.get("run_signature") == plan["run_signature"]
        )
        if not baseline_valid:
            _wait_launch_guard(config, mode)
            sampler = UtilSampler(); sampler.start(); start = time.perf_counter()
            try:
                pred = _train_predict(xgb, x_train_full, y_train, x_valid_full, config)
                active = time.perf_counter() - start
                sampled = sampler.stop()
                result = {
                    "status": "completed", "backend": "xgboost_cuda",
                    "dataset_signature": plan["dataset_signature"], "run_signature": plan["run_signature"],
                    "stage": "gpu_audit", "test_type": "baseline_all_valid", "condition_id": "GPU_B0",
                    "feature": "", "feature_index": -1, "outer_fold": fold.fold_id,
                    "elapsed_seconds": active, **compute_metrics(y_valid, pred, valid_dates),
                }
                atomic_json(result, baseline_path)
                elapsed_active += active
                cooldown = _cooldown(active, sampled, config, mode); elapsed_sleep += cooldown
                del pred
            except Exception:
                sampler.stop()
                raise
        for feature in plan["features_by_fold"].get(str(fold.fold_id), []):
            fi = index[feature]
            task_id, identity = _task_identity(plan["dataset_signature"], plan["run_signature"], fold, feature, fi, plan["identity_config"])
            path = result_dir / f"{task_id}.json"
            old = read_json(path, {})
            if old.get("status") == "completed" and old.get("identity") == identity:
                cached += 1
                write_progress()
                continue
            _wait_launch_guard(config, mode)
            sampler = UtilSampler(); sampler.start(); start = time.perf_counter()
            try:
                mask = np.ones(len(feature_names), dtype=bool); mask[fi] = False
                x_train = np.ascontiguousarray(x_train_full[:, mask], dtype=np.float32)
                x_valid = np.ascontiguousarray(x_valid_full[:, mask], dtype=np.float32)
                pred = _train_predict(xgb, x_train, y_train, x_valid, config)
                active = time.perf_counter() - start
                sampled = sampler.stop()
                result = {
                    "status": "completed", "identity": identity, "task_id": task_id,
                    "backend": "xgboost_cuda", "dataset_signature": plan["dataset_signature"],
                    "run_signature": plan["run_signature"], "stage": "gpu_audit",
                    "test_type": "single_feature_loo", "condition_id": f"GPU_LOO::{feature}",
                    "feature": feature, "feature_index": fi, "outer_fold": fold.fold_id,
                    "enabled_feature_count": len(feature_names) - 1, "elapsed_seconds": active,
                    **compute_metrics(y_valid, pred, valid_dates),
                }
                atomic_json(result, path)
                completed += 1; elapsed_active += active; task_times.append(active)
                cooldown = _cooldown(active, sampled, config, mode); elapsed_sleep += cooldown
                write_progress()
                del x_train, x_valid, pred
            except Exception as exc:
                sampled = sampler.stop()
                failed += 1
                atomic_json({
                    "status": "failed", "identity": identity, "task_id": task_id,
                    "backend": "xgboost_cuda", "dataset_signature": plan["dataset_signature"],
                    "run_signature": plan["run_signature"], "stage": "gpu_audit",
                    "test_type": "single_feature_loo", "condition_id": f"GPU_LOO::{feature}",
                    "feature": feature, "feature_index": fi, "outer_fold": fold.fold_id,
                    "error": repr(exc), "traceback": traceback.format_exc(),
                    "elapsed_seconds": time.perf_counter() - start,
                }, path)
                _cooldown(max(1.0, time.perf_counter() - start), sampled, config, mode)
                write_progress()
            if (completed + failed) % 8 == 0:
                gc.collect()
        del x_train_full, x_valid_full, y_train, y_valid, valid_dates
        gc.collect()
    write_progress("completed" if failed == 0 else "partial")
    return {
        "status": "completed" if failed == 0 else "partial", "completed": completed,
        "cached": cached, "failed": failed, "active_seconds": elapsed_active,
        "cooldown_seconds": elapsed_sleep, "mean_task_seconds": float(np.mean(task_times)) if task_times else None,
    }


def run_gpu_resume(
    prepared: Any,
    folds: list[Any],
    output_dir: Path,
    legacy_dataset_signature: str,
    run_signature: str,
    gpu_config: dict[str, Any],
    *,
    mode: str,
    workers: int,
    total_cpu_threads: int,
) -> dict[str, Any]:
    gpu = detect_cuda()
    if not gpu.get("available"):
        result = {"status": "skipped", "gpu": gpu}
        atomic_json(result, output_dir / "gpu_continuation_summary.json")
        return result
    metrics_path = output_dir / "all_model_metrics.csv"
    metrics = pd.read_csv(metrics_path, low_memory=False) if metrics_path.exists() else pd.DataFrame()
    completed_pairs: set[tuple[int, str]] = set()
    if not metrics.empty:
        part = metrics[(metrics.get("backend") == "xgboost_cuda") & (metrics.get("test_type") == "single_feature_loo")]
        completed_pairs = {(int(r.outer_fold), str(r.feature)) for r in part.itertuples(index=False)}
    priority_path = output_dir / "correlation" / "feature_priority.csv"
    priority = pd.read_csv(priority_path)["feature"].astype(str).tolist() if priority_path.exists() else list(prepared.feature_names)
    ordered = [f for f in priority if f in prepared.feature_names]
    ordered += [f for f in prepared.feature_names if f not in set(ordered)]
    features_by_fold: dict[str, list[str]] = {}
    total_missing = 0
    for fold in folds:
        missing = [f for f in ordered if (fold.fold_id, f) not in completed_pairs]
        features_by_fold[str(fold.fold_id)] = missing
        total_missing += len(missing)
    if total_missing == 0:
        result = {"status": "completed", "planned_missing": 0, "completed": 0, "gpu": gpu}
        atomic_json(result, output_dir / "gpu_continuation_summary.json")
        return result

    workers = max(1, min(int(workers), len(folds)))
    fold_order = [7, 6, 5, 4, 3, 2, 1, 0]
    fold_map = {f.fold_id: f for f in folds}
    assignments: list[list[Any]] = [[] for _ in range(workers)]
    weighted = sorted((fid for fid in fold_order if fid in fold_map), key=lambda fid: len(features_by_fold[str(fid)]), reverse=True)
    loads = [0] * workers
    for fid in weighted:
        w = min(range(workers), key=lambda i: loads[i])
        assignments[w].append(fold_map[fid])
        loads[w] += len(features_by_fold[str(fid)])
    logical = psutil.cpu_count(logical=True) or total_cpu_threads
    affinity_pool = list(range(max(0, logical - total_cpu_threads), logical)) if mode == "game" else list(range(logical))
    plans: list[dict[str, Any]] = []
    for worker_id, assigned in enumerate(assignments):
        worker_affinity = affinity_pool[worker_id::workers] or affinity_pool
        identity_keys = [
            "enabled", "max_features", "fold_order", "rounds", "learning_rate", "max_depth",
            "max_leaves", "grow_policy", "min_child_weight", "reg_alpha", "reg_lambda",
            "max_bin", "nthread", "minimum_free_ram_gb",
        ]
        identity_config = {k: gpu_config[k] for k in identity_keys if k in gpu_config}
        plans.append({
            "worker_id": worker_id,
            "planned": sum(len(features_by_fold[str(f.fold_id)]) for f in assigned),
            "mode": mode, "cache_root": str(prepared.root), "output_dir": str(output_dir),
            "dataset_signature": legacy_dataset_signature, "run_signature": run_signature,
            "feature_names": prepared.feature_names, "folds": [f.to_dict() for f in assigned],
            "features_by_fold": features_by_fold, "gpu_config": gpu_config,
            "identity_config": identity_config, "cpu_affinity": worker_affinity,
        })
    started = time.time()
    ctx = mp.get_context("spawn")
    results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = [executor.submit(_gpu_fold_worker, plan) for plan in plans]
        while futures:
            done, pending = cf.wait(futures, timeout=15, return_when=cf.FIRST_COMPLETED)
            for f in done:
                results.append(f.result()); futures.remove(f)
            completed = sum(int(r.get("completed", 0)) + int(r.get("cached", 0)) for r in results)
            # Workers checkpoint progress after every model, so ETA remains
            # useful even while the only game-mode worker is still running.
            live_progress = 0
            for progress_path in output_dir.glob("gpu_resume_progress_worker_*.json"):
                progress = read_json(progress_path, {})
                live_progress += int(progress.get("completed", 0)) + int(progress.get("cached", 0))
            write_eta(
                output_dir,
                "gpu_missing_resume",
                min(total_missing, max(completed, live_progress)),
                total_missing,
                time.time() - started,
            )
            if pending:
                LOGGER.info("GPU resume running workers=%s completed_finished_workers=%s/%s", workers, len(results), len(plans))
    summary = {
        "status": "completed" if sum(int(r.get("failed", 0)) for r in results) == 0 else "partial",
        "gpu": gpu, "mode": mode, "workers": workers, "planned_missing": total_missing,
        "completed": sum(int(r.get("completed", 0)) for r in results),
        "cached": sum(int(r.get("cached", 0)) for r in results),
        "failed": sum(int(r.get("failed", 0)) for r in results),
        "active_seconds": sum(float(r.get("active_seconds", 0.0)) for r in results),
        "cooldown_seconds": sum(float(r.get("cooldown_seconds", 0.0)) for r in results),
        "wall_seconds": time.time() - started, "worker_results": results,
    }
    atomic_json(summary, output_dir / "gpu_continuation_summary.json")
    return summary
