from __future__ import annotations

import csv
import gc
import json
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import get_paths
from ..io_utils import atomic_json
from ..ticker_map1h.registry import TickerTaskRegistry
from .correlation import compute_ticker_correlation_map
from .data import list_cached_tickers, load_cached_ticker, prepare_ticker_cache
from .model import run_elite_ticker_task
from .reporting import aggregate_results, pack_results

LOGGER = logging.getLogger(__name__)


def load_plan(project: Path) -> dict[str, Any]:
    plan = json.loads((project / "configs" / "ticker_elite2h_plan.json").read_text(encoding="utf-8"))
    profile = os.environ.get("CRASHWATCH_EXECUTION_PROFILE", "cpu4").strip().lower()
    if profile == "cpu4":
        plan.update({
            "execution_profile": "cpu4",
            "backend_mode": "cpu_only",
            "result_subdir": "ticker_independent_cpu4_unlimited",
            "cache_namespace": "ticker_independent_cpu4_unlimited_v3",
            "gpu_workers": 1,
            "threads_per_model_worker": 4,
            "correlation_workers": 0,
            "threads_per_correlation_worker": 0,
            "min_available_ram_hard_gb": 2.5,
            "min_available_ram_soft_gb": 4.0,
        })
    elif profile == "full":
        plan.update({
            "execution_profile": "full",
            "backend_mode": "hybrid_fixed",
            "result_subdir": "ticker_independent_full_unlimited",
            "cache_namespace": "ticker_independent_full_unlimited_v3",
            "gpu_workers": 4,
            "threads_per_model_worker": 3,
            "correlation_workers": 1,
            "threads_per_correlation_worker": 4,
            "min_available_ram_hard_gb": 2.0,
            "min_available_ram_soft_gb": 4.0,
        })
    else:
        raise ValueError(f"unsupported CRASHWATCH_EXECUTION_PROFILE: {profile}")
    os.environ["CRASHWATCH_BACKEND_MODE"] = str(plan["backend_mode"])
    return plan


def _gpu_inventory() -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        name, memory = [value.strip() for value in output.split(",", 1)]
        return {"gpu_name": name, "vram_total_gb": float(memory) / 1024.0}
    except Exception:
        return {"gpu_name": "unknown", "vram_total_gb": 0.0}


def choose_runtime_layout(plan: dict[str, Any]) -> dict[str, Any]:
    import psutil

    logical = int(psutil.cpu_count(logical=True) or 1)
    memory = psutil.virtual_memory()
    profile = str(plan.get("execution_profile", "cpu4"))
    gpu = _gpu_inventory() if profile == "full" else {"gpu_name": "sealed", "vram_total_gb": 0.0}
    if profile == "cpu4":
        model_workers = 1
        model_threads = min(4, logical)
        correlation_workers = 0
        correlation_threads = 0
    else:
        # 9800X3D: 4 model workers x 3 threads + one 4-thread correlation worker.
        model_workers = min(4, max(1, logical // 3))
        model_threads = 3 if logical >= 12 else max(1, logical // model_workers)
        correlation_workers = 1 if logical >= 8 else 0
        correlation_threads = 4 if correlation_workers and logical >= 16 else (2 if correlation_workers else 0)
        if memory.total / 1024 ** 3 < 28:
            model_workers = min(model_workers, 3)
        if float(gpu.get("vram_total_gb", 0.0)) < 14:
            model_workers = min(model_workers, 3)
        allocated = model_workers * model_threads + correlation_workers * correlation_threads
        if allocated > logical and model_workers > 1:
            model_threads = max(1, (logical - correlation_workers * correlation_threads) // model_workers)
    return {
        "execution_profile": profile,
        "backend_mode": plan.get("backend_mode"),
        "model_workers": model_workers,
        "threads_per_model_worker": model_threads,
        "correlation_workers": correlation_workers,
        "threads_per_correlation_worker": correlation_threads,
        "logical_cores": logical,
        "total_ram_gb": memory.total / 1024 ** 3,
        "available_ram_gb": memory.available / 1024 ** 3,
        **gpu,
        "allocated_threads": model_workers * model_threads + correlation_workers * correlation_threads,
        "cache_mode": "per_ticker_float32_memmap",
        "backend_policy": "all_cpu" if profile == "cpu4" else "lightgbm_cpu_xgboost_catboost_cuda",
    }


def _set_affinity(role: str, index: int, layout: dict[str, Any]) -> None:
    try:
        import psutil

        process = psutil.Process()
        logical = int(layout["logical_cores"])
        if role == "model":
            width = int(layout["threads_per_model_worker"])
            start = index * width
        else:
            width = int(layout["threads_per_correlation_worker"])
            start = int(layout["model_workers"]) * int(layout["threads_per_model_worker"]) + index * width
        cores = sorted({core % logical for core in range(start, start + width)})
        process.cpu_affinity(cores)
        if os.name == "nt":
            profile = str(layout.get("execution_profile", "cpu4"))
            if profile == "full":
                process.nice(psutil.HIGH_PRIORITY_CLASS if role == "model" else psutil.ABOVE_NORMAL_PRIORITY_CLASS)
            else:
                process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _thread_limit(threads: int):
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=threads)
    except Exception:
        class NullContext:
            def __enter__(self): return self
            def __exit__(self, *args): return False
        return NullContext()


def _task_plan(readiness: pd.DataFrame, plan: dict[str, Any]) -> list[dict[str, Any]]:
    priorities = plan["priority"]
    tasks: list[dict[str, Any]] = []
    for ticker in readiness["ticker"].astype(str).str.zfill(6):
        tasks.append({
            "task_id": f"corr__{ticker}",
            "stage": "correlation_map",
            "ticker": ticker,
            "variant": "correlation",
            "seed": -1,
            "priority": int(priorities["correlation_map"]),
            "payload": {},
        })
    ready_mask = readiness["ready"].astype(str).str.lower().isin(["true", "1", "yes"])
    ready = readiness.loc[ready_mask].sort_values(["positives", "rows"], ascending=[False, False])
    for rank, ticker in enumerate(ready["ticker"].astype(str).str.zfill(6)):
        tasks.append({
            "task_id": f"elite__{ticker}",
            "stage": "elite_model",
            "ticker": ticker,
            "variant": "elite",
            "seed": -1,
            "priority": int(priorities["elite_model"]) + max(0, len(ready) - rank),
            "payload": {},
        })
    return tasks


def _wait_for_ram(plan: dict[str, Any], result_dir: Path) -> bool:
    """Wait without a wall-clock deadline until RAM recovers or the user requests stop."""
    import psutil

    soft = float(plan["min_available_ram_soft_gb"])
    while True:
        available = psutil.virtual_memory().available / 1024 ** 3
        if available >= soft:
            return True
        if (result_dir / "STOP_REQUESTED").exists():
            return False
        time.sleep(5)


def run_worker(
    project: Path,
    result_dir: Path,
    role: str,
    worker_index: int,
) -> int:
    plan = load_plan(project)
    layout = json.loads((result_dir / "runtime_worker_plan.json").read_text(encoding="utf-8"))
    threads = int(layout["threads_per_model_worker"] if role == "model" else layout["threads_per_correlation_worker"])
    _set_affinity(role, worker_index, layout)
    registry = TickerTaskRegistry(result_dir / "task_registry.sqlite")
    worker_name = f"{role}-{worker_index}"
    stop_flag = result_dir / "STOP_REQUESTED"
    profile = str(plan.get("execution_profile", "cpu4"))
    if role == "model" and profile == "cpu4":
        # One 4-thread worker handles both stages so total CPU use never exceeds four threads.
        stages = ("elite_model", "correlation_map")
    else:
        stages = ("elite_model",) if role == "model" else ("correlation_map",)
    with _thread_limit(threads):
        while not stop_flag.exists():
            if not _wait_for_ram(plan, result_dir):
                return 0
            worker_backend = (
                "cpu_only" if profile == "cpu4"
                else ("hybrid_fixed" if role == "model" else "cpu")
            )
            task = registry.claim_next(
                stages=stages,
                worker_name=worker_name,
                backend=worker_backend,
                threads=threads,
            )
            if task is None:
                return 0
            try:
                registry.heartbeat(task.task_id)
                data = load_cached_ticker(result_dir, task.ticker, plan, mmap=True)
                if task.stage == "correlation_map":
                    summary = compute_ticker_correlation_map(project, data, result_dir, plan)
                    result_path = result_dir / "correlation_maps" / task.ticker / "correlation_summary.json"
                    registry.complete(task.task_id, str(result_path))
                else:
                    summary = run_elite_ticker_task(
                        data,
                        result_dir,
                        plan,
                        threads=threads,
                        deadline_epoch=float("inf"),
                    )
                    result_path = result_dir / "ticker_models" / task.ticker / "task_summary.json"
                    if summary.get("status") == "partial":
                        registry.release(task.task_id, "manual stop partial; fold checkpoints preserved")
                        return 0
                    registry.complete(task.task_id, str(result_path))
                LOGGER.info("completed %s status=%s", task.task_id, summary.get("status"))
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                LOGGER.exception("task failed: %s attempt=%s", task.task_id, task.attempts)
                if task.attempts < int(plan.get("max_task_attempts", 2)) and not stop_flag.exists():
                    registry.release(task.task_id, f"retry after attempt {task.attempts}: {error}")
                else:
                    registry.fail(task.task_id, error)
            finally:
                gc.collect()
    return 0


def _nvidia_stats() -> dict[str, float]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=4,
        ).strip().splitlines()[0]
        values = [float(value.strip()) for value in output.split(",")]
        return {
            "gpu_utilization_pct": values[0],
            "vram_used_mb": values[1],
            "vram_total_mb": values[2],
            "gpu_temperature_c": values[3],
            "gpu_power_w": values[4],
        }
    except Exception:
        return {key: np.nan for key in ["gpu_utilization_pct", "vram_used_mb", "vram_total_mb", "gpu_temperature_c", "gpu_power_w"]}


def _resource_monitor(result_dir: Path, plan: dict[str, Any], done: threading.Event) -> None:
    import psutil

    path = result_dir / "resource_usage.csv"
    fields = [
        "timestamp", "cpu_percent", "ram_used_gb", "ram_available_gb", "ram_percent",
        "gpu_utilization_pct", "vram_used_mb", "vram_total_mb", "gpu_temperature_c", "gpu_power_w",
    ]
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()
    low_ram_count = 0
    high_temp_count = 0
    while not done.wait(float(plan["resource_poll_seconds"])):
        memory = psutil.virtual_memory()
        row = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": (memory.total - memory.available) / 1024 ** 3,
            "ram_available_gb": memory.available / 1024 ** 3,
            "ram_percent": memory.percent,
            **_nvidia_stats(),
        }
        with path.open("a", newline="", encoding="utf-8-sig") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        low_ram_count = low_ram_count + 1 if row["ram_available_gb"] < float(plan["min_available_ram_hard_gb"]) else 0
        temperature = row["gpu_temperature_c"]
        high_temp_count = high_temp_count + 1 if np.isfinite(temperature) and temperature >= float(plan["gpu_temperature_stop"]) else 0
        if low_ram_count >= 3 or high_temp_count >= 3:
            reason = "hard_low_ram" if low_ram_count >= 3 else "high_gpu_temperature"
            (result_dir / "STOP_REQUESTED").write_text(reason, encoding="utf-8")
            atomic_json({"reason": reason, "resource_row": row}, result_dir / "resource_stop.json")
            return


def _spawn_worker(project: Path, result_dir: Path, role: str, index: int) -> subprocess.Popen:
    command = [
        sys.executable,
        str(project / "run_ticker_elite2h.py"),
        "--worker-role", role,
        "--worker-index", str(index),
        "--result-dir", str(result_dir),
    ]
    return subprocess.Popen(command, cwd=project)


def run_supervisor(project: Path, dataset: Path | None = None, result_dir: Path | None = None) -> dict[str, Any]:
    project = project.resolve()
    plan = load_plan(project)
    paths = get_paths(project)
    result_dir = (result_dir or paths.data_root / str(plan.get("result_subdir", "ticker_independent_cpu4"))).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "STOP_REQUESTED").unlink(missing_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(processName)s %(message)s",
        handlers=[logging.FileHandler(result_dir / "ticker_independent.log", encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )

    started = time.time()
    readiness = prepare_ticker_cache(project, dataset, result_dir, plan)
    layout = choose_runtime_layout(plan)
    atomic_json(layout, result_dir / "runtime_worker_plan.json")
    LOGGER.info("runtime layout: %s", layout)

    registry = TickerTaskRegistry(result_dir / "task_registry.sqlite")
    registry.reset_stale(stale_seconds=1200)
    registry.ensure_tasks(_task_plan(readiness, plan))
    registry.set_meta("schema_version", plan["schema_version"])
    registry.set_meta("runtime_layout", layout)

    registry.set_meta("completion_mode", plan.get("completion_mode", "run_until_complete"))
    registry.set_meta("time_budget_enabled", False)
    done = threading.Event()
    monitor = threading.Thread(target=_resource_monitor, args=(result_dir, plan, done), daemon=True)
    monitor.start()

    processes: list[subprocess.Popen] = []

    def spawn_all() -> list[subprocess.Popen]:
        spawned: list[subprocess.Popen] = []
        for index in range(int(layout["model_workers"])):
            spawned.append(_spawn_worker(project, result_dir, "model", index))
        for index in range(int(layout["correlation_workers"])):
            spawned.append(_spawn_worker(project, result_dir, "correlation", index))
        return spawned

    processes = spawn_all()
    while True:
        status = registry.status()
        if status["tasks_unfinished"] == 0:
            break
        if (result_dir / "STOP_REQUESTED").exists():
            break
        failed_ratio = status["tasks_failed"] / max(1, status["tasks_total"])
        if failed_ratio >= float(plan["hard_stop_failed_ratio"]):
            (result_dir / "STOP_REQUESTED").write_text("excessive_task_failures", encoding="utf-8")
            break
        if processes and all(process.poll() is not None for process in processes):
            registry.reset_stale(stale_seconds=60)
            status = registry.status()
            if status["tasks_unfinished"] == 0:
                break
            LOGGER.warning("all workers exited with unfinished tasks; restarting workers")
            time.sleep(float(plan.get("worker_restart_delay_seconds", 5)))
            processes = spawn_all()
        time.sleep(5)

    for process in processes:
        process.wait()
    done.set()
    monitor.join(timeout=15)
    status = registry.status()
    summary = aggregate_results(result_dir, plan, status, paths.data_root)
    summary["elapsed_hours"] = (time.time() - started) / 3600.0
    summary["completion_mode"] = plan.get("completion_mode", "run_until_complete")
    summary["time_budget_enabled"] = False
    atomic_json(summary, result_dir / "run_summary.json")
    archive = pack_results(result_dir, int(plan["result_max_zip_mb"]))
    summary["result_archive"] = str(archive)
    atomic_json(summary, result_dir / "run_summary.json")
    LOGGER.info("Result archive: %s", archive)
    return summary
