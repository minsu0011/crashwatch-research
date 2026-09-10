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
from .weak_search import (
    build_improvement_report, copy_previous_diagnostics, load_strength_map,
    plan_for_tier, strength_lookup, write_locked_manifest,
)

LOGGER = logging.getLogger(__name__)


def load_plan(project: Path) -> dict[str, Any]:
    plan = json.loads((project / "configs" / "ticker_elite2h_plan.json").read_text(encoding="utf-8"))
    profile = os.environ.get("CRASHWATCH_EXECUTION_PROFILE", "full").strip().lower()
    if profile != "full":
        raise ValueError("This package is the RTX 5080 full-load weak-search profile only.")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "4"
    plan.update({
        "execution_profile": "full",
        "backend_mode": "hybrid_fixed",
        "result_subdir": "ticker_weaksearch10h_full",
        "cache_namespace": "ticker_weaksearch10h_full_v1",
    })
    os.environ["CRASHWATCH_BACKEND_MODE"] = "hybrid_fixed"
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
    gpu = _gpu_inventory()
    # 9800X3D / RTX 5080 / 32GB target: four mixed-family workers.
    # Each worker runs CPU LightGBM or CUDA XGBoost/CatBoost sequentially, so this layout
    # Five concurrent CatBoost pools can deadlock on a final 2.39GB allocation on a
    # 16GB card. Four workers still saturate the GPU and use all 16 logical threads.
    model_workers = 4 if logical >= 16 else max(1, min(4, logical // 4))
    model_threads = 4 if logical >= 16 else max(1, logical // model_workers)
    if memory.total / 1024 ** 3 < 30 or memory.available / 1024 ** 3 < 12:
        model_workers = min(model_workers, 4)
    if float(gpu.get("vram_total_gb", 0.0)) < 14:
        model_workers = min(model_workers, 3)
    allocated = model_workers * model_threads
    return {
        "execution_profile": "full",
        "backend_mode": "hybrid_fixed",
        "model_workers": model_workers,
        "threads_per_model_worker": model_threads,
        "correlation_workers": 0,
        "threads_per_correlation_worker": 0,
        "logical_cores": logical,
        "total_ram_gb": memory.total / 1024 ** 3,
        "available_ram_gb": memory.available / 1024 ** 3,
        **gpu,
        "allocated_threads": allocated,
        "cache_mode": "per_ticker_float32_memmap",
        "backend_policy": "lightgbm_cpu_xgboost_catboost_cuda",
        "search_policy": "strong_locked_weak_broad_middle_focused",
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


def _task_plan(readiness: pd.DataFrame, plan: dict[str, Any], project: Path) -> list[dict[str, Any]]:
    strength = load_strength_map(project)
    ready_mask = readiness["ready"].astype(str).str.lower().isin(["true", "1", "yes"])
    ready = readiness.loc[ready_mask].copy()
    ready["ticker"] = ready["ticker"].astype(str).str.zfill(6)
    targets = strength.loc[strength["training_enabled"]].copy()
    merged = targets.merge(ready, on="ticker", how="inner", suffixes=("_strength", ""))
    tasks: list[dict[str, Any]] = []
    for _, row in merged.sort_values(["priority", "positives", "rows"], ascending=[False, False, False]).iterrows():
        ticker = str(row["ticker"]).zfill(6)
        tier = str(row["search_tier"])
        tasks.append({
            "task_id": f"weaksearch__{ticker}",
            "stage": "elite_model",
            "ticker": ticker,
            "variant": tier,
            "seed": -1,
            "priority": int(row["priority"]),
            "payload": {"search_tier": tier, "name": str(row.get("name", ""))},
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
        if (result_dir / "STOP_REQUESTED").exists() or (result_dir / "NO_NEW_TASKS").exists():
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
    no_new_flag = result_dir / "NO_NEW_TASKS"
    stages = ("elite_model",)
    lookup = strength_lookup(project)
    with _thread_limit(threads):
        while not stop_flag.exists():
            if no_new_flag.exists():
                return 0
            if not _wait_for_ram(plan, result_dir):
                return 0
            worker_backend = "hybrid_fixed"
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
                tier = str(task.payload.get("search_tier") or lookup.get(task.ticker, {}).get("search_tier", "middle_search"))
                local_plan = plan_for_tier(plan, tier)
                data = load_cached_ticker(result_dir, task.ticker, local_plan, mmap=True)
                summary = run_elite_ticker_task(
                    data, result_dir, local_plan, threads=threads, deadline_epoch=float("inf"),
                )
                result_path = result_dir / "ticker_models" / task.ticker / "task_summary.json"
                if summary.get("status") == "partial":
                    registry.release(task.task_id, "manual/hard stop partial; checkpoints preserved")
                    return 0
                if summary.get("status") == "failed":
                    raise RuntimeError(str(summary.get("reason", "ticker task failed")))
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
    (result_dir / "NO_NEW_TASKS").unlink(missing_ok=True)
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
    write_locked_manifest(project, result_dir, paths.data_root, plan)
    copy_previous_diagnostics(paths.data_root, result_dir, plan)
    registry.ensure_tasks(_task_plan(readiness, plan, project))
    registry.set_meta("schema_version", plan["schema_version"])
    registry.set_meta("runtime_layout", layout)

    registry.set_meta("completion_mode", plan.get("completion_mode", "soft_10h_then_finish_running_tickers"))
    registry.set_meta("time_budget_enabled", True)
    registry.set_meta("soft_budget_hours", float(plan.get("soft_budget_hours", 10.0)))
    registry.set_meta("hard_cap_hours", float(plan.get("hard_cap_hours", 12.0)))
    done = threading.Event()
    monitor = threading.Thread(target=_resource_monitor, args=(result_dir, plan, done), daemon=True)
    monitor.start()

    processes: dict[tuple[str, int], subprocess.Popen] = {}

    def spawn_all() -> dict[tuple[str, int], subprocess.Popen]:
        spawned: dict[tuple[str, int], subprocess.Popen] = {}
        for index in range(int(layout["model_workers"])):
            spawned[("model", index)] = _spawn_worker(project, result_dir, "model", index)
        for index in range(int(layout["correlation_workers"])):
            spawned[("correlation", index)] = _spawn_worker(project, result_dir, "correlation", index)
        return spawned

    processes = spawn_all()
    soft_budget = float(plan.get("soft_budget_hours", 10.0)) * 3600.0
    hard_cap = float(plan.get("hard_cap_hours", 12.0)) * 3600.0
    soft_reached = False
    while True:
        status = registry.status()
        elapsed = time.time() - started
        if status["tasks_unfinished"] == 0:
            break
        if (result_dir / "STOP_REQUESTED").exists():
            break
        if elapsed >= soft_budget and not soft_reached:
            soft_reached = True
            (result_dir / "NO_NEW_TASKS").write_text("soft_10h_budget_reached", encoding="utf-8")
            atomic_json({"elapsed_hours": elapsed / 3600.0, "status": status}, result_dir / "soft_budget_reached.json")
            LOGGER.warning("soft 10-hour budget reached; no new ticker tasks will be claimed")
        if elapsed >= hard_cap:
            (result_dir / "STOP_REQUESTED").write_text("hard_12h_cap_reached", encoding="utf-8")
            LOGGER.warning("hard cap reached; current ticker will stop at its next fold checkpoint")
            break
        failed_ratio = status["tasks_failed"] / max(1, status["tasks_total"])
        if failed_ratio >= float(plan["hard_stop_failed_ratio"]):
            (result_dir / "STOP_REQUESTED").write_text("excessive_task_failures", encoding="utf-8")
            break
        dead_slots: list[tuple[str, int]] = []
        for slot, process in list(processes.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            role, index = slot
            worker_name = f"{role}-{index}"
            recovered = registry.recover_dead_worker(
                worker_name,
                int(plan.get("max_task_attempts", 2)),
                f"worker {worker_name} exited unexpectedly with code {return_code}",
            )
            LOGGER.warning(
                "worker exited: worker=%s code=%s released=%s failed=%s",
                worker_name, return_code, recovered["released"], recovered["failed"],
            )
            dead_slots.append(slot)
            del processes[slot]
        if dead_slots and not soft_reached and not (result_dir / "STOP_REQUESTED").exists():
            status = registry.status()
            if status["tasks_pending"] > 0:
                time.sleep(float(plan.get("worker_restart_delay_seconds", 5)))
                for role, index in dead_slots:
                    processes[(role, index)] = _spawn_worker(project, result_dir, role, index)
                    LOGGER.warning("restarted worker: %s-%s", role, index)
        time.sleep(5)

    for process in processes.values():
        process.wait()
    done.set()
    monitor.join(timeout=15)
    status = registry.status()
    summary = aggregate_results(result_dir, plan, status, paths.data_root)
    improvement = build_improvement_report(project, result_dir, paths.data_root, plan)
    summary["strong_locked_count"] = int((load_strength_map(project)["search_tier"] == "strong_lock").sum())
    summary["weak_search_compared_count"] = int((improvement.get("comparison_status", pd.Series(dtype=str)) == "compared").sum()) if not improvement.empty else 0
    summary["new_strong_candidate_count"] = int(improvement.get("promoted_to_strong_candidate", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not improvement.empty else 0
    summary["elapsed_hours"] = (time.time() - started) / 3600.0
    summary["completion_mode"] = plan.get("completion_mode", "soft_10h_then_finish_running_tickers")
    summary["time_budget_enabled"] = True
    summary["soft_budget_hours"] = float(plan.get("soft_budget_hours", 10.0))
    summary["hard_cap_hours"] = float(plan.get("hard_cap_hours", 12.0))
    atomic_json(summary, result_dir / "run_summary.json")
    archive = pack_results(result_dir, int(plan["result_max_zip_mb"]))
    summary["result_archive"] = str(archive)
    atomic_json(summary, result_dir / "run_summary.json")
    LOGGER.info("Result archive: %s", archive)
    return summary
