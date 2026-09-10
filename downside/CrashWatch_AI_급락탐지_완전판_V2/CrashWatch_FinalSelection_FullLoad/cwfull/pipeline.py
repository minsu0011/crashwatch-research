from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
import threading
import time
from pathlib import Path
from typing import Any

import psutil

from cw7h.utils import atomic_json, hash_strings, read_json
from .aggregate import aggregate
from .common import (
    acquire_lock,
    load_context,
    nvml_snapshot,
    release_lock,
    resolve_output,
    set_full_load_mode,
    setup_logging,
)
from .lgb_engine import run_profiles as run_lgb_profiles
from .profiles import build_experiment_profiles, unpack_profiles
from .xgb_engine import run_profiles as run_xgb_profiles

LOGGER = logging.getLogger(__name__)


def _resource_monitor(path: Path, stop_event: threading.Event, started: float, interval: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch", "elapsed_seconds", "cpu_percent", "ram_used_gb", "ram_available_gb",
        "gpu_util_percent", "gpu_memory_util_percent", "gpu_memory_used_gb",
        "gpu_temperature_c", "gpu_power_w",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        while not stop_event.wait(interval):
            memory = psutil.virtual_memory()
            gpu = nvml_snapshot()
            writer.writerow(
                {
                    "epoch": time.time(),
                    "elapsed_seconds": time.time() - started,
                    "cpu_percent": psutil.cpu_percent(interval=None),
                    "ram_used_gb": memory.used / 1024**3,
                    "ram_available_gb": memory.available / 1024**3,
                    "gpu_util_percent": gpu.get("gpu_util_percent"),
                    "gpu_memory_util_percent": gpu.get("gpu_memory_util_percent"),
                    "gpu_memory_used_gb": gpu.get("gpu_memory_used_gb"),
                    "gpu_temperature_c": gpu.get("gpu_temperature_c"),
                    "gpu_power_w": gpu.get("gpu_power_w"),
                }
            )
            stream.flush()


def _xgb_entry(queue: mp.Queue, kwargs: dict[str, Any]) -> None:
    try:
        queue.put({"ok": True, "result": run_xgb_profiles(**kwargs)})
    except Exception as exc:
        import traceback
        queue.put({"ok": False, "error": repr(exc), "traceback": traceback.format_exc()})


def _config_variant(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(base))
    result.update(overrides)
    return result


def run_pipeline(
    package_root: Path,
    project_root: Path,
    cfg: dict[str, Any],
    *,
    dataset_override: str | None,
    output_override: str | None,
    stages: set[str] | None = None,
) -> dict[str, Any]:
    selected_stages = stages or {"core", "targeted", "diagnostic", "aggregate"}
    started = time.time()
    output = resolve_output(project_root, output_override, str(cfg["output_dir"]))
    setup_logging(output / "final_selection_full_load.log")
    lock_path = output / "pipeline.lock.json"
    acquire_lock(lock_path)
    monitor_stop = threading.Event()
    monitor = None
    xgb_process: mp.Process | None = None
    try:
        hardware = set_full_load_mode(int(cfg["cpu"]["total_threads"]), str(cfg["cpu"]["priority"]))
        atomic_json(hardware, output / "hardware_full_load.json")
        if int(hardware.get("logical_cpus", 0)) < 32:
            LOGGER.warning("32 logical threads보다 적습니다: %s", hardware)
        if float(hardware.get("ram_gb", 0)) < 80:
            LOGGER.warning("권장 RAM 96GB보다 적습니다: %.1fGB", float(hardware.get("ram_gb", 0)))

        if selected_stages == {"aggregate"} and (output / "experiment_profile_manifest.json").exists():
            manifest = read_json(output / "experiment_profile_manifest.json", {})
            selection_summary = aggregate(output, manifest, cfg)
            final = {
                "status": "completed",
                "elapsed_seconds": time.time() - started,
                "output_dir": str(output),
                "selected_stages": ["aggregate"],
                "hardware": hardware,
                "stage_results": {},
                "selection_summary": selection_summary,
                "sealed_evaluation_executed": False,
            }
            atomic_json(final, output / "FINAL_RUN_STATUS.json")
            return final

        prepared, refs, folds, legacy, best_iterations = load_context(
            package_root, project_root, dataset_override, output
        )
        manifest = build_experiment_profiles(package_root, prepared.feature_names, output)
        core_profiles = unpack_profiles(manifest, "core_operational")
        targeted_profiles = unpack_profiles(manifest, "targeted_operational")
        diagnostic_profiles = unpack_profiles(manifest, "diagnostic_full_column")
        xgb_profiles = {name: list(features) for name, features in manifest["xgboost_profiles"].items()}

        monitor = threading.Thread(
            target=_resource_monitor,
            args=(output / "resource_usage.csv", monitor_stop, started, float(cfg["runtime"]["resource_poll_seconds"])),
            daemon=True,
        )
        monitor.start()

        lightgbm_operational = _config_variant(cfg["lightgbm"], cfg["lightgbm_operational_overrides"])
        lightgbm_diagnostic = _config_variant(cfg["lightgbm"], cfg["lightgbm_diagnostic_overrides"])
        seeds = [int(seed) for seed in cfg["selection"]["seeds"]]
        task_root = output / "task_results"
        stage_results: dict[str, Any] = {}

        # Phase A: XGBoost consumes the RTX 5080 while LightGBM uses the remaining 24 CPU threads.
        if "core" in selected_stages:
            context = mp.get_context("spawn")
            xgb_queue: mp.Queue = context.Queue()
            xgb_kwargs = {
                "prepared": prepared,
                "folds": folds,
                "profiles": xgb_profiles,
                "seeds": seeds,
                "config": cfg["xgboost"],
                "task_dir": task_root / "xgboost",
                "workers": int(cfg["gpu"]["workers"]),
                "summary_path": output / "xgboost_stage_summary.json",
            }
            xgb_process = context.Process(
                target=_xgb_entry,
                args=(xgb_queue, xgb_kwargs),
                name="cw-final-xgboost-stage",
                daemon=False,
            )
            xgb_process.start()
            LOGGER.info("XGBoost P0/P2/P7 stage started, PID=%s", xgb_process.pid)

            stage_results["lightgbm_core_operational"] = run_lgb_profiles(
                prepared=prepared,
                folds=folds,
                profiles=core_profiles,
                seeds=seeds,
                best_iterations=best_iterations,
                config=lightgbm_operational,
                config_name="operational",
                task_dir=task_root / "lightgbm",
                workers=int(cfg["cpu"]["concurrent_lightgbm_workers"]),
                threads_per_worker=int(cfg["cpu"]["threads_per_lightgbm_worker"]),
                min_free_ram_gb=float(cfg["cpu"]["minimum_free_ram_gb"]),
                priority="above_normal",
                retries=int(cfg["runtime"]["failed_task_retries"]),
            )
            xgb_process.join()
            try:
                xgb_message = xgb_queue.get(timeout=15)
            except Exception:
                xgb_message = {"ok": False, "error": f"XGBoost process exit code={xgb_process.exitcode}"}
            stage_results["xgboost_p0_p2_p7"] = xgb_message.get("result", xgb_message)
            xgb_process = None

            # A failed/OOM task is retried after CPU/GPU concurrency has ended.
            xgb_result = stage_results["xgboost_p0_p2_p7"]
            if xgb_result.get("status") not in {"completed", "skipped"} or int(xgb_result.get("failed", 0)) > 0:
                LOGGER.warning("XGBoost partial result detected. Retrying without concurrent LightGBM.")
                stage_results["xgboost_retry"] = run_xgb_profiles(
                    prepared=prepared,
                    folds=folds,
                    profiles=xgb_profiles,
                    seeds=seeds,
                    config=cfg["xgboost"],
                    task_dir=task_root / "xgboost",
                    workers=max(1, int(cfg["gpu"]["retry_workers"])),
                    summary_path=output / "xgboost_retry_summary.json",
                )

        # Phase B: all CPU threads are available for operational context tests.
        if "targeted" in selected_stages:
            stage_results["lightgbm_targeted_operational"] = run_lgb_profiles(
                prepared=prepared,
                folds=folds,
                profiles=targeted_profiles,
                seeds=seeds,
                best_iterations=best_iterations,
                config=lightgbm_operational,
                config_name="operational",
                task_dir=task_root / "lightgbm",
                workers=int(cfg["cpu"]["full_lightgbm_workers"]),
                threads_per_worker=int(cfg["cpu"]["threads_per_lightgbm_worker"]),
                min_free_ram_gb=float(cfg["cpu"]["minimum_free_ram_gb"]),
                priority="above_normal",
                retries=int(cfg["runtime"]["failed_task_retries"]),
            )

        # Phase C: feature_fraction/bagging are disabled to separate true feature effects from sampling effects.
        if "diagnostic" in selected_stages:
            stage_results["lightgbm_full_column_diagnostic"] = run_lgb_profiles(
                prepared=prepared,
                folds=folds,
                profiles=diagnostic_profiles,
                seeds=seeds,
                best_iterations=best_iterations,
                config=lightgbm_diagnostic,
                config_name="diagnostic",
                task_dir=task_root / "lightgbm",
                workers=int(cfg["cpu"]["full_lightgbm_workers"]),
                threads_per_worker=int(cfg["cpu"]["threads_per_lightgbm_worker"]),
                min_free_ram_gb=float(cfg["cpu"]["minimum_free_ram_gb"]),
                priority="above_normal",
                retries=int(cfg["runtime"]["failed_task_retries"]),
            )

        selection_summary = {}
        if "aggregate" in selected_stages:
            selection_summary = aggregate(output, manifest, cfg)

        # Core and targeted operational profiles intentionally overlap (P0/P2/P7).
        # They share the same task cache, so count unique feature lists within each
        # model configuration rather than summing stage inputs and double-counting.
        operational_hashes: set[str] = set()
        if "core" in selected_stages:
            operational_hashes.update(hash_strings(features) for features in core_profiles.values())
        if "targeted" in selected_stages:
            operational_hashes.update(hash_strings(features) for features in targeted_profiles.values())
        diagnostic_hashes: set[str] = set()
        if "diagnostic" in selected_stages:
            diagnostic_hashes.update(hash_strings(features) for features in diagnostic_profiles.values())
        expected_lgb = (len(operational_hashes) + len(diagnostic_hashes)) * len(folds) * len(seeds)
        expected_xgb = len(xgb_profiles) * len(folds) * len(seeds) if "core" in selected_stages else 0
        actual_lgb = int(selection_summary.get("lightgbm_unique_completed_models", 0))
        actual_xgb = int(selection_summary.get("xgboost_completed_models", 0))
        required_stage_results = []
        for key in [
            "lightgbm_core_operational",
            "lightgbm_targeted_operational",
            "lightgbm_full_column_diagnostic",
        ]:
            if key in stage_results:
                required_stage_results.append(stage_results[key])
        if "core" in selected_stages:
            required_stage_results.append(stage_results.get("xgboost_retry", stage_results.get("xgboost_p0_p2_p7", {})))
        stages_ok = all(item.get("status") == "completed" for item in required_stage_results)
        counts_ok = (
            ("aggregate" not in selected_stages)
            or (actual_lgb >= expected_lgb and actual_xgb >= expected_xgb)
        )
        completion_audit = {
            "expected_lightgbm_models": expected_lgb,
            "actual_lightgbm_models": actual_lgb,
            "expected_xgboost_models": expected_xgb,
            "actual_xgboost_models": actual_xgb,
            "unique_operational_profiles": len(operational_hashes),
            "unique_diagnostic_profiles": len(diagnostic_hashes),
            "stage_statuses_ok": stages_ok,
            "model_counts_ok": counts_ok,
        }
        atomic_json(completion_audit, output / "completion_audit.json")

        final = {
            "status": "completed" if stages_ok and counts_ok else "incomplete_retry_required",
            "elapsed_seconds": time.time() - started,
            "output_dir": str(output),
            "selected_stages": sorted(selected_stages),
            "hardware": hardware,
            "stage_results": stage_results,
            "selection_summary": selection_summary,
            "completion_audit": completion_audit,
            "sealed_evaluation_executed": False,
        }
        atomic_json(final, output / "FINAL_RUN_STATUS.json")
        return final
    finally:
        if xgb_process is not None and xgb_process.is_alive():
            xgb_process.terminate()
            xgb_process.join(timeout=15)
            if xgb_process.is_alive():
                xgb_process.kill()
        monitor_stop.set()
        if monitor is not None:
            monitor.join(timeout=5)
        release_lock(lock_path)
