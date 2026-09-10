from __future__ import annotations

import concurrent.futures as cf
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Iterable

import psutil

from .aggregate import aggregate_results
from .cache import PreparedCache
from .config import RuntimePaths
from .folds import FoldSlice, build_fold_slices
from .gpu import detect_nvidia_gpu, run_gpu_audit_sequence
from .model import BlockTask, TuneTask, run_lightgbm_block, tune_best_iteration
from .monitor import ResourceMonitor
from .references import ReferenceBundle
from .utils import atomic_json, read_json

LOGGER = logging.getLogger(__name__)


def _run_tuning(
    prepared: PreparedCache,
    output_dir: Path,
    config: dict[str, Any],
    folds_by_profile: dict[str, list[FoldSlice]],
) -> dict[str, dict[int, int]]:
    result_path = output_dir / "best_iterations.json"
    best: dict[str, dict[int, int]] = {"full_reduced": {}, "common_period": {}}
    tasks: list[TuneTask] = []
    for profile, folds in folds_by_profile.items():
        for fold in folds:
            if not fold.eligible:
                continue
            tasks.append(TuneTask(
                cache_root=str(prepared.root),
                profile=profile,
                fold=fold.to_dict(),
                dataset_signature=prepared.dataset_signature,
                lightgbm_config=config["lightgbm"],
                threads=int(config["threads_per_worker"]),
                inner_validation_days=int(config["inner_validation_days"]),
                inner_purge_days=int(config["inner_purge_days"]),
                min_train_days=int(config["min_train_days"]),
                seed=17,
            ))
    ctx = mp.get_context("spawn")
    max_workers = min(int(config["workers"]), max(1, len(tasks)))
    LOGGER.info("tuning best_iteration: %s tasks, workers=%s", len(tasks), max_workers)
    results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = [executor.submit(tune_best_iteration, task) for task in tasks]
        for future in cf.as_completed(futures):
            result = future.result()
            results.append(result)
            if result.get("status") != "completed":
                raise RuntimeError(f"best_iteration tuning failed: {result}")
            best[result["profile"]][int(result["outer_fold"])] = int(result["best_iteration"])
            LOGGER.info("tuned %s fold=%s best_iteration=%s cache=%s", result["profile"], result["outer_fold"], result["best_iteration"], result.get("cache_status"))
    atomic_json({"best_iterations": best, "tasks": results}, result_path)
    return best


def _build_blocks(
    prepared: PreparedCache,
    config: dict[str, Any],
    folds_by_profile: dict[str, list[FoldSlice]],
    best_iterations: dict[str, dict[int, int]],
    phase: str,
    profiles: list[str],
    conditions: list[str],
    seeds: list[int],
    fold_ids: list[int],
    deadline_epoch: float,
) -> list[BlockTask]:
    fold_maps = {profile: {fold.fold_id: fold for fold in folds} for profile, folds in folds_by_profile.items()}
    blocks: list[BlockTask] = []
    for fold_id in fold_ids:
        for profile in profiles:
            fold = fold_maps[profile].get(int(fold_id))
            if fold is None:
                continue
            for condition in conditions:
                if not prepared.matrix_path(profile, condition).exists():
                    continue
                blocks.append(BlockTask(
                    cache_root=str(prepared.root),
                    profile=profile,
                    condition=condition,
                    phase=phase,
                    fold=fold.to_dict(),
                    seeds=list(seeds),
                    best_iteration=int(best_iterations[profile].get(int(fold_id), min(600, int(config["lightgbm"]["max_rounds"])))),
                    dataset_signature=prepared.dataset_signature,
                    lightgbm_config=config["lightgbm"],
                    threads=int(config["threads_per_worker"]),
                    deadline_epoch=deadline_epoch,
                    save_predictions=bool(config.get("save_predictions", True)),
                    prediction_compression=str(config.get("prediction_compression", "zstd")),
                ))
    return blocks


def _execute_blocks(
    blocks: list[BlockTask],
    workers: int,
    soft_deadline_epoch: float,
    hard_deadline_epoch: float,
    minimum_free_ram_gb: float,
    phase_name: str,
) -> dict[str, Any]:
    if not blocks:
        return {"phase": phase_name, "submitted": 0, "completed_blocks": 0, "failed_blocks": 0, "skipped_blocks": 0}
    ctx = mp.get_context("spawn")
    iterator = iter(blocks)
    in_flight: dict[cf.Future, BlockTask] = {}
    completed_blocks = 0
    failed_blocks = 0
    skipped_blocks = 0
    submitted = 0
    stop_submitting = False

    def try_submit(executor: cf.ProcessPoolExecutor) -> bool:
        nonlocal submitted, stop_submitting, skipped_blocks
        if stop_submitting:
            return False
        if time.time() >= soft_deadline_epoch and phase_name not in {"canary", "core", "battery"}:
            stop_submitting = True
            return False
        if time.time() >= hard_deadline_epoch:
            stop_submitting = True
            return False
        available_gb = psutil.virtual_memory().available / 1024**3
        if available_gb < minimum_free_ram_gb:
            return False
        try:
            block = next(iterator)
        except StopIteration:
            stop_submitting = True
            return False
        future = executor.submit(run_lightgbm_block, block)
        in_flight[future] = block
        submitted += 1
        return True

    LOGGER.info("phase=%s blocks=%s workers=%s", phase_name, len(blocks), workers)
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        while len(in_flight) < workers * 2 and try_submit(executor):
            pass
        while in_flight:
            done, _ = cf.wait(in_flight, timeout=5, return_when=cf.FIRST_COMPLETED)
            if not done:
                if time.time() >= hard_deadline_epoch:
                    stop_submitting = True
                while len(in_flight) < workers * 2 and try_submit(executor):
                    pass
                continue
            for future in done:
                block = in_flight.pop(future)
                try:
                    result = future.result()
                    status = result.get("status")
                    if status == "completed":
                        completed_blocks += 1
                    elif status in {"deadline_skipped", "ineligible"}:
                        skipped_blocks += 1
                    else:
                        failed_blocks += 1
                    LOGGER.info(
                        "phase=%s fold=%s profile=%s condition=%s status=%s cache_only=%s",
                        phase_name, block.fold["fold_id"], block.profile, block.condition, status, result.get("cache_only")
                    )
                except Exception as exc:
                    failed_blocks += 1
                    LOGGER.exception("block crashed: %s", exc)
            while len(in_flight) < workers * 2 and try_submit(executor):
                pass
        if submitted < len(blocks):
            skipped_blocks += len(blocks) - submitted
    return {
        "phase": phase_name,
        "planned_blocks": len(blocks),
        "submitted": submitted,
        "completed_blocks": completed_blocks,
        "failed_blocks": failed_blocks,
        "skipped_blocks": skipped_blocks,
    }


def run_experiment(
    prepared: PreparedCache,
    paths: RuntimePaths,
    config: dict[str, Any],
    refs: ReferenceBundle,
    *,
    started_epoch: float | None = None,
) -> dict[str, Any]:
    started_epoch = started_epoch or time.time()
    output_dir = paths.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    soft_deadline = started_epoch + float(config["soft_runtime_minutes"]) * 60
    hard_deadline = started_epoch + float(config["hard_runtime_minutes"]) * 60
    monitor = ResourceMonitor(
        output_dir / "resource_usage.csv",
        output_dir,
        float(config.get("resource_poll_seconds", 5)),
        started_epoch,
        sample_gpu=bool(config.get("monitor_gpu", True)),
    )
    monitor.start()
    phase_summaries: list[dict[str, Any]] = []
    gpu_process: mp.Process | None = None
    gpu_enabled = False
    try:
        folds_by_profile: dict[str, list[FoldSlice]] = {}
        for profile in ["full_reduced", "common_period"]:
            folds = build_fold_slices(
                prepared.profile_dir(profile),
                refs.folds,
                int(config["min_train_days"]),
                int(config["validation_days"]),
                int(config["purge_days"]),
                output_dir / f"fold_manifest_{profile}.json",
            )
            folds_by_profile[profile] = folds
            ineligible = [fold.to_dict() for fold in folds if not fold.eligible]
            if ineligible:
                LOGGER.warning("profile=%s ineligible folds=%s", profile, [(f["fold_id"], f["reason"]) for f in ineligible])

        best_iterations = _run_tuning(prepared, output_dir, config, folds_by_profile)

        # Canary is part of the main cache: same task keys, only seed 17.
        canary_blocks = _build_blocks(
            prepared, config, folds_by_profile, best_iterations,
            phase="canary", profiles=["full_reduced", "common_period"],
            conditions=["B0", "A1", "A2", "A3"], seeds=[17], fold_ids=[7],
            deadline_epoch=hard_deadline,
        )
        canary_summary = _execute_blocks(
            canary_blocks,
            workers=min(2, int(config["workers"])),
            soft_deadline_epoch=soft_deadline,
            hard_deadline_epoch=hard_deadline,
            minimum_free_ram_gb=float(config["minimum_free_ram_gb"]),
            phase_name="canary",
        )
        phase_summaries.append(canary_summary)
        if canary_summary["failed_blocks"] > 0 or canary_summary["completed_blocks"] < len(canary_blocks):
            raise RuntimeError(f"canary failed: {canary_summary}")

        if config.get("enable_gpu_audit", True):
            gpu_info = detect_nvidia_gpu()
        else:
            gpu_info = {"available": False, "reason": "sealed_by_config"}
        gpu_enabled = bool(config.get("enable_gpu_audit", True) and gpu_info.get("available"))
        worker_count = int(config["workers_with_gpu_audit"] if gpu_enabled else config["workers"])
        if gpu_enabled:
            gpu_payload = {
                "cache_root": str(prepared.root),
                "output_dir": str(output_dir),
                "dataset_signature": prepared.dataset_signature,
                "profiles": list(config["gpu_audit_profiles"]),
                "fold_ids": list(config["gpu_audit_folds"]),
                "seeds": list(config["gpu_audit_seeds"]),
                "conditions": list(config["gpu_audit_conditions"]),
                "folds_by_profile": {k: [f.to_dict() for f in v] for k, v in folds_by_profile.items()},
                "xgboost_config": config["xgboost_gpu"],
                "threads": int(config["gpu_audit_threads"]),
                "deadline_epoch": hard_deadline,
                "prediction_compression": config["prediction_compression"],
            }
            ctx = mp.get_context("spawn")
            gpu_process = ctx.Process(target=run_gpu_audit_sequence, args=(gpu_payload,), name="xgboost-gpu-audit")
            gpu_process.start()
            LOGGER.info("GPU audit started pid=%s gpu=%s", gpu_process.pid, gpu_info)
        else:
            atomic_json({"status": "disabled_or_unavailable", "gpu": gpu_info}, output_dir / "gpu_audit_status.json")

        core_blocks = _build_blocks(
            prepared, config, folds_by_profile, best_iterations,
            phase="core", profiles=["full_reduced", "common_period"],
            conditions=list(config["core_conditions"]), seeds=list(config["core_seeds"]),
            fold_ids=list(config["recent_fold_order"]), deadline_epoch=hard_deadline,
        )
        phase_summaries.append(_execute_blocks(
            core_blocks, worker_count, soft_deadline, hard_deadline,
            float(config["minimum_free_ram_gb"]), "core"
        ))

        # Battery is confirmatory and therefore runs before the optional ETF extension.
        if time.time() < hard_deadline:
            battery_blocks = _build_blocks(
                prepared, config, folds_by_profile, best_iterations,
                phase="battery", profiles=["common_period"],
                conditions=list(config["battery_conditions"]), seeds=list(config["optional_seeds"]),
                fold_ids=list(config["recent_fold_order"]), deadline_epoch=hard_deadline,
            )
            phase_summaries.append(_execute_blocks(
                battery_blocks, worker_count, soft_deadline, hard_deadline,
                float(config["minimum_free_ram_gb"]), "battery"
            ))
        else:
            phase_summaries.append({"phase": "battery", "status": "hard_deadline_skipped"})

        if time.time() < soft_deadline:
            etf_blocks = _build_blocks(
                prepared, config, folds_by_profile, best_iterations,
                phase="etf", profiles=["full_reduced", "common_period"],
                conditions=list(config["etf_conditions"]), seeds=list(config["optional_seeds"]),
                fold_ids=list(config["recent_fold_order"]), deadline_epoch=hard_deadline,
            )
            phase_summaries.append(_execute_blocks(
                etf_blocks, worker_count, soft_deadline, hard_deadline,
                float(config["minimum_free_ram_gb"]), "etf"
            ))
        else:
            phase_summaries.append({"phase": "etf", "status": "soft_deadline_skipped"})

        if gpu_process is not None:
            remaining = max(0.0, hard_deadline - time.time())
            gpu_process.join(timeout=remaining)
            if gpu_process.is_alive():
                LOGGER.warning("GPU audit still alive at hard deadline; terminating")
                gpu_process.terminate()
                gpu_process.join(timeout=10)
                atomic_json({"status": "terminated_at_hard_deadline", "pid": gpu_process.pid}, output_dir / "gpu_audit_status.json")

        run_metadata = {
            "dataset_path": str(prepared.dataset_path),
            "dataset_signature": prepared.dataset_signature,
            "cache_root": str(prepared.root),
            "cache_bytes": prepared.manifest.get("cache_bytes"),
            "config_hash": config.get("config_hash"),
            "common_period_mode": config.get("common_period_mode"),
            "phase_summaries": phase_summaries,
            "gpu_audit_enabled": gpu_enabled,
            "started_epoch": started_epoch,
            "elapsed_seconds": time.time() - started_epoch,
            "soft_runtime_minutes": config["soft_runtime_minutes"],
            "hard_runtime_minutes": config["hard_runtime_minutes"],
        }
        atomic_json(run_metadata, output_dir / "execution_audit.json")
        return aggregate_results(prepared.root, output_dir, config, run_metadata)
    finally:
        if gpu_process is not None and gpu_process.is_alive():
            gpu_process.terminate()
            gpu_process.join(timeout=10)
        monitor.stop()
