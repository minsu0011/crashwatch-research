from __future__ import annotations

import concurrent.futures as cf
import json
import logging
import math
import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import psutil

from .aggregate import aggregate_results
from .correlation import run_correlation_audit
from .data import PreparedData, References, discover_dataset, load_references, prepare_data
from .folds import FoldSlice, build_fold_slices
from .gpu_worker import run_gpu_feature_audit
from .lgb_worker import LgbBlock, ModelCondition, run_lgb_block, tune_fold_iteration
from .monitor import ResourceMonitor
from .utils import atomic_json, canonical_hash, ensure_thread_env, read_json

LOGGER = logging.getLogger(__name__)


def _set_full_load_priority(config: dict[str, Any]) -> dict[str, Any]:
    process = psutil.Process(os.getpid())
    result: dict[str, Any] = {"logical_cpus": psutil.cpu_count(logical=True), "affinity": [], "priority": "unchanged"}
    try:
        cpus = list(range(psutil.cpu_count(logical=True) or 1))
        process.cpu_affinity(cpus)
        result["affinity"] = process.cpu_affinity()
    except Exception as exc:
        result["affinity_error"] = repr(exc)
    if bool(config.get("above_normal_priority", True)):
        try:
            if os.name == "nt":
                process.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
                result["priority"] = "above_normal"
            else:
                result["priority"] = "normal_non_windows"
        except Exception as exc:
            result["priority_error"] = repr(exc)
    return result


def _run_tuning(
    prepared: PreparedData,
    folds: list[FoldSlice],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[int, int]:
    tasks: list[dict[str, Any]] = []
    for fold in folds:
        if not fold.eligible:
            continue
        tasks.append({
            "cache_root": str(prepared.root),
            "output_dir": str(output_dir),
            "dataset_signature": prepared.signature,
            "fold": fold.to_dict(),
            "model_config": config["lightgbm"],
            "threads": int(config["cpu"]["threads_per_worker"]),
            "rolling_windows": int(config["tuning"]["rolling_windows"]),
            "rolling_step_days": int(config["tuning"]["rolling_step_days"]),
            "inner_validation_days": int(config["tuning"]["inner_validation_days"]),
            "inner_purge_days": int(config["tuning"]["inner_purge_days"]),
            "min_train_days": int(config["min_train_days"]),
        })
    ctx = mp.get_context("spawn")
    workers = min(int(config["cpu"]["workers"]), len(tasks))
    results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx) as executor:
        futures = [executor.submit(tune_fold_iteration, task) for task in tasks]
        for future in cf.as_completed(futures):
            result = future.result()
            results.append(result)
            LOGGER.info("tuning fold=%s status=%s best=%s", result.get("fold_id"), result.get("status"), result.get("best_iteration"))
    failed = [r for r in results if r.get("status") != "completed"]
    if failed:
        raise RuntimeError(f"best-iteration tuning failed: {failed[:2]}")
    cap = int(config["lightgbm"].get("effective_round_cap", config["lightgbm"]["max_rounds"]))
    best = {int(r["fold_id"]): min(cap, int(r["best_iteration"])) for r in results}
    atomic_json({"best_iterations": best, "tasks": results, "effective_round_cap": cap}, output_dir / "best_iterations.json")
    return best


def _condition_all_baseline() -> ModelCondition:
    return ModelCondition("B0_ALL_VALID", "baseline_all_valid", "", -1, None, None, True)


def _loo_conditions(feature_names: list[str], save_predictions: bool) -> list[ModelCondition]:
    return [
        ModelCondition(
            condition_id=f"LOO::{feature}",
            test_type="single_feature_loo",
            feature=feature,
            feature_index=i,
            cluster_id=None,
            enabled_indices=None,
            save_prediction=save_predictions,
        )
        for i, feature in enumerate(feature_names)
    ]


def _conditional_conditions(feature_names: list[str], primary_clusters: pd.DataFrame) -> tuple[ModelCondition, list[ModelCondition]]:
    primary_clusters = primary_clusters.copy()
    rep_features = primary_clusters.loc[primary_clusters["is_representative"].astype(bool), "feature"].astype(str).tolist()
    index = {name: i for i, name in enumerate(feature_names)}
    rep_indices = tuple(sorted(index[name] for name in rep_features if name in index))
    baseline = ModelCondition("B0_CORRELATION_PRUNED", "baseline_pruned", "", -1, None, rep_indices, False)
    by_feature = primary_clusters.set_index("feature")
    conditions: list[ModelCondition] = []
    rep_set = set(rep_features)
    for i, feature in enumerate(feature_names):
        row = by_feature.loc[feature]
        cluster_id = int(row["cluster_id"])
        if feature in rep_set:
            enabled = tuple(x for x in rep_indices if x != i)
            test_type = "conditional_drop_rep"
            cid = f"COND_DROP_REP::{feature}"
        else:
            enabled = tuple(sorted(set(rep_indices) | {i}))
            test_type = "conditional_add_nonrep"
            cid = f"COND_ADD_NONREP::{feature}"
        conditions.append(ModelCondition(cid, test_type, feature, i, cluster_id, enabled, False))
    return baseline, conditions


def _cluster_conditions(feature_names: list[str], primary_clusters: pd.DataFrame) -> list[ModelCondition]:
    index = {name: i for i, name in enumerate(feature_names)}
    all_indices = set(range(len(feature_names)))
    conditions: list[ModelCondition] = []
    for cluster_id, part in primary_clusters.groupby("cluster_id", sort=True):
        members = [index[name] for name in part["feature"].astype(str) if name in index]
        enabled = tuple(sorted(all_indices - set(members)))
        representative = str(part["representative"].iloc[0])
        conditions.append(ModelCondition(
            condition_id=f"CLUSTER_LOO::{int(cluster_id)}",
            test_type="cluster_loo",
            feature=representative,
            feature_index=index.get(representative, -1),
            cluster_id=int(cluster_id),
            enabled_indices=enabled,
            save_prediction=False,
        ))
    return conditions


def _shard_conditions(conditions: list[ModelCondition], size: int) -> list[tuple[ModelCondition, ...]]:
    return [tuple(conditions[i:i + size]) for i in range(0, len(conditions), size)]


def _build_blocks(
    prepared: PreparedData,
    output_dir: Path,
    folds: list[FoldSlice],
    best_iterations: dict[int, int],
    conditions: list[ModelCondition],
    config: dict[str, Any],
    stage: str,
    deadline_epoch: float,
    shard_size: int,
    run_signature: str,
) -> list[LgbBlock]:
    shards = _shard_conditions(conditions, max(1, shard_size))
    fold_order = list(config["cpu"].get("fold_order", [7, 6, 5, 4, 3, 2, 1, 0]))
    fold_map = {f.fold_id: f for f in folds if f.eligible}
    blocks: list[LgbBlock] = []
    # Interleave folds per shard. This avoids completing every feature for one fold while another fold has not started.
    for shard in shards:
        for fold_id in fold_order:
            fold = fold_map.get(int(fold_id))
            if fold is None:
                continue
            blocks.append(LgbBlock(
                cache_root=str(prepared.root),
                output_dir=str(output_dir),
                dataset_signature=prepared.signature,
                run_signature=run_signature,
                fold=fold.to_dict(),
                conditions=shard,
                best_iteration=int(best_iterations[fold.fold_id]),
                model_config=config["lightgbm"],
                threads=int(config["cpu"]["threads_per_worker"]),
                deadline_epoch=deadline_epoch,
                feature_names=tuple(prepared.feature_names),
                stage=stage,
            ))
    return blocks


def _execute_blocks(blocks: list[LgbBlock], workers: int, stage: str) -> dict[str, Any]:
    if not blocks:
        return {"stage": stage, "blocks": 0, "trained": 0, "cached": 0, "failed": 0, "deadline_skipped": 0, "elapsed_seconds": 0.0}
    started = time.time()
    ctx = mp.get_context("spawn")
    totals = {"trained": 0, "cached": 0, "failed": 0, "deadline_skipped": 0}
    with cf.ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx) as executor:
        futures = [executor.submit(run_lgb_block, block) for block in blocks]
        completed_blocks = 0
        for future in cf.as_completed(futures):
            result = future.result()
            completed_blocks += 1
            for key in totals:
                totals[key] += int(result.get(key, 0))
            if completed_blocks % max(1, min(10, len(blocks))) == 0 or result.get("failed"):
                LOGGER.info(
                    "stage=%s blocks=%s/%s trained=%s cached=%s failed=%s skipped=%s",
                    stage, completed_blocks, len(blocks), totals["trained"], totals["cached"], totals["failed"], totals["deadline_skipped"],
                )
    return {"stage": stage, "blocks": len(blocks), **totals, "elapsed_seconds": time.time() - started}


def _canary_and_adjust_iterations(
    prepared: PreparedData,
    output_dir: Path,
    folds: list[FoldSlice],
    best_iterations: dict[int, int],
    priority: list[str],
    config: dict[str, Any],
    deadline_epoch: float,
) -> dict[int, int]:
    fold = max((f for f in folds if f.eligible), key=lambda f: f.fold_id)
    index = {name: i for i, name in enumerate(prepared.feature_names)}
    selected = [f for f in priority if f in index][:int(config["runtime"].get("canary_models", 6))]
    conditions = [
        ModelCondition(f"CANARY::{f}", "canary_loo", f, index[f], None, None, False) for f in selected
    ]
    block = LgbBlock(
        cache_root=str(prepared.root),
        output_dir=str(output_dir),
        dataset_signature=prepared.signature,
        run_signature=f"canary-{config.get('config_hash', '')}",
        fold=fold.to_dict(),
        conditions=tuple(conditions),
        best_iteration=int(best_iterations[fold.fold_id]),
        model_config=config["lightgbm"],
        threads=int(config["cpu"]["threads_per_worker"]),
        deadline_epoch=deadline_epoch,
        feature_names=tuple(prepared.feature_names),
        stage="canary",
    )
    result = run_lgb_block(block)
    mean_seconds = float(result.get("mean_model_seconds", float("nan")))
    if not np.isfinite(mean_seconds):
        elapsed = []
        result_dir = output_dir / "task_results" / "lightgbm_cpu"
        for path in result_dir.glob("*.json") if result_dir.exists() else []:
            row = read_json(path, {})
            if (
                row.get("status") == "completed"
                and row.get("test_type") == "canary_loo"
                and row.get("dataset_signature") == prepared.signature
                and row.get("run_signature") == f"canary-{config.get('config_hash', '')}"
            ):
                value = float(row.get("elapsed_seconds", float("nan")))
                if np.isfinite(value):
                    elapsed.append(value)
        if elapsed:
            mean_seconds = float(np.mean(elapsed))
    workers = int(config["cpu"]["workers"])
    model_count = len(prepared.feature_names) * len([f for f in folds if f.eligible])
    safety = float(config["runtime"].get("estimate_safety_factor", 1.35))
    estimated_primary = mean_seconds * model_count / max(1.0, workers * 0.82) * safety if np.isfinite(mean_seconds) else float("nan")
    remaining = max(1.0, deadline_epoch - time.time())
    target_fraction = float(config["runtime"].get("mandatory_budget_fraction", 0.55))
    target = remaining * target_fraction
    adjusted = dict(best_iterations)
    scale = 1.0
    if np.isfinite(estimated_primary) and estimated_primary > target:
        scale = max(float(config["runtime"].get("minimum_iteration_scale", 0.45)), target / estimated_primary)
        minimum = int(config["lightgbm"].get("min_effective_rounds", 40))
        adjusted = {fold_id: max(minimum, int(round(value * scale))) for fold_id, value in best_iterations.items()}
    plan = {
        "canary": result,
        "mean_seconds_per_model_worst_fold": mean_seconds,
        "estimated_primary_wall_seconds": estimated_primary,
        "remaining_seconds_after_canary": remaining,
        "mandatory_target_seconds": target,
        "iteration_scale": scale,
        "best_iterations_before": best_iterations,
        "best_iterations_after": adjusted,
    }
    atomic_json(plan, output_dir / "runtime_calibration.json")
    return adjusted


def _task_completion(output_dir: Path, test_type: str, dataset_signature: str, run_signature: str) -> int:
    result_dir = output_dir / "task_results" / "lightgbm_cpu"
    count = 0
    for path in result_dir.glob("*.json") if result_dir.exists() else []:
        row = read_json(path, {})
        if (
            row.get("status") == "completed"
            and row.get("test_type") == test_type
            and row.get("dataset_signature") == dataset_signature
            and row.get("run_signature") == run_signature
        ):
            count += 1
    return count


def run_experiment(
    package_root: Path,
    project_root: Path,
    config: dict[str, Any],
    dataset_override: str | None = None,
    *,
    force_cache: bool = False,
    force_correlation: bool = False,
) -> dict[str, Any]:
    started_epoch = time.time()
    hard_deadline = started_epoch + float(config["runtime"]["hard_runtime_minutes"]) * 60.0
    model_deadline = hard_deadline - float(config["runtime"].get("aggregation_reserve_minutes", 7)) * 60.0
    gpu_soft_deadline = started_epoch + float(config["runtime"].get("gpu_soft_runtime_minutes", 390)) * 60.0
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    cache_dir = Path(config["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(output_dir / "resource_usage.csv", float(config["runtime"].get("resource_poll_seconds", 5)))
    monitor.start()
    gpu_process: mp.Process | None = None
    phase_summaries: list[dict[str, Any]] = []
    try:
        runtime_limits = _set_full_load_priority(config["cpu"])
        atomic_json(runtime_limits, output_dir / "runtime_hardware_plan.json")
        refs = load_references(package_root / "reference", strict_counts=bool(config.get("strict_reference_counts", True)))
        dataset = discover_dataset(project_root, str(config.get("dataset_path", "AUTO")), dataset_override, list(config.get("sealed_path_tokens", [])))
        prepared = prepare_data(dataset, cache_dir, config, refs, force=force_cache)
        dates = np.load(prepared.dates_path, mmap_mode="r")
        folds = build_fold_slices(
            dates,
            refs.folds,
            min_train_days=int(config["min_train_days"]),
            validation_days=int(config["validation_days"]),
            purge_days=int(config["purge_days"]),
            output_path=output_dir / "fold_manifest.json",
        )
        invalid = [f for f in folds if not f.eligible]
        if invalid:
            raise RuntimeError(f"ineligible folds: {[f.to_dict() for f in invalid]}")
        atomic_json({
            "config": config,
            "project_root": str(project_root),
            "package_root": str(package_root),
            "dataset_path": str(dataset),
            "dataset_signature": prepared.signature,
            "started_epoch": started_epoch,
            "hard_deadline_epoch": hard_deadline,
            "model_deadline_epoch": model_deadline,
        }, output_dir / "resolved_run_config.json")

        correlation_manifest = run_correlation_audit(prepared, refs, folds, output_dir, config, force=force_correlation)
        phase_summaries.append({"phase": "correlation", **correlation_manifest})
        best_iterations = _run_tuning(prepared, folds, output_dir, config)
        priority = pd.read_csv(output_dir / "correlation" / "feature_priority.csv")["feature"].astype(str).tolist()
        best_iterations = _canary_and_adjust_iterations(prepared, output_dir, folds, best_iterations, priority, config, model_deadline)
        atomic_json(best_iterations, output_dir / "effective_best_iterations.json")
        run_signature = canonical_hash({
            "dataset_signature": prepared.signature,
            "config_hash": config.get("config_hash", ""),
            "best_iterations": best_iterations,
            "correlation_hash": correlation_manifest.get("correlation_hash", ""),
        })
        atomic_json({"run_signature": run_signature}, output_dir / "run_signature.json")

        if bool(config["gpu"].get("enabled", True)):
            gpu_plan = {
                "cache_root": str(prepared.root),
                "output_dir": str(output_dir),
                "dataset_signature": prepared.signature,
                "run_signature": run_signature,
                "feature_names": prepared.feature_names,
                "folds": [f.to_dict() for f in folds],
                "gpu_config": config["gpu"],
                "soft_deadline_epoch": min(gpu_soft_deadline, model_deadline),
            }
            ctx = mp.get_context("spawn")
            gpu_process = ctx.Process(target=run_gpu_feature_audit, args=(gpu_plan,), name="cw7h-xgb-cuda-audit", daemon=False)
            gpu_process.start()
            LOGGER.info("started optional XGBoost CUDA audit pid=%s", gpu_process.pid)

        workers = int(config["cpu"]["workers"])
        baseline_blocks = _build_blocks(
            prepared, output_dir, folds, best_iterations, [_condition_all_baseline()], config,
            stage="baseline", deadline_epoch=model_deadline, shard_size=1, run_signature=run_signature,
        )
        phase_summaries.append(_execute_blocks(baseline_blocks, workers, "baseline"))
        expected_baselines = len([f for f in folds if f.eligible])
        completed_baselines = _task_completion(output_dir, "baseline_all_valid", prepared.signature, run_signature)
        if completed_baselines < expected_baselines and time.time() < model_deadline - 60:
            LOGGER.warning("baseline incomplete (%s/%s); retrying once", completed_baselines, expected_baselines)
            retry = _execute_blocks(baseline_blocks, workers, "baseline_retry_once")
            retry["stage"] = "baseline_retry_once"
            phase_summaries.append(retry)
            completed_baselines = _task_completion(output_dir, "baseline_all_valid", prepared.signature, run_signature)
        if completed_baselines < expected_baselines:
            raise RuntimeError(f"baseline models incomplete: {completed_baselines}/{expected_baselines}")

        loo = _loo_conditions(prepared.feature_names, bool(config["output"].get("save_primary_predictions", False)))
        primary_blocks = _build_blocks(
            prepared, output_dir, folds, best_iterations, loo, config,
            stage="primary_single_feature_loo", deadline_epoch=model_deadline,
            shard_size=int(config["cpu"].get("models_per_block", 24)), run_signature=run_signature,
        )
        primary_summary = _execute_blocks(primary_blocks, workers, "primary_single_feature_loo")
        phase_summaries.append(primary_summary)
        expected_primary = len(prepared.feature_names) * len(folds)
        completed_primary = _task_completion(output_dir, "single_feature_loo", prepared.signature, run_signature)
        if completed_primary < expected_primary and time.time() < model_deadline - 90:
            LOGGER.warning("primary LOO incomplete (%s/%s); retrying failed/missing tasks once", completed_primary, expected_primary)
            retry = _execute_blocks(primary_blocks, workers, "primary_single_feature_loo_retry_once")
            retry["stage"] = "primary_single_feature_loo_retry_once"
            phase_summaries.append(retry)
            completed_primary = _task_completion(output_dir, "single_feature_loo", prepared.signature, run_signature)
        atomic_json({
            "expected": expected_primary,
            "completed": completed_primary,
            "complete": completed_primary >= expected_primary,
            "retry_once_enabled": True,
        }, output_dir / "primary_completion_audit.json")

        primary_clusters = pd.read_csv(output_dir / "correlation" / "primary_clusters.csv")
        pruned_baseline, conditional_conditions = _conditional_conditions(prepared.feature_names, primary_clusters)
        conditional_plan = [pruned_baseline] + conditional_conditions
        if time.time() < model_deadline and bool(config["tests"].get("conditional_all_features", True)):
            conditional_blocks = _build_blocks(
                prepared, output_dir, folds, best_iterations, conditional_plan, config,
                stage="conditional_all_features", deadline_epoch=model_deadline,
                shard_size=int(config["cpu"].get("models_per_block", 24)), run_signature=run_signature,
            )
            phase_summaries.append(_execute_blocks(conditional_blocks, workers, "conditional_all_features"))

        if time.time() < model_deadline and bool(config["tests"].get("cluster_ablation", True)):
            cluster_conditions = _cluster_conditions(prepared.feature_names, primary_clusters)
            cluster_blocks = _build_blocks(
                prepared, output_dir, folds, best_iterations, cluster_conditions, config,
                stage="cluster_loo", deadline_epoch=model_deadline,
                shard_size=int(config["cpu"].get("cluster_models_per_block", 16)), run_signature=run_signature,
            )
            phase_summaries.append(_execute_blocks(cluster_blocks, workers, "cluster_loo"))

        if gpu_process is not None:
            remaining = max(0.0, model_deadline - time.time())
            gpu_process.join(timeout=remaining)
            if gpu_process.is_alive():
                LOGGER.warning("GPU audit exceeded model deadline; terminating optional process")
                gpu_process.terminate()
                gpu_process.join(timeout=20)
                if gpu_process.is_alive():
                    gpu_process.kill()
                    gpu_process.join(timeout=10)

        aggregation = aggregate_results(output_dir, len(prepared.feature_names), len(folds), prepared.signature, run_signature)
        final = {
            "status": aggregation["status"],
            "started_epoch": started_epoch,
            "elapsed_seconds": time.time() - started_epoch,
            "hard_runtime_minutes": config["runtime"]["hard_runtime_minutes"],
            "hard_deadline_respected": time.time() <= hard_deadline,
            "dataset_path": str(dataset),
            "dataset_signature": prepared.signature,
            "run_signature": run_signature,
            "output_dir": str(output_dir),
            "cache_root": str(prepared.root),
            "feature_count": len(prepared.feature_names),
            "fold_count": len(folds),
            "phase_summaries": phase_summaries,
            "aggregation": aggregation,
        }
        atomic_json(final, output_dir / "run_summary.json")
        return final
    finally:
        if gpu_process is not None and gpu_process.is_alive():
            gpu_process.terminate()
            gpu_process.join(timeout=10)
        monitor.stop()
