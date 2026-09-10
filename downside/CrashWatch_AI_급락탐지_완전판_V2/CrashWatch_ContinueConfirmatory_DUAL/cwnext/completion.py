from __future__ import annotations

import concurrent.futures as cf
import logging
import multiprocessing as mp
import time
from pathlib import Path
from typing import Any

import pandas as pd

from cw7h.aggregate import aggregate_results
from cw7h.lgb_worker import LgbBlock, run_lgb_block
from cw7h.runner import _build_blocks, _cluster_conditions, _conditional_conditions
from cw7h.utils import atomic_json

from .common import write_eta

LOGGER = logging.getLogger(__name__)


def _execute_blocks(blocks: list[LgbBlock], workers: int, output_dir: Path, stage: str) -> dict[str, Any]:
    started = time.time()
    summaries: list[dict[str, Any]] = []
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=max(1, workers), mp_context=ctx) as executor:
        futures = [executor.submit(run_lgb_block, block) for block in blocks]
        for i, future in enumerate(cf.as_completed(futures), start=1):
            result = future.result()
            summaries.append(result)
            trained = sum(int(x.get("trained", 0)) for x in summaries)
            cached = sum(int(x.get("cached", 0)) for x in summaries)
            failed = sum(int(x.get("failed", 0)) for x in summaries)
            planned = trained + cached + failed + sum(int(x.get("deadline_skipped", 0)) for x in summaries)
            write_eta(output_dir, stage, i, len(blocks), time.time() - started)
            LOGGER.info(
                "%s blocks=%s/%s trained=%s cached=%s failed=%s",
                stage, i, len(blocks), trained, cached, failed,
            )
    result = {
        "stage": stage,
        "blocks": len(blocks),
        "trained": sum(int(x.get("trained", 0)) for x in summaries),
        "cached": sum(int(x.get("cached", 0)) for x in summaries),
        "failed": sum(int(x.get("failed", 0)) for x in summaries),
        "deadline_skipped": sum(int(x.get("deadline_skipped", 0)) for x in summaries),
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(result, output_dir / f"continuation_{stage}_summary.json")
    return result


def run_cpu_completion(
    prepared: Any,
    folds: list[Any],
    previous_output: Path,
    legacy_config: dict[str, Any],
    legacy_dataset_signature: str,
    run_signature: str,
    best_iterations: dict[int, int],
    *,
    workers: int,
    threads_per_worker: int,
    block_size: int,
) -> dict[str, Any]:
    config = dict(legacy_config)
    config["cpu"] = dict(legacy_config["cpu"])
    config["cpu"]["workers"] = int(workers)
    config["cpu"]["threads_per_worker"] = int(threads_per_worker)
    config["cpu"]["models_per_block"] = int(block_size)
    config["cpu"]["cluster_models_per_block"] = int(block_size)
    deadline = time.time() + 3650 * 24 * 3600
    primary_clusters = pd.read_csv(previous_output / "correlation" / "primary_clusters.csv")
    pruned_baseline, conditional = _conditional_conditions(prepared.feature_names, primary_clusters)
    conditional_blocks = _build_blocks(
        prepared,
        previous_output,
        folds,
        best_iterations,
        [pruned_baseline] + conditional,
        config,
        stage="conditional_all_features",
        deadline_epoch=deadline,
        shard_size=int(block_size),
        run_signature=run_signature,
    )
    conditional_summary = _execute_blocks(conditional_blocks, workers, previous_output, "conditional_resume")

    cluster_conditions = _cluster_conditions(prepared.feature_names, primary_clusters)
    cluster_blocks = _build_blocks(
        prepared,
        previous_output,
        folds,
        best_iterations,
        cluster_conditions,
        config,
        stage="cluster_loo",
        deadline_epoch=deadline,
        shard_size=int(block_size),
        run_signature=run_signature,
    )
    cluster_summary = _execute_blocks(cluster_blocks, workers, previous_output, "cluster_loo_resume")
    aggregation = aggregate_results(
        previous_output,
        len(prepared.feature_names),
        len(folds),
        legacy_dataset_signature,
        run_signature,
    )
    final = {
        "conditional": conditional_summary,
        "cluster": cluster_summary,
        "aggregation": aggregation,
    }
    atomic_json(final, previous_output / "cpu_completion_final.json")
    return final
