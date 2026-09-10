from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STARTER = ROOT / "starter_code"
sys.path.insert(0, str(STARTER))

from run_surge_organic_ablation_v8 import _init_worker, _run_worker_task  # noqa: E402
from surge_ablation_common import atomic_write_json, load_json  # noqa: E402


def run_case(gpu_workers: int, tasks: int, iterations: int, xgboost_threads: int) -> dict[str, float | int]:
    output = ROOT / "outputs" / "surge_organic_ablation_v8"
    cache = load_json(output / "cache" / "matrix_cache_manifest.json")
    features = list(cache["features"])
    feature_to_index = {feature: index for index, feature in enumerate(features)}
    plan = pd.read_csv(output / "organic_condition_plan.csv")
    candidates = plan[plan["test_type"].ne("baseline")].head(tasks).reset_index(drop=True)
    if len(candidates) < tasks:
        raise RuntimeError(f"benchmark plan too small: {len(candidates)} < {tasks}")

    with tempfile.TemporaryDirectory(prefix=f"v8_gpu_bench_{gpu_workers}_") as temp_text:
        temp = Path(temp_text)
        context = {
            "matrix_path": cache["artifacts"]["matrix"]["path"],
            "arrays_path": cache["artifacts"]["arrays"]["path"],
            "fold_indices_path": cache["artifacts"]["fold_indices"]["path"],
            "prediction_dir": str(temp / "predictions"),
            "threads_per_worker": 1,
            "xgboost_threads": int(xgboost_threads),
            "lightgbm_params": {},
            "xgboost_params": {},
            "top_fractions": [0.01, 0.03, 0.05, 0.10],
            "scope_codebooks": cache.get("scope_codebooks", {}),
            "minimum_scope_rows": 20,
            "precision_target": 0.70,
            "precision_min_alerts": 30,
        }
        context_path = temp / "worker_context.json"
        atomic_write_json(context_path, context)
        payloads: list[dict[str, object]] = []
        for index, row in candidates.iterrows():
            dropped = [token for token in str(row.get("dropped_features", "")).split("|") if token and token != "nan"]
            dropped_set = set(dropped)
            enabled = [position for feature, position in feature_to_index.items() if feature not in dropped_set]
            identity_hash = hashlib.sha256(f"{gpu_workers}:{index}:{iterations}:{xgboost_threads}".encode()).hexdigest()
            payloads.append(
                {
                    "status": "planned",
                    "identity_hash": identity_hash,
                    "result_path": str(temp / "results" / f"{identity_hash}.json"),
                    "run_signature": "gpu_benchmark",
                    "backend": "xgboost_gpu",
                    "stage": str(row["stage"]),
                    "test_type": str(row["test_type"]),
                    "condition_id": str(row["condition_id"]),
                    "fold_id": int(index % 8),
                    "fold_role": "benchmark",
                    "seed": 17,
                    "best_iteration": int(iterations),
                    "enabled_feature_indices": enabled,
                    "dropped_features": dropped,
                    "representative_feature": None,
                    "cluster_id": None,
                    "feature_group": None,
                    "profile_name": None,
                    "feature_count_total": len(features),
                    "save_prediction": False,
                }
            )

        started = time.monotonic()
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=gpu_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(context_path),),
        ) as executor:
            results = list(executor.map(_run_worker_task, payloads))
        elapsed = time.monotonic() - started
        completed = sum(result.get("status") == "completed" for result in results)
        failed = len(results) - completed
        return {
            "gpu_workers": gpu_workers,
            "tasks": len(payloads),
            "iterations": iterations,
            "xgboost_threads": xgboost_threads,
            "completed": completed,
            "failed": failed,
            "wall_seconds": elapsed,
            "tasks_per_second": len(payloads) / elapsed,
            "seconds_per_task_effective": elapsed / len(payloads),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-counts", default="1,2,3")
    parser.add_argument("--tasks", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--xgboost-threads", type=int, default=1)
    args = parser.parse_args()
    results = [
        run_case(int(token), args.tasks, args.iterations, args.xgboost_threads)
        for token in args.worker_counts.split(",")
        if token.strip()
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
