from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parents[1]
STARTER_DIR = PROJECT_DIR / "starter_code"
if str(STARTER_DIR) not in sys.path:
    sys.path.insert(0, str(STARTER_DIR))

from run_surge_organic_ablation_v8 import (  # noqa: E402
    _init_worker,
    _run_worker_task,
    build_task_payload,
)
from surge_ablation_common import (  # noqa: E402
    FileLock,
    TaskSpec,
    atomic_write_json,
    load_json,
    result_file_is_valid,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def optional_text(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def optional_int(value: Any) -> int | None:
    text = optional_text(value)
    return int(float(text)) if text is not None else None


def load_payloads(output_dir: Path) -> list[dict[str, Any]]:
    config = load_json(output_dir / "resolved_run_config.json")
    cache = load_json(output_dir / "cache" / "matrix_cache_manifest.json")
    features = [str(item) for item in cache["features"]]
    feature_to_index = {feature: index for index, feature in enumerate(features)}
    all_indices = tuple(range(len(features)))

    with (output_dir / "organic_condition_plan.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        condition_rows = {row["condition_id"]: row for row in csv.DictReader(handle)}
    with (output_dir / "task_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        manifest_rows = [row for row in csv.DictReader(handle) if row["backend"] == "lightgbm_cpu"]

    payloads: list[dict[str, Any]] = []
    for row in manifest_rows:
        if row["test_type"] == "baseline":
            dropped: tuple[str, ...] = ()
            enabled = all_indices
            representative = None
            cluster_id = None
            feature_group = None
            profile_name = row["condition_id"].split("::", 1)[1]
        else:
            condition = condition_rows.get(row["condition_id"])
            if condition is None:
                raise RuntimeError(f"Condition missing from plan: {row['condition_id']}")
            dropped = tuple(item for item in condition["dropped_features"].split("|") if item)
            missing = [item for item in dropped if item not in feature_to_index]
            if missing:
                raise RuntimeError(f"Unknown dropped features for {row['condition_id']}: {missing}")
            dropped_set = set(dropped)
            enabled = tuple(index for index, feature in enumerate(features) if feature not in dropped_set)
            representative = optional_text(condition.get("representative_feature"))
            cluster_id = optional_int(condition.get("cluster_id"))
            feature_group = optional_text(condition.get("feature_group"))
            profile_name = None

        spec = TaskSpec(
            backend="lightgbm_cpu",
            stage=row["stage"],
            test_type=row["test_type"],
            condition_id=row["condition_id"],
            fold_id=int(row["fold_id"]),
            seed=int(row["seed"]),
            enabled_feature_indices=enabled,
            dropped_features=dropped,
            representative_feature=representative,
            cluster_id=cluster_id,
            feature_group=feature_group,
            profile_name=profile_name,
        )
        payload = build_task_payload(
            spec,
            features,
            int(config["best_iterations"]["lightgbm_cpu"][str(spec.fold_id)]),
            config["fold_roles"],
            str(config["task_run_signature"]),
            output_dir / "task_results",
            save_prediction=spec.test_type == "baseline",
        )
        if payload["identity_hash"] != row["identity_hash"]:
            raise RuntimeError(
                f"Task identity mismatch for {row['condition_id']} fold={row['fold_id']}: "
                f"generated={payload['identity_hash']} expected={row['identity_hash']}"
            )
        payloads.append(payload)
    return payloads


def write_status(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = utc_now()
    atomic_write_json(path, payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume only the original V8 LightGBM tasks in PUBG/game mode.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "surge_organic_ablation_v8",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    if args.workers < 1 or args.threads_per_worker < 1:
        raise ValueError("workers and threads-per-worker must be positive")

    output_dir = args.output.resolve()
    runtime_dir = output_dir / "runtime_logs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    status_path = runtime_dir / "GAME_MODE_STATUS.json"
    lock_path = runtime_dir / ".surge_organic_game_mode.lock"

    with FileLock(lock_path):
        payloads = load_payloads(output_dir)
        pending = [
            payload
            for payload in payloads
            if not result_file_is_valid(Path(payload["result_path"]), str(payload["identity_hash"]))
        ]
        cached = len(payloads) - len(pending)

        context = load_json(output_dir / "worker_context.json")
        context["threads_per_worker"] = int(args.threads_per_worker)
        context["xgboost_threads"] = 1
        context_path = runtime_dir / "game_worker_context.json"
        atomic_write_json(context_path, context)

        status: dict[str, Any] = {
            "schema": "crashwatch_surge_v8_game_mode_v1",
            "status": "RUNNING",
            "started_at": utc_now(),
            "pid": os.getpid(),
            "backend": "lightgbm_cpu",
            "gpu_sealed": True,
            "workers": int(args.workers),
            "threads_per_worker": int(args.threads_per_worker),
            "logical_thread_budget": int(args.workers * args.threads_per_worker),
            "affinity_expected": "logical CPUs 16-31 (mask 0xFFFF0000)",
            "task_run_signature": load_json(output_dir / "resolved_run_config.json")["task_run_signature"],
            "lightgbm_tasks_total": len(payloads),
            "cached_at_start": cached,
            "pending_at_start": len(pending),
            "session_completed": 0,
            "session_failed": 0,
        }
        write_status(status_path, status)
        print(
            f"Game mode start: total={len(payloads)} cached={cached} pending={len(pending)} "
            f"workers={args.workers} threads={args.threads_per_worker}",
            flush=True,
        )
        if not pending:
            status["status"] = "COMPLETE_LIGHTGBM"
            write_status(status_path, status)
            return 0

        completed = 0
        failed = 0
        started = time.monotonic()
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(str(context_path),),
        )
        try:
            futures = {executor.submit(_run_worker_task, payload): payload for payload in pending}
            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    if result.get("status") in {"completed", "cached"}:
                        completed += 1
                    else:
                        failed += 1
                        print(f"Task failed: {result}", flush=True)
                except Exception as exc:
                    failed += 1
                    print(
                        f"Task exception: {task['identity_hash']} {task['condition_id']} fold={task['fold_id']}: {exc!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                done = completed + failed
                if done == 1 or done % args.progress_every == 0 or done == len(pending):
                    elapsed = time.monotonic() - started
                    rate = done / elapsed if elapsed > 0 else 0.0
                    remaining = (len(pending) - done) / rate if rate > 0 else None
                    status.update(
                        {
                            "session_completed": completed,
                            "session_failed": failed,
                            "session_processed": done,
                            "rate_tasks_per_second": rate,
                            "eta_seconds_for_lightgbm_only": remaining,
                        }
                    )
                    write_status(status_path, status)
                    print(
                        f"Game mode progress {done}/{len(pending)} complete={completed} failed={failed} "
                        f"cached={cached} rate={rate:.3f}/s",
                        flush=True,
                    )
        except KeyboardInterrupt:
            status["status"] = "PAUSED"
            status["stop_reason"] = "KeyboardInterrupt"
            write_status(status_path, status)
            return 130
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

        status["status"] = "COMPLETE_LIGHTGBM" if failed == 0 else "FAILED"
        status["finished_at"] = utc_now()
        write_status(status_path, status)
        return 0 if failed == 0 else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
