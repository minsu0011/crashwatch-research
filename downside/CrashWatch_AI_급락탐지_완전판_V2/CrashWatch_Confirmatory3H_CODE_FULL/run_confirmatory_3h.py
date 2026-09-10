from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

from cw3h.aggregate import aggregate_results
from cw3h.cache import discover_dataset, prepare_cache
from cw3h.config import load_config, resolve_paths
from cw3h.experiment import run_experiment
from cw3h.references import load_references
from cw3h.utils import atomic_json, setup_logging


def apply_runtime_profile(config: dict) -> dict:
    """Apply hard process limits before any worker is spawned."""
    threads = int(config["threads_per_worker"])
    for name, value in {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }.items():
        os.environ[name] = value

    info = {
        "runtime_profile": str(config.get("runtime_profile", "full")),
        "workers": int(config["workers"]),
        "threads_per_worker": threads,
        "gpu_audit_enabled": bool(config.get("enable_gpu_audit", True)),
    }
    if not config.get("enable_gpu_audit", True):
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        info["cuda_visible_devices"] = "-1"

    if info["runtime_profile"] == "game_cpu4":
        if int(config["workers"]) != 1 or threads != 4 or bool(config.get("enable_gpu_audit", True)):
            raise ValueError("game_cpu4 requires workers=1, threads_per_worker=4, enable_gpu_audit=false")
        import psutil

        process = psutil.Process()
        available = list(process.cpu_affinity())
        selected = available[-4:] if len(available) >= 4 else available
        process.cpu_affinity(selected)
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        info.update({
            "cpu_affinity": selected,
            "logical_cpu_limit": len(selected),
            "priority": "below_normal",
            "gpu_compute": "sealed",
        })
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrashWatch 3-hour confirmatory paired-ablation runner")
    parser.add_argument("command", choices=["prepare", "run", "aggregate", "all"], nargs="?", default="all")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--project-root", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--no-gpu-audit", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    config, config_path = load_config(args.config)
    if args.no_gpu_audit:
        config["enable_gpu_audit"] = False
    runtime_limits = apply_runtime_profile(config)
    paths = resolve_paths(config, config_path, args.project_root)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(paths.output_dir / "confirmatory_3h.log", args.verbose)
    refs = load_references(paths.reference_dir, bool(config.get("strict_reference_counts", True)))
    dataset_path = discover_dataset(config, paths, args.dataset)
    atomic_json({
        "config": config,
        "config_path": str(config_path),
        "project_root": str(paths.project_root),
        "dataset_path": str(dataset_path),
        "reference_hashes": refs.hashes,
        "runtime_limits": runtime_limits,
    }, paths.output_dir / "resolved_run_config.json")

    prepared = prepare_cache(dataset_path, paths, config, refs, force=args.force_cache)
    atomic_json(prepared.manifest, paths.output_dir / "dataset_and_feature_manifest.json")

    if args.command == "prepare":
        print(f"Prepared cache: {prepared.root}")
        return 0
    if args.command == "aggregate":
        summary = aggregate_results(prepared.root, paths.output_dir, config, {
            "dataset_path": str(dataset_path),
            "dataset_signature": prepared.dataset_signature,
            "aggregate_only": True,
        })
        print(summary)
        return 0
    summary = run_experiment(prepared, paths, config, refs, started_epoch=started)
    print(summary)
    return 0 if summary.get("status") in {"completed", "partial"} else 2


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
