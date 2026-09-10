from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import psutil

from dual_ablation.config import get_paths
from dual_ablation.finance11h.crawler import run_finance_crawl, verify_mandatory_short_data
from dual_ablation.finance11h.features import build_finance_features
from dual_ablation.io_utils import atomic_json


def _set_runtime(threads: int) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = str(threads)
    if os.name == "nt":
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance11H 전체 11시간 파이프라인")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--training-path", type=Path)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="auto")
    parser.add_argument("--hours", type=float, default=11.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seeds", default="17,43")
    parser.add_argument("--cache-namespace", default="finance11h_v2")
    parser.add_argument("--overwrite-crawl", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error("--threads는 1~16 범위여야 합니다.")

    project = args.project.resolve()
    _set_runtime(args.threads)
    paths = get_paths(project)
    result_dir = paths.data_root / "ablation_finance11h"
    result_dir.mkdir(parents=True, exist_ok=True)
    log_path = result_dir / "finance11h.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )
    started = time.monotonic()
    deadline = started + args.hours * 3600
    state_path = result_dir / "overall_run_state.json"
    end = (
        pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None).strftime("%Y-%m-%d")
        if args.end == "auto" else args.end
    )
    state: dict = {
        "status": "running", "started_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "budget_hours": args.hours, "stage": "startup",
    }
    atomic_json(state, state_path)
    try:
        state["stage"] = "crawl"
        atomic_json(state, state_path)
        if args.skip_crawl:
            crawl = {"skipped": True, "mandatory_short": verify_mandatory_short_data(paths)}
        else:
            crawl = run_finance_crawl(project, args.start, end, overwrite=args.overwrite_crawl, strict=True)

        state["stage"] = "features"
        atomic_json(state, state_path)
        if args.skip_features:
            dataset = paths.data_root / "development" / "training_dataset_finance11h.parquet"
            if not dataset.exists():
                raise FileNotFoundError(f"--skip-features 대상 데이터셋이 없습니다: {dataset}")
            feature_summary = {"skipped": True, "output_dataset": str(dataset)}
        else:
            feature_summary = build_finance_features(project, args.training_path, strict_short=True)
            dataset = Path(feature_summary["output_dataset"])

        remaining_seconds = deadline - time.monotonic()
        if remaining_seconds <= 12 * 60:
            raise RuntimeError(f"결과 집계 12분을 제외하면 모델 실행 시간이 없습니다: remaining={remaining_seconds:.1f}s")
        state["stage"] = "model"
        state["remaining_hours_at_model_start"] = remaining_seconds / 3600
        atomic_json(state, state_path)
        from dual_ablation.finance11h.runner import Finance11HRunner

        runner = Finance11HRunner(
            project, dataset, hours=remaining_seconds / 3600,
            threads=args.threads, folds=args.folds,
            seeds=[int(value.strip()) for value in args.seeds.split(",") if value.strip()],
            validation_days=60, purge_days=20, prefer_gpu=True,
            overwrite_cache=args.overwrite_cache, cache_namespace=args.cache_namespace,
        )
        model_summary = runner.run()
        state.update({
            "status": "completed", "stage": "completed",
            "finished_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "elapsed_hours": (time.monotonic() - started) / 3600,
            "crawl": crawl, "features": feature_summary, "model": model_summary,
        })
        atomic_json(state, state_path)
        print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
    except Exception as exc:
        state.update({
            "status": "failed",
            "finished_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "elapsed_hours": (time.monotonic() - started) / 3600,
            "error": f"{type(exc).__name__}: {exc}",
        })
        atomic_json(state, state_path)
        raise


if __name__ == "__main__":
    main()
