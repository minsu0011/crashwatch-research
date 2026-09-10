"""Runnable V3 dual-ablation entry point.

It keeps raw/features/results under this V3 directory and reads the existing
parent development panel only as the supervised-learning base.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dual_ablation.crawlers.orchestrator import run_crawlers
from dual_ablation.experiment.runner import run_ablation
from dual_ablation.features.pipeline import run_feature_pipeline


def _csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch V3 dual-ablation runner")
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2026-03-24")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--overwrite-crawl", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--with-market-panel", action="store_true")
    parser.add_argument("--modes", default="universe,ticker_global,bucket")
    parser.add_argument("--groups")
    parser.add_argument("--buckets")
    parser.add_argument("--tickers")
    parser.add_argument("--seeds", default="17,43,101")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--min-train-days", type=int, default=500)
    parser.add_argument("--max-train-rows", type=int, default=300000)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--calibration", choices=("none", "sigmoid", "isotonic"), default="sigmoid")
    parser.add_argument("--allow-partial-data", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    project = Path(__file__).resolve().parent
    result: dict[str, object] = {"project": str(project)}
    if not args.skip_crawl:
        result["crawl"] = run_crawlers(project, args.start, args.end, overwrite=args.overwrite_crawl,
                                        with_market_panel=args.with_market_panel)
    if not args.skip_features:
        result["features"] = run_feature_pipeline(project, strict_quality_check=True,
                                                   allow_partial_data=args.allow_partial_data)
    result["ablation"] = run_ablation(
        project, modes=_csv_set(args.modes) or set(), groups=_csv_set(args.groups),
        buckets=_csv_set(args.buckets), tickers=_csv_set(args.tickers),
        seeds=[int(x) for x in args.seeds.split(",") if x.strip()], n_folds=args.folds,
        validation_days=args.validation_days, purge_days=args.purge_days,
        min_train_days=args.min_train_days, max_train_rows=args.max_train_rows,
        overwrite_cache=args.overwrite_cache, prefer_gpu=not args.cpu,
        calibration=args.calibration, allow_partial_data=args.allow_partial_data,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
