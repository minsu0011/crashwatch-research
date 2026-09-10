#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_ablation.experiment.runner import run_ablation


def csv_set(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="시장 공통 피처와 종목 피처의 독립 이탈테스트")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--target", default="label_abs_crash_20")
    parser.add_argument("--modes", default="universe,ticker_global,bucket", help="universe,ticker_global,bucket,ticker")
    parser.add_argument("--groups", default=None)
    parser.add_argument("--buckets", default=None)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--seeds", default="17,43,101")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--min-train-days", type=int, default=500)
    parser.add_argument("--max-train-rows", type=int, default=300000)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_ablation(
        Path(__file__).resolve().parent, args.dataset,
        target=args.target, modes=csv_set(args.modes) or set(), groups=csv_set(args.groups),
        buckets=csv_set(args.buckets), tickers=csv_set(args.tickers),
        seeds=[int(x) for x in args.seeds.split(",") if x.strip()],
        n_folds=args.folds, validation_days=args.validation_days, purge_days=args.purge_days,
        min_train_days=args.min_train_days, max_train_rows=args.max_train_rows,
        overwrite_cache=args.overwrite_cache, prefer_gpu=not args.cpu,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
