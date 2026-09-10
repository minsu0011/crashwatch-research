#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path


def parse_set(value: str | None) -> set[str] | None:
    return {x.strip() for x in value.split(",") if x.strip()} if value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="장기 이탈테스트 단일 작업 워커")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--modes", required=True)
    parser.add_argument("--groups", default=None)
    parser.add_argument("--buckets", default=None)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--seeds", default="17,43,101,151,197")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--min-train-days", type=int, default=500)
    parser.add_argument("--max-train-rows", type=int, default=300000)
    parser.add_argument("--calibration", choices=["none", "sigmoid"], default="sigmoid")
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["CRASHWATCH_RUN_TAG"] = args.run_tag
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    from dual_ablation.experiment.runner import run_ablation

    result = run_ablation(
        Path(__file__).resolve().parent,
        args.dataset,
        target="label_abs_crash_20",
        modes=parse_set(args.modes) or set(),
        groups=parse_set(args.groups),
        buckets=parse_set(args.buckets),
        tickers=parse_set(args.tickers),
        seeds=[int(x) for x in args.seeds.split(",") if x.strip()],
        n_folds=args.folds,
        validation_days=args.validation_days,
        purge_days=args.purge_days,
        min_train_days=args.min_train_days,
        max_train_rows=args.max_train_rows,
        overwrite_cache=args.overwrite_cache,
        prefer_gpu=not args.cpu,
        calibration=args.calibration,
        calibration_days=args.calibration_days,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
