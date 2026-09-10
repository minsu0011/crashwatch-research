"""Run V2/V3 comparison and the four V3 dual-ablation modes."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dual_ablation.experiment.runner import run_ablation


def _csv_set(value: str | None) -> set[str] | None:
    return {part.strip() for part in value.split(",") if part.strip()} if value else None


def _seeds(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) == 1 and parts[0].isdigit() and int(parts[0]) <= 10:
        return [17, 43, 101][: int(parts[0])]
    return [int(part) for part in parts]


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch V3 dual-ablation runner")
    parser.add_argument("--dataset-path", type=Path)
    parser.add_argument("--modes", default="universe,ticker_global,bucket,ticker")
    parser.add_argument("--groups")
    parser.add_argument("--buckets")
    parser.add_argument("--tickers")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--min-train-days", type=int, default=500)
    parser.add_argument("--seeds", default="3", help="seed count (1-3) or comma-separated seed values")
    parser.add_argument("--calibration", choices=("none", "sigmoid", "isotonic"), default="sigmoid")
    parser.add_argument("--max-train-rows", type=int, default=300000)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--allow-partial-data", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.purge_days < 20:
        parser.error("--purge-days must be at least 20 for label_abs_crash_20")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_ablation(
        Path(__file__).resolve().parent, dataset_path=args.dataset_path,
        modes=_csv_set(args.modes) or set(), groups=_csv_set(args.groups),
        buckets=_csv_set(args.buckets), tickers=_csv_set(args.tickers),
        seeds=_seeds(args.seeds), n_folds=args.folds,
        validation_days=args.validation_days, purge_days=args.purge_days,
        min_train_days=args.min_train_days, max_train_rows=args.max_train_rows,
        overwrite_cache=args.overwrite_cache, prefer_gpu=not args.cpu,
        calibration=args.calibration, allow_partial_data=args.allow_partial_data,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
