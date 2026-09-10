from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import psutil


def _set_runtime(threads: int) -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[name] = str(threads)
    if os.name == "nt":
        psutil.Process().nice(psutil.HIGH_PRIORITY_CLASS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance11H 금융 피처 이탈테스트")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--hours", type=float, default=11.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seeds", default="17,43")
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--cache-namespace", default="finance11h_v2")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.purge_days < 20:
        parser.error("--purge-days는 label horizon 때문에 최소 20이어야 합니다.")
    if not 1 <= args.threads <= 16:
        parser.error("--threads는 1~16 범위여야 합니다.")
    _set_runtime(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    from dual_ablation.finance11h.runner import Finance11HRunner

    runner = Finance11HRunner(
        args.project, args.dataset,
        hours=args.hours, threads=args.threads, folds=args.folds,
        seeds=[int(value.strip()) for value in args.seeds.split(",") if value.strip()],
        validation_days=args.validation_days, purge_days=args.purge_days,
        prefer_gpu=not args.cpu, overwrite_cache=args.overwrite_cache,
        cache_namespace=args.cache_namespace,
    )
    print(json.dumps(runner.run(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
