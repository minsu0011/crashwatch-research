#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _set_threads(threads: int, *, gpu_enabled: bool) -> None:
    value = str(threads)
    for key in [
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS",
    ]:
        os.environ[key] = value
    if gpu_enabled:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    else:
        # CPU 게임 병행 모드에서는 CUDA 런타임에도 GPU를 노출하지 않는다.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent / ".numba_cache"))


def main() -> None:
    parser = argparse.ArgumentParser(description="11시간 금융 피처 집중 이탈테스트")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--hours", type=float, default=11.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seeds", default="17,43")
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--cache-namespace", default="finance11h_v3")
    args = parser.parse_args()
    _set_threads(args.threads, gpu_enabled=not args.cpu)

    import logging
    import psutil
    from dual_ablation.finance11h.runner import Finance11HRunner

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        process = psutil.Process()
        if os.name == "nt":
            if args.cpu and args.threads <= 4:
                process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
                logical = psutil.cpu_count(logical=True) or args.threads
                process.cpu_affinity(
                    list(range(max(0, logical - args.threads), logical))
                )
            else:
                process.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            process.nice(-5)
    except Exception:
        pass

    runner = Finance11HRunner(
        Path(__file__).resolve().parent,
        args.dataset,
        hours=args.hours,
        threads=args.threads,
        folds=args.folds,
        seeds=[int(x) for x in args.seeds.split(",") if x.strip()],
        validation_days=args.validation_days,
        purge_days=args.purge_days,
        prefer_gpu=not args.cpu,
        overwrite_cache=args.overwrite_cache,
        cache_namespace=args.cache_namespace,
    )
    result = runner.run()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
