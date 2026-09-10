#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def _set_runtime(profile: str, threads: int) -> None:
    value = str(threads)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[key] = value
    os.environ.setdefault(
        "NUMBA_CACHE_DIR",
        str(Path(__file__).resolve().parent / ".numba_cache"),
    )
    if profile == "game":
        # Import하기 전에 CUDA를 숨겨 XGBoost/Numba가 게임용 GPU를 건드리지 못하게 한다.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def _parse_seeds(raw: str) -> list[int]:
    seeds = [int(value.strip()) for value in raw.split(",") if value.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("seed를 하나 이상 지정해야 합니다.")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finance11H 집중 Nested 이탈테스트 (게임/풀로드 캐시 공유)"
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--profile", choices=("game", "full"), default="game")
    parser.add_argument("--hours", type=float, default=0.0)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seeds", default="17,43,101")
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--inner-validation-days", type=int, default=40)
    parser.add_argument("--calibration-days", type=int, default=60)
    parser.add_argument("--calibration-purge-days", type=int, default=20)
    parser.add_argument(
        "--cache-namespace",
        default="finance_nested_focus_v1",
    )
    parser.add_argument("--clear-stop-on-start", action="store_true")
    args = parser.parse_args()

    threads = args.threads if args.threads is not None else (4 if args.profile == "game" else 16)
    if not 1 <= threads <= 16:
        parser.error("--threads는 1~16 범위여야 합니다.")
    if args.profile == "game" and threads > 4:
        parser.error("game profile은 CPU 4스레드를 초과할 수 없습니다.")
    if args.purge_days < 20 or args.calibration_purge_days < 20:
        parser.error("label_abs_crash_20 때문에 purge는 모두 최소 20거래일이어야 합니다.")
    if args.folds != 8:
        parser.error("집중 검증 계약상 outer fold는 8이어야 합니다.")

    seeds = _parse_seeds(args.seeds)
    _set_runtime(args.profile, threads)

    import logging

    from dual_ablation.finance_nested.control import apply_process_profile
    from dual_ablation.finance_nested.runner import FocusedNestedRunner
    from dual_ablation.io_utils import atomic_json

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    runtime = apply_process_profile(args.profile, threads)
    runtime.update(
        {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "prefer_gpu": args.profile == "full",
            "seeds": seeds,
            "outer_folds": args.folds,
            "inner_folds": args.inner_folds,
            "outer_purge_days": args.purge_days,
            "calibration_purge_days": args.calibration_purge_days,
        }
    )
    runtime_path = (
        args.project.resolve()
        / "crashwatch_ai_data"
        / "ablation_finance_nested_focus"
        / "runtime_profile_latest.json"
    )
    atomic_json(runtime, runtime_path)

    runner = FocusedNestedRunner(
        args.project,
        args.dataset,
        profile=args.profile,
        threads=threads,
        prefer_gpu=args.profile == "full",
        hours=args.hours,
        cache_namespace=args.cache_namespace,
        outer_folds=args.folds,
        seeds=seeds,
        validation_days=args.validation_days,
        purge_days=args.purge_days,
        inner_folds=args.inner_folds,
        inner_validation_days=args.inner_validation_days,
        calibration_days=args.calibration_days,
        calibration_purge_days=args.calibration_purge_days,
        clear_stop_on_start=args.clear_stop_on_start,
    )
    print(json.dumps(runner.run(), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
