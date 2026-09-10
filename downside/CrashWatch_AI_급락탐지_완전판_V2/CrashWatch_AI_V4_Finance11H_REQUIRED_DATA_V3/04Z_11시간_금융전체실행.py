#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd


def _set_threads(threads: int) -> None:
    for key in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"]:
        os.environ[key] = str(threads)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(__file__).resolve().parent / ".numba_cache"))


def main() -> None:
    parser = argparse.ArgumentParser(description="필수 실데이터 수집부터 이탈테스트까지 총 11시간 예산으로 실행")
    parser.add_argument("--hours", type=float, default=11.0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="auto")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--seeds", default="17,43")
    parser.add_argument("--sources", default="macro,krx,lending,dart,naver")
    parser.add_argument("--skip-data-download", action="store_true")
    parser.add_argument("--no-dart-documents", action="store_true")
    parser.add_argument("--overwrite-data", action="store_true")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--krx-chunk-months", type=int, default=24)
    parser.add_argument("--lending-chunk-months", type=int, default=12)
    parser.add_argument("--cache-namespace", default="finance11h_v3")
    args = parser.parse_args()
    if args.hours <= 0.5:
        raise ValueError("hours는 0.5보다 커야 합니다.")
    _set_threads(args.threads)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from dual_ablation.config import get_paths
    from dual_ablation.data_acquisition.common import load_local_env
    from dual_ablation.data_acquisition.runner import run_required_data_download
    from dual_ablation.data_acquisition.validation import validate_required_data
    from dual_ablation.finance11h.features import build_finance_features
    from dual_ablation.finance11h.runner import Finance11HRunner

    project = Path(__file__).resolve().parent
    load_local_env(project)
    paths = get_paths(project)
    started = time.monotonic()
    end = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d") if args.end == "auto" else args.end
    sources = [x.strip().lower() for x in args.sources.split(",") if x.strip()]

    if args.skip_data_download:
        acquisition = {"status": "skipped_by_user", "validation": validate_required_data(paths, strict=False)}
    else:
        acquisition = run_required_data_download(
            project,
            args.start,
            end,
            sources=sources,
            overwrite=args.overwrite_data,
            strict=False,
            dart_documents=not args.no_dart_documents,
            krx_chunk_months=args.krx_chunk_months,
            lending_chunk_months=args.lending_chunk_months,
        )

    # All independent sources have now had a chance to run. Do not start the
    # expensive model experiment unless actual short and auxiliary coverage pass.
    validation = validate_required_data(paths, strict=True)
    features = build_finance_features(project, strict_short=True)

    elapsed = time.monotonic() - started
    remaining_hours = args.hours - elapsed / 3600 - 0.20
    if remaining_hours <= 0.35:
        raise RuntimeError(
            "필수 데이터 수집·검증·피처 생성에 11시간 예산이 소진됐습니다. "
            "수집 캐시는 저장됐으므로 같은 명령을 다시 실행하면 학습 단계부터 이어집니다."
        )
    runner = Finance11HRunner(
        project,
        paths.data_root / "development" / "training_dataset_finance11h.parquet",
        hours=remaining_hours,
        threads=args.threads,
        folds=args.folds,
        seeds=[int(x) for x in args.seeds.split(",") if x.strip()],
        prefer_gpu=True,
        overwrite_cache=args.overwrite_cache,
        cache_namespace=args.cache_namespace,
    )
    ablation = runner.run()
    print(json.dumps({
        "acquisition": acquisition,
        "validation": validation,
        "features": features,
        "ablation": ablation,
        "cache_namespace": args.cache_namespace,
        "total_elapsed_hours": (time.monotonic() - started) / 3600,
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
