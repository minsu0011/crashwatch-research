#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from dual_ablation.data_acquisition.runner import run_required_data_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance11H 필수 실데이터 통합 수집")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="auto")
    parser.add_argument("--sources", default="macro,krx,lending,dart,naver")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true", help="실제 공매도 80%% 커버리지 미달 시 실패")
    parser.add_argument("--no-dart-documents", action="store_true")
    parser.add_argument("--naver-start", default="2020-01-01")
    parser.add_argument("--krx-chunk-months", type=int, default=24)
    parser.add_argument("--lending-chunk-months", type=int, default=12)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    end = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d") if args.end == "auto" else args.end
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = run_required_data_download(
        project,
        args.start,
        end,
        sources=[x.strip().lower() for x in args.sources.split(",") if x.strip()],
        overwrite=args.overwrite,
        strict=args.strict,
        dart_documents=not args.no_dart_documents,
        naver_start=args.naver_start,
        krx_chunk_months=args.krx_chunk_months,
        lending_chunk_months=args.lending_chunk_months,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
