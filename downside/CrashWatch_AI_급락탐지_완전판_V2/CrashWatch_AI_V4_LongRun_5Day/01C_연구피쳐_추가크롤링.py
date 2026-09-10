#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from dual_ablation.config import get_paths, load_baskets
from dual_ablation.crawlers.attention import collect_naver_attention
from dual_ablation.crawlers.dart_enhanced import collect_financial_quality, collect_ownership_events
from dual_ablation.crawlers.etf_pressure import collect_etf_pressure
from dual_ablation.crawlers.fred_credit import collect_fred_credit
from dual_ablation.crawlers.optional_derivatives import import_derivatives_csv
from dual_ablation.io_utils import atomic_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V4 연구 피처용 추가 데이터 수집")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--skip-dart-enhanced", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    project = Path(__file__).resolve().parent
    paths = get_paths(project)
    paths.raw_dual.mkdir(parents=True, exist_ok=True)
    baskets = load_baskets(paths)
    result = {"created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(), "errors": {}}

    def collect(name: str, operation) -> int:
        try:
            return len(operation())
        except Exception as exc:  # noqa: BLE001
            logging.exception("%s 수집 실패; 다른 데이터 소스는 계속 진행합니다.", name)
            result["errors"][name] = f"{type(exc).__name__}: {exc}"
            return 0

    result["fred_credit_rows"] = collect(
        "fred_credit", lambda: collect_fred_credit(paths, args.start, args.end, overwrite=args.overwrite)
    )
    result["etf_pressure_rows"] = collect(
        "etf_pressure", lambda: collect_etf_pressure(paths, args.start, args.end, overwrite=args.overwrite)
    )
    result["derivatives_rows"] = collect(
        "derivatives", lambda: import_derivatives_csv(paths, overwrite=args.overwrite)
    )
    if not args.skip_attention:
        result["naver_attention_rows"] = collect(
            "naver_attention",
            lambda: collect_naver_attention(
                paths, baskets, args.start, args.end, overwrite=args.overwrite, sleep_seconds=args.sleep
            ),
        )
    else:
        result["naver_attention_rows"] = 0
    if not args.skip_dart_enhanced:
        result["dart_ownership_rows"] = collect(
            "dart_ownership",
            lambda: collect_ownership_events(paths, baskets, overwrite=args.overwrite, sleep_seconds=args.sleep),
        )
        result["dart_financial_rows"] = collect(
            "dart_financial",
            lambda: collect_financial_quality(
                paths, baskets, args.start, args.end, overwrite=args.overwrite, sleep_seconds=args.sleep
            ),
        )
    else:
        result["dart_ownership_rows"] = 0
        result["dart_financial_rows"] = 0
    atomic_json(result, paths.raw_dual / "research_crawl_summary.json")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
