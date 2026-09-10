#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from dual_ablation.crawlers.orchestrator import run_crawlers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="국장 전체용 + 종목 특화용 이원화 raw 데이터 수집")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--with-market-panel", action="store_true", help="전 종목 일별 횡단면 수집. 호출량이 많음")
    parser.add_argument("--sleep", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    result = run_crawlers(Path(__file__).resolve().parent, args.start, args.end, overwrite=args.overwrite, with_market_panel=args.with_market_panel, sleep_seconds=args.sleep)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
