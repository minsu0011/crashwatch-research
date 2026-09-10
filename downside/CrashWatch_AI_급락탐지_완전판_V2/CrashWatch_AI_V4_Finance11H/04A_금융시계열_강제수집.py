from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from dual_ablation.finance11h.crawler import run_finance_crawl


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance11H 금융·공매도 시계열 수집")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.15)
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()
    end = (
        pd.Timestamp.now(tz="Asia/Seoul").normalize().tz_localize(None).strftime("%Y-%m-%d")
        if args.end == "auto" else args.end
    )
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    result = run_finance_crawl(
        args.project, args.start, end,
        overwrite=args.overwrite, sleep_seconds=args.sleep_seconds,
        strict=not args.no_strict,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
