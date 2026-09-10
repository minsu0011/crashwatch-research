#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from dual_ablation.config import get_paths
from dual_ablation.data_acquisition.common import load_local_env
from dual_ablation.data_acquisition.krx_actual import collect_krx_actual_data, verify_krx_actual_data


def main() -> None:
    parser = argparse.ArgumentParser(description="실제 KRX 공매도·수급·외국인보유·밸류에이션 시계열 수집")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.45)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--chunk-months", type=int, default=24)
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    load_local_env(project)
    paths = get_paths(project)
    end = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d") if args.end == "auto" else args.end
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = collect_krx_actual_data(
        paths,
        args.start,
        end,
        overwrite=args.overwrite,
        sleep_seconds=args.sleep_seconds,
        chunk_months=args.chunk_months,
    )
    if not args.allow_partial:
        result["mandatory_short"] = verify_krx_actual_data(paths)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
