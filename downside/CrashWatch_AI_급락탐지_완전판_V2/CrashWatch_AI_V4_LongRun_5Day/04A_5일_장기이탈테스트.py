#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dual_ablation.longrun.orchestrator import run_longrun


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrashWatch V4 4~5일 안전 장기 이탈테스트")
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parent / "configs" / "longrun_5day.json",
    )
    parser.add_argument("--reset", action="store_true", help="작업 상태만 초기화. prediction cache는 유지")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    result = run_longrun(Path(__file__).resolve().parent, args.config, reset=args.reset)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
