#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dual_ablation.finance11h.features import build_finance_features


def main() -> None:
    parser = argparse.ArgumentParser(description="NumPy·Numba 금융 피처 생성")
    parser.add_argument("--training", type=Path, default=None)
    parser.add_argument("--allow-partial-short", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = build_finance_features(
        Path(__file__).resolve().parent,
        args.training,
        strict_short=not args.allow_partial_short,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
