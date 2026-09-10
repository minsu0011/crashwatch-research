#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_ablation.features.pipeline import run_feature_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="universe/ticker 피처를 독립 생성하고 기존 training dataset에 결합")
    parser.add_argument("--training-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_feature_pipeline(Path(__file__).resolve().parent, args.training_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
