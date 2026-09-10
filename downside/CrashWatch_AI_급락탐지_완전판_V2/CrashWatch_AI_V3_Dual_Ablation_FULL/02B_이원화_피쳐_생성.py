"""Build the V3 universe/ticker features and their quality/coverage audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_ablation.features.pipeline import run_feature_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch V3 dual feature builder")
    parser.add_argument("--training-path", type=Path)
    parser.add_argument("--strict-quality-check", action="store_true")
    parser.add_argument("--allow-partial-data", action="store_true")
    args = parser.parse_args()
    result = run_feature_pipeline(
        Path(__file__).resolve().parent,
        training_path=args.training_path,
        strict_quality_check=args.strict_quality_check,
        allow_partial_data=args.allow_partial_data,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
