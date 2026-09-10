from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from dual_ablation.finance11h.features import build_finance_features
from dual_ablation.finance11h.numba_kernels import warmup


def main() -> None:
    parser = argparse.ArgumentParser(description="Finance11H Numba 금융 피처 생성")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--training-path", type=Path)
    parser.add_argument("--no-strict-short", action="store_true")
    parser.add_argument("--numba-diagnostics-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.numba_diagnostics_only:
        result = warmup()
    else:
        result = build_finance_features(
            args.project, args.training_path,
            strict_short=not args.no_strict_short,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
