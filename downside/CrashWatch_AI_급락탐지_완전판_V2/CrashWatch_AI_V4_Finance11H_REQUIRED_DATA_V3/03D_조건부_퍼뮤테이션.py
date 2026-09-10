#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

from dual_ablation.experiment.permutation import run_grouped_permutation


def main() -> None:
    parser = argparse.ArgumentParser(description="그룹 단위 조건부 permutation importance")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--target", default="label_abs_crash_20")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--unconditional", action="store_true")
    args = parser.parse_args()
    result = run_grouped_permutation(Path(__file__).resolve().parent, args.dataset, args.target, args.seed, conditional=not args.unconditional)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
