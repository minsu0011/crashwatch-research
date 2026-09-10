#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_ablation.experiment.runner import run_ablation

RESEARCH_GROUPS = {
    "u_market_microstructure", "u_tail_network", "u_attention", "u_derivatives_risk",
    "u_credit_funding", "u_etf_pressure", "t_range_volatility", "t_microstructure_proxy",
    "t_tail_dependence", "t_volume_price_pressure", "t_limit_stress", "t_attention",
    "t_ownership_governance", "t_fundamental_quality", "t_network_contagion",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V4 추가 연구 피처만 계층형 이탈테스트")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--modes", default="universe,ticker_global,bucket")
    parser.add_argument("--groups", default=None, help="생략하면 V4 연구 그룹 전체")
    parser.add_argument("--buckets", default=None)
    parser.add_argument("--tickers", default=None)
    parser.add_argument("--seeds", default="17,43,101")
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--validation-days", type=int, default=60)
    parser.add_argument("--purge-days", type=int, default=20)
    parser.add_argument("--min-train-days", type=int, default=500)
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def parse_set(value: str | None) -> set[str] | None:
    return {x.strip() for x in value.split(",") if x.strip()} if value else None


def main() -> None:
    args = parse_args()
    groups = parse_set(args.groups) or RESEARCH_GROUPS
    result = run_ablation(
        Path(__file__).resolve().parent,
        args.dataset,
        target="label_abs_crash_20",
        modes=parse_set(args.modes) or set(),
        groups=groups,
        buckets=parse_set(args.buckets),
        tickers=parse_set(args.tickers),
        seeds=[int(x) for x in args.seeds.split(",") if x.strip()],
        n_folds=args.folds,
        validation_days=args.validation_days,
        purge_days=args.purge_days,
        min_train_days=args.min_train_days,
        max_train_rows=300000,
        overwrite_cache=args.overwrite_cache,
        prefer_gpu=not args.cpu,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
