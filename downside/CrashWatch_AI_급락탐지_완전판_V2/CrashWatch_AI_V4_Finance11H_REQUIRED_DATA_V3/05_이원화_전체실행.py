#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dual_ablation.crawlers.orchestrator import run_crawlers
from dual_ablation.config import get_paths, load_baskets
from dual_ablation.crawlers.attention import collect_naver_attention
from dual_ablation.crawlers.dart_enhanced import collect_financial_quality, collect_ownership_events
from dual_ablation.crawlers.etf_pressure import collect_etf_pressure
from dual_ablation.crawlers.fred_credit import collect_fred_credit
from dual_ablation.crawlers.optional_derivatives import import_derivatives_csv
from dual_ablation.experiment.runner import run_ablation
from dual_ablation.features.pipeline import run_feature_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="이원화 크롤링 -> 피처 -> 이탈테스트 전체 실행")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--with-market-panel", action="store_true")
    parser.add_argument("--skip-crawl", action="store_true")
    parser.add_argument("--skip-research-crawl", action="store_true")
    parser.add_argument("--skip-attention", action="store_true")
    parser.add_argument("--skip-dart-enhanced", action="store_true")
    parser.add_argument("--modes", default="universe,ticker_global,bucket")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    project = Path(__file__).resolve().parent
    if not args.skip_crawl:
        run_crawlers(project, args.start, args.end, with_market_panel=args.with_market_panel)
    if not args.skip_research_crawl:
        paths = get_paths(project)
        baskets = load_baskets(paths)
        collect_fred_credit(paths, args.start, args.end)
        collect_etf_pressure(paths, args.start, args.end)
        import_derivatives_csv(paths)
        if not args.skip_attention:
            collect_naver_attention(paths, baskets, args.start, args.end)
        if not args.skip_dart_enhanced:
            collect_ownership_events(paths, baskets)
            collect_financial_quality(paths, baskets, args.start, args.end)
    run_feature_pipeline(project)
    run_ablation(project, modes={x.strip() for x in args.modes.split(",") if x.strip()}, prefer_gpu=not args.cpu)


if __name__ == "__main__":
    main()
