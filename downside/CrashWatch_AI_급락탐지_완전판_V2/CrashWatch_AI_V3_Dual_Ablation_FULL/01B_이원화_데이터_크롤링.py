"""Collect the V3 dual-ablation raw sources and write truthful audit metadata."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from dual_ablation.audit import build_crawl_manifest
from dual_ablation.config import get_paths, load_baskets
from dual_ablation.crawlers.orchestrator import run_crawlers


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch V3 dual-ablation data collector")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="auto")
    parser.add_argument("--retry", type=int, default=3, choices=(1, 2, 3))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--with-market-panel", action="store_true")
    args = parser.parse_args()
    end = pd.Timestamp.now(tz="Asia/Seoul").date().isoformat() if args.end == "auto" else args.end
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    project = Path(__file__).resolve().parent
    result = run_crawlers(
        project, args.start, end, overwrite=args.overwrite,
        with_market_panel=args.with_market_panel,
    )
    paths = get_paths(project)
    manifest = build_crawl_manifest(paths, load_baskets(paths), args.start, end)
    result["manifest_rows"] = len(manifest)
    result["manifest_status"] = manifest["status"].value_counts(dropna=False).to_dict()
    result["retry_limit"] = args.retry
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
