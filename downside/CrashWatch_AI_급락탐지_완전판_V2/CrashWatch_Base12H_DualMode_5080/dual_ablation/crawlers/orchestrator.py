from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..config import get_paths, load_assets, load_baskets
from ..io_utils import atomic_json
from .dart import collect_disclosures
from .global_assets import collect_global_assets
from .krx import collect_basket_tickers, collect_indices, collect_market_aggregate_flows, collect_market_cross_section
from .macro import collect_ecos, collect_kosis

LOGGER = logging.getLogger(__name__)


def run_crawlers(
    project: Path | None,
    start: str,
    end: str,
    *,
    overwrite: bool = False,
    with_market_panel: bool = False,
    sleep_seconds: float = 0.35,
) -> dict:
    paths = get_paths(project)
    paths.raw_dual.mkdir(parents=True, exist_ok=True)
    baskets = load_baskets(paths)
    assets = load_assets(paths)
    results: dict[str, object] = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "start": start, "end": end, "basket_ticker_count": len(baskets),
    }
    manifest = collect_basket_tickers(paths, baskets, start, end, overwrite=overwrite, sleep_seconds=sleep_seconds)
    results["krx_ticker_success"] = int(manifest["status"].isin(["ok", "cached"]).sum()) if not manifest.empty else 0
    results["krx_index_rows"] = len(collect_indices(paths, start, end, overwrite=overwrite))
    results["krx_market_flow_rows"] = len(collect_market_aggregate_flows(paths, start, end, overwrite=overwrite))
    results["global_asset_rows"] = len(collect_global_assets(paths, assets, start, end, overwrite=overwrite))
    results["dart_rows"] = len(collect_disclosures(paths, baskets, start, end, overwrite=overwrite, sleep_seconds=sleep_seconds))
    results["ecos_rows"] = len(collect_ecos(paths, start, end, overwrite=overwrite))
    results["kosis_rows"] = len(collect_kosis(paths, overwrite=overwrite))
    if with_market_panel:
        panel_manifest = collect_market_cross_section(paths, start, end, overwrite=overwrite, sleep_seconds=sleep_seconds)
        results["market_panel_days"] = int(panel_manifest["status"].isin(["ok", "cached"]).sum()) if not panel_manifest.empty else 0
    else:
        results["market_panel_days"] = 0
        results["market_panel_note"] = "--with-market-panel 옵션을 사용해야 전 종목 일별 횡단면을 수집합니다."
    atomic_json(results, paths.raw_dual / "crawl_run_summary.json")
    return results
