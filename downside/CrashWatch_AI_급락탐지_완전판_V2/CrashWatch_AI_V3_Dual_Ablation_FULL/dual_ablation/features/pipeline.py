from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..audit import (
    build_basket_coverage_report,
    build_crawl_manifest,
    feature_quality_audit,
    preflight_validate,
)
from ..config import ProjectPaths, get_paths, load_baskets
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_date, normalize_ticker
from .catalogs import build_catalogs
from .ticker import build_ticker_features
from .universe import build_universe_features

TARGET = "label_abs_crash_20"


def _future_crash_label(raw: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, block in raw.groupby("ticker", sort=False):
        block = block.sort_values("date").copy()
        close = pd.to_numeric(block["close"], errors="coerce")
        future = pd.concat([close.shift(-offset) for offset in range(1, 21)], axis=1)
        complete = future.notna().sum(axis=1).eq(20)
        future_min = future.min(axis=1, skipna=False)
        block[TARGET] = (safe_ratio(future_min, close) - 1.0).le(-0.12).where(complete).astype("float32")
        parts.append(block)
    return pd.concat(parts, ignore_index=True, sort=False)


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def _raw_labeled_base(paths: ProjectPaths, training_path: Path | None) -> tuple[pd.DataFrame, pd.DataFrame, Path | None]:
    raw_path = paths.raw_dual / "krx_ticker_timeseries.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(f"ticker OHLCV not found: {raw_path}")
    raw = normalize_date(pd.read_parquet(raw_path))
    raw["ticker"] = normalize_ticker(raw["ticker"])
    baskets = load_baskets(paths)
    raw = raw.drop(columns=[c for c in ["name", "bucket", "market", "role"] if c in raw], errors="ignore")
    raw = raw.merge(baskets[["ticker", "name", "bucket", "market", "role"]], on="ticker", how="inner")
    raw = raw.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    labeled = _future_crash_label(raw)

    candidates = [
        training_path,
        paths.project.parent / "crashwatch_ai_data" / "development" / "training_dataset.parquet",
        paths.data_root / "development" / "training_dataset.parquet",
    ]
    source = next((Path(p) for p in candidates if p is not None and Path(p).exists()), None)
    legacy = pd.read_parquet(source) if source else pd.DataFrame()
    if not legacy.empty:
        legacy = normalize_date(legacy)
        legacy["ticker"] = normalize_ticker(legacy["ticker"])
    labeled = labeled.loc[labeled[TARGET].notna()].copy()
    labeled[TARGET] = labeled[TARGET].astype("int8")
    return labeled, legacy, source


def _merge_legacy_features(base: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    if legacy.empty:
        base["v2_source_available"] = 0
        return base
    keys = ["date", "ticker"]
    reserved = set(keys + [
        "name", "market", "bucket", "role", TARGET, "open", "high", "low", "close", "volume",
        "sealed_do_not_train_or_tune",
    ])
    legacy_features = [c for c in legacy.columns if c not in reserved]
    source_rows = legacy[keys].drop_duplicates().assign(v2_source_available=1)
    out = base.merge(legacy[keys + legacy_features], on=keys, how="left", validate="one_to_one")
    out = out.merge(source_rows, on=keys, how="left", validate="one_to_one")
    out["v2_source_available"] = out["v2_source_available"].fillna(0).astype("int8")
    return out


def merge_with_training(
    paths: ProjectPaths,
    universe: pd.DataFrame,
    ticker: pd.DataFrame,
    training_path: Path | None = None,
) -> tuple[pd.DataFrame, Path, Path | None]:
    base, legacy, source = _raw_labeled_base(paths, training_path)
    merged = _merge_legacy_features(base, legacy)
    merged = merged.merge(universe, on="date", how="left", validate="many_to_one")
    merged = merged.merge(
        ticker.drop(columns=[c for c in ["name", "bucket", "market", "role"] if c in ticker]),
        on=["date", "ticker"], how="left", validate="one_to_one",
    )
    merged["sealed_do_not_train_or_tune"] = 0
    merged = merged.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
    output = paths.data_root / "development" / "training_dataset_dual.parquet"
    atomic_parquet(merged, output)
    return merged, output, source


def run_feature_pipeline(
    project: Path | None = None,
    training_path: Path | None = None,
    *,
    strict_quality_check: bool = False,
    allow_partial_data: bool = False,
) -> dict:
    paths = get_paths(project)
    paths.feature_dual.mkdir(parents=True, exist_ok=True)
    paths.meta_dual.mkdir(parents=True, exist_ok=True)
    baskets = load_baskets(paths)
    universe = build_universe_features(paths)
    ticker = build_ticker_features(paths, universe)
    atomic_parquet(universe, paths.feature_dual / "universe_features.parquet")
    atomic_parquet(ticker, paths.feature_dual / "ticker_features.parquet")
    universe_catalog, ticker_catalog = build_catalogs(paths, universe, ticker)
    merged, merged_path, legacy_source = merge_with_training(paths, universe, ticker, training_path)

    source_status = {
        "u_aggregate_flow": "source_failed", "u_aggregate_shorting": "source_failed",
        "u_realestate_korea": "unsafe_timestamp", "t_investor_flow": "source_failed",
        "t_short_selling": "source_failed",
    }
    catalogs = {"universe": universe_catalog, "ticker": ticker_catalog}
    quality = feature_quality_audit(merged, catalogs, source_status)
    atomic_csv(quality, paths.meta_dual / "feature_quality_audit.csv")
    atomic_csv(quality, paths.feature_dual / "feature_quality_audit.csv")
    coverage = build_basket_coverage_report(merged, baskets, TARGET)
    atomic_csv(coverage, paths.meta_dual / "basket_coverage_report.csv")
    crawl_manifest = build_crawl_manifest(
        paths, baskets, str(merged["date"].min().date()), str(merged["date"].max().date()),
    )
    preflight_errors = preflight_validate(
        merged, quality, coverage, allow_partial_data=allow_partial_data or not strict_quality_check,
        target=TARGET,
    )
    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "universe_rows": len(universe), "universe_feature_count": sum(map(len, universe_catalog.values())),
        "ticker_rows": len(ticker), "ticker_count": int(ticker["ticker"].nunique()),
        "ticker_feature_count": sum(map(len, ticker_catalog.values())),
        "merged_rows": len(merged), "merged_tickers": int(merged["ticker"].nunique()),
        "merged_path": str(merged_path), "legacy_source": str(legacy_source) if legacy_source else None,
        "ready_buckets": int(coverage["status"].eq("ready").sum()),
        "valid_features": int(quality["status"].eq("valid").sum()),
        "preflight_errors": preflight_errors, "allow_partial_data": allow_partial_data,
        "crawl_success": int(crawl_manifest["status"].eq("success").sum()),
        "crawl_partial_or_failed": int((~crawl_manifest["status"].eq("success")).sum()),
    }
    atomic_json(summary, paths.feature_dual / "feature_run_summary.json")
    return summary
