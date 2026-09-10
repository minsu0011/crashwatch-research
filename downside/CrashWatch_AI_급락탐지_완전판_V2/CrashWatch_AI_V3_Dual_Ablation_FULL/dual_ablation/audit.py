from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .config import ProjectPaths
from .io_utils import atomic_csv, normalize_date, normalize_ticker

MIN_TICKERS_PER_BUCKET = 5
CORE_GROUPS = {
    "u_market_trend", "u_market_breadth", "u_market_volatility",
    "u_market_liquidity", "u_aggregate_flow", "u_macro_rates_fx",
    "t_price_trend", "t_volatility_tail", "t_investor_flow",
    "t_short_selling", "t_valuation_size", "t_disclosure_event",
}


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _status(frame: pd.DataFrame, date_col: str, critical: Iterable[str]) -> tuple[str, str, float, int]:
    if frame.empty:
        return "empty", "no rows returned", 0.0, 0
    if date_col not in frame.columns:
        return "failed", f"missing date column: {date_col}", 0.0, 0
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    if dates.isna().all():
        return "failed", "date column contains no valid dates", 0.0, 0
    if dates.duplicated().any():
        return "partial", "duplicate dates", 0.0, 0
    cols = [c for c in critical if c in frame.columns]
    if not cols:
        return "failed", "critical value columns missing", 0.0, 0
    non_null = float(frame[cols].notna().mean().mean())
    unique = int(max((frame[c].nunique(dropna=True) for c in cols), default=0))
    if non_null < 0.5:
        return "partial", "critical non-null ratio below 0.5", non_null, unique
    return "success", "", non_null, unique


def build_crawl_manifest(
    paths: ProjectPaths,
    baskets: pd.DataFrame,
    requested_start: str,
    requested_end: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    collected_at = pd.Timestamp.now(tz="Asia/Seoul").isoformat()
    ticker_raw_path = paths.raw_dual / "krx_ticker_timeseries.parquet"
    ticker_raw = pd.read_parquet(ticker_raw_path) if ticker_raw_path.exists() else pd.DataFrame()
    if not ticker_raw.empty:
        ticker_raw = normalize_date(ticker_raw)
        ticker_raw["ticker"] = normalize_ticker(ticker_raw["ticker"])
    disclosures_path = paths.raw_dual / "dart_disclosures.parquet"
    disclosures = pd.read_parquet(disclosures_path) if disclosures_path.exists() else pd.DataFrame()
    if not disclosures.empty:
        disclosures = normalize_date(disclosures)
        disclosures["ticker"] = normalize_ticker(disclosures["ticker"])
    raw_manifest_path = paths.raw_dual / "krx_tickers" / "manifest.csv"
    raw_manifest = pd.read_csv(raw_manifest_path, dtype={"ticker": str}) if raw_manifest_path.exists() else pd.DataFrame()
    attempt_map = (
        raw_manifest.assign(ticker=normalize_ticker(raw_manifest["ticker"])).set_index("ticker")
        .get("attempt_count", pd.Series(dtype=float)).to_dict()
        if not raw_manifest.empty else {}
    )
    krx_authenticated = bool(os.getenv("KRX_ID") and os.getenv("KRX_PW"))

    ticker_specs = [
        ("KRX", "ohlcv", ["close", "volume"]),
        ("KRX", "market_cap", ["market_cap"]),
        ("KRX", "fundamental_per", ["per"]),
        ("KRX", "fundamental_pbr", ["pbr"]),
        ("KRX", "dividend_yield", ["dividend_yield"]),
        ("KRX", "investor_foreign", ["flow_value_foreign"]),
        ("KRX", "investor_institution", ["flow_value_institution"]),
        ("KRX", "investor_individual", ["flow_value_individual"]),
        ("KRX", "foreign_ownership", ["foreign_ownership_rate"]),
        ("KRX", "short_trading", ["short_volume", "short_value"]),
        ("KRX", "short_balance", ["short_balance_value"]),
        ("OpenDART", "disclosure", ["report_nm"]),
    ]
    for _, basket in baskets.iterrows():
        ticker = str(basket["ticker"]).zfill(6)
        for source, dataset_type, critical in ticker_specs:
            block = disclosures.loc[disclosures.get("ticker", pd.Series(dtype=str)).eq(ticker)] if source == "OpenDART" else ticker_raw.loc[ticker_raw.get("ticker", pd.Series(dtype=str)).eq(ticker)]
            present_critical = [c for c in critical if c in block.columns]
            source_block = block[["date"] + present_critical].copy() if present_critical and "date" in block else pd.DataFrame()
            if source == "OpenDART" and not source_block.empty:
                # Multiple disclosures on one date are valid event rows.  Use
                # one row per date only for the time-series continuity test.
                status_block = source_block.drop_duplicates("date", keep="last")
                status, message, ratio, unique = _status(status_block, "date", critical)
            elif source == "KRX" and dataset_type != "ohlcv" and not present_critical and not krx_authenticated:
                status, message, ratio, unique = "skipped", "KRX_ID/KRX_PW not configured", 0.0, 0
            else:
                status, message, ratio, unique = _status(source_block, "date", critical)
            actual_start = source_block["date"].min() if not source_block.empty else pd.NaT
            actual_end = source_block["date"].max() if not source_block.empty else pd.NaT
            attempted = int(attempt_map.get(ticker, 1) or 0) if dataset_type == "ohlcv" else int(status != "skipped")
            rows.append({
                "ticker": ticker, "name": basket.get("name", ""), "bucket": basket.get("bucket", ""),
                "source": source, "dataset_type": dataset_type,
                "requested_start": requested_start, "requested_end": requested_end,
                "actual_start": actual_start, "actual_end": actual_end, "row_count": len(source_block),
                "non_null_ratio": ratio, "unique_value_count": unique, "status": status,
                "error_type": "" if status == "success" else ("MissingCredential" if status == "skipped" else ("SourceUnavailable" if not present_critical else "QualityFailure")),
                "error_message": message, "attempt_count": attempted, "collected_at": collected_at,
            })

    global_specs = [
        ("Yahoo", "KOSPI", paths.raw_dual / "global_assets.parquet", ["kospi_close"]),
        ("Yahoo", "KOSDAQ", paths.raw_dual / "global_assets.parquet", ["kosdaq_close"]),
        ("Yahoo", "VIX", paths.raw_dual / "global_assets.parquet", ["vix_close"]),
        ("Yahoo", "US_INDEX", paths.raw_dual / "global_assets.parquet", ["sp500_close", "nasdaq_close"]),
        ("Yahoo", "FX", paths.raw_dual / "global_assets.parquet", ["usdkrw_close", "dxy_close"]),
        ("Yahoo", "COMMODITY", paths.raw_dual / "global_assets.parquet", ["wti_close", "copper_close", "gold_close"]),
        ("FRED_ECOS", "RATES", paths.project.parent / "crashwatch_ai_data" / "raw" / "macro_fred.parquet", ["value"]),
        ("FRED_ECOS", "REAL_ESTATE", paths.project.parent / "crashwatch_ai_data" / "raw" / "macro_ecos.parquet", ["value"]),
    ]
    for source, dataset_type, path, critical in global_specs:
        block = pd.read_parquet(path) if path.exists() else pd.DataFrame()
        status_block = block.drop_duplicates("date", keep="last") if "date" in block else block
        status, message, ratio, unique = _status(status_block, "date", critical)
        if dataset_type == "REAL_ESTATE" and not block.empty:
            status, message = "partial", "unsafe_timestamp: release_date/available_from unavailable"
        rows.append({
            "ticker": "ALL", "name": "", "bucket": "universe", "source": source,
            "dataset_type": dataset_type, "requested_start": requested_start, "requested_end": requested_end,
            "actual_start": pd.to_datetime(block["date"], errors="coerce").min() if "date" in block else pd.NaT,
            "actual_end": pd.to_datetime(block["date"], errors="coerce").max() if "date" in block else pd.NaT,
            "row_count": len(block), "non_null_ratio": ratio, "unique_value_count": unique,
            "status": status, "error_type": "" if status == "success" else "QualityFailure",
            "error_message": message, "attempt_count": 1, "collected_at": collected_at,
        })
    derived_specs = [
        ("DERIVED_KRX_OHLCV", "MARKET_VALUE_TRADED", "partial", "close*volume proxy; authenticated market total unavailable"),
        ("DERIVED_48_TICKERS", "MARKET_BREADTH", "partial", "breadth is limited to the configured 48-ticker basket"),
        ("KRX", "MARKET_AGGREGATE_FLOW", "failed", "KRX endpoint returned non-JSON/login response"),
        ("UNAVAILABLE", "INVESTOR_SENTIMENT", "skipped", "no verified point-in-time source configured"),
    ]
    ticker_dates = pd.to_datetime(ticker_raw.get("date"), errors="coerce") if not ticker_raw.empty else pd.Series(dtype="datetime64[ns]")
    for source, dataset_type, status, message in derived_specs:
        has_proxy = not ticker_raw.empty and dataset_type in {"MARKET_VALUE_TRADED", "MARKET_BREADTH"}
        rows.append({
            "ticker": "ALL", "name": "", "bucket": "universe", "source": source,
            "dataset_type": dataset_type, "requested_start": requested_start, "requested_end": requested_end,
            "actual_start": ticker_dates.min() if has_proxy else pd.NaT,
            "actual_end": ticker_dates.max() if has_proxy else pd.NaT,
            "row_count": int(ticker_dates.nunique()) if has_proxy else 0,
            "non_null_ratio": 1.0 if has_proxy else 0.0, "unique_value_count": int(ticker_dates.nunique()) if has_proxy else 0,
            "status": status, "error_type": "SourceLimitation", "error_message": message,
            "attempt_count": int(dataset_type == "MARKET_AGGREGATE_FLOW"), "collected_at": collected_at,
        })
    out = pd.DataFrame(rows)
    paths.meta_dual.mkdir(parents=True, exist_ok=True)
    atomic_csv(out, paths.meta_dual / "crawl_manifest.csv")
    return out


def build_basket_coverage_report(
    dataset: pd.DataFrame, baskets: pd.DataFrame, target: str = "label_abs_crash_20",
    min_tickers: int = MIN_TICKERS_PER_BUCKET,
) -> pd.DataFrame:
    df = normalize_date(dataset)
    df["ticker"] = normalize_ticker(df["ticker"])
    planned_by_bucket = baskets.groupby("bucket")["ticker"].apply(lambda s: sorted(set(normalize_ticker(s))))
    rows = []
    for bucket, planned in planned_by_bucket.items():
        block = df.loc[df["ticker"].isin(planned)]
        available = sorted(block.loc[block[target].notna(), "ticker"].unique()) if target in block else []
        missing = sorted(set(planned) - set(available))
        positives = int(pd.to_numeric(block.get(target), errors="coerce").fillna(0).sum()) if target in block else 0
        negatives = int((pd.to_numeric(block.get(target), errors="coerce") == 0).sum()) if target in block else 0
        if not available:
            status, reason = "no_data", "no labeled rows"
        elif len(available) < min_tickers:
            status, reason = "insufficient_tickers", f"available_ticker_count={len(available)} < {min_tickers}"
        elif positives == 0 or negatives == 0:
            status, reason = "single_class", "target contains one class"
        else:
            status, reason = "ready", ""
        rows.append({
            "bucket": bucket, "planned_ticker_count": len(planned), "available_ticker_count": len(available),
            "missing_ticker_count": len(missing), "available_tickers": "|".join(available),
            "missing_tickers": "|".join(missing), "start_date": block["date"].min(),
            "end_date": block["date"].max(), "row_count": len(block), "positive_count": positives,
            "positive_rate": positives / len(block) if len(block) else np.nan,
            "status": status, "failure_reason": reason,
        })
    return pd.DataFrame(rows)


def feature_quality_audit(
    dataset: pd.DataFrame,
    catalogs: dict[str, dict[str, list[str]]],
    source_status: dict[str, str] | None = None,
) -> pd.DataFrame:
    rows = []
    source_status = source_status or {}
    for domain, groups in catalogs.items():
        for group, features in groups.items():
            for feature in features:
                source = group
                if feature not in dataset:
                    rows.append({"feature": feature, "domain": domain, "group": group, "source": source,
                                 "status": "missing_dependency", "failure_reason": "column missing"})
                    continue
                s = pd.to_numeric(dataset[feature], errors="coerce")
                inf_count = int(np.isinf(s).sum())
                clean = s.replace([np.inf, -np.inf], np.nan)
                non_null = int(clean.notna().sum())
                unique = int(clean.nunique(dropna=True))
                missing = float(clean.isna().mean())
                zero_ratio = float(clean.eq(0).mean())
                if source_status.get(group) == "unsafe_timestamp":
                    status, reason = "unsafe_timestamp", "release timestamp is not verifiable"
                elif source_status.get(group) == "source_failed" and non_null == 0:
                    status, reason = "source_failed", "source crawl failed"
                elif non_null == 0:
                    status, reason = "all_missing", "no non-null values"
                elif unique < 2:
                    status, reason = "constant", "fewer than two unique values"
                elif missing > 0.995:
                    status, reason = "mostly_missing", "missing ratio above 0.995"
                elif non_null < 200:
                    status, reason = "mostly_missing", "fewer than 200 non-null observations"
                else:
                    active_days = dataset.loc[clean.notna(), "date"].nunique() if "date" in dataset else 0
                    if active_days < 100:
                        status, reason = "mostly_missing", "fewer than 100 active trading days"
                    else:
                        status, reason = "valid", ""
                valid = clean.dropna()
                rows.append({
                    "feature": feature, "domain": domain, "group": group, "source": source,
                    "row_count": len(clean), "non_null_count": non_null, "missing_ratio": missing,
                    "unique_count": unique, "zero_ratio": zero_ratio, "inf_count": inf_count,
                    "min": valid.min() if len(valid) else np.nan, "max": valid.max() if len(valid) else np.nan,
                    "mean": valid.mean() if len(valid) else np.nan, "std": valid.std() if len(valid) else np.nan,
                    "first_valid_date": dataset.loc[clean.notna(), "date"].min() if "date" in dataset else pd.NaT,
                    "last_valid_date": dataset.loc[clean.notna(), "date"].max() if "date" in dataset else pd.NaT,
                    "status": status, "failure_reason": reason,
                })
    return pd.DataFrame(rows)


def preflight_validate(
    dataset: pd.DataFrame, quality: pd.DataFrame, coverage: pd.DataFrame,
    *, allow_partial_data: bool = False, target: str = "label_abs_crash_20",
) -> list[str]:
    errors: list[str] = []
    valid_groups = set(quality.loc[quality["status"].eq("valid"), "group"])
    missing_core = sorted(CORE_GROUPS - valid_groups)
    if missing_core:
        errors.append("core groups without a valid feature: " + ", ".join(missing_core))
    not_ready = coverage.loc[~coverage["status"].eq("ready"), ["bucket", "status"]]
    if not not_ready.empty:
        errors.append("buckets not ready: " + ", ".join(f"{r.bucket}={r.status}" for r in not_ready.itertuples()))
    dates = pd.to_datetime(dataset.get("date"), errors="coerce")
    if dates.isna().any() or getattr(dates.dt, "tz", None) is not None:
        errors.append("dates are invalid or timezone-aware")
    tickers = dataset.get("ticker", pd.Series(dtype=str)).astype(str)
    if not tickers.str.fullmatch(r"\d{6}").all():
        errors.append("ticker is not a six-digit string")
    if dataset.duplicated(["date", "ticker"]).any():
        errors.append("duplicate ticker-date rows")
    labels = set(pd.to_numeric(dataset.get(target), errors="coerce").dropna().unique())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        errors.append("label is not binary with both classes")
    if "sealed_do_not_train_or_tune" in dataset and pd.to_numeric(dataset["sealed_do_not_train_or_tune"], errors="coerce").fillna(0).ne(0).any():
        errors.append("sealed rows are present")
    if errors and not allow_partial_data:
        raise ValueError("Preflight failed:\n- " + "\n- ".join(errors))
    return errors
