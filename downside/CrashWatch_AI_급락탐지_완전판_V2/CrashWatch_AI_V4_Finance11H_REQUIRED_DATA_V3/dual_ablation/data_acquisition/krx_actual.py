from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import ProjectPaths
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_date, normalize_ticker, read_parquet_tree
from .common import (
    SourceRecord,
    append_manifest,
    date8,
    date_chunks,
    normalize_frame,
    now_iso,
    read_enabled_tickers,
    retry_call,
    sanitize_secret_error,
    sha256_file,
    sleep_with_jitter,
)

LOGGER = logging.getLogger(__name__)

# pykrx returns Korean display names. Keep the raw meaning explicit and only
# apply fs_ prefixes at the Finance11H integration boundary.
SHORT_VOLUME_MAP = {
    "공매도": "short_trade_volume",
    "공매도량": "short_trade_volume",
    "매수": "total_trade_volume",
    "거래량": "total_trade_volume",
    "비중": "short_volume_ratio",
}
SHORT_BALANCE_MAP = {
    "공매도잔고": "short_balance_shares",
    "잔고": "short_balance_shares",
    "잔고수량": "short_balance_shares",
    "상장주식수": "listed_shares",
    "공매도금액": "short_balance_value",
    "잔고금액": "short_balance_value",
    "시가총액": "market_cap",
    "비중": "short_balance_ratio",
}
SHORT_STATUS_MAP = {
    "공매도": "short_trade_volume_status",
    "공매도량": "short_trade_volume_status",
    "잔고": "short_balance_shares_status",
    "공매도잔고": "short_balance_shares_status",
    "공매도금액": "short_trade_value_status",
    "거래대금": "short_trade_value_status",
    "잔고금액": "short_balance_value_status",
}
INVESTOR_MAP = {
    "금융투자": "financial_investment",
    "보험": "insurance",
    "투신": "investment_trust",
    "사모": "private_fund",
    "은행": "bank",
    "기타금융": "other_finance",
    "연기금": "pension",
    "연기금등": "pension",
    "기관합계": "institution",
    "기관": "institution",
    "기타법인": "other_corporation",
    "개인": "individual",
    "외국인": "foreign",
    "외국인합계": "foreign",
    "기타외국인": "other_foreign",
    "기타": "other",
    "전체": "total",
    "합계": "total",
}
FOREIGN_MAP = {
    "상장주식수": "listed_shares",
    "보유수량": "foreign_owned_shares",
    "외국인보유주식수": "foreign_owned_shares",
    "지분율": "foreign_ownership_rate",
    "한도수량": "foreign_limit_shares",
    "한도소진률": "foreign_limit_exhaustion_rate",
}
MARKET_CAP_MAP = {
    "시가총액": "market_cap",
    "거래량": "market_volume",
    "거래대금": "market_trading_value",
    "상장주식수": "listed_shares",
    "외국인보유주식수": "foreign_owned_shares",
}
FUNDAMENTAL_MAP = {
    "BPS": "bps",
    "PER": "per",
    "PBR": "pbr",
    "EPS": "eps",
    "DIV": "dividend_yield",
    "배당수익률": "dividend_yield",
    "DPS": "dps",
}
SHORT_INVESTOR_MAP = {
    "기관": "institution",
    "개인": "individual",
    "외국인": "foreign",
    "기타": "other",
    "합계": "total",
}


def _pykrx_stock():
    missing = [key for key in ("KRX_ID", "KRX_PW") if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "KRX 로그인 세션에 필요한 환경변수가 없습니다: " + ", ".join(missing)
            + ". 06_자격증명_로컬설정.py를 실행해 이 PC의 .env.data.local에만 저장하세요."
        )
    try:
        from pykrx import stock
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pykrx==1.2.8 설치가 필요합니다. SETUP_REQUIRED_DATA.bat을 실행하세요.") from exc
    return stock


def _call_first(stock: Any, names: list[str], *args: Any, **kwargs: Any) -> pd.DataFrame:
    errors: list[str] = []
    for name in names:
        fn = getattr(stock, name, None)
        if fn is None:
            continue
        try:
            return fn(*args, **kwargs)
        except TypeError as exc:
            errors.append(f"{name}: {exc}")
    raise AttributeError(f"지원 함수 없음/인자 불일치: {names}; {' | '.join(errors)}")


def _prefix_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return frame.rename(columns={column: f"{prefix}{column}" for column in frame.columns if column != "date"})


def _normalize_percent_points(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert KRX/pykrx percentage-point outputs to unit fractions.

    pykrx examples report 0.06 for 0.06%, and 13.02 for 13.02%. Those are
    percent points, not fractions. Finance11H internally uses fractions.
    """
    out = frame.copy()
    for column in columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce") / 100.0
    return out


def _collect_one_source(stock: Any, source: str, ticker: str, start8: str, end8: str) -> pd.DataFrame:
    if source == "short_status":
        raw = _call_first(stock, ["get_shorting_status_by_date"], start8, end8, ticker)
        return normalize_frame(raw, SHORT_STATUS_MAP)
    if source == "short_volume":
        raw = _call_first(stock, ["get_shorting_volume_by_date"], start8, end8, ticker)
        return _normalize_percent_points(normalize_frame(raw, SHORT_VOLUME_MAP), ["short_volume_ratio"])
    if source == "short_balance":
        raw = _call_first(stock, ["get_shorting_balance_by_date"], start8, end8, ticker)
        return _normalize_percent_points(normalize_frame(raw, SHORT_BALANCE_MAP), ["short_balance_ratio"])
    if source == "investor_value":
        raw = _call_first(stock, ["get_market_trading_value_by_date"], start8, end8, ticker)
        return _prefix_columns(normalize_frame(raw, INVESTOR_MAP), "net_value_")
    if source == "investor_volume":
        raw = _call_first(stock, ["get_market_trading_volume_by_date"], start8, end8, ticker)
        return _prefix_columns(normalize_frame(raw, INVESTOR_MAP), "net_volume_")
    if source == "foreign_ownership":
        raw = _call_first(stock, ["get_exhaustion_rates_of_foreign_investment"], start8, end8, ticker)
        return _normalize_percent_points(
            normalize_frame(raw, FOREIGN_MAP),
            ["foreign_ownership_rate", "foreign_limit_exhaustion_rate"],
        )
    if source == "market_cap":
        raw = _call_first(stock, ["get_market_cap"], start8, end8, ticker)
        return normalize_frame(raw, MARKET_CAP_MAP)
    if source == "fundamental":
        raw = _call_first(stock, ["get_market_fundamental"], start8, end8, ticker)
        return _normalize_percent_points(normalize_frame(raw, FUNDAMENTAL_MAP), ["dividend_yield"])
    raise KeyError(source)


def _source_valid(frame: pd.DataFrame, source: str) -> bool:
    if frame.empty:
        return False
    required = {
        "short_status": ["short_trade_volume_status", "short_trade_value_status", "short_balance_shares_status", "short_balance_value_status"],
        "short_volume": ["short_trade_volume", "total_trade_volume", "short_volume_ratio"],
        "short_balance": ["short_balance_shares", "short_balance_value", "market_cap"],
        "investor_value": ["net_value_institution", "net_value_foreign", "net_value_individual"],
        "investor_volume": ["net_volume_institution", "net_volume_foreign", "net_volume_individual"],
        "foreign_ownership": ["foreign_owned_shares", "foreign_ownership_rate"],
        "market_cap": ["market_cap", "market_trading_value", "listed_shares"],
        "fundamental": ["per", "pbr"],
    }.get(source, [])
    existing = [column for column in required if column in frame.columns]
    return bool(existing) and float(frame[existing].notna().to_numpy().mean()) > 0.01


def _coalescing_merge(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [normalize_date(frame).sort_values("date").drop_duplicates("date", keep="last") for frame in frames if not frame.empty]
    if not valid:
        return pd.DataFrame(columns=["date"])
    all_dates = pd.Index(sorted(set().union(*(set(frame["date"].dropna()) for frame in valid))), name="date")
    out = pd.DataFrame({"date": all_dates})
    for frame in valid:
        indexed = frame.set_index("date")
        for column in indexed.columns:
            values = indexed[column].reindex(all_dates).reset_index(drop=True)
            if column in out.columns:
                out[column] = out[column].combine_first(values)
            else:
                out[column] = values
    return out.sort_values("date").reset_index(drop=True)


def _resolve_overlaps(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    pairs = [
        ("short_trade_volume", "short_trade_volume_status"),
        ("short_trade_value", "short_trade_value_status"),
        ("short_balance_shares", "short_balance_shares_status"),
        ("short_balance_value", "short_balance_value_status"),
    ]
    for primary, fallback in pairs:
        if primary not in out.columns and fallback in out.columns:
            out[primary] = out[fallback]
        elif primary in out.columns and fallback in out.columns:
            out[primary] = out[primary].combine_first(out[fallback])

    if "short_volume_ratio" not in out.columns and {"short_trade_volume", "total_trade_volume"}.issubset(out.columns):
        denominator = pd.to_numeric(out["total_trade_volume"], errors="coerce").replace(0, np.nan)
        out["short_volume_ratio"] = pd.to_numeric(out["short_trade_volume"], errors="coerce") / denominator
    if "short_value_ratio" not in out.columns and {"short_trade_value", "market_trading_value"}.issubset(out.columns):
        denominator = pd.to_numeric(out["market_trading_value"], errors="coerce").replace(0, np.nan)
        out["short_value_ratio"] = pd.to_numeric(out["short_trade_value"], errors="coerce") / denominator
    if "short_balance_ratio" not in out.columns and {"short_balance_shares", "listed_shares"}.issubset(out.columns):
        denominator = pd.to_numeric(out["listed_shares"], errors="coerce").replace(0, np.nan)
        out["short_balance_ratio"] = pd.to_numeric(out["short_balance_shares"], errors="coerce") / denominator
    return out


def _collect_market_short_investor(
    stock: Any,
    paths: ProjectPaths,
    start: str,
    end: str,
    *,
    overwrite: bool,
    chunk_months: int,
    sleep_seconds: float,
) -> dict[str, Any]:
    root = paths.raw_dual / "required_data_v3" / "krx_actual" / "market_short_investor"
    source_root = root / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "market_short_investor_manifest.csv"
    records: list[SourceRecord] = []

    for market in ("KOSPI", "KOSDAQ"):
        for left, right in date_chunks(start, end, months=chunk_months):
            start8, end8 = date8(left), date8(right)
            chunk_id = f"{start8}_{end8}"
            for kind, function_name in (
                ("volume", "get_shorting_investor_volume_by_date"),
                ("value", "get_shorting_investor_value_by_date"),
            ):
                path = source_root / f"market={market}" / f"kind={kind}" / f"{chunk_id}.parquet"
                started = now_iso()
                timer = time.perf_counter()
                if path.exists() and not overwrite:
                    frame = pd.read_parquet(path)
                    record = SourceRecord(
                        "KRX/pykrx",
                        f"market_short_investor_{kind}",
                        f"{market}:{chunk_id}",
                        "cached",
                        len(frame),
                        started,
                        now_iso(),
                        time.perf_counter() - timer,
                        str(path.relative_to(paths.project)),
                        "",
                        sha256_file(path),
                    )
                else:
                    try:
                        raw = retry_call(
                            lambda: getattr(stock, function_name)(start8, end8, market),
                            attempts=4,
                            base_delay=2.0,
                            max_delay=45.0,
                        )
                        frame = _prefix_columns(normalize_frame(raw, SHORT_INVESTOR_MAP), f"short_{kind}_")
                        frame["market"] = market
                        if not frame.empty:
                            atomic_parquet(frame, path)
                        record = SourceRecord(
                            "KRX/pykrx",
                            f"market_short_investor_{kind}",
                            f"{market}:{chunk_id}",
                            "success" if len(frame) else "empty",
                            len(frame),
                            started,
                            now_iso(),
                            time.perf_counter() - timer,
                            str(path.relative_to(paths.project)) if path.exists() else "",
                            "",
                            sha256_file(path) if path.exists() else "",
                        )
                    except Exception as exc:  # noqa: BLE001
                        record = SourceRecord(
                            "KRX/pykrx",
                            f"market_short_investor_{kind}",
                            f"{market}:{chunk_id}",
                            "failed",
                            0,
                            started,
                            now_iso(),
                            time.perf_counter() - timer,
                            "",
                            sanitize_secret_error(f"{type(exc).__name__}: {exc}"),
                            "",
                        )
                records.append(record)
                append_manifest([record], manifest_path)
                sleep_with_jitter(sleep_seconds)

    source_files = sorted(source_root.rglob("*.parquet"))
    frames = [pd.read_parquet(path) for path in source_files]
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not combined.empty:
        combined = normalize_date(combined).sort_values(["market", "date"]).drop_duplicates(["market", "date"], keep="last")
        atomic_csv(combined, root / "krx_market_short_investor_daily.csv")
        atomic_parquet(combined, root / "krx_market_short_investor_daily.parquet")
    return {
        "rows": len(combined),
        "markets": sorted(combined["market"].dropna().unique().tolist()) if "market" in combined else [],
        "manifest_rows": len(records),
    }


def collect_krx_actual_data(
    paths: ProjectPaths,
    start: str,
    end: str,
    *,
    overwrite: bool = False,
    chunk_months: int = 24,
    sleep_seconds: float = 0.45,
    sources: tuple[str, ...] = (
        "short_status",
        "short_volume",
        "short_balance",
        "investor_value",
        "investor_volume",
        "foreign_ownership",
        "market_cap",
        "fundamental",
    ),
) -> dict[str, Any]:
    stock = _pykrx_stock()
    baskets = read_enabled_tickers(paths)
    root = paths.raw_dual / "required_data_v3" / "krx_actual"
    source_root = root / "sources"
    source_root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "krx_actual_manifest.csv"
    records: list[SourceRecord] = []
    started_all = time.perf_counter()

    for ticker_row in baskets.itertuples(index=False):
        ticker = str(ticker_row.ticker).zfill(6)
        LOGGER.info("KRX actual %s %s", ticker, getattr(ticker_row, "name", ""))
        for left, right in date_chunks(start, end, months=chunk_months):
            start8, end8 = date8(left), date8(right)
            chunk_id = f"{start8}_{end8}"
            for source in sources:
                path = source_root / f"ticker={ticker}" / f"source={source}" / f"{chunk_id}.parquet"
                record_started = now_iso()
                timer = time.perf_counter()
                if path.exists() and not overwrite:
                    try:
                        cached = pd.read_parquet(path)
                        record = SourceRecord(
                            "KRX/pykrx", source, f"{ticker}:{chunk_id}", "cached", len(cached),
                            record_started, now_iso(), time.perf_counter() - timer,
                            str(path.relative_to(paths.project)), "", sha256_file(path),
                        )
                        records.append(record)
                        append_manifest([record], manifest_path)
                        continue
                    except Exception:
                        path.unlink(missing_ok=True)
                try:
                    frame = retry_call(
                        lambda: _collect_one_source(stock, source, ticker, start8, end8),
                        attempts=4,
                        base_delay=2.0,
                        max_delay=45.0,
                    )
                    if not frame.empty:
                        frame["ticker"] = ticker
                        frame["bucket"] = str(getattr(ticker_row, "bucket", ""))
                        frame["name"] = str(getattr(ticker_row, "name", ""))
                        atomic_parquet(frame, path)
                        status = "success" if _source_valid(frame, source) else "partial"
                        checksum = sha256_file(path)
                    else:
                        status = "empty"
                        checksum = ""
                    record = SourceRecord(
                        "KRX/pykrx", source, f"{ticker}:{chunk_id}", status, len(frame),
                        record_started, now_iso(), time.perf_counter() - timer,
                        str(path.relative_to(paths.project)) if path.exists() else "", "", checksum,
                    )
                except Exception as exc:  # noqa: BLE001
                    error = sanitize_secret_error(f"{type(exc).__name__}: {exc}")
                    LOGGER.error("KRX 실패 %s %s %s: %s", ticker, chunk_id, source, error)
                    record = SourceRecord(
                        "KRX/pykrx", source, f"{ticker}:{chunk_id}", "failed", 0,
                        record_started, now_iso(), time.perf_counter() - timer, "", error, "",
                    )
                records.append(record)
                append_manifest([record], manifest_path)
                sleep_with_jitter(sleep_seconds)

    market_short = _collect_market_short_investor(
        stock,
        paths,
        start,
        end,
        overwrite=overwrite,
        chunk_months=chunk_months,
        sleep_seconds=sleep_seconds,
    )
    combined = build_krx_combined(paths)
    summary = summarize_krx_actual(paths, combined)
    summary.update({
        "started_at": records[0].started_at if records else now_iso(),
        "completed_at": now_iso(),
        "elapsed_seconds": time.perf_counter() - started_all,
        "requested_start": start,
        "requested_end": end,
        "chunk_months": chunk_months,
        "manifest_rows_this_run": len(records),
        "market_short_investor": market_short,
    })
    atomic_json(summary, root / "krx_actual_summary.json")
    return summary


def build_krx_combined(paths: ProjectPaths) -> pd.DataFrame:
    root = paths.raw_dual / "required_data_v3" / "krx_actual"
    source_root = root / "sources"
    baskets = read_enabled_tickers(paths)
    combined_tickers: list[pd.DataFrame] = []
    for ticker_row in baskets.itertuples(index=False):
        ticker = str(ticker_row.ticker).zfill(6)
        ticker_root = source_root / f"ticker={ticker}"
        source_frames: list[pd.DataFrame] = []
        if ticker_root.exists():
            for source_dir in sorted(ticker_root.glob("source=*")):
                # A licensed import source can contain complementary files for
                # the same date (for example, one trade CSV and one balance
                # CSV). Concatenating and dropping duplicate dates discards
                # one side. Coalesce partitions by date first so both column
                # families survive.
                partition_frames: list[pd.DataFrame] = []
                for partition_path in sorted(source_dir.rglob("*.parquet")):
                    partition = pd.read_parquet(partition_path)
                    if partition.empty:
                        continue
                    partition = (
                        normalize_date(partition)
                        .sort_values("date")
                        .drop_duplicates("date", keep="last")
                        .drop(columns=["ticker", "bucket", "name"], errors="ignore")
                    )
                    partition_frames.append(partition)
                frame = _coalescing_merge(partition_frames)
                if frame.empty:
                    continue
                source_frames.append(frame)
        merged = _resolve_overlaps(_coalescing_merge(source_frames))
        if merged.empty:
            continue
        merged["ticker"] = ticker
        merged["bucket"] = str(getattr(ticker_row, "bucket", ""))
        merged["name"] = str(getattr(ticker_row, "name", ""))
        combined_tickers.append(merged)

    combined = pd.concat(combined_tickers, ignore_index=True, sort=False) if combined_tickers else pd.DataFrame()
    if combined.empty:
        return combined

    combined["ticker"] = normalize_ticker(combined["ticker"])
    combined = normalize_date(combined).sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    atomic_parquet(combined, root / "krx_actual_ticker_daily.parquet")
    atomic_csv(combined, root / "krx_actual_ticker_daily.csv")

    # Finance11H integration. Only actual KRX short data receive fs_short_* names;
    # lending data are intentionally kept in a separate namespace.
    finance = combined.copy()
    prefix_map = {
        "short_trade_volume": "fs_short_trade_volume",
        "total_trade_volume": "fs_total_trade_volume",
        "short_volume_ratio": "fs_short_volume_ratio",
        "short_trade_value": "fs_short_trade_value",
        "market_trading_value": "fv_trading_value",
        "short_value_ratio": "fs_short_value_ratio",
        "short_balance_shares": "fs_short_balance_shares",
        "short_balance_value": "fs_short_balance_value",
        "short_balance_ratio": "fs_short_balance_ratio",
        "market_cap": "fv_market_cap",
        "market_volume": "fv_volume",
        "listed_shares": "fv_listed_shares",
        "foreign_owned_shares": "ff_foreign_owned_shares",
        "foreign_ownership_rate": "ff_foreign_ownership_rate",
        "foreign_limit_shares": "ff_foreign_limit_shares",
        "foreign_limit_exhaustion_rate": "ff_foreign_limit_exhaustion_rate",
        "per": "fv_per",
        "pbr": "fv_pbr",
        "eps": "fv_eps",
        "bps": "fv_bps",
        "dividend_yield": "fv_dividend_yield",
        "dps": "fv_dps",
    }
    prefix_map.update({
        column: f"ff_{column}"
        for column in finance.columns
        if column.startswith("net_value_") or column.startswith("net_volume_")
    })
    finance = finance.rename(columns=prefix_map)
    expected_root = paths.raw_dual / "finance11h"
    expected_root.mkdir(parents=True, exist_ok=True)
    atomic_parquet(finance, expected_root / "finance_ticker_timeseries.parquet")
    atomic_csv(finance, expected_root / "finance_ticker_timeseries.csv")
    return combined


def summarize_krx_actual(paths: ProjectPaths, combined: pd.DataFrame | None = None) -> dict[str, Any]:
    root = paths.raw_dual / "required_data_v3" / "krx_actual"
    if combined is None:
        path = root / "krx_actual_ticker_daily.parquet"
        combined = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    baskets = read_enabled_tickers(paths)
    required_columns = [
        "short_trade_volume",
        "short_trade_value",
        "short_volume_ratio",
        "short_balance_shares",
        "short_balance_value",
        "market_cap",
    ]
    rows: list[dict[str, Any]] = []
    for ticker in baskets["ticker"]:
        block = combined.loc[combined.get("ticker", pd.Series(dtype=str)).eq(ticker)].copy() if not combined.empty else pd.DataFrame()
        row: dict[str, Any] = {"ticker": ticker, "rows": len(block)}
        if len(block):
            start_date = pd.to_datetime(block["date"], errors="coerce").min()
            end_date = pd.to_datetime(block["date"], errors="coerce").max()
            span_days = int((end_date - start_date).days) if pd.notna(start_date) and pd.notna(end_date) else 0
        else:
            start_date = end_date = pd.NaT
            span_days = 0
        row["start_date"] = str(start_date.date()) if pd.notna(start_date) else ""
        row["end_date"] = str(end_date.date()) if pd.notna(end_date) else ""
        row["span_days"] = span_days
        for column in required_columns:
            row[f"{column}_non_null_ratio"] = float(block[column].notna().mean()) if column in block and len(block) else 0.0
        minimum_ratio = min(row[f"{column}_non_null_ratio"] for column in required_columns)
        row["required_min_non_null_ratio"] = minimum_ratio
        # At least ~3 years of calendar coverage and ~2.4 years of trading rows.
        row["status"] = "ready" if len(block) >= 600 and span_days >= 900 and minimum_ratio >= 0.50 else "insufficient"
        rows.append(row)
    coverage = pd.DataFrame(rows)
    atomic_csv(coverage, root / "krx_actual_coverage_48_tickers.csv")
    ready = int(coverage["status"].eq("ready").sum()) if not coverage.empty else 0
    non_null_actual = bool(
        len(combined)
        and all(column in combined.columns for column in required_columns)
        and combined[required_columns].notna().any().all()
    )
    return {
        "ticker_count": len(baskets),
        "ready_tickers": ready,
        "ticker_coverage": ready / len(baskets) if len(baskets) else 0.0,
        "combined_rows": len(combined),
        "required_columns": required_columns,
        "ready_rule": {"minimum_rows": 600, "minimum_span_days": 900, "minimum_non_null_ratio": 0.50},
        "actual_short_data": non_null_actual,
    }


def verify_krx_actual_data(paths: ProjectPaths, minimum_ticker_coverage: float = 0.80) -> dict[str, Any]:
    summary = summarize_krx_actual(paths)
    if not summary["actual_short_data"] or summary["ticker_coverage"] < minimum_ticker_coverage:
        raise RuntimeError(
            f"실제 KRX 공매도 커버리지 부족: {summary['ticker_coverage']:.1%} < {minimum_ticker_coverage:.1%}. "
            "KRX_ID/KRX_PW를 로컬에 설정해 수집하거나, 합법적으로 내려받은 KRX CSV를 06C로 가져오세요."
        )
    return summary
