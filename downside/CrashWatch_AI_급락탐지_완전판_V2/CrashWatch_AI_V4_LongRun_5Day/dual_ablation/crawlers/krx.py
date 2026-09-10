from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from ..config import ProjectPaths
from ..io_utils import atomic_csv, atomic_parquet, merge_newer, normalize_date

LOGGER = logging.getLogger(__name__)

COLUMN_MAP = {
    "시가": "open", "고가": "high", "저가": "low", "종가": "close",
    "거래량": "volume", "거래대금": "trading_value", "등락률": "return_pct",
    "시가총액": "market_cap", "상장주식수": "listed_shares",
    "PER": "per", "PBR": "pbr", "EPS": "eps", "BPS": "bps", "DIV": "dividend_yield", "DPS": "dps",
    "외국인합계": "foreign", "외국인": "foreign", "기관합계": "institution", "기관": "institution", "개인": "individual", "기타법인": "other_corp",
    "금융투자": "financial_investment", "보험": "insurance", "투신": "investment_trust", "사모": "private_fund",
    "은행": "bank", "기타금융": "other_finance", "연기금": "pension", "기타외국인": "other_foreign", "합계": "total",
    "보유수량": "foreign_owned_shares", "지분율": "foreign_ownership_rate", "한도수량": "foreign_limit_shares",
    "공매도": "short_volume", "비중": "short_ratio", "잔고수량": "short_balance_shares", "잔고금액": "short_balance_value",
}


def _date8(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
def _call(fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    return fn()


def _normalize(df: pd.DataFrame | None, prefix: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].reset_index()
    out = out.rename(columns={out.columns[0]: "date", **COLUMN_MAP})
    out.columns = [str(c).strip() for c in out.columns]
    if prefix:
        out = out.rename(columns={c: f"{prefix}{c}" for c in out.columns if c != "date"})
    for col in out.columns:
        if col != "date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return normalize_date(out).drop_duplicates("date", keep="last")


def _safe(label: str, fn: Callable[[], pd.DataFrame], prefix: str = "") -> tuple[pd.DataFrame, str | None]:
    try:
        return _normalize(_call(fn), prefix), None
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("KRX %s 실패: %s", label, exc)
        return pd.DataFrame(columns=["date"]), f"{label}: {exc}"


def _merge_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    valid = [x for x in frames if not x.empty and "date" in x.columns]
    if not valid:
        return pd.DataFrame()
    out = valid[0]
    for frame in valid[1:]:
        duplicate = [c for c in frame.columns if c != "date" and c in out.columns]
        out = out.merge(frame.drop(columns=duplicate), on="date", how="outer")
    return out.sort_values("date").drop_duplicates("date", keep="last")


def collect_one_ticker(row: pd.Series, start: str, end: str, sleep_seconds: float = 0.35) -> tuple[pd.DataFrame, list[str]]:
    from pykrx import stock

    ticker = str(row["ticker"]).zfill(6)
    start8, end8 = _date8(start), _date8(end)
    errors: list[str] = []
    jobs: list[tuple[str, Callable[[], pd.DataFrame], str]] = [
        ("ohlcv", lambda: stock.get_market_ohlcv_by_date(start8, end8, ticker, adjusted=True), ""),
        ("market_cap", lambda: stock.get_market_cap_by_date(start8, end8, ticker), ""),
        ("fundamental", lambda: stock.get_market_fundamental_by_date(start8, end8, ticker), ""),
        ("trading_value", lambda: stock.get_market_trading_value_by_date(start8, end8, ticker), "flow_value_"),
        ("trading_volume", lambda: stock.get_market_trading_volume_by_date(start8, end8, ticker), "flow_volume_"),
        ("foreign", lambda: stock.get_exhaustion_rates_of_foreign_investment(start8, end8, ticker), "foreign_"),
        ("short_volume", lambda: stock.get_shorting_volume_by_date(start8, end8, ticker), "short_volume_"),
        ("short_balance", lambda: stock.get_shorting_balance_by_date(start8, end8, ticker), "short_balance_"),
    ]
    frames: list[pd.DataFrame] = []
    for label, fn, prefix in jobs:
        frame, error = _safe(label, fn, prefix)
        frames.append(frame)
        if error:
            errors.append(error)
        time.sleep(sleep_seconds)
    out = _merge_frames(frames)
    if out.empty:
        return out, errors
    out["ticker"] = ticker
    out["name"] = row.get("name", "")
    out["bucket"] = row.get("bucket", "")
    out["market"] = row.get("market", "")
    out["role"] = row.get("role", "")
    return out.sort_values(["date", "ticker"]), errors


def collect_basket_tickers(paths: ProjectPaths, baskets: pd.DataFrame, start: str, end: str, *, overwrite: bool = False, sleep_seconds: float = 0.35) -> pd.DataFrame:
    out_root = paths.raw_dual / "krx_tickers"
    records: list[dict[str, Any]] = []
    for _, row in baskets.iterrows():
        ticker = str(row["ticker"]).zfill(6)
        path = out_root / f"ticker={ticker}" / "data.parquet"
        if path.exists() and not overwrite:
            frame = pd.read_parquet(path)
            records.append({"ticker": ticker, "status": "cached", "rows": len(frame), "path": str(path)})
            continue
        frame, errors = collect_one_ticker(row, start, end, sleep_seconds)
        if frame.empty:
            records.append({"ticker": ticker, "status": "failed", "rows": 0, "errors": " | ".join(errors)})
            continue
        if path.exists() and not overwrite:
            frame = merge_newer(pd.read_parquet(path), frame, ["date", "ticker"])
        atomic_parquet(frame, path)
        records.append({
            "ticker": ticker, "name": row.get("name", ""), "bucket": row.get("bucket", ""),
            "status": "ok", "rows": len(frame), "start": frame["date"].min(), "end": frame["date"].max(),
            "errors": " | ".join(errors), "path": str(path),
        })
    manifest = pd.DataFrame(records)
    atomic_csv(manifest, out_root / "manifest.csv")
    frames = [pd.read_parquet(p) for p in sorted(out_root.glob("ticker=*/data.parquet"))]
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not combined.empty:
        combined = combined.sort_values(["date", "ticker"])
        atomic_parquet(combined, paths.raw_dual / "krx_ticker_timeseries.parquet")
    return manifest


def collect_indices(paths: ProjectPaths, start: str, end: str, overwrite: bool = False) -> pd.DataFrame:
    from pykrx import stock

    index_codes = {
        "kospi": "1001", "kospi200": "1028", "kosdaq": "2001", "kosdaq150": "2203",
    }
    frames: list[pd.DataFrame] = []
    for alias, code in index_codes.items():
        frame, _ = _safe(alias, lambda code=code: stock.get_index_ohlcv_by_date(_date8(start), _date8(end), code), f"{alias}_")
        if not frame.empty:
            frames.append(frame)
        time.sleep(0.3)
    out = _merge_frames(frames)
    path = paths.raw_dual / "krx_indices.parquet"
    if path.exists() and not overwrite and not out.empty:
        out = merge_newer(pd.read_parquet(path), out, ["date"])
    if not out.empty:
        atomic_parquet(out, path)
    return out



def collect_market_aggregate_flows(paths: ProjectPaths, start: str, end: str, overwrite: bool = False) -> pd.DataFrame:
    """KOSPI/KOSDAQ 투자자별 순매수와 투자자별 공매도 거래대금을 일별 수집한다."""
    from pykrx import stock

    frames: list[pd.DataFrame] = []
    for market in ("KOSPI", "KOSDAQ"):
        flow, _ = _safe(
            f"market_flow_{market}",
            lambda market=market: stock.get_market_trading_value_by_date(
                _date8(start), _date8(end), market, detail=True
            ),
            f"flow_{market.lower()}_",
        )
        short_value, _ = _safe(
            f"market_short_{market}",
            lambda market=market: stock.get_shorting_investor_value_by_date(
                _date8(start), _date8(end), market
            ),
            f"short_{market.lower()}_",
        )
        if not flow.empty:
            frames.append(flow)
        if not short_value.empty:
            frames.append(short_value)
        time.sleep(0.4)
    out = _merge_frames(frames)
    path = paths.raw_dual / "krx_market_flows.parquet"
    if path.exists() and not overwrite and not out.empty:
        out = merge_newer(pd.read_parquet(path), out, ["date"])
    if not out.empty:
        atomic_parquet(out, path)
    return out

def collect_market_cross_section(paths: ProjectPaths, start: str, end: str, *, markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"), overwrite: bool = False, sleep_seconds: float = 0.25) -> pd.DataFrame:
    """일별 전 종목 OHLCV/시가총액 스냅샷을 저장한다. 호출량이 많아 명시적으로 실행하는 것을 권장한다."""
    from pykrx import stock

    index = stock.get_index_ohlcv_by_date(_date8(start), _date8(end), "1001")
    dates = pd.DatetimeIndex(pd.to_datetime(index.index)).sort_values()
    out_root = paths.raw_dual / "market_cross_section"
    records: list[dict[str, Any]] = []
    for dt in dates:
        ds = dt.strftime("%Y%m%d")
        path = out_root / f"date={dt.strftime('%Y-%m-%d')}" / "data.parquet"
        if path.exists() and not overwrite:
            records.append({"date": dt, "status": "cached", "path": str(path)})
            continue
        day_frames: list[pd.DataFrame] = []
        for market in markets:
            try:
                ohlcv = _call(lambda market=market: stock.get_market_ohlcv_by_ticker(ds, market=market))
                cap = _call(lambda market=market: stock.get_market_cap_by_ticker(ds, market=market))
                ohlcv = ohlcv.rename(columns=COLUMN_MAP).reset_index().rename(columns={"티커": "ticker", ohlcv.index.name or "index": "ticker"})
                cap = cap.rename(columns=COLUMN_MAP).reset_index().rename(columns={"티커": "ticker", cap.index.name or "index": "ticker"})
                if "ticker" not in ohlcv.columns:
                    ohlcv = ohlcv.rename(columns={ohlcv.columns[0]: "ticker"})
                if "ticker" not in cap.columns:
                    cap = cap.rename(columns={cap.columns[0]: "ticker"})
                frame = ohlcv.merge(cap, on="ticker", how="left", suffixes=("", "_cap"))
                frame["date"] = dt
                frame["market"] = market
                day_frames.append(frame)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("시장 횡단면 %s %s 실패: %s", ds, market, exc)
            time.sleep(sleep_seconds)
        if day_frames:
            day = pd.concat(day_frames, ignore_index=True, sort=False)
            day["ticker"] = day["ticker"].astype(str).str.zfill(6)
            atomic_parquet(day, path)
            records.append({"date": dt, "status": "ok", "rows": len(day), "path": str(path)})
        else:
            records.append({"date": dt, "status": "failed", "rows": 0})
    manifest = pd.DataFrame(records)
    atomic_csv(manifest, out_root / "manifest.csv")
    return manifest
