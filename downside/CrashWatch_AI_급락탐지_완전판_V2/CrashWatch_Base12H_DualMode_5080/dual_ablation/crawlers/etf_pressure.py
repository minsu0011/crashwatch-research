from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import ProjectPaths
from ..io_utils import atomic_parquet, normalize_ticker

LOGGER = logging.getLogger(__name__)


def _pykrx_etf(ticker: str, start: str, end: str) -> pd.DataFrame:
    from pykrx import stock
    raw = stock.get_market_ohlcv_by_date(pd.Timestamp(start).strftime("%Y%m%d"), pd.Timestamp(end).strftime("%Y%m%d"), ticker)
    if raw is None or raw.empty:
        return pd.DataFrame()
    raw = raw.reset_index().rename(columns={"날짜": "date", "종가": "close", "거래량": "volume", "거래대금": "trading_value"})
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    return raw[[c for c in ["date", "close", "volume", "trading_value"] if c in raw.columns]]


def _yfinance_etf(ticker: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(f"{ticker}.KS", start=start, end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), progress=False, threads=False, auto_adjust=False)
    if raw is None or raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index().rename(columns={"Date": "date", "Close": "close", "Volume": "volume"})
    raw["trading_value"] = pd.to_numeric(raw.get("close"), errors="coerce") * pd.to_numeric(raw.get("volume"), errors="coerce")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.tz_localize(None)
    return raw[["date", "close", "volume", "trading_value"]]


def collect_etf_pressure(paths: ProjectPaths, start: str, end: str, *, overwrite: bool = False) -> pd.DataFrame:
    path = paths.raw_dual / "etf_pressure.parquet"
    if path.exists() and not overwrite:
        return pd.read_parquet(path)
    cfg = pd.read_csv(paths.configs / "etf_pressure_universe.csv", dtype={"ticker": str})
    cfg["ticker"] = normalize_ticker(cfg["ticker"])
    cfg = cfg.loc[pd.to_numeric(cfg.get("enabled", 1), errors="coerce").fillna(1).astype(int).eq(1)]
    frames = []
    meta = []
    for _, row in cfg.iterrows():
        ticker, alias = row["ticker"], str(row["alias"])
        try:
            raw = _pykrx_etf(ticker, start, end)
            if raw.empty:
                raw = _yfinance_etf(ticker, start, end)
            if raw.empty:
                LOGGER.warning("ETF 데이터 없음: %s", ticker)
                continue
            raw = raw.rename(columns={
                "close": f"_{alias}_close", "volume": f"_{alias}_volume", "trading_value": f"_{alias}_value"
            })
            frames.append(raw)
            meta.append(row.to_dict())
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ETF %s 실패: %s", ticker, exc)
    if not frames:
        return pd.DataFrame()
    wide = frames[0]
    for frame in frames[1:]:
        wide = wide.merge(frame, on="date", how="outer")
    wide = wide.sort_values("date")
    out = wide[["date"]].copy()
    long_z, inverse_z, leveraged_z = [], [], []
    for item in meta:
        alias = str(item["alias"])
        close = pd.to_numeric(wide.get(f"_{alias}_close"), errors="coerce")
        value = pd.to_numeric(wide.get(f"_{alias}_value"), errors="coerce")
        out[f"u_etf_{alias}_ret_1"] = close.pct_change(fill_method=None)
        out[f"u_etf_{alias}_value_z20"] = (value - value.rolling(20, min_periods=5).mean()) / value.rolling(20, min_periods=5).std()
        role = str(item.get("role", "long"))
        if role == "inverse":
            inverse_z.append(out[f"u_etf_{alias}_value_z20"])
        elif role == "leveraged":
            leveraged_z.append(out[f"u_etf_{alias}_value_z20"])
        else:
            long_z.append(out[f"u_etf_{alias}_value_z20"])
    def mean_series(items: list[pd.Series]) -> pd.Series:
        return pd.concat(items, axis=1).mean(axis=1) if items else pd.Series(np.nan, index=out.index)
    out["u_etf_inverse_pressure"] = mean_series(inverse_z) - mean_series(long_z)
    out["u_etf_leverage_pressure"] = mean_series(leveraged_z) - mean_series(long_z)
    out["u_etf_risk_off_pressure"] = mean_series(inverse_z + leveraged_z)
    atomic_parquet(out, path)
    return out
