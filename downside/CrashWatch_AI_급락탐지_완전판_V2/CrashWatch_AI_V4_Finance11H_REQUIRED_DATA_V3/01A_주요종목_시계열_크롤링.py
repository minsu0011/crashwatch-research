#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""대표 종목 18개의 원천 시계열을 별도로 수집한다.

기본 소스는 pykrx이며, OHLCV만 pykrx에서 실패할 때 yfinance를 보조 소스로 사용한다.
기존 01_데이터_크롤링.py를 대체하지 않고, 이탈테스트용 소형 데이터셋을 빠르게
만들기 위한 보조 스크립트다.

실행 예:
    python 01A_주요종목_시계열_크롤링.py --start 2018-01-01 --end 2026-07-21
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR
DATA_ROOT = PROJECT_DIR / "crashwatch_ai_data"
OUT_DIR = DATA_ROOT / "raw" / "sentinel_stocks"
UNIVERSE_PATH = BASE_DIR / "주요종목_18선.csv"
LOG_DIR = DATA_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "01A_주요종목_시계열_크롤링.log", encoding="utf-8"),
    ],
)
LOGGER = logging.getLogger("sentinel_crawler")

COLUMN_MAP = {
    "시가": "open",
    "고가": "high",
    "저가": "low",
    "종가": "close",
    "거래량": "volume",
    "거래대금": "trading_value",
    "등락률": "return_pct",
    "시가총액": "market_cap",
    "상장주식수": "listed_shares",
    "PER": "per",
    "PBR": "pbr",
    "EPS": "eps",
    "BPS": "bps",
    "DIV": "dividend_yield",
    "DPS": "dps",
    "외국인합계": "foreign_value",
    "기관합계": "institution_value",
    "개인": "individual_value",
    "기타법인": "other_corp_value",
    "지분율": "foreign_ownership_rate",
    "보유수량": "foreign_owned_shares",
    "공매도": "short_volume",
    "비중": "short_volume_ratio",
    "잔고수량": "short_balance_shares",
    "잔고금액": "short_balance_value",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--sleep", type=float, default=0.45)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def date8(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=20), reraise=True)
def call_with_retry(fn: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    return fn()


def normalize_frame(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["date"])
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[out.index.notna()].reset_index()
    first = out.columns[0]
    out = out.rename(columns={first: "date", **COLUMN_MAP})
    out.columns = [str(c).strip() for c in out.columns]
    if prefix:
        out = out.rename(columns={c: f"{prefix}{c}" for c in out.columns if c != "date"})
    for col in out.columns:
        if col != "date":
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.sort_values("date").drop_duplicates("date", keep="last")


def safe_collect(label: str, fn: Callable[[], pd.DataFrame], prefix: str = "") -> tuple[pd.DataFrame, str | None]:
    try:
        frame = normalize_frame(call_with_retry(fn), prefix=prefix)
        return frame, None
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("%s 수집 실패: %s", label, exc)
        return pd.DataFrame(columns=["date"]), f"{label}: {exc}"


def yfinance_ohlcv(symbol: str, start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        symbol,
        start=start,
        end=(pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["date"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.tz_localize(None)
    return raw.sort_values("date")


def collect_ticker(row: pd.Series, start: str, end: str, sleep_seconds: float) -> dict[str, object]:
    from pykrx import stock

    ticker = str(row["ticker"]).zfill(6)
    name = str(row["name"])
    start8, end8 = date8(start), date8(end)
    errors: list[str] = []

    frames: list[pd.DataFrame] = []
    ohlcv, error = safe_collect(
        "ohlcv",
        lambda: stock.get_market_ohlcv_by_date(start8, end8, ticker, adjusted=True),
    )
    if error:
        errors.append(error)
    if ohlcv.empty:
        try:
            ohlcv = yfinance_ohlcv(str(row["yahoo_symbol"]), start, end)
            ohlcv["ohlcv_source_yfinance"] = 1
        except Exception as exc:  # noqa: BLE001
            errors.append(f"yfinance_ohlcv: {exc}")
    frames.append(ohlcv)

    collectors = [
        ("market_cap", lambda: stock.get_market_cap(start8, end8, ticker), ""),
        ("fundamental", lambda: stock.get_market_fundamental(start8, end8, ticker), ""),
        (
            "trading_value",
            lambda: stock.get_market_trading_value_by_date(start8, end8, ticker),
            "flow_value_",
        ),
        (
            "trading_volume",
            lambda: stock.get_market_trading_volume_by_date(start8, end8, ticker),
            "flow_volume_",
        ),
        (
            "foreign_ownership",
            lambda: stock.get_exhaustion_rates_of_foreign_investment(start8, end8, ticker),
            "foreign_",
        ),
        ("short_volume", lambda: stock.get_shorting_volume_by_date(start8, end8, ticker), "short_volume_"),
        ("short_balance", lambda: stock.get_shorting_balance_by_date(start8, end8, ticker), "short_balance_"),
    ]
    for label, fn, prefix in collectors:
        frame, error = safe_collect(label, fn, prefix=prefix)
        frames.append(frame)
        if error:
            errors.append(error)
        time.sleep(sleep_seconds)

    nonempty = [f for f in frames if not f.empty and "date" in f.columns]
    if not nonempty:
        return {"ticker": ticker, "name": name, "status": "failed", "errors": errors}

    merged = nonempty[0]
    for frame in nonempty[1:]:
        duplicate_cols = [c for c in frame.columns if c != "date" and c in merged.columns]
        if duplicate_cols:
            frame = frame.drop(columns=duplicate_cols)
        merged = merged.merge(frame, on="date", how="outer")

    merged["ticker"] = ticker
    merged["name"] = name
    merged["market"] = str(row["market"])
    merged["sector"] = str(row["sector"])
    merged["priority"] = str(row["priority"])
    merged = merged.sort_values("date").drop_duplicates(["date", "ticker"], keep="last")

    # 최소한의 파생 열. 본 프로젝트의 정식 피처는 02_피쳐_생성_봉인.py에서 생성한다.
    if "close" in merged.columns:
        close = pd.to_numeric(merged["close"], errors="coerce")
        merged["ret_1_raw"] = close.pct_change(fill_method=None)
        merged["ret_5_raw"] = close.pct_change(5, fill_method=None)
        merged["ret_20_raw"] = close.pct_change(20, fill_method=None)
        merged["vol_20_raw"] = merged["ret_1_raw"].rolling(20, min_periods=10).std()
        merged["drawdown_60_raw"] = close / close.rolling(60, min_periods=20).max() - 1
    else:
        for col in ("ret_1_raw", "ret_5_raw", "ret_20_raw", "vol_20_raw", "drawdown_60_raw"):
            merged[col] = np.nan

    ticker_dir = OUT_DIR / f"ticker={ticker}"
    ticker_dir.mkdir(parents=True, exist_ok=True)
    path = ticker_dir / "data.parquet"
    merged.to_parquet(path, index=False)
    return {
        "ticker": ticker,
        "name": name,
        "status": "ok",
        "rows": len(merged),
        "start_date": merged["date"].min(),
        "end_date": merged["date"].max(),
        "column_count": len(merged.columns),
        "path": str(path),
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    universe = pd.read_csv(UNIVERSE_PATH, dtype={"ticker": str})
    universe["ticker"] = universe["ticker"].str.zfill(6)

    records: list[dict[str, object]] = []
    for _, row in tqdm(universe.iterrows(), total=len(universe), desc="대표 종목 수집"):
        ticker = str(row["ticker"])
        path = OUT_DIR / f"ticker={ticker}" / "data.parquet"
        if path.exists() and not args.overwrite:
            df = pd.read_parquet(path, columns=["date", "ticker"])
            records.append(
                {
                    "ticker": ticker,
                    "name": row["name"],
                    "status": "skipped",
                    "rows": len(df),
                    "start_date": df["date"].min(),
                    "end_date": df["date"].max(),
                    "path": str(path),
                }
            )
            continue
        result = collect_ticker(row, args.start, args.end, args.sleep)
        records.append(result)
        time.sleep(args.sleep)

    manifest = pd.DataFrame(records)
    manifest.to_csv(OUT_DIR / "manifest.csv", index=False, encoding="utf-8-sig")

    paths = sorted(OUT_DIR.glob("ticker=*/data.parquet"))
    if paths:
        combined = pd.concat((pd.read_parquet(p) for p in paths), ignore_index=True, sort=False)
        combined = combined.sort_values(["date", "ticker"])
        combined.to_parquet(OUT_DIR / "sentinel_raw_timeseries.parquet", index=False)
        compact_cols = [
            c
            for c in (
                "date", "ticker", "name", "market", "sector", "open", "high", "low", "close",
                "volume", "trading_value", "market_cap", "ret_1_raw", "ret_5_raw", "ret_20_raw",
                "vol_20_raw", "drawdown_60_raw",
            )
            if c in combined.columns
        ]
        combined[compact_cols].to_csv(
            OUT_DIR / "sentinel_raw_timeseries_compact.csv.gz",
            index=False,
            encoding="utf-8-sig",
            compression="gzip",
        )

    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "start": args.start,
        "end": args.end,
        "ticker_count": len(universe),
        "success_count": int(manifest["status"].isin(["ok", "skipped"]).sum()) if not manifest.empty else 0,
        "warning": "pykrx는 KRX/웹 구조 변경 및 호출 제한의 영향을 받을 수 있다.",
    }
    (OUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
