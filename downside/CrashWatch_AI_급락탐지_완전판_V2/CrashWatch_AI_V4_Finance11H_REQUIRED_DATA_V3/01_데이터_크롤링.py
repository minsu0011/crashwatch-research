#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
01_데이터_크롤링.py
==================
한국 주식 급락 사전탐지 AI용 원천 데이터를 원터치로 수집한다.

설치:
    pip install pandas numpy pyarrow requests beautifulsoup4 lxml tqdm tenacity \
        python-dotenv pykrx yfinance duckdb

선택 API 키(.env 또는 시스템 환경변수):
    OPENDART_API_KEY=...
    FRED_API_KEY=...
    ECOS_API_KEY=...
    KRX_ID=...        # pykrx 최신 KRX 로그인 정책에서 필요할 수 있음
    KRX_PW=...

실행:
    python 01_데이터_크롤링.py

출력:
    crashwatch_ai_data/raw/krx/stocks/ticker=XXXXXX/data.parquet
    crashwatch_ai_data/raw/krx/indices/*.parquet
    crashwatch_ai_data/raw/cross_assets.parquet
    crashwatch_ai_data/raw/macro_fred.parquet
    crashwatch_ai_data/raw/macro_ecos.parquet
    crashwatch_ai_data/raw/policy_events.parquet
    crashwatch_ai_data/raw/dart/*.parquet
    crashwatch_ai_data/meta/*.parquet

주의:
- KRX/pykrx는 웹 구조 변경이나 호출 제한의 영향을 받을 수 있다.
- 이미 정상 저장된 종목은 건너뛰므로 중단 후 재실행 가능하다.
- 과도한 병렬 호출은 차단 위험이 있어 KRX는 의도적으로 순차 수집한다.
"""
from __future__ import annotations

import io
import json
import logging
import os
import random
import re
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tqdm import tqdm
import yfinance as yf

from 공통_도구 import 결과압축_생성, 수집_품질_보고서_생성


# 사용자 설정

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
from pykrx import stock  # .env의 KRX_ID/KRX_PW를 먼저 읽은 뒤 import
DATA_ROOT = BASE_DIR / "crashwatch_ai_data"
START_DATE = "2018-01-01"
END_DATE = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

# 월별 과거 시점에서 각 시장 시총 상위 N개의 합집합을 사용한다.
# 350 + 350이면 보통 700~1,100개 고유 종목이 남아 5만 행을 훨씬 넘는다.
TOP_N_PER_MARKET_PER_MONTH = 350
MAX_UNIVERSE = 1_200
MARKETS = ("KOSPI", "KOSDAQ")
REQUEST_SLEEP_SECONDS = 0.45
MAX_RETRIES = 6
OVERWRITE = False
DOWNLOAD_RISK_DOCUMENTS = True
MAX_RISK_DOCUMENTS = 6_000

# 반드시 포함할 주요 국내 종목
FORCED_TICKERS: dict[str, str] = {
    "005930": "삼성전자",
    "000660": "SK하이닉스",
    "005380": "현대차",
    "000270": "기아",
    "373220": "LG에너지솔루션",
    "207940": "삼성바이오로직스",
    "068270": "셀트리온",
    "035420": "NAVER",
    "035720": "카카오",
    "005490": "POSCO홀딩스",
    "012330": "현대모비스",
    "006400": "삼성SDI",
    "051910": "LG화학",
    "028260": "삼성물산",
    "066570": "LG전자",
    "105560": "KB금융",
    "055550": "신한지주",
    "086790": "하나금융지주",
    "032830": "삼성생명",
    "096770": "SK이노베이션",
    "017670": "SK텔레콤",
    "030200": "KT",
    "034730": "SK",
    "009150": "삼성전기",
    "042700": "한미반도체",
}

# 한국시장에 전염될 수 있는 해외 증시·코인·원자재·부동산 프록시
# yfinance 데이터는 연구/개인 분석 용도로 사용하고 재배포 조건은 별도 확인할 것.
CROSS_ASSETS: dict[str, str] = {
    "sp500": "^GSPC",
    "nasdaq": "^IXIC",
    "dow": "^DJI",
    "russell2000": "^RUT",
    "semiconductor_sox": "^SOX",
    "nikkei225": "^N225",
    "hangseng": "^HSI",
    "shanghai": "000001.SS",
    "shenzhen": "399001.SZ",
    "eurostoxx50": "^STOXX50E",
    "dax": "^GDAXI",
    "ftse100": "^FTSE",
    "india_nifty50": "^NSEI",
    "btc": "BTC-USD",
    "eth": "ETH-USD",
    "gold": "GC=F",
    "wti": "CL=F",
    "copper": "HG=F",
    "dollar_index": "DX-Y.NYB",
    "us_reit_vnq": "VNQ",
    "us_realestate_xlre": "XLRE",
    "us_realestate_iyr": "IYR",
    "us_homebuilders_xhb": "XHB",
    "us_homebuilders_itb": "ITB",
}

# FRED 공식 계열
FRED_SERIES: dict[str, str] = {
    "VIXCLS": "vix",
    "DGS10": "us_10y",
    "DGS2": "us_2y",
    "DFF": "fed_effective_rate",
    "DFEDTARU": "fed_target_upper",
    "DFEDTARL": "fed_target_lower",
    "BAMLH0A0HYM2": "us_high_yield_spread",
    "DEXKOUS": "usdkrw",
    "DTWEXBGS": "dollar_index_broad",
    "DCOILWTICO": "wti_spot",
    "SP500": "sp500_fred",
    "NASDAQCOM": "nasdaq_fred",
}

# KRX 업종/테마 지수 이름에서 자동 탐색할 키워드
KRX_INDEX_KEYWORDS = (
    "건설업",
    "부동산",
    "리츠",
    "은행",
    "증권",
    "보험",
    "운수장비",
    "자동차",
    "전기전자",
    "반도체",
    "화학",
    "철강",
)

DART_RISK_KEYWORDS = (
    "유상증자",
    "감자",
    "전환사채",
    "신주인수권부사채",
    "교환사채",
    "최대주주 변경",
    "횡령",
    "배임",
    "부도",
    "회생절차",
    "영업정지",
    "상장폐지",
    "관리종목",
    "감사의견",
    "계속기업",
    "소송",
    "공급계약 해지",
    "불성실공시",
    "담보제공",
    "자기주식 처분",
)


# 뉴스 수집 설정
NEWS_MAX_PAGES_PER_QUERY = int(os.getenv("NEWS_MAX_PAGES_PER_QUERY", "3"))
NEWS_REQUEST_SLEEP_SECONDS = float(os.getenv("NEWS_REQUEST_SLEEP_SECONDS", "0.15"))
NEWS_RECENT_DAYS_GDELT = int(os.getenv("NEWS_RECENT_DAYS_GDELT", "365"))

NEWS_NEGATIVE_KEYWORDS = (
    "급락", "폭락", "하락", "부진", "적자", "손실", "위기", "충격", "우려", "경고",
    "부도", "파산", "회생", "횡령", "배임", "소송", "해지", "감자", "유상증자",
    "상장폐지", "관리종목", "거래정지", "리콜", "제재", "조사", "압수수색",
)
NEWS_RISK_KEYWORDS = (
    "반대매매", "신용잔고", "마진콜", "공매도", "유동성", "차입금", "부채",
    "환율 급등", "금리 인상", "부동산 PF", "채무불이행", "디폴트", "감사의견",
)

DEFAULT_MARKET_NEWS_QUERIES = (
    ("market_crash", "코스피 급락 OR 증시 폭락", "South Korea stock market crash"),
    ("credit_stress", "반대매매 OR 신용잔고 OR 유동성 위기", "South Korea credit market stress"),
    ("realestate_pf", "부동산 PF OR 건설사 부도", "South Korea real estate crisis"),
    ("semiconductor", "반도체 업황 OR 메모리 가격", "semiconductor industry downturn"),
    ("rates_fx", "금리 인상 OR 환율 급등", "Federal Reserve rates dollar surge"),
    ("crypto_risk", "비트코인 급락 OR 가상자산 폭락", "Bitcoin cryptocurrency crash"),
)


# 공통 유틸리티

load_dotenv(BASE_DIR / ".env")
DATA_ROOT.mkdir(parents=True, exist_ok=True)
RAW_DIR = DATA_ROOT / "raw"
META_DIR = DATA_ROOT / "meta"
LOG_DIR = DATA_ROOT / "logs"
for d in (RAW_DIR, META_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "01_데이터_크롤링.log", encoding="utf-8"),
    ],
)
LOGGER = logging.getLogger("collector")
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/150 Safari/537.36"
        )
    }
)


def date8(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def safe_name(text: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣_]+", "_", str(text)).strip("_").lower()


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def merge_parquet(
    df: pd.DataFrame,
    path: Path,
    keys: Sequence[str],
    sort_cols: Sequence[str] | None = None,
) -> None:
    frames = []
    if path.exists():
        try:
            frames.append(pd.read_parquet(path))
        except Exception:
            LOGGER.warning("기존 parquet 읽기 실패, 새로 작성: %s", path)
    frames.append(df)
    out = pd.concat(frames, ignore_index=True, sort=False)
    existing_keys = [c for c in keys if c in out.columns]
    if existing_keys:
        out = out.drop_duplicates(existing_keys, keep="last")
    if sort_cols:
        cols = [c for c in sort_cols if c in out.columns]
        if cols:
            out = out.sort_values(cols, kind="mergesort")
    atomic_parquet(out.reset_index(drop=True), path)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def retry_call(
    func: Callable[..., Any],
    *args: Any,
    sleep_seconds: float = REQUEST_SLEEP_SECONDS,
    **kwargs: Any,
) -> Any:
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        time.sleep(sleep_seconds)
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt >= MAX_RETRIES:
                break
            delay = min(60.0, 1.7**attempt) + random.random()
            LOGGER.warning("호출 실패 %d/%d: %s, %.1f초 후 재시도", attempt, MAX_RETRIES, exc, delay)
            time.sleep(delay)
    assert last is not None
    raise last


def http_get_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(url, params=params, timeout=(15, 120))
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(min(30.0, 1.7**attempt) + random.random())
    assert last is not None
    raise last


def http_get_bytes(url: str, params: dict[str, Any] | None = None) -> bytes:
    last: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(url, params=params, timeout=(15, 180))
            response.raise_for_status()
            return response.content
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt >= MAX_RETRIES:
                break
            time.sleep(min(30.0, 1.7**attempt) + random.random())
    assert last is not None
    raise last


def normalize_indexed_frame(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~out.index.isna()].sort_index()
    out.index.name = "date"
    out = out.reset_index()
    mapping = {
        "시가": "open",
        "고가": "high",
        "저가": "low",
        "종가": "close",
        "거래량": "volume",
        "거래대금": "trading_value",
        "등락률": "change_pct",
        "시가총액": "market_cap",
        "상장주식수": "listed_shares",
        "외국인보유주식수": "foreign_held_shares",
        "보유수량": "foreign_held_shares",
        "지분율": "foreign_ownership_pct",
        "한도수량": "foreign_limit_shares",
        "한도소진율": "foreign_limit_exhaustion_pct",
        "BPS": "bps",
        "PER": "per",
        "PBR": "pbr",
        "EPS": "eps",
        "DIV": "dividend_yield",
        "DPS": "dps",
        "기관합계": "institution_net",
        "외국인합계": "foreign_net",
        "외국인": "foreign_net",
        "개인": "individual_net",
        "기타법인": "other_corp_net",
        "전체": "total_net",
        "공매도": "short_volume",
        "매수": "total_buy_volume",
        "비중": "ratio_pct",
        "공매도잔고": "short_balance_shares",
        "공매도금액": "short_balance_value",
    }
    out = out.rename(columns=mapping)
    if prefix:
        out = out.rename(columns={c: f"{prefix}_{c}" for c in out.columns if c != "date"})
    for col in out.columns:
        if col != "date":
            converted = pd.to_numeric(out[col], errors="coerce")
            if converted.notna().any():
                out[col] = converted
    return out



# KRX

def krx_business_days() -> pd.DatetimeIndex:
    path = META_DIR / "krx_business_days.parquet"
    if path.exists() and not OVERWRITE:
        return pd.DatetimeIndex(pd.read_parquet(path)["date"])
    try:
        df = retry_call(stock.get_index_ohlcv_by_date, date8(START_DATE), date8(END_DATE), "1001")
        days = pd.DatetimeIndex(pd.to_datetime(df.index)).sort_values().unique()
    except Exception:
        LOGGER.exception("KRX 영업일 조회 실패. 평일 달력으로 대체")
        days = pd.bdate_range(START_DATE, END_DATE)
    atomic_parquet(pd.DataFrame({"date": days}), path)
    return days


def month_end_trading_days(days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    s = pd.Series(days, index=days)
    out = s.resample("ME").last().dropna()
    if len(days) and (out.empty or out.iloc[-1] != days[-1]):
        out.loc[days[-1]] = days[-1]
    return pd.DatetimeIndex(out.values)


def get_market_cap_by_ticker_compat(ds: str, market: str) -> pd.DataFrame:
    candidates = [
        getattr(stock, "get_market_cap_by_ticker", None),
        getattr(stock, "get_market_cap", None),
    ]
    for fn in candidates:
        if not callable(fn):
            continue
        try:
            return retry_call(fn, ds, market=market)
        except TypeError:
            try:
                return retry_call(fn, ds, market)
            except Exception:
                continue
    raise RuntimeError("pykrx 시가총액 횡단면 함수를 찾지 못했습니다.")


def build_historical_universe() -> pd.DataFrame:
    path = META_DIR / "selected_universe.parquet"
    if path.exists() and not OVERWRITE:
        return pd.read_parquet(path)

    days = krx_business_days()
    snapshots = month_end_trading_days(days)
    rows: list[dict[str, Any]] = []
    for dt in tqdm(snapshots, desc="월별 시총 유니버스"):
        ds = date8(dt)
        for market in MARKETS:
            try:
                cap = get_market_cap_by_ticker_compat(ds, market)
                if cap is None or cap.empty:
                    continue
                cap = cap.rename(columns={"시가총액": "market_cap"})
                cap.index = cap.index.astype(str).str.zfill(6)
                cap.index.name = "ticker"
                cap = cap.reset_index()
                cap["market_cap"] = pd.to_numeric(cap["market_cap"], errors="coerce")
                cap = cap.nlargest(TOP_N_PER_MARKET_PER_MONTH, "market_cap")
                for item in cap.itertuples(index=False):
                    rows.append(
                        {
                            "snapshot_date": pd.Timestamp(dt),
                            "ticker": str(item.ticker).zfill(6),
                            "market": market,
                            "market_cap": float(item.market_cap),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("유니버스 스냅샷 실패 %s %s: %s", ds, market, exc)

    hist = pd.DataFrame(rows)
    if hist.empty:
        raise RuntimeError("KRX 유니버스를 만들지 못했습니다.")
    selected = (
        hist.groupby("ticker", as_index=False)
        .agg(
            market=("market", "last"),
            max_market_cap=("market_cap", "max"),
            first_snapshot=("snapshot_date", "min"),
            last_snapshot=("snapshot_date", "max"),
        )
        .sort_values("max_market_cap", ascending=False)
        .head(MAX_UNIVERSE)
    )

    # 필수 종목 강제 포함
    forced_rows = []
    latest = days[-1]
    for ticker, name in FORCED_TICKERS.items():
        market = "KOSPI"
        try:
            if ticker in retry_call(stock.get_market_ticker_list, date8(latest), market="KOSDAQ"):
                market = "KOSDAQ"
        except Exception:
            pass
        forced_rows.append(
            {
                "ticker": ticker,
                "market": market,
                "max_market_cap": np.nan,
                "first_snapshot": pd.NaT,
                "last_snapshot": latest,
                "name": name,
                "forced": True,
            }
        )
    selected["forced"] = False
    selected = pd.concat([selected, pd.DataFrame(forced_rows)], ignore_index=True, sort=False)
    selected = selected.drop_duplicates("ticker", keep="last")

    names = {}
    for ticker in tqdm(selected["ticker"], desc="종목명 조회"):
        if ticker in FORCED_TICKERS:
            names[ticker] = FORCED_TICKERS[ticker]
            continue
        try:
            names[ticker] = retry_call(stock.get_market_ticker_name, ticker)
        except Exception:
            names[ticker] = ""
    selected["name"] = selected["ticker"].map(names).fillna(selected.get("name", ""))
    atomic_parquet(selected.reset_index(drop=True), path)
    atomic_parquet(hist, META_DIR / "historical_universe_snapshots.parquet")
    return selected.reset_index(drop=True)


def collect_one_krx_ticker(row: pd.Series) -> dict[str, Any]:
    ticker = str(row["ticker"]).zfill(6)
    out_path = RAW_DIR / "krx" / "stocks" / f"ticker={ticker}" / "data.parquet"
    if out_path.exists() and not OVERWRITE:
        return {"ticker": ticker, "status": "skipped", "path": str(out_path)}

    start8, end8 = date8(START_DATE), date8(END_DATE)
    frames: list[pd.DataFrame] = []
    errors: dict[str, str] = {}

    jobs: list[tuple[str, Callable[[], pd.DataFrame], str]] = [
        (
            "ohlcv",
            lambda: stock.get_market_ohlcv_by_date(start8, end8, ticker, adjusted=True),
            "",
        ),
        ("market_cap", lambda: stock.get_market_cap(start8, end8, ticker), ""),
        (
            "fundamental",
            lambda: stock.get_market_fundamental(start8, end8, ticker),
            "",
        ),
        (
            "investor_value",
            lambda: stock.get_market_trading_value_by_date(start8, end8, ticker),
            "value",
        ),
        (
            "investor_volume",
            lambda: stock.get_market_trading_volume_by_date(start8, end8, ticker),
            "shares",
        ),
        (
            "foreign_ownership",
            lambda: stock.get_exhaustion_rates_of_foreign_investment(start8, end8, ticker),
            "",
        ),
        (
            "short_volume",
            lambda: stock.get_shorting_volume_by_date(start8, end8, ticker),
            "short",
        ),
        (
            "short_balance",
            lambda: stock.get_shorting_balance_by_date(start8, end8, ticker),
            "shortbal",
        ),
    ]

    for name, fn, prefix in jobs:
        try:
            raw = retry_call(fn)
            part = normalize_indexed_frame(raw, prefix)
            if not part.empty:
                frames.append(part)
        except Exception as exc:  # noqa: BLE001
            errors[name] = repr(exc)
            LOGGER.warning("%s %s 실패: %s", ticker, name, exc)

    if not frames:
        return {"ticker": ticker, "status": "failed", "errors": json.dumps(errors, ensure_ascii=False)}

    merged = frames[0]
    for part in frames[1:]:
        overlap = sorted((set(merged.columns) & set(part.columns)) - {"date"})
        rename = {c: f"{c}__new" for c in overlap}
        part = part.rename(columns=rename)
        merged = merged.merge(part, on="date", how="outer")
        for col in overlap:
            new_col = f"{col}__new"
            merged[col] = merged[col].combine_first(merged[new_col])
            merged = merged.drop(columns=[new_col])

    # 공매도 거래비중과 잔고비중 이름 구분
    if "short_ratio_pct" in merged:
        merged = merged.rename(columns={"short_ratio_pct": "short_volume_ratio_pct"})
    if "shortbal_ratio_pct" in merged:
        merged = merged.rename(columns={"shortbal_ratio_pct": "short_balance_ratio_pct"})

    merged["ticker"] = ticker
    merged["name"] = str(row.get("name", ""))
    merged["market"] = str(row.get("market", ""))
    merged = merged.sort_values("date").drop_duplicates(["date", "ticker"], keep="last")
    atomic_parquet(merged.reset_index(drop=True), out_path)
    return {
        "ticker": ticker,
        "status": "ok",
        "rows": len(merged),
        "errors": json.dumps(errors, ensure_ascii=False),
        "path": str(out_path),
    }


def collect_krx_stocks(universe: pd.DataFrame) -> None:
    manifest_path = META_DIR / "krx_collection_manifest.parquet"
    old = pd.read_parquet(manifest_path) if manifest_path.exists() else pd.DataFrame()
    done = set(old.loc[old.get("status", pd.Series(dtype=str)).isin(["ok", "skipped"]), "ticker"]) if not old.empty else set()
    buffer: list[dict[str, Any]] = []
    for _, row in tqdm(universe.iterrows(), total=len(universe), desc="KRX 종목별 데이터"):
        ticker = str(row["ticker"]).zfill(6)
        file_path = RAW_DIR / "krx" / "stocks" / f"ticker={ticker}" / "data.parquet"
        if ticker in done and file_path.exists() and not OVERWRITE:
            continue
        try:
            result = collect_one_krx_ticker(row)
        except Exception:  # noqa: BLE001
            result = {
                "ticker": ticker,
                "status": "failed",
                "errors": traceback.format_exc(),
            }
        result["updated_at"] = pd.Timestamp.now(tz="Asia/Seoul")
        buffer.append(result)
        if len(buffer) >= 10:
            merge_parquet(pd.DataFrame(buffer), manifest_path, ["ticker"], ["ticker"])
            buffer.clear()
    if buffer:
        merge_parquet(pd.DataFrame(buffer), manifest_path, ["ticker"], ["ticker"])


def collect_krx_indices() -> None:
    out_dir = RAW_DIR / "krx" / "indices"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_indices = {"kospi": "1001", "kosdaq": "2001"}
    for alias, ticker in base_indices.items():
        try:
            df = retry_call(stock.get_index_ohlcv_by_date, date8(START_DATE), date8(END_DATE), ticker)
            out = normalize_indexed_frame(df)
            out["alias"] = alias
            out["index_ticker"] = ticker
            atomic_parquet(out, out_dir / f"{alias}.parquet")
        except Exception:
            LOGGER.exception("기본 지수 수집 실패: %s", alias)

    # 업종/부동산 관련 지수 자동 탐색
    latest = date8(krx_business_days()[-1])
    found: list[dict[str, str]] = []
    for market in ("KOSPI", "KOSDAQ", "KRX", "테마"):
        try:
            tickers = retry_call(stock.get_index_ticker_list, latest, market=market)
        except Exception:
            continue
        for ticker in tickers:
            try:
                name = retry_call(stock.get_index_ticker_name, ticker)
            except Exception:
                continue
            if any(keyword in str(name) for keyword in KRX_INDEX_KEYWORDS):
                found.append({"ticker": str(ticker), "name": str(name), "market": market})

    found_df = pd.DataFrame(found).drop_duplicates("ticker") if found else pd.DataFrame()
    if not found_df.empty:
        atomic_parquet(found_df, META_DIR / "krx_selected_indices.parquet")
        for row in tqdm(found_df.itertuples(index=False), total=len(found_df), desc="KRX 업종지수"):
            alias = safe_name(row.name) or f"index_{row.ticker}"
            path = out_dir / f"{alias}_{row.ticker}.parquet"
            if path.exists() and not OVERWRITE:
                continue
            try:
                df = retry_call(
                    stock.get_index_ohlcv_by_date,
                    date8(START_DATE),
                    date8(END_DATE),
                    row.ticker,
                )
                out = normalize_indexed_frame(df)
                out["alias"] = alias
                out["index_name"] = row.name
                out["index_ticker"] = row.ticker
                atomic_parquet(out, path)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("업종지수 실패 %s %s: %s", row.ticker, row.name, exc)



# 해외 증시·코인·원자재·부동산 프록시

def collect_cross_assets() -> None:
    path = RAW_DIR / "cross_assets.parquet"
    if path.exists() and not OVERWRITE:
        LOGGER.info("cross_assets 존재, 건너뜀")
        return
    reverse = {symbol: alias for alias, symbol in CROSS_ASSETS.items()}
    symbols = list(reverse)
    LOGGER.info("yfinance 교차자산 %d개 다운로드", len(symbols))
    data = yf.download(
        tickers=symbols,
        start=START_DATE,
        end=(pd.Timestamp(END_DATE) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=True,
        repair=True,
        actions=False,
        threads=True,
        group_by="column",
        progress=True,
        timeout=30,
    )
    rows: list[pd.DataFrame] = []
    if isinstance(data.columns, pd.MultiIndex):
        level0 = set(data.columns.get_level_values(0))
        field_first = "Close" in level0
        for symbol in symbols:
            try:
                block = data.xs(symbol, axis=1, level=1 if field_first else 0).copy()
            except Exception:
                continue
            block.index = pd.to_datetime(block.index, errors="coerce").tz_localize(None)
            block.index.name = "date"
            block = block.reset_index()
            block.columns = [safe_name(c) for c in block.columns]
            block["symbol"] = symbol
            block["alias"] = reverse[symbol]
            rows.append(block)
    else:
        block = data.copy()
        block.index = pd.to_datetime(block.index, errors="coerce").tz_localize(None)
        block.index.name = "date"
        block = block.reset_index()
        block.columns = [safe_name(c) for c in block.columns]
        symbol = symbols[0]
        block["symbol"] = symbol
        block["alias"] = reverse[symbol]
        rows.append(block)
    if not rows:
        raise RuntimeError("교차자산 데이터를 받지 못했습니다.")
    out = pd.concat(rows, ignore_index=True, sort=False)
    atomic_parquet(out, path)




# 뉴스: NAVER 기사 + GDELT 글로벌 뉴스량

def _clean_news_text(text: Any) -> str:
    value = BeautifulSoup(str(text or ""), "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", value)


def _news_keyword_counts(text: str) -> tuple[int, int]:
    lowered = str(text)
    negative = sum(lowered.count(k) for k in NEWS_NEGATIVE_KEYWORDS)
    risk = sum(lowered.count(k) for k in NEWS_RISK_KEYWORDS)
    return negative, risk


def load_news_queries(universe: pd.DataFrame) -> pd.DataFrame:
    config_path = BASE_DIR / "뉴스_검색어_설정.csv"
    if config_path.exists():
        try:
            q = pd.read_csv(config_path, encoding="utf-8-sig")
            if not q.empty:
                q["enabled"] = q.get("enabled", 1).fillna(1).astype(int)
                return q[q["enabled"].eq(1)].copy()
        except Exception:
            LOGGER.exception("뉴스 검색어 설정 읽기 실패")

    rows = []
    for ticker, company in FORCED_TICKERS.items():
        rows.append(
            {
                "alias": f"company_{ticker}",
                "query": f'"{company}" 주가',
                "gdelt_query": company,
                "scope": "company",
                "ticker": ticker,
                "enabled": 1,
            }
        )
    for alias, query, gdelt_query in DEFAULT_MARKET_NEWS_QUERIES:
        rows.append(
            {
                "alias": alias,
                "query": query,
                "gdelt_query": gdelt_query,
                "scope": "market",
                "ticker": "",
                "enabled": 1,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(config_path, index=False, encoding="utf-8-sig")
    return out


def collect_naver_news(queries: pd.DataFrame) -> None:
    client_id = os.getenv("NAVER_CLIENT_ID", "").strip()
    client_secret = os.getenv("NAVER_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        LOGGER.warning("NAVER_CLIENT_ID/SECRET이 없어 NAVER 뉴스 수집을 건너뜁니다.")
        return

    path = RAW_DIR / "news" / "naver_news.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    endpoint = os.getenv(
        "NAVER_NEWS_API_URL",
        "https://openapi.naver.com/v1/search/news.json",
    )
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    records = []
    for row in tqdm(queries.itertuples(index=False), total=len(queries), desc="NAVER 뉴스"):
        query = str(row.query)
        for page in range(NEWS_MAX_PAGES_PER_QUERY):
            params = {
                "query": query,
                "display": 100,
                "start": 1 + page * 100,
                "sort": "date",
            }
            try:
                time.sleep(NEWS_REQUEST_SLEEP_SECONDS)
                response = SESSION.get(endpoint, headers=headers, params=params, timeout=(10, 60))
                response.raise_for_status()
                items = response.json().get("items", []) or []
            except Exception as exc:
                LOGGER.warning("NAVER 뉴스 실패 alias=%s page=%d: %s", row.alias, page, exc)
                break
            if not items:
                break
            for item in items:
                title = _clean_news_text(item.get("title"))
                description = _clean_news_text(item.get("description"))
                combined = f"{title} {description}"
                negative_count, risk_count = _news_keyword_counts(combined)
                pub = pd.to_datetime(item.get("pubDate"), utc=True, errors="coerce")
                if pd.notna(pub):
                    pub = pub.tz_convert("Asia/Seoul")
                link = str(item.get("originallink") or item.get("link") or "")
                records.append(
                    {
                        "source": "NAVER",
                        "alias": str(row.alias),
                        "scope": str(row.scope),
                        "ticker": (
                            str(getattr(row, "ticker", "")).replace(".0", "").zfill(6)
                            if str(getattr(row, "ticker", "")).strip().lower() not in {"", "nan", "none"}
                            else ""
                        ),
                        "query": query,
                        "published_at_kst": pub,
                        "date": pub.tz_localize(None).normalize() if pd.notna(pub) else pd.NaT,
                        "title": title,
                        "description": description,
                        "link": link,
                        "source_domain": urlparse(link).netloc.lower(),
                        "negative_keyword_count": negative_count,
                        "risk_keyword_count": risk_count,
                    }
                )
            if len(items) < 100:
                break

    if records:
        out = pd.DataFrame(records)
        out = out.dropna(subset=["published_at_kst"])
        out["article_key"] = (
            out["link"].fillna("")
            + "|"
            + out["title"].fillna("")
            + "|"
            + out["published_at_kst"].astype(str)
        )
        out = out.drop_duplicates("article_key", keep="last")
        if path.exists() and not OVERWRITE:
            old = pd.read_parquet(path)
            out = pd.concat([old, out], ignore_index=True, sort=False).drop_duplicates(
                "article_key", keep="last"
            )
        atomic_parquet(out.sort_values("published_at_kst").reset_index(drop=True), path)


def _parse_gdelt_timeline(payload: dict[str, Any], alias: str, query: str) -> list[dict[str, Any]]:
    rows = []
    timeline = payload.get("timeline", [])
    if isinstance(timeline, dict):
        timeline = [timeline]
    for series in timeline if isinstance(timeline, list) else []:
        data = series.get("data", []) if isinstance(series, dict) else []
        for point in data:
            if not isinstance(point, dict):
                continue
            raw_date = point.get("date") or point.get("datetime")
            dt = pd.to_datetime(raw_date, errors="coerce")
            if pd.isna(dt) and raw_date:
                dt = pd.to_datetime(str(raw_date), format="%Y%m%dT%H%M%SZ", errors="coerce")
            value = point.get("value", point.get("count", point.get("volume")))
            norm = point.get("norm", point.get("total"))
            rows.append(
                {
                    "source": "GDELT",
                    "alias": alias,
                    "query": query,
                    "date": dt.normalize() if pd.notna(dt) else pd.NaT,
                    "news_volume": pd.to_numeric(value, errors="coerce"),
                    "news_norm": pd.to_numeric(norm, errors="coerce"),
                }
            )
    return rows


def collect_gdelt_news_timeline(queries: pd.DataFrame) -> None:
    path = RAW_DIR / "news" / "gdelt_timeline.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    endpoint = "https://api.gdeltproject.org/api/v2/doc/doc"
    end = pd.Timestamp(END_DATE).normalize()
    start = max(pd.Timestamp(START_DATE), end - pd.Timedelta(days=NEWS_RECENT_DAYS_GDELT))
    records = []
    # 호출량을 제한하기 위해 시장 공통 쿼리와 주요 10개 기업만 사용
    work = pd.concat(
        [
            queries[queries["scope"].eq("market")],
            queries[queries["scope"].eq("company")].head(10),
        ],
        ignore_index=True,
    )
    for row in tqdm(work.itertuples(index=False), total=len(work), desc="GDELT 뉴스량"):
        query = str(getattr(row, "gdelt_query", "") or getattr(row, "query", "")).strip()
        if not query:
            continue
        params = {
            "query": query,
            "mode": "timelinevolraw",
            "format": "json",
            "maxrecords": 250,
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime": end.strftime("%Y%m%d%H%M%S"),
        }
        try:
            payload = http_get_json(endpoint, params=params)
            records.extend(_parse_gdelt_timeline(payload, str(row.alias), query))
        except Exception as exc:
            LOGGER.warning("GDELT 실패 alias=%s: %s", row.alias, exc)
    if records:
        out = pd.DataFrame(records).dropna(subset=["date"])
        out = out.drop_duplicates(["alias", "date"], keep="last")
        atomic_parquet(out.sort_values(["alias", "date"]).reset_index(drop=True), path)


def collect_news(universe: pd.DataFrame) -> None:
    queries = load_news_queries(universe)
    collect_naver_news(queries)
    collect_gdelt_news_timeline(queries)



# FRED

def collect_fred() -> None:
    key = os.getenv("FRED_API_KEY", "").strip()
    if not key:
        LOGGER.warning("FRED_API_KEY가 없어 FRED 수집을 건너뜁니다.")
        return
    path = RAW_DIR / "macro_fred.parquet"
    if path.exists() and not OVERWRITE:
        LOGGER.info("macro_fred 존재, 건너뜀")
        return
    rows: list[dict[str, Any]] = []
    for series_id, alias in tqdm(FRED_SERIES.items(), desc="FRED"):
        params = {
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "observation_start": START_DATE,
            "observation_end": END_DATE,
            "sort_order": "asc",
            # vintage-date 정보가 포함되는 형식. 과거 수정치 검증에 활용 가능.
            "output_type": 4,
            "realtime_start": START_DATE,
            "realtime_end": END_DATE,
        }
        try:
            payload = http_get_json(
                "https://api.stlouisfed.org/fred/series/observations", params=params
            )
            for obs in payload.get("observations", []):
                rows.append(
                    {
                        "series_id": series_id,
                        "alias": alias,
                        "date": pd.to_datetime(obs.get("date"), errors="coerce"),
                        "value": pd.to_numeric(obs.get("value"), errors="coerce"),
                        "realtime_start": pd.to_datetime(obs.get("realtime_start"), errors="coerce"),
                        "realtime_end": pd.to_datetime(obs.get("realtime_end"), errors="coerce"),
                    }
                )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("FRED %s 실패: %s", series_id, exc)
    if rows:
        atomic_parquet(pd.DataFrame(rows).dropna(subset=["date"]), path)



# ECOS: 기준금리 + 주택가격/주택금융 자동 탐색

def ecos_service_rows(service: str, key: str, start_no: int, end_no: int, *parts: str) -> list[dict[str, Any]]:
    suffix = "/".join(str(p).strip("/") for p in parts if str(p) != "")
    url = f"https://ecos.bok.or.kr/api/{service}/{key}/json/kr/{start_no}/{end_no}"
    if suffix:
        url += "/" + suffix
    payload = http_get_json(url)
    block = payload.get(service, {})
    result = block.get("RESULT", {})
    if result and result.get("CODE") not in {"INFO-000", None}:
        raise RuntimeError(f"ECOS {service}: {result}")
    return block.get("row", []) or []


def ecos_statistic_search(
    key: str,
    stat_code: str,
    cycle: str,
    start: str,
    end: str,
    item_codes: Sequence[str],
    max_rows: int = 100_000,
) -> list[dict[str, Any]]:
    parts = [stat_code, cycle, start, end, *item_codes]
    return ecos_service_rows("StatisticSearch", key, 1, max_rows, *parts)


def collect_ecos() -> None:
    key = os.getenv("ECOS_API_KEY", "").strip()
    if not key:
        LOGGER.warning("ECOS_API_KEY가 없어 ECOS 수집을 건너뜁니다.")
        return
    path = RAW_DIR / "macro_ecos.parquet"
    if path.exists() and not OVERWRITE:
        LOGGER.info("macro_ecos 존재, 건너뜀")
        return

    rows: list[dict[str, Any]] = []

    # 한국은행 기준금리: 722Y001 / 0101000
    try:
        data = ecos_statistic_search(
            key,
            "722Y001",
            "D",
            date8(START_DATE),
            date8(END_DATE),
            ["0101000"],
        )
        for item in data:
            rows.append(
                {
                    "source": "ECOS",
                    "stat_code": "722Y001",
                    "stat_name": "한국은행 기준금리",
                    "alias": "bok_base_rate",
                    "cycle": "D",
                    "date": pd.to_datetime(item.get("TIME"), format="%Y%m%d", errors="coerce"),
                    "value": pd.to_numeric(item.get("DATA_VALUE"), errors="coerce"),
                    "item_name": item.get("ITEM_NAME1"),
                    "unit": item.get("UNIT_NAME"),
                }
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("ECOS 기준금리 실패: %s", exc)

    # 통계표 목록에서 주택가격/아파트가격/주택담보대출 관련 표를 자동 탐색
    keywords = ("주택매매가격", "아파트매매가격", "주택가격", "전세가격", "주택담보대출")
    try:
        tables = ecos_service_rows("StatisticTableList", key, 1, 20_000)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("ECOS 통계표 목록 실패: %s", exc)
        tables = []

    candidates = []
    for table in tables:
        name = str(table.get("STAT_NAME", ""))
        if table.get("SRCH_YN") == "Y" and any(k in name for k in keywords):
            candidates.append(table)
    # 너무 많은 표를 무차별 호출하지 않는다.
    candidates = candidates[:12]
    metadata_rows = []
    for table in tqdm(candidates, desc="ECOS 주택 관련 통계"):
        stat_code = str(table.get("STAT_CODE", ""))
        cycle = str(table.get("CYCLE", "M"))
        stat_name = str(table.get("STAT_NAME", stat_code))
        if not stat_code:
            continue
        try:
            items = ecos_service_rows("StatisticItemList", key, 1, 10_000, stat_code)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("ECOS 항목목록 실패 %s: %s", stat_code, exc)
            continue
        # 전국/종합/총지수 우선. 없으면 첫 검색 가능 항목.
        preferred = [
            item
            for item in items
            if any(k in str(item.get("ITEM_NAME", "")) for k in ("전국", "종합", "총지수"))
        ]
        chosen = (preferred or items)[:3]
        for item in chosen:
            item_code = str(item.get("ITEM_CODE", ""))
            if not item_code:
                continue
            try:
                if cycle in {"M", "MM"}:
                    start, end = pd.Timestamp(START_DATE).strftime("%Y%m"), pd.Timestamp(END_DATE).strftime("%Y%m")
                elif cycle in {"Q", "QY"}:
                    start = f"{pd.Timestamp(START_DATE).year}Q1"
                    end = f"{pd.Timestamp(END_DATE).year}Q4"
                elif cycle in {"A", "YY"}:
                    start, end = str(pd.Timestamp(START_DATE).year), str(pd.Timestamp(END_DATE).year)
                else:
                    start, end = date8(START_DATE), date8(END_DATE)
                data = ecos_statistic_search(key, stat_code, cycle, start, end, [item_code])
                alias = "ecos_" + safe_name(stat_name + "_" + str(item.get("ITEM_NAME", item_code)))[:100]
                for obs in data:
                    raw_time = str(obs.get("TIME", ""))
                    if cycle in {"M", "MM"}:
                        dt = pd.to_datetime(raw_time, format="%Y%m", errors="coerce")
                    elif cycle in {"Q", "QY"}:
                        try:
                            dt = pd.Period(raw_time, freq="Q").end_time.normalize()
                        except Exception:
                            dt = pd.NaT
                    elif cycle in {"A", "YY"}:
                        dt = pd.to_datetime(raw_time, format="%Y", errors="coerce")
                    else:
                        dt = pd.to_datetime(raw_time, errors="coerce")
                    rows.append(
                        {
                            "source": "ECOS",
                            "stat_code": stat_code,
                            "stat_name": stat_name,
                            "alias": alias,
                            "cycle": cycle,
                            "date": dt,
                            "value": pd.to_numeric(obs.get("DATA_VALUE"), errors="coerce"),
                            "item_name": obs.get("ITEM_NAME1") or item.get("ITEM_NAME"),
                            "unit": obs.get("UNIT_NAME"),
                        }
                    )
                metadata_rows.append({**table, **item})
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("ECOS 시계열 실패 %s %s: %s", stat_code, item_code, exc)

    if rows:
        out = pd.DataFrame(rows).dropna(subset=["date"])
        atomic_parquet(out, path)
    if metadata_rows:
        atomic_parquet(pd.DataFrame(metadata_rows), META_DIR / "ecos_selected_series.parquet")



# 연준·한국은행 정책회의 일정

def collect_fomc_dates() -> list[pd.Timestamp]:
    years = range(pd.Timestamp(START_DATE).year, pd.Timestamp(END_DATE).year + 2)
    main_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
    urls = [main_url]
    urls.extend(f"https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm" for year in years)
    dates: set[pd.Timestamp] = set()
    statement_pattern = re.compile(r"fomc(?:pressconf|statement)(\d{8})[a-z]?\.htm", re.I)
    main_html = ""
    for url in urls:
        try:
            html = http_get_bytes(url).decode("utf-8", errors="ignore")
            if url == main_url:
                main_html = html
        except Exception:
            continue
        for match in statement_pattern.findall(html):
            dt = pd.to_datetime(match, format="%Y%m%d", errors="coerce")
            if pd.notna(dt):
                dates.add(pd.Timestamp(dt).normalize())

    # 아직 statement 링크가 생성되지 않은 미래 회의도 캘린더 본문에서 읽는다.
    if main_html:
        text = BeautifulSoup(main_html, "lxml").get_text(" ", strip=True)
        month_map = {
            "January": 1, "February": 2, "March": 3, "April": 4,
            "May": 5, "June": 6, "July": 7, "August": 8,
            "September": 9, "October": 10, "November": 11, "December": 12,
        }
        for year in years:
            start_marker = re.search(rf"{year}\s+FOMC\s+Meetings", text, re.I)
            if not start_marker:
                continue
            tail = text[start_marker.end():]
            next_marker = re.search(r"20\d{2}\s+FOMC\s+Meetings", tail, re.I)
            section = tail[: next_marker.start()] if next_marker else tail
            # 회의는 보통 'January 27-28' 형식. 정책결정일은 마지막 날이다.
            matches = re.findall(
                r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?",
                section,
            )
            for month_name, first_day, last_day in matches[:12]:
                day = int(last_day or first_day)
                try:
                    dates.add(pd.Timestamp(year, month_map[month_name], day))
                except ValueError:
                    continue
    return sorted(dates)


def collect_bok_meeting_dates() -> list[pd.Timestamp]:
    years = range(pd.Timestamp(START_DATE).year, pd.Timestamp(END_DATE).year + 2)
    dates: set[pd.Timestamp] = set()
    for year in years:
        urls = [
            f"https://www.bok.or.kr/portal/singl/crncyPolicyDrcMtg/listYear.do?menuNo=200755&mtgSe=A&year={year}",
            f"https://www.bok.or.kr/portal/singl/crncyPolicyDrcMtg/listYear.do?menuNo=200755&mtgSe=A&searchYear={year}",
        ]
        for url in urls:
            try:
                text = BeautifulSoup(http_get_bytes(url), "lxml").get_text(" ", strip=True)
            except Exception:
                continue
            for y, m, d in re.findall(r"(20\d{2})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})", text):
                if int(y) != year:
                    continue
                try:
                    dates.add(pd.Timestamp(int(y), int(m), int(d)))
                except ValueError:
                    continue
    return sorted(dates)


def collect_policy_events() -> None:
    rows: list[dict[str, Any]] = []
    for dt in collect_fomc_dates():
        rows.append(
            {
                "source": "FED",
                "event_type": "FOMC_RATE_DECISION",
                "announcement_date_local": dt,
                # 미국 오후 발표는 한국 주식 종가 이후이므로 다음 KRX 거래일부터 사용
                "availability_rule": "NEXT_KRX_TRADING_DAY",
            }
        )
    for dt in collect_bok_meeting_dates():
        rows.append(
            {
                "source": "BOK",
                "event_type": "BOK_RATE_DECISION",
                "announcement_date_local": dt,
                # 일봉 종가 이후 예측 기준: 발표 당일 종가 계산 시 사용 가능
                "availability_rule": "SAME_KRX_TRADING_DAY",
            }
        )
    manual = BASE_DIR / "수동_정책일정.csv"
    if manual.exists():
        try:
            m = pd.read_csv(manual)
            rows.extend(m.to_dict("records"))
        except Exception:
            LOGGER.exception("수동 정책일정 읽기 실패")
    if rows:
        out = pd.DataFrame(rows)
        out["announcement_date_local"] = pd.to_datetime(out["announcement_date_local"], errors="coerce")
        out = out.dropna(subset=["announcement_date_local"]).drop_duplicates(
            ["source", "event_type", "announcement_date_local"], keep="last"
        )
        atomic_parquet(out, RAW_DIR / "policy_events.parquet")



# OpenDART

DART_BASE = "https://opendart.fss.or.kr/api"


def dart_request(endpoint: str, params: dict[str, Any], binary: bool = False) -> Any:
    key = os.getenv("OPENDART_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENDART_API_KEY가 없습니다.")
    params = dict(params)
    params["crtfc_key"] = key
    if binary:
        content = http_get_bytes(f"{DART_BASE}/{endpoint}", params=params)
        if content[:1] == b"{" or content[:1] == b"[":
            payload = json.loads(content.decode("utf-8"))
            raise RuntimeError(f"DART 오류: {payload}")
        return content
    payload = http_get_json(f"{DART_BASE}/{endpoint}", params=params)
    status = str(payload.get("status", "000"))
    if status not in {"000", ""}:
        if status == "013":
            return payload
        raise RuntimeError(f"DART {status}: {payload.get('message')}")
    return payload


def collect_dart_corp_codes() -> pd.DataFrame:
    path = RAW_DIR / "dart" / "corp_codes.parquet"
    if path.exists() and not OVERWRITE:
        return pd.read_parquet(path)
    content = dart_request("corpCode.xml", {}, binary=True)
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        xml = zf.read(zf.namelist()[0])
    soup = BeautifulSoup(xml, "xml")
    rows = []
    for item in soup.find_all("list"):
        stock_code = item.stock_code.get_text(strip=True) if item.stock_code else ""
        rows.append(
            {
                "corp_code": item.corp_code.get_text(strip=True),
                "corp_name": item.corp_name.get_text(strip=True),
                "stock_code": stock_code.zfill(6) if stock_code else "",
                "modify_date": pd.to_datetime(
                    item.modify_date.get_text(strip=True) if item.modify_date else "",
                    errors="coerce",
                ),
            }
        )
    out = pd.DataFrame(rows)
    atomic_parquet(out, path)
    return out


def collect_dart_profiles(corps: pd.DataFrame, universe: pd.DataFrame) -> None:
    path = RAW_DIR / "dart" / "company_profiles.parquet"
    if path.exists() and not OVERWRITE:
        return
    allowed = set(universe["ticker"].astype(str).str.zfill(6))
    work = corps[corps["stock_code"].isin(allowed)]
    rows: list[dict[str, Any]] = []
    for corp_code in tqdm(work["corp_code"], desc="DART 기업개황"):
        try:
            payload = dart_request("company.json", {"corp_code": corp_code})
            rows.append(payload)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("DART 기업개황 %s 실패: %s", corp_code, exc)
    if rows:
        out = pd.DataFrame(rows)
        if "stock_code" in out:
            out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
        atomic_parquet(out, path)


def month_chunks(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    cursor = start.replace(day=1)
    while cursor <= end:
        next_month = cursor + pd.offsets.MonthBegin(1)
        yield max(start, cursor), min(end, next_month - pd.Timedelta(days=1))
        cursor = next_month


def collect_dart_disclosures() -> pd.DataFrame:
    path = RAW_DIR / "dart" / "disclosures.parquet"
    if path.exists() and not OVERWRITE:
        return pd.read_parquet(path)
    rows = []
    chunks = list(month_chunks(pd.Timestamp(START_DATE), pd.Timestamp(END_DATE)))
    for start, end in tqdm(chunks, desc="DART 공시목록"):
        for corp_cls in ("Y", "K"):
            page = 1
            while True:
                try:
                    payload = dart_request(
                        "list.json",
                        {
                            "bgn_de": start.strftime("%Y%m%d"),
                            "end_de": end.strftime("%Y%m%d"),
                            "corp_cls": corp_cls,
                            "page_no": page,
                            "page_count": 100,
                            "sort": "date",
                            "sort_mth": "asc",
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("공시목록 실패 %s~%s %s: %s", start, end, corp_cls, exc)
                    break
                items = payload.get("list", []) or []
                rows.extend(items)
                total_page = int(payload.get("total_page", 0) or 0)
                if not items or page >= total_page:
                    break
                page += 1
    out = pd.DataFrame(rows)
    if not out.empty:
        out["rcept_dt"] = pd.to_datetime(out["rcept_dt"], errors="coerce")
        if "stock_code" in out:
            out["stock_code"] = out["stock_code"].astype(str).str.zfill(6)
        out = out.drop_duplicates("rcept_no", keep="last")
        atomic_parquet(out, path)
    return out


def chunks(values: Sequence[str], size: int) -> Iterator[list[str]]:
    for i in range(0, len(values), size):
        yield list(values[i : i + size])


def collect_dart_financials(corps: pd.DataFrame, universe: pd.DataFrame) -> None:
    path = RAW_DIR / "dart" / "financials_major_accounts.parquet"
    allowed = set(universe["ticker"].astype(str).str.zfill(6))
    work = corps[corps["stock_code"].isin(allowed)].drop_duplicates("corp_code")
    corp_codes = work["corp_code"].astype(str).tolist()
    stock_map = dict(zip(work["corp_code"].astype(str), work["stock_code"].astype(str).str.zfill(6)))
    name_map = dict(zip(work["corp_code"].astype(str), work["corp_name"].astype(str)))
    old = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    done = set()
    if not old.empty and {"bsns_year", "reprt_code"}.issubset(old.columns):
        done = set(map(tuple, old[["bsns_year", "reprt_code"]].astype(str).drop_duplicates().itertuples(index=False, name=None)))

    report_codes = ("11013", "11012", "11014", "11011")
    for year in tqdm(range(pd.Timestamp(START_DATE).year, pd.Timestamp(END_DATE).year + 1), desc="DART 재무연도"):
        for report_code in report_codes:
            if (str(year), report_code) in done and not OVERWRITE:
                continue
            rows = []
            for batch in chunks(corp_codes, 100):
                try:
                    payload = dart_request(
                        "fnlttMultiAcnt.json",
                        {
                            "corp_code": ",".join(batch),
                            "bsns_year": str(year),
                            "reprt_code": report_code,
                        },
                    )
                    for item in payload.get("list", []) or []:
                        corp_code = str(item.get("corp_code", ""))
                        item["stock_code"] = str(item.get("stock_code") or stock_map.get(corp_code, "")).zfill(6)
                        item["corp_name_master"] = name_map.get(corp_code, "")
                        rows.append(item)
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("DART 재무 %s %s batch 실패: %s", year, report_code, exc)
            if rows:
                batch_df = pd.DataFrame(rows)
                merge_parquet(
                    batch_df,
                    path,
                    ["corp_code", "stock_code", "bsns_year", "reprt_code", "fs_div", "sj_div", "account_nm"],
                    ["stock_code", "bsns_year", "reprt_code"],
                )


def extract_dart_document_text(content: bytes) -> str:
    texts = []
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        for name in zf.namelist():
            if not name.lower().endswith((".xml", ".html", ".htm", ".txt")):
                continue
            raw = zf.read(name)
            decoded = None
            for enc in ("utf-8", "euc-kr", "cp949"):
                try:
                    decoded = raw.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if decoded:
                text = BeautifulSoup(decoded, "lxml").get_text(" ", strip=True)
                text = re.sub(r"\s+", " ", text)
                if text:
                    texts.append(text)
    return "\n".join(texts)


def collect_dart_risk_documents(disclosures: pd.DataFrame) -> None:
    if not DOWNLOAD_RISK_DOCUMENTS or disclosures.empty:
        return
    pattern = "|".join(re.escape(x) for x in DART_RISK_KEYWORDS)
    work = disclosures[
        disclosures["report_nm"].astype(str).str.contains(pattern, regex=True, na=False)
    ].sort_values("rcept_dt").tail(MAX_RISK_DOCUMENTS)
    out_dir = RAW_DIR / "dart" / "documents"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = RAW_DIR / "dart" / "document_manifest.parquet"
    old = pd.read_parquet(manifest_path) if manifest_path.exists() else pd.DataFrame()
    done = set(old.loc[old.get("status", pd.Series(dtype=str)).eq("ok"), "rcept_no"].astype(str)) if not old.empty else set()
    buffer = []
    for row in tqdm(work.itertuples(index=False), total=len(work), desc="DART 위험공시 원문"):
        rcept_no = str(row.rcept_no)
        text_path = out_dir / f"{rcept_no}.txt"
        if rcept_no in done and text_path.exists() and not OVERWRITE:
            continue
        try:
            content = dart_request("document.xml", {"rcept_no": rcept_no}, binary=True)
            text = extract_dart_document_text(content)
            text_path.write_text(text, encoding="utf-8")
            rec = {
                "rcept_no": rcept_no,
                "stock_code": str(getattr(row, "stock_code", "")).zfill(6),
                "rcept_dt": getattr(row, "rcept_dt", None),
                "report_nm": getattr(row, "report_nm", ""),
                "text_path": str(text_path),
                "text_length": len(text),
                "status": "ok",
            }
        except Exception as exc:  # noqa: BLE001
            rec = {"rcept_no": rcept_no, "status": "failed", "error": repr(exc)}
        buffer.append(rec)
        if len(buffer) >= 100:
            merge_parquet(pd.DataFrame(buffer), manifest_path, ["rcept_no"], ["rcept_no"])
            buffer.clear()
    if buffer:
        merge_parquet(pd.DataFrame(buffer), manifest_path, ["rcept_no"], ["rcept_no"])


def collect_dart(universe: pd.DataFrame) -> None:
    if not os.getenv("OPENDART_API_KEY", "").strip():
        LOGGER.warning("OPENDART_API_KEY가 없어 DART 수집을 건너뜁니다.")
        return
    corps = collect_dart_corp_codes()
    collect_dart_profiles(corps, universe)
    disclosures = collect_dart_disclosures()
    collect_dart_financials(corps, universe)
    collect_dart_risk_documents(disclosures)



# 메인

def main() -> None:
    started = pd.Timestamp.now(tz="Asia/Seoul")
    LOGGER.info("수집 시작 %s ~ %s", START_DATE, END_DATE)
    if not os.getenv("KRX_ID") or not os.getenv("KRX_PW"):
        LOGGER.warning("KRX_ID/KRX_PW가 비어 있습니다. 일부 KRX 조회가 실패할 수 있습니다.")

    universe = build_historical_universe()
    LOGGER.info("최종 유니버스: %d개, 강제 포함: %d개", len(universe), universe["ticker"].isin(FORCED_TICKERS).sum())

    steps: list[tuple[str, Callable[[], None]]] = [
        ("KRX 종목", lambda: collect_krx_stocks(universe)),
        ("KRX 지수", collect_krx_indices),
        ("교차자산", collect_cross_assets),
        ("뉴스", lambda: collect_news(universe)),
        ("FRED", collect_fred),
        ("ECOS", collect_ecos),
        ("정책회의", collect_policy_events),
        ("OpenDART", lambda: collect_dart(universe)),
    ]
    status = []
    for name, fn in steps:
        try:
            LOGGER.info("=== %s 시작 ===", name)
            fn()
            status.append({"step": name, "status": "ok"})
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("%s 전체 단계 실패", name)
            status.append({"step": name, "status": "failed", "error": repr(exc)})

    ended = pd.Timestamp.now(tz="Asia/Seoul")
    manifest = {
        "started_at": started,
        "ended_at": ended,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "universe_count": len(universe),
        "forced_tickers": FORCED_TICKERS,
        "steps": status,
    }
    write_json(manifest, META_DIR / "collection_run_manifest.json")
    LOGGER.info("수집 종료. 소요시간: %s", ended - started)
    수집_품질_보고서_생성()
    결과압축_생성("01_데이터크롤링")
    print("\n다음 실행: python 02_피쳐_생성_봉인.py")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            결과압축_생성("01_데이터크롤링")
        finally:
            raise
