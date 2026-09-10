#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
02_피쳐_생성_봉인.py
====================
01_데이터_크롤링.py가 저장한 원천 데이터에서 급락 사전탐지용 피처·라벨을 만든다.
그 뒤 마지막 20거래일을 라벨 완성용 tail로 남기고, 그 직전 60거래일을
15거래일 × 4개 봉인구간(S00~S03)으로 물리적으로 분리한다.

실행:
    python 02_피쳐_생성_봉인.py

예측 시점 가정:
    한국 주식시장 종가가 확정된 뒤, 다음 거래일부터 향후 5/10/20거래일의
    급락 가능성을 예측한다.

시점 정렬 원칙:
- 미국·유럽 시장: 현지 날짜 D의 종가는 한국 거래일 D+1부터 사용
- 중국 시장: 한국장 종료 뒤 종가가 확정되므로 다음 KRX 거래일부터 사용
- 일본 시장: 한국장 종료 전에 종가가 확정되므로 같은 KRX 거래일 사용
- BTC/ETH: 완성된 UTC 일봉만 사용하기 위해 다음 KRX 거래일부터 사용
- FOMC: 미국 오후 발표이므로 다음 KRX 거래일부터 사용
- 한국은행 결정: 한국 종가 후 예측 기준으로 발표 당일부터 사용
- 월별 주택가격 통계: 발표시차를 보수적으로 45일 적용
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import duckdb
import numpy as np
import pandas as pd

from 공통_도구 import 결과압축_생성, 피쳐_품질_보고서_생성


# 설정

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR / "crashwatch_ai_data"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
DEVELOPMENT_DIR = DATA_ROOT / "development"
SEALED_DIR = DATA_ROOT / "sealed"
META_DIR = DATA_ROOT / "meta"
LOG_DIR = DATA_ROOT / "logs"
for d in (PROCESSED_DIR, DEVELOPMENT_DIR, SEALED_DIR, META_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

WINDOWS = (2, 3, 5, 10, 20, 60, 120, 250)
LABEL_HORIZONS = (5, 10, 20)
ABS_CRASH_THRESHOLDS = {5: -0.07, 10: -0.09, 20: -0.12}
IDIO_CRASH_THRESHOLDS = {5: -0.05, 10: -0.07, 20: -0.10}
MIN_HISTORY_DAYS = 260
SEAL_COUNT = 4
SEAL_DAYS = 15
LABEL_TAIL_DAYS = max(LABEL_HORIZONS)
EPS = 1e-12

MAJOR_TICKERS: dict[str, str] = {
    "005930": "samsung_electronics",
    "000660": "sk_hynix",
    "005380": "hyundai_motor",
    "000270": "kia",
    "373220": "lg_energy_solution",
    "207940": "samsung_biologics",
    "068270": "celltrion",
    "035420": "naver",
    "035720": "kakao",
    "005490": "posco_holdings",
    "012330": "hyundai_mobis",
    "006400": "samsung_sdi",
    "051910": "lg_chem",
    "105560": "kb_financial",
    "055550": "shinhan_financial",
    "086790": "hana_financial",
    "042700": "hanmi_semiconductor",
}

# 자산의 현지 날짜 종가가 한국 종가 시점에 사용 가능한지에 따른 보수적 거래일 지연
SAME_DAY_ASSETS = {"nikkei225"}
NEXT_KRX_DAY_ASSETS = {
    "sp500",
    "nasdaq",
    "dow",
    "russell2000",
    "semiconductor_sox",
    "hangseng",
    "shanghai",
    "shenzhen",
    "eurostoxx50",
    "dax",
    "ftse100",
    "india_nifty50",
    "btc",
    "eth",
    "gold",
    "wti",
    "copper",
    "dollar_index",
    "us_reit_vnq",
    "us_realestate_xlre",
    "us_realestate_iyr",
    "us_homebuilders_xhb",
    "us_homebuilders_itb",
}
ANCHOR_ASSETS = (
    "sp500",
    "nasdaq",
    "semiconductor_sox",
    "nikkei225",
    "shanghai",
    "btc",
    "us_reit_vnq",
    "dollar_index",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "02_피쳐_생성_봉인.log", encoding="utf-8"),
    ],
)
LOGGER = logging.getLogger("feature_builder")


# 유틸리티

def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_div(a: pd.Series, b: pd.Series | float) -> pd.Series:
    return a / (b + EPS)


def to_numeric_series(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            result = result.combine_first(pd.to_numeric(df[col], errors="coerce"))
    return result


def positive_streak(s: pd.Series) -> pd.Series:
    mask = s.fillna(0).gt(0)
    group = (~mask).cumsum()
    return mask.astype("int16").groupby(group).cumsum().astype("int16")


def negative_streak(s: pd.Series) -> pd.Series:
    mask = s.fillna(0).lt(0)
    group = (~mask).cumsum()
    return mask.astype("int16").groupby(group).cumsum().astype("int16")


def rolling_z(s: pd.Series, window: int) -> pd.Series:
    minp = min(window, max(2, window // 3))
    mean = s.rolling(window, min_periods=minp).mean()
    std = s.rolling(window, min_periods=minp).std(ddof=0)
    return (s - mean) / (std + EPS)


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-diff.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / (loss + EPS)
    return 100 - 100 / (1 + rs)


def days_since_extreme(s: pd.Series, window: int, kind: str) -> pd.Series:
    values = s.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    for i in range(window - 1, len(values)):
        x = values[i - window + 1 : i + 1]
        if np.isnan(x).all():
            continue
        idx = np.nanargmax(x) if kind == "max" else np.nanargmin(x)
        out[i] = window - 1 - idx
    return pd.Series(out, index=s.index)


def next_available_krx_date(dt: pd.Timestamp, krx_dates: pd.DatetimeIndex, same_day: bool) -> pd.Timestamp | pd.NaT:
    if pd.isna(dt):
        return pd.NaT
    side = "left" if same_day else "right"
    pos = np.searchsorted(krx_dates.values, np.datetime64(pd.Timestamp(dt).normalize()), side=side)
    if pos >= len(krx_dates):
        return pd.NaT
    return pd.Timestamp(krx_dates[pos])


def future_min(s: pd.Series, horizon: int) -> pd.Series:
    # t+1 ... t+h만 포함한다.
    return s.shift(-1)[::-1].rolling(horizon, min_periods=horizon).min()[::-1]


def future_max(s: pd.Series, horizon: int) -> pd.Series:
    return s.shift(-1)[::-1].rolling(horizon, min_periods=horizon).max()[::-1]


def read_parquet_glob(pattern: str) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        escaped = pattern.replace("'", "''")
        return con.execute(
            f"SELECT * FROM read_parquet('{escaped}', union_by_name=true)"
        ).df()
    finally:
        con.close()



# 원천 데이터 로드 및 표준화

def load_stock_raw() -> pd.DataFrame:
    pattern = str(RAW_DIR / "krx" / "stocks" / "ticker=*" / "*.parquet")
    LOGGER.info("KRX 원천 데이터 로드")
    df = read_parquet_glob(pattern)
    if df.empty:
        raise RuntimeError("KRX 원천 데이터가 없습니다. 01_데이터_크롤링.py를 먼저 실행하세요.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)
    df = df.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])

    # 01 수집기의 버전 차이/접두사 차이를 표준 열로 합친다.
    canonical: dict[str, Sequence[str]] = {
        "short_volume": ("short_short_volume", "short_volume"),
        "short_total_buy_volume": ("short_total_buy_volume", "total_buy_volume"),
        "short_volume_ratio_pct": ("short_volume_ratio_pct", "short_ratio_pct"),
        "short_balance_shares": (
            "shortbal_short_balance_shares",
            "short_balance_shares",
        ),
        "short_balance_value": (
            "shortbal_short_balance_value",
            "short_balance_value",
        ),
        "short_balance_ratio_pct": (
            "short_balance_ratio_pct",
            "shortbal_ratio_pct",
        ),
        "foreign_ownership_pct": (
            "foreign_ownership_pct",
            "foreign_limit_exhaustion_pct",
        ),
    }
    for target, candidates in canonical.items():
        df[target] = to_numeric_series(df, candidates)
    return df


def load_industry_map() -> pd.DataFrame:
    path = RAW_DIR / "dart" / "company_profiles.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["ticker", "industry_code", "industry_name"])
    p = pd.read_parquet(path)
    if p.empty or "stock_code" not in p.columns:
        return pd.DataFrame(columns=["ticker", "industry_code", "industry_name"])
    out = pd.DataFrame(
        {
            "ticker": p["stock_code"].astype(str).str.zfill(6),
            "industry_code": p.get("induty_code", pd.Series("UNKNOWN", index=p.index)).astype(str),
            "industry_name": p.get("corp_name", pd.Series("", index=p.index)).astype(str),
        }
    ).drop_duplicates("ticker", keep="last")
    return out



# 종목 자체 피처

def build_one_stock_features(g: pd.DataFrame) -> pd.DataFrame:
    """종목 단위 피처를 사전에 dict로 모아 한 번에 결합해 DataFrame 파편화를 방지한다."""
    g = g.sort_values("date").copy()

    def num(col: str) -> pd.Series:
        if col in g.columns:
            return pd.to_numeric(g[col], errors="coerce")
        return pd.Series(np.nan, index=g.index, dtype="float64")

    open_ = num("open")
    high = num("high")
    low = num("low")
    close = num("close")
    volume = num("volume")
    trading_value = num("trading_value")
    market_cap = num("market_cap")
    listed_shares = num("listed_shares")

    f: dict[str, pd.Series | np.ndarray] = {}
    ret_1 = close.pct_change(fill_method=None)
    log_ret_1 = np.log(close.where(close > 0)).diff()
    up_day = ret_1.gt(0).astype("int8")
    down_day = ret_1.lt(0).astype("int8")
    f["ret_1"] = ret_1
    f["log_ret_1"] = log_ret_1
    f["up_day"] = up_day
    f["down_day"] = down_day
    f["up_streak_days"] = positive_streak(ret_1)
    f["down_streak_days"] = negative_streak(ret_1)
    f["gap_return"] = safe_div(open_, close.shift(1)) - 1
    f["intraday_return"] = safe_div(close, open_) - 1
    f["range_pct"] = safe_div(high - low, close.shift(1))
    f["close_location"] = safe_div(close - low, high - low)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    f["upper_wick_pct"] = safe_div(high - body_high, close)
    f["lower_wick_pct"] = safe_div(body_low - low, close)
    true_range = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1
    ).max(axis=1)
    f["rsi_14"] = rsi(close, 14)

    for w in WINDOWS:
        minp = min(w, max(2, w // 3))
        ret_w = close.pct_change(w, fill_method=None)
        f[f"ret_{w}"] = ret_w
        f[f"logret_sum_{w}"] = log_ret_1.rolling(w, min_periods=minp).sum()
        f[f"vol_{w}"] = log_ret_1.rolling(w, min_periods=minp).std(ddof=0) * np.sqrt(252)
        f[f"downside_vol_{w}"] = (
            log_ret_1.where(log_ret_1 < 0).rolling(w, min_periods=minp).std(ddof=0) * np.sqrt(252)
        )
        f[f"skew_{w}"] = log_ret_1.rolling(w, min_periods=minp).skew()
        f[f"kurt_{w}"] = log_ret_1.rolling(w, min_periods=minp).kurt()
        f[f"ma_ratio_{w}"] = safe_div(close, close.rolling(w, min_periods=minp).mean()) - 1
        f[f"ema_ratio_{w}"] = safe_div(close, close.ewm(span=w, adjust=False).mean()) - 1
        f[f"atr_pct_{w}"] = safe_div(true_range.rolling(w, min_periods=minp).mean(), close)
        rolling_high = high.rolling(w, min_periods=minp).max()
        rolling_low = low.rolling(w, min_periods=minp).min()
        f[f"dist_high_{w}"] = safe_div(close, rolling_high) - 1
        f[f"dist_low_{w}"] = safe_div(close, rolling_low) - 1
        f[f"days_since_high_{w}"] = days_since_extreme(close, w, "max")
        f[f"days_since_low_{w}"] = days_since_extreme(close, w, "min")
        f[f"up_ratio_{w}"] = up_day.rolling(w, min_periods=minp).mean()
        f[f"down_ratio_{w}"] = down_day.rolling(w, min_periods=minp).mean()
        f[f"volume_ratio_{w}"] = safe_div(volume, volume.rolling(w, min_periods=minp).mean())
        f[f"value_ratio_{w}"] = safe_div(trading_value, trading_value.rolling(w, min_periods=minp).mean())
        f[f"volume_z_{w}"] = rolling_z(volume, w)
        f[f"value_z_{w}"] = rolling_z(trading_value, w)
        f[f"amihud_{w}"] = safe_div(ret_1.abs(), trading_value).rolling(w, min_periods=minp).mean() * 1e12
        f[f"drawdown_{w}"] = safe_div(close, rolling_high) - 1

    f["log_market_cap"] = np.log1p(market_cap.clip(lower=0))
    f["turnover_value_pct"] = safe_div(trading_value, market_cap)
    f["turnover_shares_pct"] = safe_div(volume, listed_shares)

    flow_cols = (
        "value_foreign_net", "value_institution_net", "value_individual_net", "value_other_corp_net",
        "shares_foreign_net", "shares_institution_net", "shares_individual_net", "shares_other_corp_net",
    )
    flow_series: dict[str, pd.Series] = {}
    for col in flow_cols:
        if col not in g.columns:
            continue
        value = num(col)
        flow_series[col] = value
        f[f"{col}_to_value"] = safe_div(value, trading_value.abs())
        if col.startswith("value_"):
            f[f"{col}_to_mcap"] = safe_div(value, market_cap)
        f[f"{col}_buy_streak"] = positive_streak(value)
        f[f"{col}_sell_streak"] = negative_streak(value)
        for w in (3, 5, 10, 20, 60):
            minp = min(w, max(2, w // 3))
            f[f"{col}_sum_{w}"] = value.rolling(w, min_periods=minp).sum()
            f[f"{col}_positive_ratio_{w}"] = value.gt(0).rolling(w, min_periods=minp).mean()
            f[f"{col}_z_{w}"] = rolling_z(value, w)

    foreign = flow_series.get("value_foreign_net")
    inst = flow_series.get("value_institution_net")
    if foreign is not None and inst is not None:
        smart = foreign + inst
        joint_buy = (foreign > 0) & (inst > 0)
        joint_sell = (foreign < 0) & (inst < 0)
        f["smart_money_net"] = smart
        f["smart_money_to_value"] = safe_div(smart, trading_value.abs())
        f["foreign_institution_joint_buy"] = joint_buy.astype("int8")
        f["foreign_institution_joint_sell"] = joint_sell.astype("int8")
        f["joint_buy_streak"] = positive_streak(joint_buy.astype(int))
        f["joint_sell_streak"] = positive_streak(joint_sell.astype(int))
        f["foreign_buy_price_down"] = ((foreign > 0) & (ret_1 < 0)).astype("int8")
        f["foreign_sell_price_up"] = ((foreign < 0) & (ret_1 > 0)).astype("int8")
        f["inst_buy_price_down"] = ((inst > 0) & (ret_1 < 0)).astype("int8")
        f["inst_sell_price_up"] = ((inst < 0) & (ret_1 > 0)).astype("int8")
        for w in (5, 10, 20, 60):
            f[f"smart_money_sum_{w}"] = smart.rolling(w, min_periods=max(2, w // 3)).sum()
        f["smart_money_accel_5_20"] = f["smart_money_sum_5"] - f["smart_money_sum_20"] / 4

    foreign_pct = num("foreign_ownership_pct")
    for w in (1, 5, 20, 60):
        f[f"foreign_ownership_chg_{w}"] = foreign_pct.diff(w)

    short_volume = num("short_volume")
    short_volume_ratio = num("short_volume_ratio_pct")
    short_balance_to_mcap = safe_div(num("short_balance_value"), market_cap)
    f["short_volume_to_volume"] = safe_div(short_volume, volume)
    f["short_balance_value_to_mcap"] = short_balance_to_mcap
    for w in (5, 20, 60):
        f[f"short_volume_ratio_z_{w}"] = rolling_z(short_volume_ratio, w)
        f[f"short_balance_chg_{w}"] = short_balance_to_mcap.diff(w)
        f[f"short_volume_sum_{w}"] = short_volume.rolling(w, min_periods=max(2, w // 3)).sum()
    f["short_price_divergence_20"] = f["short_balance_chg_20"] * f["ret_20"]

    for col in ("per", "pbr", "eps", "bps", "dividend_yield", "dps"):
        if col in g.columns:
            value = num(col)
            f[col] = value
            f[f"{col}_chg_20"] = value.pct_change(20, fill_method=None)

    f["history_days"] = np.arange(1, len(g) + 1)
    feature_frame = pd.DataFrame(f, index=g.index)
    duplicate_cols = [c for c in feature_frame.columns if c in g.columns]
    if duplicate_cols:
        g = g.drop(columns=duplicate_cols)
    return pd.concat([g, feature_frame], axis=1).copy()

def build_stock_features(raw: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for ticker, block in raw.groupby("ticker", sort=False):
        enriched = build_one_stock_features(block)
        enriched["ticker"] = ticker
        blocks.append(enriched)
    return pd.concat(blocks, ignore_index=True) if blocks else raw.copy()



# 시장·업종·타 회사 급락 전염 피처

def add_breadth_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # KRX 원천에는 있지만 Yahoo Finance 대체 원천에는 없을 수 있는 열을 보완한다.
    # 거래대금은 가격×거래량 프록시로 계산하며, 시가총액이 없으면 가격 순위를 사용한다.
    if "trading_value" not in out.columns:
        close = pd.to_numeric(out.get("close", 0), errors="coerce").fillna(0)
        volume = pd.to_numeric(out.get("volume", 0), errors="coerce").fillna(0)
        out["trading_value"] = close * volume
    else:
        out["trading_value"] = pd.to_numeric(out["trading_value"], errors="coerce").fillna(0)
    if "market_cap" not in out.columns:
        out["market_cap"] = pd.to_numeric(out.get("close", 0), errors="coerce").fillna(0)
    out["crash_1d_3pct"] = out["ret_1"].le(-0.03).astype("int8")
    out["crash_1d_5pct"] = out["ret_1"].le(-0.05).astype("int8")
    out["crash_5d_8pct"] = out["ret_5"].le(-0.08).astype("int8")

    all_daily = (
        out.groupby("date", as_index=False)
        .agg(
            all_market_median_ret=("ret_1", "median"),
            all_market_cross_vol=("ret_1", "std"),
            all_market_down_ratio=("down_day", "mean"),
            all_market_crash3_ratio=("crash_1d_3pct", "mean"),
            all_market_crash5_ratio=("crash_1d_5pct", "mean"),
            all_market_crash5d8_ratio=("crash_5d_8pct", "mean"),
            all_market_trading_value=("trading_value", "sum"),
        )
    )
    out = out.merge(all_daily, on="date", how="left")

    market_daily = (
        out.groupby(["date", "market"], as_index=False)
        .agg(
            market_median_ret=("ret_1", "median"),
            market_cross_vol=("ret_1", "std"),
            market_down_ratio=("down_day", "mean"),
            market_crash3_ratio=("crash_1d_3pct", "mean"),
            market_crash5_ratio=("crash_1d_5pct", "mean"),
            market_crash5d8_ratio=("crash_5d_8pct", "mean"),
            market_trading_value=("trading_value", "sum"),
        )
    )
    out = out.merge(market_daily, on=["date", "market"], how="left")
    out["relative_ret_1"] = out["ret_1"] - out["market_median_ret"]

    # 시총 상위 50개 종목의 연쇄 급락은 시장 시스템 리스크 프록시
    cap_rank = out.groupby("date")["market_cap"].rank(method="first", ascending=False)
    top50 = out[cap_rank <= 50]
    top50_daily = (
        top50.groupby("date", as_index=False)
        .agg(
            top50_median_ret=("ret_1", "median"),
            top50_crash3_ratio=("crash_1d_3pct", "mean"),
            top50_crash5_ratio=("crash_1d_5pct", "mean"),
            top50_down_ratio=("down_day", "mean"),
        )
    )
    out = out.merge(top50_daily, on="date", how="left")

    if "industry_code" in out.columns:
        keys = ["date", "industry_code"]
        peer = (
            out.groupby(keys, as_index=False)
            .agg(
                industry_count=("ticker", "size"),
                industry_ret_sum=("ret_1", "sum"),
                industry_down_sum=("down_day", "sum"),
                industry_crash3_sum=("crash_1d_3pct", "sum"),
                industry_crash5_sum=("crash_1d_5pct", "sum"),
                industry_cross_vol=("ret_1", "std"),
            )
        )
        out = out.merge(peer, on=keys, how="left")
        denom = (out["industry_count"] - 1).clip(lower=1)
        out["peer_mean_ret_ex_self"] = (out["industry_ret_sum"] - out["ret_1"].fillna(0)) / denom
        out["peer_down_ratio_ex_self"] = (out["industry_down_sum"] - out["down_day"]) / denom
        out["peer_crash3_ratio_ex_self"] = (
            out["industry_crash3_sum"] - out["crash_1d_3pct"]
        ) / denom
        out["peer_crash5_ratio_ex_self"] = (
            out["industry_crash5_sum"] - out["crash_1d_5pct"]
        ) / denom
        out["industry_relative_ret_1"] = out["ret_1"] - out["peer_mean_ret_ex_self"]
    return out


def add_major_company_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    keep_cols = [
        "date",
        "ticker",
        "ret_1",
        "ret_3",
        "ret_5",
        "drawdown_20",
        "down_streak_days",
        "value_foreign_net_sell_streak",
        "value_institution_net_sell_streak",
        "short_balance_chg_20",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    major = out[out["ticker"].isin(MAJOR_TICKERS)][keep_cols].copy()
    blocks = []
    for ticker, alias in MAJOR_TICKERS.items():
        block = major[major["ticker"] == ticker].drop(columns=["ticker"])
        if block.empty:
            continue
        block = block.rename(columns={c: f"major_{alias}_{c}" for c in block.columns if c != "date"})
        blocks.append(block)
    for block in blocks:
        out = out.merge(block, on="date", how="left")

    crash_cols = []
    for alias in MAJOR_TICKERS.values():
        col = f"major_{alias}_ret_1"
        if col in out.columns:
            crash_col = f"major_{alias}_crash_3pct"
            out[crash_col] = out[col].le(-0.03).astype("float32")
            crash_cols.append(crash_col)
    if crash_cols:
        out["major_company_crash_count"] = out[crash_cols].sum(axis=1)
        out["major_company_crash_ratio"] = out[crash_cols].mean(axis=1)
        out["major_company_worst_ret_1"] = out[
            [c.replace("_crash_3pct", "_ret_1") for c in crash_cols]
        ].min(axis=1)

    # 자기 자신이 주요종목이면 자기 수익률 복제값은 제거한다.
    for ticker, alias in MAJOR_TICKERS.items():
        mask = out["ticker"].eq(ticker)
        own_cols = [c for c in out.columns if c.startswith(f"major_{alias}_")]
        if own_cols:
            out.loc[mask, own_cols] = np.nan
    return out



# 해외 증시·코인·부동산 전염 피처

def load_cross_asset_features(krx_dates: pd.DatetimeIndex) -> pd.DataFrame:
    path = RAW_DIR / "cross_assets.parquet"
    base = pd.DataFrame({"date": krx_dates})
    if not path.exists():
        LOGGER.warning("cross_assets.parquet 없음")
        return base
    raw = pd.read_parquet(path)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.tz_localize(None)
    raw["close"] = pd.to_numeric(raw.get("close"), errors="coerce")
    raw = raw.dropna(subset=["date", "alias", "close"])

    asset_blocks: list[pd.DataFrame] = []
    for alias, g in raw.groupby("alias", sort=False):
        g = g.sort_values("date").drop_duplicates("date", keep="last").copy()
        close = g["close"]
        g[f"xa_{alias}_close"] = close
        for w in (1, 2, 3, 5, 10, 20, 60):
            g[f"xa_{alias}_ret_{w}"] = close.pct_change(w, fill_method=None)
        logret = np.log(close.where(close > 0)).diff()
        for w in (5, 20, 60):
            g[f"xa_{alias}_vol_{w}"] = logret.rolling(w, min_periods=max(3, w // 3)).std(ddof=0) * np.sqrt(252)
            peak = close.rolling(w, min_periods=max(3, w // 3)).max()
            g[f"xa_{alias}_drawdown_{w}"] = close / (peak + EPS) - 1
        threshold = -0.06 if alias in {"btc", "eth"} else -0.03
        g[f"xa_{alias}_crash_1d"] = g[f"xa_{alias}_ret_1"].le(threshold).astype("int8")
        g[f"xa_{alias}_crash_3d"] = g[f"xa_{alias}_ret_3"].le(threshold * 1.5).astype("int8")
        g[f"xa_{alias}_down_streak"] = negative_streak(g[f"xa_{alias}_ret_1"])
        same_day = alias in SAME_DAY_ASSETS
        g["availability_date"] = [
            next_available_krx_date(dt, krx_dates, same_day=same_day) for dt in g["date"]
        ]
        g = g.dropna(subset=["availability_date"])
        feature_cols = [c for c in g.columns if c.startswith(f"xa_{alias}_")]
        block = g[["availability_date", *feature_cols]].rename(columns={"availability_date": "date"})
        block = block.drop_duplicates("date", keep="last")
        asset_blocks.append(block)

    out = base.copy()
    for block in asset_blocks:
        out = out.merge(block, on="date", how="left")
    out = out.sort_values("date")
    # 반복 merge 이후 조각난 블록을 한 번만 정리해 뒤의 파생열 삽입 비용을 줄인다.
    out = out.copy()
    feature_cols = [c for c in out.columns if c != "date"]
    out[feature_cols] = out[feature_cols].ffill(limit=7)

    equity_aliases = (
        "sp500",
        "nasdaq",
        "dow",
        "russell2000",
        "semiconductor_sox",
        "nikkei225",
        "hangseng",
        "shanghai",
        "shenzhen",
        "eurostoxx50",
        "dax",
        "ftse100",
        "india_nifty50",
    )
    equity_crash = [f"xa_{a}_crash_1d" for a in equity_aliases if f"xa_{a}_crash_1d" in out]
    equity_rets = [f"xa_{a}_ret_1" for a in equity_aliases if f"xa_{a}_ret_1" in out]
    if equity_crash:
        out["global_equity_crash_breadth"] = out[equity_crash].mean(axis=1)
        out["global_equity_crash_count"] = out[equity_crash].sum(axis=1)
    if equity_rets:
        out["global_equity_worst_ret_1"] = out[equity_rets].min(axis=1)
        out["global_equity_median_ret_1"] = out[equity_rets].median(axis=1)
        out["global_equity_cross_dispersion"] = out[equity_rets].std(axis=1)

    crypto_crash = [c for c in ("xa_btc_crash_1d", "xa_eth_crash_1d") if c in out]
    if crypto_crash:
        out["crypto_crash_breadth"] = out[crypto_crash].mean(axis=1)
    realestate_rets = [
        c
        for c in (
            "xa_us_reit_vnq_ret_1",
            "xa_us_realestate_xlre_ret_1",
            "xa_us_realestate_iyr_ret_1",
            "xa_us_homebuilders_xhb_ret_1",
            "xa_us_homebuilders_itb_ret_1",
        )
        if c in out
    ]
    if realestate_rets:
        out["global_realestate_worst_ret_1"] = out[realestate_rets].min(axis=1)
        out["global_realestate_median_ret_1"] = out[realestate_rets].median(axis=1)
    return out


def load_krx_index_features(krx_dates: pd.DatetimeIndex) -> pd.DataFrame:
    paths = sorted((RAW_DIR / "krx" / "indices").glob("*.parquet"))
    out = pd.DataFrame({"date": krx_dates})
    for path in paths:
        try:
            df = pd.read_parquet(path)
        except Exception:
            continue
        if df.empty or "close" not in df.columns:
            continue
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        alias = str(df.get("alias", pd.Series(path.stem, index=df.index)).iloc[0])
        alias = re.sub(r"[^0-9a-zA-Z가-힣_]+", "_", alias).lower()
        close = pd.to_numeric(df["close"], errors="coerce")
        block = pd.DataFrame({"date": df["date"]})
        for w in (1, 3, 5, 20, 60):
            block[f"krxidx_{alias}_ret_{w}"] = close.pct_change(w, fill_method=None)
        block[f"krxidx_{alias}_crash_1d"] = block[f"krxidx_{alias}_ret_1"].le(-0.03).astype("int8")
        out = out.merge(block.drop_duplicates("date", keep="last"), on="date", how="left")
    return out



# 정책 이벤트·거시·부동산 통계

def add_policy_event_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame({"date": dates})
    path = RAW_DIR / "policy_events.parquet"
    if not path.exists():
        return out
    events = pd.read_parquet(path)
    events["announcement_date_local"] = pd.to_datetime(events["announcement_date_local"], errors="coerce")
    effective_rows = []
    for row in events.itertuples(index=False):
        same_day = str(getattr(row, "availability_rule", "")) == "SAME_KRX_TRADING_DAY"
        effective = next_available_krx_date(row.announcement_date_local, dates, same_day=same_day)
        if pd.isna(effective):
            continue
        effective_rows.append(
            {
                "source": str(row.source).lower(),
                "event_type": str(row.event_type).lower(),
                "effective_date": effective,
            }
        )
    eff = pd.DataFrame(effective_rows)
    if eff.empty:
        return out

    date_to_pos = {pd.Timestamp(d): i for i, d in enumerate(dates)}
    for source in sorted(eff["source"].unique()):
        event_dates = sorted(pd.DatetimeIndex(eff.loc[eff["source"] == source, "effective_date"].unique()))
        event_pos = np.array([date_to_pos[d] for d in event_dates if d in date_to_pos], dtype=int)
        if len(event_pos) == 0:
            continue
        current_pos = np.arange(len(dates))
        prev_idx = np.searchsorted(event_pos, current_pos, side="right") - 1
        next_idx = np.searchsorted(event_pos, current_pos, side="left")
        days_since = np.where(prev_idx >= 0, current_pos - event_pos[np.clip(prev_idx, 0, len(event_pos) - 1)], np.nan)
        days_to = np.where(next_idx < len(event_pos), event_pos[np.clip(next_idx, 0, len(event_pos) - 1)] - current_pos, np.nan)
        out[f"event_{source}_today"] = np.isin(current_pos, event_pos).astype("int8")
        out[f"event_{source}_days_since"] = days_since
        out[f"event_{source}_days_to"] = days_to
        for w in (1, 3, 5, 10):
            out[f"event_{source}_within_next_{w}d"] = pd.Series(days_to).between(0, w).astype("int8")
            out[f"event_{source}_within_prev_{w}d"] = pd.Series(days_since).between(0, w).astype("int8")
    return out


def load_macro_features(krx_dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame({"date": krx_dates})

    fred_path = RAW_DIR / "macro_fred.parquet"
    if fred_path.exists():
        f = pd.read_parquet(fred_path)
        f["date"] = pd.to_datetime(f["date"], errors="coerce")
        # 동일 관측일의 여러 vintage가 있으면 최초 공개치 우선
        if "realtime_start" in f.columns:
            f = f.sort_values("realtime_start").drop_duplicates(["alias", "date"], keep="first")
        for alias, g in f.groupby("alias", sort=False):
            g = g.sort_values("date")
            # 미국 날짜 D 수치는 한국 거래일 D+1부터 사용
            g["availability_date"] = [next_available_krx_date(dt, krx_dates, same_day=False) for dt in g["date"]]
            block = g.dropna(subset=["availability_date"])[["availability_date", "value"]].rename(
                columns={"availability_date": "date", "value": f"macro_{alias}"}
            )
            out = out.merge(block.drop_duplicates("date", keep="last"), on="date", how="left")

    ecos_path = RAW_DIR / "macro_ecos.parquet"
    if ecos_path.exists():
        e = pd.read_parquet(ecos_path)
        e["date"] = pd.to_datetime(e["date"], errors="coerce")
        for alias, g in e.groupby("alias", sort=False):
            g = g.sort_values("date").copy()
            cycle = str(g.get("cycle", pd.Series("D", index=g.index)).iloc[0])
            is_housing = any(k in str(alias) for k in ("주택", "아파트", "전세", "housing"))
            if is_housing or cycle in {"M", "MM", "Q", "QY", "A", "YY"}:
                # 발표일 메타가 없으므로 보수적으로 45일 뒤부터 사용
                candidate = g["date"] + pd.Timedelta(days=45)
                same_day = True
            else:
                candidate = g["date"]
                same_day = True
            g["availability_date"] = [next_available_krx_date(dt, krx_dates, same_day=same_day) for dt in candidate]
            safe_alias = re.sub(r"[^0-9a-zA-Z가-힣_]+", "_", str(alias))[:120]
            block = g.dropna(subset=["availability_date"])[["availability_date", "value"]].rename(
                columns={"availability_date": "date", "value": f"macro_{safe_alias}"}
            )
            out = out.merge(block.drop_duplicates("date", keep="last"), on="date", how="left")

    out = out.sort_values("date")
    macro_cols = [c for c in out.columns if c.startswith("macro_")]
    if macro_cols:
        out[macro_cols] = out[macro_cols].ffill()
        for col in list(macro_cols):
            out[f"{col}_chg_1"] = out[col].pct_change(fill_method=None)
            out[f"{col}_diff_1"] = out[col].diff()
            out[f"{col}_z_60"] = rolling_z(out[col], 60)
        if {"macro_us_10y", "macro_us_2y"}.issubset(out.columns):
            out["macro_us_10y_2y_spread"] = out["macro_us_10y"] - out["macro_us_2y"]
        if {"macro_fed_target_upper", "macro_fed_target_lower"}.issubset(out.columns):
            out["macro_fed_target_mid"] = (
                out["macro_fed_target_upper"] + out["macro_fed_target_lower"]
            ) / 2
            out["macro_fed_target_width"] = out["macro_fed_target_upper"] - out["macro_fed_target_lower"]
            out["macro_fed_target_mid_chg_bp"] = out["macro_fed_target_mid"].diff() * 100
        if "macro_bok_base_rate" in out.columns:
            out["macro_bok_base_rate_chg_bp"] = out["macro_bok_base_rate"].diff() * 100
    return out




# 뉴스 피처

NEWS_CUTOFF_HOUR_KST = 18
NEWS_NEGATIVE_KEYWORDS = (
    "급락", "폭락", "하락", "부진", "적자", "손실", "위기", "충격", "우려", "경고",
    "부도", "파산", "회생", "횡령", "배임", "소송", "해지", "감자", "유상증자",
    "상장폐지", "관리종목", "거래정지", "리콜", "제재", "조사", "압수수색",
)
NEWS_RISK_KEYWORDS = (
    "반대매매", "신용잔고", "마진콜", "공매도", "유동성", "차입금", "부채",
    "환율 급등", "금리 인상", "부동산 PF", "채무불이행", "디폴트", "감사의견",
)


def _news_available_date(ts: pd.Timestamp, krx_dates: pd.DatetimeIndex) -> pd.Timestamp | pd.NaT:
    if pd.isna(ts):
        return pd.NaT
    value = pd.Timestamp(ts)
    if value.tzinfo is not None:
        value = value.tz_convert("Asia/Seoul").tz_localize(None)
    same_day = value.hour < NEWS_CUTOFF_HOUR_KST
    return next_available_krx_date(value.normalize(), krx_dates, same_day=same_day)


def _rolling_news_features(daily: pd.DataFrame, prefix: str, key_cols: list[str]) -> pd.DataFrame:
    sort_cols = [*key_cols, "date"]
    daily = daily.sort_values(sort_cols).copy()
    groups = daily.groupby(key_cols, sort=False) if key_cols else [(None, daily)]
    blocks = []
    for key, g in groups:
        g = g.sort_values("date").copy()
        for base_col in (
            f"{prefix}_article_count",
            f"{prefix}_negative_count",
            f"{prefix}_risk_count",
        ):
            if base_col not in g:
                continue
            for w in (3, 7, 20, 60):
                g[f"{base_col}_sum_{w}"] = g[base_col].rolling(w, min_periods=1).sum()
            g[f"{base_col}_z_60"] = rolling_z(g[base_col], 60)
        neg = g.get(f"{prefix}_negative_count", pd.Series(0.0, index=g.index))
        cnt = g.get(f"{prefix}_article_count", pd.Series(0.0, index=g.index))
        g[f"{prefix}_negative_ratio"] = safe_div(neg, cnt)
        blocks.append(g)
    return pd.concat(blocks, ignore_index=True) if blocks else daily


def add_news_features(base: pd.DataFrame, krx_dates: pd.DatetimeIndex) -> pd.DataFrame:
    out = base.copy()
    news_dir = RAW_DIR / "news"
    naver_path = news_dir / "naver_news.parquet"

    # 모든 종목이 시장뉴스 피처를 공유한다.
    if naver_path.exists():
        n = pd.read_parquet(naver_path)
        if not n.empty:
            n["published_at_kst"] = pd.to_datetime(n["published_at_kst"], errors="coerce")
            n["available_date"] = [
                _news_available_date(ts, krx_dates) for ts in n["published_at_kst"]
            ]
            n = n.dropna(subset=["available_date"])
            negative_raw = n["negative_keyword_count"] if "negative_keyword_count" in n else pd.Series(0, index=n.index)
            risk_raw = n["risk_keyword_count"] if "risk_keyword_count" in n else pd.Series(0, index=n.index)
            scope_raw = n["scope"] if "scope" in n else pd.Series("", index=n.index)
            n["negative_keyword_count"] = pd.to_numeric(negative_raw, errors="coerce").fillna(0)
            n["risk_keyword_count"] = pd.to_numeric(risk_raw, errors="coerce").fillna(0)

            market = n[scope_raw.astype(str).eq("market")].copy()
            if not market.empty:
                md = (
                    market.groupby("available_date", as_index=False)
                    .agg(
                        news_market_article_count=("title", "size"),
                        news_market_negative_count=("negative_keyword_count", "sum"),
                        news_market_risk_count=("risk_keyword_count", "sum"),
                        news_market_source_count=("source_domain", "nunique"),
                        news_market_query_count=("alias", "nunique"),
                    )
                    .rename(columns={"available_date": "date"})
                )
                md = _rolling_news_features(md, "news_market", [])
                out = out.merge(md, on="date", how="left")

            company = n[scope_raw.astype(str).eq("company")].copy()
            ticker_raw = company["ticker"] if "ticker" in company else pd.Series("", index=company.index)
            company["ticker"] = ticker_raw.astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
            if not company.empty:
                cd = (
                    company.groupby(["available_date", "ticker"], as_index=False)
                    .agg(
                        news_company_article_count=("title", "size"),
                        news_company_negative_count=("negative_keyword_count", "sum"),
                        news_company_risk_count=("risk_keyword_count", "sum"),
                        news_company_source_count=("source_domain", "nunique"),
                    )
                    .rename(columns={"available_date": "date"})
                )
                cd = _rolling_news_features(cd, "news_company", ["ticker"])
                out = out.merge(cd, on=["date", "ticker"], how="left")

    gdelt_path = news_dir / "gdelt_timeline.parquet"
    if gdelt_path.exists():
        g = pd.read_parquet(gdelt_path)
        if not g.empty:
            g["date"] = pd.to_datetime(g["date"], errors="coerce")
            # 글로벌 일간 뉴스량은 해당 일 종료 후 다음 KRX 거래일부터 사용
            g["available_date"] = [
                next_available_krx_date(dt, krx_dates, same_day=False) for dt in g["date"]
            ]
            g = g.dropna(subset=["available_date", "alias"])
            g["news_volume"] = pd.to_numeric(g.get("news_volume"), errors="coerce")
            wide = g.pivot_table(
                index="available_date",
                columns="alias",
                values="news_volume",
                aggfunc="last",
            ).sort_index()
            wide.columns = [f"news_gdelt_{str(c)}_volume" for c in wide.columns]
            wide = wide.reset_index().rename(columns={"available_date": "date"})
            for col in [c for c in wide if c.startswith("news_gdelt_")]:
                wide[f"{col}_chg_1"] = wide[col].pct_change(fill_method=None)
                wide[f"{col}_z_60"] = rolling_z(wide[col], 60)
            out = out.merge(wide, on="date", how="left")

    news_cols = [c for c in out.columns if c.startswith("news_")]
    if news_cols:
        # fillna 전에 실제 뉴스 원천이 병합된 행인지 기록한다.
        availability_cols = [c for c in news_cols if c.endswith(("article_count", "_volume"))]
        out["news_any_available"] = out[availability_cols].notna().any(axis=1).astype("int8") if availability_cols else 0
        # 뉴스가 없었던 날은 기사량 계열만 0으로, z-score/변화율은 그대로 결측으로 둔다.
        count_cols = [
            c for c in news_cols
            if any(x in c for x in ("article_count", "negative_count", "risk_count", "source_count", "query_count"))
            and not c.endswith(("_z_60", "_chg_1"))
        ]
        out[count_cols] = out[count_cols].fillna(0)
    else:
        out["news_any_available"] = 0
    return out



# DART 공시·재무 피처

def add_disclosure_features(base: pd.DataFrame, krx_dates: pd.DatetimeIndex) -> pd.DataFrame:
    path = RAW_DIR / "dart" / "disclosures.parquet"
    if not path.exists():
        return base
    d = pd.read_parquet(path)
    if d.empty or "stock_code" not in d.columns:
        return base
    d["ticker"] = d["stock_code"].astype(str).str.zfill(6)
    d["rcept_dt"] = pd.to_datetime(d["rcept_dt"], errors="coerce")
    # 종가 이후 예측에서 공시는 접수 다음 KRX 거래일부터 사용해 보수적으로 처리
    d["date"] = [next_available_krx_date(dt, krx_dates, same_day=False) for dt in d["rcept_dt"]]
    title = d["report_nm"].astype(str)
    patterns = {
        "dart_rights_issue": r"유상증자",
        "dart_capital_reduction": r"감자",
        "dart_convertible_bond": r"전환사채|신주인수권부사채|교환사채",
        "dart_major_shareholder_change": r"최대주주.*변경",
        "dart_embezzlement_breach": r"횡령|배임",
        "dart_default_rehabilitation": r"부도|회생절차",
        "dart_business_suspension": r"영업정지",
        "dart_delisting_warning": r"상장폐지|관리종목",
        "dart_audit_risk": r"감사의견|계속기업",
        "dart_lawsuit": r"소송",
        "dart_contract_cancel": r"계약.*해지|공급계약.*해지",
        "dart_unfaithful_disclosure": r"불성실공시",
        "dart_collateral": r"담보제공",
        "dart_treasury_disposal": r"자기주식.*처분",
    }
    d["dart_disclosure_count"] = 1
    for name, pattern in patterns.items():
        d[name] = title.str.contains(pattern, regex=True, na=False).astype("int8")
    daily = d.groupby(["date", "ticker"], as_index=False)[["dart_disclosure_count", *patterns]].sum()
    out = base.merge(daily, on=["date", "ticker"], how="left")
    cols = ["dart_disclosure_count", *patterns]
    out[cols] = out[cols].fillna(0)
    # 최근 N일 위험 공시 누적
    blocks = []
    for ticker, g in out.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        for col in cols:
            for w in (5, 20, 60):
                g[f"{col}_sum_{w}"] = g[col].rolling(w, min_periods=1).sum()
        g["ticker"] = ticker
        blocks.append(g)
    return pd.concat(blocks, ignore_index=True) if blocks else out


def canonical_account(account_name: str) -> str | None:
    text = str(account_name).lower()
    rules = [
        ("assets", ("자산총계", "total assets")),
        ("current_assets", ("유동자산", "current assets")),
        ("cash", ("현금및현금성자산", "cash and cash equivalents")),
        ("liabilities", ("부채총계", "total liabilities")),
        ("current_liabilities", ("유동부채", "current liabilities")),
        ("equity", ("자본총계", "total equity")),
        ("revenue", ("매출액", "영업수익", "revenue")),
        ("operating_income", ("영업이익", "operating income")),
        ("net_income", ("당기순이익", "분기순이익", "net income")),
        ("operating_cashflow", ("영업활동으로 인한 현금흐름", "operating activities")),
        ("short_borrowings", ("단기차입금", "short-term borrowings")),
        ("long_borrowings", ("장기차입금", "long-term borrowings")),
    ]
    for canonical, names in rules:
        if any(name.lower() in text for name in names):
            return canonical
    return None


def add_financial_features(base: pd.DataFrame, krx_dates: pd.DatetimeIndex) -> pd.DataFrame:
    path = RAW_DIR / "dart" / "financials_major_accounts.parquet"
    if not path.exists():
        return base
    f = pd.read_parquet(path)
    if f.empty:
        return base
    f["ticker"] = f["stock_code"].astype(str).str.zfill(6)
    f["canonical"] = f.get("account_nm", pd.Series("", index=f.index)).map(canonical_account)
    f["value"] = pd.to_numeric(
        f.get("thstrm_amount", pd.Series(np.nan, index=f.index))
        .astype(str)
        .str.replace(",", "", regex=False)
        .replace("-", np.nan),
        errors="coerce",
    )
    f = f.dropna(subset=["canonical", "value"])
    f["fs_priority"] = f.get("fs_div", pd.Series("", index=f.index)).map({"CFS": 0, "OFS": 1}).fillna(2)
    f = f.sort_values("fs_priority").drop_duplicates(
        ["ticker", "bsns_year", "reprt_code", "canonical"], keep="first"
    )
    wide = f.pivot_table(
        index=["ticker", "bsns_year", "reprt_code"],
        columns="canonical",
        values="value",
        aggfunc="last",
    ).reset_index()
    wide.columns.name = None

    # 정기보고서 공시일 연결
    dpath = RAW_DIR / "dart" / "disclosures.parquet"
    d = pd.read_parquet(dpath) if dpath.exists() else pd.DataFrame()
    report_patterns = {
        "11013": r"분기보고서.*1분기",
        "11012": r"반기보고서",
        "11014": r"분기보고서.*3분기",
        "11011": r"사업보고서",
    }
    dates = []
    if not d.empty and "stock_code" in d.columns:
        d["ticker"] = d["stock_code"].astype(str).str.zfill(6)
        d["rcept_dt"] = pd.to_datetime(d["rcept_dt"], errors="coerce")
        for row in wide.itertuples(index=False):
            subset = d[d["ticker"] == row.ticker]
            pattern = report_patterns.get(str(row.reprt_code), "")
            if pattern:
                subset = subset[subset["report_nm"].astype(str).str.contains(pattern, regex=True, na=False)]
            subset = subset[subset["rcept_dt"].dt.year.between(int(row.bsns_year), int(row.bsns_year) + 1)]
            dates.append(subset["rcept_dt"].min() if not subset.empty else pd.NaT)
    else:
        dates = [pd.NaT] * len(wide)
    wide["raw_date"] = dates
    fallback_month = {"11013": 6, "11012": 9, "11014": 12, "11011": 4}
    missing = wide["raw_date"].isna()
    wide.loc[missing, "raw_date"] = [
        pd.Timestamp(int(y) + (1 if str(r) == "11011" else 0), fallback_month.get(str(r), 12), 1)
        for y, r in zip(wide.loc[missing, "bsns_year"], wide.loc[missing, "reprt_code"])
    ]
    wide["date"] = [next_available_krx_date(dt, krx_dates, same_day=False) for dt in wide["raw_date"]]
    wide = wide.dropna(subset=["date"])

    for col in (
        "assets",
        "current_assets",
        "cash",
        "liabilities",
        "current_liabilities",
        "equity",
        "revenue",
        "operating_income",
        "net_income",
        "operating_cashflow",
        "short_borrowings",
        "long_borrowings",
    ):
        if col not in wide:
            wide[col] = np.nan
    wide["fin_debt_ratio"] = safe_div(wide["liabilities"], wide["equity"])
    wide["fin_current_ratio"] = safe_div(wide["current_assets"], wide["current_liabilities"])
    wide["fin_cash_to_assets"] = safe_div(wide["cash"], wide["assets"])
    wide["fin_operating_margin"] = safe_div(wide["operating_income"], wide["revenue"])
    wide["fin_net_margin"] = safe_div(wide["net_income"], wide["revenue"])
    wide["fin_ocf_to_assets"] = safe_div(wide["operating_cashflow"], wide["assets"])
    wide["fin_borrowings_to_assets"] = safe_div(
        wide["short_borrowings"].fillna(0) + wide["long_borrowings"].fillna(0), wide["assets"]
    )
    wide["fin_negative_equity"] = wide["equity"].lt(0).astype("int8")
    wide["fin_negative_ocf"] = wide["operating_cashflow"].lt(0).astype("int8")
    wide = wide.sort_values(["ticker", "date"])
    for col in ("assets", "liabilities", "equity", "revenue", "operating_income", "net_income", "operating_cashflow"):
        wide[f"fin_{col}_yoy"] = wide.groupby("ticker")[col].pct_change(4, fill_method=None)

    feature_cols = [c for c in wide.columns if c.startswith("fin_")]
    right_groups = {
        ticker: g[["date", *feature_cols]].sort_values("date").drop_duplicates("date", keep="last")
        for ticker, g in wide.groupby("ticker", sort=False)
    }
    blocks = []
    for ticker, g in base.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        right = right_groups.get(ticker)
        if right is not None and not right.empty:
            g = pd.merge_asof(g, right, on="date", direction="backward", allow_exact_matches=True)
        g["ticker"] = ticker
        blocks.append(g)
    return pd.concat(blocks, ignore_index=True) if blocks else base



# 교차자산 민감도·상호작용

def add_cross_asset_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        for alias in ANCHOR_ASSETS:
            asset_ret = f"xa_{alias}_ret_1"
            if asset_ret not in g.columns:
                continue
            cov = g["ret_1"].rolling(60, min_periods=30).cov(g[asset_ret])
            var = g[asset_ret].rolling(60, min_periods=30).var()
            g[f"contagion_beta_{alias}_60"] = cov / (var + EPS)
            g[f"contagion_corr_{alias}_60"] = g["ret_1"].rolling(60, min_periods=30).corr(g[asset_ret])
            crash = f"xa_{alias}_crash_1d"
            if crash in g.columns:
                g[f"contagion_exposure_{alias}"] = g[f"contagion_beta_{alias}_60"] * g[crash]
        g["ticker"] = ticker
        blocks.append(g)
    return pd.concat(blocks, ignore_index=True) if blocks else df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pairs = [
        ("global_equity_crash_breadth", "beta_60", "inter_global_crash_x_beta"),
        ("crypto_crash_breadth", "vol_20", "inter_crypto_crash_x_vol"),
        ("global_realestate_worst_ret_1", "fin_borrowings_to_assets", "inter_realestate_x_borrowing"),
        ("event_fed_within_next_3d", "vol_20", "inter_fomc_near_x_vol"),
        ("event_bok_within_next_3d", "value_foreign_net_sell_streak", "inter_bok_near_x_foreign_sell"),
        ("macro_usdkrw_chg_1", "value_foreign_net_to_value", "inter_fx_x_foreign_flow"),
        ("peer_crash3_ratio_ex_self", "downside_vol_20", "inter_peer_crash_x_downvol"),
        ("major_company_crash_ratio", "market_crash3_ratio", "inter_major_x_market_crash"),
        ("macro_us_10y_2y_spread", "fin_debt_ratio", "inter_curve_x_debt"),
        ("macro_bok_base_rate", "fin_borrowings_to_assets", "inter_bok_rate_x_borrowing"),
    ]
    for a, b, name in pairs:
        if a in out.columns and b in out.columns:
            out[name] = pd.to_numeric(out[a], errors="coerce") * pd.to_numeric(out[b], errors="coerce")
    return out



# 시장 베타·횡단면 순위·달력

def add_market_beta_and_ranks(df: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        cov = g["ret_1"].rolling(60, min_periods=30).cov(g["market_median_ret"])
        var = g["market_median_ret"].rolling(60, min_periods=30).var()
        g["beta_60"] = cov / (var + EPS)
        g["market_corr_60"] = g["ret_1"].rolling(60, min_periods=30).corr(g["market_median_ret"])
        g["idio_ret_1"] = g["ret_1"] - g["beta_60"] * g["market_median_ret"]
        g["idio_vol_60"] = g["idio_ret_1"].rolling(60, min_periods=30).std(ddof=0) * np.sqrt(252)
        g["ticker"] = ticker
        blocks.append(g)
    out = pd.concat(blocks, ignore_index=True) if blocks else df

    rank_cols = (
        "ret_1",
        "ret_5",
        "ret_20",
        "vol_20",
        "downside_vol_20",
        "drawdown_60",
        "log_market_cap",
        "turnover_value_pct",
        "value_foreign_net_to_value",
        "value_institution_net_to_value",
        "smart_money_to_value",
        "foreign_ownership_chg_20",
        "short_volume_to_volume",
        "short_balance_value_to_mcap",
        "per",
        "pbr",
        "peer_crash3_ratio_ex_self",
    )
    for col in rank_cols:
        if col in out.columns:
            out[f"cs_rank_{col}"] = out.groupby(["date", "market"])[col].rank(pct=True, method="average")

    out["weekday"] = out["date"].dt.weekday.astype("int8")
    out["month"] = out["date"].dt.month.astype("int8")
    out["quarter"] = out["date"].dt.quarter.astype("int8")
    out["is_month_end"] = out["date"].dt.is_month_end.astype("int8")
    out["is_quarter_end"] = out["date"].dt.is_quarter_end.astype("int8")
    return out



# 라벨

def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    blocks = []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = pd.to_numeric(g["close"], errors="coerce")
        low = pd.to_numeric(g["low"], errors="coerce")
        high = pd.to_numeric(g["high"], errors="coerce")
        market_ret = pd.to_numeric(g["market_median_ret"], errors="coerce")
        for h in LABEL_HORIZONS:
            g[f"future_min_close_ret_{h}"] = future_min(close, h) / close - 1
            g[f"future_min_low_ret_{h}"] = future_min(low, h) / close - 1
            g[f"future_max_high_ret_{h}"] = future_max(high, h) / close - 1
            g[f"future_close_ret_{h}"] = close.shift(-h) / close - 1
            label = g[f"future_min_low_ret_{h}"].le(ABS_CRASH_THRESHOLDS[h]).astype("float32")
            label[g[f"future_min_low_ret_{h}"].isna()] = np.nan
            g[f"label_abs_crash_{h}"] = label

            market_future = market_ret.shift(-1)[::-1].rolling(h, min_periods=h).sum()[::-1]
            g[f"future_market_ret_{h}"] = market_future
            g[f"future_idio_ret_{h}"] = g[f"future_close_ret_{h}"] - market_future
            idio_label = g[f"future_idio_ret_{h}"].le(IDIO_CRASH_THRESHOLDS[h]).astype("float32")
            idio_label[g[f"future_idio_ret_{h}"].isna()] = np.nan
            g[f"label_idio_crash_{h}"] = idio_label
        g["ticker"] = ticker
        blocks.append(g)
    return pd.concat(blocks, ignore_index=True) if blocks else df



# 피처 그룹 분류

def build_feature_catalog(df: pd.DataFrame) -> dict[str, list[str]]:
    excluded_exact = {
        "date",
        "ticker",
        "name",
        "market",
        "industry_code",
        "industry_name",
        "feature_ready",
        "history_days",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trading_value",
        "market_cap",
        "listed_shares",
    }
    excluded_prefixes = (
        "label_",
        "future_",
    )
    numeric = [
        c
        for c in df.select_dtypes(include=[np.number, "bool"]).columns
        if c not in excluded_exact and not c.startswith(excluded_prefixes)
    ]

    def group_of(col: str) -> str:
        if col.startswith(("ret_", "log_ret", "logret", "up_", "down_", "gap_", "intraday_", "range_", "close_location", "upper_wick", "lower_wick", "ma_ratio", "ema_ratio", "atr_", "rsi_", "dist_high", "dist_low", "days_since", "drawdown")):
            return "price_trend"
        if col.startswith(("vol_", "downside_vol", "skew_", "kurt_", "idio_vol", "market_corr")):
            return "volatility_tail"
        if col.startswith(("volume_", "value_ratio", "value_z", "amihud", "turnover_")):
            return "liquidity_volume"
        if any(x in col for x in ("foreign", "institution", "individual", "other_corp", "smart_money", "joint_buy", "joint_sell")):
            return "investor_flow"
        if col.startswith("short_") or "short_balance" in col:
            return "short_selling"
        if col.startswith(("per", "pbr", "eps", "bps", "dps", "dividend", "log_market_cap")):
            return "valuation_size"
        if col.startswith("news_"):
            return "news_sentiment_attention"
        if col.startswith("dart_"):
            return "dart_disclosure"
        if col.startswith("fin_"):
            return "financial_statement"
        if col.startswith("xa_") or col.startswith(("global_equity", "crypto_", "global_realestate")):
            if any(x in col for x in ("reit", "realestate", "homebuilder")):
                return "realestate_global"
            if any(x in col for x in ("btc", "eth", "crypto")):
                return "crypto_contagion"
            return "global_market_contagion"
        if col.startswith("krxidx_"):
            if any(x in col for x in ("건설", "부동산", "리츠")):
                return "realestate_korea"
            return "krx_sector_index"
        if col.startswith("major_"):
            return "major_company_contagion"
        if col.startswith(("peer_", "industry_")):
            return "industry_peer_contagion"
        if col.startswith(("all_market_", "market_", "top50_")):
            return "market_breadth"
        if col.startswith("macro_"):
            if any(x in col for x in ("주택", "아파트", "전세", "housing")):
                return "realestate_korea"
            return "macro_rates_fx"
        if col.startswith("event_"):
            return "policy_event"
        if col.startswith("contagion_"):
            return "rolling_contagion_sensitivity"
        if col.startswith("inter_"):
            return "interaction"
        if col.startswith("cs_rank_"):
            return "cross_sectional_rank"
        if col in {"weekday", "month", "quarter", "is_month_end", "is_quarter_end"}:
            return "calendar"
        if col.startswith(("beta_", "relative_ret", "idio_ret")):
            return "market_relative"
        return "other_numeric"

    groups: dict[str, list[str]] = {}
    for col in numeric:
        groups.setdefault(group_of(col), []).append(col)
    groups = {k: sorted(v) for k, v in sorted(groups.items()) if v}
    return groups



# 봉인 분리

def split_and_seal(df: pd.DataFrame, catalog: dict[str, list[str]]) -> None:
    dates = pd.DatetimeIndex(sorted(df["date"].dropna().unique()))
    required = LABEL_TAIL_DAYS + SEAL_COUNT * SEAL_DAYS + 260
    if len(dates) < required:
        raise RuntimeError(f"거래일이 너무 적습니다: {len(dates)} < {required}")

    tail_dates = dates[-LABEL_TAIL_DAYS:]
    seal_area_end = len(dates) - LABEL_TAIL_DAYS
    seal_area_start = seal_area_end - SEAL_COUNT * SEAL_DAYS
    development_dates = dates[:seal_area_start]
    seal_dates: dict[str, pd.DatetimeIndex] = {}
    for i in range(SEAL_COUNT):
        start = seal_area_start + i * SEAL_DAYS
        seal_dates[f"S{i:02d}"] = dates[start : start + SEAL_DAYS]

    # 학습 파일에는 봉인과 label tail을 절대 넣지 않는다.
    development = df[df["date"].isin(development_dates)].copy()
    development = development[development["feature_ready"].eq(1)]
    dev_path = DEVELOPMENT_DIR / "training_dataset.parquet"
    atomic_parquet(development.reset_index(drop=True), dev_path)

    manifests = []
    for seal_id, sd in seal_dates.items():
        seal_dir = SEALED_DIR / seal_id
        seal_dir.mkdir(parents=True, exist_ok=True)
        sealed = df[df["date"].isin(sd)].copy()
        sealed["seal_id"] = seal_id
        sealed["sealed_do_not_train_or_tune"] = 1
        path = seal_dir / "data.parquet"
        atomic_parquet(sealed.reset_index(drop=True), path)
        manifests.append(
            {
                "seal_id": seal_id,
                "start_date": sd.min(),
                "end_date": sd.max(),
                "trading_days": len(sd),
                "rows": len(sealed),
                "path": str(path),
                "sha256": sha256_file(path),
                "status": "SEALED_DO_NOT_TRAIN_OR_TUNE",
            }
        )

    tail = df[df["date"].isin(tail_dates)].copy()
    tail["seal_id"] = "LABEL_TAIL"
    tail["sealed_do_not_train_or_tune"] = 1
    tail_path = SEALED_DIR / "label_tail.parquet"
    atomic_parquet(tail.reset_index(drop=True), tail_path)

    catalog_path = META_DIR / "feature_catalog.json"
    write_json(catalog, catalog_path)
    flat_features = sorted({c for cols in catalog.values() for c in cols})
    feature_hash = hashlib.sha256("\n".join(flat_features).encode("utf-8")).hexdigest()
    manifest = {
        "policy": "4 x 15 KRX trading-day seals; seals never used for feature/model selection",
        "target_label": "label_abs_crash_20",
        "development": {
            "start_date": development_dates.min(),
            "end_date": development_dates.max(),
            "trading_days": len(development_dates),
            "rows": len(development),
            "path": str(dev_path),
            "sha256": sha256_file(dev_path),
        },
        "seals": manifests,
        "label_tail": {
            "start_date": tail_dates.min(),
            "end_date": tail_dates.max(),
            "trading_days": len(tail_dates),
            "rows": len(tail),
            "path": str(tail_path),
            "sha256": sha256_file(tail_path),
        },
        "feature_count": len(flat_features),
        "feature_list_sha256": feature_hash,
        "feature_catalog_path": str(catalog_path),
    }
    write_json(manifest, SEALED_DIR / "seal_manifest.json")



# 메인

def main() -> None:
    LOGGER.info("1/12 원천 주식 로드")
    raw = load_stock_raw()
    industry = load_industry_map()
    raw = raw.merge(industry, on="ticker", how="left")
    raw["industry_code"] = raw.get("industry_code", "UNKNOWN").fillna("UNKNOWN")

    LOGGER.info("2/12 종목 자체 시계열 피처")
    df = build_stock_features(raw)
    krx_dates = pd.DatetimeIndex(sorted(df["date"].unique()))

    LOGGER.info("3/12 시장·업종 breadth")
    df = add_breadth_features(df)

    LOGGER.info("4/12 주요 타 회사 급락")
    df = add_major_company_features(df)

    LOGGER.info("5/12 해외 증시·코인·글로벌 부동산")
    cross = load_cross_asset_features(krx_dates)
    krxidx = load_krx_index_features(krx_dates)
    global_daily = cross.merge(krxidx, on="date", how="outer").sort_values("date")

    LOGGER.info("6/12 FOMC·한국은행·거시·주택가격")
    events = add_policy_event_features(krx_dates)
    macro = load_macro_features(krx_dates)
    global_daily = global_daily.merge(events, on="date", how="left").merge(macro, on="date", how="left")
    df = df.merge(global_daily, on="date", how="left")

    LOGGER.info("7/13 뉴스 기사량·위험어·감성")
    df = add_news_features(df, krx_dates)

    LOGGER.info("8/13 DART 공시")
    df = add_disclosure_features(df, krx_dates)

    LOGGER.info("9/13 재무제표")
    df = add_financial_features(df, krx_dates)

    LOGGER.info("10/13 시장 베타·횡단면 순위")
    df = add_market_beta_and_ranks(df)

    LOGGER.info("11/13 시장 간 전염 민감도·상호작용")
    df = add_cross_asset_sensitivity(df)
    df = add_interaction_features(df)

    LOGGER.info("12/13 급락 라벨")
    df["feature_ready"] = df["history_days"].ge(MIN_HISTORY_DAYS).astype("int8")
    df = add_labels(df)
    df = df.replace([np.inf, -np.inf], np.nan).sort_values(["date", "ticker"])

    # 전체 중간 결과. 이것은 디버깅용이며 03은 development/sealed 파일만 읽는다.
    full_path = PROCESSED_DIR / "full_feature_dataset.parquet"
    atomic_parquet(df.reset_index(drop=True), full_path)

    LOGGER.info("13/13 15거래일 × 4 봉인 분리")
    catalog = build_feature_catalog(df)
    split_and_seal(df, catalog)

    feature_count = sum(len(v) for v in catalog.values())
    LOGGER.info("완료: %d행, %d개 피처, %d개 그룹", len(df), feature_count, len(catalog))
    피쳐_품질_보고서_생성(df, catalog)
    결과압축_생성("02_피쳐생성_봉인")
    print("\n다음 실행: python 03_이탈테스트_봉인인증.py")
    print(f"개발 데이터: {DEVELOPMENT_DIR / 'training_dataset.parquet'}")
    print(f"봉인 목록: {SEALED_DIR / 'seal_manifest.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        try:
            결과압축_생성("02_피쳐생성_봉인")
        finally:
            raise
