from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ProjectPaths
from ..io_utils import normalize_date, read_parquet_tree
from .common import drawdown, downside_semivol, first_numeric, rolling_z, safe_div
from .research import build_research_universe_features

LOGGER = logging.getLogger(__name__)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _index_features(indices: pd.DataFrame) -> pd.DataFrame:
    if indices.empty:
        return pd.DataFrame(columns=["date"])
    out = normalize_date(indices)
    close_cols = [c for c in out.columns if c.endswith("_close")]
    for col in close_cols:
        alias = col.removesuffix("_close")
        close = pd.to_numeric(out[col], errors="coerce")
        ret1 = close.pct_change(fill_method=None)
        out[f"u_market_{alias}_ret_1"] = ret1
        out[f"u_market_{alias}_ret_5"] = close.pct_change(5, fill_method=None)
        out[f"u_market_{alias}_ret_20"] = close.pct_change(20, fill_method=None)
        out[f"u_vol_{alias}_20"] = ret1.rolling(20, min_periods=10).std()
        out[f"u_tail_{alias}_downside_20"] = downside_semivol(ret1, 20)
        out[f"u_market_{alias}_drawdown_60"] = drawdown(close, 60)
    return out[[c for c in out.columns if c == "date" or c.startswith("u_")]]


def _global_features(global_assets: pd.DataFrame) -> pd.DataFrame:
    if global_assets.empty:
        return pd.DataFrame(columns=["date"])
    raw = normalize_date(global_assets).sort_values("date")
    # 한국 장 개장 시점에 확실히 관측 가능한 값만 사용하기 위해 모든 해외 일봉을 1행 지연한다.
    value_cols = [c for c in raw.columns if c != "date"]
    raw[value_cols] = raw[value_cols].shift(1)
    out = raw[["date"]].copy()
    for col in [c for c in raw.columns if c.endswith("_close") or c.endswith("_adj_close")]:
        alias = col.rsplit("_", 1)[0]
        s = pd.to_numeric(raw[col], errors="coerce")
        prefix = "u_crypto_" if alias in {"btc", "eth"} else "u_global_"
        out[f"{prefix}{alias}_ret_1"] = s.pct_change(fill_method=None)
        out[f"{prefix}{alias}_ret_5"] = s.pct_change(5, fill_method=None)
        out[f"{prefix}{alias}_vol_20"] = s.pct_change(fill_method=None).rolling(20, min_periods=10).std()
        out[f"{prefix}{alias}_drawdown_60"] = drawdown(s, 60)
    return out


def _cross_section_features(paths: ProjectPaths) -> pd.DataFrame:
    panel = read_parquet_tree(paths.raw_dual / "market_cross_section")
    if panel.empty:
        return pd.DataFrame(columns=["date"])
    panel = normalize_date(panel)
    close = first_numeric(panel, ["close", "종가"])
    ret = first_numeric(panel, ["return_pct", "등락률"]) / 100.0
    if ret.isna().all() and close.notna().any():
        panel = panel.sort_values(["ticker", "date"])
        ret = panel.groupby("ticker", sort=False)[close.name if close.name in panel.columns else "close"].pct_change(fill_method=None)
    volume = first_numeric(panel, ["volume", "거래량"])
    trading_value = first_numeric(panel, ["trading_value", "거래대금"])
    market_cap = first_numeric(panel, ["market_cap", "시가총액"])
    panel = panel.assign(_ret=ret, _volume=volume, _trading_value=trading_value, _market_cap=market_cap)

    def aggregate(g: pd.DataFrame) -> pd.Series:
        r = g["_ret"]
        tv = g["_trading_value"]
        cap = g["_market_cap"]
        return pd.Series({
            "u_breadth_advance_ratio": float((r > 0).mean()),
            "u_breadth_decline_ratio": float((r < 0).mean()),
            "u_breadth_crash_3pct_ratio": float((r <= -0.03).mean()),
            "u_breadth_crash_5pct_ratio": float((r <= -0.05).mean()),
            "u_dispersion_return_std": r.std(),
            "u_dispersion_return_iqr": r.quantile(0.75) - r.quantile(0.25),
            "u_liq_total_trading_value": tv.sum(min_count=1),
            "u_liq_median_turnover": safe_div(tv, cap).median(),
            "u_liq_top10_value_share": safe_div(tv.nlargest(min(10, len(tv))).sum(), tv.sum()),
        })

    out = panel.groupby("date", sort=True).apply(aggregate, include_groups=False).reset_index()
    for col in [c for c in out.columns if c.startswith("u_liq_")]:
        out[f"{col}_z20"] = rolling_z(out[col], 20)
    return out



def _market_flow_features(paths: ProjectPaths) -> pd.DataFrame:
    raw = _read(paths.raw_dual / "krx_market_flows.parquet")
    if raw.empty:
        return pd.DataFrame(columns=["date"])
    out = normalize_date(raw).sort_values("date")
    rename = {}
    for col in out.columns:
        if col == "date":
            continue
        if col.startswith("flow_"):
            rename[col] = f"u_flow_{col.removeprefix('flow_')}"
        elif col.startswith("short_"):
            rename[col] = f"u_short_{col.removeprefix('short_')}"
    out = out.rename(columns=rename)
    for col in [c for c in out.columns if c != "date"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}_z20"] = rolling_z(out[col], 20)
    return out

def _basket_aggregate_features(ticker_raw: pd.DataFrame) -> pd.DataFrame:
    if ticker_raw.empty:
        return pd.DataFrame(columns=["date"])
    df = normalize_date(ticker_raw)
    trading_value = first_numeric(df, ["trading_value"])
    market_cap = first_numeric(df, ["market_cap"])
    foreign = first_numeric(df, ["flow_value_외국인합계", "flow_value_foreign", "foreign_value"])
    institution = first_numeric(df, ["flow_value_기관합계", "flow_value_institution", "institution_value"])
    individual = first_numeric(df, ["flow_value_개인", "flow_value_individual", "individual_value"])
    short_value = first_numeric(df, ["short_balance_잔고금액", "short_balance_short_balance_value", "short_balance_value"])
    short_volume = first_numeric(df, ["short_volume_공매도", "short_volume_short_volume", "short_volume"])
    df = df.assign(_tv=trading_value, _cap=market_cap, _foreign=foreign, _institution=institution, _individual=individual, _short_value=short_value, _short_volume=short_volume)
    out = df.groupby("date", as_index=False).agg(
        u_liq_basket_trading_value=("_tv", "sum"),
        u_flow_basket_foreign=("_foreign", "sum"),
        u_flow_basket_institution=("_institution", "sum"),
        u_flow_basket_individual=("_individual", "sum"),
        u_short_basket_balance_value=("_short_value", "sum"),
        u_short_basket_volume=("_short_volume", "sum"),
        _basket_cap=("_cap", "sum"),
    )
    out["u_short_basket_balance_to_cap"] = safe_div(out["u_short_basket_balance_value"], out["_basket_cap"])
    out = out.drop(columns="_basket_cap")
    for col in [c for c in out.columns if c != "date"]:
        out[f"{col}_z20"] = rolling_z(out[col], 20)
    return out


def _macro_features(paths: ProjectPaths) -> pd.DataFrame:
    ecos = _read(paths.raw_dual / "ecos_macro.parquet")
    kosis = _read(paths.raw_dual / "kosis_macro.parquet")
    frames: list[pd.DataFrame] = []
    if not ecos.empty:
        e = normalize_date(ecos).sort_values("date")
        cols = [c for c in e.columns if c != "date"]
        e[cols] = e[cols].shift(1)
        e = e.rename(columns={c: f"u_macro_{c}" for c in cols})
        frames.append(e)
    if not kosis.empty:
        k = normalize_date(kosis).sort_values("date")
        cols = [c for c in k.columns if c != "date"]
        # 월간 통계의 실제 공표일 메타데이터가 없으면 보수적으로 31일 지연한다.
        k["date"] = k["date"] + pd.Timedelta(days=31)
        k = k.rename(columns={c: f"u_realestate_{c}" for c in cols})
        frames.append(k)
    if not frames:
        return pd.DataFrame(columns=["date"])
    out = frames[0]
    for frame in frames[1:]:
        out = pd.merge_asof(out.sort_values("date"), frame.sort_values("date"), on="date", direction="backward")
    return out


def build_universe_features(paths: ProjectPaths) -> pd.DataFrame:
    indices = _read(paths.raw_dual / "krx_indices.parquet")
    global_assets = _read(paths.raw_dual / "global_assets.parquet")
    ticker_raw = _read(paths.raw_dual / "krx_ticker_timeseries.parquet")
    frames = [
        _index_features(indices), _global_features(global_assets), _cross_section_features(paths),
        _market_flow_features(paths), _basket_aggregate_features(ticker_raw), _macro_features(paths),
        build_research_universe_features(paths),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError("이원화 raw 데이터가 없습니다. 먼저 01B_이원화_데이터_크롤링.py를 실행하세요.")
    date_source = max(frames, key=len)[["date"]].drop_duplicates().sort_values("date")
    out = date_source
    for frame in frames:
        out = pd.merge_asof(out.sort_values("date"), frame.sort_values("date"), on="date", direction="backward")
    # 시장 레짐 교호항
    market_ret = next((out[c] for c in out.columns if c == "u_market_kospi_ret_1"), pd.Series(np.nan, index=out.index))
    global_ret = next((out[c] for c in out.columns if c == "u_global_sp500_ret_1"), pd.Series(np.nan, index=out.index))
    breadth = out.get("u_breadth_advance_ratio", pd.Series(np.nan, index=out.index))
    vix = out.get("u_global_vix_ret_1", pd.Series(np.nan, index=out.index))
    out["u_interaction_global_x_market"] = global_ret * market_ret
    out["u_interaction_breadth_x_market"] = (breadth - 0.5) * market_ret
    out["u_interaction_vix_x_breadth"] = vix * (0.5 - breadth)
    return out.sort_values("date").drop_duplicates("date", keep="last")
