from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import ProjectPaths, load_baskets
from ..io_utils import normalize_date, normalize_ticker
from .common import drawdown, downside_semivol, event_decay, rolling_beta, rolling_z, safe_div

EVENT_CATEGORIES = [
    "earnings", "contract_order", "capital_increase", "bond_issue", "merger_split",
    "major_shareholder", "treasury_stock", "litigation", "regulatory", "governance",
    "audit_opinion", "business_suspension", "investment_disposal",
]

EVENT_RULES: dict[str, list[str]] = {
    "earnings": ["영업실적", "잠정실적", "매출액", "손익구조", "사업보고서", "분기보고서", "반기보고서"],
    "contract_order": ["단일판매", "공급계약", "수주", "계약체결", "계약해지"],
    "capital_increase": ["유상증자", "무상증자", "감자", "신주인수권"],
    "bond_issue": ["전환사채", "교환사채", "신주인수권부사채", "회사채"],
    "merger_split": ["합병", "분할", "영업양수", "영업양도"],
    "major_shareholder": ["최대주주", "주식등의대량보유", "임원ㆍ주요주주"],
    "treasury_stock": ["자기주식", "자사주", "소각"],
    "litigation": ["소송", "횡령", "배임"],
    "regulatory": ["상장폐지", "불성실공시", "거래정지", "관리종목"],
    "governance": ["대표이사", "임원", "경영권", "주주총회"],
    "audit_opinion": ["감사의견", "감사보고서", "내부회계관리제도"],
    "business_suspension": ["영업정지", "생산중단", "부도", "회생"],
    "investment_disposal": ["타법인주식", "출자", "유형자산", "투자", "처분"],
}

NEGATIVE_EVENTS = {"litigation", "regulatory", "audit_opinion", "business_suspension"}
POSITIVE_EVENTS = {"contract_order", "earnings"}


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _first(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _streak(series: pd.Series) -> pd.Series:
    signs = np.sign(pd.to_numeric(series, errors="coerce").fillna(0))
    groups = signs.ne(signs.shift()).cumsum()
    return (signs.groupby(groups).cumcount() + 1) * signs


def _legacy_enrichment(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.project.parent / "crashwatch_ai_data" / "development" / "training_dataset.parquet"
    if not path.exists():
        return pd.DataFrame()
    available = set(pq.ParquetFile(path).schema_arrow.names)
    candidates = [
        "date", "ticker", "trading_value", "market_cap", "per", "pbr", "dividend_yield",
        "foreign_ownership_pct", "short_volume", "short_total_buy_volume",
        "short_volume_ratio_pct", "short_balance_shares", "short_balance_value",
        "foreign_net_buy_value", "institution_net_buy_value", "individual_net_buy_value",
    ]
    cols = [c for c in candidates if c in available]
    out = pd.read_parquet(path, columns=cols)
    out = normalize_date(out)
    out["ticker"] = normalize_ticker(out["ticker"])
    return out.drop_duplicates(["date", "ticker"], keep="last")


def _prepare_raw(paths: ProjectPaths) -> pd.DataFrame:
    raw = _read(paths.raw_dual / "krx_ticker_timeseries.parquet")
    if raw.empty:
        raise FileNotFoundError("KRX ticker OHLCV is required")
    raw = normalize_date(raw)
    raw["ticker"] = normalize_ticker(raw["ticker"])
    baskets = load_baskets(paths)
    raw = raw.drop(columns=[c for c in ["name", "bucket", "market", "role"] if c in raw], errors="ignore")
    raw = raw.merge(baskets[["ticker", "name", "bucket", "market", "role"]], on="ticker", how="inner")
    legacy = _legacy_enrichment(paths)
    if not legacy.empty:
        raw = raw.merge(legacy, on=["date", "ticker"], how="left", suffixes=("", "_legacy"))
    return raw.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")


def _base_features(raw: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, g in raw.groupby("ticker", sort=False):
        g = g.sort_values("date")
        out = g[["date", "ticker", "name", "bucket", "market", "role"]].copy()
        close = pd.to_numeric(g["close"], errors="coerce")
        open_ = pd.to_numeric(g["open"], errors="coerce")
        high = pd.to_numeric(g["high"], errors="coerce")
        low = pd.to_numeric(g["low"], errors="coerce")
        volume = pd.to_numeric(g["volume"], errors="coerce")
        ret1 = close.pct_change(fill_method=None)
        out["t_price_ret_1"] = ret1
        for window in (3, 5, 10, 20, 60, 120):
            out[f"t_price_ret_{window}"] = close.pct_change(window, fill_method=None)
        for window in (20, 60, 120):
            average = close.rolling(window, min_periods=window).mean()
            out[f"t_price_ma_gap_{window}"] = safe_div(close, average) - 1
            out[f"t_price_drawdown_{window}"] = drawdown(close, window)
        out["t_price_gap_open"] = safe_div(open_, close.shift(1)) - 1
        out["t_price_intraday_return"] = safe_div(close, open_) - 1
        out["t_vol_realized_5"] = ret1.rolling(5, min_periods=5).std()
        out["t_vol_realized_20"] = ret1.rolling(20, min_periods=20).std()
        out["t_vol_realized_60"] = ret1.rolling(60, min_periods=60).std()
        out["t_tail_downside_20"] = downside_semivol(ret1, 20)
        out["t_tail_skew_20"] = ret1.rolling(20, min_periods=20).skew()
        out["t_tail_kurt_60"] = ret1.rolling(60, min_periods=60).kurt()
        out["t_vol_range_1"] = safe_div(high - low, close.shift(1))

        market_cap = _first(g, ["market_cap"])
        trading_value = _first(g, ["trading_value"])
        trading_value = trading_value.where(trading_value.notna(), close * volume)
        out["t_liq_turnover"] = safe_div(trading_value, market_cap)
        out["t_liq_volume_z20"] = rolling_z(volume, 20)
        out["t_liq_value_z20"] = rolling_z(trading_value, 20)
        out["t_liq_down_day_turnover"] = out["t_liq_turnover"].where(ret1 < 0, 0.0)

        foreign = _first(g, ["foreign_net_buy_value"])
        institution = _first(g, ["institution_net_buy_value"])
        individual = _first(g, ["individual_net_buy_value"])
        flow_total = pd.concat([foreign.abs(), institution.abs(), individual.abs()], axis=1).sum(axis=1, min_count=1)
        out["t_foreign_net_buy_value"] = foreign
        out["t_institution_net_buy_value"] = institution
        out["t_individual_net_buy_value"] = individual
        out["t_foreign_net_buy_ratio"] = safe_div(foreign, flow_total)
        out["t_institution_net_buy_ratio"] = safe_div(institution, flow_total)
        out["t_individual_net_buy_ratio"] = safe_div(individual, flow_total)
        out["t_foreign_flow_sum_5"] = foreign.rolling(5, min_periods=5).sum()
        out["t_foreign_flow_sum_20"] = foreign.rolling(20, min_periods=20).sum()
        out["t_institution_flow_sum_5"] = institution.rolling(5, min_periods=5).sum()
        out["t_institution_flow_sum_20"] = institution.rolling(20, min_periods=20).sum()
        out["t_foreign_flow_zscore_20"] = rolling_z(foreign, 20)
        out["t_institution_flow_zscore_20"] = rolling_z(institution, 20)
        out["t_foreign_flow_streak"] = _streak(foreign).where(foreign.notna())
        out["t_institution_flow_streak"] = _streak(institution).where(institution.notna())
        ownership = _first(g, ["foreign_ownership_pct"])
        out["t_foreign_ownership_rate"] = ownership
        out["t_foreign_ownership_change_5"] = ownership.diff(5)
        out["t_foreign_ownership_change_20"] = ownership.diff(20)

        short_volume = _first(g, ["short_volume"])
        short_value = pd.Series(np.nan, index=g.index, dtype=float)
        short_ratio = _first(g, ["short_volume_ratio_pct"]) / 100.0
        short_balance_value = _first(g, ["short_balance_value"])
        short_balance_shares = _first(g, ["short_balance_shares"])
        out["t_short_volume"] = short_volume
        out["t_short_value"] = short_value
        out["t_short_volume_ratio"] = short_ratio.where(short_ratio.notna(), safe_div(short_volume, volume))
        out["t_short_value_ratio"] = safe_div(short_value, trading_value)
        out["t_short_balance_value"] = short_balance_value
        out["t_short_balance_shares"] = short_balance_shares
        out["t_short_balance_to_market_cap"] = safe_div(short_balance_value, market_cap)
        out["t_short_balance_change_5"] = short_balance_value.pct_change(5, fill_method=None)
        out["t_short_balance_change_20"] = short_balance_value.pct_change(20, fill_method=None)
        out["t_short_ratio_zscore_20"] = rolling_z(out["t_short_volume_ratio"], 20)
        out["t_short_balance_slope_20"] = out["t_short_balance_to_market_cap"].diff(20) / 20
        out["t_short_squeeze_proxy"] = out["t_short_balance_change_20"] * out["t_price_ret_20"].clip(lower=0)

        per = _first(g, ["per"])
        pbr = _first(g, ["pbr"])
        dividend = _first(g, ["dividend_yield"])
        out["t_market_cap"] = market_cap
        out["t_log_market_cap"] = np.log1p(market_cap.clip(lower=0))
        out["t_per"] = per.replace([np.inf, -np.inf], np.nan)
        out["t_pbr"] = pbr.replace([np.inf, -np.inf], np.nan)
        out["t_dividend_yield"] = dividend
        out["t_per_missing"] = out["t_per"].isna().astype("int8")
        out["t_pbr_missing"] = out["t_pbr"].isna().astype("int8")
        out["t_dividend_missing"] = out["t_dividend_yield"].isna().astype("int8")
        parts.append(out)
    features = pd.concat(parts, ignore_index=True, sort=False)
    features["t_market_cap_rank_pct"] = features.groupby("date")["t_market_cap"].rank(pct=True)
    features["t_per_industry_zscore"] = features.groupby(["date", "bucket"])["t_per"].transform(lambda s: (s - s.mean()) / s.std())
    features["t_pbr_industry_zscore"] = features.groupby(["date", "bucket"])["t_pbr"].transform(lambda s: (s - s.mean()) / s.std())
    features["t_size_bucket"] = np.floor(features["t_market_cap_rank_pct"] * 5).clip(upper=4)
    features["t_value_distress_score"] = (
        -features["t_market_cap_rank_pct"] + features["t_per_industry_zscore"].clip(lower=0)
        + features["t_pbr_industry_zscore"].clip(lower=0)
    )
    return features


def _add_peer_features(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    grouped = out.groupby(["date", "bucket"], sort=False)["t_price_ret_1"]
    count = grouped.transform("count")
    total = grouped.transform("sum")
    out["t_peer_leave_one_out_ret_1"] = safe_div(total - out["t_price_ret_1"], count - 1)
    out["t_peer_relative_ret_1"] = out["t_price_ret_1"] - out["t_peer_leave_one_out_ret_1"]
    crashes = out["t_price_ret_1"].le(-0.03).astype(int)
    out["t_peer_down_3pct_count"] = crashes.groupby([out["date"], out["bucket"]]).transform("sum") - crashes
    out["t_peer_bucket_dispersion"] = grouped.transform("std")
    out = out.sort_values(["ticker", "date"])
    out["t_peer_lead_lag_1"] = out.groupby("ticker")["t_peer_leave_one_out_ret_1"].shift(1)
    out["t_peer_lead_lag_3"] = out.groupby("ticker")["t_peer_leave_one_out_ret_1"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    return out


def _classify_report(report_name: str) -> str | None:
    text = str(report_name)
    for category, keywords in EVENT_RULES.items():
        if any(keyword in text for keyword in keywords):
            return category
    return None


def _add_disclosures(paths: ProjectPaths, features: pd.DataFrame) -> pd.DataFrame:
    events = _read(paths.raw_dual / "dart_disclosures.parquet")
    out_parts: list[pd.DataFrame] = []
    if not events.empty:
        events = normalize_date(events)
        events["ticker"] = normalize_ticker(events["ticker"])
        events["category"] = events["report_nm"].map(_classify_report)
        events = events.loc[events["category"].notna()]
    for ticker, block in features.groupby("ticker", sort=False):
        block = block.sort_values("date").copy()
        ticker_events = events.loc[events["ticker"].eq(ticker)].copy() if not events.empty else pd.DataFrame()
        trading_dates = pd.DatetimeIndex(block["date"])
        if not ticker_events.empty:
            positions = trading_dates.searchsorted(pd.DatetimeIndex(ticker_events["date"]), side="right")
            ticker_events["effective_date"] = [trading_dates[pos] if pos < len(trading_dates) else pd.NaT for pos in positions]
            ticker_events = ticker_events.dropna(subset=["effective_date"])
        total5 = pd.Series(0.0, index=block.index)
        total20 = pd.Series(0.0, index=block.index)
        negative20 = pd.Series(0.0, index=block.index)
        positive20 = pd.Series(0.0, index=block.index)
        for category in EVENT_CATEGORIES:
            counts = ticker_events.loc[ticker_events["category"].eq(category)].groupby("effective_date").size() if not ticker_events.empty else pd.Series(dtype=float)
            daily = block["date"].map(counts).fillna(0).astype(float)
            count5 = daily.rolling(5, min_periods=1).sum()
            count20 = daily.rolling(20, min_periods=1).sum()
            last_event = block["date"].where(daily.gt(0)).ffill()
            days_since = (block["date"] - last_event).dt.days
            block[f"t_event_{category}_count_5"] = count5
            block[f"t_event_{category}_count_20"] = count20
            block[f"t_event_{category}_decay"] = event_decay(days_since, 5.0)
            total5 += count5
            total20 += count20
            if category in NEGATIVE_EVENTS:
                negative20 += count20
            if category in POSITIVE_EVENTS:
                positive20 += count20
        block["t_event_total_count_5"] = total5
        block["t_event_total_count_20"] = total20
        block["t_event_negative_count_20"] = negative20
        block["t_event_positive_count_20"] = positive20
        block["t_event_risk_score"] = negative20 - 0.5 * positive20
        block["t_dart_source_available"] = int(not events.empty and ticker in set(events["ticker"]))
        block["t_dart_crawl_failed"] = int(events.empty)
        out_parts.append(block)
    return pd.concat(out_parts, ignore_index=True, sort=False)


def _add_macro_links(features: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    out = features.merge(universe, on="date", how="left", validate="many_to_one")
    out = out.sort_values(["ticker", "date"])
    links = {"usdkrw": "u_usdkrw_ret_1", "kospi": "u_kospi_ret_1", "vix": "u_vix_change_1"}
    for alias, column in links.items():
        if column not in out:
            continue
        out[f"t_link_{alias}_beta_60"] = out.groupby("ticker", group_keys=False).apply(
            lambda g: rolling_beta(g["t_price_ret_1"], g[column], 60), include_groups=False,
        ).reset_index(level=0, drop=True)
        out[f"t_link_{alias}_corr_60"] = out.groupby("ticker", group_keys=False).apply(
            lambda g: g["t_price_ret_1"].rolling(60, min_periods=30).corr(g[column]), include_groups=False,
        ).reset_index(level=0, drop=True)
    out["t_price_trend_x_high_vol"] = out["t_price_ret_20"] * out.get("u_regime_high_vol")
    out["t_foreign_sell_x_market_stress"] = (-out["t_foreign_net_buy_ratio"]).clip(lower=0) * out.get("u_regime_risk_off")
    out["t_short_rise_x_negative_momentum"] = out["t_short_balance_change_20"].clip(lower=0) * (-out["t_price_ret_20"]).clip(lower=0)
    out["t_liquidity_drop_x_drawdown"] = (-out["t_liq_value_z20"]).clip(lower=0) * (-out["t_price_drawdown_60"]).clip(lower=0)
    out["t_macro_beta_x_fx_shock"] = out.get("t_link_usdkrw_beta_60") * out.get("u_regime_fx_shock")
    out["t_peer_crash_x_low_liquidity"] = out["t_peer_down_3pct_count"] * (-out["t_liq_value_z20"]).clip(lower=0)
    out["t_disclosure_risk_x_high_vol"] = out["t_event_risk_score"] * out.get("u_regime_high_vol")
    universe_columns = [c for c in universe.columns if c != "date"]
    return out.drop(columns=universe_columns, errors="ignore")


def build_ticker_features(paths: ProjectPaths, universe_features: pd.DataFrame) -> pd.DataFrame:
    out = _base_features(_prepare_raw(paths))
    out = _add_peer_features(out)
    out = _add_disclosures(paths, out)
    out = _add_macro_links(out, universe_features)
    return out.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
