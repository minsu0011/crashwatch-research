from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..config import ProjectPaths, load_baskets, load_ticker_rules
from ..io_utils import normalize_date, normalize_ticker
from .common import drawdown, downside_semivol, event_decay, first_numeric, rolling_beta, rolling_z, safe_div
from .research import add_research_ticker_features

LOGGER = logging.getLogger(__name__)


def _read(path):
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _base_ticker_features(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=["date", "ticker"])
    df = normalize_date(raw)
    df["ticker"] = normalize_ticker(df["ticker"])
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")

    close = first_numeric(df, ["close", "종가"])
    high = first_numeric(df, ["high", "고가"])
    low = first_numeric(df, ["low", "저가"])
    open_ = first_numeric(df, ["open", "시가"])
    volume = first_numeric(df, ["volume", "거래량"])
    trading_value = first_numeric(df, ["trading_value", "거래대금"])
    market_cap = first_numeric(df, ["market_cap", "시가총액"])
    listed_shares = first_numeric(df, ["listed_shares", "상장주식수"])
    per = first_numeric(df, ["per", "PER"])
    pbr = first_numeric(df, ["pbr", "PBR"])
    dividend_yield = first_numeric(df, ["dividend_yield", "DIV"])
    foreign_rate = first_numeric(df, ["foreign_지분율", "foreign_foreign_ownership_rate", "foreign_ownership_rate"])
    foreign_flow = first_numeric(df, ["flow_value_외국인합계", "flow_value_foreign", "foreign_value"])
    institution_flow = first_numeric(df, ["flow_value_기관합계", "flow_value_institution", "institution_value"])
    individual_flow = first_numeric(df, ["flow_value_개인", "flow_value_individual", "individual_value"])
    short_volume = first_numeric(df, ["short_volume_공매도", "short_volume_short_volume", "short_volume"])
    short_ratio = first_numeric(df, ["short_volume_비중", "short_volume_short_ratio", "short_ratio"])
    short_balance_value = first_numeric(df, ["short_balance_잔고금액", "short_balance_short_balance_value", "short_balance_value"])
    short_balance_shares = first_numeric(df, ["short_balance_잔고수량", "short_balance_short_balance_shares", "short_balance_shares"])

    df = df.assign(
        _close=close, _high=high, _low=low, _open=open_, _volume=volume, _trading_value=trading_value,
        _market_cap=market_cap, _listed_shares=listed_shares, _per=per, _pbr=pbr, _dividend_yield=dividend_yield,
        _foreign_rate=foreign_rate, _foreign_flow=foreign_flow, _institution_flow=institution_flow,
        _individual_flow=individual_flow, _short_volume=short_volume, _short_ratio=short_ratio,
        _short_balance_value=short_balance_value, _short_balance_shares=short_balance_shares,
    )

    def build(g: pd.DataFrame) -> pd.DataFrame:
        out = g[[c for c in ["date", "ticker", "name", "bucket", "market", "role"] if c in g.columns]].copy()
        c = g["_close"]
        r1 = c.pct_change(fill_method=None)
        out["t_price_ret_1"] = r1
        for w in (3, 5, 10, 20, 60, 120):
            out[f"t_price_ret_{w}"] = c.pct_change(w, fill_method=None)
        for w in (20, 60, 120):
            ma = c.rolling(w, min_periods=max(5, w // 3)).mean()
            out[f"t_price_ma_gap_{w}"] = safe_div(c, ma) - 1.0
            out[f"t_price_drawdown_{w}"] = drawdown(c, w)
        out["t_price_gap_open"] = safe_div(g["_open"], c.shift(1)) - 1.0
        out["t_price_intraday_return"] = safe_div(c, g["_open"]) - 1.0
        out["t_vol_realized_5"] = r1.rolling(5, min_periods=3).std()
        out["t_vol_realized_20"] = r1.rolling(20, min_periods=10).std()
        out["t_vol_realized_60"] = r1.rolling(60, min_periods=20).std()
        out["t_tail_downside_20"] = downside_semivol(r1, 20)
        out["t_tail_skew_20"] = r1.rolling(20, min_periods=10).skew()
        out["t_tail_kurt_60"] = r1.rolling(60, min_periods=20).kurt()
        out["t_vol_range_1"] = safe_div(g["_high"] - g["_low"], c.shift(1))
        out["t_liq_turnover"] = safe_div(g["_trading_value"], g["_market_cap"])
        out["t_liq_volume_z20"] = rolling_z(g["_volume"], 20)
        out["t_liq_value_z20"] = rolling_z(g["_trading_value"], 20)
        out["t_liq_down_day_turnover"] = out["t_liq_turnover"].where(r1 < 0, 0.0)
        out["t_flow_foreign_to_value"] = safe_div(g["_foreign_flow"], g["_trading_value"])
        out["t_flow_institution_to_value"] = safe_div(g["_institution_flow"], g["_trading_value"])
        out["t_flow_individual_to_value"] = safe_div(g["_individual_flow"], g["_trading_value"])
        out["t_flow_foreign_20"] = g["_foreign_flow"].rolling(20, min_periods=5).sum()
        out["t_flow_institution_20"] = g["_institution_flow"].rolling(20, min_periods=5).sum()
        out["t_foreign_ownership_rate"] = g["_foreign_rate"]
        out["t_foreign_exhaustion_slope_20"] = g["_foreign_rate"].diff(20) / 20.0
        out["t_foreign_exhaustion_accel_5"] = out["t_foreign_exhaustion_slope_20"].diff(5)
        out["t_short_volume_ratio"] = g["_short_ratio"].where(g["_short_ratio"].notna(), safe_div(g["_short_volume"], g["_volume"]))
        out["t_short_volume_z40"] = rolling_z(g["_short_volume"], 40)
        out["t_short_balance_to_cap"] = safe_div(g["_short_balance_value"], g["_market_cap"])
        out["t_short_balance_slope_20"] = out["t_short_balance_to_cap"].diff(20) / 20.0
        out["t_value_per"] = g["_per"]
        out["t_value_pbr"] = g["_pbr"]
        out["t_value_dividend_yield"] = g["_dividend_yield"]
        out["t_size_log_market_cap"] = np.log1p(g["_market_cap"].clip(lower=0))
        out["t_size_log_listed_shares"] = np.log1p(g["_listed_shares"].clip(lower=0))
        out["t_interaction_short_x_vol"] = out["t_short_balance_to_cap"] * out["t_vol_realized_20"]
        out["t_interaction_flow_x_drawdown"] = out["t_flow_foreign_to_value"] * out["t_price_drawdown_60"]
        out["t_interaction_turnover_x_downside"] = out["t_liq_turnover"] * out["t_tail_downside_20"]
        for family in ("flow", "foreign", "short", "value"):
            cols = [c for c in out.columns if c.startswith(f"t_{family}_")]
            out[f"t_{family}_is_available"] = out[cols].notna().any(axis=1).astype("int8") if cols else 0
        return out

    parts = [build(g.copy()) for _, g in df.groupby("ticker", sort=False)]
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame(columns=["date", "ticker"])


def _add_peer_features(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or "bucket" not in features.columns:
        return features
    out = features.copy()
    ret_col = "t_price_ret_1"
    group = out.groupby(["date", "bucket"], sort=False)[ret_col]
    count = group.transform("count")
    total = group.transform("sum")
    out["t_peer_leave_one_out_ret_1"] = safe_div(total - out[ret_col], count - 1)
    out["t_peer_relative_ret_1"] = out[ret_col] - out["t_peer_leave_one_out_ret_1"]
    out["t_peer_down_3pct_count"] = out.assign(_crash=(out[ret_col] <= -0.03).astype(int)).groupby(["date", "bucket"])["_crash"].transform("sum") - (out[ret_col] <= -0.03).astype(int)
    out["t_peer_bucket_dispersion"] = out.groupby(["date", "bucket"])[ret_col].transform("std")
    out = out.sort_values(["ticker", "date"])
    out["t_peer_lead_lag_1"] = out.groupby("ticker")["t_peer_leave_one_out_ret_1"].shift(1)
    out["t_peer_lead_lag_3"] = out.groupby("ticker", group_keys=False)["t_peer_leave_one_out_ret_1"].transform(lambda s: s.rolling(3, min_periods=1).mean().shift(1))
    return out


def _event_daily(events: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return base
    out = base.copy()
    categories = ["capital_change", "contract_order", "governance", "buyback_dividend", "risk_legal", "earnings", "clinical_license", "facility_investment", "other"]
    if events.empty:
        for category in categories:
            out[f"t_event_{category}_count_20"] = 0.0
            out[f"t_event_{category}_decay"] = 0.0
        out["t_event_any_count_5"] = 0.0
        out["t_event_max_severity_20"] = 0.0
        out["t_event_is_available"] = 0
        return out
    ev = normalize_date(events)
    ev["ticker"] = normalize_ticker(ev["ticker"])
    # DART 목록 API는 접수 날짜만 제공하므로 장중/장후를 구분할 수 없다.
    # 누수를 피하기 위해 해당 접수일 다음 거래일부터 피처를 활성화한다.
    trading_dates = pd.DatetimeIndex(sorted(out["date"].dropna().unique()))
    event_dates = pd.DatetimeIndex(ev["date"])
    positions = trading_dates.searchsorted(event_dates, side="right")
    effective = [trading_dates[pos] if pos < len(trading_dates) else pd.NaT for pos in positions]
    ev["date"] = effective
    ev = ev.loc[ev["date"].notna()]
    ev["event_category"] = ev.get("event_category", "other").fillna("other")
    ev["event_severity"] = pd.to_numeric(ev.get("event_severity", 1), errors="coerce").fillna(1)
    daily = ev.groupby(["date", "ticker", "event_category"], as_index=False).agg(event_count=("event_category", "size"), event_severity=("event_severity", "max"))
    result_parts: list[pd.DataFrame] = []
    for ticker, g in out.groupby("ticker", sort=False):
        block = g.sort_values("date").copy()
        e = daily.loc[daily["ticker"].eq(ticker)]
        block["t_event_any_count_5"] = 0.0
        block["t_event_max_severity_20"] = 0.0
        for category in categories:
            ec = e.loc[e["event_category"].eq(category), ["date", "event_count", "event_severity"]]
            series = block[["date"]].merge(ec, on="date", how="left").fillna({"event_count": 0, "event_severity": 0})
            block[f"t_event_{category}_count_20"] = series["event_count"].rolling(20, min_periods=1).sum().to_numpy()
            event_dates = series["date"].where(series["event_count"] > 0).ffill()
            days_since = (series["date"] - event_dates).dt.days
            block[f"t_event_{category}_decay"] = event_decay(days_since, 5.0).to_numpy()
            block["t_event_any_count_5"] += series["event_count"].rolling(5, min_periods=1).sum().to_numpy()
            block["t_event_max_severity_20"] = np.maximum(block["t_event_max_severity_20"], series["event_severity"].rolling(20, min_periods=1).max().to_numpy())
        block["t_event_is_available"] = 1
        result_parts.append(block)
    return pd.concat(result_parts, ignore_index=True, sort=False)


def _add_macro_links(features: pd.DataFrame, universe: pd.DataFrame, rules: dict) -> pd.DataFrame:
    if features.empty or universe.empty:
        return features
    out = features.merge(universe, on="date", how="left", suffixes=("", "_u"))
    out = out.sort_values(["ticker", "date"])
    available = {c: c for c in universe.columns if c != "date"}
    for bucket, bucket_rules in rules.items():
        mask = out["bucket"].eq(bucket)
        for link in bucket_rules.get("links", []):
            candidates = [c for c in available if link in c and (c.endswith("ret_1") or "ret_1" in c)]
            if not candidates:
                continue
            x_col = candidates[0]
            for ticker, idx in out.loc[mask].groupby("ticker").groups.items():
                y = out.loc[idx, "t_price_ret_1"]
                x = out.loc[idx, x_col]
                out.loc[idx, f"t_link_{link}_beta_60"] = rolling_beta(y, x, 60).to_numpy()
                out.loc[idx, f"t_link_{link}_corr_60"] = y.rolling(60, min_periods=20).corr(x).to_numpy()
                out.loc[idx, f"t_interaction_{link}_x_foreign"] = (x * out.loc[idx, "t_flow_foreign_to_value"]).to_numpy()
    # 병합했던 universe 원본 열은 ticker namespace에 중복 저장하지 않는다.
    return out.drop(columns=[c for c in universe.columns if c != "date" and c in out.columns], errors="ignore")


def build_ticker_features(paths: ProjectPaths, universe_features: pd.DataFrame) -> pd.DataFrame:
    raw = _read(paths.raw_dual / "krx_ticker_timeseries.parquet")
    if raw.empty:
        raise FileNotFoundError("종목 raw 시계열이 없습니다. 먼저 01B_이원화_데이터_크롤링.py를 실행하세요.")
    baskets = load_baskets(paths)
    raw["ticker"] = normalize_ticker(raw["ticker"])
    raw = raw.merge(baskets[["ticker", "bucket", "bucket_name", "name", "market", "role"]], on="ticker", how="left", suffixes=("", "_cfg"))
    for col in ["bucket", "name", "market", "role"]:
        cfg = f"{col}_cfg"
        if cfg in raw.columns:
            raw[col] = raw[col].where(raw[col].notna(), raw[cfg]) if col in raw.columns else raw[cfg]
    raw = raw.drop(columns=[c for c in raw.columns if c.endswith("_cfg")], errors="ignore")
    out = _base_ticker_features(raw)
    out = _add_peer_features(out)
    events = _read(paths.raw_dual / "dart_disclosures.parquet")
    out = _event_daily(events, out)
    out = _add_macro_links(out, universe_features, load_ticker_rules(paths))
    out = add_research_ticker_features(out, raw, universe_features, paths)
    return out.sort_values(["date", "ticker"]).drop_duplicates(["date", "ticker"], keep="last")
