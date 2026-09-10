from __future__ import annotations

"""Research-backed feature families for CrashWatch AI V4.

The module intentionally separates features that can be built from point-in-time daily
OHLCV from optional external datasets. Every rolling calculation is backward-looking.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ProjectPaths
from ..io_utils import normalize_date, normalize_ticker
from .common import first_numeric, rolling_beta, rolling_z, safe_div

EPS = 1e-12


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _positive_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    a = pd.to_numeric(a, errors="coerce").where(lambda x: x > 0)
    b = pd.to_numeric(b, errors="coerce").where(lambda x: x > 0)
    return np.log(a / b)


def _corwin_schultz_spread(high: pd.Series, low: pd.Series) -> pd.Series:
    """Corwin-Schultz spread estimator from daily high/low prices."""
    log_hl = _positive_log_ratio(high, low)
    beta = log_hl.pow(2) + log_hl.shift(1).pow(2)
    high_2 = pd.concat([high, high.shift(1)], axis=1).max(axis=1)
    low_2 = pd.concat([low, low.shift(1)], axis=1).min(axis=1)
    gamma = _positive_log_ratio(high_2, low_2).pow(2)
    denom = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    alpha = alpha.clip(lower=0, upper=20)
    return 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))


def _rolling_expected_shortfall(ret: pd.Series, window: int, quantile: float = 0.10) -> pd.Series:
    def es(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        if values.size < max(10, window // 3):
            return np.nan
        threshold = np.quantile(values, quantile)
        tail = values[values <= threshold]
        return float(tail.mean()) if tail.size else np.nan

    return ret.rolling(window, min_periods=max(10, window // 3)).apply(es, raw=True)


def _drawdown_duration(close: pd.Series) -> pd.Series:
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    peak = -np.inf
    duration = 0
    for i, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if value >= peak:
            peak = value
            duration = 0
        else:
            duration += 1
        result[i] = duration
    return pd.Series(result, index=close.index)


def _range_volatility_block(g: pd.DataFrame) -> pd.DataFrame:
    out = g[["date", "ticker"]].copy()
    o, h, l, c = g["_open"], g["_high"], g["_low"], g["_close"]
    log_hl = _positive_log_ratio(h, l)
    log_co = _positive_log_ratio(c, o)
    log_hc = _positive_log_ratio(h, c)
    log_ho = _positive_log_ratio(h, o)
    log_lc = _positive_log_ratio(l, c)
    log_lo = _positive_log_ratio(l, o)
    overnight = _positive_log_ratio(o, c.shift(1))

    parkinson = log_hl.pow(2) / (4.0 * np.log(2.0))
    garman_klass = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    rogers_satchell = log_hc * log_ho + log_lc * log_lo

    for window in (5, 20, 60):
        minp = max(3, window // 3)
        out[f"t_rangevol_parkinson_{window}"] = np.sqrt(parkinson.clip(lower=0).rolling(window, min_periods=minp).mean())
        out[f"t_rangevol_garman_klass_{window}"] = np.sqrt(garman_klass.clip(lower=0).rolling(window, min_periods=minp).mean())
        out[f"t_rangevol_rogers_satchell_{window}"] = np.sqrt(rogers_satchell.clip(lower=0).rolling(window, min_periods=minp).mean())

        k = 0.34 / (1.34 + (window + 1.0) / max(window - 1.0, 1.0))
        yz_var = (
            overnight.rolling(window, min_periods=minp).var()
            + k * log_co.rolling(window, min_periods=minp).var()
            + (1.0 - k) * rogers_satchell.rolling(window, min_periods=minp).mean()
        )
        out[f"t_rangevol_yang_zhang_{window}"] = np.sqrt(yz_var.clip(lower=0))

    log_ret = _positive_log_ratio(c, c.shift(1))
    for window in (20, 60):
        rv = log_ret.pow(2).rolling(window, min_periods=max(10, window // 2)).sum()
        bpv = (np.pi / 2.0) * (log_ret.abs() * log_ret.shift(1).abs()).rolling(
            window, min_periods=max(10, window // 2)
        ).sum()
        out[f"t_rangevol_bipower_{window}"] = bpv
        out[f"t_rangevol_jump_proxy_{window}"] = (rv - bpv).clip(lower=0)
        downside = log_ret.where(log_ret < 0, 0.0).pow(2).rolling(window, min_periods=max(10, window // 2)).sum()
        upside = log_ret.where(log_ret > 0, 0.0).pow(2).rolling(window, min_periods=max(10, window // 2)).sum()
        out[f"t_rangevol_bad_good_ratio_{window}"] = safe_div(downside, upside)
    return out


def _microstructure_block(g: pd.DataFrame) -> pd.DataFrame:
    out = g[["date", "ticker"]].copy()
    c, h, l = g["_close"], g["_high"], g["_low"]
    volume, value, cap = g["_volume"], g["_trading_value"], g["_market_cap"]
    ret = c.pct_change(fill_method=None)
    log_ret = _positive_log_ratio(c, c.shift(1))
    spread = _corwin_schultz_spread(h, l)
    clv = safe_div(2.0 * c - h - l, h - l).clip(-1, 1)
    signed_volume = clv * volume

    out["t_micro_amihud_1"] = safe_div(ret.abs(), value.abs())
    out["t_micro_corwin_schultz_spread"] = spread
    out["t_micro_close_location"] = clv
    out["t_micro_high_low_pct"] = safe_div(h - l, c.shift(1))
    out["t_micro_turnover"] = safe_div(value, cap)
    out["t_micro_kyle_proxy"] = safe_div(ret.abs(), np.sqrt(value.abs()))

    for window in (20, 60):
        minp = max(5, window // 3)
        covariance = log_ret.rolling(window, min_periods=minp).cov(log_ret.shift(1))
        out[f"t_micro_roll_spread_{window}"] = 2.0 * np.sqrt((-covariance).clip(lower=0))
        out[f"t_micro_amihud_{window}"] = out["t_micro_amihud_1"].rolling(window, min_periods=minp).mean()
        out[f"t_micro_cs_spread_{window}"] = spread.rolling(window, min_periods=minp).mean()
        out[f"t_micro_zero_return_ratio_{window}"] = ret.abs().lt(1e-12).rolling(window, min_periods=minp).mean()
        out[f"t_pressure_signed_volume_{window}"] = safe_div(
            signed_volume.rolling(window, min_periods=minp).sum(),
            volume.abs().rolling(window, min_periods=minp).sum(),
        )
        out[f"t_pressure_down_up_volume_{window}"] = safe_div(
            volume.where(ret < 0, 0.0).rolling(window, min_periods=minp).sum(),
            volume.where(ret > 0, 0.0).rolling(window, min_periods=minp).sum(),
        )
        out[f"t_pressure_return_volume_corr_{window}"] = ret.rolling(window, min_periods=minp).corr(volume.pct_change(fill_method=None))
        obv = (np.sign(ret.fillna(0.0)) * volume.fillna(0.0)).cumsum()
        out[f"t_pressure_obv_slope_{window}"] = safe_div(obv.diff(window), volume.rolling(window, min_periods=minp).mean())
        out[f"t_pressure_turnover_shock_{window}"] = rolling_z(out["t_micro_turnover"], window)
    return out


def _limit_stress_block(g: pd.DataFrame) -> pd.DataFrame:
    out = g[["date", "ticker"]].copy()
    o, h, l, c = g["_open"], g["_high"], g["_low"], g["_close"]
    prev = c.shift(1)
    ret = c.pct_change(fill_method=None)
    out["t_limit_close_to_low"] = safe_div(c - l, h - l)
    out["t_limit_gap_down"] = (safe_div(o, prev) - 1.0).clip(upper=0)
    out["t_limit_intraday_selloff"] = (safe_div(c, h) - 1.0).clip(upper=0)
    out["t_limit_near_down_limit"] = ret.le(-0.27).astype("int8")
    out["t_limit_extreme_down_5"] = ret.le(-0.05).rolling(5, min_periods=1).sum()
    out["t_limit_extreme_down_20"] = ret.le(-0.05).rolling(20, min_periods=5).sum()
    out["t_limit_consecutive_down"] = ret.lt(0).groupby((~ret.lt(0)).cumsum()).cumsum()
    out["t_limit_drawdown_duration"] = _drawdown_duration(c)
    out["t_limit_time_since_20d_low"] = c.rolling(20, min_periods=5).apply(lambda x: len(x) - 1 - int(np.nanargmin(x)), raw=True)
    return out


def _tail_dependence_block(g: pd.DataFrame, market_ret: pd.Series) -> pd.DataFrame:
    out = g[["date", "ticker"]].copy()
    ret = g["_close"].pct_change(fill_method=None)
    mkt = pd.to_numeric(market_ret, errors="coerce")
    for window in (60, 120):
        minp = max(20, window // 3)
        down_mask = mkt < 0
        tail_threshold = mkt.rolling(window, min_periods=minp).quantile(0.20)
        tail_mask = mkt <= tail_threshold
        out[f"t_taildep_downside_beta_{window}"] = rolling_beta(ret.where(down_mask), mkt.where(down_mask), window)
        out[f"t_taildep_tail_beta_{window}"] = rolling_beta(ret.where(tail_mask), mkt.where(tail_mask), window)
        out[f"t_taildep_down_corr_{window}"] = ret.where(down_mask).rolling(window, min_periods=minp).corr(mkt.where(down_mask))
        out[f"t_taildep_cocrash_freq_{window}"] = ((ret <= -0.05) & (mkt <= -0.02)).rolling(window, min_periods=minp).mean()
        beta = rolling_beta(ret, mkt, window)
        residual = ret - beta * mkt
        out[f"t_taildep_idio_vol_{window}"] = residual.rolling(window, min_periods=minp).std()
        out[f"t_taildep_idio_skew_{window}"] = residual.rolling(window, min_periods=minp).skew()
        out[f"t_taildep_expected_shortfall_{window}"] = _rolling_expected_shortfall(ret, window)
    return out


def _merge_optional_ticker_sources(base: pd.DataFrame, paths: ProjectPaths) -> pd.DataFrame:
    out = base.copy()
    attention = _read(paths.raw_dual / "naver_attention.parquet")
    if not attention.empty:
        a = normalize_date(attention)
        a["ticker"] = normalize_ticker(a["ticker"])
        ratio = pd.to_numeric(a.get("ratio"), errors="coerce")
        a = a.assign(_ratio=ratio).sort_values(["ticker", "date"])
        parts = []
        for _, g in a.groupby("ticker", sort=False):
            block = g[["date", "ticker"]].copy()
            block["t_attention_naver_level"] = g["_ratio"]
            block["t_attention_naver_change_1"] = g["_ratio"].pct_change(fill_method=None)
            block["t_attention_naver_change_5"] = g["_ratio"].pct_change(5, fill_method=None)
            block["t_attention_naver_z20"] = rolling_z(g["_ratio"], 20)
            block["t_attention_naver_shock"] = block["t_attention_naver_z20"].gt(2).astype("int8")
            block["t_attention_revised_scale_flag"] = 1
            parts.append(block)
        att = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not att.empty:
            out = out.merge(att, on=["date", "ticker"], how="left", validate="one_to_one")

    ownership = _read(paths.raw_dual / "dart_ownership_events.parquet")
    if not ownership.empty:
        ev = normalize_date(ownership)
        ev["ticker"] = normalize_ticker(ev["ticker"])
        trading_dates = pd.DatetimeIndex(sorted(out["date"].dropna().unique()))
        pos = trading_dates.searchsorted(pd.DatetimeIndex(ev["date"]), side="right")
        ev["date"] = [trading_dates[p] if p < len(trading_dates) else pd.NaT for p in pos]
        ev = ev.dropna(subset=["date"])
        ev["change_shares"] = pd.to_numeric(ev.get("change_shares"), errors="coerce").fillna(0.0)
        ev["event_type"] = ev.get("event_type", "unknown").fillna("unknown")
        daily = ev.groupby(["date", "ticker", "event_type"], as_index=False).agg(
            change_shares=("change_shares", "sum"), event_count=("event_type", "size")
        )
        parts = []
        for ticker, g in out.groupby("ticker", sort=False):
            block = g[["date", "ticker"]].sort_values("date").copy()
            e = daily[daily["ticker"].eq(ticker)]
            for event_type in ("major", "insider"):
                x = e[e["event_type"].eq(event_type)][["date", "change_shares", "event_count"]]
                merged = block[["date"]].merge(x, on="date", how="left").fillna(0.0)
                change = merged["change_shares"]
                block[f"t_owner_{event_type}_net_change_20"] = change.rolling(20, min_periods=1).sum().to_numpy()
                block[f"t_owner_{event_type}_sell_count_20"] = change.lt(0).rolling(20, min_periods=1).sum().to_numpy()
                block[f"t_owner_{event_type}_buy_count_20"] = change.gt(0).rolling(20, min_periods=1).sum().to_numpy()
                block[f"t_owner_{event_type}_event_count_20"] = merged["event_count"].rolling(20, min_periods=1).sum().to_numpy()
            parts.append(block)
        own = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not own.empty:
            out = out.merge(own, on=["date", "ticker"], how="left", validate="one_to_one")

    financials = _read(paths.raw_dual / "dart_financial_quality.parquet")
    if not financials.empty:
        f = normalize_date(financials, "available_from").rename(columns={"available_from": "date"})
        f["ticker"] = normalize_ticker(f["ticker"])
        f = f.sort_values(["ticker", "date"])
        blocks = []
        for ticker, g in out.groupby("ticker", sort=False):
            source = f[f["ticker"].eq(ticker)].drop(columns="ticker", errors="ignore")
            merged = pd.merge_asof(g.sort_values("date"), source.sort_values("date"), on="date", direction="backward") if not source.empty else g.copy()
            blocks.append(merged)
        out = pd.concat(blocks, ignore_index=True, sort=False)
    return out



def _add_ticker_network_features(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or "t_price_ret_1" not in features.columns:
        return features
    out = features.sort_values(["date", "ticker"]).copy()
    ret = pd.to_numeric(out["t_price_ret_1"], errors="coerce")
    out["_network_ret"] = ret

    market_group = out.groupby("date", sort=False)["_network_ret"]
    market_count = market_group.transform("count")
    market_total = market_group.transform("sum")
    out["_network_market_loo"] = safe_div(market_total - ret, market_count - 1)

    if "bucket" in out.columns:
        bucket_group = out.groupby(["date", "bucket"], sort=False)["_network_ret"]
        bucket_count = bucket_group.transform("count")
        bucket_total = bucket_group.transform("sum")
        out["_network_bucket_loo"] = safe_div(bucket_total - ret, bucket_count - 1)
    else:
        out["_network_bucket_loo"] = np.nan

    parts = []
    for _, g in out.groupby("ticker", sort=False):
        block = g.copy()
        r = block["_network_ret"]
        market = block["_network_market_loo"]
        bucket = block["_network_bucket_loo"]
        block["t_network_market_corr_60"] = r.rolling(60, min_periods=20).corr(market)
        block["t_network_bucket_corr_60"] = r.rolling(60, min_periods=20).corr(bucket)
        block["t_network_peer_lead_beta_60"] = rolling_beta(r, bucket.shift(1), 60)
        block["t_network_market_lead_beta_60"] = rolling_beta(r, market.shift(1), 60)
        block["t_network_tail_sync_60"] = ((r <= -0.05) & (bucket <= -0.03)).rolling(60, min_periods=20).mean()
        block["t_network_centrality_proxy_60"] = pd.concat([
            block["t_network_market_corr_60"], block["t_network_bucket_corr_60"]
        ], axis=1).mean(axis=1)
        parts.append(block)
    return pd.concat(parts, ignore_index=True, sort=False).drop(
        columns=["_network_ret", "_network_market_loo", "_network_bucket_loo"], errors="ignore"
    )

def add_research_ticker_features(
    base_features: pd.DataFrame,
    raw: pd.DataFrame,
    universe_features: pd.DataFrame,
    paths: ProjectPaths,
) -> pd.DataFrame:
    if base_features.empty or raw.empty:
        return base_features
    df = normalize_date(raw)
    df["ticker"] = normalize_ticker(df["ticker"])
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    df = df.assign(
        _open=first_numeric(df, ["open", "시가"]),
        _high=first_numeric(df, ["high", "고가"]),
        _low=first_numeric(df, ["low", "저가"]),
        _close=first_numeric(df, ["close", "종가"]),
        _volume=first_numeric(df, ["volume", "거래량"]),
        _trading_value=first_numeric(df, ["trading_value", "거래대금"]),
        _market_cap=first_numeric(df, ["market_cap", "시가총액"]),
    )

    market_candidates = ["u_market_kospi_ret_1", "u_market_kosdaq_ret_1", "u_market_ret_1"]
    market_col = next((c for c in market_candidates if c in universe_features.columns), None)
    market = universe_features[["date", market_col]].copy() if market_col else pd.DataFrame({"date": df["date"].unique(), "_market_ret": np.nan})
    if market_col:
        market = market.rename(columns={market_col: "_market_ret"})

    pieces = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date")
        m = g[["date"]].merge(market, on="date", how="left")["_market_ret"]
        blocks = [
            _range_volatility_block(g),
            _microstructure_block(g),
            _limit_stress_block(g),
            _tail_dependence_block(g, m),
        ]
        block = blocks[0]
        for extra in blocks[1:]:
            block = block.merge(extra, on=["date", "ticker"], how="left", validate="one_to_one")
        pieces.append(block)
    research = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
    out = base_features.merge(research, on=["date", "ticker"], how="left", validate="one_to_one")
    out = _add_ticker_network_features(out)
    return _merge_optional_ticker_sources(out, paths)


def _daily_panel_for_research(raw: pd.DataFrame) -> pd.DataFrame:
    df = normalize_date(raw)
    df["ticker"] = normalize_ticker(df["ticker"])
    df = df.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    close = first_numeric(df, ["close", "종가"])
    high = first_numeric(df, ["high", "고가"])
    low = first_numeric(df, ["low", "저가"])
    volume = first_numeric(df, ["volume", "거래량"])
    value = first_numeric(df, ["trading_value", "거래대금"])
    cap = first_numeric(df, ["market_cap", "시가총액"])
    df = df.assign(_close=close, _high=high, _low=low, _volume=volume, _value=value, _cap=cap)
    df["_ret"] = df.groupby("ticker")["_close"].pct_change(fill_method=None)
    df["_amihud"] = safe_div(df["_ret"].abs(), df["_value"].abs())
    df["_turnover"] = safe_div(df["_value"], df["_cap"])
    df["_spread"] = df.groupby("ticker", group_keys=False).apply(
        lambda g: _corwin_schultz_spread(g["_high"], g["_low"]), include_groups=False
    ).reset_index(level=0, drop=True) if len(df) else np.nan
    df["_clv"] = safe_div(2.0 * df["_close"] - df["_high"] - df["_low"], df["_high"] - df["_low"]).clip(-1, 1)
    df["_signed_volume"] = df["_clv"] * df["_volume"]
    return df


def _network_timeseries(panel: pd.DataFrame, window: int = 60, min_periods: int = 30) -> pd.DataFrame:
    pivot = panel.pivot(index="date", columns="ticker", values="_ret").sort_index()
    rows: list[dict] = []
    for i, date in enumerate(pivot.index):
        if i + 1 < min_periods:
            rows.append({"date": date})
            continue
        sample = pivot.iloc[max(0, i - window + 1): i + 1]
        valid_cols = sample.columns[sample.notna().sum() >= min_periods]
        if len(valid_cols) < 5:
            rows.append({"date": date})
            continue
        corr = sample[valid_cols].corr(min_periods=min_periods).to_numpy(dtype=float)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        upper = corr[mask]
        upper = upper[np.isfinite(upper)]
        if upper.size == 0:
            rows.append({"date": date})
            continue
        corr_clean = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr_clean, 1.0)
        eigvals = np.linalg.eigvalsh(corr_clean)
        rows.append({
            "date": date,
            "u_tailnet_avg_corr_60": float(np.mean(upper)),
            "u_tailnet_corr_q90_60": float(np.quantile(upper, 0.90)),
            "u_tailnet_density_corr60": float(np.mean(upper >= 0.60)),
            "u_tailnet_largest_eigen_share_60": float(max(eigvals[-1], 0.0) / max(np.maximum(eigvals, 0.0).sum(), EPS)),
        })
    return pd.DataFrame(rows)


def _optional_universe_sources(paths: ProjectPaths) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    fred = _read(paths.raw_dual / "fred_credit.parquet")
    if not fred.empty:
        f = normalize_date(fred).sort_values("date")
        for col in [c for c in f.columns if c != "date"]:
            f[col] = pd.to_numeric(f[col], errors="coerce")
            f[f"u_credit_{col}_change_5"] = f[col].diff(5)
            f[f"u_credit_{col}_z60"] = rolling_z(f[col], 60)
        f = f.rename(columns={c: f"u_credit_{c}" for c in list(f.columns) if c != "date" and not c.startswith("u_credit_")})
        frames.append(f)

    etf = _read(paths.raw_dual / "etf_pressure.parquet")
    if not etf.empty:
        e = normalize_date(etf).sort_values("date")
        frames.append(e)

    deriv = _read(paths.raw_dual / "derivatives_daily.parquet")
    if not deriv.empty:
        d = normalize_date(deriv).sort_values("date")
        cols = [c for c in d.columns if c not in {"date", "available_from"}]
        if "available_from" not in deriv.columns:
            d[cols] = d[cols].shift(1)
        d = d.rename(columns={c: c if c.startswith("u_deriv_") else f"u_deriv_{c}" for c in cols})
        frames.append(d)

    attention = _read(paths.raw_dual / "naver_attention.parquet")
    if not attention.empty:
        a = normalize_date(attention)
        a["ratio"] = pd.to_numeric(a.get("ratio"), errors="coerce")
        daily = a.groupby("date", as_index=False).agg(
            u_attention_median=("ratio", "median"),
            u_attention_max=("ratio", "max"),
            u_attention_dispersion=("ratio", "std"),
        ).sort_values("date")
        daily["u_attention_shock_ratio"] = a.assign(
            _z=a.groupby("ticker", group_keys=False)["ratio"].transform(lambda s: rolling_z(s, 20))
        ).groupby("date")["_z"].apply(lambda s: float((s > 2).mean())).to_numpy()
        daily["u_attention_revised_scale_flag"] = 1
        frames.append(daily)
    return frames


def build_research_universe_features(paths: ProjectPaths) -> pd.DataFrame:
    raw = _read(paths.raw_dual / "krx_ticker_timeseries.parquet")
    if raw.empty:
        return pd.DataFrame(columns=["date"])
    panel = _daily_panel_for_research(raw)

    def aggregate(g: pd.DataFrame) -> pd.Series:
        ret = g["_ret"].dropna()
        return pd.Series({
            "u_micro_amihud_median": g["_amihud"].median(),
            "u_micro_spread_median": g["_spread"].median(),
            "u_micro_zero_return_ratio": g["_ret"].abs().lt(1e-12).mean(),
            "u_micro_turnover_median": g["_turnover"].median(),
            "u_micro_signed_volume_pressure": safe_div(g["_signed_volume"].sum(), g["_volume"].abs().sum()),
            "u_tailnet_return_q01": ret.quantile(0.01) if len(ret) else np.nan,
            "u_tailnet_return_q05": ret.quantile(0.05) if len(ret) else np.nan,
            "u_tailnet_return_q10": ret.quantile(0.10) if len(ret) else np.nan,
            "u_tailnet_crash_ratio_3pct": (ret <= -0.03).mean() if len(ret) else np.nan,
            "u_tailnet_crash_ratio_5pct": (ret <= -0.05).mean() if len(ret) else np.nan,
            "u_tailnet_crash_ratio_7pct": (ret <= -0.07).mean() if len(ret) else np.nan,
            "u_tailnet_left_tail_mean": ret[ret <= ret.quantile(0.10)].mean() if len(ret) >= 10 else np.nan,
        })

    daily = panel.groupby("date", sort=True).apply(aggregate, include_groups=False).reset_index()
    for col in [c for c in daily.columns if c.startswith(("u_micro_", "u_tailnet_"))]:
        daily[f"{col}_z20"] = rolling_z(pd.to_numeric(daily[col], errors="coerce"), 20)
    network = _network_timeseries(panel)
    out = daily.merge(network, on="date", how="left", validate="one_to_one")
    for frame in _optional_universe_sources(paths):
        out = pd.merge_asof(out.sort_values("date"), frame.sort_values("date"), on="date", direction="backward")
    return out.sort_values("date").drop_duplicates("date", keep="last")
