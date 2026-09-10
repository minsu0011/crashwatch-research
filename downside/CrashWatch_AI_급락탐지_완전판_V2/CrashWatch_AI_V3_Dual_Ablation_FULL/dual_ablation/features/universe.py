from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ..config import ProjectPaths
from ..io_utils import normalize_date, normalize_ticker
from .common import drawdown, downside_semivol, rolling_z, safe_div


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _rolling_streak(values: pd.Series) -> pd.Series:
    signs = np.sign(pd.to_numeric(values, errors="coerce").fillna(0))
    groups = signs.ne(signs.shift()).cumsum()
    lengths = signs.groupby(groups).cumcount() + 1
    return lengths * signs


def _ohlcv_cross_section(paths: ProjectPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = _read(paths.raw_dual / "krx_ticker_timeseries.parquet")
    if raw.empty:
        raise FileNotFoundError("KRX ticker OHLCV is required")
    raw = normalize_date(raw)
    raw["ticker"] = normalize_ticker(raw["ticker"])
    raw = raw.sort_values(["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    raw["close"] = pd.to_numeric(raw["close"], errors="coerce")
    raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
    raw["ret_1"] = raw.groupby("ticker")["close"].pct_change(fill_method=None)
    raw["ret_5"] = raw.groupby("ticker")["close"].pct_change(5, fill_method=None)
    raw["ma20"] = raw.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=20).mean())
    raw["ma60"] = raw.groupby("ticker")["close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
    raw["high20"] = raw.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=20).max())
    raw["low20"] = raw.groupby("ticker")["close"].transform(lambda s: s.rolling(20, min_periods=20).min())
    raw["trading_value_proxy"] = raw["close"] * raw["volume"]
    raw["amihud"] = safe_div(raw["ret_1"].abs(), raw["trading_value_proxy"].replace(0, np.nan))

    def aggregate(block: pd.DataFrame) -> pd.Series:
        ret = block["ret_1"].dropna()
        ret5 = block["ret_5"].dropna()
        tv = block["trading_value_proxy"]
        active = block.loc[block["volume"].fillna(0).gt(0)]
        return pd.Series({
            "u_advancing_ratio": float((ret > 0).mean()) if len(ret) else np.nan,
            "u_declining_ratio": float((ret < 0).mean()) if len(ret) else np.nan,
            "u_new_high_ratio_20": float(active["close"].ge(active["high20"]).mean()) if len(active) else np.nan,
            "u_new_low_ratio_20": float(active["close"].le(active["low20"]).mean()) if len(active) else np.nan,
            "u_above_ma20_ratio": float(active["close"].gt(active["ma20"]).mean()) if len(active) else np.nan,
            "u_above_ma60_ratio": float(active["close"].gt(active["ma60"]).mean()) if len(active) else np.nan,
            "u_market_dispersion_1": ret.std(),
            "u_market_dispersion_5": ret5.std(),
            "u_cross_section_negative_ratio": float((ret < 0).mean()) if len(ret) else np.nan,
            "u_cross_section_crash_ratio": float((ret <= -0.05).mean()) if len(ret) else np.nan,
            "u_cross_section_volatility": ret.std(),
            "u_cross_section_downside_dispersion": ret.loc[ret < 0].std(),
            "u_market_value_traded": tv.sum(min_count=1),
            "u_market_turnover": safe_div(tv, block["close"].abs()).median(),
            "u_zero_return_ratio": float(ret.eq(0).mean()) if len(ret) else np.nan,
            "u_low_volume_ratio": float(block["volume"].le(block["volume"].median() * 0.1).mean()),
            "u_amihud_market": block["amihud"].median(),
        })

    cross = raw.groupby("date", sort=True).apply(aggregate, include_groups=False).reset_index()
    cross["u_advance_decline_spread"] = cross["u_advancing_ratio"] - cross["u_declining_ratio"]
    cross["u_advance_decline_cumulative"] = cross["u_advance_decline_spread"].cumsum()
    cross["u_market_value_traded_zscore_20"] = rolling_z(cross["u_market_value_traded"], 20)
    cross["u_market_turnover_zscore_20"] = rolling_z(cross["u_market_turnover"], 20)
    cross["u_liquidity_stress_index"] = (
        -cross["u_market_turnover_zscore_20"] + rolling_z(cross["u_amihud_market"], 20)
        + rolling_z(cross["u_low_volume_ratio"], 20)
    ) / 3.0
    return raw, cross


def _market_features(paths: ProjectPaths) -> pd.DataFrame:
    assets = normalize_date(_read(paths.raw_dual / "global_assets.parquet"))
    out = assets[["date"]].copy()
    for alias in ("kospi", "kosdaq"):
        close = pd.to_numeric(assets.get(f"{alias}_close"), errors="coerce")
        out[f"u_{alias}_ret_1"] = close.pct_change(fill_method=None)
        out[f"u_{alias}_ret_5"] = close.pct_change(5, fill_method=None)
        out[f"u_{alias}_ret_20"] = close.pct_change(20, fill_method=None)
        if alias == "kospi":
            out[f"u_{alias}_ret_60"] = close.pct_change(60, fill_method=None)
        out[f"u_{alias}_drawdown_20"] = drawdown(close, 20)
        out[f"u_{alias}_ma_gap_20"] = safe_div(close, close.rolling(20, min_periods=20).mean()) - 1
        if alias == "kospi":
            out[f"u_{alias}_drawdown_60"] = drawdown(close, 60)
            out[f"u_{alias}_ma_gap_60"] = safe_div(close, close.rolling(60, min_periods=60).mean()) - 1
            r1 = out["u_kospi_ret_1"]
            out["u_kospi_vol_5"] = r1.rolling(5, min_periods=5).std()
            out["u_kospi_vol_20"] = r1.rolling(20, min_periods=20).std()
            out["u_kospi_downside_vol_20"] = downside_semivol(r1, 20)
            out["u_kospi_skew_20"] = r1.rolling(20, min_periods=20).skew()
            out["u_kospi_kurtosis_20"] = r1.rolling(20, min_periods=20).kurt()
    out["u_kospi_kosdaq_spread_20"] = out["u_kospi_ret_20"] - out["u_kosdaq_ret_20"]

    # US/global data are shifted one row because their same-calendar-day close
    # is not known at the Korean close.
    vix = pd.to_numeric(assets.get("vix_close"), errors="coerce").shift(1)
    out["u_vix_level"] = vix
    out["u_vix_change_1"] = vix.diff()
    out["u_vix_zscore_60"] = rolling_z(vix, 60)
    return out


def _legacy_panel(paths: ProjectPaths) -> pd.DataFrame:
    path = paths.project.parent / "crashwatch_ai_data" / "development" / "training_dataset.parquet"
    if not path.exists():
        return pd.DataFrame()
    wanted = [
        "date", "ticker", "trading_value", "market_cap", "foreign_ownership_pct",
        "short_volume", "short_balance_value", "short_balance_shares",
    ]
    schema = set(pq.ParquetFile(path).schema_arrow.names)
    cols = [c for c in wanted if c in schema]
    legacy = pd.read_parquet(path, columns=cols)
    legacy = normalize_date(legacy)
    legacy["ticker"] = normalize_ticker(legacy["ticker"])
    return legacy


def _aggregate_flows_and_shorts(paths: ProjectPaths, dates: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame({"date": pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()})
    legacy = _legacy_panel(paths)
    # The current KRX session does not provide investor-flow fields.  Keep NaN
    # values explicit so quality audit marks source_failed instead of inventing
    # zeros.
    for col in [
        "u_foreign_net_buy_value", "u_institution_net_buy_value", "u_individual_net_buy_value",
        "u_short_sell_value", "u_short_sell_ratio", "u_short_balance_value",
        "u_short_balance_to_market_cap",
    ]:
        out[col] = np.nan
    if not legacy.empty:
        by_date = legacy.groupby("date", as_index=False).agg(
            _market_cap=("market_cap", "sum"),
            _short_balance=("short_balance_value", "sum"),
            _short_volume=("short_volume", "sum"),
            _trading_value=("trading_value", "sum"),
        )
        out = out.merge(by_date, on="date", how="left")
        # min_count semantics: all-missing source must stay missing.
        if legacy["short_balance_value"].notna().any():
            out["u_short_balance_value"] = out["_short_balance"]
            out["u_short_balance_to_market_cap"] = safe_div(out["_short_balance"], out["_market_cap"])
        if legacy["short_volume"].notna().any():
            out["u_short_sell_value"] = out["_short_volume"]
            out["u_short_sell_ratio"] = safe_div(out["_short_volume"], out["_trading_value"])
        out = out.drop(columns=[c for c in out if c.startswith("_")])
    total_value = out[["u_foreign_net_buy_value", "u_institution_net_buy_value", "u_individual_net_buy_value"]].abs().sum(axis=1, min_count=1)
    for who in ("foreign", "institution", "individual"):
        out[f"u_{who}_net_buy_ratio"] = safe_div(out[f"u_{who}_net_buy_value"], total_value)
    out["u_foreign_flow_zscore_20"] = rolling_z(out["u_foreign_net_buy_value"], 20)
    out["u_institution_flow_zscore_20"] = rolling_z(out["u_institution_net_buy_value"], 20)
    out["u_foreign_flow_streak"] = _rolling_streak(out["u_foreign_net_buy_value"])
    out["u_institution_flow_streak"] = _rolling_streak(out["u_institution_net_buy_value"])
    out["u_short_ratio_zscore_20"] = rolling_z(out["u_short_sell_ratio"], 20)
    out["u_short_balance_change_5"] = out["u_short_balance_value"].pct_change(5, fill_method=None)
    out["u_short_balance_change_20"] = out["u_short_balance_value"].pct_change(20, fill_method=None)
    out["u_short_stress_index"] = (out["u_short_ratio_zscore_20"] + rolling_z(out["u_short_balance_change_20"], 20)) / 2
    return out


def _macro_features(paths: ProjectPaths, dates: pd.Series) -> pd.DataFrame:
    base = pd.DataFrame({"date": pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()})
    fred_path = paths.project.parent / "crashwatch_ai_data" / "raw" / "macro_fred.parquet"
    fred = _read(fred_path)
    if not fred.empty:
        fred = normalize_date(fred)
        pivot = fred.pivot_table(index="date", columns="alias", values="value", aggfunc="last").sort_index()
        pivot = pivot.shift(1).reset_index()  # publication/timezone-safe one-observation lag
        base = pd.merge_asof(base, pivot.sort_values("date"), on="date", direction="backward")
    ecos_path = paths.project.parent / "crashwatch_ai_data" / "raw" / "macro_ecos.parquet"
    ecos = _read(ecos_path)
    if not ecos.empty:
        ecos = normalize_date(ecos)
        bok = ecos.loc[ecos["alias"].eq("bok_base_rate"), ["date", "value"]].drop_duplicates("date", keep="last")
        bok["value"] = pd.to_numeric(bok["value"], errors="coerce").shift(1)
        base = pd.merge_asof(base.sort_values("date"), bok.sort_values("date"), on="date", direction="backward")
    mapping = {
        "usdkrw": "u_usdkrw_level", "us_2y": "u_us_2y_yield", "us_10y": "u_us_10y_yield",
        "dollar_index_broad": "u_dollar_index", "us_high_yield_spread": "u_credit_spread_proxy",
        "value": "u_korea_base_rate",
    }
    out = base[["date"]].copy()
    for source, target in mapping.items():
        out[target] = pd.to_numeric(base.get(source), errors="coerce")
    out["u_usdkrw_ret_1"] = out["u_usdkrw_level"].pct_change(fill_method=None)
    out["u_usdkrw_ret_5"] = out["u_usdkrw_level"].pct_change(5, fill_method=None)
    out["u_usdkrw_vol_20"] = out["u_usdkrw_ret_1"].rolling(20, min_periods=20).std()
    out["u_us_10y_change_5"] = out["u_us_10y_yield"].diff(5)
    out["u_term_spread_us"] = out["u_us_10y_yield"] - out["u_us_2y_yield"]
    out["u_korea_3y_yield"] = np.nan
    out["u_korea_10y_yield"] = np.nan
    out["u_term_spread_kr"] = np.nan
    return out


def _realestate_unsafe(dates: pd.Series) -> pd.DataFrame:
    # Existing ECOS real-estate extracts do not contain verified release_date
    # or available_from.  They are deliberately unavailable until that audit
    # metadata is collected.
    out = pd.DataFrame({"date": pd.DatetimeIndex(pd.to_datetime(dates).dropna().unique()).sort_values()})
    for col in [
        "u_housing_price_change", "u_apartment_price_change", "u_jeonse_price_change",
        "u_unsold_housing_change", "u_housing_transaction_change", "u_realestate_sentiment",
        "u_realestate_stress_index",
    ]:
        out[col] = np.nan
    return out


def _regimes(out: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "u_regime_high_vol": out.get("u_kospi_vol_20"),
        "u_regime_risk_off": out.get("u_vix_level"),
        "u_regime_liquidity_stress": out.get("u_liquidity_stress_index"),
        "u_regime_foreign_selloff": -out.get("u_foreign_net_buy_ratio", pd.Series(np.nan, index=out.index)),
        "u_regime_rate_shock": out.get("u_us_10y_change_5"),
        "u_regime_fx_shock": out.get("u_usdkrw_vol_20"),
        "u_regime_bear_market": -out.get("u_kospi_drawdown_60", pd.Series(np.nan, index=out.index)),
    }
    for name, series in specs.items():
        threshold = series.shift(1).rolling(252, min_periods=126).quantile(0.8)
        out[name] = series.gt(threshold).where(series.notna() & threshold.notna()).astype("float32")
    return out


def build_universe_features(paths: ProjectPaths) -> pd.DataFrame:
    ticker_raw, cross = _ohlcv_cross_section(paths)
    market = _market_features(paths)
    dates = cross["date"]
    flow_short = _aggregate_flows_and_shorts(paths, dates)
    macro = _macro_features(paths, dates)
    realestate = _realestate_unsafe(dates)
    out = cross
    for frame in (market, flow_short, macro, realestate):
        out = pd.merge_asof(out.sort_values("date"), frame.sort_values("date"), on="date", direction="backward")
    out = _regimes(out)
    return out.sort_values("date").drop_duplicates("date", keep="last")
