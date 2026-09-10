from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from ..config import ProjectPaths
from ..io_utils import atomic_json

UNIVERSE_GROUPS: dict[str, list[str]] = {
    "u_market_trend": [
        "u_kospi_ret_1", "u_kospi_ret_5", "u_kospi_ret_20", "u_kospi_ret_60",
        "u_kospi_drawdown_20", "u_kospi_drawdown_60", "u_kospi_ma_gap_20", "u_kospi_ma_gap_60",
        "u_kosdaq_ret_1", "u_kosdaq_ret_5", "u_kosdaq_ret_20", "u_kosdaq_drawdown_20",
        "u_kosdaq_ma_gap_20", "u_kospi_kosdaq_spread_20",
    ],
    "u_market_breadth": [
        "u_advancing_ratio", "u_declining_ratio", "u_advance_decline_spread",
        "u_advance_decline_cumulative", "u_new_high_ratio_20", "u_new_low_ratio_20",
        "u_above_ma20_ratio", "u_above_ma60_ratio", "u_market_dispersion_1",
        "u_market_dispersion_5", "u_cross_section_negative_ratio", "u_cross_section_crash_ratio",
    ],
    "u_market_volatility": [
        "u_kospi_vol_5", "u_kospi_vol_20", "u_kospi_downside_vol_20", "u_kospi_skew_20",
        "u_kospi_kurtosis_20", "u_cross_section_volatility", "u_cross_section_downside_dispersion",
        "u_vix_level", "u_vix_change_1", "u_vix_zscore_60",
    ],
    "u_market_liquidity": [
        "u_market_value_traded", "u_market_value_traded_zscore_20", "u_market_turnover",
        "u_market_turnover_zscore_20", "u_zero_return_ratio", "u_low_volume_ratio",
        "u_amihud_market", "u_liquidity_stress_index",
    ],
    "u_aggregate_flow": [
        "u_foreign_net_buy_value", "u_institution_net_buy_value", "u_individual_net_buy_value",
        "u_foreign_net_buy_ratio", "u_institution_net_buy_ratio", "u_individual_net_buy_ratio",
        "u_foreign_flow_zscore_20", "u_institution_flow_zscore_20", "u_foreign_flow_streak",
        "u_institution_flow_streak",
    ],
    "u_aggregate_shorting": [
        "u_short_sell_value", "u_short_sell_ratio", "u_short_balance_value",
        "u_short_balance_to_market_cap", "u_short_ratio_zscore_20", "u_short_balance_change_5",
        "u_short_balance_change_20", "u_short_stress_index",
    ],
    "u_macro_rates_fx": [
        "u_usdkrw_level", "u_usdkrw_ret_1", "u_usdkrw_ret_5", "u_usdkrw_vol_20",
        "u_korea_base_rate", "u_korea_3y_yield", "u_korea_10y_yield", "u_us_2y_yield",
        "u_us_10y_yield", "u_us_10y_change_5", "u_term_spread_us", "u_term_spread_kr",
        "u_credit_spread_proxy", "u_dollar_index",
    ],
    "u_realestate_korea": [
        "u_housing_price_change", "u_apartment_price_change", "u_jeonse_price_change",
        "u_unsold_housing_change", "u_housing_transaction_change", "u_realestate_sentiment",
        "u_realestate_stress_index",
    ],
    "u_regime_interaction": [
        "u_regime_high_vol", "u_regime_risk_off", "u_regime_liquidity_stress",
        "u_regime_foreign_selloff", "u_regime_rate_shock", "u_regime_fx_shock",
        "u_regime_bear_market",
    ],
}

TICKER_GROUPS: dict[str, list[str]] = {
    "t_price_trend": [
        "t_price_ret_1", "t_price_ret_3", "t_price_ret_5", "t_price_ret_10", "t_price_ret_20",
        "t_price_ret_60", "t_price_ret_120", "t_price_ma_gap_20", "t_price_ma_gap_60",
        "t_price_drawdown_20", "t_price_drawdown_60", "t_price_gap_open", "t_price_intraday_return",
    ],
    "t_volatility_tail": [
        "t_vol_realized_5", "t_vol_realized_20", "t_vol_realized_60", "t_tail_downside_20",
        "t_tail_skew_20", "t_tail_kurt_60", "t_vol_range_1",
    ],
    "t_liquidity_turnover": [
        "t_liq_turnover", "t_liq_volume_z20", "t_liq_value_z20", "t_liq_down_day_turnover",
    ],
    "t_investor_flow": [
        "t_foreign_net_buy_value", "t_institution_net_buy_value", "t_individual_net_buy_value",
        "t_foreign_net_buy_ratio", "t_institution_net_buy_ratio", "t_individual_net_buy_ratio",
        "t_foreign_flow_sum_5", "t_foreign_flow_sum_20", "t_institution_flow_sum_5",
        "t_institution_flow_sum_20", "t_foreign_flow_zscore_20", "t_institution_flow_zscore_20",
        "t_foreign_flow_streak", "t_institution_flow_streak", "t_foreign_ownership_rate",
        "t_foreign_ownership_change_5", "t_foreign_ownership_change_20",
    ],
    "t_short_selling": [
        "t_short_volume", "t_short_value", "t_short_volume_ratio", "t_short_value_ratio",
        "t_short_balance_value", "t_short_balance_shares", "t_short_balance_to_market_cap",
        "t_short_balance_change_5", "t_short_balance_change_20", "t_short_ratio_zscore_20",
        "t_short_balance_slope_20", "t_short_squeeze_proxy",
    ],
    "t_valuation_size": [
        "t_market_cap", "t_log_market_cap", "t_per", "t_pbr", "t_dividend_yield",
        "t_market_cap_rank_pct", "t_per_industry_zscore", "t_pbr_industry_zscore",
        "t_size_bucket", "t_value_distress_score", "t_per_missing", "t_pbr_missing",
        "t_dividend_missing",
    ],
    "t_peer_contagion": [
        "t_peer_leave_one_out_ret_1", "t_peer_relative_ret_1", "t_peer_down_3pct_count",
        "t_peer_bucket_dispersion", "t_peer_lead_lag_1", "t_peer_lead_lag_3",
    ],
    "t_disclosure_event": [],
    "t_macro_linkage": [],
    "t_interaction": [
        "t_price_trend_x_high_vol", "t_foreign_sell_x_market_stress",
        "t_short_rise_x_negative_momentum", "t_liquidity_drop_x_drawdown",
        "t_macro_beta_x_fx_shock", "t_peer_crash_x_low_liquidity",
        "t_disclosure_risk_x_high_vol",
    ],
}


def _present(columns: Iterable[str], requested: list[str]) -> list[str]:
    available = set(columns)
    return [feature for feature in requested if feature in available]


def build_catalogs(paths: ProjectPaths, universe: pd.DataFrame, ticker: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    universe_catalog = {group: _present(universe.columns, features) for group, features in UNIVERSE_GROUPS.items()}
    ticker_groups = {group: list(features) for group, features in TICKER_GROUPS.items()}
    ticker_groups["t_disclosure_event"] = sorted(c for c in ticker.columns if c.startswith("t_event_") or c.startswith("t_dart_"))
    ticker_groups["t_macro_linkage"] = sorted(c for c in ticker.columns if c.startswith("t_link_"))
    ticker_catalog = {group: _present(ticker.columns, features) for group, features in ticker_groups.items()}
    atomic_json(universe_catalog, paths.feature_dual / "feature_catalog_universe.json")
    atomic_json(ticker_catalog, paths.feature_dual / "feature_catalog_ticker.json")
    return universe_catalog, ticker_catalog
