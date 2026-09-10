from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ProjectPaths, get_paths
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_date, normalize_ticker
from .numba_kernels import (
    diff_lag,
    pct_change,
    rolling_corr,
    rolling_slope,
    rolling_sum,
    rolling_zscore,
    safe_divide,
    signed_streak,
    warmup,
)

LOGGER = logging.getLogger(__name__)


def _first_numeric(df: pd.DataFrame, candidates: list[str]) -> np.ndarray:
    output = np.full(len(df), np.nan, dtype=np.float64)
    for col in candidates:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
            values[~np.isfinite(values)] = np.nan
            missing = ~np.isfinite(output)
            output[missing] = values[missing]
            if np.isfinite(output).all():
                break
    return output


def _series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _to32(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _delay_short_balance_availability(
    frame: pd.DataFrame,
    periods: int = 2,
) -> pd.DataFrame:
    """공매도 순보유잔고를 보고 공개시점(T+2 거래일) 이후로 지연한다."""
    out = frame.sort_values(["ticker", "date"]).copy()
    columns = [
        column
        for column in [
            "fs_short_balance_shares",
            "fs_short_balance_value",
            "fs_short_balance_ratio",
        ]
        if column in out.columns
    ]
    if columns:
        out[columns] = out.groupby("ticker", sort=False)[columns].shift(periods)
    return out


def _build_ticker_block(block: pd.DataFrame) -> pd.DataFrame:
    block = block.sort_values("date").reset_index(drop=True)
    n = len(block)
    out = block[["date", "ticker"]].copy()

    close = _first_numeric(block, ["close", "종가", "t_price_close"])
    volume = _first_numeric(block, ["volume", "거래량", "fv_volume", "t_liq_volume"])
    trading_value = _first_numeric(block, ["trading_value", "거래대금", "fv_trading_value", "t_liq_trading_value"])
    market_cap = _first_numeric(block, ["fv_market_cap", "market_cap", "시가총액", "t_size_market_cap"])
    ret1 = _first_numeric(block, ["t_price_ret_1", "ret_1", "return_1"])
    if not np.isfinite(ret1).any() and np.isfinite(close).any():
        ret1 = pct_change(close, 1).astype(np.float64)
    ret5 = _first_numeric(block, ["t_price_ret_5", "ret_5"])
    if not np.isfinite(ret5).any() and np.isfinite(close).any():
        ret5 = pct_change(close, 5).astype(np.float64)
    drawdown20 = _first_numeric(block, ["t_price_drawdown_20", "drawdown_20"])
    vol20 = _first_numeric(block, ["t_vol_realized_20", "vol_20", "t_rangevol_gk_20"])
    illiq = _first_numeric(block, ["t_micro_amihud_20", "t_liq_amihud_20", "amihud_20"])

    short_volume = _first_numeric(block, ["fs_short_trade_volume", "fs_status_short_volume", "fs_volume_short_volume", "fs_volume_공매도"])
    short_value = _first_numeric(block, ["fs_short_trade_value", "fs_status_short_value", "fs_value_short_value", "fs_value_short_volume", "fs_value_공매도금액"])
    short_ratio = _first_numeric(block, ["fs_short_volume_ratio", "fs_short_value_ratio", "fs_short_balance_ratio", "fs_volume_short_ratio", "fs_value_short_ratio", "fs_balance_short_ratio", "fs_status_short_ratio"])
    short_balance_shares = _first_numeric(block, ["fs_short_balance_shares", "fs_balance_short_balance_shares", "fs_status_short_balance_shares"])
    short_balance_value = _first_numeric(block, ["fs_short_balance_value", "fs_balance_short_balance_value", "fs_status_short_balance_value"])
    listed_shares = _first_numeric(block, ["fs_balance_listed_shares", "fv_listed_shares", "listed_shares"])

    foreign_value = _first_numeric(block, [
        "ff_net_value_foreign", "ff_value_foreign", "flow_value_foreign",
        "t_flow_foreign_net_buy_value", "ff_naver_foreign_net_volume",
    ])
    institution_value = _first_numeric(block, [
        "ff_net_value_institution", "ff_value_institution",
        "flow_value_institution", "t_flow_institution_net_buy_value",
        "ff_naver_institution_net_volume",
    ])
    individual_value = _first_numeric(block, ["ff_net_value_individual", "ff_value_individual", "flow_value_individual", "t_flow_individual_net_buy_value"])
    derived_individual = -(foreign_value + institution_value)
    individual_value = np.where(
        np.isfinite(individual_value),
        individual_value,
        derived_individual,
    )
    foreign_ownership = _first_numeric(block, [
        "ff_foreign_ownership_rate", "ff_foreign_foreign_ownership_rate",
        "foreign_foreign_ownership_rate", "t_foreign_ownership_rate",
        "ff_naver_foreign_ownership_rate",
    ])

    per = _first_numeric(block, ["fv_per", "per", "t_value_per"])
    pbr = _first_numeric(block, ["fv_pbr", "pbr", "t_value_pbr"])
    dividend = _first_numeric(block, ["fv_dividend_yield", "dividend_yield", "t_value_dividend_yield"])

    lending_contract_shares = _first_numeric(block, ["fl_lending_contract_shares", "lending_contract_shares"])
    lending_contract_value = _first_numeric(block, ["fl_lending_contract_value", "lending_contract_value"])
    lending_repayment_shares = _first_numeric(block, ["fl_lending_repayment_shares", "lending_repayment_shares"])
    lending_repayment_value = _first_numeric(block, ["fl_lending_repayment_value", "lending_repayment_value"])
    lending_balance_shares = _first_numeric(block, ["fl_lending_balance_shares", "lending_balance_shares"])
    lending_balance_value = _first_numeric(block, ["fl_lending_balance_value", "lending_balance_value"])

    short_volume_ratio = safe_divide(short_volume, volume)
    short_value_ratio = safe_divide(short_value, trading_value)
    short_balance_to_cap = safe_divide(short_balance_value, market_cap)
    short_balance_to_shares = safe_divide(short_balance_shares, listed_shares)

    out["t_finshort_volume_ratio"] = short_volume_ratio
    out["t_finshort_value_ratio"] = short_value_ratio
    out["t_finshort_balance_to_cap"] = short_balance_to_cap
    out["t_finshort_balance_to_shares"] = short_balance_to_shares
    out["t_finshort_source_ratio"] = _to32(short_ratio)

    for window, min_periods in ((5, 3), (20, 8), (60, 20)):
        out[f"t_finshort_volume_sum_{window}"] = rolling_sum(short_volume, window, min_periods)
        out[f"t_finshort_value_sum_{window}"] = rolling_sum(short_value, window, min_periods)
        out[f"t_finshort_volume_z_{window}"] = rolling_zscore(short_volume_ratio.astype(np.float64), window, min_periods)
        out[f"t_finshort_value_z_{window}"] = rolling_zscore(short_value_ratio.astype(np.float64), window, min_periods)
        out[f"t_finshort_balance_z_{window}"] = rolling_zscore(short_balance_to_cap.astype(np.float64), window, min_periods)
        out[f"t_finshort_balance_change_{window}"] = diff_lag(short_balance_to_cap.astype(np.float64), window)
    out["t_finshort_balance_slope_20"] = rolling_slope(short_balance_to_cap.astype(np.float64), 20, 8)
    out["t_finshort_balance_slope_60"] = rolling_slope(short_balance_to_cap.astype(np.float64), 60, 20)
    out["t_finshort_balance_acceleration"] = _to32(out["t_finshort_balance_slope_20"].to_numpy(dtype=float) - out["t_finshort_balance_slope_60"].to_numpy(dtype=float))
    out["t_finshort_price_divergence_20"] = _to32(out["t_finshort_balance_change_20"].to_numpy(dtype=float) * ret5)
    out["t_finshort_crowding"] = _to32(
        np.nan_to_num(out["t_finshort_balance_z_60"].to_numpy(dtype=float), nan=0.0)
        + np.nan_to_num(out["t_finshort_volume_z_20"].to_numpy(dtype=float), nan=0.0)
    )
    out["t_finshort_squeeze_pressure"] = _to32(
        np.nan_to_num(out["t_finshort_balance_z_60"].to_numpy(dtype=float), nan=0.0)
        * np.maximum(np.nan_to_num(ret5, nan=0.0), 0.0)
        * np.maximum(np.nan_to_num(out["t_finshort_volume_z_20"].to_numpy(dtype=float), nan=0.0), 0.0)
    )
    out["t_finshort_crash_pressure"] = _to32(
        np.maximum(np.nan_to_num(out["t_finshort_balance_z_60"].to_numpy(dtype=float), nan=0.0), 0.0)
        * np.maximum(-np.nan_to_num(ret5, nan=0.0), 0.0)
    )
    out["t_finshort_stress"] = _to32(
        np.maximum(np.nan_to_num(out["t_finshort_balance_z_60"].to_numpy(dtype=float), nan=0.0), 0.0)
        * (1.0 + np.maximum(-np.nan_to_num(drawdown20, nan=0.0), 0.0))
        * (1.0 + np.maximum(np.nan_to_num(illiq, nan=0.0), 0.0))
    )

    lending_net_shares = lending_contract_shares - lending_repayment_shares
    lending_net_value = lending_contract_value - lending_repayment_value
    lending_balance_to_cap = safe_divide(lending_balance_value, market_cap)
    lending_balance_to_shares = safe_divide(lending_balance_shares, listed_shares)
    out["t_lending_net_shares"] = _to32(lending_net_shares)
    out["t_lending_net_value"] = _to32(lending_net_value)
    out["t_lending_balance_to_cap"] = lending_balance_to_cap
    out["t_lending_balance_to_shares"] = lending_balance_to_shares
    for window, min_periods in ((5, 3), (20, 8), (60, 20)):
        out[f"t_lending_contract_z_{window}"] = rolling_zscore(lending_contract_shares, window, min_periods)
        out[f"t_lending_net_z_{window}"] = rolling_zscore(lending_net_shares, window, min_periods)
        out[f"t_lending_balance_z_{window}"] = rolling_zscore(lending_balance_to_cap.astype(np.float64), window, min_periods)
        out[f"t_lending_balance_change_{window}"] = diff_lag(lending_balance_to_cap.astype(np.float64), window)
    out["t_lending_balance_slope_20"] = rolling_slope(lending_balance_to_cap.astype(np.float64), 20, 8)
    out["t_lending_balance_slope_60"] = rolling_slope(lending_balance_to_cap.astype(np.float64), 60, 20)
    out["t_lending_balance_acceleration"] = _to32(
        out["t_lending_balance_slope_20"].to_numpy(dtype=float)
        - out["t_lending_balance_slope_60"].to_numpy(dtype=float)
    )
    out["t_lending_price_divergence_20"] = _to32(
        out["t_lending_balance_change_20"].to_numpy(dtype=float) * ret5
    )
    out["t_lending_crowding"] = _to32(
        np.nan_to_num(out["t_lending_balance_z_60"].to_numpy(dtype=float), nan=0.0)
        + np.nan_to_num(out["t_lending_net_z_20"].to_numpy(dtype=float), nan=0.0)
    )
    out["t_lending_squeeze_pressure"] = _to32(
        np.maximum(np.nan_to_num(out["t_lending_balance_z_60"].to_numpy(dtype=float), nan=0.0), 0.0)
        * np.maximum(np.nan_to_num(ret5, nan=0.0), 0.0)
    )

    flow_den = np.abs(foreign_value) + np.abs(institution_value) + np.abs(individual_value)
    out["t_finflow_foreign_ratio"] = safe_divide(foreign_value, flow_den)
    out["t_finflow_institution_ratio"] = safe_divide(institution_value, flow_den)
    out["t_finflow_individual_ratio"] = safe_divide(individual_value, flow_den)
    out["t_finflow_smart_money_ratio"] = safe_divide(foreign_value + institution_value, flow_den)
    out["t_finflow_foreign_ownership"] = _to32(foreign_ownership)
    for window, min_periods in ((5, 3), (20, 8), (60, 20)):
        out[f"t_finflow_foreign_sum_{window}"] = rolling_sum(foreign_value, window, min_periods)
        out[f"t_finflow_institution_sum_{window}"] = rolling_sum(institution_value, window, min_periods)
        out[f"t_finflow_smart_sum_{window}"] = rolling_sum(foreign_value + institution_value, window, min_periods)
        out[f"t_finflow_foreign_z_{window}"] = rolling_zscore(foreign_value, window, min_periods)
        out[f"t_finflow_institution_z_{window}"] = rolling_zscore(institution_value, window, min_periods)
    out["t_finflow_foreign_streak"] = signed_streak(foreign_value)
    out["t_finflow_institution_streak"] = signed_streak(institution_value)
    out["t_finflow_foreign_ownership_change_5"] = diff_lag(foreign_ownership, 5)
    out["t_finflow_foreign_ownership_change_20"] = diff_lag(foreign_ownership, 20)
    out["t_finflow_foreign_return_corr_20"] = rolling_corr(foreign_value, ret1, 20, 8)
    out["t_finflow_institution_return_corr_20"] = rolling_corr(institution_value, ret1, 20, 8)

    out["t_finvalue_log_market_cap"] = _to32(np.where(market_cap > 0, np.log1p(market_cap), np.nan))
    out["t_finvalue_per"] = _to32(np.where(per > 0, per, np.nan))
    out["t_finvalue_pbr"] = _to32(np.where(pbr > 0, pbr, np.nan))
    out["t_finvalue_dividend_yield"] = _to32(dividend)
    out["t_finvalue_per_missing"] = _to32((~np.isfinite(per) | (per <= 0)).astype(np.float64))
    out["t_finvalue_pbr_missing"] = _to32((~np.isfinite(pbr) | (pbr <= 0)).astype(np.float64))

    out["t_fininteraction_short_x_vol"] = _to32(np.nan_to_num(out["t_finshort_stress"].to_numpy(dtype=float), nan=0.0) * np.maximum(np.nan_to_num(vol20, nan=0.0), 0.0))
    out["t_fininteraction_short_x_illiq"] = _to32(np.nan_to_num(out["t_finshort_stress"].to_numpy(dtype=float), nan=0.0) * np.maximum(np.nan_to_num(illiq, nan=0.0), 0.0))
    out["t_fininteraction_foreign_sell_x_drawdown"] = _to32(
        np.maximum(-np.nan_to_num(out["t_finflow_foreign_z_20"].to_numpy(dtype=float), nan=0.0), 0.0)
        * np.maximum(-np.nan_to_num(drawdown20, nan=0.0), 0.0)
    )
    out["t_fininteraction_short_rise_x_negative_return"] = _to32(
        np.maximum(np.nan_to_num(out["t_finshort_balance_change_20"].to_numpy(dtype=float), nan=0.0), 0.0)
        * np.maximum(-np.nan_to_num(ret5, nan=0.0), 0.0)
    )
    return out


def _rank_by_date(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for col in columns:
        if col in out.columns:
            out[f"{col}_rank_pct"] = out.groupby("date", sort=False)[col].rank(pct=True, method="average").astype("float32")
    return out


def build_finance_features(project: Path | None = None, training_path: Path | None = None, *, strict_short: bool = True) -> dict:
    paths = get_paths(project)
    warmup()
    raw_path = paths.raw_dual / "finance11h" / "finance_ticker_timeseries.parquet"
    lending_path = paths.raw_dual / "finance11h" / "stock_lending_ticker_timeseries.parquet"
    market_path = paths.raw_dual / "finance11h" / "finance_market_timeseries.parquet"
    candidates = [
        training_path,
        paths.data_root / "development" / "training_dataset_dual.parquet",
        paths.data_root / "development" / "training_dataset.parquet",
    ]
    source = next((Path(p) for p in candidates if p is not None and Path(p).exists()), None)
    if source is None:
        raise FileNotFoundError("기존 training dataset이 없습니다.")
    if not raw_path.exists():
        raise FileNotFoundError("금융 시계열이 없습니다. 04A_금융시계열_강제수집.py를 먼저 실행하세요.")

    base = normalize_date(pd.read_parquet(source))
    base["ticker"] = normalize_ticker(base["ticker"])
    raw = normalize_date(pd.read_parquet(raw_path))
    raw["ticker"] = normalize_ticker(raw["ticker"])
    raw = raw.drop(columns=["bucket", "name"], errors="ignore")
    merged_raw = base.merge(raw, on=["date", "ticker"], how="left", validate="many_to_one", suffixes=("", "_finance_raw"))
    merged_raw = _delay_short_balance_availability(merged_raw, periods=2)
    if lending_path.exists():
        lending = normalize_date(pd.read_parquet(lending_path))
        lending["ticker"] = normalize_ticker(lending["ticker"])
        lending = lending.drop(columns=["name", "bucket"], errors="ignore")
        merged_raw = merged_raw.merge(lending, on=["date", "ticker"], how="left", validate="many_to_one")
    fallback_flow_path = (
        paths.raw_dual
        / "required_data_v3"
        / "naver_flow_fallback"
        / "naver_investor_foreign_48_tickers.parquet"
    )
    fallback_flow_rows = 0
    if fallback_flow_path.exists():
        fallback = normalize_date(pd.read_parquet(fallback_flow_path))
        fallback["ticker"] = normalize_ticker(fallback["ticker"])
        fallback = fallback.rename(columns={
            "foreign_net_volume": "ff_naver_foreign_net_volume",
            "institution_net_volume": "ff_naver_institution_net_volume",
            "foreign_owned_shares": "ff_naver_foreign_owned_shares",
            "foreign_ownership_rate": "ff_naver_foreign_ownership_rate",
        })
        keep = [
            "date",
            "ticker",
            "ff_naver_foreign_net_volume",
            "ff_naver_institution_net_volume",
            "ff_naver_foreign_owned_shares",
            "ff_naver_foreign_ownership_rate",
        ]
        fallback = (
            fallback[[column for column in keep if column in fallback.columns]]
            .sort_values(["ticker", "date"])
            .drop_duplicates(["ticker", "date"], keep="last")
        )
        fallback_flow_rows = len(fallback)
        merged_raw = merged_raw.merge(
            fallback,
            on=["date", "ticker"],
            how="left",
            validate="many_to_one",
        )

    blocks = [_build_ticker_block(g) for _, g in merged_raw.groupby("ticker", sort=False)]
    ticker_features = pd.concat(blocks, ignore_index=True, sort=False)
    ticker_features = _rank_by_date(ticker_features, ["t_finvalue_log_market_cap", "t_finvalue_per", "t_finvalue_pbr", "t_finshort_balance_to_cap"])

    # Universe financial features are true date-level aggregates over the available panel.
    temp = merged_raw[["date", "ticker"]].merge(ticker_features, on=["date", "ticker"], how="left", validate="one_to_one")
    aggregate_rows = []
    for date, g in temp.groupby("date", sort=False):
        short_balance = pd.to_numeric(g.get("t_finshort_balance_to_cap"), errors="coerce")
        short_volume = pd.to_numeric(g.get("t_finshort_volume_ratio"), errors="coerce")
        foreign = pd.to_numeric(g.get("t_finflow_foreign_ratio"), errors="coerce")
        institution = pd.to_numeric(g.get("t_finflow_institution_ratio"), errors="coerce")
        short_change = pd.to_numeric(g.get("t_finshort_balance_change_20"), errors="coerce")
        aggregate_rows.append({
            "date": date,
            "u_finshort_balance_median": short_balance.median(),
            "u_finshort_balance_mean": short_balance.mean(),
            "u_finshort_volume_mean": short_volume.mean(),
            "u_finshort_balance_rising_ratio": short_change.gt(0).mean() if short_change.notna().any() else np.nan,
            "u_finshort_crowded_ratio": short_balance.gt(short_balance.quantile(0.8)).mean() if short_balance.notna().sum() >= 5 else np.nan,
            "u_lending_balance_mean": pd.to_numeric(g.get("t_lending_balance_to_cap"), errors="coerce").mean(),
            "u_lending_balance_rising_ratio": pd.to_numeric(g.get("t_lending_balance_change_20"), errors="coerce").gt(0).mean() if pd.to_numeric(g.get("t_lending_balance_change_20"), errors="coerce").notna().any() else np.nan,
            "u_lending_crowded_ratio": pd.to_numeric(g.get("t_lending_crowding"), errors="coerce").gt(0).mean() if pd.to_numeric(g.get("t_lending_crowding"), errors="coerce").notna().any() else np.nan,
            "u_finflow_foreign_sell_ratio": foreign.lt(0).mean() if foreign.notna().any() else np.nan,
            "u_finflow_institution_sell_ratio": institution.lt(0).mean() if institution.notna().any() else np.nan,
            "u_finflow_foreign_mean": foreign.mean(),
            "u_finflow_institution_mean": institution.mean(),
        })
    universe_features = pd.DataFrame(aggregate_rows).sort_values("date")
    for col in [c for c in universe_features.columns if c.startswith("u_lending_")]:
        values = pd.to_numeric(universe_features[col], errors="coerce").to_numpy(dtype=np.float64)
        universe_features[f"{col}_z20"] = rolling_zscore(values, 20, 8)
        universe_features[f"{col}_change5"] = diff_lag(values, 5)

    if market_path.exists():
        market = normalize_date(pd.read_parquet(market_path))
        universe_features = universe_features.merge(market, on="date", how="left", validate="one_to_one")
        numeric_market = []
        for column in market.columns:
            if column in {"date", "available_from", "observation_date"} or column.startswith("source_"):
                continue
            values = pd.to_numeric(market[column], errors="coerce")
            if values.notna().sum() >= 10:
                numeric_market.append(column)
        for col in numeric_market:
            values = pd.to_numeric(universe_features[col], errors="coerce").to_numpy(dtype=np.float64)
            safe = "".join(ch if ch.isalnum() else "_" for ch in col.lower())
            universe_features[f"u_finmarket_{safe}_z20"] = rolling_zscore(values, 20, 8)
            universe_features[f"u_finmarket_{safe}_change5"] = diff_lag(values, 5)

    paths.feature_dual.mkdir(parents=True, exist_ok=True)
    atomic_parquet(ticker_features, paths.feature_dual / "finance11h_ticker_features.parquet")
    atomic_parquet(universe_features, paths.feature_dual / "finance11h_universe_features.parquet")

    final = base.merge(universe_features, on="date", how="left", validate="many_to_one")
    final = final.merge(ticker_features, on=["date", "ticker"], how="left", validate="one_to_one")
    output = paths.data_root / "development" / "training_dataset_finance11h.parquet"
    atomic_parquet(final, output)

    ticker_catalog = {
        "t_financial_shorting": sorted(c for c in ticker_features.columns if c.startswith("t_finshort_")),
        "t_financial_flow": sorted(c for c in ticker_features.columns if c.startswith("t_finflow_")),
        "t_financial_valuation": sorted(c for c in ticker_features.columns if c.startswith("t_finvalue_")),
        "t_financial_interaction": sorted(c for c in ticker_features.columns if c.startswith("t_fininteraction_")),
        "t_stock_lending": sorted(c for c in ticker_features.columns if c.startswith("t_lending_")),
    }
    universe_catalog = {
        "u_financial_shorting": sorted(c for c in universe_features.columns if c.startswith("u_finshort_")),
        "u_financial_flow": sorted(c for c in universe_features.columns if c.startswith("u_finflow_")),
        "u_financial_market": sorted(c for c in universe_features.columns if c.startswith("u_finmarket_")),
        "u_stock_lending": sorted(c for c in universe_features.columns if c.startswith("u_lending_")),
    }
    atomic_json(ticker_catalog, paths.feature_dual / "feature_catalog_finance11h_ticker.json")
    atomic_json(universe_catalog, paths.feature_dual / "feature_catalog_finance11h_universe.json")

    quality_rows = []
    for namespace, frame, keys in [
        ("ticker", ticker_features, {"date", "ticker"}),
        ("universe", universe_features, {"date"}),
    ]:
        for col in frame.columns:
            if col in keys:
                continue
            values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            quality_rows.append({
                "namespace": namespace, "feature": col, "rows": len(values),
                "missing_ratio": float(values.isna().mean()), "unique_count": int(values.nunique(dropna=True)),
                "non_null_count": int(values.notna().sum()), "status": "valid" if values.notna().sum() >= 200 and values.nunique(dropna=True) >= 2 else "invalid",
            })
    quality = pd.DataFrame(quality_rows)
    atomic_csv(quality, paths.feature_dual / "finance11h_feature_quality.csv")

    short_features = ticker_catalog["t_financial_shorting"]
    valid_short = quality.loc[quality["feature"].isin(short_features) & quality["status"].eq("valid")]
    if strict_short and len(valid_short) < 8:
        raise RuntimeError(f"필수 공매도 피처가 부족합니다: valid={len(valid_short)}, required>=8")

    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "source_dataset": str(source), "output_dataset": str(output),
        "rows": len(final), "tickers": int(final["ticker"].nunique()),
        "ticker_finance_features": sum(len(v) for v in ticker_catalog.values()),
        "universe_finance_features": sum(len(v) for v in universe_catalog.values()),
        "valid_short_features": int(len(valid_short)),
        "fallback_flow_rows": int(fallback_flow_rows),
        "flow_source": (
            "naver_fallback_exploratory_not_strict"
            if fallback_flow_rows
            else "official_or_none"
        ),
        "short_balance_availability_lag_trading_days": 2,
        "numba_enabled": True,
    }
    atomic_json(summary, paths.feature_dual / "finance11h_feature_summary.json")
    return summary
