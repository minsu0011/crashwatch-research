from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ProjectPaths, get_paths
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_date, normalize_ticker
from .numba_kernels import (
    NUMBA_AVAILABLE,
    diff_lag,
    numba_diagnostics,
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
    for col in candidates:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
            return np.ascontiguousarray(values, dtype=np.float64)
    return np.full(len(df), np.nan, dtype=np.float64)


def _series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _to32(values: np.ndarray | pd.Series) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


def _required_sum(*values: np.ndarray | pd.Series) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    valid = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    out = np.full(arrays[0].shape, np.nan, dtype=np.float32)
    if valid.any():
        out[valid] = np.sum(np.vstack([value[valid] for value in arrays]), axis=0).astype(np.float32)
    return out


def _required_product(*values: np.ndarray | pd.Series) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    valid = np.logical_and.reduce([np.isfinite(value) for value in arrays])
    out = np.full(arrays[0].shape, np.nan, dtype=np.float32)
    if valid.any():
        product = np.ones(int(valid.sum()), dtype=np.float64)
        for value in arrays:
            product *= value[valid]
        out[valid] = product.astype(np.float32)
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

    short_volume = _first_numeric(block, ["fs_status_short_volume", "fs_volume_short_volume", "fs_volume_공매도"])
    short_value = _first_numeric(block, ["fs_status_short_value", "fs_value_short_value", "fs_value_공매도금액"])
    short_ratio = _first_numeric(block, ["fs_volume_short_ratio", "fs_value_short_ratio", "fs_balance_short_ratio", "fs_status_short_ratio"])
    short_balance_shares = _first_numeric(block, ["fs_balance_short_balance_shares", "fs_status_short_balance_shares"])
    short_balance_value = _first_numeric(block, ["fs_balance_short_balance_value", "fs_status_short_balance_value"])
    listed_shares = _first_numeric(block, ["fs_balance_listed_shares", "fv_listed_shares", "listed_shares"])

    foreign_value = _first_numeric(block, ["ff_value_foreign", "flow_value_foreign", "t_flow_foreign_net_buy_value"])
    institution_value = _first_numeric(block, ["ff_value_institution", "flow_value_institution", "t_flow_institution_net_buy_value"])
    individual_value = _first_numeric(block, ["ff_value_individual", "flow_value_individual", "t_flow_individual_net_buy_value"])
    foreign_ownership = _first_numeric(block, ["ff_foreign_foreign_ownership_rate", "foreign_foreign_ownership_rate", "t_foreign_ownership_rate"])

    per = _first_numeric(block, ["fv_per", "per", "t_value_per"])
    pbr = _first_numeric(block, ["fv_pbr", "pbr", "t_value_pbr"])
    dividend = _first_numeric(block, ["fv_dividend_yield", "dividend_yield", "t_value_dividend_yield"])

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
    balance_z60 = out["t_finshort_balance_z_60"].to_numpy(dtype=np.float64)
    volume_z20 = out["t_finshort_volume_z_20"].to_numpy(dtype=np.float64)
    out["t_finshort_crowding"] = _required_sum(balance_z60, volume_z20)
    out["t_finshort_squeeze_pressure"] = _required_product(
        balance_z60, np.maximum(ret5, 0.0), np.maximum(volume_z20, 0.0)
    )
    out["t_finshort_crash_pressure"] = _required_product(
        np.maximum(balance_z60, 0.0), np.maximum(-ret5, 0.0)
    )
    out["t_finshort_stress"] = _required_product(
        np.maximum(balance_z60, 0.0),
        1.0 + np.maximum(-drawdown20, 0.0),
        1.0 + np.maximum(illiq, 0.0),
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

    stress = out["t_finshort_stress"].to_numpy(dtype=np.float64)
    foreign_z20 = out["t_finflow_foreign_z_20"].to_numpy(dtype=np.float64)
    balance_change20 = out["t_finshort_balance_change_20"].to_numpy(dtype=np.float64)
    out["t_fininteraction_short_x_vol"] = _required_product(stress, np.maximum(vol20, 0.0))
    out["t_fininteraction_short_x_illiq"] = _required_product(stress, np.maximum(illiq, 0.0))
    out["t_fininteraction_foreign_sell_x_drawdown"] = _required_product(
        np.maximum(-foreign_z20, 0.0), np.maximum(-drawdown20, 0.0)
    )
    out["t_fininteraction_short_rise_x_negative_return"] = _required_product(
        np.maximum(balance_change20, 0.0), np.maximum(-ret5, 0.0)
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
    numba_benchmark = warmup()
    atomic_json(numba_benchmark, paths.feature_dual / "finance11h_numba_benchmark.json")
    raw_path = paths.raw_dual / "finance11h" / "finance_ticker_timeseries.parquet"
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
            "u_finflow_foreign_sell_ratio": foreign.lt(0).mean() if foreign.notna().any() else np.nan,
            "u_finflow_institution_sell_ratio": institution.lt(0).mean() if institution.notna().any() else np.nan,
            "u_finflow_foreign_mean": foreign.mean(),
            "u_finflow_institution_mean": institution.mean(),
        })
    universe_features = pd.DataFrame(aggregate_rows).sort_values("date")
    for column in ("u_finshort_balance_median", "u_finshort_balance_mean", "u_finshort_volume_mean"):
        values = np.ascontiguousarray(
            pd.to_numeric(universe_features[column], errors="coerce").to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
        universe_features[f"{column}_z20"] = rolling_zscore(values, 20, 8)
        universe_features[f"{column}_z60"] = rolling_zscore(values, 60, 20)
        universe_features[f"{column}_change5"] = diff_lag(values, 5)
        universe_features[f"{column}_change20"] = diff_lag(values, 20)
    universe_features["u_finshort_stress"] = _required_sum(
        np.maximum(universe_features["u_finshort_balance_mean_z60"].to_numpy(dtype=float), 0.0),
        np.maximum(universe_features["u_finshort_volume_mean_z20"].to_numpy(dtype=float), 0.0),
        universe_features["u_finshort_balance_rising_ratio"].to_numpy(dtype=float),
    )
    for column in ("u_finflow_foreign_mean", "u_finflow_institution_mean"):
        values = np.ascontiguousarray(
            pd.to_numeric(universe_features[column], errors="coerce").to_numpy(dtype=np.float64),
            dtype=np.float64,
        )
        universe_features[f"{column}_z20"] = rolling_zscore(values, 20, 8)
        universe_features[f"{column}_sum20"] = rolling_sum(values, 20, 8)

    if market_path.exists():
        market = normalize_date(pd.read_parquet(market_path))
        universe_features = universe_features.merge(market, on="date", how="left", validate="one_to_one")
        numeric_market = [c for c in market.columns if c != "date"]
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
    }
    universe_catalog = {
        "u_financial_shorting": sorted(c for c in universe_features.columns if c.startswith("u_finshort_")),
        "u_financial_flow": sorted(c for c in universe_features.columns if c.startswith("u_finflow_")),
        "u_financial_market": sorted(c for c in universe_features.columns if c.startswith("u_finmarket_")),
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

    valid_features = set(quality.loc[quality["status"].eq("valid"), "feature"])
    required_groups = {
        "t_financial_shorting": (ticker_catalog["t_financial_shorting"], 8),
        "u_financial_shorting": (universe_catalog["u_financial_shorting"], 8),
        "t_financial_flow": (ticker_catalog["t_financial_flow"], 1),
        "u_financial_flow": (universe_catalog["u_financial_flow"], 1),
        "t_financial_valuation": (ticker_catalog["t_financial_valuation"], 1),
        "t_financial_interaction": (ticker_catalog["t_financial_interaction"], 1),
    }
    valid_group_counts = {
        group: len(set(features) & valid_features)
        for group, (features, _) in required_groups.items()
    }
    if strict_short:
        insufficient = {
            group: {"valid": valid_group_counts[group], "required": minimum}
            for group, (_, minimum) in required_groups.items()
            if valid_group_counts[group] < minimum
        }
        if insufficient:
            raise RuntimeError(f"필수 금융 피처 그룹이 부족합니다: {insufficient}")

    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "source_dataset": str(source), "output_dataset": str(output),
        "rows": len(final), "tickers": int(final["ticker"].nunique()),
        "ticker_finance_features": sum(len(v) for v in ticker_catalog.values()),
        "universe_finance_features": sum(len(v) for v in universe_catalog.values()),
        "valid_short_features": int(valid_group_counts["t_financial_shorting"]),
        "valid_finance_features_by_group": valid_group_counts,
        "numba_enabled": NUMBA_AVAILABLE,
        "numba_nopython_compiled": bool(numba_diagnostics()["nopython_compiled"]),
        "numba_benchmark": numba_benchmark,
    }
    atomic_json(summary, paths.feature_dual / "finance11h_feature_summary.json")
    return summary
