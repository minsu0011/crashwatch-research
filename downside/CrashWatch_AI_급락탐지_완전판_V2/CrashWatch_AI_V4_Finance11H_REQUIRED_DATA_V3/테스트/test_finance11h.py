from __future__ import annotations

import os

import numpy as np
import pandas as pd

from dual_ablation.finance11h.features import (
    _build_ticker_block,
    _delay_short_balance_availability,
)
from dual_ablation.finance11h.numba_kernels import (
    diff_lag,
    pct_change,
    rolling_slope,
    rolling_sum,
    rolling_zscore,
    safe_divide,
    signed_streak,
)
from dual_ablation.finance11h.model import ModelConfig
from dual_ablation.finance11h.runner import (
    Experiment,
    Finance11HRunner,
    _exact_sign_p,
    _hash_file_metadata,
)


def test_numba_rolling_matches_pandas():
    values = np.arange(1.0, 101.0, dtype=np.float64)
    expected_sum = pd.Series(values).rolling(20, min_periods=8).sum().to_numpy()
    actual_sum = rolling_sum(values, 20, 8)
    np.testing.assert_allclose(actual_sum, expected_sum, rtol=1e-5, atol=1e-5, equal_nan=True)
    expected_z = ((pd.Series(values) - pd.Series(values).rolling(20, min_periods=8).mean()) / pd.Series(values).rolling(20, min_periods=8).std(ddof=0)).to_numpy()
    actual_z = rolling_zscore(values, 20, 8)
    np.testing.assert_allclose(actual_z, expected_z, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_basic_numba_kernels():
    x = np.array([1.0, 2.0, 4.0, np.nan, 8.0], dtype=np.float64)
    y = np.array([1.0, 2.0, 0.0, 4.0, 2.0], dtype=np.float64)
    assert np.isnan(safe_divide(x, y)[2])
    assert np.isclose(diff_lag(x, 1)[2], 2.0)
    assert np.isclose(pct_change(x, 1)[2], 1.0)
    streak = signed_streak(np.array([1.0, 2.0, -1.0, -2.0, 0.0]))
    np.testing.assert_array_equal(streak, np.array([1, 2, -1, -2, 0], dtype=np.float32))
    slope = rolling_slope(np.arange(20.0), 10, 5)
    assert np.isclose(slope[-1], 1.0)


def _synthetic_block(rows: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2024-01-01", periods=rows)
    close = 100 * np.cumprod(1 + rng.normal(0, 0.01, rows))
    volume = rng.integers(100_000, 2_000_000, rows).astype(float)
    value = volume * close
    short_volume = volume * np.clip(rng.normal(0.04, 0.015, rows), 0, 0.15)
    short_balance = np.cumsum(rng.normal(0, 1000, rows)) + 1_000_000
    foreign = rng.normal(0, 2e9, rows)
    institution = rng.normal(0, 1.5e9, rows)
    individual = -(foreign + institution) + rng.normal(0, 2e8, rows)
    return pd.DataFrame({
        "date": dates, "ticker": "005930", "close": close, "volume": volume,
        "trading_value": value, "fv_market_cap": 4e14,
        "fs_volume_short_volume": short_volume,
        "fs_value_short_volume": short_volume * close,
        "fs_balance_short_balance_shares": short_balance,
        "fs_balance_short_balance_value": short_balance * close,
        "fs_balance_listed_shares": 5.9e9,
        "ff_value_foreign": foreign, "ff_value_institution": institution,
        "ff_value_individual": individual, "ff_foreign_foreign_ownership_rate": 50 + rng.normal(0, .1, rows),
        "fv_per": 12 + rng.normal(0, .5, rows), "fv_pbr": 1.4 + rng.normal(0, .05, rows),
        "fv_dividend_yield": 2.2 + rng.normal(0, .05, rows),
        "t_price_ret_1": pd.Series(close).pct_change(),
        "t_price_ret_5": pd.Series(close).pct_change(5),
        "t_price_drawdown_20": pd.Series(close) / pd.Series(close).rolling(20).max() - 1,
        "t_vol_realized_20": pd.Series(close).pct_change().rolling(20).std(),
        "t_micro_amihud_20": pd.Series(np.abs(pd.Series(close).pct_change()) / value).rolling(20).mean(),
    })


def test_finance_feature_block_has_mandatory_short_features():
    out = _build_ticker_block(_synthetic_block())
    mandatory = [
        "t_finshort_volume_ratio", "t_finshort_balance_to_cap",
        "t_finshort_balance_change_20", "t_finshort_balance_slope_20",
        "t_finshort_crowding", "t_finshort_squeeze_pressure", "t_finshort_stress",
    ]
    assert set(mandatory) <= set(out.columns)
    assert out[mandatory].notna().sum().sum() > 100


def test_future_mutation_does_not_change_past_features():
    raw = _synthetic_block()
    first = _build_ticker_block(raw)
    mutated = raw.copy()
    cutoff = 120
    mutated.loc[cutoff:, "fs_balance_short_balance_value"] *= 100
    mutated.loc[cutoff:, "fs_volume_short_volume"] *= 50
    second = _build_ticker_block(mutated)
    cols = [c for c in first.columns if c.startswith("t_finshort_")]
    np.testing.assert_allclose(
        first.loc[: cutoff - 1, cols].to_numpy(dtype=float),
        second.loc[: cutoff - 1, cols].to_numpy(dtype=float),
        rtol=1e-5, atol=1e-5, equal_nan=True,
    )


def test_short_balance_is_available_after_two_trading_days():
    frame = pd.DataFrame({
        "date": pd.bdate_range("2024-01-01", periods=5),
        "ticker": "005930",
        "fs_short_balance_shares": [10, 20, 30, 40, 50],
        "fs_short_balance_value": [100, 200, 300, 400, 500],
        "fs_short_balance_ratio": [0.1, 0.2, 0.3, 0.4, 0.5],
    })
    delayed = _delay_short_balance_availability(frame)
    assert delayed["fs_short_balance_shares"].iloc[:2].isna().all()
    assert delayed["fs_short_balance_shares"].iloc[2:].tolist() == [10, 20, 30]
    assert delayed["fs_short_balance_value"].iloc[2:].tolist() == [100, 200, 300]


def test_naver_flow_fallback_produces_exploratory_flow_features():
    raw = _synthetic_block()
    raw = raw.drop(columns=[
        "ff_value_foreign",
        "ff_value_institution",
        "ff_value_individual",
        "ff_foreign_foreign_ownership_rate",
    ])
    raw["ff_naver_foreign_net_volume"] = np.linspace(-1000, 1000, len(raw))
    raw["ff_naver_institution_net_volume"] = np.linspace(500, -500, len(raw))
    raw["ff_naver_foreign_ownership_rate"] = np.linspace(0.50, 0.55, len(raw))
    result = _build_ticker_block(raw)
    assert result["t_finflow_foreign_ratio"].notna().sum() == len(raw)
    assert result["t_finflow_institution_ratio"].notna().sum() == len(raw)
    assert result["t_finflow_foreign_ownership"].notna().sum() == len(raw)


def test_dataset_signature_detects_same_size_same_mtime_content_change(
    tmp_path,
):
    path = tmp_path / "dataset.parquet"
    path.write_bytes(b"abcdefgh")
    original_stat = path.stat()
    first = _hash_file_metadata(path)
    path.write_bytes(b"abcdEfgh")
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    second = _hash_file_metadata(path)
    assert first != second


def test_cache_namespace_changes_cache_key(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    first = Finance11HRunner(project, None, cache_namespace="full")
    second = Finance11HRunner(project, None, cache_namespace="game")
    first.dataset_signature = second.dataset_signature = "same-dataset"
    experiment = Experiment("baseline", "baseline", "none")
    config = ModelConfig("test", n_estimators=1)
    first_path = first._cache_path(experiment, 0, 17, config, ["a"])
    second_path = second._cache_path(experiment, 0, 17, config, ["a"])
    assert first_path.name != second_path.name


def test_exact_sign_test_does_not_invent_sample_size():
    assert np.isclose(_exact_sign_p(np.array([1.0, 1.0, 1.0])), 0.25)
    assert np.isclose(_exact_sign_p(np.array([1.0, -1.0, 1.0, -1.0])), 1.0)
