from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dual_ablation.config import get_paths
from dual_ablation.features.research import (
    _corwin_schultz_spread,
    _daily_panel_for_research,
    _network_timeseries,
    add_research_ticker_features,
)


def synthetic_raw(days: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=days)
    rows = []
    for ticker in ["005930", "000660", "035420", "105560", "068270", "005380"]:
        ret = rng.normal(0, 0.018, days)
        close = 100 * np.cumprod(1 + ret)
        open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, days))
        high = np.maximum(open_, close) * (1 + rng.random(days) * 0.015)
        low = np.minimum(open_, close) * (1 - rng.random(days) * 0.015)
        for i, date in enumerate(dates):
            rows.append({
                "date": date, "ticker": ticker, "open": open_[i], "high": high[i], "low": low[i],
                "close": close[i], "volume": 1_000_000 + rng.integers(0, 200_000),
                "trading_value": close[i] * 1_000_000, "market_cap": close[i] * 100_000_000,
            })
    return pd.DataFrame(rows)


def test_corwin_schultz_nonnegative() -> None:
    raw = synthetic_raw(40).query("ticker == '005930'")
    spread = _corwin_schultz_spread(raw["high"], raw["low"])
    assert (spread.dropna() >= 0).all()
    assert (spread.dropna() <= 2).all()


def test_network_features_are_backward_rolling() -> None:
    raw = synthetic_raw(100)
    panel = _daily_panel_for_research(raw)
    first = _network_timeseries(panel)
    altered = raw.copy()
    cutoff = sorted(raw["date"].unique())[70]
    altered.loc[altered["date"] > cutoff, "close"] *= 3.0
    second = _network_timeseries(_daily_panel_for_research(altered))
    cols = [c for c in first.columns if c != "date"]
    left = first.loc[first["date"] <= cutoff, cols].reset_index(drop=True)
    right = second.loc[second["date"] <= cutoff, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_ticker_research_features_do_not_use_future(tmp_path: Path) -> None:
    raw = synthetic_raw(140)
    dates = sorted(raw["date"].unique())
    market = pd.DataFrame({"date": dates, "u_market_kospi_ret_1": np.sin(np.arange(len(dates)) / 10) / 100})
    base = raw[["date", "ticker"]].copy()
    paths = get_paths(tmp_path)
    paths.raw_dual.mkdir(parents=True, exist_ok=True)
    first = add_research_ticker_features(base, raw, market, paths)

    cutoff = dates[90]
    changed = raw.copy()
    changed.loc[changed["date"] > cutoff, ["open", "high", "low", "close", "volume", "trading_value"]] *= 5
    second = add_research_ticker_features(base, changed, market, paths)
    cols = [c for c in first.columns if c.startswith(("t_rangevol_", "t_micro_", "t_taildep_", "t_pressure_", "t_limit_"))]
    left = first.loc[first["date"] <= cutoff, ["date", "ticker"] + cols].reset_index(drop=True)
    right = second.loc[second["date"] <= cutoff, ["date", "ticker"] + cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_feature_families_created(tmp_path: Path) -> None:
    raw = synthetic_raw(180)
    dates = sorted(raw["date"].unique())
    market = pd.DataFrame({"date": dates, "u_market_kospi_ret_1": np.random.default_rng(7).normal(0, 0.01, len(dates))})
    paths = get_paths(tmp_path)
    paths.raw_dual.mkdir(parents=True, exist_ok=True)
    out = add_research_ticker_features(raw[["date", "ticker"]], raw, market, paths)
    for prefix in ["t_rangevol_", "t_micro_", "t_taildep_", "t_pressure_", "t_limit_"]:
        cols = [c for c in out.columns if c.startswith(prefix)]
        assert cols
        assert out[cols].notna().sum().sum() > 0
