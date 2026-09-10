from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dual_ablation.config import get_paths, load_baskets
from dual_ablation.experiment.runner import ExperimentSpec, _apply_mask, build_specs
from dual_ablation.experiment.splits import make_walk_forward_folds
from dual_ablation.experiment.statistics import paired_delta_table, summarize_deltas
from dual_ablation.features.ticker import _add_peer_features, _base_ticker_features


PROJECT = Path(__file__).resolve().parents[1]


def test_sector_baskets_have_six_companies_each():
    baskets = load_baskets(get_paths(PROJECT))
    counts = baskets.groupby("bucket")["ticker"].nunique()
    assert len(counts) >= 8
    assert counts.eq(6).all()
    assert baskets["ticker"].str.len().eq(6).all()


def test_walk_forward_purge_is_20_trading_days():
    dates = pd.bdate_range("2019-01-01", periods=1200)
    folds = make_walk_forward_folds(dates, n_folds=4, validation_days=75, purge_days=20, min_train_days=500)
    assert len(folds) == 4
    for fold in folds:
        train = fold["train_dates_index"]
        val = fold["validation_dates_index"]
        positions = {date: i for i, date in enumerate(dates)}
        assert positions[val[0]] - positions[train[-1]] - 1 == 20
        assert len(val) == 75


def test_ticker_mask_changes_only_target_rows():
    df = pd.DataFrame({
        "ticker": ["005930", "005930", "000660", "000660"],
        "bucket": ["semiconductor"] * 4,
        "t_flow_x": [1.0, 2.0, 3.0, 4.0],
        "t_flow_is_available": [1, 1, 1, 1],
    })
    spec = ExperimentSpec("x", "ticker", "ticker_mask", "t_investor_flow", "semiconductor", "005930")
    out = _apply_mask(df, spec, ["t_flow_x", "t_flow_is_available"])
    assert out.loc[out["ticker"].eq("005930"), "t_flow_x"].isna().all()
    assert out.loc[out["ticker"].eq("005930"), "t_flow_is_available"].eq(0).all()
    assert out.loc[out["ticker"].eq("000660"), "t_flow_x"].tolist() == [3.0, 4.0]


def test_feature_specs_include_universe_bucket_and_ticker_modes():
    baskets = load_baskets(get_paths(PROJECT)).head(6)
    u = {"u_market": ["u_market_x"]}
    t = {"t_flow": ["t_flow_x"], "t_event": ["t_event_x"]}
    specs = build_specs(u, t, baskets, {"universe", "ticker_global", "bucket", "ticker"})
    names = {s.ablation_mode for s in specs}
    assert {"none", "global_drop", "ticker_global_drop", "bucket_mask", "ticker_mask"}.issubset(names)


def test_metric_delta_sign_and_summary():
    metrics = pd.DataFrame([
        {"experiment": "baseline", "fold": 0, "seed": 1, "scope_type": "all", "scope_value": "all_validation", "pr_auc": .30, "roc_auc": .60, "brier": .20, "logloss": .60, "alert_precision": .30, "alert_recall": .10},
        {"experiment": "drop", "fold": 0, "seed": 1, "scope_type": "all", "scope_value": "all_validation", "pr_auc": .25, "roc_auc": .55, "brier": .22, "logloss": .65, "alert_precision": .25, "alert_recall": .08, "namespace": "universe", "ablation_mode": "global_drop", "target_group": "u_market", "target_bucket": None, "target_ticker": None},
    ])
    deltas = paired_delta_table(metrics)
    assert np.isclose(deltas.iloc[0]["pr_auc_loss_when_removed"], .05)
    assert np.isclose(deltas.iloc[0]["brier_increase_when_removed"], .02)
    summary = summarize_deltas(deltas, bootstrap_samples=50, seed=1)
    assert summary.iloc[0]["pr_auc_loss_when_removed_mean"] > 0


def test_empty_scope_with_nan_baseline_is_still_paired():
    metrics = pd.DataFrame([
        {
            "experiment": "baseline", "fold": 0, "seed": 1,
            "scope_type": "ticker", "scope_value": "999999",
            "pr_auc": np.nan, "roc_auc": np.nan, "brier": np.nan,
            "logloss": np.nan, "alert_precision": np.nan, "alert_recall": np.nan,
        },
        {
            "experiment": "drop", "fold": 0, "seed": 1,
            "scope_type": "ticker", "scope_value": "999999",
            "pr_auc": np.nan, "roc_auc": np.nan, "brier": np.nan,
            "logloss": np.nan, "alert_precision": np.nan, "alert_recall": np.nan,
            "namespace": "ticker", "ablation_mode": "ticker_global_drop",
            "target_group": "t_test", "target_bucket": None, "target_ticker": None,
        },
    ])
    deltas = paired_delta_table(metrics)
    assert len(deltas) == 1
    assert np.isnan(deltas.iloc[0]["pr_auc_loss_when_removed"])


def test_ticker_feature_generation_and_peer_leave_one_out():
    dates = pd.bdate_range("2022-01-03", periods=140)
    rows = []
    for ticker, offset in [("005930", 0.0), ("000660", 0.5)]:
        for i, date in enumerate(dates):
            close = 100 + offset + i * 0.1 + np.sin(i / 5)
            rows.append({
                "date": date, "ticker": ticker, "name": ticker, "bucket": "semiconductor", "market": "KOSPI", "role": "x",
                "open": close * .995, "high": close * 1.01, "low": close * .99, "close": close,
                "volume": 1_000_000 + i * 1000, "trading_value": close * (1_000_000 + i * 1000),
                "market_cap": close * 1_000_000_000, "listed_shares": 1_000_000_000,
                "per": 10 + i / 100, "pbr": 1.2, "dividend_yield": 2.0,
                "flow_value_외국인합계": 1000 + i, "flow_value_기관합계": 500 - i,
                "flow_value_개인": -1500, "foreign_지분율": 50 + i / 1000,
                "short_volume_공매도": 10000 + i, "short_volume_비중": 1.0,
                "short_balance_잔고금액": 100_000_000 + i * 1000,
            })
    base = _base_ticker_features(pd.DataFrame(rows))
    assert "t_price_ret_20" in base.columns
    assert "t_short_balance_to_cap" in base.columns
    peer = _add_peer_features(base)
    assert peer["t_peer_leave_one_out_ret_1"].notna().sum() > 100
