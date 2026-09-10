from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "03B_종목별_그룹_이탈테스트.py"
SPEC = importlib.util.spec_from_file_location("ticker_ablation_under_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_single_class_keeps_rank_metrics_nan_but_brier_available() -> None:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=4),
            "ticker": ["A"] * 4,
            "label_abs_crash_20": [0, 0, 0, 0],
            "pred": [0.1, 0.2, 0.3, 0.4],
            "alert_top3": [0, 0, 0, 1],
        }
    )
    metrics = MODULE.safe_metrics(df)
    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["roc_auc"])
    assert np.isfinite(metrics["brier"])
    assert np.isfinite(metrics["logloss"])


def test_validation_cap_preserves_sentinel() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01"] * 10),
            "ticker": ["S"] + list("ABCDEFGHI"),
            "label_abs_crash_20": [0] * 10,
        }
    )
    capped = MODULE.cap_validation_rows(df, max_rows=3, sentinel_tickers={"S"})
    assert "S" in set(capped["ticker"])


def test_ablation_sign_definition() -> None:
    aggregated = pd.DataFrame(
        [
            {
                "experiment": "baseline", "scope": "ticker", "ticker": "A", "name": "A",
                "market": "K", "sector": "X", "priority": "CORE", "pr_auc": 0.20,
                "roc_auc": 0.60, "brier": 0.10, "logloss": 0.30,
                "alert_recall_top3_daily": 0.40, "alert_precision_top3_daily": 0.50,
                "fold_count": 4, "rows": 10, "positives": 2, "positive_rate": 0.2,
                "mean_pred": 0.2, "alert_count": 1, "alerts_per_250d": 10,
            },
            {
                "experiment": "drop_group::g", "scope": "ticker", "ticker": "A", "name": "A",
                "market": "K", "sector": "X", "priority": "CORE", "pr_auc": 0.15,
                "roc_auc": 0.55, "brier": 0.12, "logloss": 0.35,
                "alert_recall_top3_daily": 0.30, "alert_precision_top3_daily": 0.40,
                "fold_count": 4, "rows": 10, "positives": 2, "positive_rate": 0.2,
                "mean_pred": 0.2, "alert_count": 1, "alerts_per_250d": 10,
            },
        ]
    )
    result = MODULE.compare_with_baseline(aggregated).iloc[0]
    assert np.isclose(result["pr_auc_loss_when_removed"], 0.05)
    assert np.isclose(result["brier_increase_when_removed"], 0.02)


def test_walk_forward_has_required_purge_gap() -> None:
    dates = pd.bdate_range("2018-01-01", periods=1200)
    folds = MODULE.make_walk_forward_folds(dates, cv_folds=4, valid_days=75, purge_days=20)
    for fold in folds:
        train_end_pos = dates.get_loc(fold["train_end"])
        valid_start_pos = dates.get_loc(fold["valid_start"])
        assert valid_start_pos - train_end_pos - 1 == 20


def test_daily_top3_alert_marks_at_least_one_per_date() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2026-01-01")] * 10,
            "ticker": [f"T{i}" for i in range(10)],
            "label_abs_crash_20": [0] * 10,
            "pred": np.linspace(0, 1, 10),
        }
    )
    out = MODULE.add_daily_top3_alert(df)
    assert out["alert_top3"].sum() == 1
