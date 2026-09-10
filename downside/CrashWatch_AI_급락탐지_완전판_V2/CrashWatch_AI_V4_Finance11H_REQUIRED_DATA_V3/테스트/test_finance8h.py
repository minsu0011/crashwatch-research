from __future__ import annotations

import numpy as np
import pandas as pd

from dual_ablation.finance8h.runner import (
    PRIMARY_SEEDS,
    _conditional_source_rows,
    _summarize_deltas,
)
from dual_ablation.io_utils import _temporary_path


def test_seed_policy_is_deterministic_and_not_best_seed_selection():
    assert PRIMARY_SEEDS[:5] == [17, 43, 79, 101, 137]
    assert len(PRIMARY_SEEDS) == len(set(PRIMARY_SEEDS))


def test_conditional_permutation_preserves_ticker_identity():
    dates = np.repeat(pd.bdate_range("2024-01-01", periods=20).to_numpy(), 3)
    tickers = np.tile(np.array(["000001", "000002", "000003"]), 20)
    regime = np.repeat(np.linspace(-1, 1, 20), 3)
    source = _conditional_source_rows(dates, tickers, regime, seed=17)
    assert len(source) == len(dates)
    assert np.all(tickers[source] == tickers)
    assert np.all((source >= 0) & (source < len(source)))


def test_delta_summary_keeps_positive_loss_sign():
    rows = []
    for fold, loss in enumerate([0.03, 0.01, -0.01]):
        row = {
            "experiment": "drop_g",
            "phase": "test",
            "ablation_method": "retrain_drop",
            "model_config": "m",
            "stage": "test",
            "mode": "global_drop",
            "group": "g",
            "target_bucket": np.nan,
            "target_ticker": np.nan,
            "scope_type": "all",
            "scope_value": "all_validation",
            "fold": fold,
            "seed": 17,
        }
        for metric in [
            "pr_auc_loss",
            "roc_auc_loss",
            "balanced_accuracy_loss",
            "accuracy_loss",
            "brier_increase",
            "logloss_increase",
            "top_3pct_precision_loss",
            "top_3pct_recall_loss",
        ]:
            row[metric] = loss
        rows.append(row)
    summary = _summarize_deltas(pd.DataFrame(rows))
    assert np.isclose(summary.iloc[0]["pr_auc_loss_mean"], 0.01)
    assert np.isclose(
        summary.iloc[0]["pr_auc_loss_positive_fold_ratio"],
        2 / 3,
    )


def test_atomic_temporary_name_stays_short_for_windows_paths(tmp_path):
    destination = tmp_path / (("very_long_result_name_" * 8) + ".parquet")
    temporary = _temporary_path(destination)
    assert temporary.parent == destination.parent
    assert len(temporary.name) < 32
    assert destination.name not in temporary.name
