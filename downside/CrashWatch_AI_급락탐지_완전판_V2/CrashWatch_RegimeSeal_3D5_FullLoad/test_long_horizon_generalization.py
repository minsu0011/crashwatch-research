from __future__ import annotations

import numpy as np
import pandas as pd

from cwgeneralization.core import (
    benjamini_hochberg,
    binary_metrics,
    choose_threshold_for_recall,
)
from cwgeneralization.engine import _control_features


def test_recall_threshold_is_locked_without_confirmation_data() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    probability = np.array([.9, .8, .7, .6, .5, .4, .3, .2])
    threshold = choose_threshold_for_recall(y, probability, .70)
    assert threshold == .5
    assert (y[probability >= threshold].sum() / y.sum()) >= .70


def test_single_class_roc_and_pr_stay_nan() -> None:
    frame = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=20),
        "target": np.zeros(20, dtype=np.uint8),
        "prediction": np.linspace(.1, .2, 20),
    })
    metrics = binary_metrics(frame)
    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["roc_auc"])
    assert np.isfinite(metrics["brier"])


def test_bh_and_random_control_are_deterministic() -> None:
    adjusted = benjamini_hochberg([.01, .03, .20])
    assert np.allclose(adjusted, [.03, .045, .20])
    first = _control_features(np.array([1, 2, 3]), 12)
    second = _control_features(np.array([1, 2, 3]), 12)
    assert first.shape == (3, 12)
    assert np.array_equal(first, second)


if __name__ == "__main__":
    test_recall_threshold_is_locked_without_confirmation_data()
    test_single_class_roc_and_pr_stay_nan()
    test_bh_and_random_control_are_deterministic()
    print("long-horizon generalization tests: PASS")
