from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from dual_ablation.refine12h.calibration import apply_calibrator, fit_calibrator, select_safe_policy


def test_positive_platt_cannot_invert_rank() -> None:
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, 1200)
    raw = np.clip(0.05 + 0.55 * y + rng.normal(0, 0.22, len(y)), 1e-4, 1 - 1e-4)
    params = fit_calibrator("platt_positive", y, raw)
    assert params["slope"] >= 0.0
    calibrated = apply_calibrator("platt_positive", params, raw)
    assert roc_auc_score(y, calibrated) >= roc_auc_score(y, raw) - 1e-12


def test_fold2_style_negative_relation_falls_back_without_inversion() -> None:
    rng = np.random.default_rng(11)
    y = rng.integers(0, 2, 1200)
    raw = np.clip(0.95 - 0.80 * y + rng.normal(0, 0.08, len(y)), 1e-4, 1 - 1e-4)
    dates = np.repeat(np.arange(120), 10)
    policy = select_safe_policy(y, raw, dates)
    calibrated = apply_calibrator(policy.method, policy.params, raw)
    assert policy.method == "none" or policy.rank_correlation >= 0.995
    assert roc_auc_score(y, calibrated) >= roc_auc_score(y, raw) - 0.005
