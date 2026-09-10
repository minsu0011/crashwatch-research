from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from dual_ablation.ticker_elite2h.data import choose_adaptive_folds
from dual_ablation.ticker_elite2h.model import feature_audit, select_features_from_audit
from dual_ablation.ticker_elite2h.reporting import exact_pooled_comparison
from dual_ablation.ticker_elite2h.runner import load_plan


def plan() -> dict:
    return load_plan(Path(__file__).resolve().parents[1])


def test_adaptive_folds_prefers_feasible_window():
    p = plan()
    dates = np.arange("2019-01-01", "2026-01-01", dtype="datetime64[D]")
    y = np.zeros(len(dates), dtype=np.int8)
    y[::13] = 1
    folds, audit = choose_adaptive_folds(dates, y, p)
    assert len(folds) >= p["minimum_outer_folds"]
    assert audit["validation_days"] in p["candidate_validation_days"]
    assert all(len(np.unique(y[fold.validation_idx])) == 2 for fold in folds)


def test_all_temporal_purges_cover_label_horizon():
    p = plan()
    assert int(p["purge_days"]) >= 20
    assert int(p["calibration_purge_days"]) >= 20
    assert int(p["inner_purge_days"]) >= 20


def test_feature_policies_return_nonredundant_indices():
    rng = np.random.default_rng(17)
    X = rng.normal(size=(900, 30)).astype(np.float32)
    X[:, 1] = X[:, 0] * 0.999 + rng.normal(scale=1e-4, size=900)
    y = (X[:, 0] + 0.4 * X[:, 5] + rng.normal(size=900) > 1.0).astype(np.int8)
    names = [f"f_{i}" for i in range(X.shape[1])]
    p = plan()
    audit = feature_audit(X, y, names, p["stable_sign_segments"])
    for policy in p["feature_policies"]:
        selected = select_features_from_audit(X, audit, policy, 12, p)
        assert 1 <= len(selected) <= 12
        assert not ({0, 1} <= set(selected.tolist()))


def test_exact_pooled_comparison_uses_same_rows(tmp_path: Path):
    import pytest
    pytest.importorskip("pyarrow")
    result = tmp_path / "ticker_elite2h"
    model_dir = result / "ticker_models" / "005930"
    model_dir.mkdir(parents=True)
    dates = pd.date_range("2025-01-01", periods=80, freq="D")
    target = np.asarray(([0, 1] * 40), dtype=np.int8)
    independent = pd.DataFrame({
        "date": dates,
        "ticker": "005930",
        "target": target,
        "raw_prediction": np.where(target == 1, 0.8, 0.2),
    })
    independent.to_parquet(model_dir / "outer_predictions.parquet", index=False)

    pooled_dir = tmp_path / "refine12h" / "predictions" / "outer" / "lightgbm" / "full_reduced"
    pooled_dir.mkdir(parents=True)
    pooled = pd.DataFrame({
        "date": dates.append(pd.DatetimeIndex([pd.Timestamp("2026-01-01")])),
        "ticker": "005930",
        "target": np.concatenate([target, [1]]),
        "raw_prediction": np.concatenate([np.where(target == 1, 0.7, 0.3), [0.9]]),
        "seed": 17,
    })
    pooled.to_parquet(pooled_dir / "pred.parquet", index=False)
    comparison = exact_pooled_comparison(result, tmp_path)
    assert len(comparison) == 1
    assert int(comparison.iloc[0]["paired_rows"]) == 80
    assert comparison.iloc[0]["independent_minus_pooled_pr_auc"] >= 0


def test_rolling_platt_preserves_ranking_and_uses_recent_window():
    from dual_ablation.ticker_elite2h.model import _rolling_platt_policy
    from dual_ablation.refine12h.calibration import apply_calibrator

    rng = np.random.default_rng(20260803)
    dates = np.arange("2024-01-01", "2025-01-01", dtype="datetime64[D]")
    latent = rng.normal(size=len(dates))
    raw = 1.0 / (1.0 + np.exp(-(0.7 * latent - 0.8)))
    y = (latent + rng.normal(scale=0.8, size=len(dates)) > 0.6).astype(np.int8)
    p = plan()
    policy = _rolling_platt_policy(y, raw, dates, p)
    calibrated = apply_calibrator(policy.method, policy.params, raw)
    assert policy.method in {"platt_positive", "none"}
    if policy.method == "platt_positive":
        assert float(policy.params["slope"]) > 0
        assert int(policy.params["rolling_oof_blocks"]) >= 2
        assert np.array_equal(np.argsort(raw), np.argsort(calibrated))


def test_final_recipe_ranking_ignores_alphabetic_family_order():
    from dual_ablation.ticker_elite2h.model import _rank_final_recipes

    recipes = pd.DataFrame([
        {"fold": 0, "family": "catboost", "config_name": "cat_a", "feature_policy": "standard", "feature_budget": 56, "training_window_days": 0, "inner_selection_score": 1.0},
        {"fold": 1, "family": "catboost", "config_name": "cat_a", "feature_policy": "standard", "feature_budget": 56, "training_window_days": 0, "inner_selection_score": 1.0},
        {"fold": 2, "family": "xgboost", "config_name": "xgb_z", "feature_policy": "consensus", "feature_budget": 56, "training_window_days": 750, "inner_selection_score": 1.4},
        {"fold": 3, "family": "xgboost", "config_name": "xgb_z", "feature_policy": "consensus", "feature_budget": 56, "training_window_days": 750, "inner_selection_score": 1.4},
    ])
    folds = pd.DataFrame([
        {"fold": 0, "raw_pr_auc": 0.40, "raw_pr_lift": 1.4, "raw_roc_auc": 0.56, "balanced_accuracy": 0.52, "top_10pct_precision": 0.40, "brier_skill": -0.2, "logloss_skill": -0.2},
        {"fold": 1, "raw_pr_auc": 0.41, "raw_pr_lift": 1.45, "raw_roc_auc": 0.57, "balanced_accuracy": 0.53, "top_10pct_precision": 0.42, "brier_skill": -0.2, "logloss_skill": -0.2},
        {"fold": 2, "raw_pr_auc": 0.62, "raw_pr_lift": 2.0, "raw_roc_auc": 0.72, "balanced_accuracy": 0.61, "top_10pct_precision": 0.70, "brier_skill": 0.0, "logloss_skill": 0.0},
        {"fold": 3, "raw_pr_auc": 0.60, "raw_pr_lift": 1.95, "raw_roc_auc": 0.70, "balanced_accuracy": 0.60, "top_10pct_precision": 0.68, "brier_skill": 0.0, "logloss_skill": 0.0},
    ])
    ranked = _rank_final_recipes(recipes, folds, plan())
    assert ranked.iloc[0]["family"] == "xgboost"
    assert ranked.iloc[0]["selection_rule"] == "outer_stability_score_v2_no_alphabetic_tie_break"


def test_runtime_profiles_have_separate_backends_and_result_dirs(monkeypatch):
    from dual_ablation.ticker_elite2h.runner import load_plan
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("CRASHWATCH_EXECUTION_PROFILE", "cpu4")
    cpu = load_plan(project)
    monkeypatch.setenv("CRASHWATCH_EXECUTION_PROFILE", "full")
    full = load_plan(project)
    assert cpu["backend_mode"] == "cpu_only"
    assert cpu["result_subdir"] == "ticker_independent_cpu4_unlimited"
    assert cpu["gpu_workers"] == 1
    assert cpu["threads_per_model_worker"] == 4
    assert cpu["correlation_workers"] == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert full["backend_mode"] == "hybrid_fixed"
    assert full["result_subdir"] == "ticker_independent_full_unlimited"


def test_backend_policy_is_fixed_by_profile(monkeypatch):
    import dual_ablation.ticker_elite2h.model as model_module

    calls = []
    class Dummy:
        pass
    def fake_fit(family, config, x_train, y_train, **kwargs):
        calls.append((family, kwargs["requested_backend"], kwargs["force_backend"], kwargs["threads"]))
        return Dummy()
    monkeypatch.setattr(model_module, "fit_tree_model", fake_fit)
    X = np.zeros((10, 2), dtype=np.float32)
    y = np.asarray([0, 1] * 5, dtype=np.int8)

    monkeypatch.setenv("CRASHWATCH_BACKEND_MODE", "cpu_only")
    model_module._fit_candidate("xgboost", {"name": "x"}, X, y, 17, 4)
    model_module._fit_candidate("catboost", {"name": "c"}, X, y, 17, 4)
    assert calls[-2][1:3] == ("cpu", "cpu")
    assert calls[-1][1:3] == ("cpu", "cpu")

    monkeypatch.setenv("CRASHWATCH_BACKEND_MODE", "hybrid_fixed")
    model_module._fit_candidate("lightgbm", {"name": "l"}, X, y, 17, 3)
    model_module._fit_candidate("xgboost", {"name": "x"}, X, y, 17, 3)
    model_module._fit_candidate("catboost", {"name": "c"}, X, y, 17, 3)
    assert calls[-3][1:3] == ("cpu", "cpu")
    assert calls[-2][1:3] == ("cuda", "cuda")
    assert calls[-1][1:3] == ("cuda", "cuda")


def test_unlimited_plan_has_no_time_budget(monkeypatch):
    project = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("CRASHWATCH_EXECUTION_PROFILE", "cpu4")
    p = load_plan(project)
    assert p["time_budget_enabled"] is False
    assert p["completion_mode"] == "run_until_all_registered_tasks_terminal"
    assert "default_hours" not in p
    assert "hard_max_hours" not in p
    assert int(p["max_task_attempts"]) >= 2


def test_registry_tracks_retry_attempts(tmp_path: Path):
    from dual_ablation.ticker_map1h.registry import TickerTaskRegistry

    registry = TickerTaskRegistry(tmp_path / "tasks.sqlite")
    registry.ensure_tasks([{
        "task_id": "elite__005930",
        "stage": "elite_model",
        "ticker": "005930",
        "variant": "elite",
        "seed": -1,
        "priority": 10,
        "payload": {},
    }])
    first = registry.claim_next(stages=("elite_model",), worker_name="w0", backend="cpu", threads=4)
    assert first is not None and first.attempts == 1
    registry.release(first.task_id, "transient")
    second = registry.claim_next(stages=("elite_model",), worker_name="w0", backend="cpu", threads=4)
    assert second is not None and second.attempts == 2
