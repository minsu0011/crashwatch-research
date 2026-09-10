from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from dual_ablation.experiment.metrics import add_daily_alerts, safe_metrics
from dual_ablation.experiment.runner import (
    ExperimentSpec, _apply_mask, _deterministic_train_sample, _fit_predict, _prediction_path,
)
from dual_ablation.experiment.splits import make_walk_forward_folds
from dual_ablation.experiment.statistics import _bootstrap_fold_means, paired_delta_table, summarize_deltas


METRICS = ["pr_auc", "roc_auc", "brier", "logloss", "alert_precision", "alert_recall"]


def _metric_row(experiment: str, scope_type: str, scope_value: str, row_hash: str = "same") -> dict:
    row = {
        "experiment": experiment, "fold": 0, "seed": 17, "scope_type": scope_type,
        "scope_value": scope_value, "validation_row_hash": row_hash,
        "namespace": "ticker", "ablation_mode": "bucket_mask", "target_group": "t_x",
        "target_bucket": "b", "target_ticker": None,
    }
    row.update({name: 0.5 for name in METRICS})
    return row


def test_walk_forward_purge_is_twenty_trading_days() -> None:
    dates = pd.Series(pd.date_range("2020-01-01", periods=900, freq="B"))
    folds = make_walk_forward_folds(dates, n_folds=8, validation_days=60, purge_days=20, min_train_days=500)
    assert len(folds) == 8
    for fold in folds:
        train_end = fold["train_dates_index"][-1]
        validation_start = fold["validation_dates_index"][0]
        positions = pd.Index(dates).get_indexer([train_end, validation_start])
        assert positions[1] - positions[0] - 1 == 20


def test_daily_top_three_percent_always_selects_at_least_one() -> None:
    block = pd.DataFrame({
        "date": [pd.Timestamp("2025-01-02")] * 2,
        "prediction": [0.2, 0.8], "target": [0, 1], "ticker": ["000001", "000002"],
    })
    out = add_daily_alerts(block)
    assert out["alert"].sum() == 1
    assert out.loc[out["alert"].eq(1), "ticker"].item() == "000002"


def test_single_class_auc_metrics_remain_nan() -> None:
    block = pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=4), "ticker": ["000001"] * 4,
        "target": [0, 0, 0, 0], "prediction": [0.1, 0.2, 0.3, 0.4],
        "prediction_uncalibrated": [0.1, 0.2, 0.3, 0.4], "alert": [1, 0, 0, 0],
    })
    metrics = safe_metrics(block)
    assert np.isnan(metrics["pr_auc"]) and np.isnan(metrics["roc_auc"])
    assert np.isfinite(metrics["brier"]) and np.isfinite(metrics["logloss"])


def test_empty_scope_has_stable_hash_and_is_not_zero_effect() -> None:
    metrics = safe_metrics(pd.DataFrame())
    assert metrics["rows"] == 0 and metrics["metric_status"] == "skipped_no_rows"
    assert isinstance(metrics["validation_row_hash"], str)
    assert np.isnan(metrics["pr_auc"])


def test_bucket_and_ticker_scope_pairing_and_delta_sign() -> None:
    baseline_bucket = _metric_row("baseline", "bucket", "b")
    ablated_bucket = _metric_row("bucket__b__t_x", "target_bucket", "b")
    ablated_bucket["pr_auc"] = 0.4
    bucket_delta = paired_delta_table(pd.DataFrame([baseline_bucket, ablated_bucket]))
    assert np.isclose(bucket_delta.loc[0, "pr_auc_loss_when_removed"], 0.1)

    baseline_ticker = _metric_row("baseline", "ticker", "005930")
    ablated_ticker = _metric_row("ticker__005930__t_x", "target_ticker", "005930")
    ticker_delta = paired_delta_table(pd.DataFrame([baseline_ticker, ablated_ticker]))
    assert ticker_delta.loc[0, "pair_scope_type"] == "ticker"


def test_pairing_rejects_different_validation_rows() -> None:
    baseline = _metric_row("baseline", "bucket", "b", "left")
    ablated = _metric_row("bucket__b__t_x", "target_bucket", "b", "right")
    try:
        paired_delta_table(pd.DataFrame([baseline, ablated]))
    except AssertionError:
        pass
    else:
        raise AssertionError("row-hash mismatch was not rejected")


def test_mask_only_changes_target_ticker_rows() -> None:
    frame = pd.DataFrame({"ticker": ["005930", "000660"], "bucket": ["semi", "semi"], "x": [1.0, 2.0]})
    spec = ExperimentSpec("e", "ticker", "ticker_mask", "t_x", "semi", "005930")
    masked = _apply_mask(frame, spec, ["x"])
    assert np.isnan(masked.loc[0, "x"])
    assert masked.loc[1, "x"] == frame.loc[1, "x"]


def test_fold_statistics_average_seeds_before_inference() -> None:
    rows = []
    for fold, values in [(0, [1.0, 3.0]), (1, [5.0, 7.0])]:
        for seed, value in enumerate(values):
            row = {
                "experiment": "e", "namespace": "u", "ablation_mode": "global_drop",
                "target_group": "u_x", "target_bucket": None, "target_ticker": None,
                "pair_scope_type": "all", "scope_value": "all_validation", "fold": fold, "seed": seed,
            }
            for col in [
                "pr_auc_loss_when_removed", "roc_auc_loss_when_removed", "brier_increase_when_removed",
                "logloss_increase_when_removed", "alert_precision_loss_when_removed",
                "alert_recall_loss_when_removed",
            ]:
                row[col] = value
            rows.append(row)
    summary = summarize_deltas(pd.DataFrame(rows), bootstrap_samples=100)
    assert summary.loc[0, "fold_count"] == 2 and summary.loc[0, "seed_count"] == 2
    assert summary.loc[0, "mean_delta"] == 4.0
    assert np.isclose(summary.loc[0, "std_delta"], np.std([2.0, 6.0], ddof=1))


def test_bootstrap_p_value_never_reports_exact_zero() -> None:
    _, _, pvalue = _bootstrap_fold_means(
        np.ones(8, dtype=float), np.random.default_rng(7), samples=4000,
    )
    assert 0.0 < pvalue <= 1.0


def test_cache_key_changes_with_dataset_hash(tmp_path) -> None:
    paths = SimpleNamespace(cache_dual=tmp_path)
    fold = {"metadata": {"fold_id": 0, "validation_start": "2025-01-01"}}
    spec = ExperimentSpec("baseline", "baseline", "none")
    left = _prediction_path(paths, spec, fold, 17, ["x"], dataset_hash="a", catalog_hash="c", calibration="none")
    right = _prediction_path(paths, spec, fold, 17, ["x"], dataset_hash="b", catalog_hash="c", calibration="none")
    assert left != right


def test_large_deterministic_sample_hash_is_supported() -> None:
    frame = pd.DataFrame({
        "ticker": [f"{i % 10:06d}" for i in range(100)],
        "date": pd.date_range("2025-01-01", periods=100),
        "target": [1] * 5 + [0] * 95,
    })
    left = _deterministic_train_sample(frame, "target", 30, 17)
    right = _deterministic_train_sample(frame, "target", 30, 17)
    assert len(left) == 30 and left.index.equals(right.index) and left["target"].sum() == 5


def test_calibration_uses_training_tail_not_validation(monkeypatch) -> None:
    fit_sizes: list[int] = []

    class Dummy:
        backend = "dummy"

        def predict_proba(self, x):
            return np.clip(0.2 + 0.6 * np.nan_to_num(x["x"].to_numpy()), 0.01, 0.99)

    def fake_fit(x, y, seed, prefer_gpu=True):
        fit_sizes.append(len(x))
        return Dummy()

    monkeypatch.setattr("dual_ablation.experiment.runner.fit_model", fake_fit)
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    train = pd.DataFrame({"date": dates, "x": np.tile([0.0, 1.0], 100), "y": np.tile([0, 1], 100)})
    validation = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10), "x": 0.5, "y": 1})
    _fit_predict(train, validation, ["x"], "y", 17, False, "sigmoid")
    assert fit_sizes == [140]
