from __future__ import annotations

import py_compile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from dual_ablation.config import ProjectPaths
from dual_ablation.experiment.splits import make_walk_forward_folds
from dual_ablation.finance11h import crawler
from dual_ablation.finance11h.crawler import _year_chunks, preflight_short_sources
from dual_ablation.finance11h.features import _build_ticker_block, build_finance_features
from dual_ablation.finance11h.model import CANDIDATES, fit_xgb
from dual_ablation.finance11h.numba_kernels import (
    numba_diagnostics,
    rolling_corr,
    rolling_slope,
    rolling_sum,
    rolling_zscore,
    warmup,
)
from dual_ablation.finance11h.runner import (
    Experiment,
    Finance11HRunner,
    _build_sensitivity_matrices,
    _hash_file_metadata,
    _metrics,
    _pair_ablation_metrics,
    _summarize_deltas,
)


def _project_paths(tmp_path: Path) -> ProjectPaths:
    configs = tmp_path / "configs"
    configs.mkdir(parents=True)
    pd.DataFrame([
        {"ticker": "005930", "name": "삼성전자", "bucket": "semiconductor", "enabled": 1}
    ]).to_csv(configs / "sector_baskets.csv", index=False, encoding="utf-8-sig")
    data = tmp_path / "crashwatch_ai_data"
    return ProjectPaths(
        project=tmp_path,
        data_root=data,
        raw_dual=data / "raw" / "dual_ablation",
        feature_dual=data / "features" / "dual_ablation",
        result_dual=data / "ablation_dual",
        cache_dual=data / "ablation_dual" / "prediction_cache",
        configs=configs,
    )


def _finance_block(rows: int = 100) -> pd.DataFrame:
    index = np.arange(rows, dtype=float)
    return pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=rows),
        "ticker": "005930",
        "close": 100 + index,
        "volume": 1_000_000 + index * 100,
        "trading_value": 100_000_000 + index * 10_000,
        "fv_market_cap": 1_000_000_000_000 + index * 1_000_000,
        "fv_listed_shares": 5_000_000_000,
        "t_price_ret_1": np.sin(index / 13) / 100,
        "t_price_ret_5": np.sin(index / 17) / 50,
        "t_price_drawdown_20": -np.abs(np.sin(index / 19)) / 10,
        "t_vol_realized_20": 0.1 + np.abs(np.sin(index / 23)) / 10,
        "t_micro_amihud_20": 0.01 + np.abs(np.sin(index / 29)) / 100,
        "fs_status_short_volume": 10_000 + index * 10,
        "fs_status_short_value": 2_000_000 + index * 1_000,
        "fs_volume_short_ratio": 1.0 + np.sin(index / 11),
        "fs_balance_short_balance_shares": 100_000 + index * 20,
        "fs_balance_short_balance_value": 20_000_000 + index * 5_000,
        "ff_value_foreign": np.sin(index / 7) * 1_000_000,
        "ff_value_institution": np.cos(index / 9) * 800_000,
        "ff_value_individual": -np.sin(index / 8) * 900_000,
        "ff_foreign_foreign_ownership_rate": 45 + np.sin(index / 31),
        "fv_per": 10 + np.sin(index / 21),
        "fv_pbr": 1.2 + np.sin(index / 25) / 10,
        "fv_dividend_yield": 1.5 + np.sin(index / 27) / 10,
    })


def _metric_row(experiment: str, scope_type: str, scope_value: str, pr_auc: float) -> dict:
    row = {
        "experiment": experiment, "fold": 0, "seed": 17,
        "scope_type": scope_type, "scope_value": scope_value,
        "pr_auc": pr_auc, "roc_auc": 0.6, "balanced_accuracy": 0.55,
        "accuracy": 0.8, "brier": 0.18, "logloss": 0.55,
    }
    for fraction in (1, 3, 5):
        row[f"top_{fraction}pct_precision"] = 0.4
        row[f"top_{fraction}pct_recall"] = 0.1
    return row


def test_empty_short_sources_stop_strict_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _project_paths(tmp_path)
    jobs = [
        ("short_status", "", lambda: pd.DataFrame()),
        ("short_volume", "", lambda: pd.DataFrame()),
        ("short_value", "", lambda: pd.DataFrame()),
        ("short_balance", "", lambda: pd.DataFrame()),
    ]
    monkeypatch.setattr(crawler, "_ticker_jobs", lambda *args: jobs)
    with pytest.raises(RuntimeError, match="전부 실패/빈 응답"):
        preflight_short_sources(paths, "2020-01-01", "2020-02-20")


def test_recent_short_balance_missing_is_not_zero() -> None:
    block = _finance_block()
    block.loc[block.index[-2:], ["fs_balance_short_balance_shares", "fs_balance_short_balance_value"]] = np.nan
    result = _build_ticker_block(block)
    assert result["t_finshort_balance_to_cap"].tail(2).isna().all()
    assert result["t_finshort_balance_change_20"].tail(2).isna().all()
    assert result["t_finshort_stress"].tail(2).isna().all()


def test_year_chunks_have_no_gap_or_overlap() -> None:
    chunks = _year_chunks("2018-03-15", "2022-04-02")
    assert chunks[0][0] == pd.Timestamp("2018-03-15")
    assert chunks[-1][1] == pd.Timestamp("2022-04-02")
    for previous, current in zip(chunks, chunks[1:]):
        assert previous[1] + pd.Timedelta(days=1) == current[0]


def test_numba_kernels_match_pandas_reference() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=300).astype(np.float64)
    other = rng.normal(size=300).astype(np.float64)
    values[[3, 40, 99]] = np.nan
    series = pd.Series(values)
    expected_sum = series.rolling(20, min_periods=8).sum().to_numpy()
    expected_mean = series.rolling(20, min_periods=8).mean().to_numpy()
    expected_std = series.rolling(20, min_periods=8).std(ddof=0).to_numpy()
    expected_z = (values - expected_mean) / expected_std
    expected_corr = series.rolling(20, min_periods=8).corr(pd.Series(other)).to_numpy()

    def slope(window: pd.Series) -> float:
        valid = window.notna().to_numpy()
        if valid.sum() < 8:
            return np.nan
        x = np.arange(len(window), dtype=float)[valid]
        y = window.to_numpy()[valid]
        return float(np.polyfit(x, y, 1)[0])

    expected_slope = series.rolling(20, min_periods=8).apply(slope, raw=False).to_numpy()
    np.testing.assert_allclose(rolling_sum(values, 20, 8), expected_sum, rtol=1e-5, atol=1e-5, equal_nan=True)
    np.testing.assert_allclose(rolling_zscore(values, 20, 8), expected_z, rtol=2e-4, atol=2e-4, equal_nan=True)
    np.testing.assert_allclose(rolling_slope(values, 20, 8), expected_slope, rtol=2e-4, atol=2e-4, equal_nan=True)
    np.testing.assert_allclose(rolling_corr(values, other, 20, 8), expected_corr, rtol=2e-4, atol=2e-4, equal_nan=True)


def test_numba_is_nopython_and_reports_warm_vs_repeat() -> None:
    benchmark = warmup()
    diagnostics = numba_diagnostics()
    assert diagnostics["nopython_compiled"] is True
    assert benchmark["first_dispatch_seconds"] >= 0
    assert benchmark["repeated_seconds"] >= 0


def test_future_mutation_does_not_change_past_features() -> None:
    source = _finance_block(140)
    original = _build_ticker_block(source)
    mutated = source.copy()
    columns = [c for c in mutated.columns if c not in {"date", "ticker"}]
    mutated.loc[80:, columns] = mutated.loc[80:, columns] * 17 + 123
    changed = _build_ticker_block(mutated)
    pd.testing.assert_frame_equal(original.iloc[:80], changed.iloc[:80], check_dtype=True)


def test_purge_is_at_least_twenty_trading_days() -> None:
    dates = pd.Series(pd.bdate_range("2018-01-01", periods=1200))
    folds = make_walk_forward_folds(dates, n_folds=4, validation_days=60, purge_days=20)
    assert all(item["metadata"]["purge_start"] < item["metadata"]["validation_start"] for item in folds)
    assert all(
        len(pd.bdate_range(item["metadata"]["purge_start"], item["metadata"]["purge_end"])) >= 20
        for item in folds
    )


def test_cached_validation_rows_are_identical() -> None:
    runner = Finance11HRunner.__new__(Finance11HRunner)
    runner.df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
        "ticker": ["005930", "000660", "005930"],
        "bucket": ["semi", "semi", "semi"],
    })
    runner.y = np.array([0, 1, 0], dtype=np.int8)
    pred = runner.df.copy()
    pred["target"] = runner.y
    pred["prediction"] = np.array([0.1, 0.8, 0.2], dtype=np.float32)
    assert runner._validate_cached_prediction(pred.iloc[::-1].reset_index(drop=True), np.arange(3)) == ""
    assert "row_count" in runner._validate_cached_prediction(pred.iloc[:2], np.arange(3))


def test_target_bucket_pairing_uses_normalized_scope() -> None:
    metrics = pd.DataFrame([
        _metric_row("baseline", "bucket", "finance", 0.50),
        _metric_row("bucket__finance__g", "target_bucket", "finance", 0.40),
    ])
    paired = _pair_ablation_metrics(metrics)
    assert paired.iloc[0]["pr_auc_loss"] == pytest.approx(0.10)


def test_target_ticker_pairing_preserves_six_digits() -> None:
    metrics = pd.DataFrame([
        _metric_row("baseline", "ticker", "005930", 0.50),
        _metric_row("ticker__005930__g", "target_ticker", "005930", 0.35),
    ])
    paired = _pair_ablation_metrics(metrics)
    assert paired.iloc[0]["pr_auc_loss"] == pytest.approx(0.15)


def test_sensitivity_matrices_only_include_own_target() -> None:
    summary = pd.DataFrame([
        {"scope_type": "target_bucket", "scope_value": "finance", "target_bucket": "finance", "target_ticker": None, "group": "g", "pr_auc_loss_mean": 0.1},
        {"scope_type": "target_bucket", "scope_value": "bio", "target_bucket": "finance", "target_ticker": None, "group": "g", "pr_auc_loss_mean": 0.9},
        {"scope_type": "target_ticker", "scope_value": "005930", "target_bucket": None, "target_ticker": "005930", "group": "g", "pr_auc_loss_mean": 0.2},
        {"scope_type": "target_ticker", "scope_value": "000660", "target_bucket": None, "target_ticker": "005930", "group": "g", "pr_auc_loss_mean": 0.8},
    ])
    bucket, ticker = _build_sensitivity_matrices(summary)
    assert bucket.loc[0, "g"] == pytest.approx(0.1)
    assert ticker.loc[0, "g"] == pytest.approx(0.2)


def test_dataset_metadata_and_content_change_cache_signature(tmp_path: Path) -> None:
    path = tmp_path / "dataset.bin"
    path.write_bytes(b"first")
    before = _hash_file_metadata(path)
    path.write_bytes(b"second")
    after = _hash_file_metadata(path)
    assert before != after


def test_cache_key_reuse_and_namespace_separation(tmp_path: Path) -> None:
    runner = Finance11HRunner.__new__(Finance11HRunner)
    runner.cache_dir = tmp_path
    runner.dataset_signature = "dataset-a"
    runner.cache_namespace = "finance11h_v2"
    exp = Experiment("baseline", "baseline", "none")
    config = CANDIDATES[0]
    first = runner._cache_path(exp, 0, 17, config, ["a", "b"])
    second = runner._cache_path(exp, 0, 17, config, ["a", "b"])
    assert first == second
    runner.cache_namespace = "other"
    assert runner._cache_path(exp, 0, 17, config, ["a", "b"]) != first


def test_deadline_stops_new_task() -> None:
    runner = Finance11HRunner.__new__(Finance11HRunner)
    runner.deadline = time.monotonic() + 100
    runner.reserve_seconds = 12 * 60
    runner.task_times = []
    runner.task_times_by_stage = {}
    runner.monitor = SimpleNamespace(abort_event=threading.Event())
    assert runner.can_start("baseline") is False


def test_gpu_oom_reduction_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    import xgboost

    calls: list[dict] = []

    class FakeClassifier:
        def __init__(self, **params):
            self.params = params

        def fit(self, x, y):
            calls.append(self.params)
            if len(calls) == 1:
                raise RuntimeError("CUDA out of memory")
            return self

    monkeypatch.setattr(xgboost, "XGBClassifier", FakeClassifier)
    x = np.ones((100, 4), dtype=np.float32)
    y = np.array([0, 1] * 50, dtype=np.int8)
    _, backend, diagnostics = fit_xgb(x, y, 17, CANDIDATES[0], threads=2, prefer_gpu=True)
    assert backend == "xgboost_cuda_max_bin_128"
    assert diagnostics["oom_detected"] is True
    assert diagnostics["attempts"][0]["status"] == "failed"
    assert diagnostics["attempts"][1]["max_bin"] == 128


def test_accuracy_and_balanced_accuracy_are_not_confused() -> None:
    y = np.array([0] * 90 + [1] * 10, dtype=np.int8)
    p = np.full(100, 0.1)
    dates = np.repeat(pd.Timestamp("2025-01-02"), 100)
    result = _metrics(y, p, dates)
    assert result["accuracy"] == pytest.approx(0.90)
    assert result["balanced_accuracy"] == pytest.approx(0.50)
    assert result["top_1pct_precision"] in {0.0, 1.0}


def test_inference_requires_eight_valid_folds() -> None:
    rows = []
    for fold in range(8):
        rows.append({
            "experiment": "g", "stage": "global", "mode": "global_drop",
            "group": "g", "target_bucket": None, "target_ticker": None,
            "scope_type": "all", "scope_value": "all_validation",
            "fold": fold, "seed": 17,
            "pr_auc_loss": 0.01, "roc_auc_loss": 0.01,
            "balanced_accuracy_loss": 0.01, "accuracy_loss": 0.01,
            "brier_increase": 0.01, "logloss_increase": 0.01,
            "top_1pct_precision_loss": 0.01, "top_1pct_recall_loss": 0.01,
            "top_3pct_precision_loss": 0.01, "top_3pct_recall_loss": 0.01,
            "top_5pct_precision_loss": 0.01, "top_5pct_recall_loss": 0.01,
        })
    summary = _summarize_deltas(pd.DataFrame(rows))
    assert summary.iloc[0]["inference_level"] == "statistical"
    assert np.isfinite(summary.iloc[0]["pr_auc_loss_sign_p"])
    short = _summarize_deltas(pd.DataFrame(rows[:3]))
    assert short.iloc[0]["inference_level"] == "exploratory"
    assert np.isnan(short.iloc[0]["pr_auc_loss_sign_p"])


def test_sealed_development_rows_are_rejected(tmp_path: Path) -> None:
    paths = _project_paths(tmp_path)
    dataset = paths.data_root / "development" / "training_dataset_finance11h.parquet"
    dataset.parent.mkdir(parents=True)
    pd.DataFrame({
        "date": pd.bdate_range("2020-01-01", periods=3),
        "ticker": ["005930"] * 3,
        "label_abs_crash_20": [0, 1, 0],
        "sealed_do_not_train_or_tune": [0, 1, 0],
    }).to_parquet(dataset, index=False)
    runner = Finance11HRunner(tmp_path, dataset, hours=0.1, threads=1, folds=1, seeds=[17], prefer_gpu=False)
    with pytest.raises(RuntimeError, match="sealed 행"):
        runner.load()


def test_finance_python_files_compile() -> None:
    root = Path(__file__).resolve().parents[1]
    files = list((root / "dual_ablation" / "finance11h").glob("*.py")) + list(root.glob("04*.py"))
    assert files
    for path in files:
        py_compile.compile(str(path), doraise=True)


def test_synthetic_finance_pipeline_builds_all_required_groups(tmp_path: Path) -> None:
    paths = _project_paths(tmp_path)
    source = _finance_block(650)
    source["label_abs_crash_20"] = (np.arange(len(source)) % 9 == 0).astype(np.int8)
    source["sealed_do_not_train_or_tune"] = 0
    training = paths.data_root / "development" / "training_dataset_dual.parquet"
    training.parent.mkdir(parents=True)
    base_columns = [
        "date", "ticker", "close", "volume", "trading_value",
        "t_price_ret_1", "t_price_ret_5", "t_price_drawdown_20",
        "t_vol_realized_20", "t_micro_amihud_20",
        "label_abs_crash_20", "sealed_do_not_train_or_tune",
    ]
    source[base_columns].to_parquet(training, index=False)
    raw_columns = [
        "date", "ticker", "fv_market_cap", "fv_listed_shares",
        "fs_status_short_volume", "fs_status_short_value", "fs_volume_short_ratio",
        "fs_balance_short_balance_shares", "fs_balance_short_balance_value",
        "ff_value_foreign", "ff_value_institution", "ff_value_individual",
        "ff_foreign_foreign_ownership_rate", "fv_per", "fv_pbr", "fv_dividend_yield",
    ]
    raw_path = paths.raw_dual / "finance11h" / "finance_ticker_timeseries.parquet"
    raw_path.parent.mkdir(parents=True)
    source[raw_columns].to_parquet(raw_path, index=False)
    result = build_finance_features(tmp_path, training, strict_short=True)
    counts = result["valid_finance_features_by_group"]
    assert counts["t_financial_shorting"] >= 8
    assert counts["u_financial_shorting"] >= 8
    assert all(counts[group] >= 1 for group in (
        "t_financial_flow", "u_financial_flow",
        "t_financial_valuation", "t_financial_interaction",
    ))
    runner = Finance11HRunner(
        tmp_path, Path(result["output_dataset"]),
        hours=0.1, threads=1, folds=1, seeds=[17],
        validation_days=60, purge_days=20, prefer_gpu=False,
    )
    runner.load()
    assert runner.X.dtype == np.float32
    assert len(runner.folds) == 1
