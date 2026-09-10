from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dual_ablation.experiment import runner


class DummyModel:
    backend = "synthetic_dummy"

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        signal = np.nan_to_num(x.to_numpy(dtype=float), nan=0.0).mean(axis=1)
        return np.clip(1.0 / (1.0 + np.exp(-signal)), 1e-5, 1 - 1e-5)


def _make_project(root: Path) -> Path:
    project = root / "v3"
    data = project / "crashwatch_ai_data"
    (project / "configs").mkdir(parents=True)
    (data / "development").mkdir(parents=True)
    (data / "features" / "dual_ablation").mkdir(parents=True)
    (data / "dual_ablation" / "meta").mkdir(parents=True)
    tickers = [f"{i:06d}" for i in range(1, 6)]
    baskets = pd.DataFrame({
        "bucket": "synthetic", "bucket_name": "synthetic", "ticker": tickers,
        "name": tickers, "market": "KOSPI", "role": "test", "enabled": 1,
    })
    baskets.to_csv(project / "configs" / "sector_baskets.csv", index=False)
    dates = pd.date_range("2023-01-02", periods=130, freq="B")
    rows = []
    for day, date in enumerate(dates):
        for position, ticker in enumerate(tickers):
            rows.append({
                "date": date, "ticker": ticker, "bucket": "synthetic",
                "label_abs_crash_20": int((day + position) % 7 == 0),
                "u_signal": np.sin(day / 10), "t_signal": (position - 2) / 2 + day / 500,
                "sealed_do_not_train_or_tune": 0,
            })
    dataset = pd.DataFrame(rows)
    dataset.to_parquet(data / "development" / "training_dataset_dual.parquet", index=False)
    (data / "features" / "dual_ablation" / "feature_catalog_universe.json").write_text(
        json.dumps({"u_market_trend": ["u_signal"]}), encoding="utf-8",
    )
    (data / "features" / "dual_ablation" / "feature_catalog_ticker.json").write_text(
        json.dumps({"t_price_trend": ["t_signal"]}), encoding="utf-8",
    )
    quality = pd.DataFrame([
        {"feature": "u_signal", "domain": "universe", "group": "u_market_trend", "status": "valid"},
        {"feature": "t_signal", "domain": "ticker", "group": "t_price_trend", "status": "valid"},
    ])
    quality.to_csv(data / "dual_ablation" / "meta" / "feature_quality_audit.csv", index=False)
    pd.DataFrame([{
        "bucket": "synthetic", "planned_ticker_count": 5, "available_ticker_count": 5,
        "missing_ticker_count": 0, "status": "ready",
    }]).to_csv(data / "dual_ablation" / "meta" / "basket_coverage_report.csv", index=False)
    pd.DataFrame([{"status": "success"}]).to_csv(
        data / "dual_ablation" / "meta" / "crawl_manifest.csv", index=False,
    )
    return project


def test_synthetic_run_reuses_cache_and_invalidates_on_dataset_change(tmp_path, monkeypatch) -> None:
    project = _make_project(tmp_path)
    fit_calls = 0

    def fake_fit(x, y, seed, prefer_gpu=True):
        nonlocal fit_calls
        fit_calls += 1
        return DummyModel()

    monkeypatch.setattr(runner, "fit_model", fake_fit)
    kwargs = dict(
        modes={"universe"}, groups={"u_market_trend"}, seeds=[17], n_folds=4,
        validation_days=10, purge_days=20, min_train_days=20, calibration="none",
        allow_partial_data=True, prefer_gpu=False,
    )
    first = runner.run_ablation(project, **kwargs)
    first_calls = fit_calls
    assert first["executed_experiments"] == 2 and first_calls > 0
    second = runner.run_ablation(project, **kwargs)
    assert second["dataset_hash"] == first["dataset_hash"] and fit_calls == first_calls

    dataset_path = project / "crashwatch_ai_data" / "development" / "training_dataset_dual.parquet"
    changed = pd.read_parquet(dataset_path)
    changed.loc[0, "t_signal"] += 0.123
    changed.to_parquet(dataset_path, index=False)
    third = runner.run_ablation(project, **kwargs)
    assert third["dataset_hash"] != first["dataset_hash"] and fit_calls > first_calls

