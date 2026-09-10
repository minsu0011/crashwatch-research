from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from dual_ablation.config import get_paths
from dual_ablation.experiment.models import get_model_settings
from dual_ablation.experiment.runner import ExperimentSpec, _deterministic_train_sample, _prediction_path
from dual_ablation.longrun.hardware import query_gpu, query_power_limits
from dual_ablation.longrun.planner import Job, initial_jobs, load_jobs, save_jobs


PROJECT = Path(__file__).resolve().parents[1]


def test_longrun_config_is_safe_profile():
    config = json.loads((PROJECT / "configs" / "longrun_5day.json").read_text(encoding="utf-8"))
    assert 96 <= config["runtime_hours"] <= 120
    assert config["safety"]["gpu_power_ratio"] <= 0.80
    assert config["safety"]["cpu_logical_cores"] <= 8
    assert config["safety"]["gpu_abort_temp_c"] < config["safety"]["gpu_hard_temp_c"]
    assert config["experiment"]["purge_days"] == 20
    assert len(config["experiment"]["seeds_primary"]) >= 5


def test_run_tag_separates_result_and_shares_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("CRASHWATCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CRASHWATCH_RUN_TAG", "job A")
    paths = get_paths(PROJECT)
    assert paths.result_dual.name == "job_A"
    assert paths.cache_dual.name == "prediction_cache"
    assert "runs" in paths.result_dual.parts


def test_model_settings_respect_thread_limit(monkeypatch):
    monkeypatch.setenv("CRASHWATCH_XGB_N_JOBS", "6")
    monkeypatch.setenv("CRASHWATCH_XGB_N_ESTIMATORS", "333")
    settings = get_model_settings(prefer_gpu=True)
    assert settings["n_jobs"] == min(6, os.cpu_count() or 6)
    assert settings["n_estimators"] == 333
    assert settings["prefer_gpu"] is True


def test_cache_key_changes_with_fold_boundaries(monkeypatch, tmp_path):
    monkeypatch.setenv("CRASHWATCH_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CRASHWATCH_RUN_TAG", "cache_test")
    paths = get_paths(PROJECT)
    spec = ExperimentSpec("baseline", "baseline", "none")
    fold1 = {"fold_id": 0, "metadata": {"fold_id": 0, "validation_start": "2024-01-01", "validation_end": "2024-03-01"}}
    fold2 = {"fold_id": 0, "metadata": {"fold_id": 0, "validation_start": "2025-01-01", "validation_end": "2025-03-01"}}
    p1 = _prediction_path(paths, spec, fold1, 17, ["x"], "data", "label_abs_crash_20", True)
    p2 = _prediction_path(paths, spec, fold2, 17, ["x"], "data", "label_abs_crash_20", True)
    assert p1 != p2


def test_job_state_roundtrip(tmp_path):
    jobs = [Job("a", "screen", ["universe"], "a", seeds=[17, 43])]
    state = tmp_path / "state.json"
    save_jobs(state, jobs, {"x": 1})
    loaded, metadata = load_jobs(state)
    assert loaded[0].job_id == "a"
    assert loaded[0].seeds == [17, 43]
    assert metadata["x"] == 1


def test_initial_plan_has_global_and_eight_buckets(monkeypatch, tmp_path):
    # 실제 V4 basket 설정을 사용하되 데이터 경로만 임시로 분리한다.
    monkeypatch.setenv("CRASHWATCH_DATA_DIR", str(tmp_path / "data"))
    config = json.loads((PROJECT / "configs" / "longrun_5day.json").read_text(encoding="utf-8"))
    jobs = initial_jobs(PROJECT, config)
    assert jobs[0].stage == "global_screen"
    assert sum(job.stage == "bucket_screen" for job in jobs) == 8
    assert all(len(job.seeds) == 5 for job in jobs)


def test_gpu_queries_fail_softly(monkeypatch):
    import dual_ablation.longrun.hardware as hw

    monkeypatch.setattr(hw, "_run_nvidia_smi", lambda *args, **kwargs: None)
    assert query_gpu() == {}
    assert query_power_limits() == {}


def test_deterministic_sampling_supports_large_hashes() -> None:
    frame = pd.DataFrame({
        "date": pd.bdate_range("2024-01-02", periods=200),
        "ticker": [f"{i % 17:06d}" for i in range(200)],
        "label_abs_crash_20": [1 if i % 23 == 0 else 0 for i in range(200)],
    })
    first = _deterministic_train_sample(frame, "label_abs_crash_20", 60, 17)
    second = _deterministic_train_sample(frame, "label_abs_crash_20", 60, 17)
    assert len(first) == 60
    assert first.index.tolist() == second.index.tolist()
    assert frame.loc[frame["label_abs_crash_20"].eq(1)].index.isin(first.index).all()


def test_preflight_accepts_minimal_v4_dataset(tmp_path):
    pytest.importorskip("pyarrow")
    from dual_ablation.longrun.orchestrator import _preflight

    data_root = tmp_path / "crashwatch_ai_data"
    feature_dir = data_root / "features" / "dual_ablation"
    dev = data_root / "development"
    feature_dir.mkdir(parents=True)
    dev.mkdir(parents=True)
    dates = pd.bdate_range("2022-01-03", periods=720)
    tickers = [f"{i:06d}" for i in range(1, 6)]
    rows = []
    for ticker in tickers:
        for i, date in enumerate(dates):
            rows.append({
                "date": date,
                "ticker": ticker,
                "label_abs_crash_20": int((i + int(ticker)) % 17 == 0),
                "t_rangevol_parkinson_20": float(i % 19) / 19,
            })
    dataset = dev / "training_dataset_dual.parquet"
    pd.DataFrame(rows).to_parquet(dataset, index=False)
    (feature_dir / "feature_catalog_universe.json").write_text("{}", encoding="utf-8")
    (feature_dir / "feature_catalog_ticker.json").write_text(
        json.dumps({"t_range_volatility": ["t_rangevol_parkinson_20"]}), encoding="utf-8"
    )
    config = json.loads((PROJECT / "configs" / "longrun_5day.json").read_text(encoding="utf-8"))
    config["experiment"]["require_base_catalog"] = False
    config["safety"]["min_disk_free_gb"] = 0
    config["safety"]["min_free_ram_gb"] = 0
    report = _preflight(PROJECT, dataset, data_root, config, data_root / "ablation_longrun")
    assert report["status"] == "pass"
    assert report["fold_count"] == 10
    assert "t_range_volatility" in report["present_v4_groups"]
