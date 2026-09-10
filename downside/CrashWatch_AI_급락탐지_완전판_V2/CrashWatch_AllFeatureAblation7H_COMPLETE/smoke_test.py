from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cw7h.aggregate import aggregate_results
from cw7h.correlation import _group_residualize, run_correlation_audit
from cw7h.data import PreparedData, References
from cw7h.folds import FoldSlice
from cw7h.lgb_worker import LgbBlock, ModelCondition, run_lgb_block, tune_fold_iteration
from cw7h.runner import _cluster_conditions, _conditional_conditions, _execute_blocks
from cw7h.utils import atomic_json


def main() -> int:
    root = Path(tempfile.mkdtemp(prefix="cw7h_smoke_"))
    try:
        residual_fixture = np.array([[1.0, np.nan], [2.0, np.nan], [5.0, np.nan]], dtype=np.float32)
        residualized = _group_residualize(residual_fixture, np.array([0, 0, 1], dtype=np.int32))
        assert np.isfinite(residualized[:, 0]).all()
        assert np.isnan(residualized[:, 1]).all()
        cache = root / "cache"
        output = root / "output"
        cache.mkdir(parents=True)
        output.mkdir(parents=True)
        rng = np.random.default_rng(17)
        dates_unique = pd.bdate_range("2022-01-03", periods=320)
        tickers = [f"{i:06d}" for i in range(8)]
        rows = len(dates_unique) * len(tickers)
        dates = np.repeat(dates_unique.view("int64"), len(tickers))
        ticker_values = np.tile(np.array(tickers, dtype="U16"), len(dates_unique))
        X = rng.normal(size=(rows, 8)).astype(np.float32)
        X[:, 1] = X[:, 0] * 0.96 + rng.normal(scale=0.08, size=rows)
        X[:, 3] = X[:, 2] * -0.93 + rng.normal(scale=0.12, size=rows)
        X[rng.random(X.shape) < 0.03] = np.nan
        signal = np.nan_to_num(X[:, 0]) + 0.55 * np.nan_to_num(X[:, 4]) + rng.normal(scale=0.7, size=rows)
        y = (signal > np.quantile(signal, 0.72)).astype(np.uint8)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
        np.save(cache / "X_all_valid.npy", X)
        np.save(cache / "target.npy", y)
        np.save(cache / "dates_ns.npy", dates)
        np.save(cache / "tickers.npy", ticker_values)
        np.save(cache / "buckets.npy", np.array(["test"] * rows, dtype="U32"))
        np.save(cache / "original_row_id.npy", np.arange(rows, dtype=np.int64))
        atomic_json(feature_names, cache / "feature_names.json")
        atomic_json({f: "synthetic" for f in feature_names}, cache / "feature_groups.json")
        manifest = {"status": "complete", "rows": rows, "features": len(feature_names)}
        atomic_json(manifest, cache / "cache_manifest.json")
        prepared = PreparedData(cache, root / "synthetic.parquet", "synthetic-signature", manifest, feature_names, {f: "synthetic" for f in feature_names})
        audit = pd.DataFrame({
            "feature": feature_names,
            "status": "valid",
            "missing_ratio": np.isnan(X).mean(axis=0),
            "unique_count": [rows] * len(feature_names),
            "ticker_coverage": [1.0] * len(feature_names),
            "group": ["synthetic"] * len(feature_names),
        })
        refs = References(feature_names, {"synthetic": feature_names}, audit, {t: "test" for t in tickers}, [], {})
        folds = [
            FoldSlice(0, 0, 200 * 8, 220 * 8, 240 * 8, "2022-01-03", str(dates_unique[199].date()), str(dates_unique[220].date()), str(dates_unique[239].date()), 200, 20, 1600, 160, True, ""),
            FoldSlice(1, 0, 260 * 8, 280 * 8, 300 * 8, "2022-01-03", str(dates_unique[259].date()), str(dates_unique[280].date()), str(dates_unique[299].date()), 260, 20, 2080, 160, True, ""),
        ]
        config = {
            "correlation": {
                "sample_rows": 1600,
                "mi_sample_rows": 1200,
                "seed": 17,
                "min_pair_fraction": 0.1,
                "mi_neighbors": 3,
                "mi_jobs": 1,
                "cluster_thresholds": [0.8, 0.9, 0.92, 0.95],
                "primary_cluster_threshold": 0.92,
                "edge_output_threshold": 0.7,
            }
        }
        run_correlation_audit(prepared, refs, folds, output, config, force=True)
        model_config = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_data_in_leaf": 20,
            "lambda_l1": 0.0,
            "lambda_l2": 0.5,
            "max_bin": 127,
            "min_gain_to_split": 1e-12,
            "max_rounds": 80,
            "early_stopping_rounds": 10,
            "min_effective_rounds": 10,
            "fallback_rounds": 30,
        }
        best = {}
        for fold in folds:
            result = tune_fold_iteration({
                "cache_root": str(cache),
                "output_dir": str(output),
                "dataset_signature": prepared.signature,
                "fold": fold.to_dict(),
                "model_config": model_config,
                "threads": 2,
                "rolling_windows": 2,
                "rolling_step_days": 40,
                "inner_validation_days": 20,
                "inner_purge_days": 10,
                "min_train_days": 80,
            })
            assert result["status"] == "completed", result
            best[fold.fold_id] = result["best_iteration"]
        deadline = time.time() + 300
        primary_clusters = pd.read_csv(output / "correlation" / "primary_clusters.csv")
        pruned, conditional = _conditional_conditions(feature_names, primary_clusters)
        cluster_conditions = _cluster_conditions(feature_names, primary_clusters)
        primary_blocks = []
        conditional_blocks = []
        cluster_blocks = []
        for fold in folds:
            baseline = ModelCondition("B0_ALL_VALID", "baseline_all_valid", "", -1, None, None, True)
            loo = [ModelCondition(f"LOO::{f}", "single_feature_loo", f, i, None, None, True) for i, f in enumerate(feature_names)]
            primary_blocks.append(LgbBlock(str(cache), str(output), prepared.signature, "smoke-run", fold.to_dict(), tuple([baseline] + loo), best[fold.fold_id], model_config, 2, deadline, tuple(feature_names), "smoke_primary"))
            conditional_blocks.append(LgbBlock(str(cache), str(output), prepared.signature, "smoke-run", fold.to_dict(), tuple([pruned] + conditional), best[fold.fold_id], model_config, 2, deadline, tuple(feature_names), "smoke_conditional"))
            cluster_blocks.append(LgbBlock(str(cache), str(output), prepared.signature, "smoke-run", fold.to_dict(), tuple(cluster_conditions), best[fold.fold_id], model_config, 2, deadline, tuple(feature_names), "smoke_cluster"))
        result = _execute_blocks(primary_blocks, 2, "smoke_primary")
        assert result["failed"] == 0, result
        result = _execute_blocks(conditional_blocks, 2, "smoke_conditional")
        assert result["failed"] == 0, result
        result = _execute_blocks(cluster_blocks, 2, "smoke_cluster")
        assert result["failed"] == 0, result
        rerun = _execute_blocks(primary_blocks, 2, "smoke_primary_resume")
        assert rerun["trained"] == 0 and rerun["cached"] == len(folds) * (len(feature_names) + 1), rerun
        summary = aggregate_results(output, len(feature_names), len(folds), prepared.signature, "smoke-run")
        assert summary["completed_primary_single_feature_tasks"] == len(feature_names) * len(folds), summary
        assert (output / "feature_master_decision.csv").exists()
        assert (output / "conditional_ablation_summary.csv").stat().st_size > 0
        assert (output / "cluster_ablation_summary.csv").stat().st_size > 0
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("SMOKE_TEST_OK", root)
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
