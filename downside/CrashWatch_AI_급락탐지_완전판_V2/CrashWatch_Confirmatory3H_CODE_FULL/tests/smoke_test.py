from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cw3h.aggregate import aggregate_results
from cw3h.cache import _write_frame_matrix, _write_masked_copy, _write_subset_matrix
from cw3h.folds import FoldSlice, build_fold_slices
from cw3h.model import BlockTask, TuneTask, run_lightgbm_block, tune_best_iteration
from cw3h.utils import atomic_json, hash_strings


def make_profile(cache_root: Path, profile: str, seed: int = 7) -> tuple[FoldSlice, list[str]]:
    rng = np.random.default_rng(seed)
    profile_dir = cache_root / "profiles" / profile
    matrix_dir = profile_dir / "matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    trading_days = np.arange(np.datetime64("2022-01-03"), np.datetime64("2025-01-01"), dtype="datetime64[D]")
    weekdays = trading_days[np.is_busday(trading_days)][:700]
    tickers = np.array(["003670", "006400", "005930", "000660", "035420", "000270"], dtype="U16")
    n_dates = len(weekdays)
    n_tickers = len(tickers)
    n_rows = n_dates * n_tickers
    dates = np.repeat(weekdays.astype("datetime64[ns]").astype(np.int64), n_tickers)
    ticker_rows = np.tile(tickers, n_dates)
    buckets = np.where(np.isin(ticker_rows, ["003670", "006400"]), "battery_materials", "other").astype("U32")

    n_features = 72 if profile == "full_reduced" else 64
    features = [f"f_{i:03d}" for i in range(n_features)]
    x = rng.normal(size=(n_rows, n_features)).astype(np.float32)
    date_signal = np.repeat(rng.normal(size=n_dates).astype(np.float32), n_tickers)
    latent = 1.5 * x[:, 0] - 1.0 * x[:, 1] + 0.55 * x[:, 2] + 0.3 * date_signal + rng.normal(0, 1.2, n_rows)
    threshold = np.quantile(latent, 0.88)
    y = (latent > threshold).astype(np.uint8)

    np.save(profile_dir / "dates_ns.npy", dates, allow_pickle=False)
    np.save(profile_dir / "tickers.npy", ticker_rows, allow_pickle=False)
    np.save(profile_dir / "buckets.npy", buckets, allow_pickle=False)
    np.save(profile_dir / "target.npy", y, allow_pickle=False)
    np.save(profile_dir / "original_row_id.npy", np.arange(n_rows, dtype=np.int64), allow_pickle=False)
    np.save(matrix_dir / "B0.npy", x, allow_pickle=False)
    np.save(matrix_dir / "A1.npy", x[:, 3:], allow_pickle=False)

    manifest = {
        "profile": profile,
        "conditions": {
            "B0": {"features": features, "feature_hash": hash_strings(features), "mode": "baseline"},
            "A1": {"features": features[3:], "feature_hash": hash_strings(features[3:]), "mode": "global_drop"},
        },
    }
    atomic_json(manifest, profile_dir / "profile_manifest.json")

    train_dates = 560
    purge_dates = 20
    validation_dates = 60
    train_stop = train_dates * n_tickers
    validation_start = (train_dates + purge_dates) * n_tickers
    validation_stop = (train_dates + purge_dates + validation_dates) * n_tickers
    fold = FoldSlice(
        fold_id=7,
        train_start=0,
        train_stop=train_stop,
        validation_start=validation_start,
        validation_stop=validation_stop,
        train_date_min=str(weekdays[0]),
        train_date_max=str(weekdays[train_dates - 1]),
        validation_date_min=str(weekdays[train_dates + purge_dates]),
        validation_date_max=str(weekdays[train_dates + purge_dates + validation_dates - 1]),
        train_dates=train_dates,
        validation_dates=validation_dates,
        train_rows=train_stop,
        validation_rows=validation_stop - validation_start,
        eligible=True,
        reason="",
    )
    return fold, features


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cw3h_smoke_") as temp:
        temp_root = Path(temp)

        # Matrix cache primitives: dataframe -> canonical memmap -> feature subset -> masked copy.
        cache_primitive_dir = temp_root / "cache_primitives"
        cache_primitive_dir.mkdir(parents=True, exist_ok=True)
        tiny = pd.DataFrame({f"z{i}": np.arange(120, dtype=np.float32) + i for i in range(8)})
        canonical = cache_primitive_dir / "canonical.npy"
        subset = cache_primitive_dir / "subset.npy"
        masked = cache_primitive_dir / "masked.npy"
        _write_frame_matrix(tiny, list(tiny.columns), canonical, chunk_rows=17)
        _write_subset_matrix(canonical, None, np.array([0, 3, 7], dtype=np.int64), subset, chunk_rows=19)
        row_mask = np.zeros(len(tiny), dtype=bool)
        row_mask[::5] = True
        _write_masked_copy(subset, masked, row_mask, np.array([1], dtype=np.int64), chunk_rows=23)
        subset_arr = np.load(subset)
        masked_arr = np.load(masked)
        assert subset_arr.shape == (120, 3)
        assert np.isnan(masked_arr[row_mask, 1]).all()
        assert np.allclose(masked_arr[~row_mask], subset_arr[~row_mask], equal_nan=True)

        # Outer-fold date boundaries and the minimum 20-trading-day purge are validated.
        fold_check_dir = temp_root / "fold_check"
        fold_check_dir.mkdir(parents=True, exist_ok=True)
        check_days = np.arange(np.datetime64("2020-01-01"), np.datetime64("2024-01-01"), dtype="datetime64[D]")
        check_days = check_days[np.is_busday(check_days)][:700]
        check_dates = np.repeat(check_days.astype("datetime64[ns]").astype(np.int64), 2)
        np.save(fold_check_dir / "dates_ns.npy", check_dates, allow_pickle=False)
        good_def = [{
            "fold_id": 0,
            "train_start": str(check_days[0]),
            "train_end": str(check_days[499]),
            "validation_start": str(check_days[520]),
            "validation_end": str(check_days[579]),
        }]
        good_fold = build_fold_slices(fold_check_dir, good_def, 500, 60, 20)[0]
        assert good_fold.eligible, good_fold
        bad_def = [dict(good_def[0], validation_start=str(check_days[519]), validation_end=str(check_days[578]))]
        bad_fold = build_fold_slices(fold_check_dir, bad_def, 500, 60, 20)[0]
        assert not bad_fold.eligible and "purge_dates=19" in bad_fold.reason, bad_fold

        cache_root = temp_root / "cache" / "synthetic-signature"
        output_dir = temp_root / "output"
        folds = {}
        for index, profile in enumerate(["full_reduced", "common_period"]):
            fold, _ = make_profile(cache_root, profile, seed=7 + index)
            folds[profile] = fold

        lgb_cfg = {
            "objective": "binary",
            "learning_rate": 0.08,
            "num_leaves": 31,
            "max_depth": -1,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.9,
            "bagging_fraction": 1.0,
            "bagging_freq": 0,
            "lambda_l1": 0.0,
            "lambda_l2": 0.5,
            "max_bin": 127,
            "max_rounds": 60,
            "early_stopping_rounds": 10,
            "deterministic": True,
            "force_col_wise": True,
        }
        best = {}
        for profile, fold in folds.items():
            tune = tune_best_iteration(TuneTask(
                cache_root=str(cache_root),
                profile=profile,
                fold=fold.to_dict(),
                dataset_signature="synthetic-signature",
                lightgbm_config=lgb_cfg,
                threads=2,
                inner_validation_days=40,
                inner_purge_days=20,
                min_train_days=400,
                seed=17,
            ))
            assert tune["status"] == "completed", tune
            best[profile] = int(tune["best_iteration"])

        for profile, fold in folds.items():
            for condition in ["B0", "A1"]:
                result = run_lightgbm_block(BlockTask(
                    cache_root=str(cache_root),
                    profile=profile,
                    condition=condition,
                    phase="smoke",
                    fold=fold.to_dict(),
                    seeds=[17, 43],
                    best_iteration=best[profile],
                    dataset_signature="synthetic-signature",
                    lightgbm_config=lgb_cfg,
                    threads=2,
                    deadline_epoch=time.time() + 300,
                    save_predictions=False,
                    prediction_compression="zstd",
                ))
                assert result["status"] == "completed", result

        # Cache-hit contract: exact same task must not retrain.
        cached = run_lightgbm_block(BlockTask(
            cache_root=str(cache_root),
            profile="common_period",
            condition="B0",
            phase="smoke_rerun",
            fold=folds["common_period"].to_dict(),
            seeds=[17, 43],
            best_iteration=best["common_period"],
            dataset_signature="synthetic-signature",
            lightgbm_config=lgb_cfg,
            threads=2,
            deadline_epoch=time.time() + 300,
            save_predictions=False,
            prediction_compression="zstd",
        ))
        assert cached["cache_only"] is True, cached

        config = {
            "decision_thresholds": {
                "global_min_mean_loss": 0.005,
                "global_min_positive_fold_ratio": 0.75,
                "global_worst_fold_floor": -0.005,
                "joint_incremental_min": 0.003,
                "etf_incremental_min": 0.001,
                "battery_min_mean_loss": 0.010,
            }
        }
        summary = aggregate_results(cache_root, output_dir, config, {"smoke_test": True})
        assert summary["status"] == "completed", summary
        assert summary["completed_task_count"] == 8, summary
        assert (output_dir / "paired_ablation_deltas.csv").exists()
        assert (output_dir / "global_feature_utility_summary.csv").exists()
        print(json.dumps({
            "status": "PASS",
            "best_iterations": best,
            "completed_task_count": summary["completed_task_count"],
            "paired_delta_row_count": summary["paired_delta_row_count"],
            "cache_hit_verified": True,
            "matrix_cache_primitives_verified": True,
            "purge_boundary_validation_verified": True,
        }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
