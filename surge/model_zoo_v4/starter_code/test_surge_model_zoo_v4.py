from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from surge_model_zoo_common import (  # noqa: E402
    apply_threshold_policy,
    choose_threshold_policy,
    daily_fraction_selection,
    derive_forward_targets,
    maximum_recall_at_alert_rate,
    theoretical_minimum_alert_rate,
    threshold_for_target_recall,
)
from run_surge_model_zoo_v4 import (  # noqa: E402
    DataBundle,
    Recipe,
    fit_predict_recipe,
    fit_predict_xgboost,
)
from surge_model_zoo_deployment import (  # noqa: E402
    predict_single_saved_model,
    save_catboost_model,
    save_extra_trees_model,
    save_lightgbm_model,
    save_xgboost_model,
)


class DummyArgs:
    max_tuning_rounds = 30
    early_stopping_rounds = 5
    xgboost_threads = 2
    threads_per_model = 2


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def build_synthetic_package(root: Path, date_count: int = 300, ticker_count: int = 6) -> dict[str, Path]:
    rng = np.random.default_rng(20260810)
    dates = pd.bdate_range("2020-01-01", periods=date_count)
    records: list[dict[str, object]] = []
    row_id = 0
    for ticker_id in range(ticker_count):
        ticker = f"T{ticker_id:02d}"
        signal = np.zeros(date_count, dtype=np.float64)
        regime = np.sin(np.linspace(0.0, 8.0 * np.pi, date_count) + ticker_id * 0.13)
        for index in range(1, date_count):
            signal[index] = 0.78 * signal[index - 1] + rng.normal(0.0, 0.72)
        returns = rng.normal(0.0002, 0.008, size=date_count)
        for index in range(1, date_count):
            up_probability = float(sigmoid(np.asarray([-3.05 + 1.05 * signal[index - 1] + 0.25 * regime[index - 1]]))[0])
            down_probability = float(sigmoid(np.asarray([-3.55 - 0.45 * signal[index - 1]]))[0])
            draw = rng.random()
            if draw < up_probability:
                returns[index] = 0.058 + rng.normal(0.0, 0.004)
            elif draw < up_probability + down_probability:
                returns[index] = -0.058 + rng.normal(0.0, 0.004)
        for index, date in enumerate(dates):
            records.append(
                {
                    "source_row_id": row_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "sealed_do_not_train_or_tune": 0,
                    "industry_name": f"I{ticker_id % 2}",
                    "market": "KOSPI" if ticker_id % 2 == 0 else "KOSDAQ",
                    "bucket": f"B{ticker_id % 3}",
                    "t_price_ret_1": float(returns[index]),
                    "f_signal": float(signal[index]),
                    "f_signal_noise": float(signal[index] + rng.normal(0.0, 0.35)),
                    "f_regime": float(regime[index]),
                    "f_ret_lag": float(returns[index]),
                    "f_interaction": float(signal[index] * (1.0 + 0.4 * regime[index])),
                    "f_noise_1": float(rng.normal()),
                    "f_noise_2": float(rng.normal()),
                    "f_sparse": float(signal[index]) if rng.random() > 0.16 else np.nan,
                }
            )
            row_id += 1
    source = pd.DataFrame.from_records(records)
    derived = derive_forward_targets(
        source,
        return_column="t_price_ret_1",
        ticker_column="ticker",
        date_column="date",
        threshold=0.05,
    )
    sidecar = source[["source_row_id", "date", "ticker"]].copy()
    sidecar["target_valid"] = derived["forward_valid"].eq(1.0)
    sidecar["label_abs_surge_3d_5pct"] = derived["surge_d3"]
    first_hit: list[float] = []
    best_forward: list[float] = []
    for _, group in source.groupby("ticker", sort=False):
        local_returns = group["t_price_ret_1"].to_numpy(dtype=np.float64)
        for local in range(len(group)):
            if local + 3 >= len(group):
                first_hit.append(np.nan)
                best_forward.append(np.nan)
                continue
            cumulative = np.cumprod(1.0 + local_returns[local + 1 : local + 4]) - 1.0
            hits = np.flatnonzero(cumulative >= 0.05 - 1e-12)
            first_hit.append(float(hits[0] + 1) if len(hits) else np.nan)
            best_forward.append(float(np.max(cumulative)))
    # Match the production target sidecar convention: 0 means no +5% hit.
    sidecar["first_hit_day"] = pd.Series(first_hit).fillna(0).astype(np.int8)
    sidecar["best_forward_return_3d"] = best_forward

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = data_dir / "synthetic_training.csv"
    target_path = data_dir / "synthetic_target.csv"
    source.to_csv(dataset_path, index=False)
    sidecar.to_csv(target_path, index=False)

    date_values = list(dates)
    folds = []
    boundaries = [
        (0, 119, 120, 124, 125, 164),
        (0, 159, 160, 164, 165, 204),
        (0, 199, 200, 204, 205, 244),
        (0, 239, 240, 244, 245, 284),
    ]
    for fold_id, (tr0, tr1, p0, p1, va0, va1) in enumerate(boundaries):
        folds.append(
            {
                "fold_id": fold_id,
                "train_start": date_values[tr0].strftime("%Y-%m-%d"),
                "train_end": date_values[tr1].strftime("%Y-%m-%d"),
                "purge_start": date_values[p0].strftime("%Y-%m-%d"),
                "purge_end": date_values[p1].strftime("%Y-%m-%d"),
                "validation_start": date_values[va0].strftime("%Y-%m-%d"),
                "validation_end": date_values[va1].strftime("%Y-%m-%d"),
            }
        )
    folds_path = root / "folds.json"
    folds_path.write_text(json.dumps({"folds": folds}, ensure_ascii=False, indent=2), encoding="utf-8")

    all_features = [
        "f_signal",
        "f_signal_noise",
        "f_regime",
        "f_ret_lag",
        "f_interaction",
        "f_noise_1",
        "f_noise_2",
        "f_sparse",
    ]
    profiles = {
        "profiles": {
            "P0_ALL_VALID": {"features": all_features},
            "P1_SELECTION_TOP": {"features": all_features[:6]},
            "P7S_UNIVARIATE_TOP3_LIFT": {"features": ["f_signal", "f_signal_noise", "f_interaction", "f_sparse"]},
            "P4S_METRIC_CONSISTENT_DIRECTIONAL": {"features": ["f_signal", "f_ret_lag", "f_regime"]},
        }
    }
    profiles_path = root / "profiles.json"
    profiles_path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2), encoding="utf-8")

    recipes = {
        "schema": "synthetic_model_zoo_test_v1",
        "custom_profiles": {},
        "quick_recipe_names": ["lgb_signal", "extra_signal"],
        "recipes": [
            {
                "name": "lgb_signal",
                "family": "lightgbm",
                "profile": "P7S_UNIVARIATE_TOP3_LIFT",
                "target_variant": "surge_d3",
                "train_policy": "expanding",
                "positive_weight_mode": "sqrt_balance",
                "params": {
                    "learning_rate": 0.08,
                    "num_leaves": 15,
                    "min_data_in_leaf": 15,
                    "feature_fraction": 0.8,
                    "bagging_fraction": 0.8,
                    "bagging_freq": 1,
                },
            },
            {
                "name": "extra_signal",
                "family": "extra_trees",
                "profile": "P1_SELECTION_TOP",
                "target_variant": "surge_d3",
                "train_policy": "rolling",
                "rolling_days": 100,
                "positive_weight_mode": "sqrt_balance",
                "params": {
                    "n_estimators": 80,
                    "max_features": 0.75,
                    "min_samples_leaf": 3,
                    "max_depth": 10,
                },
            },
        ],
    }
    recipes_path = root / "recipes.json"
    recipes_path.write_text(json.dumps(recipes, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "dataset": dataset_path,
        "target": target_path,
        "folds": folds_path,
        "profiles": profiles_path,
        "recipes": recipes_path,
    }


class ThresholdTests(unittest.TestCase):
    def test_exact_recall_threshold(self) -> None:
        target = np.asarray([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        score = np.asarray([0.99, 0.97, 0.90, 0.85, 0.80, 0.79, 0.75, 0.70, 0.60, 0.10])
        threshold, metrics = threshold_for_target_recall(target, score, 0.70)
        self.assertTrue(math.isfinite(threshold))
        self.assertGreaterEqual(float(metrics["recall"]), 0.70)
        self.assertLess(float(metrics["alert_rate"]), 1.0)

    def test_top3_is_structurally_incompatible_with_recall70(self) -> None:
        selection_positive_rate = 0.132068
        top_two_of_48 = 2.0 / 48.0
        maximum = maximum_recall_at_alert_rate(selection_positive_rate, top_two_of_48)
        self.assertLess(maximum, 0.32)
        self.assertAlmostEqual(theoretical_minimum_alert_rate(selection_positive_rate, 0.70), 0.0924476, places=6)

    def test_daily_policy_transfer_shape(self) -> None:
        dates = np.repeat(pd.date_range("2024-01-01", periods=10), 10).to_numpy()
        target = np.tile(np.asarray([1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8), 10)
        scores = np.tile(np.linspace(1.0, 0.1, 10), 10)
        policy = choose_threshold_policy(
            target,
            scores,
            dates,
            target_recall=0.70,
            daily_fraction_grid=[0.1, 0.2, 0.3, 0.5, 1.0],
            minimum_precision_lift=1.0,
        )
        selected = apply_threshold_policy(policy, scores, dates)
        self.assertEqual(len(selected), len(target))
        self.assertGreaterEqual(int(selected.sum()), 10)


class ModelRandomnessTests(unittest.TestCase):
    def test_xgboost_stochastic_seeds_are_not_identical(self) -> None:
        rng = np.random.default_rng(17)
        x = rng.normal(size=(500, 12)).astype(np.float32)
        y = (x[:, 0] + 0.4 * x[:, 1] + rng.normal(scale=0.8, size=500) > 0.8).astype(np.uint8)
        weights = np.ones(len(y), dtype=np.float64)
        params = {
            "learning_rate": 0.08,
            "max_depth": 5,
            "max_leaves": 31,
            "subsample": 0.72,
            "colsample_bytree": 0.70,
            "colsample_bynode": 0.75,
        }
        first = fit_predict_xgboost(
            x[:400], y[:400], weights[:400], x[400:], y[400:], params, 17, 2, "cpu", 40, None, 40
        ).prediction
        second = fit_predict_xgboost(
            x[:400], y[:400], weights[:400], x[400:], y[400:], params, 29, 2, "cpu", 40, None, 40
        ).prediction
        self.assertGreater(float(np.max(np.abs(first - second))), 1e-7)

    def test_two_stage_xgboost_smoke(self) -> None:
        rng = np.random.default_rng(2026)
        rows = 900
        features = [f"f{index}" for index in range(6)]
        matrix = rng.normal(size=(rows, len(features))).astype(np.float32)
        large_probability = sigmoid(-0.8 + 1.0 * matrix[:, 0] + 0.5 * matrix[:, 1])
        large_move = (rng.random(rows) < large_probability).astype(np.uint8)
        direction_probability = sigmoid(0.2 + 1.1 * matrix[:, 3] - 0.5 * matrix[:, 4])
        direction_up = (rng.random(rows) < direction_probability).astype(np.uint8)
        direction_valid = large_move.copy()
        surge = (large_move.astype(bool) & direction_up.astype(bool)).astype(np.uint8)
        crash = (large_move.astype(bool) & ~direction_up.astype(bool)).astype(np.uint8)
        with tempfile.TemporaryDirectory(prefix="surge_two_stage_") as temp:
            matrix_path = Path(temp) / "matrix.npy"
            np.save(matrix_path, matrix)
            dates = pd.bdate_range("2022-01-03", periods=rows).to_numpy(dtype="datetime64[ns]")
            frame = pd.DataFrame({"row_index": np.arange(rows), "date": dates, "ticker": "T00"})
            bundle = DataBundle(
                matrix_path=matrix_path,
                metadata_path=Path(temp) / "manifest.json",
                features=features,
                feature_to_index={feature: index for index, feature in enumerate(features)},
                frame=frame,
                targets={
                    "surge_d1": surge,
                    "surge_d2": surge,
                    "surge_d3": surge,
                    "crash_d3": crash,
                    "large_move": large_move,
                    "direction_valid": direction_valid,
                    "direction_up": direction_up,
                },
                dates=dates,
                dataset_sha256="synthetic",
                target_sha256="synthetic",
                cache_identity_hash="synthetic",
            )
            recipe = Recipe(
                name="two_stage_smoke",
                family="two_stage_xgboost",
                profile="P7S",
                direction_profile="P4S",
                target_variant="two_stage",
                train_policy="expanding",
                positive_weight_mode="sqrt_balance",
                params={
                    "learning_rate": 0.08,
                    "max_depth": 4,
                    "max_leaves": 31,
                    "subsample": 0.75,
                    "colsample_bytree": 0.8,
                },
                direction_params={
                    "learning_rate": 0.08,
                    "max_depth": 4,
                    "subsample": 0.75,
                    "colsample_bytree": 0.8,
                },
            )
            result = fit_predict_recipe(
                recipe,
                np.load(matrix_path, mmap_mode="r"),
                bundle,
                {"P7S": features[:3], "P4S": features[3:]},
                np.arange(0, 700, dtype=np.int64),
                np.arange(700, rows, dtype=np.int64),
                seed=17,
                args=DummyArgs(),
                device="cpu",
                fixed_iterations=None,
            )
            self.assertEqual(len(result.prediction), rows - 700)
            self.assertTrue(np.isfinite(result.prediction).all())
            self.assertTrue(np.all((result.prediction > 0.0) & (result.prediction < 1.0)))
            self.assertIsInstance(result.best_iteration, dict)


class DeploymentBackendTests(unittest.TestCase):
    def test_all_backend_serialization_smoke(self) -> None:
        rng = np.random.default_rng(88)
        x = rng.normal(size=(360, 7)).astype(np.float32)
        y = (1.1 * x[:, 0] - 0.7 * x[:, 1] + rng.normal(scale=0.9, size=len(x)) > 0.8).astype(np.uint8)
        weights = np.ones(len(y), dtype=np.float64)
        validation = rng.normal(size=(40, 7)).astype(np.float32)
        with tempfile.TemporaryDirectory(prefix="surge_backend_models_") as temp:
            root = Path(temp)
            records = [
                save_lightgbm_model(
                    x,
                    y,
                    weights,
                    {
                        "objective": "binary",
                        "metric": "average_precision",
                        "learning_rate": 0.08,
                        "num_leaves": 15,
                        "min_data_in_leaf": 10,
                        "verbosity": -1,
                        "num_threads": 2,
                        "feature_fraction": 0.8,
                        "bagging_fraction": 0.8,
                        "bagging_freq": 1,
                        "seed": 17,
                    },
                    20,
                    root / "lgb.txt",
                ),
                save_xgboost_model(
                    x,
                    y,
                    weights,
                    {
                        "objective": "binary:logistic",
                        "eval_metric": "aucpr",
                        "tree_method": "hist",
                        "device": "cpu",
                        "learning_rate": 0.08,
                        "max_depth": 4,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "seed": 17,
                        "nthread": 2,
                    },
                    20,
                    root / "xgb.json",
                ),
                save_catboost_model(
                    x,
                    y,
                    weights,
                    {
                        "loss_function": "Logloss",
                        "eval_metric": "AUC",
                        "learning_rate": 0.08,
                        "depth": 5,
                        "random_seed": 17,
                        "thread_count": 2,
                        "task_type": "CPU",
                        "verbose": False,
                        "allow_writing_files": False,
                    },
                    20,
                    root / "cat.cbm",
                ),
                save_extra_trees_model(
                    x,
                    y,
                    weights,
                    {"n_estimators": 40, "max_features": 0.8, "min_samples_leaf": 2, "max_depth": 8},
                    17,
                    2,
                    root / "extra.joblib",
                ),
            ]
            for record in records:
                prediction = predict_single_saved_model(record, validation, root)
                self.assertEqual(len(prediction), len(validation))
                self.assertTrue(np.isfinite(prediction).all())
                self.assertTrue(np.all((prediction >= 0.0) & (prediction <= 1.0)))
                self.assertEqual(len(record["sha256"]), 64)


class EndToEndTests(unittest.TestCase):
    def test_synthetic_end_to_end_and_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="surge_model_zoo_test_") as temp:
            root = Path(temp)
            paths = build_synthetic_package(root)
            output = root / "outputs" / "model_zoo"
            command = [
                sys.executable,
                str(HERE / "run_surge_model_zoo_v4.py"),
                "--package-root", str(root),
                "--dataset", str(paths["dataset"]),
                "--target-sidecar", str(paths["target"]),
                "--folds", str(paths["folds"]),
                "--profiles", str(paths["profiles"]),
                "--recipes", str(paths["recipes"]),
                "--output", str(output),
                "--selection-folds", "0,1",
                "--confirmation-folds", "2",
                "--recent-folds", "3",
                "--screen-seeds", "17",
                "--final-seeds", "17,29",
                "--families", "lightgbm,extra_trees",
                "--quick",
                "--finalist-count", "2",
                "--device", "cpu",
                "--threads-per-model", "2",
                "--target-recall", "0.70",
                "--threshold-recall-buffer", "0.02",
                "--minimum-oof-precision-lift", "1.0",
                "--minimum-holdout-precision-lift", "0.9",
                "--max-alert-rate", "0.90",
                "--daily-fraction-grid", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                "--top-fractions", "0.03,0.10,0.30,0.50",
                "--inner-windows", "2",
                "--inner-validation-days", "20",
                "--inner-purge-days", "5",
                "--inner-step-days", "20",
                "--minimum-inner-train-days", "60",
                "--minimum-purge-trading-days", "3",
                "--max-tuning-rounds", "45",
                "--early-stopping-rounds", "7",
                "--minimum-iterations", "5",
                "--maximum-iterations", "40",
                "--minimum-calibration-rows", "40",
                "--minimum-scope-rows", "10",
                "--minimum-scope-positives", "1",
            ]
            first = subprocess.run(command, capture_output=True, text=True, timeout=240)
            if first.returncode != 0:
                self.fail(f"first run failed\nSTDOUT:\n{first.stdout}\nSTDERR:\n{first.stderr}")
            status = json.loads((output / "RUN_STATUS.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "SUCCESS")
            verify_command = [
                sys.executable,
                str(HERE / "verify_surge_model_zoo_v4.py"),
                "--output", str(output),
            ]
            verification = subprocess.run(verify_command, capture_output=True, text=True, timeout=180)
            if verification.returncode != 0:
                self.fail(
                    f"verification failed\nSTDOUT:\n{verification.stdout}\nSTDERR:\n{verification.stderr}"
                )
            recommendation = json.loads((output / "FINAL_RECOMMENDATION.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(float(recommendation["target_recall"]), 0.70)
            self.assertTrue((output / "PERFORMANCE_GAP_REPORT_KO.md").exists())
            self.assertTrue((output / "PRODUCTION_MODEL_REGISTRY.json").exists())
            role_metrics = pd.read_csv(output / "frozen_ensemble_metrics_by_role.csv")
            self.assertEqual(set(role_metrics["fold_role"]), {"selection", "confirmation", "recent_audit"})
            diversity = pd.read_csv(output / "seed_prediction_diversity.csv")
            self.assertFalse(diversity.empty)

            scoring_output = root / "outputs" / "scored"
            scoring_command = [
                sys.executable,
                str(HERE / "score_surge_frozen_ensemble_v4.py"),
                "--model-dir", str(output),
                "--dataset", str(paths["dataset"]),
                "--target-sidecar", str(paths["target"]),
                "--output", str(scoring_output),
            ]
            scoring = subprocess.run(scoring_command, capture_output=True, text=True, timeout=180)
            if scoring.returncode != 0:
                self.fail(f"scoring failed\nSTDOUT:\n{scoring.stdout}\nSTDERR:\n{scoring.stderr}")
            scored = pd.read_csv(scoring_output / "surge_scores.csv")
            self.assertEqual(len(scored), 300 * 6)
            self.assertTrue(scored["surge_probability"].between(0.0, 1.0).all())
            evaluation = json.loads((scoring_output / "SCORING_EVALUATION.json").read_text(encoding="utf-8"))
            self.assertIn("frozen_policy", evaluation)

            predictions_before = (output / "frozen_ensemble_predictions.npz").read_bytes()
            registry_before = (output / "PRODUCTION_MODEL_REGISTRY.json").read_bytes()

            second = subprocess.run(command + ["--resume"], capture_output=True, text=True, timeout=180)
            if second.returncode != 0:
                self.fail(f"resume failed\nSTDOUT:\n{second.stdout}\nSTDERR:\n{second.stderr}")
            predictions_after = (output / "frozen_ensemble_predictions.npz").read_bytes()
            registry_after = (output / "PRODUCTION_MODEL_REGISTRY.json").read_bytes()
            self.assertEqual(predictions_before, predictions_after)
            self.assertEqual(registry_before, registry_after)

            registry_payload = json.loads(registry_after.decode("utf-8"))
            model_path: Path | None = None
            for recipe_record in registry_payload.get("models", []):
                for key in ("model", "large_move_model", "direction_model"):
                    model_record = recipe_record.get(key)
                    if not isinstance(model_record, dict):
                        continue
                    if model_record.get("format") == "constant_probability":
                        continue
                    raw_path = Path(str(model_record["path"]))
                    model_path = raw_path if raw_path.is_absolute() else output / raw_path
                    break
                if model_path is not None:
                    break
            self.assertIsNotNone(model_path)
            assert model_path is not None
            with model_path.open("ab") as handle:
                handle.write(b"\nSURGE_V4_TAMPER_TEST")
            broken_verification = subprocess.run(verify_command, capture_output=True, text=True, timeout=180)
            self.assertNotEqual(broken_verification.returncode, 0)

            repaired = subprocess.run(command + ["--resume"], capture_output=True, text=True, timeout=240)
            if repaired.returncode != 0:
                self.fail(f"repair resume failed\nSTDOUT:\n{repaired.stdout}\nSTDERR:\n{repaired.stderr}")
            repaired_verification = subprocess.run(verify_command, capture_output=True, text=True, timeout=180)
            if repaired_verification.returncode != 0:
                self.fail(
                    f"repaired verification failed\nSTDOUT:\n{repaired_verification.stdout}"
                    f"\nSTDERR:\n{repaired_verification.stderr}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
