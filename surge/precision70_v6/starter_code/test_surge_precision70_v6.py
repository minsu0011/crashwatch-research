from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import pandas as pd

from score_surge_precision70_v6 import apply_frozen_method, population_stability_index
from run_surge_precision70_v6 import (
    METHOD_CANDIDATES,
    _episode_balance_weights,
    temporal_oof_method_scores,
)
from surge_model_zoo_common import sha256_file, with_payload_checksum
from update_surge_precision_controller_v6 import clamp_policy_to_base, mature_cutoff
from surge_precision_common_v6 import (
    CalibratorSpec,
    PrecisionPolicy,
    apply_precision_policy,
    build_meta_feature_frame,
    evaluate_alerts,
    fit_best_calibrator,
    forward_training_folds,
    rank_before_seed_average,
    select_global_precision_policy,
    select_online_precision_threshold_fast,
    select_scope_precision_policy,
    wilson_lower_bound,
)


class PrecisionCommonTests(unittest.TestCase):
    def test_global_policy_has_no_alert_count_cap_and_maximizes_coverage(self) -> None:
        records = []
        rng = np.random.default_rng(7)
        for fold in range(1, 5):
            for row in range(120):
                target = int(row < 36)
                score = (0.95 - row * 0.004) if target else (0.62 - (row - 36) * 0.003)
                score += rng.normal(scale=0.001)
                records.append(
                    {
                        "fold_id": fold,
                        "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=fold * 150 + row // 12),
                        "ticker": f"T{row % 12:02d}",
                        "target": target,
                        "score": score,
                    }
                )
        frame = pd.DataFrame(records)
        policy, search, selected = select_global_precision_policy(
            frame,
            target_precision=0.70,
            minimum_precision_lcb=0.55,
            minimum_alerts_per_fold=20,
            minimum_alert_days_per_fold=2,
            required_fold_pass_rate=1.0,
            maximum_threshold_candidates=120,
        )
        self.assertTrue(policy.gate_pass)
        self.assertEqual(policy.kind, "global_threshold")
        self.assertFalse(hasattr(policy, "max_alerts_per_day"))
        self.assertGreaterEqual(policy.achieved_alerts or 0, 80)
        self.assertTrue(bool(search.iloc[0]["gate_pass"]))
        self.assertEqual(set(selected["fold_id"]), {1, 2, 3, 4})


    def test_online_threshold_abstains_when_precision_gate_is_impossible(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=100, freq="D"),
                "target": np.tile([1, 0, 0, 0, 0], 20),
                "score": np.linspace(0.0, 1.0, 100),
            }
        )
        threshold, gate, metrics = select_online_precision_threshold_fast(
            frame,
            target_precision=0.90,
            minimum_precision_lcb=0.70,
            minimum_alerts=20,
            minimum_alert_days=10,
            maximum_threshold_candidates=80,
        )
        self.assertFalse(gate)
        self.assertTrue(np.isinf(threshold))
        self.assertEqual(metrics["status"], "NO_SAFE_THRESHOLD")

    def test_online_threshold_can_alert_all_without_hidden_cap(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=100, freq="D"),
                "target": np.r_[np.ones(80, dtype=np.int8), np.zeros(20, dtype=np.int8)],
                "score": np.linspace(0.0, 1.0, 100),
            }
        )
        threshold, gate, metrics = select_online_precision_threshold_fast(
            frame,
            target_precision=0.70,
            minimum_precision_lcb=0.60,
            minimum_alerts=20,
            minimum_alert_days=10,
            maximum_threshold_candidates=80,
        )
        self.assertTrue(gate)
        self.assertLessEqual(threshold, float(frame["score"].min()))
        self.assertEqual(metrics["alerts"], 100)

    def test_no_alert_cap_allows_threshold_below_median(self) -> None:
        records = []
        rng = np.random.default_rng(812)
        for fold in range(1, 5):
            for row in range(100):
                # 80% prevalence means alerting every row still has precision 80%.
                # The optimal no-cap policy must therefore select all rows.
                target = int(row < 80)
                records.append(
                    {
                        "fold_id": fold,
                        "date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=fold * 120 + row // 10),
                        "ticker": f"T{row % 10:02d}",
                        "target": target,
                        "score": float(rng.uniform(0.0, 1.0)),
                    }
                )
        frame = pd.DataFrame(records)
        policy, _, _ = select_global_precision_policy(
            frame,
            target_precision=0.70,
            minimum_precision_lcb=0.55,
            minimum_alerts_per_fold=20,
            minimum_alert_days_per_fold=2,
            required_fold_pass_rate=1.0,
            maximum_threshold_candidates=120,
        )
        alert = apply_precision_policy(policy, frame["score"].to_numpy(dtype=np.float64))
        self.assertTrue(policy.gate_pass)
        self.assertGreater(float(alert.mean()), 0.95)

    def test_crash_and_agreement_are_not_averaged_as_surge_probability(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-02", "2023-01-02"]),
                "surge_signal": [0.80, 0.60],
                "crash_signal": [0.90, 0.20],
                "surge_signal__seed_agreement": [0.95, 0.70],
            }
        )
        meta, spec = build_meta_feature_frame(
            frame,
            ["surge_signal", "crash_signal", "surge_signal__seed_agreement"],
            {
                "surge_signal": "lightgbm",
                "crash_signal": "lightgbm_crash",
                "surge_signal__seed_agreement": "seed_agreement",
            },
            scope_columns=[],
        )
        np.testing.assert_allclose(meta["ensemble_mean"], [0.80, 0.60])
        np.testing.assert_allclose(meta["crash_risk_mean"], [0.90, 0.20])
        np.testing.assert_allclose(meta["seed_agreement_mean"], [0.95, 0.70])
        np.testing.assert_allclose(meta["surge_safety_product"], [0.08, 0.48], atol=1e-6)
        self.assertEqual(spec["signal_roles"]["crash_signal"], "crash")
        self.assertEqual(
            spec["signal_roles"]["surge_signal__seed_agreement"],
            "agreement",
        )

    def test_short_history_temporal_oof_never_scores_training_dates(self) -> None:
        dates = pd.bdate_range("2022-01-03", periods=8)
        rows = []
        for date_index, date in enumerate(dates):
            for ticker_index in range(5):
                target = int((date_index + ticker_index) % 3 == 0)
                rows.append(
                    {
                        "fold_id": 0,
                        "date": date,
                        "target": target,
                        "ticker": f"T{ticker_index}",
                        "signal_a": 0.75 * target + 0.02 * ticker_index,
                    }
                )
        frame = pd.DataFrame(rows)
        args = SimpleNamespace(
            meta_oof_initial_days=100,
            meta_oof_block_days=40,
            minimum_meta_train_rows=5,
            scope_columns=[],
            minimum_calibration_rows=5,
            meta_time_decay_half_life_days=None,
            meta_logistic_c=1.0,
            threads_per_model=1,
            meta_early_stopping_rounds=5,
            meta_max_rounds=20,
            resolved_xgboost_device="cpu",
            xgboost_threads=1,
        )
        scored = temporal_oof_method_scores(
            METHOD_CANDIDATES[0],
            frame,
            ["signal_a"],
            {"signal_a": "synthetic"},
            args,
            seed=17,
        )
        self.assertGreater(len(scored), 0)
        self.assertGreater(pd.to_datetime(scored["date"]).min(), dates[0])
        self.assertLess(len(scored), len(frame))

    def test_minimum_sample_gate_blocks_one_alert_precision_trick(self) -> None:
        frame = pd.DataFrame(
            {
                "fold_id": np.repeat([1, 2, 3], 100),
                "date": pd.date_range("2020-01-01", periods=300, freq="D"),
                "ticker": [f"T{i % 10}" for i in range(300)],
                "target": np.tile(np.asarray([1] + [0] * 99, dtype=np.int8), 3),
                "score": np.tile(np.asarray([1.0] + list(np.linspace(0.99, 0.01, 99))), 3),
            }
        )
        policy, _, _ = select_global_precision_policy(
            frame,
            target_precision=0.70,
            minimum_precision_lcb=0.10,
            minimum_alerts_per_fold=20,
            minimum_alert_days_per_fold=2,
            required_fold_pass_rate=1.0,
            maximum_threshold_candidates=40,
        )
        self.assertFalse(policy.gate_pass)

    def test_scope_policy_and_fallback(self) -> None:
        rows = []
        for market in ["KOSPI", "KOSDAQ"]:
            for index in range(300):
                target = int(index % 5 == 0)
                score = 0.9 if target else 0.1 + 0.2 * ((index % 11) / 10)
                if market == "KOSDAQ":
                    score = score * 0.8
                rows.append(
                    {
                        "fold_id": 1,
                        "date": pd.Timestamp("2021-01-01") + pd.Timedelta(days=index // 10),
                        "ticker": f"{market}_{index % 10}",
                        "market": market,
                        "target": target,
                        "score": score,
                    }
                )
        frame = pd.DataFrame(rows)
        policy, audit = select_scope_precision_policy(
            frame,
            "market",
            target_precision=0.70,
            minimum_precision_lcb=0.50,
            minimum_alerts_per_scope=20,
            minimum_alert_days_per_scope=5,
        )
        alert = apply_precision_policy(policy, frame["score"].to_numpy(), frame["market"].to_numpy())
        metrics = evaluate_alerts(
            frame["target"].to_numpy(),
            frame["score"].to_numpy(),
            alert,
            frame["date"].to_numpy(),
            frame["ticker"].to_numpy(),
        )
        self.assertGreaterEqual(metrics["precision"], 0.70)
        self.assertIn("__COMBINED__", set(audit["scope"].astype(str)))

    def test_seed_rank_aggregation_is_scale_invariant(self) -> None:
        dates = np.asarray(
            [np.datetime64("2025-01-01")] * 4 + [np.datetime64("2025-01-02")] * 4
        )
        seed_a = np.asarray([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        seed_b = seed_a * 1000.0 + 13.0
        mean, std = rank_before_seed_average([seed_a, seed_b], dates)
        reference, reference_std = rank_before_seed_average([seed_a, seed_a], dates)
        np.testing.assert_allclose(mean, reference)
        np.testing.assert_allclose(std, reference_std)

    def test_forward_meta_training_never_uses_future_fold(self) -> None:
        self.assertEqual(forward_training_folds([0, 1, 2, 3, 4], 0), [])
        self.assertEqual(forward_training_folds([0, 1, 2, 3, 4], 3), [0, 1, 2])

    def test_target_specific_episode_weights_do_not_reuse_surge_runs(self) -> None:
        bundle = SimpleNamespace(
            targets={
                "surge_d3": np.asarray([1, 1, 0, 0, 0, 0], dtype=np.int8),
                "crash_d3": np.asarray([0, 0, 1, 1, 1, 0], dtype=np.int8),
            },
            frame=pd.DataFrame({"ticker": ["T"] * 6}),
            dates=pd.date_range("2025-01-01", periods=6, freq="B").to_numpy(),
        )
        surge = _episode_balance_weights(bundle, "surge_d3")
        crash = _episode_balance_weights(bundle, "crash_d3")
        np.testing.assert_allclose(surge[:2], [0.5, 0.5])
        np.testing.assert_allclose(crash[2:5], [1.0 / 3.0] * 3)
        self.assertEqual(float(crash[0]), 1.0)
        self.assertEqual(float(surge[2]), 1.0)

    def test_calibrator_and_wilson(self) -> None:
        rng = np.random.default_rng(12)
        score = rng.uniform(0, 1, 1000)
        target = (score + rng.normal(scale=0.20, size=1000) > 0.72).astype(np.int8)
        dates = pd.date_range("2020-01-01", periods=1000, freq="D").to_numpy()
        spec, audit = fit_best_calibrator(target, score, dates, minimum_rows=200)
        self.assertIn(spec.kind, {"identity", "platt", "beta", "isotonic"})
        self.assertFalse(audit.empty)
        self.assertGreater(wilson_lower_bound(80, 100, 0.90), 0.70)

    def test_frozen_recipe_mean_method(self) -> None:
        frame = pd.DataFrame(
            {
                "s1": [0.2, 0.8, 0.6],
                "s2": [0.4, 0.6, 0.9],
                "date": pd.to_datetime(["2026-01-01"] * 3),
            }
        )
        # Build the same minimal feature spec used by build_meta_feature_frame.
        from surge_precision_common_v6 import build_meta_feature_frame

        meta, feature_spec = build_meta_feature_frame(
            frame,
            ["s1", "s2"],
            {"s1": "a", "s2": "b"},
            date_column="date",
            scope_columns=[],
        )
        freeze = {
            "method_spec": {
                "model_kind": "recipe_mean",
                "feature_spec": feature_spec,
                "model_parameters": {},
                "model": None,
                "calibrator": CalibratorSpec("identity", {}, "unit_interval").to_dict(),
            }
        }
        probability = apply_frozen_method(
            freeze,
            frame,
            frame["date"].to_numpy(dtype="datetime64[ns]"),
            Path("."),
        )
        np.testing.assert_allclose(probability, meta["ensemble_mean"].to_numpy(), atol=1e-7)

    def test_controller_maturity_and_conservative_threshold(self) -> None:
        dates = pd.Series(pd.date_range("2026-07-01", periods=10, freq="B"))
        self.assertEqual(mature_cutoff(dates, 3), pd.Timestamp(dates.iloc[-4]))
        base = PrecisionPolicy(kind="global_threshold", target_precision=0.70, threshold=0.80)
        lower_candidate = PrecisionPolicy(kind="global_threshold", target_precision=0.73, threshold=0.55)
        conservative = clamp_policy_to_base(base, lower_candidate, allow_decrease=False)
        permissive = clamp_policy_to_base(base, lower_candidate, allow_decrease=True)
        self.assertEqual(conservative.threshold, 0.80)
        self.assertEqual(permissive.threshold, 0.55)

    def test_population_stability_index(self) -> None:
        reference = {
            "psi_inner_edges": [0.2, 0.4, 0.6, 0.8],
            "psi_reference_proportions": [0.2] * 5,
        }
        stable = np.linspace(0.01, 0.99, 1000)
        shifted = np.linspace(0.75, 0.99, 1000)
        self.assertLess(population_stability_index(reference, stable), 0.05)
        self.assertGreater(population_stability_index(reference, shifted), 0.5)


class SyntheticIntegrationTest(unittest.TestCase):
    def test_synthetic_smoke_creates_freeze_without_alert_cap(self) -> None:
        script = Path(__file__).resolve().parent / "run_surge_precision70_v6.py"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            command = [
                sys.executable,
                str(script),
                "--synthetic-smoke",
                "--synthetic-date-count",
                "120",
                "--synthetic-ticker-count",
                "10",
                "--output",
                str(output),
                "--methods",
                "calibrated_recipe_mean",
                "--policy-kinds",
                "global_threshold",
                "--target-precision",
                "0.70",
                "--selection-precision-buffer",
                "0.00",
                "--minimum-precision-lcb",
                "0.30",
                "--minimum-alerts-per-fold",
                "5",
                "--minimum-alert-days-per-fold",
                "2",
                "--minimum-calibration-rows",
                "40",
                "--minimum-meta-train-rows",
                "60",
                "--meta-oof-initial-days",
                "20",
                "--meta-oof-block-days",
                "10",
                "--maximum-threshold-candidates",
                "20",
                "--bootstrap-samples",
                "10",
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                [str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")]
            ).strip(os.pathsep)
            completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
            if completed.returncode != 0:
                self.fail(f"synthetic smoke failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
            freeze = json.loads((output / "PRECISION_FREEZE_V6.json").read_text(encoding="utf-8"))
            self.assertEqual(freeze["precision_policy"]["kind"], "global_threshold")
            self.assertNotIn("daily_fraction", freeze["precision_policy"])
            self.assertTrue((output / "precision_candidate_metrics_by_role.csv").exists())
            verifier = Path(__file__).resolve().parent / "verify_surge_precision70_v6.py"
            verified = subprocess.run(
                [sys.executable, str(verifier), "--output", str(output), "--allow-gate-failed"],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if verified.returncode != 0:
                self.fail(f"verifier failed\nSTDOUT:\n{verified.stdout}\nSTDERR:\n{verified.stderr}")


class FrozenScorerIntegrationTest(unittest.TestCase):
    def test_frozen_scorer_runs_without_daily_cap(self) -> None:
        from surge_precision_common_v6 import build_meta_feature_frame

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v4 = root / "v4"
            v5 = root / "v5"
            v6 = root / "v6"
            for path in [v4, v5, v6]:
                path.mkdir(parents=True)
            dataset = root / "new.csv"
            dates = pd.date_range("2026-07-01", periods=5, freq="B")
            frame = pd.DataFrame(
                {
                    "source_row_id": np.arange(20),
                    "date": np.repeat(dates, 4),
                    "ticker": [f"T{index % 4}" for index in range(20)],
                    "market": ["KOSPI", "KOSDAQ", "KOSPI", "KOSDAQ"] * 5,
                    "industry_name": ["A"] * 20,
                    "bucket": ["B"] * 20,
                    "dummy": np.linspace(0.0, 1.0, 20),
                }
            )
            frame.to_csv(dataset, index=False)

            def constant_model(probability: float) -> dict[str, object]:
                return {"format": "constant_probability", "probability": probability}

            v4_registry = with_payload_checksum(
                {
                    "schema": "crashwatch_surge_production_model_registry_v4",
                    "recipes": ["v4a"],
                    "models": [
                        {"recipe": "v4a", "seed": 17, "features": ["dummy"], "model_type": "single", "model": constant_model(0.75)},
                        {"recipe": "v4a", "seed": 29, "features": ["dummy"], "model_type": "single", "model": constant_model(0.85)},
                    ],
                }
            )
            (v4 / "PRODUCTION_MODEL_REGISTRY.json").write_text(json.dumps(v4_registry), encoding="utf-8")
            v5_registry = with_payload_checksum(
                {
                    "schema": "crashwatch_surge_ranker_production_registry_v5",
                    "recipes": ["v5a"],
                    "models": [
                        {"recipe": "v5a", "seed": 17, "features": ["dummy"], "model": constant_model(0.2)},
                        {"recipe": "v5a", "seed": 29, "features": ["dummy"], "model": constant_model(0.9)},
                    ],
                }
            )
            (v5 / "V5_RANKER_PRODUCTION_REGISTRY.json").write_text(json.dumps(v5_registry), encoding="utf-8")
            v6_registry = with_payload_checksum(
                {
                    "schema": "crashwatch_surge_precision_specialist_registry_v6",
                    "recipes": ["v6a"],
                    "models": [
                        {"recipe": "v6a", "seed": 17, "features": ["dummy"], "seed_aggregation": "probability_mean", "model": constant_model(0.78)},
                        {"recipe": "v6a", "seed": 29, "features": ["dummy"], "seed_aggregation": "probability_mean", "model": constant_model(0.82)},
                    ],
                }
            )
            (v6 / "V6_SPECIALIST_PRODUCTION_REGISTRY.json").write_text(json.dumps(v6_registry), encoding="utf-8")

            signals = [
                "v4a", "v4a__seed_agreement",
                "v5a", "v5a__seed_agreement",
                "v6a", "v6a__seed_agreement",
            ]
            signal_frame = pd.DataFrame({signal: np.full(20, 0.8) for signal in signals})
            signal_frame["date"] = frame["date"]
            signal_frame["market"] = frame["market"]
            _, feature_spec = build_meta_feature_frame(
                signal_frame,
                signals,
                {signal: signal.split("a")[0] for signal in signals},
                date_column="date",
                scope_columns=["market"],
            )
            freeze = with_payload_checksum(
                {
                    "schema": "crashwatch_surge_precision_freeze_v6",
                    "selected_method": "calibrated_recipe_mean",
                    "selected_policy_kind": "global_threshold",
                    "precision_policy": PrecisionPolicy(
                        kind="global_threshold", target_precision=0.70, threshold=0.50, gate_pass=True
                    ).to_dict(),
                    "method_spec": {
                        "method": "calibrated_recipe_mean",
                        "model_kind": "recipe_mean",
                        "feature_spec": feature_spec,
                        "model_parameters": {},
                        "model": None,
                        "calibrator": CalibratorSpec("identity", {}, "unit_interval").to_dict(),
                    },
                    "signals": signals,
                    "family_map": {signal: signal.split("a")[0] for signal in signals},
                    "signal_calibrators": {
                        signal: CalibratorSpec("identity", {}, "unit_interval").to_dict() for signal in signals
                    },
                    "score_reference": {},
                    "target_precision": 0.70,
                    "selection_target_precision": 0.73,
                    "no_alert_count_limit": True,
                }
            )
            (v6 / "PRECISION_FREEZE_V6.json").write_text(json.dumps(freeze), encoding="utf-8")
            recommendation = with_payload_checksum(
                {
                    "schema": "crashwatch_surge_precision_recommendation_v6",
                    "status": "READY_FOR_NEW_FUTURE_HOLDOUT",
                    "target_precision": 0.70,
                    "accuracy_definition": "precision = TP / (TP + FP)",
                }
            )
            (v6 / "FINAL_RECOMMENDATION_V6.json").write_text(json.dumps(recommendation), encoding="utf-8")
            output = root / "scored"
            scorer = Path(__file__).resolve().parent / "score_surge_precision70_v6.py"
            command = [
                sys.executable, str(scorer),
                "--model-dir", str(v6),
                "--v4-model-dir", str(v4),
                "--v5-model-dir", str(v5),
                "--dataset", str(dataset),
                "--output", str(output),
            ]
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                [str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")]
            ).strip(os.pathsep)
            completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=30)
            if completed.returncode != 0:
                self.fail(f"scorer failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
            scored = pd.read_csv(output / "surge_precision70_scores.csv")
            self.assertEqual(len(scored), 20)
            self.assertEqual(int(scored["surge_alert_v6"].sum()), 20)
            self.assertTrue((output / "SCORING_AUDIT_V6.json").exists())

            controller = with_payload_checksum(
                {
                    "schema": "crashwatch_surge_precision70_controller_state_v6",
                    "status": "ACTIVE",
                    "freeze_sha256": sha256_file(v6 / "PRECISION_FREEZE_V6.json"),
                    "active_policy": PrecisionPolicy(
                        kind="global_threshold",
                        target_precision=0.70,
                        threshold=0.999999,
                        gate_pass=True,
                    ).to_dict(),
                }
            )
            controller_path = v6 / "PRECISION_CONTROLLER_STATE_V6.json"
            controller_path.write_text(json.dumps(controller), encoding="utf-8")
            controlled_output = root / "scored_controlled"
            controlled_command = command[:-2] + [
                "--output", str(controlled_output),
                "--controller-state", str(controller_path),
            ]
            controlled = subprocess.run(
                controlled_command, env=env, capture_output=True, text=True, timeout=30
            )
            if controlled.returncode != 0:
                self.fail(
                    f"controlled scorer failed\nSTDOUT:\n{controlled.stdout}\nSTDERR:\n{controlled.stderr}"
                )
            controlled_scores = pd.read_csv(controlled_output / "surge_precision70_scores.csv")
            self.assertEqual(int(controlled_scores["surge_alert_v6"].sum()), 0)
            self.assertTrue(controlled_scores["controller_state_used"].notna().all())


class ControllerIntegrationTest(unittest.TestCase):
    def test_controller_uses_only_mature_oos_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "model"
            model_dir.mkdir()
            freeze = with_payload_checksum(
                {
                    "schema": "crashwatch_surge_precision_freeze_v6",
                    "precision_policy": PrecisionPolicy(
                        kind="global_threshold", target_precision=0.70, threshold=0.60, gate_pass=True
                    ).to_dict(),
                    "training_data_end": "2025-12-31T00:00:00",
                }
            )
            (model_dir / "PRECISION_FREEZE_V6.json").write_text(
                json.dumps(freeze), encoding="utf-8"
            )
            dates = pd.date_range("2026-01-02", periods=40, freq="B")
            records = []
            row_id = 0
            for date_index, date in enumerate(dates):
                for ticker_index in range(10):
                    positive = ticker_index < 4
                    score = 0.95 - 0.01 * ticker_index if positive else 0.35 - 0.01 * ticker_index
                    records.append(
                        {
                            "source_row_id": row_id,
                            "date": date,
                            "ticker": f"T{ticker_index:02d}",
                            "market": "KOSPI",
                            "surge_probability_3d5_v6": score,
                            "target_valid": date_index < len(dates) - 3,
                            "label_abs_surge_3d_5pct": int(positive),
                        }
                    )
                    row_id += 1
            frame = pd.DataFrame(records)
            scores_path = root / "scores.csv"
            target_path = root / "target.csv"
            frame[[
                "source_row_id", "date", "ticker", "market", "surge_probability_3d5_v6"
            ]].to_csv(scores_path, index=False)
            frame[[
                "source_row_id", "date", "ticker", "target_valid", "label_abs_surge_3d_5pct"
            ]].to_csv(target_path, index=False)
            controller_script = Path(__file__).resolve().parent / "update_surge_precision_controller_v6.py"
            state_path = root / "controller.json"
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                [str(Path(__file__).resolve().parent), env.get("PYTHONPATH", "")]
            ).strip(os.pathsep)
            completed = subprocess.run(
                [
                    sys.executable, str(controller_script),
                    "--model-dir", str(model_dir),
                    "--scores", str(scores_path),
                    "--target-sidecar", str(target_path),
                    "--output", str(state_path),
                    "--minimum-alerts", "20",
                    "--minimum-alert-days", "5",
                    "--minimum-precision-lcb", "0.50",
                    "--precision-buffer", "0.00",
                    "--maximum-threshold-candidates", "100",
                ],
                env=env, capture_output=True, text=True, timeout=30,
            )
            if completed.returncode != 0:
                self.fail(
                    f"controller failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
                )
            state = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "ACTIVE")
            self.assertEqual(state["matured_through"], pd.Timestamp(dates[-4]).isoformat())
            self.assertGreaterEqual(state["active_policy_metrics"]["precision"], 0.70)
            self.assertLessEqual(pd.Timestamp(state["history_end"]), pd.Timestamp(dates[-4]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
