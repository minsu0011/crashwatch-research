from __future__ import annotations

import json
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

from surge_alert_budget_common_v5 import (  # noqa: E402
    DailyBudgetPolicy,
    apply_allocation_biases,
    apply_daily_budget_policy,
    daily_budget_selection,
    event_episode_metrics,
    select_minimax_daily_budget_policy,
)
from surge_alert_budget_test_data_v5 import build_synthetic_package  # noqa: E402
from surge_model_zoo_common import sha256_file  # noqa: E402


class DailyBudgetUnitTests(unittest.TestCase):
    def test_fraction_is_a_hard_upper_bound(self) -> None:
        dates = np.repeat(pd.Timestamp("2026-08-10"), 48)
        scores = np.linspace(1.0, 0.0, 48)
        selected = daily_budget_selection(scores, dates, 0.40, max_alerts_per_day=20)
        # floor(48 * 0.40) = 19; ceil would violate the 40% hard cap.
        self.assertEqual(int(selected.sum()), 19)
        self.assertLessEqual(float(selected.mean()), 0.40)

    def test_absolute_daily_count_cap(self) -> None:
        dates = np.repeat(pd.date_range("2026-01-01", periods=3), 50)
        scores = np.arange(150, dtype=np.float64)
        selected = daily_budget_selection(scores, dates, 0.80, max_alerts_per_day=7)
        counts = pd.Series(selected).groupby(pd.Series(dates)).sum()
        self.assertTrue(counts.eq(7).all())

    def test_tie_break_is_deterministic(self) -> None:
        dates = np.repeat(pd.Timestamp("2026-08-10"), 10)
        scores = np.ones(10, dtype=np.float64)
        first = daily_budget_selection(scores, dates, 0.30)
        second = daily_budget_selection(scores, dates, 0.30)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.flatnonzero(first), np.asarray([0, 1, 2]))

    def test_market_bias_changes_only_the_operational_order(self) -> None:
        scores = np.asarray([0.90, 0.80, 0.70, 0.60])
        groups = np.asarray(["KOSPI", "KOSPI", "KOSDAQ", "KOSDAQ"], dtype=object)
        adjusted = apply_allocation_biases(scores, groups, {"KOSDAQ": 0.40})
        selected = daily_budget_selection(
            adjusted,
            np.repeat(pd.Timestamp("2026-08-10"), 4),
            0.25,
        )
        self.assertEqual(np.flatnonzero(selected).tolist(), [2])
        np.testing.assert_allclose(scores, np.asarray([0.90, 0.80, 0.70, 0.60]))

    def test_minimax_selects_smallest_feasible_budget(self) -> None:
        rows: list[dict[str, object]] = []
        for fold_id in (0, 1):
            for day in pd.date_range("2026-01-01", periods=5):
                for rank in range(10):
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "date": day + pd.Timedelta(days=fold_id * 10),
                            "ticker": f"T{rank:02d}",
                            "target": 1 if rank < 2 else 0,
                            "score": 10.0 - rank,
                        }
                    )
        frame = pd.DataFrame(rows)
        policy, summary, _ = select_minimax_daily_budget_policy(
            frame,
            [0.10, 0.20, 0.30],
            target_recall=0.70,
            selection_recall_buffer=0.0,
            minimum_precision_lift=1.0,
            maximum_alert_rate=0.30,
            required_fold_pass_rate=1.0,
            max_alerts_per_day=3,
        )
        self.assertTrue(policy.gate_pass)
        self.assertAlmostEqual(policy.daily_fraction, 0.20)
        self.assertTrue(bool(summary.loc[summary["fraction"].eq(0.20), "gate_pass"].iloc[0]))

    def test_impossible_gate_never_relaxes_the_cap(self) -> None:
        rows: list[dict[str, object]] = []
        for fold_id in (0, 1):
            for day in pd.date_range("2026-01-01", periods=5):
                for rank in range(10):
                    rows.append(
                        {
                            "fold_id": fold_id,
                            "date": day + pd.Timedelta(days=fold_id * 10),
                            "ticker": f"T{rank:02d}",
                            "target": 1 if rank < 8 else 0,
                            "score": 10.0 - rank,
                        }
                    )
        policy, _, _ = select_minimax_daily_budget_policy(
            pd.DataFrame(rows),
            [0.10, 0.20, 0.30],
            target_recall=0.70,
            selection_recall_buffer=0.0,
            minimum_precision_lift=1.0,
            maximum_alert_rate=0.30,
            required_fold_pass_rate=1.0,
            max_alerts_per_day=3,
        )
        self.assertFalse(policy.gate_pass)
        self.assertLessEqual(policy.daily_fraction, 0.30)
        self.assertLessEqual(float(policy.achieved_mean_alert_rate or 0.0), 0.30 + 1e-12)

    def test_event_episode_recall(self) -> None:
        target = np.asarray([0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.uint8)
        alert = np.asarray([0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
        dates = pd.date_range("2026-01-01", periods=len(target))
        ticker = np.repeat("A", len(target))
        metrics = event_episode_metrics(target, alert, dates, ticker)
        self.assertEqual(metrics["event_count"], 3)
        self.assertEqual(metrics["captured_events"], 1)
        self.assertAlmostEqual(float(metrics["event_recall"]), 1.0 / 3.0)

    def test_policy_round_trip(self) -> None:
        original = DailyBudgetPolicy(
            daily_fraction=0.35,
            max_alerts_per_day=17,
            allocation_column="market",
            allocation_biases={"KOSDAQ": -0.02},
            gate_pass=True,
        )
        restored = DailyBudgetPolicy.from_dict(original.to_dict())
        self.assertEqual(restored, original)
        scores = np.arange(48, dtype=np.float64)
        selected = apply_daily_budget_policy(
            restored,
            scores,
            np.repeat(pd.Timestamp("2026-08-10"), 48),
            groups=np.repeat("KOSPI", 48),
        )
        self.assertEqual(int(selected.sum()), 16)  # floor(48 * .35)


class EndToEndTests(unittest.TestCase):
    def run_checked(self, command: list[str], timeout: int = 300) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        if result.returncode != 0:
            self.fail(
                "command failed\n"
                + " ".join(command)
                + f"\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
        return result

    def test_v4_to_v5_end_to_end_resume_score_and_tamper_repair(self) -> None:
        with tempfile.TemporaryDirectory(prefix="surge_alert_budget_v5_") as temporary:
            root = Path(temporary)
            paths = build_synthetic_package(root, date_count=300, ticker_count=6)
            v4_output = root / "outputs" / "surge_model_zoo_v4"
            v4_command = [
                sys.executable,
                str(HERE / "run_surge_model_zoo_v4.py"),
                "--package-root", str(root),
                "--dataset", str(paths["dataset"]),
                "--target-sidecar", str(paths["target"]),
                "--folds", str(paths["folds"]),
                "--profiles", str(paths["profiles"]),
                "--recipes", str(paths["recipes"]),
                "--output", str(v4_output),
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
                "--daily-fraction-grid", "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                "--top-fractions", "0.03,0.10,0.30,0.50",
                "--inner-windows", "1",
                "--inner-validation-days", "20",
                "--inner-purge-days", "5",
                "--inner-step-days", "20",
                "--minimum-inner-train-days", "60",
                "--minimum-purge-trading-days", "3",
                "--max-tuning-rounds", "35",
                "--early-stopping-rounds", "6",
                "--minimum-iterations", "5",
                "--maximum-iterations", "30",
                "--minimum-calibration-rows", "40",
                "--minimum-scope-rows", "10",
                "--minimum-scope-positives", "1",
            ]
            self.run_checked(v4_command, timeout=300)

            v5_output = root / "outputs" / "surge_alert_budget_v5"
            v5_command = [
                sys.executable,
                str(HERE / "run_surge_alert_budget_v5.py"),
                "--package-root", str(root),
                "--v4-output", str(v4_output),
                "--output", str(v5_output),
                "--dataset", str(paths["dataset"]),
                "--target-sidecar", str(paths["target"]),
                "--folds", str(paths["folds"]),
                "--profiles", str(paths["profiles"]),
                "--quick",
                "--device", "cpu",
                "--allow-cpu-fallback",
                "--selection-folds", "0,1",
                "--confirmation-folds", "2",
                "--recent-folds", "3",
                "--rank-seeds", "17",
                "--methods", "equal_recipe_rank",
                "--target-recall", "0.60",
                "--selection-recall-buffer", "0.0",
                "--minimum-precision-lift", "0.9",
                "--max-alert-rate", "0.50",
                "--max-alerts-per-day", "3",
                "--daily-fraction-grid", "0.20,0.30,0.40,0.50",
                "--diagnostic-fractions", "0.10,0.20,0.30,0.40,0.50",
                "--required-selection-fold-pass-rate", "1.0",
                "--threads-per-model", "2",
                "--xgboost-threads", "2",
                "--family-parallel",
                "--inner-windows", "1",
                "--inner-validation-days", "20",
                "--inner-purge-days", "5",
                "--inner-step-days", "20",
                "--minimum-inner-train-days", "40",
                "--minimum-purge-trading-days", "3",
                "--max-tuning-rounds", "50",
                "--early-stopping-rounds", "8",
                "--minimum-iterations", "5",
                "--maximum-iterations", "40",
            ]
            self.run_checked(v5_command, timeout=300)

            status = json.loads((v5_output / "RUN_STATUS.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "SUCCESS")
            self.assertEqual(status["schema"], "crashwatch_surge_alert_budget_runner_v5")
            metrics = pd.read_csv(v5_output / "budget_candidate_metrics_by_fold.csv")
            self.assertTrue((metrics["alert_rate"] <= 0.50 + 1e-12).all())
            self.assertTrue((metrics["alerts_per_day_max"] <= 3 + 1e-12).all())

            verify_command = [
                sys.executable,
                str(HERE / "verify_surge_alert_budget_v5.py"),
                "--output", str(v5_output),
            ]
            self.run_checked(verify_command, timeout=180)

            score_output = root / "outputs" / "scored_v5"
            score_command = [
                sys.executable,
                str(HERE / "score_surge_alert_budget_v5.py"),
                "--model-dir", str(v5_output),
                "--v4-model-dir", str(v4_output),
                "--dataset", str(paths["dataset"]),
                "--target-sidecar", str(paths["target"]),
                "--output", str(score_output),
                "--allow-gate-failed",
            ]
            self.run_checked(score_command, timeout=180)
            scored = pd.read_csv(score_output / "surge_alert_budget_scores.csv")
            self.assertIn("surge_raw_ensemble_score", scored.columns)
            per_day = scored.groupby("date")["surge_alert"].sum()
            self.assertTrue((per_day <= 3).all())
            self.assertLessEqual(float(scored["surge_alert"].mean()), 0.50 + 1e-12)

            task_jsons = sorted(v5_output.glob("rank_task_cache/*/seed_*/fold_*.json"))
            before = {path: sha256_file(path) for path in task_jsons}
            self.run_checked(v5_command + ["--resume"], timeout=240)
            after = {path: sha256_file(path) for path in task_jsons}
            self.assertEqual(before, after)

            task_payload = json.loads(task_jsons[0].read_text(encoding="utf-8"))
            prediction_path = Path(str(task_payload["prediction_path"]))
            if not prediction_path.is_absolute():
                prediction_path = v5_output / prediction_path
            with prediction_path.open("ab") as handle:
                handle.write(b"V5_TAMPER")
            broken = subprocess.run(verify_command, capture_output=True, text=True, timeout=180)
            self.assertNotEqual(broken.returncode, 0)

            self.run_checked(v5_command + ["--resume"], timeout=300)
            self.run_checked(verify_command, timeout=180)


if __name__ == "__main__":
    unittest.main()
