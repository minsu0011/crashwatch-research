from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


MODULE_PATH = Path(__file__).with_name("run_surge_correlation_map.py")
SPEC = importlib.util.spec_from_file_location("surge_corr", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
surge_corr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = surge_corr
SPEC.loader.exec_module(surge_corr)


class CorrelationMathTests(unittest.TestCase):
    def test_pairwise_complete_corr_matches_pandas(self) -> None:
        values = np.array(
            [
                [1.0, 2.0, np.nan],
                [2.0, 4.0, 3.0],
                [3.0, 6.0, 2.0],
                [4.0, np.nan, 1.0],
                [5.0, 10.0, 0.0],
            ],
            dtype=float,
        )
        expected = pd.DataFrame(values).corr(min_periods=2).to_numpy()
        actual = surge_corr.pairwise_complete_corr(values, min_periods=2)
        np.testing.assert_allclose(actual, expected, atol=1e-6, equal_nan=True)

    def test_target_correlation_direction(self) -> None:
        target = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        values = np.column_stack(
            [
                np.array([0, 1, 2, 10, 11, 12], dtype=float),
                np.array([12, 11, 10, 2, 1, 0], dtype=float),
            ]
        )
        corr, counts, positives = surge_corr.vector_target_corr(values, target, min_periods=2)
        self.assertGreater(corr[0], 0.8)
        self.assertLess(corr[1], -0.8)
        np.testing.assert_array_equal(counts, np.array([6, 6]))
        np.testing.assert_array_equal(positives, np.array([3, 3]))

    def test_within_date_residualization_removes_date_level_constant(self) -> None:
        dates = np.repeat(np.arange(4), 3)
        market = np.repeat(np.array([1.0, 2.0, 3.0, 4.0]), 3)
        cross_section = np.tile(np.array([-1.0, 0.0, 1.0]), 4)
        residual_market = surge_corr.residualize_by_group(market[:, None], dates).ravel()
        residual_cross = surge_corr.residualize_by_group(cross_section[:, None], dates).ravel()
        self.assertTrue(np.allclose(residual_market, 0.0))
        self.assertGreater(np.std(residual_cross), 0.0)

    def test_connected_components(self) -> None:
        matrix = np.eye(5, dtype=float)
        matrix[0, 1] = matrix[1, 0] = 0.95
        matrix[1, 2] = matrix[2, 1] = 0.93
        matrix[3, 4] = matrix[4, 3] = 0.91
        components = surge_corr.connected_components(matrix, threshold=0.92)
        self.assertEqual(components, [[0, 1, 2], [3], [4]])

    def test_average_linkage_does_not_single_link_a_chain(self) -> None:
        matrix = np.eye(3, dtype=float)
        matrix[0, 1] = matrix[1, 0] = 0.95
        matrix[1, 2] = matrix[2, 1] = 0.93
        matrix[0, 2] = matrix[2, 0] = 0.70
        graph_components = surge_corr.connected_components(matrix, threshold=0.92)
        average_components = surge_corr.average_linkage_components(matrix, threshold=0.92)
        self.assertEqual(graph_components, [[0, 1, 2]])
        self.assertEqual(average_components, [[0, 1], [2]])

    def test_recompute_surge_target_exact_boundary_and_tail_invalid(self) -> None:
        dates = pd.bdate_range("2026-01-02", periods=6)
        exact_second_day_return = 1.05 / 1.02 - 1.0
        source = pd.DataFrame(
            {
                "source_row_id": np.arange(6, dtype=np.int64),
                "date": dates,
                "ticker": ["000001"] * 6,
                "t_price_ret_1": [0.0, 0.02, exact_second_day_return, 0.0, 0.0, 0.0],
            }
        )
        label, valid, first_hit, best = surge_corr.recompute_surge_target_from_returns(source)
        np.testing.assert_array_equal(valid, np.array([True, True, True, False, False, False]))
        np.testing.assert_array_equal(label, np.array([1, 0, 0, 0, 0, 0], dtype=np.uint8))
        np.testing.assert_array_equal(first_hit, np.array([2, 0, 0, 0, 0, 0], dtype=np.int8))
        self.assertAlmostEqual(best[0], 0.05, places=12)
        self.assertTrue(np.isnan(best[3:]).all())

    def test_fold_validation_rejects_purge_shorter_than_horizon(self) -> None:
        dates = pd.Series(pd.bdate_range("2026-01-02", periods=10))
        target_valid = np.ones(10, dtype=bool)
        target = np.array([0, 1] * 5, dtype=np.int8)
        folds = [
            surge_corr.FoldDefinition(
                fold_id=0,
                train_start=str(dates.iloc[0].date()),
                train_end=str(dates.iloc[2].date()),
                validation_start=str(dates.iloc[5].date()),
                validation_end=str(dates.iloc[8].date()),
            )
        ]
        with self.assertRaises(ValueError):
            surge_corr.validate_fold_definitions(
                dates,
                target_valid,
                target,
                folds,
                {"selection": [0], "confirmation": [], "recent_audit": []},
                minimum_purge_days=3,
            )

    def test_leakage_name_filter_does_not_block_point_in_time_lead_lag(self) -> None:
        self.assertIsNone(surge_corr.leakage_reason("t_peer_lead_lag_1"))
        self.assertIsNotNone(surge_corr.leakage_reason("best_forward_return_3d"))
        self.assertIsNotNone(surge_corr.leakage_reason("label_abs_crash_5"))


class EndToEndCsvSmokeTest(unittest.TestCase):
    def test_csv_pipeline_creates_required_outputs(self) -> None:
        rng = np.random.default_rng(20260809)
        dates = pd.bdate_range("2020-01-02", periods=180)
        tickers = ["000001", "000002", "000003", "000004"]
        rows = []
        labels = []
        for date_index, date in enumerate(dates):
            market = rng.normal()
            for ticker_index, ticker in enumerate(tickers):
                signal = rng.normal() + 0.25 * market + 0.1 * ticker_index
                probability = 1.0 / (1.0 + np.exp(-(1.1 * signal - 0.3)))
                label = int(rng.random() < probability)
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "sealed_do_not_train_or_tune": 0,
                        "t_signal": signal,
                        "t_inverse": -signal + rng.normal(scale=0.15),
                        "t_duplicate": signal + rng.normal(scale=1e-5),
                        "t_noise": rng.normal(),
                        "u_market": market,
                        "t_sparse": rng.normal() if rng.random() > 0.75 else np.nan,
                    }
                )
                labels.append(label)
        source = pd.DataFrame(rows)
        target_valid = np.ones(len(source), dtype=bool)
        for ticker in tickers:
            ticker_indices = source.index[source["ticker"].eq(ticker)].to_numpy()
            target_valid[ticker_indices[-3:]] = False
        target = pd.DataFrame(
            {
                "source_row_id": np.arange(len(source), dtype=np.int64),
                "date": source["date"],
                "ticker": source["ticker"],
                "label_abs_surge_3d_5pct": np.asarray(labels, dtype=np.int8),
                "target_valid": target_valid,
            }
        )
        folds = [
            {
                "fold_id": 0,
                "train_start": str(dates[0].date()),
                "train_end": str(dates[69].date()),
                "purge_start": str(dates[70].date()),
                "purge_end": str(dates[72].date()),
                "validation_start": str(dates[73].date()),
                "validation_end": str(dates[92].date()),
            },
            {
                "fold_id": 1,
                "train_start": str(dates[0].date()),
                "train_end": str(dates[109].date()),
                "purge_start": str(dates[110].date()),
                "purge_end": str(dates[112].date()),
                "validation_start": str(dates[113].date()),
                "validation_end": str(dates[132].date()),
            },
            {
                "fold_id": 2,
                "train_start": str(dates[0].date()),
                "train_end": str(dates[149].date()),
                "purge_start": str(dates[150].date()),
                "purge_end": str(dates[152].date()),
                "validation_start": str(dates[153].date()),
                "validation_end": str(dates[172].date()),
            },
        ]
        features = ["t_signal", "t_inverse", "t_duplicate", "t_noise", "u_market", "t_sparse"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "source.csv"
            target_path = root / "target.csv"
            folds_path = root / "folds.json"
            features_path = root / "features.json"
            output_path = root / "output"
            source.to_csv(source_path, index=False)
            target.to_csv(target_path, index=False)
            folds_path.write_text(json.dumps(folds), encoding="utf-8")
            features_path.write_text(json.dumps(features), encoding="utf-8")
            command = [
                sys.executable,
                str(MODULE_PATH),
                "--package-root",
                str(root),
                "--input",
                str(source_path),
                "--target",
                str(target_path),
                "--folds",
                str(folds_path),
                "--feature-list",
                str(features_path),
                "--output",
                str(output_path),
                "--reuse-legacy",
                "never",
                "--selection-folds",
                "0",
                "--confirmation-folds",
                "1",
                "--recent-folds",
                "2",
                "--feature-structure-sample-rows",
                "500",
                "--spearman-sample-rows",
                "300",
                "--cluster-basis-rows",
                "300",
                "--fold-correlation-sample-rows",
                "250",
                "--target-train-sample-rows",
                "250",
                "--min-periods",
                "20",
                "--target-min-periods",
                "10",
                "--skip-mi",
                "--skip-plots",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
            if completed.returncode != 0:
                self.fail(f"pipeline failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
            required = [
                "DATASET_COMPATIBILITY.json",
                "SURGE_TARGET_AUDIT.json",
                "feature_quality_surge.csv",
                "ticker_target_quality.csv",
                "yearly_target_quality.csv",
                "walk_forward_folds.json",
                "leakage_source_column_audit.csv",
                "leakage_audit.csv",
                "surge_feature_target_correlation_by_fold.csv",
                "surge_feature_correlation_summary.csv",
                "primary_clusters.csv",
                "surge_representatives_by_fold.csv",
                "high_correlation_pairs.csv",
                "surge_vs_crash_feature_comparison.csv",
                "surge_correlation_manifest.json",
                "RUN_STATUS.json",
            ]
            for filename in required:
                self.assertTrue((output_path / filename).exists(), filename)
            summary = pd.read_csv(output_path / "surge_feature_correlation_summary.csv")
            signal = summary.set_index("feature").loc["t_signal"]
            inverse = summary.set_index("feature").loc["t_inverse"]
            self.assertGreater(signal["target_pearson"], 0.25)
            self.assertLess(inverse["target_pearson"], -0.25)
            manifest = json.loads((output_path / "surge_correlation_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["feature_count"], len(features))
            run_status = json.loads((output_path / "RUN_STATUS.json").read_text(encoding="utf-8"))
            self.assertEqual(run_status["status"], "SUCCESS")
            target_audit = json.loads((output_path / "SURGE_TARGET_AUDIT.json").read_text(encoding="utf-8"))
            self.assertEqual(target_audit["independent_revalidation"]["status"], "skipped")
            fold_audit = json.loads((output_path / "walk_forward_folds.json").read_text(encoding="utf-8"))
            self.assertEqual(fold_audit[0]["validation_rows_target_valid"], 80)

            resume_command = command + ["--resume"]
            resumed = subprocess.run(resume_command, capture_output=True, text=True, timeout=120)
            if resumed.returncode != 0:
                self.fail(f"resume failed\nSTDOUT:\n{resumed.stdout}\nSTDERR:\n{resumed.stderr}")
            self.assertIn("재사용합니다", resumed.stdout)

            summary_path = output_path / "surge_feature_correlation_summary.csv"
            original_summary_hash = surge_corr.sha256_file(summary_path)
            with summary_path.open("a", encoding="utf-8") as handle:
                handle.write("\nTAMPERED")
            repaired = subprocess.run(resume_command, capture_output=True, text=True, timeout=120)
            if repaired.returncode != 0:
                self.fail(f"repair resume failed\nSTDOUT:\n{repaired.stdout}\nSTDERR:\n{repaired.stderr}")
            self.assertIn("다시 계산합니다", repaired.stdout)
            self.assertEqual(surge_corr.sha256_file(summary_path), original_summary_hash)


if __name__ == "__main__":
    unittest.main(verbosity=2)
