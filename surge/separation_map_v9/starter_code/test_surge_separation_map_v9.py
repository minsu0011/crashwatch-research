from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


from run_surge_separation_map_v9 import (
    compute_univariate_map,
    load_external_base_oof,
    summarize_univariate_map,
)

from surge_separation_common_v9 import (
    ERROR_A_TOP_TP,
    ERROR_B_TOP_FP,
    ERROR_C_LOW_TP,
    ERROR_D_LOW_TN,
    build_error_group_codes,
    build_horizon_pairs,
    build_matched_pairs,
    cluster_innovation_values,
    compute_interaction_moment_screen,
    datewise_percentile_rank,
    groupwise_percentile_rank,
    horizon_rank_spread,
    precision_curve_metrics,
    separation_metrics,
    verify_output_inventory,
    wilson_lower_bound,
)


class SeparationCommonTests(unittest.TestCase):
    def test_datewise_rank(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        dates = np.array(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"])
        ranked = datewise_percentile_rank(values, dates)
        np.testing.assert_allclose(ranked, [0.5, 1.0, 0.5, 1.0])

    def test_groupwise_rank(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0])
        dates = np.array(["d1", "d1", "d1", "d1"])
        market = np.array(["A", "A", "B", "B"])
        ranked = groupwise_percentile_rank(values, [dates, market])
        np.testing.assert_allclose(ranked, [0.5, 1.0, 0.5, 1.0])

    def test_error_groups(self) -> None:
        target = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
        scores = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.3])
        dates = np.array(["2024-01-01"] * 6)
        codes, ranks = build_error_group_codes(target, scores, dates, top_quantile=0.7, low_quantile=0.5)
        self.assertIn(ERROR_A_TOP_TP, codes)
        self.assertIn(ERROR_B_TOP_FP, codes)
        self.assertIn(ERROR_C_LOW_TP, codes)
        self.assertIn(ERROR_D_LOW_TN, codes)
        self.assertTrue(np.isfinite(ranks).all())

    def test_matched_pairs_same_date(self) -> None:
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01"] * 4),
                "base_rank": [0.95, 0.93, 0.90, 0.88],
                "market": ["KOSPI", "KOSPI", "KOSDAQ", "KOSPI"],
                "bucket": ["a", "a", "a", "b"],
                "industry_name": ["i", "i", "j", "i"],
                "error_code": [ERROR_A_TOP_TP, ERROR_B_TOP_FP, ERROR_B_TOP_FP, ERROR_B_TOP_FP],
            }
        )
        pairs = build_matched_pairs(frame, 0, "AB", ERROR_A_TOP_TP, ERROR_B_TOP_FP, controls_per_case=2)
        self.assertEqual(len(pairs), 2)
        self.assertTrue(bool(pairs.iloc[0]["same_date"]))
        self.assertTrue(bool(pairs.iloc[0]["same_market"]))

    def test_separation_metrics_coverage(self) -> None:
        values = np.array([10.0, 9.0, np.nan, 1.0, 2.0, 3.0])
        codes = np.array([1, 1, 1, 2, 2, 2], dtype=np.int8)
        metrics = separation_metrics(values, codes, 1, 2)
        self.assertEqual(metrics["positive_valid_n"], 2)
        self.assertEqual(metrics["negative_valid_n"], 3)
        self.assertGreater(metrics["oriented_auc"], 0.9)

    def test_cluster_innovation(self) -> None:
        dates = np.array(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"])
        feature = np.array([1.0, 3.0, 1.0, 3.0])
        peer = np.array([1.0, 2.0, 3.0, 4.0])
        innovation = cluster_innovation_values(feature, [peer], dates)
        self.assertTrue(np.isfinite(innovation).all())
        self.assertEqual(len(innovation), 4)

    def test_horizon_pairs_and_spread(self) -> None:
        pairs = build_horizon_pairs(["t_price_ret_5", "t_price_ret_20", "other"])
        self.assertEqual(len(pairs), 1)
        spread = horizon_rank_spread(
            np.array([1.0, 2.0]),
            np.array([2.0, 1.0]),
            np.array(["2024-01-01", "2024-01-01"]),
        )
        np.testing.assert_allclose(spread, [-0.5, 0.5])

    def test_interaction_screen(self) -> None:
        rng = np.random.default_rng(17)
        x = rng.normal(size=(120, 4))
        codes = np.array([1] * 50 + [2] * 70, dtype=np.int8)
        x[:50, 0] *= x[:50, 1]
        result = compute_interaction_moment_screen(x, ["a", "b", "c", "d"], codes, 1, 2)
        self.assertEqual(len(result), 6)
        self.assertIn("interaction_excess_score", result.columns)

    def test_precision_curve_minimum_alerts(self) -> None:
        target = np.array([1, 1, 0, 1, 0, 0], dtype=np.int8)
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        metrics = precision_curve_metrics(target, scores, minimum_alerts=3, target_precision=0.70)
        self.assertGreaterEqual(metrics["best_precision_alerts"], 3)
        self.assertGreaterEqual(metrics["precision_target_alerts"], 3)

    def test_wilson_small_sample(self) -> None:
        self.assertLess(wilson_lower_bound(3, 3), 0.7)
        self.assertGreater(wilson_lower_bound(90, 100), 0.8)


class RunnerLogicTests(unittest.TestCase):
    def test_v6_npz_base_oof_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "precision_candidate_predictions.npz"
            np.savez_compressed(
                path,
                validation_index=np.array([0, 1, 2], dtype=np.int64),
                fold_id=np.array([0, 0, 1], dtype=np.int16),
                score=np.array([0.1, 0.8, 0.4], dtype=np.float32),
            )
            frame = pd.DataFrame(
                {
                    "source_row_id": [10, 11, 12],
                    "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                }
            )
            loaded = load_external_base_oof(path, frame)
            self.assertEqual(loaded["row_index"].tolist(), [0, 1, 2])
            self.assertEqual(loaded["fold_id"].tolist(), [0, 0, 1])
            np.testing.assert_allclose(loaded["base_score_raw"], [0.1, 0.8, 0.4], rtol=1e-6)

    def test_sparse_missing_indicator_cannot_bypass_source_coverage(self) -> None:
        rows = 100
        frame = pd.DataFrame(
            {
                "f": [1.0] * 10 + [np.nan] * 90,
                "date": pd.to_datetime(["2024-01-01"] * rows),
                "market": ["KOSPI"] * rows,
                "bucket": ["x"] * rows,
                "industry_name": ["i"] * rows,
            }
        )
        oof = pd.DataFrame(
            {
                "row_index": np.arange(rows, dtype=np.int64),
                "fold_id": [0] * rows,
                "date": frame["date"],
                "market": frame["market"],
                "bucket": frame["bucket"],
                "industry_name": frame["industry_name"],
                "error_code": [ERROR_A_TOP_TP] * 40 + [ERROR_B_TOP_FP] * 60,
            }
        )
        args = SimpleNamespace(
            date_column="date",
            market_column="market",
            bucket_column="bucket",
            industry_column="industry_name",
            minimum_coverage=0.5,
            minimum_positive_rows=10,
            minimum_negative_rows=10,
            minimum_orientation_consistency=0.5,
        )
        fold_map, _ = compute_univariate_map(
            oof,
            frame,
            ["f"],
            pd.DataFrame([[1.0]], index=["f"], columns=["f"]),
            {"f": ["f"]},
            pd.DataFrame(),
            pd.DataFrame(),
            {"selection": [0], "confirmation": [], "recent_audit": []},
            args,
            transform_plan={"f": ["raw", "missing_indicator"]},
            stage_name="TEST",
        )
        summary = summarize_univariate_map(
            fold_map,
            pd.DataFrame(),
            {"selection": [0], "confirmation": [], "recent_audit": []},
            args,
        )
        missing = summary[summary["transform"].eq("missing_indicator")].iloc[0]
        self.assertLess(float(missing["selection__mean_effective_positive_coverage"]), 0.5)
        self.assertFalse(bool(missing["separation_gate"]))

    def test_selection_ranking_does_not_use_recent(self) -> None:
        def make_map(recent_auc: float) -> pd.DataFrame:
            rows = []
            for fold_id, role, auc in [(0, "selection", 0.65), (7, "recent_audit", recent_auc)]:
                rows.append(
                    {
                        "fold_id": fold_id,
                        "fold_role": role,
                        "axis": "AB",
                        "feature": "f",
                        "transform": "raw",
                        "node_id": "raw::f",
                        "positive_rows": 100,
                        "negative_rows": 200,
                        "positive_valid_n": 100,
                        "negative_valid_n": 200,
                        "positive_coverage": 1.0,
                        "negative_coverage": 1.0,
                        "raw_auc": auc,
                        "oriented_auc": auc,
                        "orientation": 1,
                        "ks": 0.2,
                        "js_divergence": 0.1,
                        "matched_concordance": 0.62,
                        "matched_orientation": 1,
                    }
                )
            return pd.DataFrame(rows)

        args = SimpleNamespace(
            minimum_coverage=0.5,
            minimum_positive_rows=30,
            minimum_negative_rows=60,
            minimum_orientation_consistency=0.8,
        )
        roles = {"selection": [0], "confirmation": [], "recent_audit": [7]}
        score_a = summarize_univariate_map(make_map(0.51), pd.DataFrame(), roles, args).iloc[0]["separation_score"]
        score_b = summarize_univariate_map(make_map(0.99), pd.DataFrame(), roles, args).iloc[0]["separation_score"]
        self.assertAlmostEqual(float(score_a), float(score_b), places=12)


class EndToEndTest(unittest.TestCase):
    def test_csv_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir()
            (root / "references" / "feature_metadata").mkdir(parents=True)
            (root / "outputs" / "surge_organic_ablation_v8").mkdir(parents=True)
            (root / "outputs" / "surge_correlation_map_complete").mkdir(parents=True)
            rng = np.random.default_rng(123)
            dates = pd.bdate_range("2020-01-01", periods=180)
            tickers = ["A", "B", "C", "D", "E", "F"]
            rows = []
            source_row_id = 0
            for date_index, date in enumerate(dates):
                regime = np.sin(date_index / 20.0)
                for ticker_index, ticker in enumerate(tickers):
                    f0 = rng.normal() + 0.4 * regime
                    f1 = rng.normal()
                    f2 = 0.8 * f0 + rng.normal(scale=0.3)
                    f3 = rng.normal()
                    f4 = rng.normal()
                    f5 = rng.normal()
                    f6 = rng.normal()
                    f7 = rng.normal()
                    logit = 1.2 * f0 - 0.8 * f1 + 0.9 * (f3 * f4) + 0.4 * regime - 2.0
                    probability = 1.0 / (1.0 + np.exp(-logit))
                    target = int(rng.random() < probability)
                    rows.append(
                        {
                            "source_row_id": source_row_id,
                            "date": date,
                            "ticker": ticker,
                            "market": "KOSPI" if ticker_index < 4 else "KOSDAQ",
                            "bucket": "x" if ticker_index % 2 == 0 else "y",
                            "industry_name": "i1" if ticker_index < 3 else "i2",
                            "t_signal_5": f0,
                            "t_signal_20": f1,
                            "t_peer_5": f2,
                            "t_inter_a_20": f3,
                            "t_inter_b_20": f4,
                            "t_noise_20": f5,
                            "t_noise_60": f6,
                            "t_sparse_20": f7 if rng.random() > 0.2 else np.nan,
                            "target": target,
                        }
                    )
                    source_row_id += 1
            full = pd.DataFrame(rows)
            feature_columns = [
                "t_signal_5",
                "t_signal_20",
                "t_peer_5",
                "t_inter_a_20",
                "t_inter_b_20",
                "t_noise_20",
                "t_noise_60",
                "t_sparse_20",
            ]
            source = full.drop(columns=["target"])
            target = full[["source_row_id", "date", "ticker", "target"]].rename(columns={"target": "label_abs_surge_3d_5pct"})
            target["target_valid"] = True
            source.to_csv(root / "data" / "training.csv", index=False)
            target.to_csv(root / "data" / "target.csv", index=False)
            folds = []
            validation_starts = [80, 100, 120, 140]
            for fold_id, start in enumerate(validation_starts):
                folds.append(
                    {
                        "fold_id": fold_id,
                        "train_start": str(dates[0].date()),
                        "train_end": str(dates[start - 6].date()),
                        "purge_start": str(dates[start - 5].date()),
                        "purge_end": str(dates[start - 1].date()),
                        "validation_start": str(dates[start].date()),
                        "validation_end": str(dates[start + 19].date()),
                    }
                )
            (root / "references" / "feature_metadata" / "outer_walk_forward_folds.json").write_text(json.dumps(folds), encoding="utf-8")
            profile = {"profiles": {"P0_FULL_439": {"count": len(feature_columns), "features": feature_columns}}}
            (root / "references" / "feature_metadata" / "profile_manifest.json").write_text(json.dumps(profile), encoding="utf-8")
            corr = source[feature_columns].corr().abs().fillna(0.0)
            np.fill_diagonal(corr.values, 1.0)
            corr.to_csv(root / "outputs" / "surge_correlation_map_complete" / "cluster_basis_combined_abs.csv.gz", compression="gzip")
            cluster_rows = []
            for threshold in [0.92]:
                for cluster_id, feature in enumerate(feature_columns, start=1):
                    cluster_rows.append({"threshold": threshold, "cluster_id": cluster_id, "cluster_size": 1, "feature": feature})
            pd.DataFrame(cluster_rows).to_csv(root / "outputs" / "surge_organic_ablation_v8" / "correlation_cluster_assignments_v8.csv", index=False)
            pd.DataFrame({"feature": feature_columns, "consensus_decision": ["HOLD"] * len(feature_columns)}).to_csv(
                root / "outputs" / "surge_organic_ablation_v8" / "organic_feature_consensus.csv", index=False
            )
            command = [
                sys.executable,
                str(Path(__file__).with_name("run_surge_separation_map_v9.py")),
                "--package-root",
                str(root),
                "--dataset",
                str(root / "data" / "training.csv"),
                "--target-sidecar",
                str(root / "data" / "target.csv"),
                "--folds",
                str(root / "references" / "feature_metadata" / "outer_walk_forward_folds.json"),
                "--feature-profile-manifest",
                str(root / "references" / "feature_metadata" / "profile_manifest.json"),
                "--correlation-matrix",
                str(root / "outputs" / "surge_correlation_map_complete" / "cluster_basis_combined_abs.csv.gz"),
                "--v8-dir",
                str(root / "outputs" / "surge_organic_ablation_v8"),
                "--output",
                str(root / "outputs" / "v9"),
                "--target-column",
                "label_abs_surge_3d_5pct",
                "--selection-folds",
                "0,1",
                "--confirmation-folds",
                "2",
                "--recent-folds",
                "3",
                "--base-backends",
                "lightgbm_cpu",
                "--device",
                "cpu",
                "--no-require-full-439",
                "--minimum-positive-rows",
                "2",
                "--minimum-negative-rows",
                "4",
                "--minimum-coverage",
                "0.2",
                "--minimum-orientation-consistency",
                "0.5",
                "--interaction-top-per-fold",
                "20",
                "--interaction-validate-count",
                "6",
                "--axis-candidate-count",
                "4",
                "--horizon-candidate-count",
                "2",
                "--pair-candidate-count",
                "2",
                "--minimum-alerts",
                "5",
                "--minimum-probe-train-rows",
                "20",
                "--threads",
                "1",
                "--xgboost-threads",
                "1",
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=180)
            if completed.returncode != 0:
                self.fail(f"runner failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
            output = root / "outputs" / "v9"
            status = json.loads((output / "RUN_STATUS.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "SUCCESS")
            manifest = json.loads((output / "SEPARATION_MAP_MANIFEST_V9.json").read_text(encoding="utf-8"))
            valid, reasons = verify_output_inventory(output, manifest["output_inventory"])
            self.assertTrue(valid, reasons)
            self.assertTrue((output / "separation_univariate_summary.csv").exists())
            self.assertTrue((output / "precision30_probe_by_fold.csv").exists())
            self.assertTrue((output / "separation_graph_v9.graphml").exists())
            verifier = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("verify_surge_separation_map_v9.py")),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if verifier.returncode != 0:
                self.fail(f"verifier failed\nSTDOUT:\n{verifier.stdout}\nSTDERR:\n{verifier.stderr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
