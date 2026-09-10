from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from run_surge_tickerwise_correlation_map_v10 import build_parser, normalize_ticker_series
from surge_ticker_common_v10 import (
    ERROR_A_TOP_TP,
    ERROR_B_TOP_FP,
    ERROR_C_LOW_TP,
    ERROR_D_LOW_TN,
    build_ticker_error_group_codes,
    cluster_from_correlation,
    compute_ticker_correlation_matrix,
    matched_pair_concordance,
    rolling_percentile_rank,
    rolling_z_score,
    select_threshold_for_precision,
)
from surge_ticker_hierarchy_v10 import (
    HierarchyConfig,
    build_hierarchical_effect_map,
    build_ticker_driver_profiles,
    build_ticker_similarity_map,
    save_ticker_map_visualizations,
)


class TickerwiseCorrelationMapV10Tests(unittest.TestCase):
    def test_default_run_is_map_only(self) -> None:
        args = build_parser().parse_args([])
        self.assertFalse(args.run_probe)
        self.assertTrue(args.create_plots)

    def test_ticker_normalization_preserves_non_numeric_codes(self) -> None:
        values = pd.Series([100, "5930", "000270", 1234.0, "AAA", None])
        normalized = normalize_ticker_series(values).tolist()
        self.assertEqual(normalized[:5], ["000100", "005930", "000270", "001234", "AAA"])
        self.assertEqual(normalized[5], "UNKNOWN")

    def test_rolling_transforms_are_causal(self) -> None:
        values = np.arange(100, dtype=float)
        baseline_z = rolling_z_score(values, 20)
        baseline_rank = rolling_percentile_rank(values, 20)
        changed = values.copy()
        changed[80:] += 10_000
        changed_z = rolling_z_score(changed, 20)
        changed_rank = rolling_percentile_rank(changed, 20)
        np.testing.assert_allclose(baseline_z[:80], changed_z[:80], equal_nan=True)
        np.testing.assert_allclose(baseline_rank[:80], changed_rank[:80], equal_nan=True)

    def test_error_groups_are_ticker_local(self) -> None:
        target = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int8)
        score = np.asarray([0.9, 0.8, 0.1, 0.2, 0.7, 0.6, 0.3, 0.4])
        codes, rank = build_ticker_error_group_codes(target, score, top_quantile=0.75, low_quantile=0.5)
        self.assertEqual(codes[0], ERROR_A_TOP_TP)
        self.assertEqual(codes[1], ERROR_B_TOP_FP)
        self.assertEqual(codes[2], ERROR_C_LOW_TP)
        self.assertEqual(codes[3], ERROR_D_LOW_TN)
        self.assertTrue(np.all((rank >= 0) & (rank <= 1)))

    def test_matched_pair_concordance_maps_global_row_ids(self) -> None:
        values = np.asarray([0.2, 0.9, 0.1], dtype=float)
        row_ids = np.asarray([101, 205, 310], dtype=np.int64)
        pairs = pd.DataFrame(
            {
                "case_index": [205, 205, 205, 101],
                "control_index": [101, 310, 101, 310],
            }
        )
        concordance, direction, count = matched_pair_concordance(values, pairs, row_indices=row_ids)
        self.assertEqual(count, 4)
        self.assertEqual(direction, 1)
        self.assertGreater(concordance, 0.5)

    def test_ticker_correlation_clusters_recover_duplicate(self) -> None:
        rng = np.random.default_rng(123)
        base = rng.normal(size=300)
        frame = pd.DataFrame(
            {
                "f1": base,
                "f2": base + rng.normal(scale=0.001, size=300),
                "f3": rng.normal(size=300),
            }
        )
        correlation, valid_n = compute_ticker_correlation_matrix(
            frame,
            ["f1", "f2", "f3"],
            method="spearman",
            minimum_rows=50,
            shrinkage_strength=1.0,
        )
        self.assertGreater(abs(float(correlation.loc["f1", "f2"])), 0.98)
        self.assertEqual(int(valid_n.loc["f1", "f2"]), 300)
        clusters = cluster_from_correlation(correlation, threshold=0.95).set_index("feature")
        self.assertEqual(clusters.loc["f1", "cluster_id"], clusters.loc["f2", "cluster_id"])

    def test_precision_policy_never_falls_back_to_all_alerts(self) -> None:
        target = np.asarray([1, 0, 0, 0, 0, 0], dtype=np.int8)
        scores = np.asarray([0.6, 0.9, 0.8, 0.7, 0.5, 0.4])
        result = select_threshold_for_precision(target, scores, target_precision=0.70, minimum_alerts=3)
        self.assertFalse(bool(result["gate_pass"]))
        self.assertTrue(math.isinf(float(result["threshold"])))
        self.assertEqual(int(result["alerts"]), 0)

    def test_hierarchical_prior_is_leave_one_ticker_out(self) -> None:
        summary, by_fold, metadata = self._hierarchy_fixture(
            {"AAA": 0.10, "BBB": 0.03, "CCC": -0.03},
            effective_rows=400,
        )
        config = HierarchyConfig(
            prior_strength=20,
            minimum_peer_tickers=1,
            weak_effect=0.01,
            strong_effect=0.05,
            unique_delta=0.04,
            amplified_delta=0.03,
            reversal_effect=0.03,
            minimum_reliability=0.01,
            industry_weight=0.0,
            bucket_weight=0.0,
            market_weight=0.0,
            global_weight=1.0,
        )
        detailed, _ = build_hierarchical_effect_map(summary, by_fold, metadata, [0, 1], config)
        aaa = detailed.loc[detailed["ticker"].eq("AAA")].iloc[0]
        self.assertAlmostEqual(float(aaa["global_loo_effect"]), 0.0, places=6)
        self.assertGreater(float(aaa["posterior_signed_effect"]), 0.0)
        self.assertLess(float(aaa["posterior_signed_effect"]), 0.10)

    def test_hierarchy_detects_unique_and_reversal_drivers(self) -> None:
        summary, by_fold, metadata = self._hierarchy_fixture(
            {"AAA": 0.12, "BBB": -0.05, "CCC": -0.05, "DDD": -0.05},
            effective_rows=700,
        )
        config = HierarchyConfig(
            prior_strength=5,
            minimum_peer_tickers=1,
            weak_effect=0.01,
            strong_effect=0.04,
            unique_delta=0.03,
            amplified_delta=0.02,
            reversal_effect=0.02,
            minimum_reliability=0.01,
            industry_weight=0.0,
            bucket_weight=0.0,
            market_weight=0.0,
            global_weight=1.0,
        )
        detailed, _ = build_hierarchical_effect_map(summary, by_fold, metadata, [0, 1], config)
        aaa = detailed.loc[detailed["ticker"].eq("AAA")].iloc[0]
        self.assertEqual(aaa["ticker_effect_class"], "TICKER_DIRECTION_REVERSAL")
        self.assertTrue(bool(aaa["is_ticker_specific"]))

        summary2, by_fold2, metadata2 = self._hierarchy_fixture(
            {"AAA": 0.12, "BBB": 0.0, "CCC": 0.0, "DDD": 0.0},
            effective_rows=700,
        )
        detailed2, _ = build_hierarchical_effect_map(summary2, by_fold2, metadata2, [0, 1], config)
        aaa2 = detailed2.loc[detailed2["ticker"].eq("AAA")].iloc[0]
        self.assertEqual(aaa2["ticker_effect_class"], "TICKER_UNIQUE_DRIVER")

    def test_partial_pooling_shrinks_low_evidence_ticker(self) -> None:
        summary, by_fold, metadata = self._hierarchy_fixture(
            {"AAA": 0.15, "BBB": 0.02, "CCC": 0.02},
            effective_rows=8,
        )
        config = HierarchyConfig(
            prior_strength=120,
            minimum_peer_tickers=1,
            minimum_reliability=0.0,
            industry_weight=0.0,
            bucket_weight=0.0,
            market_weight=0.0,
            global_weight=1.0,
        )
        detailed, _ = build_hierarchical_effect_map(summary, by_fold, metadata, [0, 1], config)
        aaa = detailed.loc[detailed["ticker"].eq("AAA")].iloc[0]
        raw = float(aaa["ticker_signed_effect"])
        prior = float(aaa["peer_prior_signed_effect"])
        posterior = float(aaa["posterior_signed_effect"])
        self.assertLess(abs(posterior - prior), abs(raw - prior))

    def test_driver_profiles_and_ticker_similarity(self) -> None:
        rows: list[dict[str, object]] = []
        for ticker, sign in [("AAA", 1.0), ("BBB", 1.0), ("CCC", -1.0)]:
            for index in range(12):
                rows.append(
                    {
                        "ticker": ticker,
                        "axis": "AB" if index % 2 == 0 else "TARGET",
                        "node_id": f"f{index}",
                        "source_feature": f"f{index}",
                        "posterior_signed_effect": sign * (0.02 + index * 0.003),
                        "ticker_driver_score": 1.0 + index,
                        "is_ticker_specific": index < 3,
                        "is_peer_shared": index >= 3,
                        "ticker_effect_class": "TICKER_UNIQUE_DRIVER" if index < 3 else "UNIVERSAL_DRIVER",
                    }
                )
        effect = pd.DataFrame(rows)
        profiles, membership = build_ticker_driver_profiles(effect, per_axis_count=4, unique_count=3, shared_count=4)
        self.assertEqual(set(profiles), {"AAA", "BBB", "CCC"})
        self.assertFalse(membership.empty)
        config = HierarchyConfig(
            similarity_min_common=4,
            similarity_edge_threshold=0.2,
            similarity_top_k=2,
            similarity_cluster_threshold=0.5,
            similarity_feature_count=20,
        )
        edges, clusters, matrix, overlap = build_ticker_similarity_map(effect, config)
        self.assertFalse(edges.empty)
        assignments = clusters.set_index("ticker")
        self.assertEqual(assignments.loc["AAA", "ticker_map_cluster"], assignments.loc["BBB", "ticker_map_cluster"])
        self.assertNotEqual(assignments.loc["AAA", "ticker_map_cluster"], assignments.loc["CCC", "ticker_map_cluster"])
        self.assertEqual(matrix.shape, (3, 3))
        self.assertGreater(int(overlap.loc["AAA", "BBB"]), 0)

    def test_visualizations_are_created(self) -> None:
        rows = []
        for ticker in ["AAA", "BBB"]:
            for index in range(5):
                rows.append(
                    {
                        "ticker": ticker,
                        "axis": "AB",
                        "node_id": f"f{index}",
                        "posterior_signed_effect": (index + 1) * (0.02 if ticker == "AAA" else -0.02),
                        "ticker_driver_score": 1.0 + index,
                    }
                )
        effect = pd.DataFrame(rows)
        similarity = pd.DataFrame([[1.0, -0.5], [-0.5, 1.0]], index=["AAA", "BBB"], columns=["AAA", "BBB"])
        with tempfile.TemporaryDirectory(prefix="ticker_v10_plot_") as directory:
            generated = save_ticker_map_visualizations(effect, similarity, Path(directory), per_ticker_top_n=4, heatmap_node_count=4)
            self.assertGreaterEqual(len(generated), 4)
            self.assertTrue((Path(directory) / "ticker_similarity_heatmap.png").exists())
            self.assertTrue((Path(directory) / "per_ticker" / "AAA" / "ticker_driver_profile.png").exists())

    def test_end_to_end_ticker_specific_pipeline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ticker_v10_e2e_") as directory:
            root = Path(directory)
            data_dir = root / "data"
            refs_dir = root / "refs"
            output = root / "out"
            data_dir.mkdir()
            refs_dir.mkdir()
            source, target, features, folds = self._make_synthetic_data()
            source.to_csv(data_dir / "train.csv", index=False)
            target.to_csv(data_dir / "target.csv", index=False)
            (refs_dir / "profile.json").write_text(
                json.dumps({"profiles": {"P0_FULL_439": features}}, indent=2), encoding="utf-8"
            )
            (refs_dir / "folds.json").write_text(json.dumps({"folds": folds}, indent=2), encoding="utf-8")
            script = Path(__file__).with_name("run_surge_tickerwise_correlation_map_v10.py")
            command = [
                sys.executable,
                str(script),
                "--dataset", str(data_dir / "train.csv"),
                "--target-sidecar", str(data_dir / "target.csv"),
                "--folds", str(refs_dir / "folds.json"),
                "--feature-profile-manifest", str(refs_dir / "profile.json"),
                "--output", str(output),
                "--no-require-full-439",
                "--selection-folds", "0,1",
                "--confirmation-folds", "2",
                "--recent-folds", "3",
                "--base-backends", "lightgbm_cpu",
                "--probe-backends", "lightgbm_cpu",
                "--threads", "1",
                "--base-iterations", "10",
                "--base-feature-count", "6",
                "--minimum-base-feature-count", "2",
                "--minimum-train-rows-model", "60",
                "--minimum-train-positive-model", "4",
                "--minimum-train-negative-model", "10",
                "--minimum-validation-rows-model", "20",
                "--minimum-validation-positive-model", "2",
                "--minimum-validation-negative-model", "4",
                "--minimum-validation-rows-map", "15",
                "--minimum-validation-positive-map", "2",
                "--minimum-validation-negative-map", "4",
                "--minimum-feature-valid-rows", "15",
                "--minimum-feature-positive-rows", "2",
                "--minimum-feature-negative-rows", "4",
                "--minimum-feature-coverage", "0.2",
                "--minimum-map-valid-rows", "10",
                "--auc-prior-strength", "10",
                "--minimum-correlation-rows", "15",
                "--correlation-shrinkage-strength", "8",
                "--correlation-edge-threshold", "0.60",
                "--minimum-fold-abs-correlation", "0.40",
                "--minimum-correlation-sign-consistency", "0.40",
                "--cluster-threshold", "0.65",
                "--no-save-full-matrices",
                "--minimum-selection-mean-auc", "0.50",
                "--minimum-selection-min-auc", "0.45",
                "--minimum-direction-consistency", "0.40",
                "--minimum-selection-fold-count", "1",
                "--target-source-count", "4",
                "--ab-source-count", "4",
                "--cd-source-count", "4",
                "--profile-target-count", "4",
                "--profile-ab-count", "4",
                "--profile-cd-count", "4",
                "--profile-extended-target-count", "3",
                "--profile-extended-ab-count", "3",
                "--profile-extended-cd-count", "3",
                "--profile-cluster-rep-count", "6",
                "--profile-combined-max-count", "12",
                "--run-probe",
                "--probe-profiles", "BASE_TICKER,HIERARCHICAL_COMBINED,COMBINED",
                "--minimum-probe-feature-count", "2",
                "--minimum-calibration-rows", "15",
                "--minimum-calibration-positive", "2",
                "--minimum-calibration-rows-rank", "8",
                "--target-precision", "0.60",
                "--minimum-alerts-per-ticker-diagnostic", "1",
                "--minimum-alerts-per-ticker-policy", "1",
                "--minimum-portfolio-alerts", "3",
                "--required-fold-pass-rate", "0.0",
                "--controls-per-case", "1",
                "--maximum-match-day-distance", "120",
                "--match-regime-columns", "2",
                "--hierarchy-prior-strength", "20",
                "--minimum-peer-tickers", "2",
                "--ticker-similarity-min-common", "3",
                "--ticker-similarity-edge-threshold", "0.05",
                "--no-create-plots",
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=180)
            if completed.returncode != 0:
                self.fail(f"Pipeline failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}")
            status = json.loads((output / "RUN_STATUS.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "SUCCESS")
            recommendation = json.loads((output / "FINAL_RECOMMENDATION_V10.json").read_text(encoding="utf-8"))
            self.assertFalse(bool(recommendation["common_prediction_model_used"]))
            self.assertTrue(bool(recommendation["ticker_specific_models"]))
            effect = pd.read_csv(output / "ticker_hierarchical_effect_map.csv")
            self.assertEqual(set(effect["ticker"]), {"AAA", "BBB", "CCC", "DDD"})
            self.assertIn("ticker_specific_delta", effect.columns)
            self.assertIn("ticker_effect_class", effect.columns)
            self.assertTrue((output / "ticker_similarity_matrix.csv").exists())
            self.assertTrue((output / "per_ticker" / "AAA" / "hierarchical_driver_map.csv").exists())

    @staticmethod
    def _hierarchy_fixture(
        effects: dict[str, float],
        *,
        effective_rows: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        summary_rows = []
        fold_rows = []
        metadata_rows = []
        for ticker, effect in effects.items():
            direction = 1 if effect >= 0 else -1
            auc = 0.5 + abs(effect)
            summary_rows.append(
                {
                    "ticker": ticker,
                    "axis": "AB",
                    "feature": "f1",
                    "transform": "raw",
                    "selection_direction": direction,
                    "selection_mean_shrunk_auc": auc,
                    "selection_mean_fixed_auc": auc,
                    "selection_fold_count": 2,
                    "selection_direction_consistency": 1.0,
                    "selection_evidence_score": 1.0,
                    "confirmation_mean_fixed_auc": auc,
                    "recent_mean_fixed_auc": auc,
                }
            )
            positive_n = max(2, effective_rows // 3)
            negative_n = max(3, effective_rows - positive_n)
            for fold_id in [0, 1]:
                fold_rows.append(
                    {
                        "ticker": ticker,
                        "axis": "AB",
                        "feature": "f1",
                        "transform": "raw",
                        "fold_id": fold_id,
                        "positive_n": positive_n,
                        "negative_n": negative_n,
                        "valid_rows": positive_n + negative_n,
                        "matched_concordance": auc,
                        "matched_pair_count": positive_n,
                        "positive_coverage": 1.0,
                        "negative_coverage": 1.0,
                    }
                )
            metadata_rows.append(
                {
                    "ticker": ticker,
                    "name": ticker,
                    "market": "KOSPI",
                    "bucket": "B1",
                    "industry": "I1",
                }
            )
        return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), pd.DataFrame(metadata_rows)

    @staticmethod
    def _make_synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[dict[str, str | int]]]:
        rng = np.random.default_rng(20260813)
        dates = pd.bdate_range("2020-01-01", periods=420)
        features = ["f1", "f2", "f3", "f4", "f5", "f6"]
        mechanisms = {
            "AAA": np.asarray([1.3, -0.4, 0.0, 0.2, 0.0, 0.0]),
            "BBB": np.asarray([-1.1, 0.0, 1.0, 0.0, 0.2, 0.0]),
            "CCC": np.asarray([0.0, 1.1, -0.4, 0.0, 0.0, 0.5]),
            "DDD": np.asarray([0.2, 0.0, 0.0, -1.0, 1.0, 0.0]),
        }
        metadata = {
            "AAA": ("KOSPI", "TECH", "SEMICONDUCTOR"),
            "BBB": ("KOSPI", "AUTO", "AUTOMOTIVE"),
            "CCC": ("KOSDAQ", "BIO", "BIOTECH"),
            "DDD": ("KOSDAQ", "BATTERY", "MATERIALS"),
        }
        source_rows: list[dict[str, object]] = []
        target_rows: list[dict[str, object]] = []
        row_id = 0
        market_factor = rng.normal(size=len(dates))
        for ticker, beta in mechanisms.items():
            latent_state = np.zeros(len(dates), dtype=float)
            for index in range(1, len(dates)):
                latent_state[index] = 0.75 * latent_state[index - 1] + rng.normal(scale=0.65)
            matrix = rng.normal(size=(len(dates), len(features)))
            matrix[:, 0] += 0.45 * market_factor
            matrix[:, 1] += 0.35 * latent_state
            matrix[:, 2] = matrix[:, 0] + rng.normal(scale=0.08, size=len(dates))
            linear = matrix @ beta + 0.35 * latent_state - 1.35
            probability = 1.0 / (1.0 + np.exp(-linear))
            target = rng.binomial(1, probability).astype(int)
            market, bucket, industry = metadata[ticker]
            for index, date in enumerate(dates):
                source = {
                    "source_row_id": row_id,
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "name": f"NAME_{ticker}",
                    "market": market,
                    "bucket": bucket,
                    "industry_name": industry,
                }
                source.update({feature: float(matrix[index, column]) for column, feature in enumerate(features)})
                source_rows.append(source)
                target_rows.append(
                    {
                        "source_row_id": row_id,
                        "date": date.strftime("%Y-%m-%d"),
                        "ticker": ticker,
                        "label_abs_surge_3d_5pct": int(target[index]),
                        "target_valid": True,
                    }
                )
                row_id += 1
        fold_bounds = [(149, 154, 200), (199, 204, 250), (249, 254, 300), (299, 304, 350)]
        folds: list[dict[str, str | int]] = []
        for fold_id, (train_end, validation_start, validation_end) in enumerate(fold_bounds):
            folds.append(
                {
                    "fold_id": fold_id,
                    "train_start": dates[0].strftime("%Y-%m-%d"),
                    "train_end": dates[train_end].strftime("%Y-%m-%d"),
                    "validation_start": dates[validation_start].strftime("%Y-%m-%d"),
                    "validation_end": dates[validation_end].strftime("%Y-%m-%d"),
                }
            )
        return pd.DataFrame(source_rows), pd.DataFrame(target_rows), features, folds


if __name__ == "__main__":
    unittest.main(verbosity=2)
