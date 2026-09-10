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

    def test_recompute_surge_target_rejects_just_below_five_percent(self) -> None:
        dates = pd.bdate_range("2026-02-02", periods=5)
        source = pd.DataFrame(
            {
                "source_row_id": np.arange(5, dtype=np.int64),
                "date": dates,
                "ticker": ["000001"] * 5,
                "t_price_ret_1": [0.0, 0.04999, 0.0, 0.0, 0.0],
            }
        )
        label, valid, first_hit, best = surge_corr.recompute_surge_target_from_returns(source)
        self.assertTrue(bool(valid[0]))
        self.assertEqual(int(label[0]), 0)
        self.assertEqual(int(first_hit[0]), 0)
        self.assertAlmostEqual(float(best[0]), 0.04999, places=12)

    def test_binned_mutual_information_detects_nonlinear_signal(self) -> None:
        rng = np.random.default_rng(17)
        values = rng.normal(size=2000)
        target = (np.abs(values) > 0.8).astype(np.int8)
        frame = pd.DataFrame(
            {
                "nonlinear_signal": values,
                "noise": rng.normal(size=len(values)),
            }
        )
        result = surge_corr.binned_mutual_information(
            frame, target, ["nonlinear_signal", "noise"], bins=20, min_rows=100
        ).set_index("feature")
        self.assertGreater(
            float(result.loc["nonlinear_signal", "mutual_information"]),
            float(result.loc["noise", "mutual_information"]) + 0.20,
        )

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


    def test_load_fold_definitions_sorts_and_rejects_duplicate_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            valid_path = root / "folds_valid.json"
            valid_path.write_text(
                json.dumps(
                    [
                        {
                            "fold_id": 2,
                            "train_start": "2026-01-01",
                            "train_end": "2026-01-10",
                            "validation_start": "2026-01-20",
                            "validation_end": "2026-01-30",
                        },
                        {
                            "fold_id": 0,
                            "train_start": "2025-01-01",
                            "train_end": "2025-01-10",
                            "validation_start": "2025-01-20",
                            "validation_end": "2025-01-30",
                        },
                    ]
                ),
                encoding="utf-8",
            )
            loaded = surge_corr.load_fold_definitions(valid_path)
            self.assertEqual([fold.fold_id for fold in loaded], [0, 2])

            duplicate_path = root / "folds_duplicate.json"
            duplicate_payload = [
                {
                    "fold_id": 1,
                    "train_start": "2025-01-01",
                    "train_end": "2025-01-10",
                    "validation_start": "2025-01-20",
                    "validation_end": "2025-01-30",
                },
                {
                    "fold_id": 1,
                    "train_start": "2026-01-01",
                    "train_end": "2026-01-10",
                    "validation_start": "2026-01-20",
                    "validation_end": "2026-01-30",
                },
            ]
            duplicate_path.write_text(json.dumps(duplicate_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fold_id 중복"):
                surge_corr.load_fold_definitions(duplicate_path)

    def test_fold_validation_rejects_declared_purge_overlap(self) -> None:
        dates = pd.Series(pd.bdate_range("2026-01-02", periods=20))
        target_valid = np.ones(len(dates), dtype=bool)
        target = np.array([0, 1] * 10, dtype=np.int8)
        folds = [
            surge_corr.FoldDefinition(
                fold_id=0,
                train_start=str(dates.iloc[0].date()),
                train_end=str(dates.iloc[7].date()),
                purge_start=str(dates.iloc[7].date()),
                purge_end=str(dates.iloc[10].date()),
                validation_start=str(dates.iloc[12].date()),
                validation_end=str(dates.iloc[18].date()),
            )
        ]
        with self.assertRaisesRegex(ValueError, "declared_purge_overlaps_train"):
            surge_corr.validate_fold_definitions(
                dates,
                target_valid,
                target,
                folds,
                {"selection": [0], "confirmation": [], "recent_audit": []},
                minimum_purge_days=3,
            )

    def test_intersect_row_limits_treats_zero_as_unlimited(self) -> None:
        self.assertEqual(surge_corr.intersect_row_limits(0, 0), 0)
        self.assertEqual(surge_corr.intersect_row_limits(0, 500), 500)
        self.assertEqual(surge_corr.intersect_row_limits(700, 0), 700)
        self.assertEqual(surge_corr.intersect_row_limits(700, 500), 500)


class AdvancedCorrelationMapTests(unittest.TestCase):
    @staticmethod
    def _support_summary(features: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature": features,
                "selection_validation_pearson_mean": [0.10] * len(features),
                "selection_validation_spearman_mean": [0.08] * len(features),
                "selection_validation_within_date_pearson_mean": [0.07] * len(features),
                "selection_validation_within_ticker_pearson_mean": [0.06] * len(features),
            }
        )

    @staticmethod
    def _fold_row(feature: str, fold: int, pearson: float, spearman: float | None = None) -> dict[str, object]:
        spearman_value = pearson if spearman is None else spearman
        return {
            "outer_fold": fold,
            "feature": feature,
            "validation_target_pearson": pearson,
            "validation_target_spearman": spearman_value,
            "validation_target_within_date_pearson": pearson,
            "validation_target_within_ticker_pearson": pearson,
        }

    def test_strict_support_rejects_mixed_sign_and_weak_confirmation_fold(self) -> None:
        features = ["strong", "mixed", "weak_fold"]
        summary = self._support_summary(features)
        records: list[dict[str, object]] = []
        for feature in features:
            records.extend(self._fold_row(feature, fold, 0.10) for fold in range(5))
        records.extend(
            [
                self._fold_row("strong", 5, 0.05),
                self._fold_row("strong", 6, 0.06),
                self._fold_row("strong", 7, 0.05),
                self._fold_row("mixed", 5, 0.04),
                self._fold_row("mixed", 6, -0.03),
                self._fold_row("mixed", 7, 0.05),
                self._fold_row("weak_fold", 5, 0.05),
                self._fold_row("weak_fold", 6, 0.005),
                self._fold_row("weak_fold", 7, 0.05),
            ]
        )
        support = surge_corr.evaluate_support_evidence(
            summary=summary,
            fold_correlations=pd.DataFrame.from_records(records),
            roles={"selection": [0, 1, 2, 3, 4], "confirmation": [5, 6], "recent_audit": [7]},
            selection_min_abs_corr=0.02,
            selection_min_sign_consistency=0.80,
            confirmation_min_abs_corr=0.02,
            confirmation_min_sign_consistency=1.0,
            confirmation_min_retention=0.40,
            recent_min_abs_corr=0.02,
            recent_min_sign_consistency=1.0,
            recent_min_retention=0.40,
        ).set_index("feature")

        self.assertTrue(bool(support.loc["strong", "strict_stable_supported"]))
        self.assertFalse(bool(support.loc["mixed", "confirmation_supported"]))
        self.assertIn(
            "confirmation_fold_direction_mismatch",
            str(support.loc["mixed", "confirmation_support_reason"]),
        )
        self.assertFalse(bool(support.loc["weak_fold", "confirmation_supported"]))
        self.assertIn(
            "confirmation_one_or_more_folds_below_min",
            str(support.loc["weak_fold", "confirmation_support_reason"]),
        )
        self.assertAlmostEqual(float(support.loc["weak_fold", "confirmation_min_fold_abs_corr"]), 0.005)

    def test_support_uses_same_metric_that_defined_selection_direction(self) -> None:
        summary = pd.DataFrame(
            {
                "feature": ["rank_signal"],
                "selection_validation_pearson_mean": [0.01],
                "selection_validation_spearman_mean": [0.12],
                "selection_validation_within_date_pearson_mean": [0.03],
                "selection_validation_within_ticker_pearson_mean": [0.02],
            }
        )
        records = []
        for fold in range(5):
            records.append(self._fold_row("rank_signal", fold, pearson=-0.03 if fold % 2 else 0.03, spearman=0.12))
        records.extend(
            [
                self._fold_row("rank_signal", 5, pearson=-0.02, spearman=0.07),
                self._fold_row("rank_signal", 6, pearson=0.02, spearman=0.06),
                self._fold_row("rank_signal", 7, pearson=-0.03, spearman=0.06),
            ]
        )
        support = surge_corr.evaluate_support_evidence(
            summary=summary,
            fold_correlations=pd.DataFrame.from_records(records),
            roles={"selection": [0, 1, 2, 3, 4], "confirmation": [5, 6], "recent_audit": [7]},
            selection_min_abs_corr=0.02,
            selection_min_sign_consistency=0.80,
            confirmation_min_abs_corr=0.02,
            confirmation_min_sign_consistency=1.0,
            confirmation_min_retention=0.40,
            recent_min_abs_corr=0.02,
            recent_min_sign_consistency=1.0,
            recent_min_retention=0.40,
        ).iloc[0]
        self.assertEqual(support["selection_direction_source"], "spearman")
        self.assertEqual(support["selection_support_metric"], "validation_target_spearman")
        self.assertTrue(bool(support["strict_stable_supported"]))

    def test_cluster_representative_is_selection_fold_consensus_not_fold_zero_only(self) -> None:
        features = ["feature_a", "feature_b"]
        matrix = np.array([[1.0, 0.96], [0.96, 1.0]], dtype=float)
        quality = pd.DataFrame(
            {
                "feature": features,
                "data_quality_score": [1.0, 1.0],
                "missing_ratio": [0.0, 0.0],
            }
        )
        rows = []
        correlations = {
            0: {"feature_a": 0.90, "feature_b": 0.10},
            1: {"feature_a": 0.10, "feature_b": 0.90},
            2: {"feature_a": 0.15, "feature_b": 0.85},
        }
        for fold, values in correlations.items():
            for feature, value in values.items():
                rows.append(
                    {
                        "outer_fold": fold,
                        "feature": feature,
                        "train_target_pearson": value,
                        "train_target_spearman": value,
                        "train_target_within_date_pearson": value,
                        "train_target_within_ticker_pearson": value,
                    }
                )
        assignments, components, candidates, votes = surge_corr.build_cluster_assignments(
            matrix=matrix,
            features=features,
            thresholds=[0.92],
            quality=quality,
            fold_correlations=pd.DataFrame.from_records(rows),
            selection_fold_ids=[0, 1, 2],
        )
        self.assertEqual(components[0.92], [[0, 1]])
        representatives = assignments[assignments["is_representative"]]
        self.assertEqual(representatives.iloc[0]["representative"], "feature_b")
        self.assertEqual(int(representatives.iloc[0]["representative_vote_count"]), 2)
        self.assertAlmostEqual(float(representatives.iloc[0]["representative_vote_ratio"]), 2 / 3)
        vote_map = votes.set_index("outer_fold")["representative"].to_dict()
        self.assertEqual(vote_map[0], "feature_a")
        self.assertEqual(vote_map[1], "feature_b")
        self.assertEqual(vote_map[2], "feature_b")
        self.assertEqual(int(candidates["is_consensus_representative"].sum()), 1)

    def test_groupwise_literal_correlation_distribution(self) -> None:
        feature_frame = pd.DataFrame(
            {
                "positive": [0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
                "negative": [3.0, 2.0, 1.0, 0.0, 13.0, 12.0, 11.0, 10.0],
            }
        )
        target = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=float)
        groups = np.array(["d1"] * 4 + ["d2"] * 4)
        result = surge_corr.compute_groupwise_target_correlation_distribution(
            feature_frame=feature_frame,
            target=target,
            groups=groups,
            features=["positive", "negative"],
            prefix="date",
            min_group_rows=4,
            chunk_size=2,
        ).set_index("feature")
        self.assertEqual(int(result.loc["positive", "date_group_count_valid"]), 2)
        self.assertGreater(float(result.loc["positive", "date_corr_mean"]), 0.8)
        self.assertLess(float(result.loc["negative", "date_corr_mean"]), -0.8)
        self.assertEqual(float(result.loc["positive", "date_corr_positive_ratio"]), 1.0)
        self.assertEqual(float(result.loc["negative", "date_corr_negative_ratio"]), 1.0)

    def test_directionality_does_not_claim_surge_specific_without_crash_reference(self) -> None:
        summary = pd.DataFrame(
            {
                "surge_priority_rank": [1],
                "strict_stable_rank": pd.Series([1], dtype="Int64"),
                "feature": ["signal"],
                "target_pearson": [0.12],
                "selection_validation_pearson_mean": [0.10],
                "selection_validation_pearson_sign_consistency": [1.0],
                "selection_priority_score": [0.90],
                "strict_stable_supported": [True],
                "confirmation_supported": [True],
                "recent_supported": [True],
                "is_primary_representative": [True],
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            directionality = surge_corr.build_surge_crash_feature_comparison(
                summary=summary,
                legacy_summary_path=root / "missing_summary.csv",
                legacy_fold_path=root / "missing_fold.csv",
                selection_fold_ids=[0, 1],
                directional_min_abs_corr=0.03,
                directional_crash_weak_abs_corr=0.02,
            ).iloc[0]
        self.assertEqual(directionality["target_relation_class"], "CRASH_REFERENCE_UNAVAILABLE")
        self.assertTrue(pd.isna(directionality["surge_specific_strength"]))

    def test_feature_quality_includes_inf_audit_quantiles_and_constant_rejection(self) -> None:
        frame = pd.DataFrame(
            {
                "normal": [1.0, 2.0, np.nan, 4.0, 5.0],
                "constant": [7.0] * 5,
            }
        )
        quality = surge_corr.compute_feature_quality(
            feature_frame=frame,
            tickers=np.array(["a", "a", "b", "b", "b"]),
            features=["normal", "constant"],
            reference_audit=None,
            inf_counts={"normal": 1},
            source_dtypes={"normal": "float64", "constant": "float64"},
        ).set_index("feature")
        self.assertEqual(int(quality.loc["normal", "inf_count"]), 1)
        self.assertAlmostEqual(float(quality.loc["normal", "p50"]), 3.0)
        self.assertEqual(quality.loc["constant", "status"], "invalid")
        self.assertIn("fewer_than_2_unique_values", quality.loc["constant", "rejection_reason"])
        self.assertTrue(bool(quality.loc["constant", "near_zero_variance"]))

    def test_directionality_map_separates_common_opposite_specific_and_weak(self) -> None:
        features = ["common", "opposite", "specific", "weak"]
        summary = pd.DataFrame(
            {
                "surge_priority_rank": [1, 2, 3, 4],
                "strict_stable_rank": pd.Series([1, 2, 3, 4], dtype="Int64"),
                "feature": features,
                "target_pearson": [0.13, 0.11, 0.09, 0.01],
                "selection_validation_pearson_mean": [0.12, 0.10, 0.09, 0.01],
                "selection_validation_pearson_sign_consistency": [1.0, 1.0, 1.0, 1.0],
                "selection_priority_score": [0.95, 0.90, 0.85, 0.20],
                "strict_stable_supported": [True, True, True, False],
                "confirmation_supported": [True, True, True, False],
                "recent_supported": [True, True, True, False],
                "is_primary_representative": [True, True, True, True],
            }
        )
        crash_global = pd.DataFrame(
            {
                "feature": features,
                "target_pearson": [0.11, -0.08, 0.005, 0.02],
                "is_primary_representative": [True, True, False, False],
            }
        )
        crash_fold_rows = []
        crash_selection = {
            "common": [0.10, 0.11],
            "opposite": [-0.08, -0.09],
            "specific": [0.004, 0.006],
            "weak": [0.02, 0.02],
        }
        for feature, values in crash_selection.items():
            for fold, value in enumerate(values):
                crash_fold_rows.append({"outer_fold": fold, "feature": feature, "target_pearson": value})
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            global_path = root / "crash_global.csv"
            fold_path = root / "crash_fold.csv"
            crash_global.to_csv(global_path, index=False)
            pd.DataFrame.from_records(crash_fold_rows).to_csv(fold_path, index=False)
            directionality = surge_corr.build_surge_crash_feature_comparison(
                summary=summary,
                legacy_summary_path=global_path,
                legacy_fold_path=fold_path,
                selection_fold_ids=[0, 1],
                directional_min_abs_corr=0.03,
                directional_crash_weak_abs_corr=0.02,
            ).set_index("feature")
        self.assertEqual(directionality.loc["common", "target_relation_class"], "COMMON_LARGE_MOVE")
        self.assertEqual(directionality.loc["opposite", "target_relation_class"], "OPPOSITE_DIRECTION")
        self.assertEqual(directionality.loc["specific", "target_relation_class"], "SURGE_SPECIFIC")
        self.assertEqual(directionality.loc["weak", "target_relation_class"], "WEAK_OR_UNCLEAR")
        self.assertGreater(
            float(directionality.loc["opposite", "surge_specific_strength"]),
            float(directionality.loc["common", "surge_specific_strength"]),
        )


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
                "surge_feature_groupwise_correlation_summary.csv",
                "surge_feature_mi_by_selection_fold.csv",
                "surge_feature_correlation_summary.csv",
                "surge_feature_priority.csv",
                "surge_support_audit.csv",
                "surge_directional_map.csv",
                "surge_feature_profile_membership.csv",
                "surge_feature_profiles.json",
                "SURGE_CORRELATION_MAP_SUMMARY.json",
                "primary_clusters.csv",
                "surge_cluster_consensus_candidates.csv",
                "surge_primary_cluster_consensus.csv",
                "surge_selection_representative_votes.csv",
                "surge_primary_selection_representative_votes.csv",
                "surge_representatives_by_fold.csv",
                "surge_primary_representatives.json",
                "high_correlation_pairs.csv",
                "correlation_matrices.npz",
                "fold_pearson_matrices.npz",
                "surge_vs_crash_feature_comparison.csv",
                "SURGE_CORRELATION_MAP_GUIDE_KO.md",
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
            for column in (
                "selection_direction_source",
                "selection_support_metric",
                "selection_min_fold_abs_corr",
                "confirmation_mean_fold_abs_corr",
                "confirmation_min_fold_abs_corr",
                "recent_mean_fold_abs_corr",
                "recent_min_fold_abs_corr",
                "strict_stable_supported",
                "target_relation_class",
                "directional_selection_score",
                "representative_vote_ratio",
            ):
                self.assertIn(column, summary.columns)
            support_audit = pd.read_csv(output_path / "surge_support_audit.csv")
            for column in (
                "selection_support_metric",
                "selection_min_fold_abs_corr",
                "selection_strict_sign_consistency",
                "confirmation_mean_fold_abs_corr",
                "confirmation_min_fold_abs_corr",
                "recent_mean_fold_abs_corr",
                "recent_min_fold_abs_corr",
            ):
                self.assertIn(column, support_audit.columns)
            profiles = json.loads((output_path / "surge_feature_profiles.json").read_text(encoding="utf-8"))
            self.assertEqual(profiles["schema"], "crashwatch_surge_correlation_profiles_v2")
            self.assertIn("P3_STRICT_CLUSTER_REP", profiles["profiles"])
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
