from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from run_surge_all_feature_ablation import build_paired_deltas, build_parser as build_ablation_parser, run as run_ablation
from run_surge_pre_model_gate import (
    classify_direction_relation,
    compute_train_validation_alignment,
    build_parser as build_gate_parser,
    run as run_gate,
)
from surge_ablation_common import (
    daily_top_fraction_metrics,
    leakage_reason,
    payload_checksum_is_valid,
    sha256_file,
)


class SurgeAblationUnitTests(unittest.TestCase):
    def test_lead_named_historical_features_are_not_false_positive_leakage(self) -> None:
        safe_features = [
            "t_network_market_lead_beta_60",
            "t_network_peer_lead_beta_60",
            "t_peer_lead_lag_1",
            "t_peer_lead_lag_3",
        ]
        self.assertTrue(all(leakage_reason(feature) is None for feature in safe_features))
        self.assertIsNotNone(leakage_reason("future_lead_return"))

    def test_train_validation_direction_alignment(self) -> None:
        features = ["f_good", "f_bad"]
        support = pd.DataFrame(
            {
                "feature": features,
                "selection_direction_source": ["pearson", "pearson"],
                "selection_support_metric": ["validation_target_pearson"] * 2,
            }
        )
        records = []
        good_train = [0.2, 0.3, 0.1, 0.2, -0.1]
        good_valid = [0.1, 0.2, 0.2, 0.1, 0.1]
        bad_train = [-0.2] * 5
        bad_valid = [0.2] * 5
        for fold in range(5):
            records.append(
                {
                    "outer_fold": fold,
                    "feature": "f_good",
                    "train_target_pearson": good_train[fold],
                    "validation_target_pearson": good_valid[fold],
                    "train_target_spearman": good_train[fold],
                    "validation_target_spearman": good_valid[fold],
                    "train_target_within_date_pearson": good_train[fold],
                    "validation_target_within_date_pearson": good_valid[fold],
                    "train_target_within_ticker_pearson": good_train[fold],
                    "validation_target_within_ticker_pearson": good_valid[fold],
                }
            )
            records.append(
                {
                    "outer_fold": fold,
                    "feature": "f_bad",
                    "train_target_pearson": bad_train[fold],
                    "validation_target_pearson": bad_valid[fold],
                    "train_target_spearman": bad_train[fold],
                    "validation_target_spearman": bad_valid[fold],
                    "train_target_within_date_pearson": bad_train[fold],
                    "validation_target_within_date_pearson": bad_valid[fold],
                    "train_target_within_ticker_pearson": bad_train[fold],
                    "validation_target_within_ticker_pearson": bad_valid[fold],
                }
            )
        result = compute_train_validation_alignment(
            features,
            support,
            pd.DataFrame(records),
            [0, 1, 2, 3, 4],
            minimum_match_ratio=0.8,
            minimum_train_abs_mean=0.0,
        ).set_index("feature")
        self.assertTrue(bool(result.at["f_good", "train_validation_aligned"]))
        self.assertAlmostEqual(float(result.at["f_good", "train_validation_fold_sign_match_ratio"]), 0.8)
        self.assertFalse(bool(result.at["f_bad", "train_validation_aligned"]))

    def test_metric_consistent_direction_classes(self) -> None:
        self.assertEqual(classify_direction_relation(0.08, 0.001, 0.03, 0.015), "SURGE_SPECIFIC")
        self.assertEqual(classify_direction_relation(-0.08, 0.06, 0.03, 0.015), "OPPOSITE_DIRECTION")
        self.assertEqual(classify_direction_relation(0.08, 0.06, 0.03, 0.015), "COMMON_LARGE_MOVE")
        self.assertEqual(classify_direction_relation(0.01, -0.2, 0.03, 0.015), "WEAK_OR_UNCLEAR")

    def test_daily_top_fraction_selects_at_least_one_per_date(self) -> None:
        target = np.array([1, 0, 0, 0, 1, 0], dtype=np.uint8)
        scores = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])
        dates = np.array(["2026-01-02"] * 3 + ["2026-01-05"] * 3, dtype="datetime64[D]")
        result = daily_top_fraction_metrics(target, scores, dates, 0.03)
        self.assertEqual(result["alert_rows"], 2)
        self.assertEqual(result["date_count"], 2)

    def test_ablation_utility_sign(self) -> None:
        rows = [
            {
                "backend": "lightgbm_cpu",
                "fold_id": 0,
                "seed": 17,
                "test_type": "baseline",
                "pr_auc": 0.30,
                "roc_auc": 0.65,
                "pr_auc_lift": 2.0,
                "brier": 0.10,
                "logloss": 0.40,
                "ece_10": 0.02,
            },
            {
                "backend": "lightgbm_cpu",
                "fold_id": 0,
                "seed": 17,
                "test_type": "single_feature_loo",
                "condition_id": "LOO::f1",
                "pr_auc": 0.25,
                "roc_auc": 0.60,
                "pr_auc_lift": 1.7,
                "brier": 0.12,
                "logloss": 0.45,
                "ece_10": 0.03,
            },
        ]
        paired = build_paired_deltas(pd.DataFrame(rows), [0.03])
        self.assertAlmostEqual(float(paired.iloc[0]["utility_pr_auc"]), 0.05)
        self.assertAlmostEqual(float(paired.iloc[0]["utility_brier"]), 0.02)
        self.assertAlmostEqual(float(paired.iloc[0]["utility_logloss"]), 0.05)


class SurgeAblationEndToEndTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, Path, Path, Path]:
        rng = np.random.default_rng(20260809)
        dates = pd.bdate_range("2025-01-02", periods=130)
        tickers = ["A", "B", "C", "D"]
        rows = []
        for ticker_index, ticker in enumerate(tickers):
            latent = rng.normal(size=len(dates))
            for index, date in enumerate(dates):
                f_signal = latent[index] + 0.1 * ticker_index
                f_inverse = -f_signal + rng.normal(scale=0.2)
                f_noise = rng.normal()
                f_common = abs(latent[index]) + rng.normal(scale=0.1)
                probability = 1.0 / (1.0 + np.exp(-(0.9 * f_signal + 0.2 * f_common - 0.2)))
                target = int(rng.random() < probability)
                crash_probability = 1.0 / (1.0 + np.exp(-(-0.8 * f_signal + 0.1 * f_common)))
                crash = int(rng.random() < crash_probability)
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "industry_name": "sector1" if ticker in {"A", "B"} else "sector2",
                        "market": "KOSPI",
                        "bucket": "b1",
                        "label_abs_crash_20": crash,
                        "sealed_do_not_train_or_tune": 0,
                        "f_signal": f_signal,
                        "f_inverse": f_inverse,
                        "f_noise": f_noise,
                        "f_common": f_common,
                        "target": target,
                    }
                )
        data = pd.DataFrame(rows).sort_values(["date", "ticker"], kind="mergesort").reset_index(drop=True)
        data.insert(0, "source_row_id", np.arange(len(data), dtype=np.int64))
        target = data[["source_row_id", "date", "ticker", "target"]].rename(columns={"target": "label_abs_surge_3d_5pct"})
        target["target_valid"] = True
        dataset_path = root / "data.csv"
        target_path = root / "target.csv"
        data.drop(columns=["target"]).to_csv(dataset_path, index=False)
        target.to_csv(target_path, index=False)

        fold_payload = [
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
                "train_end": str(dates[99].date()),
                "purge_start": str(dates[100].date()),
                "purge_end": str(dates[102].date()),
                "validation_start": str(dates[103].date()),
                "validation_end": str(dates[122].date()),
            },
        ]
        folds_path = root / "folds.json"
        folds_path.write_text(json.dumps(fold_payload), encoding="utf-8")

        corr_dir = root / "correlation"
        corr_dir.mkdir()
        features = ["f_signal", "f_inverse", "f_noise", "f_common"]
        corr_records = []
        joined = data.merge(target[["source_row_id", "label_abs_surge_3d_5pct"]], on="source_row_id")
        for fold in fold_payload:
            for feature in features:
                train_mask = joined["date"].between(fold["train_start"], fold["train_end"])
                val_mask = joined["date"].between(fold["validation_start"], fold["validation_end"])
                train_corr = float(joined.loc[train_mask, [feature, "label_abs_surge_3d_5pct"]].corr().iloc[0, 1])
                val_corr = float(joined.loc[val_mask, [feature, "label_abs_surge_3d_5pct"]].corr().iloc[0, 1])
                corr_records.append(
                    {
                        "outer_fold": fold["fold_id"],
                        "fold_role": "selection" if fold["fold_id"] == 0 else "confirmation",
                        "feature": feature,
                        "train_target_pearson": train_corr,
                        "train_target_spearman": train_corr,
                        "train_target_within_date_pearson": train_corr,
                        "train_target_within_ticker_pearson": train_corr,
                        "validation_target_pearson": val_corr,
                        "validation_target_spearman": val_corr,
                        "validation_target_within_date_pearson": val_corr,
                        "validation_target_within_ticker_pearson": val_corr,
                    }
                )
        pd.DataFrame(corr_records).to_csv(corr_dir / "surge_feature_target_correlation_by_fold.csv", index=False)
        summary = pd.DataFrame(
            {
                "feature": features,
                "selected_for_correlation": True,
                "surge_priority_rank": [1, 2, 4, 3],
                "strict_stable_rank": [1, 2, 4, 3],
                "strict_stable_supported": True,
                "selection_supported": True,
                "selection_all_fold_sign_match": True,
                "selection_min_fold_abs_corr": 0.05,
                "is_primary_representative": [True, True, True, True],
                "primary_cluster_id": [0, 1, 2, 3],
                "primary_cluster_size": 1,
                "group": ["signal", "signal", "noise", "common"],
                "missing_ratio": 0.0,
            }
        )
        summary.to_csv(corr_dir / "surge_feature_correlation_summary.csv", index=False)
        support = pd.DataFrame(
            {
                "feature": features,
                "selection_direction_source": "pearson",
                "selection_support_metric": "validation_target_pearson",
                "strict_stable_supported": True,
                "selection_supported": True,
                "selection_all_fold_sign_match": True,
                "selection_min_fold_abs_corr": 0.05,
            }
        )
        support.to_csv(corr_dir / "surge_support_audit.csv", index=False)
        membership = pd.DataFrame({"feature": features, "P0_ALL_VALID": True, "P1_SELECTION_TOP": True})
        membership.to_csv(corr_dir / "surge_feature_profile_membership.csv", index=False)
        pd.DataFrame(
            {
                "feature": features,
                "cluster_id": [0, 1, 2, 3],
                "is_representative": True,
            }
        ).to_csv(corr_dir / "primary_clusters.csv", index=False)
        manifest = {
            "source_dataset_sha256": sha256_file(dataset_path),
            "target_sha256": sha256_file(target_path),
            "fold_roles": {"selection": [0], "confirmation": [1], "recent_audit": []},
            "feature_count": len(features),
        }
        (corr_dir / "surge_correlation_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return dataset_path, target_path, folds_path, corr_dir

    def test_small_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, target, folds, corr_dir = self.make_fixture(root)
            gate_output = root / "gate"
            gate_parser = build_gate_parser()
            gate_args = gate_parser.parse_args(
                [
                    "--package-root", str(root),
                    "--dataset", str(dataset),
                    "--target-sidecar", str(target),
                    "--folds", str(folds),
                    "--correlation-dir", str(corr_dir),
                    "--output", str(gate_output),
                    "--top-fractions", "0.03",
                    "--profile-min-selection-fold-abs", "0.0",
                    "--profile-min-train-fold-abs", "0.0",
                    "--minimum-purge-trading-days", "3",
                ]
            )
            run_gate(gate_args)
            self.assertEqual(
                json.loads((gate_output / "RUN_STATUS.json").read_text(encoding="utf-8"))["status"],
                "SUCCESS",
            )
            self.assertTrue((gate_output / "surge_feature_univariate_summary.csv").exists())
            self.assertTrue((gate_output / "surge_metric_consistent_directional_map.csv").exists())
            corrected_membership = pd.read_csv(gate_output / "surge_feature_profile_membership_corrected.csv")
            self.assertIn("P2S_SELECTION_TRAIN_ALIGNED", corrected_membership.columns)
            self.assertIn("P2A_TRAIN_ALIGNED_STRICT", corrected_membership.columns)
            self.assertEqual(int(corrected_membership["P2S_SELECTION_TRAIN_ALIGNED"].sum()), 2)

            # Gate resume도 manifest config와 모든 listed artifact hash를 확인해야 한다.
            gate_artifact = gate_output / "surge_feature_univariate_summary.csv"
            gate_artifact_hash = sha256_file(gate_artifact)
            gate_args.resume = True
            run_gate(gate_args)
            self.assertEqual(sha256_file(gate_artifact), gate_artifact_hash)

            # Gate 산출물을 변조하면 resume이 재계산해 원래의 결정론적 결과를 복구해야 한다.
            gate_artifact.write_text(gate_artifact.read_text(encoding="utf-8-sig") + "\nTAMPERED", encoding="utf-8")
            self.assertNotEqual(sha256_file(gate_artifact), gate_artifact_hash)
            run_gate(gate_args)
            self.assertEqual(sha256_file(gate_artifact), gate_artifact_hash)

            # dry-run은 task plan만 만들며 모델 tuning이나 task 학습을 실행하지 않아야 한다.
            parser = build_ablation_parser()
            dry_output = root / "ablation_dry_run"
            dry_args = parser.parse_args(
                [
                    "--package-root", str(root),
                    "--dataset", str(dataset),
                    "--target-sidecar", str(target),
                    "--folds", str(folds),
                    "--correlation-dir", str(corr_dir),
                    "--pre-model-dir", str(gate_output),
                    "--output", str(dry_output),
                    "--stages", "baseline,feature_loo",
                    "--backends", "lightgbm_cpu",
                    "--fold-ids", "0",
                    "--workers", "1",
                    "--threads-per-worker", "1",
                    "--scope-columns", "ticker",
                    "--top-fractions", "0.03",
                    "--minimum-free-disk-gb", "0.001",
                    "--dry-run",
                ]
            )
            run_ablation(dry_args)
            dry_status = json.loads((dry_output / "RUN_STATUS.json").read_text(encoding="utf-8"))
            self.assertEqual(dry_status["status"], "SUCCESS")
            self.assertTrue(bool(dry_status.get("dry_run")))
            self.assertEqual(len(pd.read_csv(dry_output / "task_manifest.csv")), 5)
            self.assertFalse((dry_output / "best_iterations.json").exists())
            self.assertFalse((dry_output / "task_results").exists())

            output = root / "ablation"
            args = parser.parse_args(
                [
                    "--package-root", str(root),
                    "--dataset", str(dataset),
                    "--target-sidecar", str(target),
                    "--folds", str(folds),
                    "--correlation-dir", str(corr_dir),
                    "--pre-model-dir", str(gate_output),
                    "--output", str(output),
                    "--stages", "baseline,feature_loo",
                    "--backends", "lightgbm_cpu",
                    "--workers", "1",
                    "--threads-per-worker", "1",
                    "--scope-columns", "ticker",
                    "--top-fractions", "0.03",
                    "--tuning-windows", "1",
                    "--tuning-step-days", "10",
                    "--inner-validation-days", "10",
                    "--inner-purge-days", "3",
                    "--minimum-inner-train-days", "20",
                    "--max-tuning-rounds", "20",
                    "--early-stopping-rounds", "5",
                    "--minimum-iterations", "5",
                    "--maximum-effective-iterations", "15",
                    "--fallback-iterations", "10",
                    "--lightgbm-params-json", '{"min_data_in_leaf":5,"num_leaves":7,"learning_rate":0.1}',
                    "--bootstrap-repetitions", "100",
                    "--minimum-free-disk-gb", "0.001",
                    "--progress-every", "2",
                ]
            )
            run_ablation(args)
            run_status = json.loads((output / "RUN_STATUS.json").read_text(encoding="utf-8"))
            self.assertEqual(run_status["status"], "SUCCESS")
            summary = pd.read_csv(output / "feature_ablation_summary.csv")
            self.assertEqual(len(summary), 4)
            run_summary_payload = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
            completion = run_summary_payload["completion_audit"]
            self.assertTrue(completion["lightgbm_cpu"]["complete"])
            self.assertEqual(run_summary_payload["primary_completion_audit"]["status"], "COMPLETE")
            self.assertEqual(run_summary_payload["baseline_pairing_audit"]["status"], "COMPLETE")
            primary_audit = json.loads(
                (output / "primary_completion_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(primary_audit["status"], "COMPLETE")
            self.assertEqual(primary_audit["expected_baseline_tasks"], 2)
            self.assertEqual(primary_audit["completed_baseline_tasks"], 2)
            pairing_audit = json.loads(
                (output / "baseline_pairing_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(pairing_audit["status"], "COMPLETE")
            self.assertEqual(pairing_audit["expected_ablated_rows"], 8)
            self.assertEqual(pairing_audit["paired_rows"], 8)
            selection_ranking = pd.read_csv(output / "feature_selection_ranking.csv")
            self.assertNotIn("confirmation_result", selection_ranking.columns)
            self.assertNotIn("recent_result", selection_ranking.columns)
            self.assertNotIn("strict_stable_supported", selection_ranking.columns)
            self.assertIn("selection_mean_pr_auc_utility", selection_ranking.columns)

            # 정상 resume은 기존 task 결과를 그대로 재사용해야 한다.
            task_files = sorted((output / "task_results").rglob("*.json"))
            self.assertEqual(len(task_files), 10)
            protected_task = task_files[0]
            original_hash = sha256_file(protected_task)
            args.resume = True
            run_ablation(args)
            self.assertEqual(sha256_file(protected_task), original_hash)

            # 같은 output에 group stage를 추가해도 기존 baseline/feature LOO task는 재사용한다.
            args.stages = "baseline,feature_loo,group_loo"
            run_ablation(args)
            self.assertEqual(sha256_file(protected_task), original_hash)
            group_summary = pd.read_csv(output / "group_ablation_summary.csv")
            self.assertEqual(len(group_summary), 3)
            self.assertEqual(len(list((output / "task_results").rglob("*.json"))), 16)

            # identity는 그대로 둔 채 metric만 변조해도 payload checksum이 깨져 재실행되어야 한다.
            tampered = json.loads(protected_task.read_text(encoding="utf-8"))
            original_pr_auc = float(tampered["metrics"]["pr_auc"])
            tampered["metrics"]["pr_auc"] = 0.999999
            protected_task.write_text(json.dumps(tampered), encoding="utf-8")
            self.assertNotEqual(sha256_file(protected_task), original_hash)
            self.assertFalse(payload_checksum_is_valid(tampered))
            run_ablation(args)
            repaired = json.loads(protected_task.read_text(encoding="utf-8"))
            self.assertEqual(repaired["identity_hash"], protected_task.stem)
            self.assertEqual(repaired["status"], "completed")
            self.assertTrue(payload_checksum_is_valid(repaired))
            self.assertAlmostEqual(float(repaired["metrics"]["pr_auc"]), original_pr_auc, places=12)

            # 현재 run에 속하지 않는 stale JSON은 집계 결과에 섞이면 안 된다.
            stale_path = output / "task_results" / "lightgbm_cpu" / "feature_loo" / "stale.json"
            stale_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "identity_hash": "stale",
                        "test_type": "single_feature_loo",
                        "representative_feature": "not_a_real_feature",
                        "metrics": {"pr_auc": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            run_ablation(args)
            resumed_summary = pd.read_csv(output / "feature_ablation_summary.csv")
            self.assertEqual(len(resumed_summary), 4)
            self.assertNotIn("not_a_real_feature", resumed_summary["representative_feature"].astype(str).tolist())


if __name__ == "__main__":
    unittest.main(verbosity=2)
