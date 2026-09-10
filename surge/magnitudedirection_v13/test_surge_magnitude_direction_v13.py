from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from surge_v13_data import (
    ABSpec,
    DIRECTION_COLUMN,
    MOVE_COLUMN,
    TARGET_COLUMN,
    build_preexposed_directed_features,
    build_self_state_features,
    build_v11_lag0_features,
    classify_feature,
    date_cross_sectional_rank,
    derive_future_path_labels_from_history,
    fit_ab_state,
    transform_ab_evidence,
)
from run_surge_magnitude_direction_v13 import (
    audit_base_oof_scope,
    ensemble_predictions,
    recalibrate_ensemble_predictions,
    runtime_data_fingerprint,
)

from surge_v13_models import (
    StageSpec,
    apply_policy,
    combine_stage_probabilities,
    discover_ab_specs_v13,
    fit_binary_model,
    match_direction_training_rows,
    past_oof_base_rank,
    predict_binary_model,
    recalibrate_policy_from_prior_oof,
    rank_direction_features,
    rank_move_features,
    select_frozen_policy,
)


def make_history(tickers: list[str], days: int = 80, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    row_id = 0
    for ti, ticker in enumerate(tickers):
        ret = rng.normal(0, 0.012, days)
        ret[15 + ti] = 0.065
        ret[33 + ti] = -0.070
        ret[52 + ti] = 0.058
        for d, r in zip(dates, ret):
            rows.append({"source_row_id": row_id, "date": d, "ticker": ticker, "t_price_ret_1": r})
            row_id += 1
    return pd.DataFrame(rows)


def target_valid_from_history(history: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, grp in history.groupby("ticker", sort=False):
        grp = grp.sort_values("date").copy()
        r = grp["t_price_ret_1"].to_numpy(float)
        c = []
        labels = []
        bests = []
        for i in range(len(grp) - 3):
            path = [r[i + 1], (1 + r[i + 1]) * (1 + r[i + 2]) - 1, (1 + r[i + 1]) * (1 + r[i + 2]) * (1 + r[i + 3]) - 1]
            best = max(path)
            labels.append(int(best >= 0.05))
            bests.append(best)
        keep = grp.iloc[:-3].copy()
        keep[TARGET_COLUMN] = labels
        keep["best_forward_return_3d"] = bests
        keep["market"] = "KOSPI"
        keep["bucket"] = "B1"
        parts.append(keep)
    return pd.concat(parts, ignore_index=True)


class V13Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = make_history(["000001", "000002", "000003"], 90)
        self.frame = target_valid_from_history(self.history)
        self.frame, self.audit = derive_future_path_labels_from_history(self.frame, self.history)
        self.frame["row_index"] = np.arange(len(self.frame))

    def test_full_history_recovers_last_target_valid_paths(self) -> None:
        self.assertEqual(self.audit["future_path_invalid_rows"], 0)
        self.assertEqual(len(self.frame), self.audit["future_path_valid_rows"])
        self.assertLessEqual(self.audit["official_target_mismatch_rate"], 1e-12)
        self.assertEqual(self.audit["history_join_missing_rows"], 0)
        self.assertEqual(self.audit["history_metadata_mismatches"], 0)

    def test_history_source_row_metadata_mismatch_fails(self) -> None:
        broken = self.history.copy()
        broken.loc[broken.index[0], "ticker"] = "999999"
        with self.assertRaises(RuntimeError):
            derive_future_path_labels_from_history(target_valid_from_history(self.history), broken)

    def test_base_oof_scope_requires_exact_rows(self) -> None:
        frame = self.frame.iloc[:24].copy().reset_index(drop=True)
        frame["row_index"] = np.arange(len(frame), dtype=np.int64)
        fold_index = {0: {"validation": np.arange(0, 12, dtype=np.int64)}, 1: {"validation": np.arange(12, 24, dtype=np.int64)}}
        folds = [type("Fold", (), {"fold_id": 0})(), type("Fold", (), {"fold_id": 1})()]
        rows = []
        for fold in folds:
            idx = fold_index[fold.fold_id]["validation"]
            part = frame.iloc[idx]
            for _, row in part.iterrows():
                rows.append({
                    "fold_id": fold.fold_id, "row_index": int(row["row_index"]),
                    "source_row_id": int(row["source_row_id"]), "ticker": str(row["ticker"]),
                    "base_score_raw": 0.5,
                })
        base = pd.DataFrame(rows)
        audit, coverage = audit_base_oof_scope(frame, fold_index, folds, base)
        self.assertTrue(audit["complete"])
        self.assertEqual(audit["missing_expected_rows"], 0)
        self.assertEqual(int(coverage["missing_rows"].sum()), 0)
        base_nan = base.copy()
        base_nan.loc[base_nan.index[-1], "base_score_raw"] = np.nan
        audit_nan, _ = audit_base_oof_scope(frame, fold_index, folds, base_nan)
        self.assertTrue(audit_nan["complete"])
        self.assertFalse(audit_nan["finite_score_complete"])
        self.assertEqual(audit_nan["nonfinite_base_score_rows"], 1)
        audit_missing, _ = audit_base_oof_scope(frame, fold_index, folds, base.iloc[:-1].copy())
        self.assertFalse(audit_missing["complete"])
        self.assertEqual(audit_missing["missing_expected_rows"], 1)

    def test_ab_transform_direction(self) -> None:
        self.frame["node_a"] = np.linspace(-2, 2, len(self.frame))
        specs = {"000001": [ABSpec("000001", "node_a", "raw_a", 1, 1.0, 1)]}
        state = fit_ab_state(self.frame.iloc[:120], specs, max_slots=2)
        transformed = transform_ab_evidence(self.frame, state)
        mask = self.frame["ticker"].eq("000001")
        self.assertGreater(transformed.loc[mask, "ab_weighted_mean"].iloc[-1], transformed.loc[mask, "ab_weighted_mean"].iloc[0])
        self.assertTrue((transformed.loc[~mask, "ab_evidence_count"] == 0).all())

    def test_lag0_features_and_directed_isolation(self) -> None:
        edges = pd.DataFrame({
            "ticker_a": ["000001"], "ticker_b": ["000002"], "directed_lag": [0],
            "discovery_best_lag_correlation": [0.4], "maxstat_q_value": [0.05],
        })
        features, manifest = build_v11_lag0_features(self.frame, edges)
        self.assertEqual(len(manifest), 1)
        self.assertTrue(any(c.startswith("v11_lag0_") for c in features.columns))
        self.assertTrue((features.loc[self.frame["ticker"].eq("000003"), "v11_lag0_peer_count"] == 0).all())

        directed = pd.DataFrame({
            "leader": ["000001"], "follower": ["000003"], "directed_lag": [2],
            "discovery_best_lag_correlation": [0.25],
        })
        pre, pre_manifest = build_preexposed_directed_features(self.frame, directed)
        self.assertEqual(len(pre_manifest), 1)
        self.assertEqual(int(pre_manifest.iloc[0]["available_horizon_components"]), 2)
        self.assertFalse(bool(pre_manifest.iloc[0]["eligible_for_primary_champion"]))
        self.assertTrue(all(c.startswith("v11_preexp_") for c in pre.columns))
        follower_rows = self.frame.loc[self.frame["ticker"].eq("000003")].sort_values("date")
        selected = follower_rows.iloc[10]
        leader = self.frame.loc[self.frame["ticker"].eq("000001")].set_index("date")["return_1d"].sort_index()
        date = pd.Timestamp(selected["date"])
        loc = leader.index.get_loc(date)
        expected = float(np.mean([leader.iloc[loc - 1], leader.iloc[loc]]))
        actual = float(pre.loc[selected.name, "v11_preexp_aligned_mean"])
        self.assertAlmostEqual(actual, expected, places=12)

    def test_self_features_are_past_only(self) -> None:
        before = build_self_state_features(self.frame)
        changed = self.frame.copy()
        last_idx = changed.groupby("ticker").tail(1).index
        changed.loc[last_idx, "return_1d"] = 0.9
        after = build_self_state_features(changed)
        cutoff = changed.groupby("ticker").tail(2).index
        stable_rows = changed.index.difference(cutoff)
        self.assertTrue(np.allclose(before.loc[stable_rows], after.loc[stable_rows], atol=1e-12))


    def test_feature_family_blocks_magnitude_shortcuts_but_keeps_signed_skew(self) -> None:
        self.assertEqual(classify_feature("u_crypto_btc_vol_20"), "magnitude")
        self.assertEqual(classify_feature("u_tailnet_density_corr60"), "magnitude")
        self.assertEqual(classify_feature("t_micro_zero_return_ratio_20"), "magnitude")
        self.assertEqual(classify_feature("t_micro_high_low_pct"), "magnitude")
        self.assertEqual(classify_feature("t_taildep_idio_skew_60"), "direction")
        self.assertEqual(classify_feature("t_finflow_foreign_sum_60__ticker_rank60"), "direction")

    def test_rankings_penalize_magnitude_shortcut(self) -> None:
        rng = np.random.default_rng(9)
        n = 900
        fold = np.repeat([0, 1, 2], n // 3)
        move = rng.binomial(1, 0.35, n)
        direction = np.where(move == 1, rng.binomial(1, 0.5, n), -1)
        mag = move + rng.normal(0, 0.25, n)
        dir_signal = np.where(direction == 1, 1.0, -1.0) + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"fold_id": fold, MOVE_COLUMN: move, DIRECTION_COLUMN: direction, "t_vol_realized_20": mag, "t_finflow_foreign_sum_20": dir_signal})
        move_rank = rank_move_features(df, ["t_vol_realized_20", "t_finflow_foreign_sum_20"], minimum_rows=30)
        dir_rank = rank_direction_features(df, ["t_vol_realized_20", "t_finflow_foreign_sum_20"], minimum_rows=20)
        self.assertEqual(move_rank.iloc[0]["feature"], "t_vol_realized_20")
        self.assertIn("t_finflow_foreign_sum_20", dir_rank["feature"].tolist())
        self.assertNotIn("t_vol_realized_20", dir_rank["feature"].tolist())

    def test_matching_balances_direction_and_magnitude(self) -> None:
        rng = np.random.default_rng(4)
        n = 300
        df = pd.DataFrame({
            "date": pd.bdate_range("2023-01-02", periods=n),
            "ticker": np.where(np.arange(n) % 2, "000001", "000002"),
            "market": "KOSPI", "bucket": np.where(np.arange(n) % 3, "B1", "B2"),
            MOVE_COLUMN: 1, DIRECTION_COLUMN: np.arange(n) % 2,
            "future_abs_excursion_3d": 0.05 + rng.uniform(0, 0.08, n),
            "f": rng.normal(size=n),
        })
        matched, manifest = match_direction_training_rows(df, max_negative_reuse=2, seed=1)
        counts = matched[DIRECTION_COLUMN].value_counts()
        self.assertEqual(int(counts[0]), int(counts[1]))
        self.assertLessEqual(int(manifest["negative_reuse_after_match"].max()), 2)
        self.assertLess(float(manifest["abs_excursion_difference"].median()), 0.02)

    def test_cpu_models_predict_probabilities(self) -> None:
        rng = np.random.default_rng(2)
        n = 260
        x = rng.normal(size=n)
        y = (x + rng.normal(scale=0.5, size=n) > 0).astype(int)
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=n), "ticker": "000001", "market": "KOSPI", "bucket": "B1", "x": x, "y": y})
        for backend in ["xgb", "logit"]:
            spec = StageSpec(backend=backend, feature_k=1, n_estimators=20, learning_rate=0.1, max_depth=2)
            bundle = fit_binary_model(df, ["x"], "y", spec, sample_weight=np.ones(n), seed=1, cpu_threads=2, use_gpu=False, ticker_onehot=False, market_bucket_onehot=False)
            pred = predict_binary_model(bundle, df)
            self.assertTrue(np.isfinite(pred).all())
            self.assertGreater(float(np.corrcoef(pred, y)[0, 1]), 0.5)

    def test_logit_supports_categorical_onehot(self) -> None:
        rng = np.random.default_rng(22)
        n = 240
        x = rng.normal(size=n)
        ticker = np.where(np.arange(n) % 3 == 0, "000001", np.where(np.arange(n) % 3 == 1, "000002", "000003"))
        market = np.where(np.arange(n) % 2 == 0, "KOSPI", "KOSDAQ")
        bucket = np.where(np.arange(n) % 4 < 2, "B1", "B2")
        y = (x + (ticker == "000001") * 0.5 + rng.normal(scale=0.6, size=n) > 0).astype(int)
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=n), "ticker": ticker,
            "market": market, "bucket": bucket, "x": x, "y": y,
        })
        spec = StageSpec(backend="logit", feature_k=1, n_estimators=1, learning_rate=0.0, max_depth=0)
        bundle = fit_binary_model(
            df, ["x"], "y", spec, sample_weight=np.ones(n), seed=1, cpu_threads=2,
            use_gpu=False, ticker_onehot=True, market_bucket_onehot=True,
        )
        pred = predict_binary_model(bundle, df)
        self.assertTrue(np.isfinite(pred).all())
        self.assertGreater(float(np.corrcoef(pred, y)[0, 1]), 0.4)

    def test_policy_frozen_across_folds(self) -> None:
        rng = np.random.default_rng(3)
        rows = []
        for fold in [0, 1, 2]:
            score = rng.uniform(size=120)
            y = (score > 0.65).astype(int)
            rows.append(pd.DataFrame({"fold_id": fold, TARGET_COLUMN: y, "policy_score": score}))
        pred = pd.concat(rows, ignore_index=True)
        policy = select_frozen_policy(pred, discovery_folds=[0, 1, 2], minimum_alerts=20, target_precision=0.7)
        metrics = apply_policy(pred, policy.threshold)
        self.assertEqual(metrics["threshold"].nunique(), 1)
        self.assertTrue(policy.safe_on_discovery)

    def test_base_rank_uses_only_prior_folds(self) -> None:
        base = pd.DataFrame({
            "fold_id": [0] * 40 + [1] * 40,
            "ticker": ["000001"] * 80,
            "base_score_raw": np.r_[np.linspace(0, 1, 40), np.linspace(100, 200, 40)],
        })
        valid = pd.DataFrame({"ticker": ["000001"], "base_score_raw": [0.5]})
        rank = past_oof_base_rank(base, valid, fold_id=1)
        self.assertAlmostEqual(float(rank[0]), 0.5, delta=0.05)

    def test_stage_policy_recalibration_uses_prior_oof_only(self) -> None:
        prior = pd.DataFrame({
            "ticker": ["000001"] * 40 + ["000002"] * 40,
            "stage_probability": np.r_[np.linspace(0.05, 0.85, 40), np.linspace(0.10, 0.90, 40)],
        })
        current = pd.DataFrame({
            "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "ticker": ["000001", "000002"],
            "stage_probability": [0.45, 0.82],
            "base_past_rank": [0.5, 0.5],
        })
        policy, hist, date_rank, source = recalibrate_policy_from_prior_oof(
            prior, current, base_rank_blend=0.0, fallback_historical_rank=[0.0, 0.0]
        )
        self.assertEqual(source, "EARLIER_OOF")
        self.assertGreater(hist[1], hist[0])
        self.assertGreater(date_rank[1], date_rank[0])
        self.assertGreater(policy[1], policy[0])
        # Current labels are intentionally absent: the function must not need them.
        self.assertTrue(np.isfinite(policy).all())

    def test_discovery_ab_specs_exclude_magnitude_family(self) -> None:
        rng = np.random.default_rng(18)
        rows = []
        for fold in [0, 1, 2]:
            for ticker in ["000001", "000002"]:
                for i in range(80):
                    y = int((i + fold) % 3 == 0)
                    rows.append({
                        "ticker": ticker,
                        "fold_id": fold,
                        "base_score_raw": 0.8 + 0.001 * i,
                        TARGET_COLUMN: y,
                        "t_vol_realized_20": y + rng.normal(0, 0.05),
                        "t_finflow_foreign_sum_20": y + rng.normal(0, 0.15),
                    })
        discovery = pd.DataFrame(rows)
        specs, ranking = discover_ab_specs_v13(
            discovery,
            ["t_vol_realized_20", "t_finflow_foreign_sum_20"],
            candidate_quantile=0.50,
            max_features_per_ticker=2,
            minimum_candidates=20,
            minimum_class=4,
        )
        self.assertTrue(specs)
        selected = {spec.node_id for values in specs.values() for spec in values}
        self.assertIn("t_finflow_foreign_sum_20", selected)
        self.assertNotIn("t_vol_realized_20", selected)
        self.assertNotIn("magnitude", ranking["family"].astype(str).tolist())

    def test_date_rank_index_independent(self) -> None:
        df = pd.DataFrame({"date": ["2024-01-01"] * 3, "s": [2.0, 1.0, 3.0]}, index=[10, 20, 30])
        ranks = date_cross_sectional_rank(df, "s")
        self.assertTrue(np.allclose(ranks, [2 / 3, 1 / 3, 1.0]))



    def test_runtime_cache_fingerprint_binds_scope_labels_and_feature_identity(self) -> None:
        base = self.frame.iloc[:30].copy()
        first = runtime_data_fingerprint(base, ["f1", "f2"])
        self.assertEqual(first, runtime_data_fingerprint(base.copy(), ["f1", "f2"]))
        changed_label = base.copy()
        changed_label.loc[changed_label.index[0], TARGET_COLUMN] = 1 - int(changed_label.loc[changed_label.index[0], TARGET_COLUMN])
        self.assertNotEqual(first, runtime_data_fingerprint(changed_label, ["f1", "f2"]))
        self.assertNotEqual(first, runtime_data_fingerprint(base, ["f2", "f1"]))

    def test_seed_ensemble_preserves_stage_decomposition_and_prior_only_recalibration(self) -> None:
        rows = []
        for fold_id in [0, 1]:
            for i in range(40):
                rows.append({
                    "row_index": fold_id * 40 + i,
                    "source_row_id": 1000 + fold_id * 40 + i,
                    "date": pd.Timestamp("2024-01-02") + pd.offsets.BDay(fold_id * 40 + i),
                    "ticker": "000001" if i % 2 == 0 else "000002",
                    "market": "KOSPI",
                    "bucket": "B1",
                    TARGET_COLUMN: int(i % 3 == 0),
                    MOVE_COLUMN: int(i % 2 == 0),
                    DIRECTION_COLUMN: int(i % 4 == 0),
                    "fold_id": fold_id,
                    "base_score_raw": 0.25 + 0.005 * i,
                    "base_past_rank": 0.5,
                    "stage_historical_rank": 0.5,
                    "stage_date_rank": 0.5,
                    "policy_score": 0.5,
                })
        base = pd.DataFrame(rows)
        first = base.copy()
        second = base.copy()
        first["p_move"] = np.linspace(0.20, 0.80, len(first))
        first["p_up_given_move"] = np.linspace(0.30, 0.70, len(first))
        second["p_move"] = np.linspace(0.40, 0.90, len(second))
        second["p_up_given_move"] = np.linspace(0.20, 0.60, len(second))
        first["stage_probability"] = first["p_move"] * first["p_up_given_move"]
        second["stage_probability"] = second["p_move"] * second["p_up_given_move"]
        ens = ensemble_predictions([first, second])
        expected = ((first["p_move"] + second["p_move"]) / 2.0) * (
            (first["p_up_given_move"] + second["p_up_given_move"]) / 2.0
        )
        self.assertTrue(np.allclose(ens["stage_probability"], expected))
        average_products = (first["stage_probability"] + second["stage_probability"]) / 2.0
        self.assertGreater(float(np.max(np.abs(expected - average_products))), 1e-6)
        recalibrated = recalibrate_ensemble_predictions(ens, base_rank_blend=0.0)
        sources = recalibrated.groupby("fold_id")["policy_calibration_source"].first().to_dict()
        self.assertNotEqual(sources[0], "EARLIER_OOF")
        self.assertEqual(sources[1], "EARLIER_OOF")
        self.assertTrue(np.isfinite(recalibrated["policy_score"]).all())

    def test_stage_probability_product(self) -> None:
        a = np.array([0.2, 0.5, 0.9])
        b = np.array([0.4, 0.6, 0.7])
        self.assertTrue(np.allclose(combine_stage_probabilities(a, b), a * b))


if __name__ == "__main__":
    unittest.main(verbosity=2)
