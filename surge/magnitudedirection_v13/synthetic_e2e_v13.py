from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from run_surge_magnitude_direction_v13 import (
    DerivedFoldFeatures,
    RuntimeData,
    runtime_data_fingerprint,
    atomic_csv,
    atomic_json,
    base_matched_metrics,
    ensemble_predictions,
    run_config_seeds,
    ticker_metrics,
)
from surge_v13_data import (
    ABSpec,
    DIRECTION_COLUMN,
    MOVE_COLUMN,
    TARGET_COLUMN,
    build_preexposed_directed_features,
    build_self_state_features,
    build_v11_lag0_features,
    derive_future_path_labels_from_history,
    feature_family_audit,
    serialize_ab_specs,
)
from surge_v13_models import (
    StageSpec,
    V13Config,
    apply_policy,
    fold_score_metrics,
    oracle_top_k,
    discover_ab_specs_v13,
    rank_direction_features,
    rank_move_features,
    select_frozen_policy,
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_synthetic(seed: int = 13) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    rng = np.random.default_rng(seed)
    tickers = [f"{i:06d}" for i in range(1, 7)]
    dates = pd.bdate_range("2021-01-04", periods=560)
    rows = []
    row_id = 0
    common = rng.normal(0, 0.004, len(dates))
    for ti, ticker in enumerate(tickers):
        mag = np.abs(rng.normal(size=len(dates)))
        direction = rng.normal(size=len(dates)) + 0.25 * np.sin(np.arange(len(dates)) / 23 + ti)
        ret = common + rng.normal(0, 0.006, len(dates))
        for t in range(1, len(dates)):
            if mag[t - 1] > 1.05:
                ret[t] += (0.060 if direction[t - 1] > 0 else -0.062) + rng.normal(0, 0.004)
        for t, date in enumerate(dates):
            rows.append({
                "source_row_id": row_id, "date": date, "ticker": ticker,
                "market": "KOSPI" if ti < 4 else "KOSDAQ", "bucket": "PAIR_A" if ti < 3 else "PAIR_B",
                "t_price_ret_1": ret[t],
                "t_vol_realized_20": mag[t] + rng.normal(0, 0.10),
                "t_rangevol_bipower_60": 0.8 * mag[t] + rng.normal(0, 0.12),
                "t_taildep_idio_vol_60": 0.65 * mag[t] + rng.normal(0, 0.15),
                "t_rangevol_garman_klass_20": 0.55 * mag[t] + rng.normal(0, 0.15),
                "t_finflow_foreign_sum_20": direction[t] + rng.normal(0, 0.18),
                "t_price_ma_gap_20": 0.8 * direction[t] + rng.normal(0, 0.20),
                "t_finshort_balance_slope_60": -0.65 * direction[t] + rng.normal(0, 0.22),
                "u_lending_balance_mean_change5": -0.50 * direction[t] + rng.normal(0, 0.25),
                "t_price_ret_5": 0.45 * direction[t] + rng.normal(0, 0.28),
                "t_idio_skew_60": 0.35 * direction[t] + rng.normal(0, 0.30),
                "unknown_move_proxy": 0.5 * mag[t] + rng.normal(0, 0.3),
                "unknown_direction_proxy": 0.5 * direction[t] + rng.normal(0, 0.3),
            })
            row_id += 1
    history = pd.DataFrame(rows)
    parts = []
    for _, grp in history.groupby("ticker", sort=False):
        grp = grp.sort_values("date").copy()
        r = grp["t_price_ret_1"].to_numpy(float)
        labels = []
        bests = []
        for i in range(len(grp) - 3):
            path = [r[i + 1], (1 + r[i + 1]) * (1 + r[i + 2]) - 1, (1 + r[i + 1]) * (1 + r[i + 2]) * (1 + r[i + 3]) - 1]
            labels.append(int(max(path) >= 0.05))
            bests.append(max(path))
        keep = grp.iloc[:-3].copy()
        keep[TARGET_COLUMN] = labels
        keep["best_forward_return_3d"] = bests
        parts.append(keep)
    frame = pd.concat(parts, ignore_index=True)
    frame, _ = derive_future_path_labels_from_history(frame, history, max_mismatch_rate=0.0)
    frame = frame.reset_index(drop=True)
    frame["row_index"] = np.arange(len(frame), dtype=np.int64)
    raw = [
        "t_vol_realized_20", "t_rangevol_bipower_60", "t_taildep_idio_vol_60",
        "t_rangevol_garman_klass_20", "t_finflow_foreign_sum_20", "t_price_ma_gap_20",
        "t_finshort_balance_slope_60", "u_lending_balance_mean_change5", "t_price_ret_5",
        "t_idio_skew_60", "unknown_move_proxy", "unknown_direction_proxy",
    ]
    return frame, history, raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.output:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        temporary = None
    else:
        temporary = tempfile.TemporaryDirectory(prefix="v13_synth_")
        output = Path(temporary.name)

    frame, history, raw_features = make_synthetic()
    starts = [180 + 40 * i for i in range(8)]
    fold_index = {}
    validation_fold_id = np.full(len(frame), -1, dtype=np.int16)
    for fold_id, start in enumerate(starts):
        train_end = start - 1
        valid_end = start + 39
        train_dates = sorted(frame["date"].unique())[: train_end + 1]
        valid_dates = sorted(frame["date"].unique())[start: valid_end + 1]
        train_idx = np.flatnonzero(frame["date"].isin(train_dates).to_numpy())
        valid_idx = np.flatnonzero(frame["date"].isin(valid_dates).to_numpy())
        fold_index[fold_id] = {"train": train_idx, "validation": valid_idx}
        validation_fold_id[valid_idx] = fold_id

    base_rows = []
    for fold_id in range(8):
        idx = fold_index[fold_id]["validation"]
        part = frame.iloc[idx]
        base = sigmoid(0.4 * part["t_vol_realized_20"].to_numpy(float) + 0.25 * part["t_finflow_foreign_sum_20"].to_numpy(float) - 0.4)
        for (_, row), score in zip(part.iterrows(), base):
            base_rows.append({
                "row_index": int(row["row_index"]), "source_row_id": int(row["source_row_id"]),
                "ticker": str(row["ticker"]), "fold_id": fold_id, "base_score_raw": float(score),
            })
    base_oof = pd.DataFrame(base_rows)
    lag0_manifest = pd.DataFrame({
        "ticker_a": ["000001", "000003"], "ticker_b": ["000002", "000004"],
        "directed_lag": [0, 0], "discovery_best_lag_correlation": [0.35, -0.30],
        "maxstat_q_value": [0.05, 0.08],
    })
    lag0_features, lag0_used = build_v11_lag0_features(frame, lag0_manifest)
    self_features = build_self_state_features(frame)
    directed_manifest = pd.DataFrame({
        "leader": ["000001"], "follower": ["000005"], "directed_lag": [3],
        "discovery_best_lag_correlation": [0.22],
    })
    pre_features, pre_used = build_preexposed_directed_features(frame, directed_manifest)
    discovery_parts = []
    for fold_id in [0, 1, 2]:
        part = frame.iloc[fold_index[fold_id]["validation"]].copy()
        part["fold_id"] = fold_id
        part = part.merge(
            base_oof.loc[base_oof["fold_id"].eq(fold_id), ["row_index", "source_row_id", "ticker", "base_score_raw"]],
            on=["row_index", "source_row_id", "ticker"], how="left", validate="one_to_one",
        )
        discovery_parts.append(part)
    discovery = pd.concat(discovery_parts, ignore_index=True)
    specs, ab_ranking = discover_ab_specs_v13(
        discovery, raw_features, candidate_quantile=0.50, max_features_per_ticker=3,
        minimum_candidates=18, minimum_class=3,
    )
    if not specs:
        raise RuntimeError("synthetic discovery A/B reconstruction produced no specs")
    runtime = RuntimeData(
        frame=frame, raw_features=raw_features, candidate_features=raw_features, v10args=argparse.Namespace(), folds=[],
        fold_index=fold_index, validation_fold_id=validation_fold_id, base_oof=base_oof,
        ab_specs=specs, lag0_features=lag0_features, self_features=self_features,
        preexposed_features=pre_features,
        cache_fingerprint=runtime_data_fingerprint(frame, raw_features),
    )
    move_ranking = rank_move_features(discovery, raw_features, minimum_rows=30)
    direction_ranking = rank_direction_features(discovery, raw_features, minimum_rows=20)
    config = V13Config(
        "SYNTH_V13", StageSpec("xgb", 4, 35, 0.08, 3), StageSpec("xgb", 4, 30, 0.08, 3),
        True, True, True, 0.0, 365, 2, ticker_onehot=True, market_bucket_onehot=True,
    )
    run_args = argparse.Namespace(cpu_threads=2, gpu_available=False, resume=False)
    derived_cache: dict[int, DerivedFoldFeatures] = {}
    matching_cache = {}
    pred, diagnostics, move_features, direction_features = run_config_seeds(
        runtime, config, move_ranking, direction_ranking, list(range(8)), [13013],
        args=run_args, output=output, derived_cache=derived_cache, matching_cache=matching_cache,
    )
    policy = select_frozen_policy(pred, discovery_folds=[0, 1, 2], minimum_alerts=5, target_precision=0.60)
    stage_metrics = fold_score_metrics(pred)
    policy_metrics = apply_policy(pred, policy.threshold)

    atomic_json(output / "RUN_STATUS.json", {"schema": "crashwatch_surge_magnitude_direction_v13", "status": "SUCCESS_VERIFIED"})
    atomic_json(output / "DATA_AUDIT_V13.json", {
        "tickers_full": int(frame["ticker"].nunique()), "raw_features": len(raw_features),
        "base_oof_tickers": int(base_oof["ticker"].nunique()),
        "base_oof_scope_complete": True, "base_oof_expected_rows": int(len(pred)),
        "base_oof_missing_expected_rows": 0, "base_oof_extra_rows": 0,
        "base_oof_source_row_id_mismatches": 0, "base_oof_nonfinite_score_rows": 0,
        "base_oof_finite_score_rate": 1.0,
        "base_oof_nonfinite_policy": "retain row; neutral base_past_rank=0.5; never filter model scope",
        "six_ticker_probe_filter_applied": False, "primary_uses_directed_v11_edges": False,
    })
    atomic_json(output / "TARGET_RECONSTRUCTION_AUDIT_V13.json", {
        "official_target_mismatch_rate": 0.0, "future_path_valid_rows": len(frame), "future_path_invalid_rows": 0,
        "history_join_missing_rows": 0, "history_metadata_mismatches": 0,
    })
    atomic_json(output / "V10_2_BASE_OOF_SCOPE_AUDIT_V13.json", {
        "expected_rows": int(len(pred)), "actual_rows": int(len(pred)),
        "missing_expected_rows": 0, "extra_rows": 0, "source_row_id_mismatches": 0,
        "nonfinite_base_score_rows": 0, "expected_tickers": int(frame["ticker"].nunique()),
        "actual_tickers": int(frame["ticker"].nunique()), "complete": True,
        "finite_score_complete": True, "finite_score_rate": 1.0,
    })
    atomic_json(output / "FINAL_RECOMMENDATION_V13.json", {
        "production_decision": "NO_ALERT", "primary_ensemble_seeds": [13013],
        "primary_findings_to_review": {"v11_directed_edges_primary": False},
    })
    atomic_json(output / "V13_CHAMPION_CONFIG.json", {
        "champion": dataclasses.asdict(config), "feature_set_frozen_before_development": True,
        "champion_frozen_before_confirmation": True,
        "champion_selection_role": "SYNTHETIC_DEVELOPMENT_VALIDATION",
        "primary_ensemble_seeds": [13013], "primary_ensemble_seed_count": 1,
        "selected_move_features": move_features, "selected_direction_raw_features": direction_features,
    })
    atomic_json(output / "V13_FROZEN_POLICY.json", {
        "threshold": policy.threshold, "minimum_alerts": 5, "target_precision": 0.60,
        "primary_ensemble_seeds": [13013],
    })
    atomic_json(output / "LEAKAGE_CONTRACT_V13.json", {
        "feature_and_config_selection_folds": [0, 1, 2], "development_folds_for_champion_selection": [3, 4],
        "confirmation_folds": [5, 6], "recent_diagnostic_folds": [7],
        "stage1_target": MOVE_COLUMN, "stage2_target": DIRECTION_COLUMN,
        "base_score_calibration": "earlier OOF folds only",
        "stage_score_calibration": "outer-train reference only for the first requested fold; earlier config-specific OOF folds thereafter",
    })
    atomic_csv(output / "V13_CONFIG_SCREENING.csv", pd.DataFrame([{"config_id": config.config_id}]))
    atomic_csv(output / "V13_ROBUST_CONFIGS.csv", pd.DataFrame([{"config_id": config.config_id}]))
    atomic_csv(output / "V13_DEVELOPMENT_CONFIG_VALIDATION.csv", pd.DataFrame([{
        "config_id": config.config_id, "development_complete": True,
        "development_safe_at_discovery_threshold": False,
    }]))
    atomic_csv(output / "V13_MOVE_FEATURE_RANKING_DISCOVERY.csv", move_ranking)
    atomic_csv(output / "V13_DIRECTION_PURITY_RANKING_DISCOVERY.csv", direction_ranking)
    atomic_csv(output / "V13_AB_SELECTION_MANIFEST.csv", serialize_ab_specs(specs))
    atomic_csv(output / "V13_DISCOVERY_AB_RANKING.csv", ab_ranking)
    atomic_csv(output / "V13_V11_LAG0_EDGE_MANIFEST.csv", lag0_used)
    atomic_csv(output / "V13_V11_DIRECTED_PREEXPOSED_MANIFEST.csv", pre_used)
    synthetic_family_audit = feature_family_audit(raw_features)
    synthetic_family_audit["origin"] = "RAW_439"
    synthetic_family_audit["eligible_stage1"] = ~synthetic_family_audit["family"].isin(["metadata", "direction"])
    synthetic_family_audit["eligible_stage2_raw"] = ~synthetic_family_audit["family"].isin(["metadata", "magnitude"])
    atomic_csv(output / "V13_FEATURE_FAMILY_AUDIT.csv", synthetic_family_audit)
    atomic_csv(output / "v13_oof_predictions.csv.gz", pred, compression="gzip")
    atomic_csv(output / "v13_stage_metrics_by_fold.csv", stage_metrics)
    atomic_csv(output / "v13_frozen_policy_by_fold.csv", policy_metrics)
    atomic_csv(output / "v13_oracle_topk_by_fold.csv", oracle_top_k(pred, minimum_alerts=5))
    atomic_csv(output / "v10_2_base_matched_by_fold.csv", base_matched_metrics(pred))
    atomic_csv(output / "v13_ticker_metrics_by_fold.csv", ticker_metrics(pred))
    atomic_csv(output / "v13_component_ablation.csv", stage_metrics.assign(variant="PRIMARY_FULL"))
    atomic_csv(output / "v13_seed_stability.csv", policy_metrics.assign(seed=13013))
    atomic_csv(output / "v13_policy_bootstrap_stability.csv", pd.DataFrame([{
        "iteration": 0, "minimum_precision": float(policy.minimum_precision),
        "mean_precision": float(policy.mean_precision), "minimum_alerts": int(policy.minimum_alerts_observed),
        "gate_success": bool(policy.safe_on_discovery),
    }]))
    atomic_csv(output / "V10_2_BASE_OOF_COVERAGE_V13.csv", (
        base_oof.groupby(["fold_id", "ticker"], sort=True)
        .agg(rows=("row_index", "size"), finite_base_score=("base_score_raw", "count"))
        .reset_index()
    ))

    verifier = Path(__file__).resolve().parent / "verify_surge_magnitude_direction_v13.py"
    result = subprocess.run([sys.executable, str(verifier), "--output", str(output), "--allow-synthetic"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit(result.returncode)
    print(json.dumps({
        "status": "PASS", "output": str(output), "rows": len(pred),
        "move_features": move_features, "direction_features": direction_features,
        "policy_threshold": policy.threshold,
    }, ensure_ascii=False, indent=2))
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
