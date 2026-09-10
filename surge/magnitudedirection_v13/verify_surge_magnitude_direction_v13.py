from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

REQUIRED_FILES = [
    "RUN_STATUS.json", "DATA_AUDIT_V13.json", "TARGET_RECONSTRUCTION_AUDIT_V13.json",
    "FINAL_RECOMMENDATION_V13.json", "V13_CHAMPION_CONFIG.json", "V13_FROZEN_POLICY.json",
    "V13_CONFIG_SCREENING.csv", "V13_ROBUST_CONFIGS.csv", "V13_DEVELOPMENT_CONFIG_VALIDATION.csv",
    "V13_MOVE_FEATURE_RANKING_DISCOVERY.csv",
    "V13_DIRECTION_PURITY_RANKING_DISCOVERY.csv", "V13_DISCOVERY_AB_RANKING.csv",
    "V13_AB_SELECTION_MANIFEST.csv", "V13_FEATURE_FAMILY_AUDIT.csv",
    "V13_V11_LAG0_EDGE_MANIFEST.csv", "V13_V11_DIRECTED_PREEXPOSED_MANIFEST.csv",
    "LEAKAGE_CONTRACT_V13.json", "v13_oof_predictions.csv.gz", "v13_stage_metrics_by_fold.csv",
    "v13_frozen_policy_by_fold.csv", "v13_oracle_topk_by_fold.csv",
    "v10_2_base_matched_by_fold.csv", "v13_ticker_metrics_by_fold.csv",
    "v13_policy_bootstrap_stability.csv", "V10_2_BASE_OOF_COVERAGE_V13.csv",
    "V10_2_BASE_OOF_SCOPE_AUDIT_V13.json",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CrashWatch Surge V13 outputs and leakage contracts.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-synthetic", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()
    root = Path(args.output).expanduser().resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = "") -> None:
        checks.append({"check": name, "passed": bool(condition), "detail": detail})

    for name in REQUIRED_FILES:
        check(f"file_exists:{name}", (root / name).exists(), str(root / name))
    missing = [x for x in REQUIRED_FILES if not (root / x).exists()]
    if missing:
        payload = {"status": "FAIL", "passed": 0, "failed": len(missing), "checks": checks}
        (root / "VERIFIER_RESULTS_V13.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        raise SystemExit(1)

    audit = load_json(root / "DATA_AUDIT_V13.json")
    target_audit = load_json(root / "TARGET_RECONSTRUCTION_AUDIT_V13.json")
    recommendation = load_json(root / "FINAL_RECOMMENDATION_V13.json")
    champion = load_json(root / "V13_CHAMPION_CONFIG.json")
    policy = load_json(root / "V13_FROZEN_POLICY.json")
    contract = load_json(root / "LEAKAGE_CONTRACT_V13.json")
    base_scope = load_json(root / "V10_2_BASE_OOF_SCOPE_AUDIT_V13.json")
    pred = pd.read_csv(root / "v13_oof_predictions.csv.gz", dtype={"ticker": str})
    stages = pd.read_csv(root / "v13_stage_metrics_by_fold.csv")
    policy_metrics = pd.read_csv(root / "v13_frozen_policy_by_fold.csv")
    robust = pd.read_csv(root / "V13_ROBUST_CONFIGS.csv")
    development = pd.read_csv(root / "V13_DEVELOPMENT_CONFIG_VALIDATION.csv")
    lag0 = pd.read_csv(root / "V13_V11_LAG0_EDGE_MANIFEST.csv", dtype={"ticker_a": str, "ticker_b": str})
    directed = pd.read_csv(root / "V13_V11_DIRECTED_PREEXPOSED_MANIFEST.csv", dtype={"leader": str, "follower": str})
    ab = pd.read_csv(root / "V13_AB_SELECTION_MANIFEST.csv", dtype={"ticker": str})
    family_audit = pd.read_csv(root / "V13_FEATURE_FAMILY_AUDIT.csv")

    expected_tickers = 1 if args.allow_synthetic else 48
    check("full_ticker_scope", int(audit.get("tickers_full", 0)) >= expected_tickers, audit.get("tickers_full"))
    if not args.allow_synthetic:
        check("exact_91775_target_valid_rows", int(audit.get("rows_full_target_valid", 0)) == 91775, audit.get("rows_full_target_valid"))
        check("target_valid_row_scope_exact", audit.get("target_valid_row_scope_exact") is True, audit.get("target_valid_row_scope_exact"))
        check("exact_48_tickers", int(audit.get("tickers_full", 0)) == 48, audit.get("tickers_full"))
        check("ticker_scope_exact", audit.get("ticker_scope_exact") is True, audit.get("ticker_scope_exact"))
        check("exact_439_raw_features", int(audit.get("raw_features", 0)) == 439, audit.get("raw_features"))
        check("exact_48_base_oof_tickers", int(audit.get("base_oof_tickers", 0)) == 48, audit.get("base_oof_tickers"))
    check("six_ticker_filter_disabled", audit.get("six_ticker_probe_filter_applied") is False, audit.get("six_ticker_probe_filter_applied"))
    check("directed_not_primary_audit", audit.get("primary_uses_directed_v11_edges") is False, audit.get("primary_uses_directed_v11_edges"))
    check("target_mismatch_finite", math.isfinite(float(target_audit.get("official_target_mismatch_rate", float("nan")))), target_audit.get("official_target_mismatch_rate"))
    check("target_path_rows_positive", int(target_audit.get("future_path_valid_rows", 0)) > 0, target_audit.get("future_path_valid_rows"))
    if not args.allow_synthetic:
        check("target_path_invalid_zero", int(target_audit.get("future_path_invalid_rows", -1)) == 0, target_audit.get("future_path_invalid_rows"))
    check("history_join_missing_zero", int(target_audit.get("history_join_missing_rows", -1)) == 0, target_audit.get("history_join_missing_rows"))
    check("history_metadata_mismatch_zero", int(target_audit.get("history_metadata_mismatches", -1)) == 0, target_audit.get("history_metadata_mismatches"))
    check("base_oof_scope_complete", audit.get("base_oof_scope_complete") is True and base_scope.get("complete") is True, {"audit": audit.get("base_oof_scope_complete"), "scope": base_scope.get("complete")})
    check("base_oof_missing_zero", int(base_scope.get("missing_expected_rows", -1)) == 0, base_scope.get("missing_expected_rows"))
    check("base_oof_extra_zero", int(base_scope.get("extra_rows", -1)) == 0, base_scope.get("extra_rows"))
    check("base_oof_source_mismatch_zero", int(base_scope.get("source_row_id_mismatches", -1)) == 0, base_scope.get("source_row_id_mismatches"))
    check(
        "base_oof_nonfinite_audited",
        int(base_scope.get("nonfinite_base_score_rows", -1)) >= 0
        and int(audit.get("base_oof_nonfinite_score_rows", -2)) == int(base_scope.get("nonfinite_base_score_rows", -1)),
        {"scope": base_scope.get("nonfinite_base_score_rows"), "audit": audit.get("base_oof_nonfinite_score_rows")},
    )
    check(
        "base_oof_nonfinite_rows_retained_not_filtered",
        "neutral" in str(audit.get("base_oof_nonfinite_policy", "")).lower(),
        audit.get("base_oof_nonfinite_policy"),
    )

    if not lag0.empty:
        lag_values = pd.to_numeric(lag0["directed_lag"], errors="coerce").fillna(0) if "directed_lag" in lag0.columns else pd.Series(0, index=lag0.index, dtype=float)
        check("lag0_manifest_all_lag0", (lag_values == 0).all(), lag_values.tolist()[:10])
        check("lag0_manifest_q_le_0p10", (pd.to_numeric(lag0["maxstat_q_value"], errors="coerce") <= 0.10 + 1e-12).all(), float(pd.to_numeric(lag0["maxstat_q_value"], errors="coerce").max()))
    else:
        check("lag0_manifest_nonempty", args.allow_synthetic, len(lag0))
    if not directed.empty:
        check("directed_preexposed_true", directed["preexposed"].astype(str).str.lower().isin(["true", "1"]).all(), directed.get("preexposed").tolist()[:10])
        check("directed_ineligible_primary", directed["eligible_for_primary_champion"].astype(str).str.lower().isin(["false", "0"]).all(), directed.get("eligible_for_primary_champion").tolist()[:10])
    check("ab_manifest_nonempty", len(ab) > 0, len(ab))
    if not family_audit.empty:
        feature_names = family_audit["feature"].astype(str)
        check("invalid_industry_rank_transform_excluded", not feature_names.str.endswith("__date_industry_rank").any(), int(feature_names.str.endswith("__date_industry_rank").sum()))
        check("full_history_cluster_innovation_excluded", not feature_names.str.endswith("__ticker_cluster_innovation").any(), int(feature_names.str.endswith("__ticker_cluster_innovation").sum()))
        direction_eligible = family_audit.loc[family_audit["eligible_stage2_raw"].astype(str).str.lower().isin(["true", "1", "yes"])]
        check("direction_raw_family_has_no_magnitude", not direction_eligible["family"].astype(str).eq("magnitude").any(), direction_eligible["family"].value_counts().to_dict())

    required_pred_columns = {
        "row_index", "source_row_id", "date", "ticker", "fold_id", "label_abs_surge_3d_5pct",
        "label_abs_move_3d_5pct", "label_up_given_abs_move", "p_move", "p_up_given_move",
        "stage_probability", "policy_score",
    }
    check("prediction_columns", required_pred_columns.issubset(pred.columns), sorted(required_pred_columns - set(pred.columns)))
    keys = ["row_index", "fold_id"]
    check("prediction_unique_row_fold", not pred.duplicated(keys).any(), int(pred.duplicated(keys).sum()))
    fold_ids = sorted(pd.to_numeric(pred["fold_id"], errors="coerce").dropna().astype(int).unique().tolist())
    development_contract = contract.get(
        "development_folds_for_champion_selection",
        contract.get("development_folds_not_used_until_champion_frozen", []),
    )
    expected_folds = sorted(set(contract.get("feature_and_config_selection_folds", []) + development_contract + contract.get("confirmation_folds", []) + contract.get("recent_diagnostic_folds", [])))
    check("all_contract_folds_present", fold_ids == expected_folds, {"actual": fold_ids, "expected": expected_folds})
    if not args.allow_synthetic:
        check("prediction_scope_48_tickers", int(pred["ticker"].nunique()) == 48, int(pred["ticker"].nunique()))
    for column in ["p_move", "p_up_given_move", "stage_probability", "policy_score"]:
        values = pd.to_numeric(pred[column], errors="coerce")
        check(f"finite:{column}", values.notna().all() and np.isfinite(values).all(), int(values.isna().sum()))
        check(f"bounded:{column}", (values >= -1e-12).all() and (values <= 1.0 + 1e-12).all(), [float(values.min()), float(values.max())])
    product = pd.to_numeric(pred["p_move"], errors="coerce") * pd.to_numeric(pred["p_up_given_move"], errors="coerce")
    check("stage_probability_product", np.allclose(product, pd.to_numeric(pred["stage_probability"], errors="coerce"), atol=1e-7), float(np.nanmax(np.abs(product - pred["stage_probability"]))))

    check("base_oof_expected_rows_match_predictions", int(base_scope.get("expected_rows", -1)) == len(pred), {"expected": base_scope.get("expected_rows"), "predictions": len(pred)})
    robust_ids = set(robust["config_id"].astype(str)) if "config_id" in robust.columns else set()
    development_ids = set(development["config_id"].astype(str)) if "config_id" in development.columns else set()
    check("robust_shortlist_nonempty", len(robust_ids) > 0, sorted(robust_ids))
    check("development_shortlist_complete", development_ids == robust_ids, {"robust": sorted(robust_ids), "development": sorted(development_ids)})
    check("stage_metrics_cover_folds", sorted(stages["fold_id"].astype(int).unique().tolist()) == expected_folds, stages["fold_id"].tolist())
    check("policy_metrics_cover_folds", sorted(policy_metrics["fold_id"].astype(int).unique().tolist()) == expected_folds, policy_metrics["fold_id"].tolist())
    thresholds = pd.to_numeric(policy_metrics["threshold"], errors="coerce")
    check("single_frozen_threshold", thresholds.nunique(dropna=True) == 1, thresholds.unique().tolist())
    check("threshold_matches_manifest", np.allclose(thresholds.dropna(), float(policy["threshold"])), {"csv": thresholds.unique().tolist(), "json": policy["threshold"]})
    check("minimum_alerts_not_lowered", int(policy.get("minimum_alerts", 0)) >= 30 or args.allow_synthetic, policy.get("minimum_alerts"))
    check("target_precision_not_lowered", float(policy.get("target_precision", 0.0)) >= 0.70 - 1e-12 or args.allow_synthetic, policy.get("target_precision"))
    check("feature_set_frozen_before_development", champion.get("feature_set_frozen_before_development") is True, champion.get("feature_set_frozen_before_development"))
    check("champion_frozen_before_confirmation", champion.get("champion_frozen_before_confirmation") is True, champion.get("champion_frozen_before_confirmation"))
    champion_seeds = champion.get("primary_ensemble_seeds", [])
    policy_seeds = policy.get("primary_ensemble_seeds", [])
    recommendation_seeds = recommendation.get("primary_ensemble_seeds", [])
    check("primary_seed_ensemble_frozen", bool(champion_seeds) and champion_seeds == policy_seeds == recommendation_seeds, {"champion": champion_seeds, "policy": policy_seeds, "recommendation": recommendation_seeds})
    check("direction_stage_separate", contract.get("stage1_target") == "label_abs_move_3d_5pct" and contract.get("stage2_target") == "label_up_given_abs_move", [contract.get("stage1_target"), contract.get("stage2_target")])
    check("current_fold_base_rank_forbidden", "earlier OOF" in str(contract.get("base_score_calibration", "")), contract.get("base_score_calibration"))
    check("stage_calibration_prior_oof_contract", "earlier config-specific OOF" in str(contract.get("stage_score_calibration", "")), contract.get("stage_score_calibration"))
    if "policy_calibration_source" in pred.columns:
        first_fold = min(fold_ids)
        later = pred.loc[pd.to_numeric(pred["fold_id"], errors="coerce").gt(first_fold), "policy_calibration_source"].astype(str)
        check("later_folds_use_earlier_oof_calibration", later.eq("EARLIER_OOF").all(), later.value_counts().to_dict())
    else:
        check("policy_calibration_source_recorded", False, "missing column")
    if "v11_directed_preexposed_included" in pred.columns:
        directed_flags = pred["v11_directed_preexposed_included"].astype(str).str.lower().isin(["true", "1", "yes"])
        check("primary_predictions_exclude_preexposed_directed", not directed_flags.any(), int(directed_flags.sum()))
    check("production_no_alert", recommendation.get("production_decision") == "NO_ALERT", recommendation.get("production_decision"))
    check("directed_primary_false_recommendation", recommendation.get("primary_findings_to_review", {}).get("v11_directed_edges_primary") is False, recommendation.get("primary_findings_to_review"))

    failed = [x for x in checks if not x["passed"]]
    payload = {"status": "PASS" if not failed else "FAIL", "passed": len(checks) - len(failed), "failed": len(failed), "checks": checks}
    (root / "VERIFIER_RESULTS_V13.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "passed": payload["passed"], "failed": payload["failed"]}, ensure_ascii=False))
    if failed:
        for item in failed:
            print(f"FAIL {item['check']}: {item['detail']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
