from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def check_output(output: Path) -> dict[str, Any]:
    checks = 0
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)

    required = [
        "RUN_STATUS.json",
        "LEADLAG_MANIFEST_V11_1.json",
        "LEADLAG_CORRECTIVE_RECOMMENDATION_V11_1.json",
        "maxstat_pair_results_v11_1.csv",
        "v11_strong_edge_recheck_v11_1.csv",
        "frozen_lag_rolling_v11_1.csv",
        "frozen_lag_rolling_summary_v11_1.csv",
        "V11_TO_V11_1_EDGE_COMPARISON.csv",
        "FROZEN_PORTFOLIO_POLICIES_V11_1.json",
        "TARGET_ALIGNED_FEATURE_MANIFESTS_V11_1.json",
    ]
    for name in required:
        check((output / name).exists(), f"missing required output: {name}")
    if failures:
        return {"status": "FAIL", "checks": checks, "failures": failures}

    status = json.loads((output / "RUN_STATUS.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "LEADLAG_MANIFEST_V11_1.json").read_text(encoding="utf-8"))
    recommendation = json.loads((output / "LEADLAG_CORRECTIVE_RECOMMENDATION_V11_1.json").read_text(encoding="utf-8"))
    policies = json.loads((output / "FROZEN_PORTFOLIO_POLICIES_V11_1.json").read_text(encoding="utf-8"))
    feature_manifests = json.loads((output / "TARGET_ALIGNED_FEATURE_MANIFESTS_V11_1.json").read_text(encoding="utf-8"))

    check(status.get("status") == "SUCCESS", "RUN_STATUS is not SUCCESS")
    check(manifest.get("target") == "label_abs_surge_3d_5pct", "target changed")
    check(manifest.get("fold_contract") == {"discovery": [0, 1, 2], "development": [3, 4], "confirmation": [5, 6], "recent_audit": [7]}, "fold contract changed")
    check(recommendation.get("target_alignment_corrected") is True, "target alignment correction flag missing")
    check(recommendation.get("matched_row_probe") is True, "matched-row probe flag missing")
    check(recommendation.get("rolling_lag_reoptimized") is False, "rolling lag must remain frozen")
    check(recommendation.get("causal_claim") is False, "causal claim must be false")

    maxstat = pd.read_csv(output / "maxstat_pair_results_v11_1.csv", dtype={"ticker_a": str, "ticker_b": str, "leader": str, "follower": str})
    check(maxstat["discovery_best_lag"].between(-5, 5).all(), "discovery lag outside -5..5")
    check(maxstat["directed_lag"].between(0, 5).all(), "directed lag outside 0..5")
    check(maxstat["maxstat_empirical_p"].dropna().between(0, 1).all(), "maxstat p-value outside [0,1]")
    check(maxstat["maxstat_q_value"].dropna().between(0, 1).all(), "maxstat q-value outside [0,1]")
    check(maxstat["permutation_method"].str.contains("max_abs_over_11_lags", regex=False).all(), "max-stat method marker missing")
    corrected = maxstat.loc[maxstat["corrected_discovery_dev_candidate"].astype(bool)]
    alpha = float(manifest.get("parameters", {}).get("maxstat_alpha", 0.10))
    if len(corrected):
        check(corrected["maxstat_q_value"].le(alpha + 1e-12).all(), "corrected edge exceeds maxstat FDR alpha")
        check(corrected["development_same_direction"].astype(bool).all(), "corrected edge changed direction in development")
        check(corrected["development_valid_folds"].ge(2).all(), "corrected edge lacks both development folds")
    else:
        check(True, "no corrected edges is a valid result")

    rolling = pd.read_csv(output / "frozen_lag_rolling_v11_1.csv")
    if len(rolling):
        check(not rolling["lag_reoptimized"].astype(bool).any(), "rolling lag was reoptimized")
        check((rolling["fixed_lag"] > 0).all(), "rolling fixed lag must be positive")
        check((rolling["residual_variant"] == "point_in_time_bucket_residual").all(), "rolling residual variant mismatch")
    else:
        check(True, "empty rolling output allowed if no candidate/reference edge")

    for edge_set, fm in feature_manifests.items():
        check(fm.get("future_offsets_used") is False, f"{edge_set}: future offset flag must be false")
        for edge in fm.get("edges", []):
            mapping = edge.get("alignment", [])
            check(all(int(item["leader_offset_from_forecast_t"]) <= 0 for item in mapping), f"{edge_set}: positive/future leader offset")
            lag = int(edge["directed_lag"])
            expected = [(k, k - lag) for k in range(1, 4) if k - lag <= 0]
            actual = [(int(item["follower_horizon_day"]), int(item["leader_offset_from_forecast_t"])) for item in mapping]
            check(actual == expected, f"{edge_set}: target alignment mapping mismatch for lag={lag}")

    pred_path = output / "matched_probe_predictions_v11_1.csv"
    if pred_path.exists() and pred_path.stat().st_size > 1:
        predictions = pd.read_csv(pred_path, dtype={"ticker": str})
        if len(predictions):
            for (edge_set, ticker, fold_id), part in predictions.groupby(["edge_set", "ticker", "fold_id"], sort=False):
                plus = part.loc[part["profile"].eq("BASE_PLUS_ALIGNED_LEAD")]
                if plus.empty:
                    continue
                base = part.loc[part["profile"].eq("BASE_TICKER")]
                check(set(base["source_row_id"]) == set(plus["source_row_id"]), f"matched support mismatch: {edge_set}/{ticker}/fold{fold_id}")

    champions_path = output / "matched_probe_champions_v11_1.csv"
    if champions_path.exists() and champions_path.stat().st_size > 1:
        champions = pd.read_csv(champions_path, dtype={"ticker": str})
        for row in champions.itertuples(index=False):
            if str(row.champion_profile) == "BASE_PLUS_ALIGNED_LEAD":
                d3 = float(row.fold3_pr_auc_delta_plus_minus_base)
                d4 = float(row.fold4_pr_auc_delta_plus_minus_base)
                check(math.isfinite(d3) and d3 > 0 and math.isfinite(d4) and d4 > 0, f"PLUS champion without wins in both dev folds: {row.ticker}")

    portfolio_path = output / "matched_probe_portfolio_v11_1.csv"
    if portfolio_path.exists() and portfolio_path.stat().st_size > 1:
        portfolio = pd.read_csv(portfolio_path)
        for edge_set, part in portfolio.groupby("edge_set", sort=False):
            thresholds = pd.to_numeric(part["frozen_threshold"], errors="coerce").dropna().unique()
            check(len(thresholds) <= 1, f"portfolio threshold was not frozen: {edge_set}")
            check((part["threshold_selected_on"] == "folds_3_4_only").all(), f"threshold selection source changed: {edge_set}")
            check(part["eligible_tickers_fixed"].nunique() <= 1, f"ticker universe changed across folds: {edge_set}")

    for edge_set, policy in policies.items():
        if isinstance(policy, dict) and math.isfinite(float(policy.get("threshold", math.nan))):
            detail = policy.get("development_fold_detail", [])
            check({int(item["fold_id"]) for item in detail} <= {3, 4}, f"future fold used for threshold selection: {edge_set}")

    return {"status": "PASS" if not failures else "FAIL", "checks": checks, "failures": failures}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/surge_leadlag_v11_1_corrective_validation")
    args = parser.parse_args()
    result = check_output(Path(args.output).resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
