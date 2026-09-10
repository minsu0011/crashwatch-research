from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cw7h.utils import atomic_json, read_json


def main() -> None:
    package = Path(__file__).resolve().parent
    project = package.parent
    first_root = project / "crashwatch_ai_data/regime_expert_router_followup_v1"
    final_root = project / "crashwatch_ai_data/regime_locked_p7_override_v1"
    first = read_json(first_root / "RUN_STATUS.json", {})
    final = read_json(final_root / "RUN_STATUS.json", {})
    first_boot = pd.read_csv(first_root / "FOLLOWUP_B4_BOOTSTRAP_SUMMARY.csv")
    final_boot = pd.read_csv(final_root / "FOLLOWUP_B4_BOOTSTRAP_SUMMARY.csv")
    route_selection = pd.read_csv(final_root / "LOCKED_OVERRIDE_ROUTE_SELECTION.csv")
    actual_p7 = route_selection[(~route_selection["disabled"]) & (route_selection["mean_p7_regime_count"] > 0)]
    calibration = final["selected_calibration"]
    b4 = final["B4_diagnostic"]

    def bootstrap_row(frame: pd.DataFrame, metric: str) -> dict:
        row = frame[frame["metric"] == metric].iloc[0]
        return {
            "mean_improvement": float(row["mean_improvement"]),
            "positive_rate": float(row["positive_rate"]),
            "ci95_low": float(row["ci95_low"]),
            "ci95_high": float(row["ci95_high"]),
        }

    report = {
        "status": "FOLLOWUP_SERIES_COMPLETE",
        "scientific_decision": "REJECT_P7_REGIME_OVERRIDE_FOR_NOW_ADOPT_SEPARATE_CALIBRATED_RISK_HEAD",
        "experiment_1_P2XGB_vs_P7LGB": {
            "candidate_count": first["candidates"],
            "selected_route": first["final_mapping"],
            "finding": "all eight regimes collapsed to P2_XGB under temporal robustness constraints",
            "B4_reused_diagnostic": first["B4_diagnostic"],
            "bootstrap": {
                "brier": bootstrap_row(first_boot, "improvement_brier"),
                "logloss": bootstrap_row(first_boot, "improvement_logloss"),
                "pr_auc": bootstrap_row(first_boot, "improvement_pr_auc"),
            },
        },
        "experiment_2_lockedP2_with_P7_override": {
            "route_candidate_count": final["route_candidates"],
            "calibration_candidate_count": final["calibration_candidates"],
            "selected_route": final["selected_route"],
            "actual_nonzero_P7_candidates": int(len(actual_p7)),
            "best_actual_P7_min_score_ratio": (
                float(actual_p7["min_score_ratio_vs_locked"].max()) if len(actual_p7) else None
            ),
            "finding": "every route that actually selected P7 lost composite score in at least one B2/B3 block",
            "selected_risk_calibration": calibration,
            "B4_reused_diagnostic": b4,
            "bootstrap": {
                "brier": bootstrap_row(final_boot, "improvement_brier"),
                "logloss": bootstrap_row(final_boot, "improvement_logloss"),
                "selection_score": bootstrap_row(final_boot, "improvement_selection_score"),
            },
        },
        "recommended_operational_outputs": {
            "ranking_score": "retain locked P2 raw score (0.1 P2_LGB + 0.9 P2_XGB)",
            "alerting": "rank by the raw score; daily top 3%, minimum one ticker",
            "risk_probability": (
                "50% raw locked-P2 score + 50% global Platt(C=0.1) probability; "
                "fit on development OOF only"
            ),
            "P7_policy": "do not switch production predictions to P7 from the current eight broad regimes",
            "research_challenger": "retain P7 and the original regime gate for future fresh-period testing",
        },
        "why_P7_failed_temporal_confirmation": [
            "the apparent P7 winners changed between B2 and B3",
            "B4 contains no CRASH_STRESS dates and only 3-6 dates in several high-volatility regimes",
            "P7 raw probabilities are less calibrated than locked P2",
            "the eight broad regimes do not provide enough stable support for a safe hard override",
        ],
        "next_data_requirement_before_more_gate_complexity": {
            "fresh_dates": "at least 120-180 additional trading dates",
            "per_regime_goal": "at least 20 dates and 30 positive rows for each candidate override regime",
            "reason": "a continuous meta-gate or finer reclustering on only 480 OOF dates would add overfitting risk",
        },
        "validation_integrity": {
            "candidate_selection": "B2/B3 prequential only",
            "B4_status": "reused development diagnostic, not fresh sealed",
            "external_sealed_target_consumed": False,
            "external_sealed_predictions_made": False,
        },
        "artifacts": {
            "first_policy": str(first_root / "FOLLOWUP_FROZEN_POLICY.json"),
            "final_policy": str(final_root / "LOCKED_OVERRIDE_FROZEN_POLICY.json"),
            "route_selection": str(final_root / "LOCKED_OVERRIDE_ROUTE_SELECTION.csv"),
            "calibration_selection": str(final_root / "LOCKED_OVERRIDE_CALIBRATION_SELECTION.csv"),
        },
    }
    atomic_json(report, final_root / "NEXT_EXPERIMENT_FINAL_REVIEW.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
