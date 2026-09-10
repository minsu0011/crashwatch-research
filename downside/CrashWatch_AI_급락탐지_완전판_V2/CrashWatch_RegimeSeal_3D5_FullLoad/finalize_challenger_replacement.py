from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from cw7h.utils import atomic_json, read_json


def main() -> None:
    package = Path(__file__).resolve().parent
    project = package.parent
    output = project / "crashwatch_ai_data/regime_continuous_meta_gate_v1"
    current = read_json(output / "CURRENT_CHALLENGER_REPLACEMENT_AUDIT.json", {})
    selected = read_json(output / "META_GATE_SELECTION_BEFORE_B4.json", {})
    active = read_json(output / "BEST_ACTIVE_META_B4_DIAGNOSTIC.json", {})
    decision = read_json(output / "FINAL_REPLACEMENT_DECISION.json", {})
    run = read_json(output / "RUN_STATUS.json", {})
    candidates = pd.read_csv(output / "META_GATE_CANDIDATE_SELECTION.csv")
    active_candidates = candidates[candidates["is_active_challenger"]]
    report = {
        "status": "CHALLENGER_REPLACEMENT_EVALUATION_AND_NEXT_EXPERIMENT_COMPLETE",
        "final_decision": "DO_NOT_REPLACE_LOCKED_P2",
        "current_regime_challenger": {
            "replacement_approved": current["replacement_approved"],
            "criteria": current["criteria"],
            "B4_bootstrap": current["B4_bootstrap_selection_score"],
        },
        "next_experiment_executed": {
            "design": "continuous date-level meta gate using trailing market state and target-free P2/P7 prediction-distribution features",
            "OOF_dates": run["daily_rows"],
            "meta_features": run["meta_features"],
            "candidate_count": run["candidates"],
            "selection_blocks": ["B2", "B3"],
            "B4_used_for_selection": False,
            "bootstrap_reps_no_override": run["bootstrap_reps"],
            "bootstrap_reps_best_active": active["bootstrap_selection_improvement"]["reps"],
        },
        "meta_gate_selection": {
            "selected_policy": selected["candidate_id"],
            "selection_fell_back_to_no_override": selected["selection_fell_back_to_no_override"],
            "active_candidate_count": int(len(active_candidates)),
            "active_noninferior_candidate_count": int(active_candidates["noninferior_both_blocks"].sum()),
            "best_active_candidate": selected["best_active_challenger"],
        },
        "best_active_candidate_B4_reused_diagnostic": active,
        "replacement_criteria": decision["criteria"],
        "replacement_approved": decision["challenger_replaces_locked_P2"],
        "interpretation": [
            "the original regime challenger is not stable enough to replace locked P2",
            "none of 42 active continuous meta-gates was noninferior in both B2 and B3",
            "the best active gate improved B4 PR-AUC/ROC point estimates but lost composite score and its 95% interval crossed zero",
            "global Platt 50% blending consistently improves Brier/log-loss without changing ranking and should be kept as a separate risk head",
        ],
        "production_recommendation": {
            "ranking_and_alerts": "locked P2 raw score = 0.1 P2_LGB + 0.9 P2_XGB",
            "risk_probability": "50% locked P2 raw + 50% global Platt(C=0.1)",
            "challenger": "retain P7/meta gate offline only; do not route production alerts to it",
        },
        "next_valid_test": {
            "requirement": "fresh future period with at least 120-180 dates and broader regime coverage",
            "promotion_rule": "active B2/B3/B4 noninferiority plus positive lower confidence bound and one untouched external sealed pass",
            "more_internal_threshold_tuning": "not recommended because high meta scores did not correspond to higher realized P7 utility",
        },
        "integrity": {
            "external_sealed_target_consumed": False,
            "external_sealed_predictions_made": False,
            "B4_status": "reused development diagnostic, not fresh sealed",
        },
        "artifacts": {
            "replacement_decision": str(output / "FINAL_REPLACEMENT_DECISION.json"),
            "candidate_selection": str(output / "META_GATE_CANDIDATE_SELECTION.csv"),
            "active_B4": str(output / "BEST_ACTIVE_META_B4_DIAGNOSTIC.json"),
            "frozen_policy": str(output / "META_GATE_FROZEN_POLICY.json"),
        },
    }
    atomic_json(report, output / "CHALLENGER_REPLACEMENT_FINAL_REVIEW.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
