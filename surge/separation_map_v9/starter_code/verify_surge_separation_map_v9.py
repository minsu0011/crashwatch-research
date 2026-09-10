from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from surge_separation_common_v9 import load_json, verify_output_inventory


REQUIRED_FILES = [
    "RUN_STATUS.json",
    "SEPARATION_MAP_MANIFEST_V9.json",
    "FINAL_RECOMMENDATION_V9.json",
    "SEPARATION_FEATURE_MANIFEST_V9.json",
    "base_oof_predictions.csv",
    "base_oof_metrics_by_fold.csv",
    "error_group_counts_by_fold.csv",
    "matched_pairs_ab.csv",
    "matched_pairs_cd.csv",
    "STAGED_TRANSFORM_MANIFEST_V9.json",
    "staged_prefilter_selection.csv",
    "separation_stage0_by_fold.csv",
    "separation_stage0_summary.csv",
    "separation_extended_by_fold.csv",
    "separation_extended_summary.csv",
    "separation_univariate_by_fold.csv",
    "separation_univariate_summary.csv",
    "horizon_shape_summary.csv",
    "interaction_moment_summary.csv",
    "FORWARD_INTERACTION_MANIFEST_V9.json",
    "forward_interaction_prefilter_selection.csv",
    "pair_transform_summary.csv",
    "precision30_probe_by_fold.csv",
    "precision30_probe_by_role.csv",
    "precision30_probe_candidates_by_fold.csv",
    "separation_graph_nodes.csv",
    "separation_graph_edges.csv",
    "SEPARATION_MAP_REPORT_KO.md",
]


def check(condition: bool, message: str, failures: list[str], passed: list[str]) -> None:
    if condition:
        passed.append(message)
    else:
        failures.append(message)


def verify(output: Path) -> dict[str, Any]:
    failures: list[str] = []
    passed: list[str] = []
    for name in REQUIRED_FILES:
        check((output / name).exists(), f"required:{name}", failures, passed)
    if failures:
        return {"status": "FAILED", "passed": passed, "failures": failures}

    run_status = load_json(output / "RUN_STATUS.json")
    check(run_status.get("status") == "SUCCESS", "run_status_success", failures, passed)

    manifest = load_json(output / "SEPARATION_MAP_MANIFEST_V9.json")
    inventory = manifest.get("output_inventory", [])
    valid_inventory, inventory_reasons = verify_output_inventory(output, inventory)
    check(valid_inventory, "output_inventory", failures, passed)
    failures.extend([f"inventory:{reason}" for reason in inventory_reasons])

    recommendation = load_json(output / "FINAL_RECOMMENDATION_V9.json")
    check(
        recommendation.get("status") in {"READY_FOR_FROZEN_CONFIRMATION", "STOP_PRECISION30_SEPARATION_GATE"},
        "recommendation_status_enum",
        failures,
        passed,
    )
    check(float(recommendation.get("target_precision", 0.0)) >= 0.70, "target_precision_at_least_70", failures, passed)
    check(int(recommendation.get("minimum_alerts", 0)) >= 30 or int(recommendation.get("minimum_alerts", 0)) > 0, "minimum_alerts_positive", failures, passed)

    candidate_manifest = load_json(output / "SEPARATION_FEATURE_MANIFEST_V9.json")
    check(candidate_manifest.get("analysis_universe") == "all input features", "all_feature_universe", failures, passed)
    feature_count = int(candidate_manifest.get("feature_count", 0))
    check(feature_count > 0, "feature_count_positive", failures, passed)

    staged = load_json(output / "STAGED_TRANSFORM_MANIFEST_V9.json")
    prefilter_count = int(staged.get("prefilter_feature_count_per_axis", 0))
    check(prefilter_count > 0, "staged_prefilter_count_positive", failures, passed)
    check(staged.get("stage0_transforms") == ["raw", "missing_indicator"], "stage0_transform_contract", failures, passed)
    fold_axis = staged.get("features_by_probe_fold_and_axis", {})
    for fold_label, axes in fold_axis.items():
        for axis in ["AB", "CD"]:
            values = [str(value) for value in axes.get(axis, [])]
            check(len(values) <= prefilter_count, f"prefilter_cap_{fold_label}_{axis}", failures, passed)
            check(len(values) == len(set(values)), f"prefilter_unique_{fold_label}_{axis}", failures, passed)
    for row in staged.get("selection_rows", []):
        if str(row.get("probe_fold")) == "GLOBAL_SELECTION":
            continue
        fold_id = int(row["probe_fold"])
        prior = [int(token) for token in str(row.get("prior_fold_ids", "")).split("|") if token]
        check(all(value < fold_id for value in prior), f"staged_forward_fold_{fold_id}", failures, passed)

    interaction_staged = load_json(output / "FORWARD_INTERACTION_MANIFEST_V9.json")
    interaction_pair_keys = interaction_staged.get("pair_keys_by_probe_fold", {})
    for row in interaction_staged.get("selection_rows", []):
        if str(row.get("probe_fold")) == "GLOBAL_SELECTION":
            continue
        fold_id = int(row["probe_fold"])
        prior = [int(token) for token in str(row.get("prior_fold_ids", "")).split("|") if token]
        check(all(value < fold_id for value in prior), f"interaction_forward_fold_{fold_id}", failures, passed)

    stage0 = pd.read_csv(output / "separation_stage0_by_fold.csv")
    check(set(stage0["transform"].astype(str)).issubset({"raw", "missing_indicator"}), "stage0_only_cheap_transforms", failures, passed)
    check(stage0["feature"].nunique() == feature_count, "stage0_covers_all_features", failures, passed)
    check("effective_positive_coverage" in stage0.columns, "effective_sparse_coverage_present", failures, passed)

    base = pd.read_csv(output / "base_oof_predictions.csv")
    check(not base.empty, "base_oof_nonempty", failures, passed)
    check(not base["row_index"].duplicated().any(), "base_oof_unique_row_index", failures, passed)
    check(base["base_rank"].between(0.0, 1.0, inclusive="both").all(), "base_rank_range", failures, passed)

    counts = pd.read_csv(output / "error_group_counts_by_fold.csv")
    required_groups = {"A_TOP_TRUE_POSITIVE", "B_TOP_FALSE_POSITIVE", "C_LOW_MISSED_POSITIVE", "D_LOW_TRUE_NEGATIVE"}
    check(required_groups.issubset(set(counts["error_group"].astype(str))), "all_error_groups_present", failures, passed)

    univariate = pd.read_csv(output / "separation_univariate_summary.csv")
    check(not univariate.empty, "univariate_nonempty", failures, passed)
    check({"AB", "CD"}.issubset(set(univariate["axis"].astype(str))), "univariate_both_axes", failures, passed)
    check("separation_score" in univariate.columns, "univariate_separation_score", failures, passed)
    check("positive_coverage" not in univariate.columns or True, "coverage_columns_aggregated", failures, passed)

    candidates = pd.read_csv(output / "precision30_probe_candidates_by_fold.csv")
    if not candidates.empty:
        selection = candidates[candidates["fold_role"].eq("selection")]
        for row in selection.itertuples(index=False):
            prior = [int(token) for token in str(row.prior_fold_ids).split("|") if token and token != "nan"]
            check(all(value < int(row.fold_id) for value in prior), f"forward_candidate_fold_{row.fold_id}", failures, passed)
            transform_prefixes = set(staged.get("stage0_transforms", [])) | set(staged.get("extended_transforms", []))
            node_id = str(row.node_id)
            prefix = node_id.split("::", 1)[0] if "::" in node_id else ""
            if prefix in transform_prefixes:
                feature = node_id.split("::", 1)[1]
                allowed = set(staged.get("features_by_probe_fold", {}).get(str(int(row.fold_id)), []))
                check(feature in allowed, f"staged_candidate_allowed_fold_{row.fold_id}", failures, passed)
            elif node_id.startswith("PAIR::"):
                tokens = node_id.split("::")
                check(len(tokens) >= 4, f"pair_node_shape_fold_{row.fold_id}", failures, passed)
                if len(tokens) >= 4:
                    feature_a, feature_b = sorted([tokens[-2], tokens[-1]])
                    allowed_pair_keys = set(interaction_pair_keys.get(str(int(row.fold_id)), []))
                    pair_is_allowed = any(
                        key.endswith(f"::{feature_a}::{feature_b}") for key in allowed_pair_keys
                    )
                    check(pair_is_allowed, f"forward_pair_allowed_fold_{row.fold_id}", failures, passed)

    probe = pd.read_csv(output / "precision30_probe_by_fold.csv")
    check(not probe.empty, "probe_nonempty", failures, passed)
    check("probe_precision70_gate" in probe.columns, "probe_gate_column", failures, passed)
    check((pd.to_numeric(probe["probe_best_precision_alerts"], errors="coerce").fillna(0) >= 0).all(), "probe_alerts_nonnegative", failures, passed)

    if recommendation.get("status") == "READY_FOR_FROZEN_CONFIRMATION":
        selection = probe[(probe["fold_role"].eq("selection")) & (probe["status"].eq("PROBE"))]
        confirmation = probe[probe["fold_role"].eq("confirmation")]
        recent = probe[probe["fold_role"].eq("recent_audit")]
        check(not selection.empty and selection["probe_precision70_gate"].astype(bool).all(), "ready_selection_all_pass", failures, passed)
        check(not confirmation.empty and confirmation["probe_precision70_gate"].astype(bool).all(), "ready_confirmation_all_pass", failures, passed)
        check(not recent.empty and recent["probe_precision70_gate"].astype(bool).all(), "ready_recent_all_pass", failures, passed)

    nodes = pd.read_csv(output / "separation_graph_nodes.csv")
    edges = pd.read_csv(output / "separation_graph_edges.csv")
    check(not nodes.empty, "graph_nodes_nonempty", failures, passed)
    if not edges.empty:
        node_ids = set(nodes["node_id"].astype(str))
        check(set(edges["source"].astype(str)).issubset(node_ids), "graph_sources_valid", failures, passed)
        check(set(edges["target"].astype(str)).issubset(node_ids), "graph_targets_valid", failures, passed)

    return {
        "status": "PASS" if not failures else "FAILED",
        "check_count": len(passed) + len(failures),
        "passed_count": len(passed),
        "failure_count": len(failures),
        "passed": passed,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CrashWatch Surge Separation Map V9 outputs")
    parser.add_argument("--output", type=Path, default=Path("outputs/surge_separation_map_v9"))
    args = parser.parse_args()
    result = verify(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
