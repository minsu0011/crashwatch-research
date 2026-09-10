from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from surge_ticker_common_v10 import load_json, verify_output_inventory


SCHEMA_VERSION = "crashwatch_surge_tickerwise_correlation_map_v10"


def check(condition: bool, message: str, failures: list[str], counter: list[int]) -> None:
    counter[0] += 1
    if not bool(condition):
        failures.append(message)


def verify_output(output: Path) -> dict[str, Any]:
    failures: list[str] = []
    counter = [0]
    required = [
        "RUN_STATUS.json",
        "FINAL_RECOMMENDATION_V10.json",
        "OUTPUT_INVENTORY_V10.json",
        "TICKER_SEPARATION_MAP_MANIFEST_V10.json",
        "ticker_target_summary.csv",
        "ticker_feature_map_summary.csv",
        "ticker_separation_map_complete.csv",
        "ticker_hierarchical_effect_map.csv",
        "ticker_peer_reference_map.csv",
        "ticker_effect_taxonomy_summary.csv",
        "TICKER_DRIVER_PROFILES_V10.json",
        "ticker_driver_profile_membership.csv",
        "ticker_similarity_matrix.csv",
        "ticker_similarity_overlap.csv",
        "ticker_map_clusters.csv",
        "ticker_separation_graph_v10.graphml",
        "ticker_similarity_graph_v10.graphml",
        "TICKER_SEPARATION_MAP_REPORT_KO.md",
    ]
    for name in required:
        check((output / name).exists(), f"Missing required artifact: {name}", failures, counter)
    if failures:
        return {"status": "FAILED", "checks": counter[0], "failures": failures}

    status = load_json(output / "RUN_STATUS.json")
    recommendation = load_json(output / "FINAL_RECOMMENDATION_V10.json")
    manifest = load_json(output / "TICKER_SEPARATION_MAP_MANIFEST_V10.json")
    inventory = load_json(output / "OUTPUT_INVENTORY_V10.json")
    check(status.get("status") == "SUCCESS", "RUN_STATUS is not SUCCESS", failures, counter)
    check(status.get("schema") == SCHEMA_VERSION, "RUN_STATUS schema mismatch", failures, counter)
    check(manifest.get("schema") == SCHEMA_VERSION, "Manifest schema mismatch", failures, counter)
    check(recommendation.get("schema") == SCHEMA_VERSION, "Recommendation schema mismatch", failures, counter)
    check(not bool(recommendation.get("common_prediction_model_used")), "Common prediction model must be disabled", failures, counter)
    check(bool(recommendation.get("peer_reference_is_leave_one_ticker_out")), "Peer reference is not marked leave-one-ticker-out", failures, counter)
    check(bool(recommendation.get("hierarchical_peer_reference_used_for_map_only")), "Peer hierarchy must be map-only", failures, counter)

    inventory_ok, inventory_errors = verify_output_inventory(output, inventory)
    check(inventory_ok, "Output inventory verification failed", failures, counter)
    failures.extend([f"Inventory: {message}" for message in inventory_errors])

    target = pd.read_csv(output / "ticker_target_summary.csv", dtype={"ticker": str})
    base = pd.read_csv(output / "ticker_feature_map_summary.csv", dtype={"ticker": str})
    hierarchy = pd.read_csv(output / "ticker_hierarchical_effect_map.csv", dtype={"ticker": str})
    references = pd.read_csv(output / "ticker_peer_reference_map.csv", dtype={"ticker": str})
    membership = pd.read_csv(output / "ticker_driver_profile_membership.csv", dtype={"ticker": str})
    clusters = pd.read_csv(output / "ticker_map_clusters.csv", dtype={"ticker": str})
    similarity = pd.read_csv(output / "ticker_similarity_matrix.csv", dtype={"ticker": str}).set_index("ticker")
    overlap = pd.read_csv(output / "ticker_similarity_overlap.csv", dtype={"ticker": str}).set_index("ticker")

    target_tickers = set(target["ticker"].astype(str))
    check(len(target_tickers) > 0, "No tickers in target summary", failures, counter)
    check(set(base["ticker"].astype(str)) == target_tickers, "Base map ticker set mismatch", failures, counter)
    check(set(hierarchy["ticker"].astype(str)) == target_tickers, "Hierarchy ticker set mismatch", failures, counter)
    check(set(clusters["ticker"].astype(str)) == target_tickers, "Ticker cluster set mismatch", failures, counter)
    check(not membership.empty, "Driver profile membership is empty", failures, counter)
    check({"TARGET", "AB", "CD"}.issubset(set(hierarchy["axis"].astype(str))), "Hierarchy is missing one or more map axes", failures, counter)
    check(hierarchy["node_id"].notna().all(), "Hierarchy contains null node IDs", failures, counter)
    check(hierarchy[["ticker", "axis", "node_id"]].duplicated().sum() == 0, "Hierarchy node key is not unique", failures, counter)
    check("ticker_specific_delta" in hierarchy.columns, "Ticker-specific delta missing", failures, counter)
    check("ticker_effect_class" in hierarchy.columns, "Ticker effect taxonomy missing", failures, counter)
    check("posterior_signed_effect" in hierarchy.columns, "Posterior signed effect missing", failures, counter)
    check("peer_prior_signed_effect" in hierarchy.columns, "Peer prior signed effect missing", failures, counter)
    finite_posterior = np.isfinite(pd.to_numeric(hierarchy["posterior_signed_effect"], errors="coerce").to_numpy(dtype=float))
    check(bool(finite_posterior.any()), "No finite posterior ticker effects", failures, counter)
    allowed_classes = {
        "LOW_EVIDENCE",
        "WEAK_OR_NEUTRAL",
        "TICKER_DIRECTION_REVERSAL",
        "TICKER_UNIQUE_DRIVER",
        "TICKER_AMPLIFIED_DRIVER",
        "TICKER_DAMPENED_DRIVER",
        "INDUSTRY_SHARED_DRIVER",
        "BUCKET_SHARED_DRIVER",
        "UNIVERSAL_DRIVER",
        "MIXED_TICKER_EFFECT",
    }
    check(set(hierarchy["ticker_effect_class"].dropna().astype(str)).issubset(allowed_classes), "Unknown ticker effect class", failures, counter)
    check(set(references["ticker"].astype(str)) == target_tickers, "Peer reference ticker set mismatch", failures, counter)

    check(similarity.shape[0] == similarity.shape[1] == len(target_tickers), "Ticker similarity matrix shape mismatch", failures, counter)
    if similarity.shape[0] > 0:
        similarity_values = similarity.to_numpy(dtype=float)
        check(np.allclose(similarity_values, similarity_values.T, atol=1e-10, equal_nan=True), "Ticker similarity matrix is not symmetric", failures, counter)
        check(np.allclose(np.diag(similarity_values), 1.0, atol=1e-10), "Ticker similarity diagonal is not one", failures, counter)
    check(overlap.shape == similarity.shape, "Ticker overlap matrix shape mismatch", failures, counter)

    per_ticker_root = output / "per_ticker"
    for ticker in sorted(target_tickers):
        ticker_dir = per_ticker_root / ticker
        check(ticker_dir.exists(), f"Missing per-ticker directory: {ticker}", failures, counter)
        check((ticker_dir / "hierarchical_driver_map.csv").exists(), f"Missing hierarchical driver map: {ticker}", failures, counter)
        check((ticker_dir / "top_precision_drivers.csv").exists(), f"Missing precision driver map: {ticker}", failures, counter)
        check((ticker_dir / "top_surge_association.csv").exists(), f"Missing surge association map: {ticker}", failures, counter)

    visualization_manifest = output / "TICKER_VISUALIZATION_MANIFEST_V10.json"
    if visualization_manifest.exists():
        visualizations = load_json(visualization_manifest)
        for relative in visualizations:
            check((output / str(relative)).exists(), f"Missing visualization: {relative}", failures, counter)

    probe_stage = status.get("stages", {}).get("ticker_probe", {})
    if probe_stage.get("status") == "SUCCESS":
        for name in [
            "ticker_probe_predictions.csv",
            "ticker_probe_metrics_by_ticker_fold.csv",
            "ticker_probe_metrics_summary.csv",
            "ticker_portfolio_metrics_by_fold.csv",
            "ticker_portfolio_metrics_by_role.csv",
            "ticker_threshold_policy_by_fold.csv",
        ]:
            check((output / name).exists(), f"Missing probe artifact: {name}", failures, counter)
    else:
        check(recommendation.get("status") == "TICKER_MAP_COMPLETE_PROBE_NOT_RUN", "Map-only run has unexpected recommendation status", failures, counter)

    result = {
        "status": "PASS" if not failures else "FAILED",
        "checks": counter[0],
        "failure_count": len(failures),
        "failures": failures,
        "tickers": len(target_tickers),
        "hierarchy_nodes": int(len(hierarchy)),
        "probe_status": probe_stage.get("status"),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CrashWatch Surge ticker-wise map V10 artifacts.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_output(args.output.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
