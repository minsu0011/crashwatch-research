from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from surge_ablation_common import load_json, verify_output_inventory


def verify(output: Path, allow_partial: bool = False) -> dict[str, Any]:
    output = output.resolve()
    checks: list[dict[str, Any]] = []

    def check(name: str, condition: bool, detail: Any = None) -> None:
        checks.append({"name": name, "pass": bool(condition), "detail": detail})

    required = [
        "RUN_STATUS.json",
        "ORGANIC_ABLATION_MANIFEST_V8.json",
        "completion_by_type.csv",
        "baseline_pairing_audit.json",
        "organic_feature_consensus.csv",
        "organic_feature_map_selection.csv",
        "correlation_cluster_synergy.csv",
        "correlated_pair_nonadditivity.csv",
        "error_group_counts_by_fold.csv",
    ]
    for name in required:
        check(f"exists:{name}", (output / name).exists())

    if (output / "RUN_STATUS.json").exists():
        status = load_json(output / "RUN_STATUS.json")
        check("run_status_success", status.get("status") == "SUCCESS", status.get("status"))

    if (output / "ORGANIC_ABLATION_MANIFEST_V8.json").exists():
        manifest = load_json(output / "ORGANIC_ABLATION_MANIFEST_V8.json")
        inventory = manifest.get("output_inventory", [])
        valid, reasons = verify_output_inventory(output, inventory)
        check("inventory", valid, reasons[:20])
        feature_count = int(manifest.get("feature_count", 0))
        check("feature_count_positive", feature_count > 0, feature_count)
        if feature_count == 439:
            check("full_439", True, 439)

    if (output / "completion_by_type.csv").exists():
        completion = pd.read_csv(output / "completion_by_type.csv")
        complete = completion["complete"].astype(str).str.lower().isin(["true", "1"]).all() if "complete" in completion else False
        check("all_task_types_complete", bool(complete) or allow_partial, completion.to_dict("records"))
        if "single_feature_loo" in set(completion.get("test_type", [])):
            row = completion[completion["test_type"].eq("single_feature_loo")].iloc[0]
            check("single_loo_complete", int(row["completed"]) == int(row["task_count"]), row.to_dict())

    if (output / "baseline_pairing_audit.json").exists():
        pairing = load_json(output / "baseline_pairing_audit.json")
        check("baseline_pairing", pairing.get("status") == "COMPLETE", pairing)

    if (output / "organic_feature_consensus.csv").exists() and (output / "ORGANIC_ABLATION_MANIFEST_V8.json").exists():
        consensus = pd.read_csv(output / "organic_feature_consensus.csv")
        manifest = load_json(output / "ORGANIC_ABLATION_MANIFEST_V8.json")
        expected = int(manifest.get("feature_count", 0))
        check("consensus_one_row_per_feature", consensus["feature"].nunique() == expected, {"unique": int(consensus["feature"].nunique()), "expected": expected})

    failures = [item for item in checks if not item["pass"]]
    report = {"status": "PASS" if not failures else "FAIL", "checks": len(checks), "failures": failures}
    (output / "VERIFICATION_REPORT_V8.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = verify(args.output, allow_partial=args.allow_partial)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
