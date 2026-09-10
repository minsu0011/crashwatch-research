from __future__ import annotations

import argparse
import json
from pathlib import Path

from surge_hardfp_common_v7 import atomic_write_json, checksum_valid, sha256_file


REQUIRED = [
    "RUN_STATUS.json",
    "EVENT_TARGET_AUDIT_V7.json",
    "base_component_metrics.csv",
    "error_group_membership_oof.csv",
    "error_contrast_map_by_fold.csv",
    "error_contrast_map_summary.csv",
    "ERROR_FEATURE_MANIFEST_V7.json",
    "FORWARD_META_AUDIT_V7.csv",
    "method_forward_precision_summary.csv",
    "final_precision_metrics_by_fold.csv",
    "final_precision_metrics_by_role.csv",
    "FINAL_POLICY_V7.json",
    "FINAL_RECOMMENDATION_V7.json",
    "BOTTLENECK_REPORT_V7.json",
    "OUTPUT_INVENTORY_V7.json",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/surge_hardfp_v7"))
    args = parser.parse_args()
    root = args.output.resolve()
    failures: list[str] = []
    checks = 0
    for name in REQUIRED:
        checks += 1
        if not (root / name).exists():
            failures.append(f"missing: {name}")
    for name in ["RUN_STATUS.json", "EVENT_TARGET_AUDIT_V7.json", "ERROR_FEATURE_MANIFEST_V7.json", "FINAL_POLICY_V7.json", "FINAL_RECOMMENDATION_V7.json", "BOTTLENECK_REPORT_V7.json", "OUTPUT_INVENTORY_V7.json"]:
        path = root / name
        if not path.exists():
            continue
        checks += 1
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not checksum_valid(payload):
            failures.append(f"payload checksum: {name}")
    inventory_path = root / "OUTPUT_INVENTORY_V7.json"
    if inventory_path.exists():
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for item in inventory.get("files", []):
            path = root / item["path"]
            checks += 1
            if not path.exists():
                failures.append(f"inventory missing: {item['path']}")
                continue
            if int(path.stat().st_size) != int(item["size"]):
                failures.append(f"inventory size: {item['path']}")
            elif sha256_file(path) != item["sha256"]:
                failures.append(f"inventory sha256: {item['path']}")
    report = {"schema": "crashwatch_surge_hardfp_verifier_v7", "checks": checks, "failures": failures, "status": "PASS" if not failures else "FAIL"}
    atomic_write_json(root / "VERIFICATION_REPORT_V7.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
