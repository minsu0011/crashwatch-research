from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/surge_tickerwise_correlation_map_v10_1")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    failures: list[str] = []
    checks = 0

    def check(condition: bool, message: str) -> None:
        nonlocal checks
        checks += 1
        if not condition:
            failures.append(message)

    required = [
        "RUN_STATUS.json",
        "FINAL_RECOMMENDATION_V10_1.json",
        "hierarchy_level_audit_v10_1.csv",
        "ticker_hierarchical_effect_map_v10_1.csv",
        "ticker_precision_separator_map_v10_1.csv",
        "ticker_similarity_matrix_v10_1.csv",
        "ticker_similarity_signature_nodes_v10_1.csv",
        "ticker_probe_eligibility_v10_1.csv",
        "TICKER_DRIVER_PROFILES_V10_1.json",
    ]
    for name in required:
        check((output / name).exists(), f"Missing {name}")
    if failures:
        raise SystemExit("\n".join(failures))

    status = json.loads((output / "RUN_STATUS.json").read_text(encoding="utf-8"))
    check(status.get("status") == "SUCCESS", "RUN_STATUS is not SUCCESS")
    recommendation = json.loads((output / "FINAL_RECOMMENDATION_V10_1.json").read_text(encoding="utf-8"))
    check(recommendation.get("selection_only_profiles_safe_for_model_selection") is True, "Selection-only flag missing")
    check(recommendation.get("confirmation_recent_are_diagnostic_only") is True, "Holdout diagnostic flag missing")

    audit = pd.read_csv(output / "hierarchy_level_audit_v10_1.csv")
    check({"industry", "bucket", "market", "global"}.issubset(set(audit["level"])), "Hierarchy audit levels missing")
    effective_weight_sum = float(pd.to_numeric(audit["effective_weight"], errors="coerce").fillna(0.0).sum())
    check(abs(effective_weight_sum - 1.0) < 1e-8, f"Hierarchy weights do not sum to 1: {effective_weight_sum}")
    invalid_nonzero = audit.loc[(~audit["valid"].astype(bool)) & (pd.to_numeric(audit["effective_weight"], errors="coerce") > 1e-12)]
    check(invalid_nonzero.empty, "Invalid hierarchy level has nonzero effective weight")

    hierarchy = pd.read_csv(output / "ticker_hierarchical_effect_map_v10_1.csv", dtype={"ticker": str})
    check(not hierarchy.empty, "Corrected hierarchy is empty")
    check(hierarchy[["ticker", "axis", "node_id"]].duplicated().sum() == 0, "Hierarchy key is not unique")
    for column in [
        "ticker_signed_effect_raw", "posterior_signed_effect_median", "specific_z_median",
        "reliability_median", "ticker_specific_robust", "ticker_effect_class_v10_1",
    ]:
        check(column in hierarchy.columns, f"Hierarchy missing {column}")
    check(set(hierarchy["axis"].astype(str)).issuperset({"TARGET", "AB", "CD"}), "Map axes missing")

    precision = pd.read_csv(output / "ticker_precision_separator_map_v10_1.csv", dtype={"ticker": str})
    check((precision["axis"].astype(str) == "AB").all(), "Precision map contains non-AB rows")
    check("precision_separator_selection_candidate" in precision.columns, "Precision selection flag missing")
    check("precision_separator_confirmed" in precision.columns, "Precision confirmation flag missing")

    similarity = pd.read_csv(output / "ticker_similarity_matrix_v10_1.csv", index_col=0)
    values = similarity.to_numpy(dtype=float)
    check(values.shape[0] == values.shape[1], "Similarity matrix not square")
    check(np.allclose(values, values.T, atol=1e-10), "Similarity matrix not symmetric")
    check(np.allclose(np.diag(values), 1.0, atol=1e-10), "Similarity diagonal not one")
    signature = pd.read_csv(output / "ticker_similarity_signature_nodes_v10_1.csv")
    check(len(signature) > 0, "Fixed signature is empty")

    eligibility = pd.read_csv(output / "ticker_probe_eligibility_v10_1.csv", dtype={"ticker": str})
    recent = eligibility.loc[eligibility["role"].astype(str).eq("recent_audit")]
    check(not recent.empty, "Recent eligibility rows missing")
    if (recent["validation_rows"] < 60).all():
        check(recent["v10_1_adaptive_row_threshold"].astype(bool).any(), "Recent row adaptation was not activated")

    inventory_path = output / "OUTPUT_INVENTORY_V10_1.csv"
    if inventory_path.exists():
        inventory = pd.read_csv(inventory_path)
        for row in inventory.itertuples(index=False):
            path = output / str(row.file)
            check(path.exists(), f"Inventory file missing: {row.file}")
            if path.exists():
                check(path.stat().st_size == int(row.size), f"Inventory size mismatch: {row.file}")
                check(sha256_file(path) == str(row.sha256), f"Inventory hash mismatch: {row.file}")

    if failures:
        print(f"FAILED {len(failures)}/{checks} checks")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)
    print(f"PASS {checks}/{checks} checks")


if __name__ == "__main__":
    main()
