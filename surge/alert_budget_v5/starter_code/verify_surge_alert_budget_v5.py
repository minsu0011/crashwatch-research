from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from surge_model_zoo_common import payload_checksum_is_valid, sha256_file


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_model_path(root: Path, record: Mapping[str, Any]) -> Path:
    raw = Path(str(record["path"]))
    return raw if raw.is_absolute() else root / raw


def verify_model_record(root: Path, record: Mapping[str, Any], failures: list[str]) -> None:
    path = resolve_model_path(root, record)
    if not path.exists():
        failures.append(f"model missing: {path}")
        return
    expected = record.get("sha256")
    if isinstance(expected, str) and sha256_file(path) != expected:
        failures.append(f"model sha256 mismatch: {path}")
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and int(path.stat().st_size) != int(expected_bytes):
        failures.append(f"model size mismatch: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CrashWatch Surge Alert Budget V5 output")
    parser.add_argument("--output", type=Path, default=Path("outputs/surge_alert_budget_v5"))
    args = parser.parse_args()
    root = args.output.resolve()
    failures: list[str] = []
    checks = 0

    required = [
        "RUN_STATUS.json",
        "ALERT_BUDGET_FREEZE_V5.json",
        "FINAL_RECOMMENDATION_V5.json",
        "BUDGET_PERFORMANCE_GAP_REPORT.json",
        "budget_candidate_metrics_by_fold.csv",
        "budget_candidate_metrics_by_role.csv",
        "method_crossfit_summary.csv",
        "budget_candidate_predictions.npz",
        "OUTPUT_INVENTORY.json",
    ]
    for name in required:
        checks += 1
        if not (root / name).exists():
            failures.append(f"required file missing: {name}")

    if (root / "RUN_STATUS.json").exists():
        status = load_json(root / "RUN_STATUS.json")
        checks += 1
        if status.get("status") != "SUCCESS":
            failures.append(f"RUN_STATUS is not SUCCESS: {status.get('status')}")
        if status.get("schema") != "crashwatch_surge_alert_budget_runner_v5":
            failures.append(f"RUN_STATUS schema mismatch: {status.get('schema')}")


    inventory_path = root / "OUTPUT_INVENTORY.json"
    if inventory_path.exists():
        inventory = load_json(inventory_path)
        for record in inventory.get("files", []):
            checks += 1
            relative = Path(str(record.get("relative_path", "")))
            path = root / relative
            if not path.exists() or not path.is_file():
                failures.append(f"inventory file missing: {relative}")
                continue
            if int(path.stat().st_size) != int(record.get("bytes", -1)):
                failures.append(f"inventory size mismatch: {relative}")
            if sha256_file(path) != str(record.get("sha256", "")):
                failures.append(f"inventory sha256 mismatch: {relative}")

    for name in [
        "ALERT_BUDGET_FREEZE_V5.json",
        "FINAL_RECOMMENDATION_V5.json",
        "BUDGET_PERFORMANCE_GAP_REPORT.json",
        "SIGNAL_REGISTRY.json",
    ]:
        path = root / name
        if not path.exists():
            continue
        checks += 1
        payload = load_json(path)
        if not payload_checksum_is_valid(payload):
            failures.append(f"payload checksum mismatch: {name}")

    registry_path = root / "V5_RANKER_PRODUCTION_REGISTRY.json"
    if registry_path.exists():
        registry = load_json(registry_path)
        checks += 1
        if not payload_checksum_is_valid(registry):
            failures.append("V5_RANKER_PRODUCTION_REGISTRY checksum mismatch")
        for record in registry.get("models", []):
            model = record.get("model")
            if isinstance(model, dict):
                checks += 1
                verify_model_record(root, model, failures)

    freeze_path = root / "ALERT_BUDGET_FREEZE_V5.json"
    if freeze_path.exists():
        freeze = load_json(freeze_path)
        model = freeze.get("method_spec", {}).get("model")
        if isinstance(model, dict):
            checks += 1
            verify_model_record(root, model, failures)
        policy = freeze.get("daily_budget_policy", {})
        checks += 1
        fraction = float(policy.get("daily_fraction", 0.0))
        maximum = float(policy.get("maximum_alert_rate", 1.0))
        if not (0 < fraction <= maximum + 1e-12):
            failures.append(f"frozen policy violates max alert rate: fraction={fraction}, max={maximum}")
        max_count = policy.get("max_alerts_per_day")
        if max_count is not None and int(max_count) < 1:
            failures.append(f"invalid max_alerts_per_day: {max_count}")

        metrics_path = root / "budget_candidate_metrics_by_fold.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            for _, row in metrics.iterrows():
                checks += 1
                alert_rate = float(row.get("alert_rate", float("nan")))
                if not pd.isna(alert_rate) and alert_rate > maximum + 1e-12:
                    failures.append(
                        f"fold alert rate exceeds hard cap: fold={row.get('fold_id')}, "
                        f"rate={alert_rate}, max={maximum}"
                    )
                if max_count is not None:
                    daily_max = float(row.get("alerts_per_day_max", float("nan")))
                    if not pd.isna(daily_max) and daily_max > int(max_count) + 1e-12:
                        failures.append(
                            f"fold daily alert count exceeds hard cap: fold={row.get('fold_id')}, "
                            f"count={daily_max}, max={max_count}"
                        )

    for result_path in root.glob("rank_task_cache/*/seed_*/fold_*.json"):
        checks += 1
        payload = load_json(result_path)
        if not payload_checksum_is_valid(payload):
            failures.append(f"task payload checksum mismatch: {result_path}")
            continue
        prediction = Path(str(payload.get("prediction_path", "")))
        if not prediction.is_absolute():
            prediction = root / prediction
        if not prediction.exists():
            failures.append(f"task prediction missing: {prediction}")
        elif payload.get("prediction_sha256") != sha256_file(prediction):
            failures.append(f"task prediction sha256 mismatch: {prediction}")

    report = {
        "schema": "crashwatch_surge_alert_budget_verification_v5",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
    }
    output_path = root / "VERIFICATION_REPORT_V5.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
