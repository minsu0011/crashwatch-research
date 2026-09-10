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


def resolve_path(root: Path, raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else root / path


def verify_model_record(root: Path, record: Mapping[str, Any], failures: list[str]) -> int:
    checks = 1
    path = resolve_path(root, str(record.get("path", "")))
    if not path.exists():
        failures.append(f"model missing: {path}")
        return checks
    expected_sha = record.get("sha256")
    if isinstance(expected_sha, str) and sha256_file(path) != expected_sha:
        failures.append(f"model sha256 mismatch: {path}")
    expected_bytes = record.get("bytes")
    if expected_bytes is not None and int(path.stat().st_size) != int(expected_bytes):
        failures.append(f"model size mismatch: {path}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify CrashWatch Surge Precision 70 V6 output")
    parser.add_argument("--output", type=Path, default=Path("outputs/surge_precision70_v6"))
    parser.add_argument(
        "--allow-gate-failed",
        action="store_true",
        help="STOP_PRECISION70_GATE 결과도 성능 성공으로 오인하지 않고 무결성만 검증",
    )
    args = parser.parse_args()
    root = args.output.resolve()
    failures: list[str] = []
    checks = 0

    required = [
        "RUN_STATUS.json",
        "PRECISION_FREEZE_V6.json",
        "FINAL_RECOMMENDATION_V6.json",
        "PRECISION_PERFORMANCE_GAP_REPORT.json",
        "precision_candidate_metrics_by_fold.csv",
        "precision_candidate_metrics_by_role.csv",
        "precision_recall_curve_by_role.csv",
        "method_forward_crossfit_summary.csv",
        "FORWARD_META_SPLIT_AUDIT.json" if (root / "FORWARD_META_SPLIT_AUDIT.json").exists() else "SIGNAL_REGISTRY_V6.json",
        "SIGNAL_REGISTRY_V6.json",
        "OUTPUT_INVENTORY.json",
    ]
    for name in list(dict.fromkeys(required)):
        checks += 1
        if not (root / name).exists():
            failures.append(f"required file missing: {name}")

    for name in [
        "PRECISION_FREEZE_V6.json",
        "FINAL_RECOMMENDATION_V6.json",
        "SIGNAL_REGISTRY_V6.json",
        "SIGNAL_CALIBRATORS_V6.json",
        "V6_SPECIALIST_PRODUCTION_REGISTRY.json",
        "OUTPUT_INVENTORY.json",
    ]:
        path = root / name
        if path.exists():
            checks += 1
            payload = load_json(path)
            if not payload_checksum_is_valid(payload):
                failures.append(f"payload checksum mismatch: {name}")

    status_path = root / "RUN_STATUS.json"
    if status_path.exists():
        status = load_json(status_path)
        checks += 2
        if status.get("status") != "SUCCESS":
            failures.append(f"RUN_STATUS is not SUCCESS: {status.get('status')}")
        if status.get("schema") != "crashwatch_surge_precision70_runner_v6":
            failures.append(f"RUN_STATUS schema mismatch: {status.get('schema')}")

    freeze_path = root / "PRECISION_FREEZE_V6.json"
    recommendation_path = root / "FINAL_RECOMMENDATION_V6.json"
    if freeze_path.exists() and recommendation_path.exists():
        freeze = load_json(freeze_path)
        recommendation = load_json(recommendation_path)
        checks += 7
        target = float(freeze.get("target_precision", recommendation.get("target_precision", 0.0)))
        selection_target = float(freeze.get("selection_target_precision", target))
        if target < 0.70 - 1e-12:
            failures.append(f"target precision below 70%: {target}")
        if selection_target + 1e-12 < target:
            failures.append(f"selection target below final target: selection={selection_target}, final={target}")
        if freeze.get("no_alert_count_limit") is not True:
            failures.append("freeze does not declare no_alert_count_limit=true")
        policy = freeze.get("precision_policy", {})
        if policy.get("kind") not in {"global_threshold", "scope_threshold"}:
            failures.append(f"invalid policy kind: {policy.get('kind')}")
        if policy.get("threshold") is None and not policy.get("scope_thresholds"):
            failures.append("frozen precision threshold missing")
        if "daily_fraction" in policy or "max_alerts_per_day" in policy:
            failures.append("daily alert budget field leaked into V6 policy")
        if recommendation.get("accuracy_definition") != "precision = TP / (TP + FP)":
            failures.append("accuracy definition is not precision")

        calibration_targets = {
            str(key): str(value)
            for key, value in freeze.get("calibration_target_map", {}).items()
        }
        signal_roles = {
            str(key): str(value)
            for key, value in freeze.get("signal_roles", {}).items()
        }
        checks += max(1, len(signal_roles))
        for signal, role in signal_roles.items():
            target_name = calibration_targets.get(signal)
            if role == "crash" and target_name != "crash_d3":
                failures.append(
                    f"crash signal calibration target mismatch: {signal} -> {target_name}"
                )
            if role == "agreement" and target_name != "identity":
                failures.append(
                    f"agreement signal calibration target mismatch: {signal} -> {target_name}"
                )
            if role in {"surge", "strong_surge", "direction_up"} and target_name != "surge_d3":
                failures.append(
                    f"positive signal calibration target mismatch: {signal} -> {target_name}"
                )

        method_model = freeze.get("method_spec", {}).get("model")
        if isinstance(method_model, Mapping) and method_model.get("path"):
            checks += verify_model_record(root, method_model, failures)

        gate_status = str(recommendation.get("status", "UNKNOWN"))
        if gate_status != "READY_FOR_NEW_FUTURE_HOLDOUT" and not args.allow_gate_failed:
            failures.append(
                f"precision gate is not READY: {gate_status}; "
                "무결성만 검증하려면 --allow-gate-failed 사용"
            )
        metrics_path = root / "precision_candidate_metrics_by_role.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
            required_roles = {"selection_forward_eval", "confirmation", "recent_audit"}
            observed = set(metrics.get("evaluation_role", pd.Series(dtype=str)).astype(str))
            checks += len(required_roles)
            missing_roles = required_roles - observed
            if missing_roles:
                failures.append(f"role metrics missing: {sorted(missing_roles)}")
            if gate_status == "READY_FOR_NEW_FUTURE_HOLDOUT":
                minimum_recall = float(freeze.get("minimum_useful_recall", 0.0))
                minimum_lcb = float(freeze.get("minimum_precision_lcb", 0.0))
                for role in required_roles:
                    part = metrics.loc[metrics["evaluation_role"].astype(str).eq(role)]
                    if part.empty:
                        continue
                    row = part.iloc[0]
                    precision = float(row.get("precision", float("nan")))
                    recall = float(row.get("recall", float("nan")))
                    precision_lcb = float(row.get("precision_wilson_lcb", float("nan")))
                    alerts = int(row.get("alerts", 0))
                    if not (precision >= target - 1e-12):
                        failures.append(f"READY gate precision violation: role={role}, precision={precision}")
                    if not (recall >= minimum_recall - 1e-12):
                        failures.append(f"READY gate recall violation: role={role}, recall={recall}")
                    if not (precision_lcb >= minimum_lcb - 1e-12):
                        failures.append(
                            f"READY gate precision LCB violation: role={role}, lcb={precision_lcb}"
                        )
                    minimum_alerts = int(policy.get("minimum_alerts", 1))
                    if alerts < minimum_alerts:
                        failures.append(f"READY gate sample violation: role={role}, alerts={alerts}")

                fold_path = root / "precision_candidate_metrics_by_fold.csv"
                if fold_path.exists():
                    fold_metrics = pd.read_csv(fold_path)
                    required_selection_rate = float(
                        freeze.get("required_selection_fold_pass_rate", 1.0)
                    )
                    required_holdout_rate = float(
                        freeze.get("required_holdout_fold_pass_rate", 1.0)
                    )
                    for role in required_roles:
                        part = fold_metrics.loc[
                            fold_metrics["evaluation_role"].astype(str).eq(role)
                        ]
                        checks += 1
                        if part.empty:
                            failures.append(f"READY fold metrics missing: role={role}")
                            continue
                        pass_rate = float(part["gate_pass"].astype(bool).mean())
                        required_rate = (
                            required_selection_rate
                            if role == "selection_forward_eval"
                            else required_holdout_rate
                        )
                        if pass_rate + 1e-12 < required_rate:
                            failures.append(
                                f"READY fold pass-rate violation: role={role}, "
                                f"observed={pass_rate}, required={required_rate}"
                            )

    controller_path = root / "PRECISION_CONTROLLER_STATE_V6.json"
    if controller_path.exists():
        checks += 1
        controller = load_json(controller_path)
        if not payload_checksum_is_valid(controller):
            failures.append("payload checksum mismatch: PRECISION_CONTROLLER_STATE_V6.json")
        else:
            status_value = str(controller.get("status", "UNKNOWN"))
            if status_value not in {"ACTIVE", "STOP_ONLINE_PRECISION70_GATE"}:
                failures.append(f"invalid controller status: {status_value}")
            active_policy = controller.get("active_policy") or controller.get("current_policy")
            if not isinstance(active_policy, Mapping):
                failures.append("controller active/current policy missing")
            else:
                if "daily_fraction" in active_policy or "max_alerts_per_day" in active_policy:
                    failures.append("alert budget field leaked into controller policy")
            expected_freeze = controller.get("freeze_sha256") or controller.get("base_freeze_sha256")
            freeze_file = root / "PRECISION_FREEZE_V6.json"
            if isinstance(expected_freeze, str) and freeze_file.exists() and sha256_file(freeze_file) != expected_freeze:
                failures.append("controller freeze SHA-256 mismatch")

    registry_path = root / "V6_SPECIALIST_PRODUCTION_REGISTRY.json"
    if registry_path.exists():
        registry = load_json(registry_path)
        for record in registry.get("models", []):
            model = record.get("model")
            if isinstance(model, Mapping):
                checks += verify_model_record(root, model, failures)

    for result_path in root.glob("specialist_task_cache/*/seed_*/fold_*.json"):
        checks += 1
        payload = load_json(result_path)
        if not payload_checksum_is_valid(payload):
            failures.append(f"task payload checksum mismatch: {result_path}")
            continue
        prediction = resolve_path(root, str(payload.get("prediction_path", "")))
        if not prediction.exists():
            # Older task payloads can hold a project-absolute path. Retry next to JSON.
            fallback = result_path.with_suffix(".npz")
            prediction = fallback if fallback.exists() else prediction
        if not prediction.exists():
            failures.append(f"task prediction missing: {prediction}")
        elif payload.get("prediction_sha256") != sha256_file(prediction):
            failures.append(f"task prediction sha256 mismatch: {prediction}")

    inventory_path = root / "OUTPUT_INVENTORY.json"
    if inventory_path.exists():
        inventory = load_json(inventory_path)
        records = inventory.get("files", inventory.get("artifacts", []))
        for record in records:
            checks += 1
            relative = Path(str(record.get("relative_path", "")))
            path = root / relative
            if not path.exists():
                failures.append(f"inventory file missing: {relative}")
                continue
            if int(record.get("bytes", path.stat().st_size)) != int(path.stat().st_size):
                failures.append(f"inventory size mismatch: {relative}")
            if str(record.get("sha256", sha256_file(path))) != sha256_file(path):
                failures.append(f"inventory sha256 mismatch: {relative}")

    report = {
        "schema": "crashwatch_surge_precision70_verification_v6",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
    }
    (root / "VERIFICATION_REPORT_V6.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
