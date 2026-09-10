from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from surge_model_zoo_common import (
    atomic_write_json,
    load_json,
    payload_checksum_is_valid,
    sha256_file,
    utc_now,
)
from surge_model_zoo_deployment import load_registry


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, **details: Any) -> None:
    checks.append({"name": name, "passed": bool(passed), **details})


def verify_inventory(output: Path, manifest: dict[str, Any], checks: list[dict[str, Any]]) -> None:
    inventory = manifest.get("output_inventory", [])
    for record in inventory:
        relative = str(record["relative_path"])
        path = output / relative
        exists = path.exists() and path.is_file()
        size_ok = exists and int(path.stat().st_size) == int(record["bytes"])
        hash_ok = size_ok and sha256_file(path) == str(record["sha256"])
        add_check(
            checks,
            f"inventory:{relative}",
            bool(exists and size_ok and hash_ok),
            exists=exists,
            size_ok=size_ok,
            hash_ok=hash_ok,
        )


def verify_task_cache(output: Path, checks: list[dict[str, Any]]) -> dict[str, int]:
    result_files = sorted((output / "task_cache").glob("*/seed_*/fold_*.json"))
    valid_count = 0
    for result_path in result_files:
        payload = load_json(result_path)
        relative = str(result_path.relative_to(output)).replace("\\", "/")
        checksum_ok = payload_checksum_is_valid(payload)
        prediction_raw = payload.get("prediction_path")
        if prediction_raw:
            prediction_path = Path(str(prediction_raw))
            if not prediction_path.is_absolute():
                prediction_path = output / prediction_path
        else:
            prediction_path = result_path.with_suffix(".npz")
        exists = prediction_path.exists()
        hash_ok = exists and payload.get("prediction_sha256") == sha256_file(prediction_path)
        passed = payload.get("status") == "completed" and checksum_ok and hash_ok
        valid_count += int(passed)
        add_check(
            checks,
            f"task:{relative}",
            passed,
            checksum_ok=checksum_ok,
            prediction_exists=exists,
            prediction_hash_ok=hash_ok,
        )
    return {"task_result_files": len(result_files), "valid_task_results": valid_count}


def verify_production_registry(output: Path, checks: list[dict[str, Any]]) -> dict[str, int]:
    path = output / "PRODUCTION_MODEL_REGISTRY.json"
    if not path.exists():
        add_check(checks, "production_registry", False, reason="missing")
        return {"production_models": 0, "valid_production_models": 0}
    registry = load_registry(path)
    registry_checksum = payload_checksum_is_valid(registry)
    add_check(checks, "production_registry_checksum", registry_checksum)
    total = 0
    valid = 0
    for recipe_record in registry.get("models", []):
        for key in ("model", "large_move_model", "direction_model"):
            model = recipe_record.get(key)
            if not isinstance(model, dict) or model.get("format") == "constant_probability":
                continue
            total += 1
            raw_path = Path(str(model.get("path", "")))
            model_path = raw_path if raw_path.is_absolute() else output / raw_path
            exists = model_path.exists()
            size_ok = exists and int(model_path.stat().st_size) == int(model.get("bytes", -1))
            hash_ok = size_ok and sha256_file(model_path) == model.get("sha256")
            passed = bool(exists and size_ok and hash_ok)
            valid += int(passed)
            add_check(
                checks,
                f"production_model:{recipe_record.get('recipe')}:{recipe_record.get('seed')}:{key}",
                passed,
                path=str(model_path),
                exists=exists,
                size_ok=size_ok,
                hash_ok=hash_ok,
            )
    return {"production_models": total, "valid_production_models": valid}


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch Surge Model Zoo V4 결과 무결성 검증")
    parser.add_argument("--output", type=Path, default=Path("outputs/surge_model_zoo_v4"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    report_path = args.report.resolve() if args.report else output / "VERIFICATION_REPORT.json"
    checks: list[dict[str, Any]] = []

    status_path = output / "RUN_STATUS.json"
    status = load_json(status_path) if status_path.exists() else {}
    add_check(checks, "run_status_success", status.get("status") == "SUCCESS", status=status.get("status"))

    manifest_path = output / "MODEL_FREEZE_MANIFEST.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    add_check(checks, "model_freeze_manifest_exists", bool(manifest), path=str(manifest_path))
    if manifest:
        verify_inventory(output, manifest, checks)

    freeze_path = output / "ENSEMBLE_FREEZE.json"
    freeze = load_json(freeze_path) if freeze_path.exists() else {}
    add_check(
        checks,
        "ensemble_freeze_checksum",
        bool(freeze) and payload_checksum_is_valid(freeze),
        path=str(freeze_path),
    )

    task_counts = verify_task_cache(output, checks)
    production_counts = verify_production_registry(output, checks)
    failures = [record for record in checks if not record["passed"]]
    report = {
        "schema": "crashwatch_surge_model_zoo_verification_v4",
        "created_at": utc_now(),
        "output": str(output),
        "status": "PASS" if not failures else "FAIL",
        "check_count": len(checks),
        "failure_count": len(failures),
        **task_counts,
        **production_counts,
        "failures": failures,
        "checks": checks,
    }
    atomic_write_json(report_path, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"checks"}}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
