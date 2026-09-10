from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_is_complete(output: Path) -> bool:
    final = read_json(output / "FINAL_RUN_STATUS.json")
    audit = read_json(output / "completion_audit.json")
    return bool(
        final.get("status") == "completed"
        and audit.get("stage_statuses_ok") is True
        and audit.get("model_counts_ok") is True
        and int(audit.get("actual_lightgbm_models", 0)) >= int(audit.get("expected_lightgbm_models", 1))
        and int(audit.get("actual_xgboost_models", 0)) >= int(audit.get("expected_xgboost_models", 1))
    )


def export_result_zip(package_root: Path, output: Path) -> dict[str, Any]:
    desktop = Path.home() / "Desktop"
    desktop.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = desktop / f"CrashWatch_FinalSelection_RESULTS_{stamp}.zip"
    final = read_json(output / "FINAL_RUN_STATUS.json")
    audit = read_json(output / "completion_audit.json")
    selection = read_json(output / "results" / "FINAL_SELECTION_SUMMARY.json")
    lock = read_json(output / "results" / "LOCKED_PROFILE.json")
    start_here = "\n".join(
        [
            "# CrashWatch FinalSelection FullLoad 결과",
            "",
            f"- pipeline_status: {final.get('status')}",
            f"- elapsed_hours: {float(final.get('elapsed_seconds', 0.0)) / 3600.0:.3f}",
            f"- LightGBM: {audit.get('actual_lightgbm_models')}/{audit.get('expected_lightgbm_models')}",
            f"- XGBoost: {audit.get('actual_xgboost_models')}/{audit.get('expected_xgboost_models')}",
            f"- locked_profile: {lock.get('locked_profile')}",
            f"- lock_status: {lock.get('status')}",
            f"- sealed_evaluation_allowed: {lock.get('sealed_evaluation_allowed')}",
            f"- sealed_data_used_in_selection: {selection.get('sealed_data_used')}",
            "",
            "먼저 results/FINAL_RESULT_GUIDE_KO.txt, results/FINAL_SELECTION_SUMMARY.json,",
            "results/LOCKED_PROFILE.json, results/P2_CAUSAL_DECOMPOSITION.json을 읽으세요.",
            "이 실험은 피처/프로필 선택 실험이며 sealed 평가는 실행하지 않았습니다.",
            "task_results의 개별 모델 JSON은 용량 때문에 제외했으며 집계 CSV와 실행 감사 파일은 포함했습니다.",
            "",
        ]
    )

    files: list[tuple[Path, str]] = []
    excluded_output_files = {"SUPERVISOR_FAILED.json", "STOP_SUPERVISOR"}
    for path in sorted(output.iterdir()):
        if path.is_file() and path.name not in excluded_output_files:
            files.append((path, f"output/{path.name}"))
    results_dir = output / "results"
    if results_dir.exists():
        for path in sorted(results_dir.rglob("*")):
            if path.is_file():
                files.append((path, f"results/{path.relative_to(results_dir).as_posix()}"))
    task_root = output / "task_results"
    if task_root.exists():
        for path in sorted(task_root.rglob("*_stage_summary.json")):
            files.append((path, f"stage_summaries/{path.relative_to(task_root).as_posix()}"))
    for relative in [
        "config_full_load.json",
        "README_KO.md",
        "EXPERIMENT_DESIGN_KO.md",
        "EXPERIMENT_MANIFEST_PREVIEW.json",
        "CODE_AUDIT.md",
        "VERSION.txt",
        "run_full_selection.py",
        "supervise_full_selection.py",
    ]:
        path = package_root / relative
        if path.exists():
            files.append((path, f"code_context/{relative}"))
    for path in sorted((package_root / "cwfull").glob("*.py")):
        files.append((path, f"code_context/cwfull/{path.name}"))

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        zipped.writestr("00_START_HERE.md", start_here.encode("utf-8"))
        seen: set[str] = set()
        for source, arcname in files:
            if arcname in seen:
                continue
            seen.add(arcname)
            zipped.write(source, arcname)
    with zipfile.ZipFile(archive, "r") as zipped:
        bad = zipped.testzip()
        names = set(zipped.namelist())
    required = {
        "00_START_HERE.md",
        "results/FINAL_SELECTION_SUMMARY.json",
        "results/LOCKED_PROFILE.json",
        "results/P2_CAUSAL_DECOMPOSITION.json",
    }
    if bad is not None or not required.issubset(names):
        raise RuntimeError(f"Result archive verification failed: bad={bad}, missing={sorted(required - names)}")
    digest = sha256(archive)
    hash_path = archive.with_suffix(archive.suffix + ".sha256.txt")
    hash_path.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
    status = {
        "status": "completed",
        "archive": str(archive),
        "size_bytes": archive.stat().st_size,
        "size_mb": archive.stat().st_size / 1024**2,
        "sha256": digest,
        "entries": len(names),
        "verified": True,
        "created_epoch": time.time(),
    }
    (output / "RESULT_EXPORT_STATUS.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise CrashWatch full selection and export results")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--max-restarts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=20.0)
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parent
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stop_marker = output / "STOP_SUPERVISOR"
    command = [
        sys.executable,
        str(package_root / "run_full_selection.py"),
        "--project-root",
        str(Path(args.project_root).expanduser().resolve()),
        "--dataset",
        str(Path(args.dataset).expanduser().resolve()),
        "--output",
        str(output),
        "--stages",
        "core,targeted,diagnostic,aggregate",
    ]
    if args.config:
        command.extend(["--config", str(Path(args.config).expanduser().resolve())])
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max(1, args.max_restarts + 1) + 1):
        if stop_marker.exists():
            return 3
        started = time.time()
        print(f"SUPERVISOR attempt={attempt} command={command}", flush=True)
        completed = subprocess.run(command, cwd=package_root, env=os.environ.copy(), check=False)
        record = {
            "attempt": attempt,
            "returncode": completed.returncode,
            "elapsed_seconds": time.time() - started,
            "result_complete": result_is_complete(output),
        }
        attempts.append(record)
        (output / "supervisor_attempts.json").write_text(
            json.dumps(attempts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        if completed.returncode == 0 and record["result_complete"]:
            export = export_result_zip(package_root, output)
            print(json.dumps({"status": "completed", "export": export}, ensure_ascii=False), flush=True)
            return 0
        if attempt > args.max_restarts or stop_marker.exists():
            break
        print(f"SUPERVISOR retrying in {args.retry_delay_seconds:.1f}s", flush=True)
        time.sleep(max(1.0, args.retry_delay_seconds))
    failure = {"status": "failed_after_retries", "attempts": attempts, "created_epoch": time.time()}
    (output / "SUPERVISOR_FAILED.json").write_text(
        json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
