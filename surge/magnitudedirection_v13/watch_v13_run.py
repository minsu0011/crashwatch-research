from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent


def pid_alive(pid: int) -> bool:
    handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)
    return True


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def append_event(path: Path, payload: dict) -> None:
    row = {"timestamp_kst": datetime.now().astimezone().isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_handoff(output: Path, health_log: Path) -> int:
    deadline = time.monotonic() + 120
    candidates: list[Path] = []
    while time.monotonic() < deadline:
        candidates = sorted(
            (Path.home() / "Desktop").glob("magnitude-direction-results-*.zip"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            break
        time.sleep(5)
    if not candidates:
        append_event(health_log, {"event": "handoff_missing"})
        return 3
    package = candidates[0]
    with zipfile.ZipFile(package) as archive:
        bad_member = archive.testzip()
        members = len(archive.namelist())
    result = {
        "status": "SUCCESS" if bad_member is None else "CRC_FAILED",
        "path": str(package),
        "bytes": package.stat().st_size,
        "members": members,
        "bad_member": bad_member,
        "sha256": file_hash(package),
        "audited_kst": datetime.now().astimezone().isoformat(),
    }
    (output / "archive_status.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    append_event(health_log, {"event": "handoff_audit", **result})
    return 0 if bad_member is None else 4


def restart_with_remaining_budget(output: Path, original_deadline: datetime, health_log: Path) -> subprocess.Popen | None:
    remaining = (original_deadline - datetime.now().astimezone()).total_seconds() / 3600.0
    if remaining < 0.25:
        append_event(health_log, {"event": "restart_skipped", "reason": "budget_nearly_exhausted"})
        return None
    checks = subprocess.run(["cmd.exe", "/d", "/c", str(HERE / "run_checks_v13.bat")], cwd=HERE, capture_output=True, text=True)
    append_event(health_log, {"event": "post_failure_checks", "returncode": checks.returncode, "stdout": checks.stdout, "stderr": checks.stderr})
    if checks.returncode:
        return None
    root = WORKSPACE
    args = [
        sys.executable,
        str(HERE / "run_surge_magnitude_direction_v13.py"),
        "--package-root", str(root),
        "--v10-output", str(root / "CrashWatch_Surge_Tickerwise_Correlation_Map_V10" / "outputs" / "surge_tickerwise_correlation_map_v10"),
        "--v10-2-output", str(root / "CrashWatch_Surge_Tickerwise_Correlation_Map_V10_2_Patch_20260815" / "outputs" / "surge_tickerwise_correlation_map_v10_2"),
        "--dataset", str(root / "CrashWatch_Surge_3D5_Reference_Package" / "data" / "training_dataset_finance11h.parquet"),
        "--target-sidecar", str(root / "CrashWatch_Surge_3D5_Reference_Package" / "data" / "surge_target_3d5.parquet"),
        "--folds", str(root / "CrashWatch_Surge_Correlation_Map_Complete_V2_Patch_20260809" / "outputs" / "surge_correlation_map_complete" / "walk_forward_folds.json"),
        "--feature-profile-manifest", str(root / "CrashWatch_Surge_AllFeature_Ablation_V3_Patch_20260809" / "outputs" / "surge_pre_model_gate_v3" / "surge_feature_profiles_corrected.json"),
        "--feature-profile", "P0_ALL_VALID",
        "--output", str(output),
        "--target-hours", f"{remaining:.6f}",
        "--cpu-threads", "16",
        "--expected-target-valid-rows", "91775",
        "--expected-ticker-count", "48",
        "--require-full-439",
        "--require-full-base-oof",
        "--require-gpu",
        "--minimum-alerts", "30",
        "--target-precision", "0.70",
        "--resume",
    ]
    retry_stdout = (HERE / "runtime_logs" / "official_v13_retry.stdout.log").open("a", encoding="utf-8")
    retry_stderr = (HERE / "runtime_logs" / "official_v13_retry.stderr.log").open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16", "NUMEXPR_MAX_THREADS": "16", "PYTHONHASHSEED": "13013"})
    process = subprocess.Popen(args, cwd=HERE, stdout=retry_stdout, stderr=retry_stderr, env=env, creationflags=0x08000000)
    append_event(health_log, {"event": "automatic_resume", "pid": process.pid, "remaining_hours": remaining, "cpu_threads": 16})
    return process


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-pid", type=int, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-start-kst", required=True)
    parser.add_argument("--interval-seconds", type=int, default=300)
    args = parser.parse_args()
    output = args.output.resolve()
    log_dir = HERE / "runtime_logs"
    log_dir.mkdir(exist_ok=True)
    health_log = log_dir / "watch_v13_health.jsonl"
    original_start = datetime.fromisoformat(args.original_start_kst)
    original_deadline = original_start + timedelta(hours=7)
    main_pid = args.main_pid
    launcher_pid = args.launcher_pid
    retries = 0
    append_event(health_log, {"event": "watch_started", "main_pid": main_pid, "launcher_pid": launcher_pid, "deadline": original_deadline.isoformat()})
    while True:
        status = load_json(output / "RUN_STATUS.json")
        alive = pid_alive(main_pid)
        append_event(
            health_log,
            {
                "event": "heartbeat",
                "main_pid": main_pid,
                "main_alive": alive,
                "launcher_alive": pid_alive(launcher_pid) if launcher_pid else False,
                "run_status": status.get("status"),
                "latest_output_mtime": max((p.stat().st_mtime for p in output.rglob("*") if p.is_file()), default=None),
            },
        )
        if status.get("status") == "SUCCESS_VERIFIED":
            if alive:
                time.sleep(10)
            result = audit_handoff(output, health_log)
            append_event(health_log, {"event": "watch_complete", "returncode": result})
            return result
        if not alive:
            time.sleep(10)
            status = load_json(output / "RUN_STATUS.json")
            if status.get("status") == "SUCCESS_VERIFIED":
                continue
            append_event(health_log, {"event": "unexpected_stop", "status": status})
            if retries < 1:
                retry = restart_with_remaining_budget(output, original_deadline, health_log)
                if retry is not None:
                    retries += 1
                    main_pid = retry.pid
                    launcher_pid = 0
                    continue
            append_event(health_log, {"event": "watch_failed", "retries": retries})
            return 2
        time.sleep(max(60, args.interval_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
