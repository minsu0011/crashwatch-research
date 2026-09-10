from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
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
    payload = {"timestamp_kst": datetime.now().astimezone().isoformat(), **payload}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def run_verifier_and_package(output: Path, health_log: Path) -> int:
    verify = subprocess.run(
        [sys.executable, str(HERE / "verify_surge_precision_gate_v12_7h.py"), "--output", str(output)],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    append_event(health_log, {"event": "verifier", "returncode": verify.returncode, "stdout": verify.stdout, "stderr": verify.stderr})
    if verify.returncode:
        return verify.returncode
    return verify.returncode


def restart_with_remaining_budget(output: Path, original_deadline: datetime, health_log: Path) -> subprocess.Popen | None:
    remaining = (original_deadline - datetime.now().astimezone()).total_seconds() / 3600.0
    if remaining < 0.25:
        append_event(health_log, {"event": "restart_skipped", "reason": "less_than_15_minutes_remaining"})
        return None
    checks = subprocess.run(["cmd.exe", "/d", "/c", str(HERE / "run_checks_7h.bat")], cwd=HERE, capture_output=True, text=True)
    append_event(health_log, {"event": "post_failure_checks", "returncode": checks.returncode, "stdout": checks.stdout, "stderr": checks.stderr})
    if checks.returncode:
        return None
    root = WORKSPACE
    args = [
        sys.executable,
        str(HERE / "run_surge_precision_gate_v12_7h.py"),
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
        "--require-gpu",
        "--search-seeds-per-family", "3",
        "--top-families", "8",
        "--robust-seeds", "13",
        "--max-ab-features-per-ticker", "24",
        "--minimum-alerts", "30",
        "--target-precision", "0.70",
        "--resume",
    ]
    stdout = (HERE / "runtime_logs" / "official_7h_retry.stdout.log").open("a", encoding="utf-8")
    stderr = (HERE / "runtime_logs" / "official_7h_retry.stderr.log").open("a", encoding="utf-8")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "16", "MKL_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "16", "NUMEXPR_NUM_THREADS": "16", "PYTHONUNBUFFERED": "1"})
    proc = subprocess.Popen(args, cwd=HERE, stdout=stdout, stderr=stderr, env=env, creationflags=0x08000000)
    append_event(health_log, {"event": "automatic_resume", "pid": proc.pid, "remaining_hours": remaining, "cpu_threads": 16})
    return proc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-pid", type=int, required=True)
    parser.add_argument("--launcher-pid", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=300)
    parser.add_argument("--original-start-kst", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    log_dir = HERE / "runtime_logs"
    log_dir.mkdir(exist_ok=True)
    health_log = log_dir / "watch_v12_7h_health.jsonl"
    original_start = datetime.fromisoformat(args.original_start_kst)
    deadline = original_start + timedelta(hours=7)
    main_pid = int(args.main_pid)
    launcher_pid = int(args.launcher_pid)
    retries = 0
    append_event(health_log, {"event": "watch_started", "main_pid": main_pid, "launcher_pid": launcher_pid, "deadline": deadline.isoformat()})
    while True:
        status = load_json(output / "RUN_STATUS.json")
        progress = load_json(output / "PROGRESS_V12_7H.json")
        alive = pid_alive(main_pid)
        append_event(
            health_log,
            {
                "event": "heartbeat",
                "main_pid": main_pid,
                "main_alive": alive,
                "launcher_alive": pid_alive(launcher_pid) if launcher_pid else False,
                "run_status": status.get("status"),
                "phase": progress.get("phase"),
                "elapsed_hours": progress.get("elapsed_hours"),
                "families_completed": progress.get("families_completed"),
            },
        )
        if status.get("status") == "SUCCESS":
            rc = run_verifier_and_package(output, health_log)
            append_event(health_log, {"event": "watch_complete", "returncode": rc})
            return rc
        if not alive:
            # Allow the process a moment to atomically replace RUN_STATUS.json.
            time.sleep(10)
            status = load_json(output / "RUN_STATUS.json")
            if status.get("status") == "SUCCESS":
                continue
            append_event(health_log, {"event": "unexpected_stop", "status": status})
            if retries < 1:
                retry = restart_with_remaining_budget(output, deadline, health_log)
                if retry is not None:
                    retries += 1
                    main_pid = retry.pid
                    launcher_pid = 0
                    continue
            append_event(health_log, {"event": "watch_failed", "retries": retries})
            return 2
        time.sleep(max(60, int(args.interval_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
