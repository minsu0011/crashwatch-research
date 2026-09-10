from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dual_ablation.base12h.registry import TaskRegistry
from dual_ablation.config import get_paths


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Absolute-deadline Base12H watchdog")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--profile", choices=("pubg", "full"), default="full")
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--max-restarts", type=int, default=6)
    args = parser.parse_args()

    project = args.project.resolve()
    result_dir = get_paths(project).data_root / "ablation_base12h"
    result_dir.mkdir(parents=True, exist_ok=True)
    registry = TaskRegistry(result_dir / "task_registry.sqlite3")
    stop_flag = result_dir / "REQUEST_SAFE_STOP.flag"
    state_path = result_dir / "base12h_watchdog_state.json"
    deadline = time.time() + max(0.1, args.hours) * 3600
    campaign_started = time.time()
    restarts = 0
    attempt = 0
    final_reason = "unknown"

    while time.time() < deadline and restarts <= args.max_restarts:
        attempt += 1
        remaining_hours = max(0.1, (deadline - time.time()) / 3600)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stdout_path = result_dir / f"watchdog_attempt_{attempt:02d}_{stamp}_stdout.log"
        stderr_path = result_dir / f"watchdog_attempt_{attempt:02d}_{stamp}_stderr.log"
        command = [
            sys.executable,
            str(project / "run_base12h.py"),
            "--project",
            str(project),
            "--profile",
            args.profile,
            "--hours",
            f"{remaining_hours:.8f}",
            "--workers",
            str(max(1, args.workers)),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "-1" if args.profile == "pubg" else "0"
        thread_limit = "4" if args.profile == "pubg" else "16"
        for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
        ):
            environment[name] = thread_limit
        environment["PYTHONHASHSEED"] = "0"
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        attempt_started = time.time()
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            child = subprocess.Popen(
                command,
                cwd=project,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
            )
            while child.poll() is None:
                status = registry.status()
                atomic_json(
                    {
                        "status": "running",
                        "watchdog_pid": os.getpid(),
                        "child_pid": child.pid,
                        "attempt": attempt,
                        "restart_count": restarts,
                        "profile": args.profile,
                        "workers": args.workers,
                        "campaign_started_at_epoch": campaign_started,
                        "campaign_deadline_epoch": deadline,
                        "remaining_hours": max(0.0, (deadline - time.time()) / 3600),
                        "registry": status,
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                        "stderr_bytes": stderr_path.stat().st_size,
                        "updated_at_epoch": time.time(),
                    },
                    state_path,
                )
                if time.time() >= deadline and not stop_flag.exists():
                    stop_flag.write_text("watchdog campaign deadline\n", encoding="utf-8")
                time.sleep(max(5.0, args.poll_seconds))
            child_return_code = int(child.returncode or 0)

        status = registry.status()
        summary_path = result_dir / "run_summary.json"
        summary = (
            read_json(summary_path)
            if summary_path.exists() and summary_path.stat().st_mtime >= attempt_started
            else {}
        )
        stop_reason = stop_flag.read_text(encoding="utf-8", errors="replace").strip() if stop_flag.exists() else ""
        worker_exit_codes = [int(code) for code in summary.get("exit_codes", []) if code is not None]
        if status["blocks_total"] and status["blocks_completed"] >= status["blocks_total"]:
            final_reason = "all_blocks_completed"
            break
        if time.time() >= deadline or "deadline" in stop_reason.lower():
            final_reason = "campaign_deadline_reached"
            break
        if stop_reason and ("user" in stop_reason.lower() or "resource" in stop_reason.lower()):
            final_reason = f"honored_safe_stop:{stop_reason}"
            break
        failed = child_return_code != 0 or any(code != 0 for code in worker_exit_codes)
        if not failed and summary.get("safe_stop"):
            final_reason = f"safe_stop:{stop_reason or 'unspecified'}"
            break
        if restarts >= args.max_restarts:
            final_reason = "restart_limit_reached"
            break
        restarts += 1
        atomic_json(
            {
                "status": "restarting",
                "watchdog_pid": os.getpid(),
                "attempt": attempt,
                "restart_count": restarts,
                "child_return_code": child_return_code,
                "worker_exit_codes": worker_exit_codes,
                "stop_reason": stop_reason,
                "registry": status,
                "remaining_hours": max(0.0, (deadline - time.time()) / 3600),
                "stderr": str(stderr_path),
                "stderr_bytes": stderr_path.stat().st_size,
                "updated_at_epoch": time.time(),
            },
            state_path,
        )
        time.sleep(min(60.0, max(10.0, args.poll_seconds * 2)))

    atomic_json(
        {
            "status": "finished",
            "watchdog_pid": os.getpid(),
            "attempts": attempt,
            "restart_count": restarts,
            "final_reason": final_reason,
            "campaign_started_at_epoch": campaign_started,
            "campaign_deadline_epoch": deadline,
            "elapsed_hours": (time.time() - campaign_started) / 3600,
            "registry": registry.status(),
            "run_summary": read_json(result_dir / "run_summary.json"),
            "updated_at_epoch": time.time(),
        },
        state_path,
    )


if __name__ == "__main__":
    main()
