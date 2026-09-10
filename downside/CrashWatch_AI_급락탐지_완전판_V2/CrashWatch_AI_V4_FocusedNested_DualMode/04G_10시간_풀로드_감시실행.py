#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _tail(path: Path, limit: int = 12_000) -> str:
    try:
        data = path.read_bytes()
        return data[-limit:].decode("utf-8", errors="replace")
    except Exception:
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="FocusedNested 풀로드/게임모드 감시 실행기")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--profile", choices=("game", "full"), default="full")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument(
        "--finish-all",
        action="store_true",
        help="시간 제한 없이 계획된 모든 task가 끝날 때까지 실행",
    )
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--max-restarts", type=int, default=8)
    parser.add_argument(
        "--seeds",
        default="17,43,79,101,137,211,307,419,541",
    )
    args = parser.parse_args()

    project = args.project.resolve()
    threads = args.threads if args.threads is not None else (4 if args.profile == "game" else 16)
    if args.profile == "game" and threads > 4:
        parser.error("game profile은 4스레드를 초과할 수 없습니다.")
    result = project / "crashwatch_ai_data" / "ablation_finance_nested_focus"
    result.mkdir(parents=True, exist_ok=True)
    state_path = result / (
        "game_finish_all_supervisor_state.json"
        if args.finish_all and args.profile == "game"
        else "full10h_supervisor_state.json"
    )
    deadline = None if args.finish_all else time.time() + args.hours * 3600
    started_at = time.time()
    attempt = 0
    restart_count = 0
    final_reason = "unknown"

    while deadline is None or time.time() < deadline:
        remaining_hours = (
            0.0 if deadline is None
            else max(0.0, (deadline - time.time()) / 3600)
        )
        if deadline is not None and remaining_hours < 0.02:
            final_reason = "campaign_deadline_reached"
            break
        attempt += 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        stdout_path = result / f"full10h_attempt_{attempt:02d}_{stamp}_stdout.log"
        stderr_path = result / f"full10h_attempt_{attempt:02d}_{stamp}_stderr.log"
        command = [
            sys.executable,
            str(project / "04E_집중Nested_이탈테스트.py"),
            "--project",
            str(project),
            "--profile",
            args.profile,
            "--threads",
            str(threads),
            "--hours",
            "0" if deadline is None else f"{remaining_hours:.8f}",
            "--folds",
            "8",
            "--seeds",
            args.seeds,
            "--inner-folds",
            "3",
            "--validation-days",
            "60",
            "--purge-days",
            "20",
            "--calibration-days",
            "60",
            "--calibration-purge-days",
            "20",
            "--cache-namespace",
            "finance_nested_focus_v1",
            "--clear-stop-on-start",
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "-1" if args.profile == "game" else "0"
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
            process = subprocess.Popen(
                command,
                cwd=project,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
            )
            while process.poll() is None:
                run_state = _read_json(result / "run_state.json")
                _atomic_json(
                    {
                        "status": "running",
                        "supervisor_pid": os.getpid(),
                        "child_pid": process.pid,
                        "attempt": attempt,
                        "restart_count": restart_count,
                        "campaign_started_at_epoch": started_at,
                        "campaign_deadline_epoch": deadline,
                        "finish_all": args.finish_all,
                        "profile": args.profile,
                        "threads": threads,
                        "remaining_hours": (
                            None if deadline is None
                            else max(0.0, (deadline - time.time()) / 3600)
                        ),
                        "child_stage": run_state.get("stage"),
                        "child_outer_fold": run_state.get("outer_fold"),
                        "child_experiment": run_state.get("experiment"),
                        "child_seed": run_state.get("seed"),
                        "stdout": str(stdout_path),
                        "stderr": str(stderr_path),
                        "updated_at_epoch": time.time(),
                    },
                    state_path,
                )
                if deadline is not None and time.time() >= deadline:
                    (result / "REQUEST_SAFE_STOP.flag").touch()
                time.sleep(max(5.0, args.poll_seconds))
            return_code = int(process.returncode or 0)

        summary = _read_json(result / "run_summary.json")
        if (
            float(summary.get("progress_ratio", 0.0)) >= 1.0
            and not summary.get("safe_stop", False)
        ):
            final_reason = "all_planned_experiments_completed"
            break
        if (
            deadline is not None
            and (time.time() >= deadline or summary.get("deadline_reached"))
        ):
            final_reason = "campaign_deadline_reached"
            break
        if restart_count >= args.max_restarts:
            final_reason = "restart_limit_reached"
            break

        restart_count += 1
        _atomic_json(
            {
                "status": "restarting",
                "supervisor_pid": os.getpid(),
                "attempt": attempt,
                "restart_count": restart_count,
                "child_return_code": return_code,
                "summary": summary,
                "stderr_tail": _tail(stderr_path),
                "remaining_hours": (
                    None if deadline is None
                    else max(0.0, (deadline - time.time()) / 3600)
                ),
                "updated_at_epoch": time.time(),
            },
            state_path,
        )
        # 온도/RAM 안전 종료와 일시적인 파일 잠금을 모두 고려해 잠깐 식힌 후 재개한다.
        time.sleep(min(120.0, max(15.0, args.poll_seconds * 2)))

    _atomic_json(
        {
            "status": "finished",
            "supervisor_pid": os.getpid(),
            "attempts": attempt,
            "restart_count": restart_count,
            "final_reason": final_reason,
            "campaign_started_at_epoch": started_at,
            "campaign_deadline_epoch": deadline,
            "elapsed_hours": (time.time() - started_at) / 3600,
            "run_summary": _read_json(result / "run_summary.json"),
            "updated_at_epoch": time.time(),
        },
        state_path,
    )


if __name__ == "__main__":
    main()
