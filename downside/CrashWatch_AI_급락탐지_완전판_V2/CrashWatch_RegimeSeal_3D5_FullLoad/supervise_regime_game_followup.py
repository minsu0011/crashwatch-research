from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Supervise the RegimeSeal game-mode follow-up")
    parser.add_argument("--selection-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--budget-seconds", type=float, default=18000.0)
    parser.add_argument("--max-restarts", type=int, default=1)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(package / "run_regime_game_followup.py"),
        "--selection-output", str(Path(args.selection_output).expanduser().resolve()),
        "--output", str(output),
        "--config", str(Path(args.config).expanduser().resolve()),
        "--cpu-threads", "8",
        "--gpu-sleep-multiplier", "3.0",
        "--budget-seconds", str(float(args.budget_seconds)),
        "--seeds", "307,701,997",
        "--bootstrap-reps", "300",
    ]
    attempts = []
    for attempt in range(1, max(1, int(args.max_restarts) + 1) + 1):
        if (output / "STOP_GAME_FOLLOWUP").exists():
            return 3
        started = time.time()
        print(f"GAME_SUPERVISOR attempt={attempt} command={command}", flush=True)
        completed = subprocess.run(command, cwd=package, env=os.environ.copy(), check=False)
        final = _read(output / "FINAL_GAME_FOLLOWUP_STATUS.json")
        ok = bool(
            completed.returncode == 0
            and final.get("status") == "completed"
            and int(final.get("actual_model_fits", 0)) == int(final.get("expected_model_fits", -1))
            and final.get("result_export", {}).get("verified") is True
        )
        attempts.append({"attempt": attempt, "returncode": completed.returncode, "elapsed_seconds": time.time() - started, "complete": ok})
        (output / "GAME_SUPERVISOR_ATTEMPTS.json").write_text(json.dumps(attempts, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if ok:
            print(json.dumps({"status": "completed", "final": final}, ensure_ascii=False), flush=True)
            return 0
        budget = _read(output / "GAME_BUDGET.json")
        if time.time() >= float(budget.get("deadline_epoch", 0)) - 900.0 or attempt > int(args.max_restarts):
            break
        time.sleep(20.0)
    failure = {"status": "failed_after_retries_or_budget", "attempts": attempts, "created_epoch": time.time()}
    (output / "SUPERVISOR_FAILED.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
