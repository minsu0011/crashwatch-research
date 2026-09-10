#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import psutil


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    project = Path(__file__).resolve().parent
    result = project / "crashwatch_ai_data" / "ablation_finance_nested_focus"
    lock = _read_json(result / "RUNNING.lock")
    state = _read_json(result / "run_state.json")
    summary = _read_json(result / "run_summary.json")
    runtime = _read_json(result / "runtime_profile_latest.json")

    pid = int(lock.get("pid", -1))
    running = pid > 0 and psutil.pid_exists(pid)
    completed = 0
    cache_hits = 0
    manifest_path = result / "task_manifest.csv"
    if manifest_path.exists():
        try:
            manifest = pd.read_csv(manifest_path)
            if "task_id" in manifest:
                manifest = manifest.drop_duplicates("task_id", keep="last")
            completed = int(manifest.get("status", pd.Series(dtype=str)).eq("completed").sum())
            cache_hits = int(manifest.get("cache_status", pd.Series(dtype=str)).eq("hit").sum())
        except Exception:
            pass

    planned = int(state.get("total_planned_outer_tasks", summary.get("planned_outer_tasks", 240)))
    payload = {
        "running": running,
        "pid": pid if running else None,
        "profile": lock.get("profile", state.get("profile")),
        "stage": state.get("stage"),
        "outer_fold": state.get("outer_fold"),
        "experiment": state.get("experiment"),
        "seed": state.get("seed"),
        "elapsed_hours": state.get("elapsed_hours"),
        "completed_outer_tasks": completed,
        "planned_outer_tasks": planned,
        "progress_ratio": completed / planned if planned else 0.0,
        "cache_hits": cache_hits,
        "runtime": runtime,
        "last_update": state.get("updated_at"),
        "safe_stop_requested": state.get("safe_stop_requested", False),
        "result_dir": str(result),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
