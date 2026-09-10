from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd
import psutil

from dual_ablation.config import get_paths


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> None:
    project = Path(__file__).resolve().parent
    paths = get_paths(project)
    result = paths.data_root / "ablation_base12h"
    state = read_json(result / "base12h_watchdog_state.json")
    plan = read_json(paths.configs / "base12h_plan.json")
    experiments_per_new_block = 1 + len(plan.get("global_groups", [])) + len(plan.get("bucket_combinations", []))
    completed_tasks = running_tasks = completed_blocks = total_blocks = completed_block_tasks = 0
    db_path = result / "task_registry.sqlite3"
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            completed_tasks = int(conn.execute("SELECT COUNT(*) FROM tasks WHERE status='completed'").fetchone()[0])
            running_tasks = int(conn.execute("SELECT COUNT(*) FROM tasks WHERE status='running'").fetchone()[0])
            completed_blocks = int(conn.execute("SELECT COUNT(*) FROM blocks WHERE status='completed'").fetchone()[0])
            total_blocks = int(conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0])
            completed_block_tasks = int(conn.execute(
                "SELECT COUNT(*) FROM tasks t JOIN blocks b ON b.block_id=t.block_id "
                "WHERE b.status='completed' AND t.status='completed'"
            ).fetchone()[0])
    expected_tasks = completed_block_tasks + max(0, total_blocks - completed_blocks) * experiments_per_new_block
    last_resource: dict = {}
    resource_path = result / "resource_usage.csv"
    if resource_path.exists():
        try:
            last_resource = pd.read_csv(resource_path).tail(1).to_dict("records")[0]
        except Exception:
            pass
    worker_errors = list((result / "worker_errors").glob("*.txt"))
    payload = {
        "time": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "status": state.get("status"),
        "watchdog_alive": psutil.pid_exists(int(state.get("watchdog_pid", -1))),
        "child_alive": psutil.pid_exists(int(state.get("child_pid", -1))),
        "attempt": state.get("attempt"),
        "restarts": state.get("restart_count"),
        "blocks_completed": completed_blocks,
        "blocks_total": total_blocks,
        "tasks_completed": completed_tasks,
        "tasks_expected": expected_tasks,
        "tasks_running": running_tasks,
        "worker_errors": len(worker_errors),
        "latest_worker_error": worker_errors[-1].name if worker_errors else None,
        "stderr_bytes": state.get("stderr_bytes"),
        "gpu_util": last_resource.get("gpu_util_pct"),
        "gpu_temp": last_resource.get("gpu_temp_c"),
        "gpu_mem_mb": last_resource.get("gpu_memory_used_mb"),
        "ram_available_gb": last_resource.get("system_ram_available_gb"),
        "remaining_hours": state.get("remaining_hours"),
    }
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
