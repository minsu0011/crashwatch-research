from __future__ import annotations

import sqlite3
from pathlib import Path

from dual_ablation.config import get_paths
from dual_ablation.ticker_elite2h.runner import load_plan


def main() -> None:
    project = Path(__file__).resolve().parent
    plan = load_plan(project)
    result_dir = get_paths(project).data_root / str(plan["result_subdir"])
    registry = result_dir / "task_registry.sqlite"
    if not registry.exists():
        print(f"No registry found: {registry}")
        return
    with sqlite3.connect(registry) as connection:
        count = connection.execute("SELECT COUNT(*) FROM tasks WHERE status='failed'").fetchone()[0]
        connection.execute(
            "UPDATE tasks SET status='pending',claimed_by=NULL,claimed_worker=NULL,claimed_at=NULL,"
            "heartbeat_at=NULL,completed_at=NULL,error='manual failed-task reset',attempts=0 "
            "WHERE status='failed'"
        )
        connection.commit()
    print(f"Reset {count} failed tasks to pending. Run the main RUN file to resume.")


if __name__ == "__main__":
    main()
