from __future__ import annotations

import csv
import json
import sqlite3
import subprocess
from pathlib import Path

from dual_ablation.config import get_paths
from dual_ablation.ticker_elite2h.runner import load_plan


def main() -> None:
    project = Path(__file__).resolve().parent
    plan = load_plan(project)
    result = get_paths(project).data_root / str(plan["result_subdir"])
    registry = result / "task_registry.sqlite"
    payload: dict = {"result_dir": str(result)}
    if registry.exists():
        with sqlite3.connect(registry) as connection:
            payload["tasks"] = dict(
                connection.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status").fetchall()
            )
            payload["running"] = connection.execute(
                "SELECT task_id, claimed_worker, attempts FROM tasks WHERE status='running' ORDER BY task_id"
            ).fetchall()
            payload["failed"] = connection.execute(
                "SELECT task_id, attempts, SUBSTR(error, 1, 500) FROM tasks WHERE status='failed' ORDER BY task_id"
            ).fetchall()
    resource = result / "resource_usage.csv"
    if resource.exists():
        with resource.open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        payload["resource"] = rows[-1] if rows else {}
    payload["initial_fold_checkpoints"] = len(list((result / "ticker_models").glob("*/fold_*_complete.json")))
    payload["replay_fold_checkpoints"] = len(
        list((result / "ticker_models").glob("*/recipe_replay/*/fold_*_complete.json"))
    )
    payload["model_cards"] = len(list((result / "ticker_models").glob("*/development_model/model_card.json")))
    errors = sorted(result.glob("TickerWeakSearch10H_FULL_*.err.log"), key=lambda path: path.stat().st_mtime)
    if errors:
        latest = errors[-1]
        payload["stderr"] = {"path": str(latest), "bytes": latest.stat().st_size}
    try:
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip()
        payload["gpu_now"] = gpu
    except Exception as exc:
        payload["gpu_now_error"] = str(exc)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
