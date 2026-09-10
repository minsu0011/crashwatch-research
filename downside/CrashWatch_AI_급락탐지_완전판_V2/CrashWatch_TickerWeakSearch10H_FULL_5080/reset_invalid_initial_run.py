from __future__ import annotations

from pathlib import Path

from dual_ablation.config import get_paths
from dual_ablation.ticker_elite2h.runner import load_plan
from dual_ablation.ticker_map1h.registry import TickerTaskRegistry


def main() -> None:
    project = Path(__file__).resolve().parent
    plan = load_plan(project)
    result = get_paths(project).data_root / str(plan["result_subdir"])
    registry = TickerTaskRegistry(result / "task_registry.sqlite")
    checkpoint_count = len(list((result / "ticker_models").glob("*/fold_*_complete.json")))
    status = registry.status()
    if checkpoint_count or status["tasks_completed"]:
        raise RuntimeError(
            f"refusing invalid-run reset: checkpoints={checkpoint_count}, completed={status['tasks_completed']}"
        )
    with registry.connect() as connection:
        connection.execute(
            "UPDATE tasks SET status='pending',attempts=0,claimed_by=NULL,claimed_worker=NULL,"
            "claimed_at=NULL,heartbeat_at=NULL,completed_at=NULL,result_path=NULL,error=NULL "
            "WHERE status IN ('pending','running','failed')"
        )
    print(registry.status())


if __name__ == "__main__":
    main()
