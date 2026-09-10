from __future__ import annotations

from pathlib import Path

from dual_ablation.config import get_paths
from dual_ablation.ticker_elite2h.reporting import aggregate_results, pack_results
from dual_ablation.ticker_elite2h.runner import load_plan
from dual_ablation.ticker_map1h.registry import TickerTaskRegistry


def main() -> None:
    project = Path(__file__).resolve().parent
    plan = load_plan(project)
    paths = get_paths(project)
    result_dir = paths.data_root / str(plan["result_subdir"])
    registry_path = result_dir / "task_registry.sqlite"
    status = TickerTaskRegistry(registry_path).status() if registry_path.exists() else {
        "tasks_total": 0, "tasks_completed": 0, "tasks_pending": 0, "tasks_running": 0,
        "tasks_failed": 0, "tasks_terminal": 0, "tasks_unfinished": 0, "progress_ratio": 0.0,
        "terminal_ratio": 0.0, "grouped": [], "running": [], "failed": [],
    }
    summary = aggregate_results(result_dir, plan, status, paths.data_root)
    archive = pack_results(result_dir, int(plan.get("result_max_zip_mb", 28)))
    print(f"Reports rebuilt: {result_dir}")
    print(f"Complete fingerprints: {summary.get('complete_fingerprint_tickers')} / {summary.get('fingerprint_tickers')}")
    print(f"Result archive: {archive}")


if __name__ == "__main__":
    main()
