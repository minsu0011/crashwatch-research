from __future__ import annotations

import json
from pathlib import Path

from dual_ablation.config import get_paths
from dual_ablation.ticker_elite2h.runner import load_plan
from dual_ablation.ticker_map1h.registry import TickerTaskRegistry


def main() -> None:
    project = Path(__file__).resolve().parent
    plan = load_plan(project)
    result_dir = get_paths(project).data_root / str(plan["result_subdir"])
    registry_path = result_dir / "task_registry.sqlite"
    print(f"Profile: {plan['execution_profile']}")
    print(f"Result directory: {result_dir}")
    if not registry_path.exists():
        print(f"No registry found: {registry_path}")
        return
    status = TickerTaskRegistry(registry_path).status()
    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    for name in ["runtime_worker_plan.json", "run_summary.json", "resource_stop.json"]:
        path = result_dir / name
        if path.exists():
            print(f"\n--- {name} ---")
            print(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
