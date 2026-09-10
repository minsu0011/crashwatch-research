from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_ablation.base12h.registry import TaskRegistry
from dual_ablation.config import get_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Base12H status")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--result-dir", type=Path, default=None)
    args = parser.parse_args()
    paths = get_paths(args.project)
    result_dir = args.result_dir or paths.data_root / "ablation_base12h"
    registry_path = result_dir / "task_registry.sqlite3"
    if not registry_path.exists():
        print(f"No registry: {registry_path}")
        return
    status = TaskRegistry(registry_path).status()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    summary_path = result_dir / "run_summary.json"
    if summary_path.exists():
        print("\n--- last run summary ---")
        print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
