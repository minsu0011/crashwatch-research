from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_ablation.config import get_paths
from dual_ablation.refine12h.registry import RefineTaskRegistry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--result-dir", type=Path, default=None)
    args = parser.parse_args()
    paths = get_paths(args.project)
    result_dir = (args.result_dir or paths.data_root / "refine12h").resolve()
    registry = RefineTaskRegistry(result_dir / "task_registry.sqlite")
    status = registry.status()
    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))
    summary = result_dir / "run_summary.json"
    if summary.exists():
        print("\n--- last run ---")
        print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
