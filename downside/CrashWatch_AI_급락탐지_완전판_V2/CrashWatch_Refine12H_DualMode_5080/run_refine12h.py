from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_ablation.refine12h.runner import run_supervisor


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch calibration/regime/model-family refinement runner")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--profile", choices=["pubg", "full"], required=True)
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--tree-workers", type=int, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--base12-source", type=Path, default=None)
    args = parser.parse_args()
    result = run_supervisor(
        args.project,
        args.dataset,
        profile=args.profile,
        hours=max(0.1, args.hours),
        tree_workers=args.tree_workers,
        result_dir=args.result_dir,
        base12_source=args.base12_source,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
