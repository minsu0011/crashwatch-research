from __future__ import annotations

import argparse
import json
from pathlib import Path

from dual_ablation.base12h.runner import run_supervisor


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch 12-hour base-model reinforcement runner")
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--profile", choices=["pubg", "full"], required=True)
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--result-dir", type=Path, default=None)
    args = parser.parse_args()
    workers = args.workers if args.workers is not None else (1 if args.profile == "pubg" else 2)
    result = run_supervisor(
        args.project,
        args.dataset,
        profile=args.profile,
        hours=max(0.1, args.hours),
        workers=max(1, workers),
        result_dir=args.result_dir,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
