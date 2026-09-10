from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path

from cwfull.pipeline import run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="CrashWatch full-load final selection: XGB P0/P2/P7 + P2 factorial + protected feature revalidation"
    )
    parser.add_argument("--dataset", default=None, help="training_dataset_finance11h.parquet 경로")
    parser.add_argument("--project-root", default=None, help="CrashWatch 프로젝트 최상위 폴더")
    parser.add_argument("--output", default=None, help="결과 폴더")
    parser.add_argument("--config", default=None, help="config_full_load.json 대체 경로")
    parser.add_argument(
        "--stages",
        default="core,targeted,diagnostic,aggregate",
        help="실행 단계: core,targeted,diagnostic,aggregate 또는 aggregate만 지정 가능",
    )
    args = parser.parse_args()
    package_root = Path(__file__).resolve().parent
    config_path = Path(args.config).expanduser() if args.config else package_root / "config_full_load.json"
    config = json.load(config_path.open(encoding="utf-8"))
    project_root = Path(args.project_root).expanduser().resolve() if args.project_root else package_root.parent.resolve()
    stages = {value.strip() for value in str(args.stages).split(",") if value.strip()}
    allowed = {"core", "targeted", "diagnostic", "aggregate"}
    unknown = stages - allowed
    if unknown:
        raise ValueError(f"알 수 없는 stages: {sorted(unknown)}")
    result = run_pipeline(
        package_root,
        project_root,
        config,
        dataset_override=args.dataset,
        output_override=args.output,
        stages=stages,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "completed" else 2


if __name__ == "__main__":
    if os.name == "nt":
        mp.freeze_support()
    raise SystemExit(main())
