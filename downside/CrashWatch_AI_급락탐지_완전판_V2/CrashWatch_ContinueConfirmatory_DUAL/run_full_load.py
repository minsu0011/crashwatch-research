from __future__ import annotations
import argparse, json, logging, multiprocessing as mp, os
from pathlib import Path
from cwnext.common import setup_logging
from cwnext.pipeline import run_pipeline


def main() -> int:
    parser=argparse.ArgumentParser(description="CrashWatch 7950X3D+96GB+RTX5080 풀로드 누락+확증 실험")
    parser.add_argument("--dataset",default=None,help="training_dataset_finance11h.parquet 경로")
    parser.add_argument("--project-root",default=None,help="CrashWatch 프로젝트 최상위 폴더")
    parser.add_argument("--previous-output",default=None,help="기존 all_feature_ablation_6h_full_5080_v1 결과 폴더")
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    cfg=json.load(open(root/"config_full_load.json",encoding="utf-8"))
    project=Path(args.project_root).expanduser().resolve() if args.project_root else root.parent.resolve()
    setup_logging(root/"launcher_full_load.log")
    result=run_pipeline(root,project,cfg,dataset_override=args.dataset,previous_output_override=args.previous_output)
    logging.info("finished status=%s elapsed_hours=%.2f",result.get("status"),result.get("elapsed_seconds",0)/3600)
    return 0 if result.get("status")=="completed" else 2

if __name__=="__main__":
    if os.name=="nt": mp.freeze_support()
    raise SystemExit(main())
