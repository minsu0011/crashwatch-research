from __future__ import annotations
import argparse, json, multiprocessing as mp, os
from pathlib import Path
from cwregime.pipeline import run_development

def main() -> int:
    parser=argparse.ArgumentParser(description="CrashWatch 3d/-5% multi-regime development holdout optimization")
    parser.add_argument("--dataset",default=None,help="development training_dataset_finance11h.parquet")
    parser.add_argument("--project-root",default=None)
    parser.add_argument("--output",default=None)
    parser.add_argument("--config",default=None)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    cfg_path=Path(args.config).expanduser() if args.config else root/"config_regime_3d5_full_load.json"
    cfg=json.load(cfg_path.open(encoding="utf-8"))
    project=Path(args.project_root).expanduser().resolve() if args.project_root else root.parent.resolve()
    result=run_development(root,project,cfg,args.dataset,args.output)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0 if result.get("status") == "completed" else 2
if __name__=="__main__":
    if os.name=="nt": mp.freeze_support()
    raise SystemExit(main())
