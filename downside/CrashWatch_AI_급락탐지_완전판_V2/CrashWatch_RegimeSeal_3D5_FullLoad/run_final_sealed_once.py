from __future__ import annotations
import argparse,json,multiprocessing as mp,os
from pathlib import Path
from cwregime.final_sealed import run_once

def main():
    p=argparse.ArgumentParser(); p.add_argument("--selection-output",required=True); p.add_argument("--sealed-dataset",required=True); p.add_argument("--confirm-final-sealed-once",required=True); p.add_argument("--config",default=None); a=p.parse_args()
    root=Path(__file__).resolve().parent; cfg=json.load((Path(a.config) if a.config else root/"config_regime_3d5_full_load.json").open(encoding="utf-8"))
    result=run_once(Path(a.selection_output),Path(a.sealed_dataset),cfg,a.confirm_final_sealed_once); print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__":
    if os.name=="nt": mp.freeze_support()
    raise SystemExit(main())
