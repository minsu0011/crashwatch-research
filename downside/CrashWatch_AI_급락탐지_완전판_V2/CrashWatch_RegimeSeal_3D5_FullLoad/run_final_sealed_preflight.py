from __future__ import annotations
import argparse,json
from pathlib import Path
from cwregime.final_sealed import preflight

def main():
    p=argparse.ArgumentParser(); p.add_argument("--selection-output",required=True); p.add_argument("--sealed-dataset",required=True); p.add_argument("--config",default=None); a=p.parse_args()
    root=Path(__file__).resolve().parent; cfg=json.load((Path(a.config) if a.config else root/"config_regime_3d5_full_load.json").open(encoding="utf-8"))
    result=preflight(Path(a.selection_output),Path(a.sealed_dataset),cfg); print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
