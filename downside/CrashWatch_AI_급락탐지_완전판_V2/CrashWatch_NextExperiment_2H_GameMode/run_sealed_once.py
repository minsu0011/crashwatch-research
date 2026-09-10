from __future__ import annotations
import argparse,json
from pathlib import Path
from cw2h.sealed import run_sealed_once

def main():
 p=argparse.ArgumentParser(description='잠긴 단일 프로필 sealed 1회 평가')
 p.add_argument('--selection-output',required=True); p.add_argument('--sealed-dataset',required=True); p.add_argument('--project-root',default=None); p.add_argument('--confirm-sealed-once',required=True)
 a=p.parse_args(); root=Path(__file__).resolve().parent; project=Path(a.project_root).resolve() if a.project_root else root.parent.resolve(); r=run_sealed_once(root,project,Path(a.selection_output).resolve(),Path(a.sealed_dataset).resolve(),a.confirm_sealed_once); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
