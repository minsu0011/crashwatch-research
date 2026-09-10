from __future__ import annotations
import argparse, json, multiprocessing as mp, os
from pathlib import Path
from cwfull.sealed import run_once

def main() -> int:
    parser=argparse.ArgumentParser(description='잠긴 프로필의 sealed 데이터 1회 평가')
    parser.add_argument('--selection-output',required=True)
    parser.add_argument('--sealed-dataset',required=True)
    parser.add_argument('--confirm-sealed-once',required=True)
    parser.add_argument('--config',default=None)
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    cfg=json.load(open(Path(args.config) if args.config else root/'config_full_load.json',encoding='utf-8'))
    result=run_once(Path(args.selection_output),Path(args.sealed_dataset),root,cfg,args.confirm_sealed_once)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__':
    if os.name=='nt': mp.freeze_support()
    raise SystemExit(main())
