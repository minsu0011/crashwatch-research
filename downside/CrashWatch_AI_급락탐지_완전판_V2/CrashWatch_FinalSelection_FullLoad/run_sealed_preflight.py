from __future__ import annotations
import argparse, json
from pathlib import Path
from cwfull.sealed import preflight

def main() -> int:
    parser=argparse.ArgumentParser(description='Sealed 1회 평가 사전 스키마 검사')
    parser.add_argument('--selection-output',required=True)
    parser.add_argument('--sealed-dataset',required=True)
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    result=preflight(Path(args.selection_output),Path(args.sealed_dataset),root)
    print(json.dumps(result,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__': raise SystemExit(main())
