from __future__ import annotations
import argparse,json,multiprocessing as mp,os
from pathlib import Path
from cw2h.pipeline import run_pipeline

def main():
 p=argparse.ArgumentParser(description='CrashWatch 2시간 배그 모드: XGB P0/P2/P7 + P2 원인분해')
 p.add_argument('--dataset',default=None); p.add_argument('--project-root',default=None); p.add_argument('--output',default=None); p.add_argument('--config',default=None)
 a=p.parse_args(); root=Path(__file__).resolve().parent; cfg=json.load(open(Path(a.config) if a.config else root/'config_game_2h.json',encoding='utf-8')); project=Path(a.project_root).expanduser().resolve() if a.project_root else root.parent.resolve()
 r=run_pipeline(root,project,cfg,a.dataset,a.output); print(json.dumps(r,ensure_ascii=False,indent=2)); return 0
if __name__=='__main__':
 if os.name=='nt': mp.freeze_support()
 raise SystemExit(main())
