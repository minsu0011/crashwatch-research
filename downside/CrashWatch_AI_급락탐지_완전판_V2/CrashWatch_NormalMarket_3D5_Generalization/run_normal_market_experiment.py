from __future__ import annotations
import argparse
from pathlib import Path
from cwnormal.pipeline import run

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default='.'); ap.add_argument('--dataset',default=None); ap.add_argument('--output',default=None); ap.add_argument('--config',default=None); args=ap.parse_args()
    root=Path(__file__).resolve().parent; cfg=Path(args.config).resolve() if args.config else root/'config_normal_market_3d5.json'
    run(root,Path(args.project_root).resolve(),cfg,args.dataset,args.output)
if __name__=='__main__': main()
