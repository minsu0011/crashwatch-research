#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from 공통_도구 import 결과압축_생성

BASE_DIR = Path(__file__).resolve().parent
D = BASE_DIR / "crashwatch_ai_data"
STATE = D / "meta" / "전체공정_상태.json"
STEPS = [
    ("데이터크롤링", "01_데이터_크롤링.py"),
    ("피쳐생성봉인", "02_피쳐_생성_봉인.py"),
    ("이탈테스트봉인인증", "03_이탈테스트_봉인인증.py"),
    ("결과점검", "04_결과_점검.py"),
]

def save(data):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

def run(name, file):
    print("\n" + "="*90)
    print(f"[시작] {name} → {file}")
    print("="*90)
    t = time.monotonic()
    result = subprocess.run([sys.executable, "-u", str(BASE_DIR/file)], cwd=BASE_DIR)
    elapsed = time.monotonic() - t
    save({"공정": name, "파일": file, "종료코드": result.returncode,
          "소요초": elapsed, "시각": datetime.now().astimezone().isoformat()})
    if result.returncode != 0:
        stage_map = {
            "데이터크롤링": "01_데이터크롤링",
            "피쳐생성봉인": "02_피쳐생성_봉인",
            "이탈테스트봉인인증": "03_이탈테스트_봉인인증",
        }
        if name in stage_map:
            결과압축_생성(stage_map[name])
        raise SystemExit(f"[실패] {name}, 종료코드={result.returncode}")
    print(f"[완료] {name}, {elapsed/60:.1f}분")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--공정", choices=[x[0] for x in STEPS], default=None)
    p.add_argument("--시작공정", choices=[x[0] for x in STEPS], default="데이터크롤링")
    p.add_argument("--사전점검생략", action="store_true")
    args = p.parse_args()
    if not args.사전점검생략:
        r = subprocess.run([sys.executable, "-u", str(BASE_DIR/"00_사전점검.py")], cwd=BASE_DIR)
        if r.returncode:
            raise SystemExit("[실패] 사전점검")
    if args.공정:
        selected = [x for x in STEPS if x[0] == args.공정]
    else:
        idx = [x[0] for x in STEPS].index(args.시작공정)
        selected = STEPS[idx:]
    for item in selected:
        run(*item)
    print("\n전체 공정 완료:", D)

if __name__ == "__main__":
    main()
