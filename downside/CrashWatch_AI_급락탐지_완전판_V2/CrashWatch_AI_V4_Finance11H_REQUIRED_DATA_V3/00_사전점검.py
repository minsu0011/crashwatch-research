#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

FILES = [
    "01_데이터_크롤링.py",
    "02_피쳐_생성_봉인.py",
    "03_이탈테스트_봉인인증.py",
    "04_결과_점검.py",
    "전체공정_실행.py",
    "공통_도구.py",
]
PACKAGES = {
    "pandas": "pandas", "numpy": "numpy", "pyarrow": "pyarrow",
    "requests": "requests", "beautifulsoup4": "bs4", "lxml": "lxml",
    "tqdm": "tqdm", "python-dotenv": "dotenv", "pykrx": "pykrx",
    "yfinance": "yfinance", "duckdb": "duckdb",
    "scikit-learn": "sklearn", "xgboost": "xgboost", "joblib": "joblib",
}
KEYS = [
    "OPENDART_API_KEY", "FRED_API_KEY", "ECOS_API_KEY",
    "NAVER_CLIENT_ID", "NAVER_CLIENT_SECRET", "KRX_ID", "KRX_PW",
]

def main() -> int:
    failed = False
    print("=" * 80)
    print("CrashWatch 실행 전 점검")
    print("=" * 80)
    print("Python:", sys.version)
    if sys.version_info < (3, 10):
        print("[실패] Python 3.10 이상 필요")
        failed = True
    for name in FILES:
        ok = (BASE_DIR / name).exists()
        print(f"[{'성공' if ok else '실패'}] 파일 {name}")
        failed |= not ok
    for package, module in PACKAGES.items():
        try:
            mod = importlib.import_module(module)
            print(f"[성공] 패키지 {package} {getattr(mod, '__version__', '')}")
        except Exception as exc:
            print(f"[실패] 패키지 {package}: {exc}")
            failed = True
    free = shutil.disk_usage(BASE_DIR).free / 1024**3
    print(f"[정보] 남은 디스크 {free:.1f}GB")
    if free < 15:
        print("[경고] 전체 수집에는 15GB 이상 권장")
    for key in KEYS:
        print(f"[{'성공' if os.getenv(key, '').strip() else '선택'}] {key}")
    print("=" * 80)
    return 1 if failed else 0

if __name__ == "__main__":
    raise SystemExit(main())
