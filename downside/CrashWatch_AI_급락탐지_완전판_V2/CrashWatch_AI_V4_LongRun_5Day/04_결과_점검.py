#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from 공통_도구 import 수집_품질_보고서_생성

BASE_DIR = Path(__file__).resolve().parent
D = BASE_DIR / "crashwatch_ai_data"

def shape(path: Path) -> str:
    if not path.exists():
        return "없음"
    try:
        df = pd.read_parquet(path)
        return f"{len(df):,}행 × {len(df.columns):,}열"
    except Exception as exc:
        return f"읽기실패: {exc}"

def csv_rows(path: Path) -> str:
    if not path.exists():
        return "없음"
    try:
        return f"{len(pd.read_csv(path)):,}행"
    except Exception as exc:
        return f"읽기실패: {exc}"

def main() -> None:
    print("=" * 88)
    print("CrashWatch 전체 결과 점검")
    print("=" * 88)
    try:
        수집_품질_보고서_생성()
    except Exception as exc:
        print("[경고] 수집 필수 데이터 점검:", exc)

    paths = {
        "전체 피쳐": D / "processed" / "full_feature_dataset.parquet",
        "개발 데이터": D / "development" / "training_dataset.parquet",
    }
    for name, path in paths.items():
        print(f"{name:<18}: {shape(path)}")
    for seal in ("S00", "S01", "S02", "S03"):
        print(f"봉인 {seal:<14}: {shape(D / 'sealed' / seal / 'data.parquet')}")
    for name, path in {
        "기준 CV": D / "ablation" / "baseline_cv.csv",
        "그룹 이탈": D / "ablation" / "group_ablation.csv",
        "개별 이탈": D / "ablation" / "feature_ablation.csv",
        "봉인 인증": D / "ablation" / "seal_certification.csv",
    }.items():
        print(f"{name:<18}: {csv_rows(path)}")

    frozen = D / "ablation" / "frozen_feature_set.json"
    if frozen.exists():
        data = json.loads(frozen.read_text(encoding="utf-8"))
        print("동결 피쳐 수       :", len(data.get("features", [])))
        print("피쳐 목록 해시    :", data.get("feature_list_sha256"))
        print("제거 그룹         :", data.get("dropped_groups"))
    zips = sorted((D / "결과압축").glob("*.zip")) if (D / "결과압축").exists() else []
    print("\n공정별 결과 압축:")
    for z in zips:
        print(f" - {z.name} ({z.stat().st_size/1024/1024:.2f}MB)")

if __name__ == "__main__":
    main()
