#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""development 패널에서 대표 종목 피처 시계열만 추출한다.

이 스크립트는 입력 development 데이터나 피처 산식을 변경하지 않는다.
출력은 crashwatch_ablation_addon/crashwatch_ai_data/ablation_sentinel/ 이다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_ROOT = PROJECT_DIR / "crashwatch_ai_data"
DEVELOPMENT_PATH = DATA_ROOT / "development" / "training_dataset.parquet"
CATALOG_PATH = DATA_ROOT / "meta" / "feature_catalog.json"
OUT_DIR = BASE_DIR / "crashwatch_ai_data" / "ablation_sentinel"
TARGET = "label_abs_crash_20"

SENTINELS = [
    ("005930", "삼성전자"), ("000660", "SK하이닉스"), ("042700", "한미반도체"),
    ("005380", "현대차"), ("373220", "LG에너지솔루션"), ("005490", "POSCO홀딩스"),
    ("034020", "두산에너빌리티"), ("012450", "한화에어로스페이스"),
    ("329180", "HD현대중공업"), ("207940", "삼성바이오로직스"), ("068270", "셀트리온"),
    ("035420", "NAVER"), ("035720", "카카오"), ("105560", "KB금융"),
    ("017670", "SK텔레콤"), ("028260", "삼성물산"), ("247540", "에코프로비엠"),
    ("196170", "알테오젠"),
]


def atomic_csv(frame: pd.DataFrame, path: Path, **kwargs: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, encoding="utf-8-sig", **kwargs)
    os.replace(tmp, path)


def core_feature_columns(panel: pd.DataFrame, catalog: dict[str, list[str]]) -> list[str]:
    """각 feature group에서 최대 5개씩 고르게 포함한 가벼운 검토용 목록."""
    identity = ["date", "ticker", "name", "market", "industry_name", "close", TARGET]
    selected = [c for c in identity if c in panel.columns]
    for group in sorted(catalog):
        selected.extend([c for c in catalog[group] if c in panel.columns and c not in selected][:5])
    return selected


def main() -> None:
    if not DEVELOPMENT_PATH.exists() or not CATALOG_PATH.exists():
        raise FileNotFoundError("development/training_dataset.parquet 또는 feature_catalog.json이 없습니다.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(DEVELOPMENT_PATH)
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    panel["ticker"] = panel["ticker"].astype(str).str.zfill(6)
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    wanted = [ticker for ticker, _ in SENTINELS]
    sentinel = panel.loc[panel["ticker"].isin(wanted)].copy().sort_values(["ticker", "date"])
    sentinel.to_parquet(OUT_DIR / "sentinel_feature_timeseries.parquet", index=False)
    core_cols = core_feature_columns(sentinel, catalog)
    sentinel.loc[:, core_cols].to_csv(
        OUT_DIR / "sentinel_core_features.csv.gz", index=False, encoding="utf-8-sig", compression="gzip"
    )
    quality_rows = []
    for ticker, requested_name in SENTINELS:
        part = sentinel.loc[sentinel["ticker"].eq(ticker)]
        numeric = part.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
        quality_rows.append({
            "ticker": ticker,
            "requested_name": requested_name,
            "name": part["name"].iloc[-1] if len(part) and "name" in part else requested_name,
            "sector": part["industry_name"].iloc[-1] if len(part) and "industry_name" in part else "UNKNOWN",
            "in_development": bool(len(part)),
            "rows": len(part),
            "start_date": part["date"].min() if len(part) else pd.NaT,
            "end_date": part["date"].max() if len(part) else pd.NaT,
            "positives": int(pd.to_numeric(part.get(TARGET), errors="coerce").fillna(0).sum()) if len(part) else 0,
            "positive_rate": float(pd.to_numeric(part.get(TARGET), errors="coerce").mean()) if len(part) else np.nan,
            "mean_missing_rate": float(numeric.isna().mean().mean()) if not numeric.empty else np.nan,
        })
    quality = pd.DataFrame(quality_rows)
    atomic_csv(quality, OUT_DIR / "sentinel_ticker_quality.csv")
    atomic_csv(quality, OUT_DIR / "sentinel_universe_checked.csv")
    print(f"대표 종목 추출: {len(sentinel):,}행 / {quality.in_development.sum()}개 종목")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
