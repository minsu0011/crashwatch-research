from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import ProjectPaths, get_paths
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_date, normalize_ticker
from .catalogs import build_catalogs
from .ticker import build_ticker_features
from .universe import build_universe_features


def _quality(namespace: str, df: pd.DataFrame, keys: set[str]) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        if col in keys:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "namespace": namespace, "feature": col, "rows": len(s),
            "missing_rate": float(s.isna().mean()), "unique_count": int(s.nunique(dropna=True)),
            "finite_rate": float(np.isfinite(s.dropna()).mean()) if s.notna().any() else 0.0,
        })
    return pd.DataFrame(rows)


def merge_with_training(paths: ProjectPaths, universe: pd.DataFrame, ticker: pd.DataFrame, training_path: Path | None = None) -> tuple[pd.DataFrame, Path | None]:
    candidates = [
        training_path,
        paths.data_root / "development" / "training_dataset.parquet",
        paths.data_root / "development" / "training_dataset_dual.parquet",
    ]
    source = next((p for p in candidates if p is not None and Path(p).exists()), None)
    if source is None:
        return pd.DataFrame(), None
    base = pd.read_parquet(source)
    if "date" not in base.columns:
        raise KeyError("training dataset에 date 열이 없습니다.")
    if "ticker" not in base.columns:
        raise KeyError("training dataset에 ticker 열이 없습니다.")
    base = normalize_date(base)
    base["ticker"] = normalize_ticker(base["ticker"])
    merged = base.merge(universe, on="date", how="left", validate="many_to_one")
    merged = merged.merge(ticker, on=["date", "ticker"], how="left", validate="many_to_one", suffixes=("", "_dual"))
    output = paths.data_root / "development" / "training_dataset_dual.parquet"
    atomic_parquet(merged, output)
    return merged, output


def run_feature_pipeline(project: Path | None = None, training_path: Path | None = None) -> dict:
    paths = get_paths(project)
    paths.feature_dual.mkdir(parents=True, exist_ok=True)
    universe = build_universe_features(paths)
    ticker = build_ticker_features(paths, universe)
    atomic_parquet(universe, paths.feature_dual / "universe_features.parquet")
    atomic_parquet(ticker, paths.feature_dual / "ticker_features.parquet")
    u_catalog, t_catalog = build_catalogs(paths, universe, ticker)
    quality = pd.concat([
        _quality("universe", universe, {"date"}),
        _quality("ticker", ticker, {"date", "ticker", "name", "bucket", "market", "role"}),
    ], ignore_index=True)
    atomic_csv(quality, paths.feature_dual / "feature_quality.csv")
    merged, merged_path = merge_with_training(paths, universe, ticker, training_path)
    summary = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "universe_rows": len(universe), "universe_features": sum(len(v) for v in u_catalog.values()),
        "ticker_rows": len(ticker), "ticker_count": int(ticker["ticker"].nunique()),
        "ticker_features": sum(len(v) for v in t_catalog.values()),
        "merged_rows": len(merged), "merged_path": str(merged_path) if merged_path else None,
        "warning": "KOSIS 월간 계열은 공표일 메타데이터가 없으면 31일 보수적 지연을 적용합니다.",
    }
    atomic_json(summary, paths.feature_dual / "feature_run_summary.json")
    return summary
