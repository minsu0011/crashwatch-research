from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from .io_utils import normalize_ticker


@dataclass(frozen=True)
class ProjectPaths:
    project: Path
    data_root: Path
    raw_dual: Path
    feature_dual: Path
    result_dual: Path
    cache_dual: Path
    configs: Path


def get_paths(project: Path | None = None) -> ProjectPaths:
    project = (project or Path(__file__).resolve().parents[1]).resolve()
    load_dotenv(project / ".env", override=False)
    load_dotenv(project / ".env.dual", override=False)
    data_root = Path(os.getenv("CRASHWATCH_DATA_DIR", project / "crashwatch_ai_data")).resolve()
    run_tag = os.getenv("CRASHWATCH_RUN_TAG", "").strip()
    if run_tag:
        safe_tag = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in run_tag)
        result_dual = data_root / "ablation_longrun" / "runs" / safe_tag
        cache_dual = data_root / "ablation_longrun" / "prediction_cache"
    else:
        result_dual = data_root / "ablation_dual"
        cache_dual = data_root / "ablation_dual" / "prediction_cache"
    return ProjectPaths(
        project=project,
        data_root=data_root,
        raw_dual=data_root / "raw" / "dual_ablation",
        feature_dual=data_root / "features" / "dual_ablation",
        result_dual=result_dual,
        cache_dual=cache_dual,
        configs=project / "configs",
    )


def load_baskets(paths: ProjectPaths, enabled_only: bool = True) -> pd.DataFrame:
    df = pd.read_csv(paths.configs / "sector_baskets.csv", dtype={"ticker": str})
    df["ticker"] = normalize_ticker(df["ticker"])
    if enabled_only and "enabled" in df.columns:
        df = df.loc[pd.to_numeric(df["enabled"], errors="coerce").fillna(0).astype(int).eq(1)]
    return df.drop_duplicates("ticker").reset_index(drop=True)


def load_assets(paths: ProjectPaths) -> pd.DataFrame:
    df = pd.read_csv(paths.configs / "global_assets.csv")
    if "enabled" in df.columns:
        df = df.loc[pd.to_numeric(df["enabled"], errors="coerce").fillna(0).astype(int).eq(1)]
    return df.reset_index(drop=True)


def load_prefix_catalog(paths: ProjectPaths) -> dict[str, dict[str, list[str]]]:
    return json.loads((paths.configs / "feature_group_prefixes.json").read_text(encoding="utf-8"))


def load_ticker_rules(paths: ProjectPaths) -> dict:
    return json.loads((paths.configs / "ticker_feature_rules.json").read_text(encoding="utf-8"))
