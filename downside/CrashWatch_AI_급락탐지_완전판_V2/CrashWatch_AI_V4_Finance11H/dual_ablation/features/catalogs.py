from __future__ import annotations

from pathlib import Path
import os

import pandas as pd

from ..config import ProjectPaths, load_prefix_catalog
from ..io_utils import atomic_json


def resolve_catalog(columns: list[str], prefix_spec: dict[str, list[str]]) -> dict[str, list[str]]:
    catalog: dict[str, list[str]] = {}
    for group, prefixes in prefix_spec.items():
        matched = sorted({col for col in columns if any(col.startswith(prefix) for prefix in prefixes)})
        catalog[group] = matched
    return catalog


def build_catalogs(paths: ProjectPaths, universe: pd.DataFrame, ticker: pd.DataFrame) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    specs = load_prefix_catalog(paths)
    u_catalog = resolve_catalog(list(universe.columns), specs["universe"])
    t_catalog = resolve_catalog(list(ticker.columns), specs["ticker"])
    policy_path = paths.configs / "research_feature_groups.csv"
    if policy_path.exists() and os.getenv("ENABLE_EXPLORATORY_FEATURES", "0").strip() != "1":
        policy = pd.read_csv(policy_path)
        disabled = set(policy.loc[pd.to_numeric(policy["default_enabled"], errors="coerce").fillna(0).eq(0), "group"].astype(str))
        u_catalog = {group: ([] if group in disabled else cols) for group, cols in u_catalog.items()}
        t_catalog = {group: ([] if group in disabled else cols) for group, cols in t_catalog.items()}
    atomic_json(u_catalog, paths.feature_dual / "feature_catalog_universe.json")
    atomic_json(t_catalog, paths.feature_dual / "feature_catalog_ticker.json")
    return u_catalog, t_catalog
