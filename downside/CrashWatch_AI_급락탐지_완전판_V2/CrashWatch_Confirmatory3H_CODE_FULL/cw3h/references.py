from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import hash_file, hash_strings, normalize_ticker, read_json

@dataclass(frozen=True)
class ReferenceBundle:
    valid_features: list[str]
    full_reduced_features: list[str]
    common_features: list[str]
    groups: dict[str, list[str]]
    ticker_to_bucket: dict[str, str]
    folds: list[dict[str, Any]]
    hashes: dict[str, str]


def load_references(reference_dir: Path, strict_counts: bool = True) -> ReferenceBundle:
    audit_path = reference_dir / "valid_feature_audit.csv"
    redundant_path = reference_dir / "redundant_features.csv"
    high_missing_path = reference_dir / "high_missing_features_removed.csv"
    ticker_map_path = reference_dir / "ticker_bucket_map.csv"
    folds_path = reference_dir / "outer_walk_forward_folds.json"
    audit = pd.read_csv(audit_path)
    if "status" in audit.columns:
        audit = audit[audit["status"].astype(str).str.lower().eq("valid")].copy()
    audit["feature"] = audit["feature"].astype(str)
    valid_features = audit["feature"].drop_duplicates().tolist()
    groups: dict[str, list[str]] = {}
    for group, frame in audit.dropna(subset=["group"]).groupby("group", sort=True):
        groups[str(group)] = sorted(frame["feature"].astype(str).drop_duplicates().tolist())
    redundant = pd.read_csv(redundant_path)["feature"].astype(str).drop_duplicates().tolist()
    high_missing = pd.read_csv(high_missing_path)["feature"].astype(str).drop_duplicates().tolist()
    full_reduced_features = [f for f in valid_features if f not in set(redundant)]
    common_features = [f for f in full_reduced_features if f not in set(high_missing)]
    ticker_map = pd.read_csv(ticker_map_path, dtype=str)
    ticker_to_bucket = {normalize_ticker(row.ticker): str(row.bucket) for row in ticker_map.itertuples(index=False)}
    folds = read_json(folds_path)
    if strict_counts:
        expected = {"valid": 439, "redundant": 75, "full_reduced": 364, "high_missing": 23, "common": 341, "tickers": 48, "folds": 8}
        actual = {"valid": len(valid_features), "redundant": len(redundant), "full_reduced": len(full_reduced_features), "high_missing": len(high_missing), "common": len(common_features), "tickers": len(ticker_to_bucket), "folds": len(folds)}
        if actual != expected:
            raise ValueError(f"reference count mismatch: expected={expected}, actual={actual}")
    required_groups = ["u_financial_shorting", "u_financial_market", "u_etf_pressure", "t_financial_shorting", "t_stock_lending", "t_financial_interaction"]
    missing_groups = [name for name in required_groups if not groups.get(name)]
    if missing_groups:
        raise ValueError(f"required feature groups missing: {missing_groups}")
    hashes = {
        "valid_feature_audit": hash_file(audit_path),
        "redundant_features": hash_file(redundant_path),
        "high_missing_features": hash_file(high_missing_path),
        "ticker_bucket_map": hash_file(ticker_map_path),
        "folds": hash_file(folds_path),
        "full_reduced_feature_hash": hash_strings(full_reduced_features),
        "common_feature_hash": hash_strings(common_features)
    }
    return ReferenceBundle(valid_features, full_reduced_features, common_features, groups, ticker_to_bucket, folds, hashes)


def condition_feature_lists(refs: ReferenceBundle) -> dict[str, list[str]]:
    short = set(refs.groups["u_financial_shorting"])
    market = set(refs.groups["u_financial_market"])
    etf = set(refs.groups["u_etf_pressure"])
    return {"B0": [], "A1": sorted(short), "A2": sorted(market), "A3": sorted(short | market), "A4": sorted(etf), "A5": sorted(market | etf), "A6": sorted(short | market | etf)}


def battery_mask_groups(refs: ReferenceBundle) -> dict[str, list[str]]:
    short = set(refs.groups["t_financial_shorting"])
    lending = set(refs.groups["t_stock_lending"])
    interaction = set(refs.groups["t_financial_interaction"])
    return {"S1": sorted(short), "S2": sorted(short | lending), "S3": sorted(short | interaction)}
