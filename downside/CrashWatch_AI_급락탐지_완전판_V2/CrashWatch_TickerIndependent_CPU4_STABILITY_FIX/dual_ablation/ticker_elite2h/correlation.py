from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from ..config import get_paths
from ..io_utils import atomic_csv, atomic_json
from ..refine12h.data import _catalog
from .data import CachedTickerData


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        if self.rank[a] < self.rank[b]:
            a, b = b, a
        self.parent[b] = a
        if self.rank[a] == self.rank[b]:
            self.rank[a] += 1


def _finite_corr(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 20:
        return math.nan
    xv = np.asarray(x[mask], dtype=np.float64)
    yv = np.asarray(y[mask], dtype=np.float64)
    if np.nanstd(xv) < 1e-12 or np.nanstd(yv) < 1e-12:
        return math.nan
    if rank:
        xv = rankdata(xv, method="average")
        yv = rankdata(yv, method="average")
    return float(np.corrcoef(xv, yv)[0, 1])


def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]
    if len(ref) < 60 or len(cur) < 30:
        return math.nan
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 4:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_hist = np.histogram(ref, bins=edges)[0].astype(float)
    cur_hist = np.histogram(cur, bins=edges)[0].astype(float)
    ref_p = np.clip(ref_hist / max(1.0, ref_hist.sum()), 1e-5, None)
    cur_p = np.clip(cur_hist / max(1.0, cur_hist.sum()), 1e-5, None)
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def _feature_groups(project: Path) -> dict[str, str]:
    paths = get_paths(project)
    catalog = _catalog(paths)
    output: dict[str, str] = {}
    for group, features in catalog.items():
        for feature in features:
            output.setdefault(feature, group)
    return output


def _temporal_corrs(values: np.ndarray, y: np.ndarray, segments: int) -> list[float]:
    indices = np.array_split(np.arange(len(y)), max(2, segments))
    return [_finite_corr(values[idx], y[idx], rank=True) for idx in indices]


def _standardized_mean_shift(reference: np.ndarray, current: np.ndarray) -> float:
    ref = reference[np.isfinite(reference)]
    cur = current[np.isfinite(current)]
    if len(ref) < 30 or len(cur) < 20:
        return math.nan
    scale = float(np.nanstd(ref))
    if not np.isfinite(scale) or scale < 1e-8:
        return 0.0
    return float(abs(np.nanmean(cur) - np.nanmean(ref)) / scale)


def compute_ticker_correlation_map(
    project: Path,
    data: CachedTickerData,
    result_dir: Path,
    plan: dict[str, Any],
) -> dict[str, Any]:
    ticker_dir = result_dir / "correlation_maps" / data.ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    X = np.asarray(data.X, dtype=np.float64)
    y = np.asarray(data.y, dtype=np.float64)
    n_rows, n_features = X.shape
    tail_rows = max(40, int(n_rows * float(plan["drift_tail_fraction"])))
    split = max(1, n_rows - tail_rows)
    group_map = _feature_groups(project)
    rows: list[dict[str, Any]] = []
    segment_count = int(plan["stable_sign_segments"])

    for index, feature in enumerate(data.features):
        values = X[:, index]
        missing = float(np.mean(~np.isfinite(values)))
        pearson = _finite_corr(values, y, rank=False)
        spearman = _finite_corr(values, y, rank=True)
        temporal = _temporal_corrs(values, y, segment_count)
        finite_temporal = np.asarray([v for v in temporal if np.isfinite(v)], dtype=float)
        sign_agreement = 0.0
        if len(finite_temporal):
            reference_sign = np.sign(np.nanmedian(finite_temporal))
            sign_agreement = float(np.mean(np.sign(finite_temporal) == reference_sign)) if reference_sign != 0 else 0.0
        temporal_std = float(np.nanstd(finite_temporal)) if len(finite_temporal) else math.nan
        psi = _psi(values[:split], values[split:])
        mean_shift = _standardized_mean_shift(values[:split], values[split:])
        unique_count = int(pd.Series(values[np.isfinite(values)]).nunique())
        score = max(abs(pearson) if np.isfinite(pearson) else 0.0, abs(spearman) if np.isfinite(spearman) else 0.0)
        rows.append({
            "ticker": data.ticker,
            "bucket": data.bucket,
            "feature": feature,
            "feature_group": group_map.get(feature, "unmapped"),
            "rows": n_rows,
            "missing_ratio": missing,
            "unique_count": unique_count,
            "pearson_target": pearson,
            "spearman_target": spearman,
            "absolute_target_score": score,
            "temporal_corr_1": temporal[0] if len(temporal) > 0 else math.nan,
            "temporal_corr_2": temporal[1] if len(temporal) > 1 else math.nan,
            "temporal_corr_3": temporal[2] if len(temporal) > 2 else math.nan,
            "target_sign_agreement": sign_agreement,
            "target_corr_temporal_std": temporal_std,
            "psi_tail": psi,
            "standardized_mean_shift": mean_shift,
        })

    nodes = pd.DataFrame(rows).sort_values(
        ["absolute_target_score", "target_sign_agreement"], ascending=[False, False]
    ).reset_index(drop=True)
    eligible = nodes.loc[
        nodes["unique_count"].ge(3)
        & nodes["missing_ratio"].le(float(plan["screening_max_missing"]))
    ].head(int(plan["correlation_feature_limit"]))
    selected_features = eligible["feature"].tolist()
    selected_indices = [data.features.index(name) for name in selected_features]
    matrix = X[:, selected_indices] if selected_indices else np.empty((n_rows, 0))
    if matrix.shape[1]:
        medians = np.nanmedian(matrix, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        filled = np.where(np.isfinite(matrix), matrix, medians)
        std = np.std(filled, axis=0)
        std = np.where(std > 1e-12, std, 1.0)
        normalized = (filled - np.mean(filled, axis=0)) / std
        corr = np.corrcoef(normalized, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        corr = np.empty((0, 0))

    edge_threshold = float(plan["edge_threshold"])
    cluster_threshold = float(plan["cluster_threshold"])
    union = UnionFind(len(selected_features))
    edges: list[dict[str, Any]] = []
    degree = np.zeros(len(selected_features), dtype=int)
    weighted_degree = np.zeros(len(selected_features), dtype=float)
    for left in range(len(selected_features)):
        for right in range(left + 1, len(selected_features)):
            value = float(corr[left, right])
            absolute = abs(value)
            if absolute >= cluster_threshold:
                union.union(left, right)
            if absolute >= edge_threshold:
                degree[left] += 1
                degree[right] += 1
                weighted_degree[left] += absolute
                weighted_degree[right] += absolute
                edges.append({
                    "ticker": data.ticker,
                    "feature_left": selected_features[left],
                    "feature_right": selected_features[right],
                    "correlation": value,
                    "absolute_correlation": absolute,
                    "same_sign": bool(value >= 0),
                })

    roots = [union.find(index) for index in range(len(selected_features))]
    root_order = {root: cluster_id for cluster_id, root in enumerate(sorted(set(roots)), start=1)}
    cluster_ids = [root_order[root] for root in roots]
    node_index = nodes.set_index("feature")
    for index, feature in enumerate(selected_features):
        node_index.loc[feature, "network_degree"] = int(degree[index])
        node_index.loc[feature, "network_weighted_degree"] = float(weighted_degree[index])
        node_index.loc[feature, "correlation_cluster"] = int(cluster_ids[index])
    nodes = node_index.reset_index()
    for column, default in [("network_degree", 0), ("network_weighted_degree", 0.0), ("correlation_cluster", 0)]:
        if column not in nodes.columns:
            nodes[column] = default
    nodes["network_degree"] = pd.to_numeric(nodes["network_degree"], errors="coerce").fillna(0).astype(int)
    nodes["network_weighted_degree"] = pd.to_numeric(nodes["network_weighted_degree"], errors="coerce").fillna(0.0)
    nodes["correlation_cluster"] = pd.to_numeric(nodes["correlation_cluster"], errors="coerce").fillna(0).astype(int)
    edge_frame = pd.DataFrame(edges).sort_values("absolute_correlation", ascending=False) if edges else pd.DataFrame(
        columns=["ticker", "feature_left", "feature_right", "correlation", "absolute_correlation", "same_sign"]
    )

    cluster_rows: list[dict[str, Any]] = []
    selected_nodes = nodes.loc[nodes["feature"].isin(selected_features)].copy()
    for cluster_id, block in selected_nodes.loc[selected_nodes["correlation_cluster"].gt(0)].groupby("correlation_cluster"):
        representative = block.sort_values(
            ["absolute_target_score", "missing_ratio"], ascending=[False, True]
        ).iloc[0]
        cluster_rows.append({
            "ticker": data.ticker,
            "cluster_id": int(cluster_id),
            "feature_count": int(len(block)),
            "representative_feature": representative["feature"],
            "representative_group": representative["feature_group"],
            "max_target_score": float(block["absolute_target_score"].max()),
            "mean_target_score": float(block["absolute_target_score"].mean()),
            "mean_missing_ratio": float(block["missing_ratio"].mean()),
            "mean_psi_tail": float(block["psi_tail"].mean(skipna=True)),
            "mean_sign_agreement": float(block["target_sign_agreement"].mean()),
            "max_network_degree": int(block["network_degree"].max()),
        })
    clusters = pd.DataFrame(cluster_rows)
    if not clusters.empty:
        clusters = clusters.sort_values(["feature_count", "max_target_score"], ascending=[False, False])

    largest_cluster = int(clusters["feature_count"].max()) if not clusters.empty else 1
    mapped_count = max(1, len(selected_features))
    summary = {
        "ticker": data.ticker,
        "bucket": data.bucket,
        "rows": int(n_rows),
        "positive_rate": float(data.y.mean()),
        "feature_count": int(n_features),
        "mapped_feature_count": int(len(selected_features)),
        "edge_count": int(len(edge_frame)),
        "cluster_count": int(len(clusters)),
        "largest_cluster_size": largest_cluster,
        "largest_cluster_ratio": float(largest_cluster / mapped_count),
        "median_abs_target_corr": float(nodes["absolute_target_score"].median()),
        "top20_mean_abs_target_corr": float(nodes.head(20)["absolute_target_score"].mean()),
        "top40_mean_sign_agreement": float(nodes.head(40)["target_sign_agreement"].mean()),
        "top40_mean_psi": float(nodes.head(40)["psi_tail"].mean(skipna=True)),
        "top40_mean_shift": float(nodes.head(40)["standardized_mean_shift"].mean(skipna=True)),
        "high_missing_feature_ratio": float(nodes["missing_ratio"].ge(0.60).mean()),
        "high_drift_feature_ratio": float(nodes["psi_tail"].ge(0.25).mean()),
    }

    atomic_csv(nodes, ticker_dir / "feature_nodes.csv")
    atomic_csv(edge_frame, ticker_dir / "feature_edges.csv")
    atomic_csv(clusters, ticker_dir / "cluster_summary.csv")
    atomic_json(summary, ticker_dir / "correlation_summary.json")
    return summary
