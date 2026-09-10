from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score


ERROR_OTHER = 0
ERROR_A_TOP_TP = 1
ERROR_B_TOP_FP = 2
ERROR_C_LOW_FN = 3
ERROR_D_LOW_TN = 4
ERROR_GROUP_NAMES = {
    ERROR_OTHER: "OTHER",
    ERROR_A_TOP_TP: "A_TOP_TRUE_POSITIVE",
    ERROR_B_TOP_FP: "B_TOP_FALSE_POSITIVE",
    ERROR_C_LOW_FN: "C_LOW_MISSED_POSITIVE",
    ERROR_D_LOW_TN: "D_LOW_TRUE_NEGATIVE",
}


@dataclass(frozen=True)
class OrganicCondition:
    stage: str
    test_type: str
    condition_id: str
    dropped_features: tuple[str, ...]
    representative_feature: str | None = None
    cluster_id: int | None = None
    feature_group: str | None = None
    relation_threshold: float | None = None
    relation_k: int | None = None
    pair_feature_a: str | None = None
    pair_feature_b: str | None = None
    relation_set_hash: str | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "test_type": self.test_type,
            "condition_id": self.condition_id,
            "dropped_feature_count": len(self.dropped_features),
            "dropped_features": "|".join(self.dropped_features),
            "representative_feature": self.representative_feature,
            "cluster_id": self.cluster_id,
            "feature_group": self.feature_group,
            "relation_threshold": self.relation_threshold,
            "relation_k": self.relation_k,
            "pair_feature_a": self.pair_feature_a,
            "pair_feature_b": self.pair_feature_b,
            "relation_set_hash": self.relation_set_hash,
        }


def _set_hash(values: Sequence[str]) -> str:
    payload = "\n".join(sorted(map(str, values))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def parse_float_tokens(text: str) -> list[float]:
    return sorted({float(token.strip()) for token in str(text).split(",") if token.strip()})


def parse_int_tokens(text: str) -> list[int]:
    return sorted({int(token.strip()) for token in str(text).split(",") if token.strip()})


def load_correlation_matrix(correlation_dir: Path, features: Sequence[str]) -> pd.DataFrame:
    candidates = [
        correlation_dir / "cluster_basis_combined_abs.csv.gz",
        correlation_dir / "combined_abs_correlation.csv.gz",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            "상관관계 행렬이 없습니다. 필요한 파일: cluster_basis_combined_abs.csv.gz 또는 combined_abs_correlation.csv.gz"
        )
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    missing_rows = [feature for feature in features if feature not in frame.index]
    missing_cols = [feature for feature in features if feature not in frame.columns]
    if missing_rows or missing_cols:
        raise ValueError(
            f"상관행렬 feature mismatch rows={missing_rows[:10]} cols={missing_cols[:10]}"
        )
    matrix = frame.loc[list(features), list(features)].apply(pd.to_numeric, errors="coerce")
    values = matrix.to_numpy(dtype=np.float64)
    if values.shape != (len(features), len(features)):
        raise ValueError("상관행렬 shape mismatch")
    values = np.abs(values)
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.maximum(values, values.T)
    np.fill_diagonal(values, 1.0)
    values = np.clip(values, 0.0, 1.0)
    return pd.DataFrame(values, index=list(features), columns=list(features))


def hierarchical_clusters(
    correlation: pd.DataFrame,
    thresholds: Sequence[float],
) -> pd.DataFrame:
    features = list(correlation.index)
    values = correlation.to_numpy(dtype=np.float64)
    distance = np.clip(1.0 - values, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="average", optimal_ordering=False)
    records: list[dict[str, Any]] = []
    for threshold in thresholds:
        labels = fcluster(tree, t=1.0 - float(threshold), criterion="distance")
        counts = pd.Series(labels).value_counts().to_dict()
        remap: dict[int, int] = {}
        ordered_labels = sorted(set(map(int, labels)), key=lambda label: min(
            feature for feature, raw in zip(features, labels) if int(raw) == label
        ))
        for ordinal, label in enumerate(ordered_labels, start=1):
            remap[label] = ordinal
        for feature, raw_label in zip(features, labels):
            raw_label = int(raw_label)
            records.append(
                {
                    "threshold": float(threshold),
                    "cluster_id": int(remap[raw_label]),
                    "cluster_size": int(counts[raw_label]),
                    "feature": feature,
                }
            )
    return pd.DataFrame.from_records(records)


def correlation_pairs(
    correlation: pd.DataFrame,
    minimum: float,
) -> pd.DataFrame:
    features = list(correlation.index)
    values = correlation.to_numpy(dtype=np.float64)
    records: list[dict[str, Any]] = []
    for i in range(len(features)):
        row = values[i, i + 1 :]
        hits = np.flatnonzero(row >= float(minimum))
        for offset in hits:
            j = i + 1 + int(offset)
            records.append(
                {
                    "feature_a": features[i],
                    "feature_b": features[j],
                    "combined_abs_corr": float(values[i, j]),
                }
            )
    result = pd.DataFrame.from_records(records)
    if not result.empty:
        result.sort_values(
            ["combined_abs_corr", "feature_a", "feature_b"],
            ascending=[False, True, True],
            kind="mergesort",
            inplace=True,
        )
        result.reset_index(drop=True, inplace=True)
    return result


def semantic_groups(features: Sequence[str], audit: pd.DataFrame) -> dict[str, list[str]]:
    lookup = audit.drop_duplicates("feature").set_index("feature") if "feature" in audit.columns else pd.DataFrame()
    groups: dict[str, list[str]] = {}
    for feature in features:
        group: str | None = None
        if not lookup.empty and feature in lookup.index and "group" in lookup.columns:
            value = lookup.at[feature, "group"]
            if pd.notna(value):
                group = str(value)
        if not group:
            tokens = feature.split("_")
            group = "_".join(tokens[:2]) if len(tokens) >= 2 else "other"
        groups.setdefault(group, []).append(feature)
    return groups


def build_organic_conditions(
    features: Sequence[str],
    correlation: pd.DataFrame,
    audit: pd.DataFrame,
    *,
    cluster_thresholds: Sequence[float],
    primary_cluster_threshold: float,
    pair_threshold: float,
    neighborhood_ks: Sequence[int],
    include_single: bool = True,
    include_pair: bool = True,
    include_clusters: bool = True,
    include_cluster_solo: bool = True,
    include_groups: bool = True,
    include_neighborhoods: bool = True,
) -> tuple[list[OrganicCondition], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_set = set(features)
    conditions: list[OrganicCondition] = []
    cluster_assignments = hierarchical_clusters(correlation, cluster_thresholds)
    pairs = correlation_pairs(correlation, pair_threshold)

    if include_single:
        for feature in features:
            conditions.append(
                OrganicCondition(
                    stage="feature_loo",
                    test_type="single_feature_loo",
                    condition_id=f"LOO::{feature}",
                    dropped_features=(feature,),
                    representative_feature=feature,
                    relation_set_hash=_set_hash([feature]),
                )
            )

    seen_cluster_sets: set[tuple[str, ...]] = set()
    if include_clusters:
        for threshold in cluster_thresholds:
            part = cluster_assignments[np.isclose(cluster_assignments["threshold"], float(threshold))]
            for cluster_id, group in part.groupby("cluster_id", sort=True):
                members = tuple(sorted(feature for feature in group["feature"].astype(str) if feature in feature_set))
                if len(members) < 2 or members in seen_cluster_sets:
                    continue
                seen_cluster_sets.add(members)
                conditions.append(
                    OrganicCondition(
                        stage="correlation_cluster_loo",
                        test_type="correlation_cluster_loo",
                        condition_id=f"CORR_CLUSTER_LOO::T{threshold:.3f}::{cluster_id}::{_set_hash(members)}",
                        dropped_features=members,
                        cluster_id=int(cluster_id),
                        relation_threshold=float(threshold),
                        relation_set_hash=_set_hash(members),
                    )
                )

    primary = cluster_assignments[
        np.isclose(cluster_assignments["threshold"], float(primary_cluster_threshold))
    ].copy()
    if primary.empty:
        raise ValueError(f"primary cluster threshold {primary_cluster_threshold} 결과가 없습니다")
    primary_members: dict[str, tuple[str, ...]] = {}
    for cluster_id, group in primary.groupby("cluster_id", sort=True):
        members = tuple(sorted(group["feature"].astype(str).tolist()))
        for feature in members:
            primary_members[feature] = members

    if include_cluster_solo:
        for feature in features:
            members = primary_members.get(feature, (feature,))
            peers = tuple(member for member in members if member != feature)
            if not peers:
                continue
            cluster_id = int(primary.loc[primary["feature"].eq(feature), "cluster_id"].iloc[0])
            conditions.append(
                OrganicCondition(
                    stage="cluster_solo",
                    test_type="cluster_solo_keep_feature",
                    condition_id=f"CLUSTER_SOLO::T{primary_cluster_threshold:.3f}::{feature}",
                    dropped_features=tuple(sorted(peers)),
                    representative_feature=feature,
                    cluster_id=cluster_id,
                    relation_threshold=float(primary_cluster_threshold),
                    relation_set_hash=_set_hash(members),
                )
            )

    if include_pair and not pairs.empty:
        for row in pairs.itertuples(index=False):
            members = tuple(sorted((str(row.feature_a), str(row.feature_b))))
            conditions.append(
                OrganicCondition(
                    stage="pair_loo",
                    test_type="correlated_pair_loo",
                    condition_id=f"PAIR_LOO::{members[0]}::{members[1]}",
                    dropped_features=members,
                    pair_feature_a=members[0],
                    pair_feature_b=members[1],
                    relation_threshold=float(row.combined_abs_corr),
                    relation_set_hash=_set_hash(members),
                )
            )

    if include_neighborhoods and neighborhood_ks:
        corr_values = correlation.to_numpy(dtype=np.float64)
        for i, feature in enumerate(features):
            order = np.argsort(-corr_values[i], kind="mergesort")
            order = [int(index) for index in order if int(index) != i]
            for k in neighborhood_ks:
                if k <= 0:
                    continue
                neighbor_indices = order[: min(int(k), len(order))]
                members = tuple(sorted([feature] + [features[index] for index in neighbor_indices]))
                if len(members) <= 1:
                    continue
                minimum_corr = min(float(correlation.at[feature, member]) for member in members if member != feature)
                conditions.append(
                    OrganicCondition(
                        stage="neighborhood_loo",
                        test_type="correlation_neighborhood_loo",
                        condition_id=f"NEIGHBORHOOD_LOO::K{k}::{feature}",
                        dropped_features=members,
                        representative_feature=feature,
                        relation_threshold=minimum_corr,
                        relation_k=int(k),
                        relation_set_hash=_set_hash(members),
                    )
                )

    if include_groups:
        for group_name, group_features in sorted(semantic_groups(features, audit).items()):
            members = tuple(sorted(group_features))
            if len(members) < 2 or len(members) >= len(features):
                continue
            conditions.append(
                OrganicCondition(
                    stage="semantic_group_loo",
                    test_type="semantic_group_loo",
                    condition_id=f"SEMANTIC_GROUP_LOO::{group_name}",
                    dropped_features=members,
                    feature_group=group_name,
                    relation_set_hash=_set_hash(members),
                )
            )

    # De-duplicate exact task meaning while preserving distinct test types where interpretation differs.
    deduped: list[OrganicCondition] = []
    seen: set[tuple[str, tuple[str, ...], str | None]] = set()
    for condition in conditions:
        key = (condition.test_type, condition.dropped_features, condition.representative_feature)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(condition)

    primary = primary.rename(columns={"cluster_id": "primary_cluster_id", "cluster_size": "primary_cluster_size"})
    return deduped, cluster_assignments, pairs, primary


def datewise_percentile_rank(scores: np.ndarray, dates: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    dates = np.asarray(dates)
    out = np.full(len(scores), np.nan, dtype=np.float64)
    frame = pd.DataFrame({"score": scores, "date": dates, "row": np.arange(len(scores), dtype=np.int64)})
    for _, group in frame.groupby("date", sort=False):
        valid = np.isfinite(group["score"].to_numpy(dtype=np.float64))
        if not valid.any():
            continue
        rows = group.loc[valid, "row"].to_numpy(dtype=np.int64)
        values = group.loc[valid, "score"].rank(method="average", pct=True).to_numpy(dtype=np.float64)
        out[rows] = values
    return out


def build_error_group_codes(
    target: np.ndarray,
    baseline_score: np.ndarray,
    dates: np.ndarray,
    *,
    top_quantile: float = 0.80,
    low_quantile: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(target, dtype=np.int8)
    rank = datewise_percentile_rank(baseline_score, dates)
    codes = np.full(len(y), ERROR_OTHER, dtype=np.int8)
    top = np.isfinite(rank) & (rank >= float(top_quantile))
    low = np.isfinite(rank) & (rank <= float(low_quantile))
    codes[top & (y == 1)] = ERROR_A_TOP_TP
    codes[top & (y == 0)] = ERROR_B_TOP_FP
    codes[low & (y == 1)] = ERROR_C_LOW_FN
    codes[low & (y == 0)] = ERROR_D_LOW_TN
    return codes, rank


def _binary_group_auc(scores: np.ndarray, positive_mask: np.ndarray, negative_mask: np.ndarray) -> float:
    use = (positive_mask | negative_mask) & np.isfinite(scores)
    if int(use.sum()) < 8:
        return float("nan")
    y = positive_mask[use].astype(np.int8)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, scores[use]))


def _rejection_at_positive_retention(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    retention: float,
) -> float:
    pos = np.asarray(scores[positive_mask & np.isfinite(scores)], dtype=np.float64)
    neg = np.asarray(scores[negative_mask & np.isfinite(scores)], dtype=np.float64)
    if len(pos) < 5 or len(neg) < 5:
        return float("nan")
    threshold = float(np.quantile(pos, max(0.0, 1.0 - float(retention)), method="lower"))
    return float(np.mean(neg < threshold))


def _positive_recall_at_negative_fpr(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    fpr: float,
) -> float:
    pos = np.asarray(scores[positive_mask & np.isfinite(scores)], dtype=np.float64)
    neg = np.asarray(scores[negative_mask & np.isfinite(scores)], dtype=np.float64)
    if len(pos) < 5 or len(neg) < 5:
        return float("nan")
    threshold = float(np.quantile(neg, max(0.0, 1.0 - float(fpr)), method="higher"))
    return float(np.mean(pos >= threshold))


def error_strata_metrics(codes: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    codes = np.asarray(codes, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    a = codes == ERROR_A_TOP_TP
    b = codes == ERROR_B_TOP_FP
    c = codes == ERROR_C_LOW_FN
    d = codes == ERROR_D_LOW_TN
    result: dict[str, Any] = {
        "error_A_rows": int(a.sum()),
        "error_B_rows": int(b.sum()),
        "error_C_rows": int(c.sum()),
        "error_D_rows": int(d.sum()),
        "ab_auc": _binary_group_auc(scores, a, b),
        "cd_auc": _binary_group_auc(scores, c, d),
        "ab_fp_rejection_at_a90": _rejection_at_positive_retention(scores, a, b, 0.90),
        "ab_fp_rejection_at_a80": _rejection_at_positive_retention(scores, a, b, 0.80),
        "cd_recall_at_d_fpr10": _positive_recall_at_negative_fpr(scores, c, d, 0.10),
        "cd_recall_at_d_fpr20": _positive_recall_at_negative_fpr(scores, c, d, 0.20),
    }
    return result


def classify_organic_role(
    single_utility: float,
    cluster_utility: float,
    rescue_by_feature: float,
    peer_recovery: float,
    *,
    epsilon: float,
) -> str:
    values = [single_utility, cluster_utility, rescue_by_feature, peer_recovery]
    if not any(np.isfinite(value) for value in values):
        return "INSUFFICIENT"
    s = single_utility if np.isfinite(single_utility) else 0.0
    c = cluster_utility if np.isfinite(cluster_utility) else 0.0
    r = rescue_by_feature if np.isfinite(rescue_by_feature) else 0.0
    p = peer_recovery if np.isfinite(peer_recovery) else 0.0
    if c < -epsilon:
        return "HARMFUL_CORRELATED_BLOCK"
    if s < -epsilon and c >= -epsilon:
        return "SUPPRESSOR_OR_HARMFUL_MEMBER"
    if c <= epsilon and abs(s) <= epsilon:
        return "NEUTRAL_OR_DISPENSABLE"
    if s > epsilon and r > epsilon and p <= epsilon:
        return "UNIQUE_CORE"
    if abs(s) <= epsilon and r > epsilon and p > epsilon:
        return "REDUNDANT_BACKUP_CORE"
    if r >= c - epsilon and c > epsilon:
        return "STRONG_REPRESENTATIVE"
    if s > epsilon and r <= epsilon and c > epsilon:
        return "INTERACTION_DEPENDENT"
    if c > epsilon and p > epsilon and r > epsilon:
        return "COMPLEMENTARY_CLUSTER_MEMBER"
    if c > epsilon and abs(s) <= epsilon:
        return "CORRELATED_BLOCK_ONLY"
    return "INCONCLUSIVE_OR_MIXED"


def _lookup_value(
    summary: pd.DataFrame,
    backend: str,
    test_type: str,
    condition_id: str,
    column: str,
) -> float:
    part = summary[
        summary["backend"].astype(str).eq(str(backend))
        & summary["test_type"].astype(str).eq(str(test_type))
        & summary["condition_id"].astype(str).eq(str(condition_id))
    ]
    if part.empty or column not in part.columns:
        return float("nan")
    value = pd.to_numeric(part.iloc[0][column], errors="coerce")
    return float(value) if pd.notna(value) else float("nan")


def build_organic_feature_map(
    summary: pd.DataFrame,
    plan: pd.DataFrame,
    features: Sequence[str],
    primary_clusters: pd.DataFrame,
    *,
    role_prefix: str = "selection",
    epsilon: float = 0.0005,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    utility_col = f"{role_prefix}_mean_pr_auc_utility"
    ab_col = f"{role_prefix}_mean_utility_ab_auc"
    cd_col = f"{role_prefix}_mean_utility_cd_auc"
    precision_col = f"{role_prefix}_mean_utility_best_precision_min_alerts"
    precision70_recall_col = f"{role_prefix}_mean_utility_max_recall_at_precision_target"
    backends = sorted(summary["backend"].dropna().astype(str).unique())
    primary_lookup = primary_clusters.drop_duplicates("feature").set_index("feature")
    plan_lookup = plan.set_index("condition_id", drop=False)
    rows: list[dict[str, Any]] = []

    # Resolve unique primary-cluster condition by set hash.
    cluster_conditions = plan[plan["test_type"].eq("correlation_cluster_loo")].copy()
    solo_conditions = plan[plan["test_type"].eq("cluster_solo_keep_feature")].copy()
    neighborhood_conditions = plan[plan["test_type"].eq("correlation_neighborhood_loo")].copy()

    for backend in backends:
        for feature in features:
            single_id = f"LOO::{feature}"
            single_u = _lookup_value(summary, backend, "single_feature_loo", single_id, utility_col)
            single_ab = _lookup_value(summary, backend, "single_feature_loo", single_id, ab_col)
            single_cd = _lookup_value(summary, backend, "single_feature_loo", single_id, cd_col)
            single_precision = _lookup_value(summary, backend, "single_feature_loo", single_id, precision_col)
            single_precision70_recall = _lookup_value(summary, backend, "single_feature_loo", single_id, precision70_recall_col)
            cluster_id = None
            cluster_size = 1
            cluster_hash = None
            if feature in primary_lookup.index:
                cluster_id = int(primary_lookup.at[feature, "primary_cluster_id"])
                cluster_size = int(primary_lookup.at[feature, "primary_cluster_size"])
                members = primary_clusters[
                    primary_clusters["primary_cluster_id"].eq(cluster_id)
                ]["feature"].astype(str).tolist()
                cluster_hash = _set_hash(members)
            cluster_u = float("nan")
            cluster_ab = float("nan")
            cluster_cd = float("nan")
            cluster_condition_id = None
            if cluster_hash is not None:
                hits = cluster_conditions[cluster_conditions["relation_set_hash"].astype(str).eq(cluster_hash)]
                if not hits.empty:
                    # Prefer the primary threshold exact cluster if present.
                    hits = hits.sort_values("relation_threshold", ascending=True)
                    row = hits.iloc[-1]
                    cluster_condition_id = str(row["condition_id"])
                    cluster_u = _lookup_value(summary, backend, "correlation_cluster_loo", cluster_condition_id, utility_col)
                    cluster_ab = _lookup_value(summary, backend, "correlation_cluster_loo", cluster_condition_id, ab_col)
                    cluster_cd = _lookup_value(summary, backend, "correlation_cluster_loo", cluster_condition_id, cd_col)
            solo_id = f"CLUSTER_SOLO::T{float(primary_clusters['threshold'].iloc[0]) if 'threshold' in primary_clusters.columns else 0.92:.3f}::{feature}"
            solo_hits = solo_conditions[solo_conditions["representative_feature"].astype(str).eq(feature)]
            peers_removed_u = float("nan")
            if not solo_hits.empty:
                solo_id = str(solo_hits.iloc[0]["condition_id"])
                peers_removed_u = _lookup_value(summary, backend, "cluster_solo_keep_feature", solo_id, utility_col)
            if cluster_size <= 1:
                cluster_u = single_u
                peers_removed_u = 0.0
            rescue = cluster_u - peers_removed_u if np.isfinite(cluster_u) and np.isfinite(peers_removed_u) else float("nan")
            peer_recovery = cluster_u - single_u if np.isfinite(cluster_u) and np.isfinite(single_u) else float("nan")
            redundancy_mask = rescue - single_u if np.isfinite(rescue) and np.isfinite(single_u) else float("nan")

            record: dict[str, Any] = {
                "backend": backend,
                "feature": feature,
                "primary_cluster_id": cluster_id,
                "primary_cluster_size": cluster_size,
                "single_utility": single_u,
                "cluster_utility": cluster_u,
                "peers_removed_utility": peers_removed_u,
                "rescue_by_feature": rescue,
                "peer_recovery_after_feature_drop": peer_recovery,
                "redundancy_masking_gain": redundancy_mask,
                "single_ab_utility": single_ab,
                "single_cd_utility": single_cd,
                "single_best_precision_utility": single_precision,
                "single_precision70_recall_utility": single_precision70_recall,
                "cluster_ab_utility": cluster_ab,
                "cluster_cd_utility": cluster_cd,
            }
            for k in sorted(pd.to_numeric(neighborhood_conditions["relation_k"], errors="coerce").dropna().astype(int).unique()):
                hits = neighborhood_conditions[
                    neighborhood_conditions["representative_feature"].astype(str).eq(feature)
                    & pd.to_numeric(neighborhood_conditions["relation_k"], errors="coerce").eq(k)
                ]
                if hits.empty:
                    record[f"neighborhood_k{k}_utility"] = np.nan
                    record[f"neighborhood_k{k}_unmasking_vs_single"] = np.nan
                    continue
                condition_id = str(hits.iloc[0]["condition_id"])
                value = _lookup_value(summary, backend, "correlation_neighborhood_loo", condition_id, utility_col)
                record[f"neighborhood_k{k}_utility"] = value
                record[f"neighborhood_k{k}_unmasking_vs_single"] = value - single_u if np.isfinite(value) and np.isfinite(single_u) else np.nan
            record["organic_role"] = classify_organic_role(
                single_u,
                cluster_u,
                rescue,
                peer_recovery,
                epsilon=epsilon,
            )
            if np.isfinite(single_ab) and np.isfinite(single_cd):
                if single_ab > epsilon and single_cd > epsilon:
                    precision_role = "BOTH_AB_CD"
                elif single_ab > epsilon:
                    precision_role = "FP_REJECTION_AB"
                elif single_cd > epsilon:
                    precision_role = "MISS_RECOVERY_CD"
                elif single_ab < -epsilon or single_cd < -epsilon:
                    precision_role = "ERROR_AXIS_HARMFUL"
                else:
                    precision_role = "ERROR_AXIS_NEUTRAL"
            else:
                precision_role = "ERROR_AXIS_INSUFFICIENT"
            record["error_axis_role"] = precision_role
            rows.append(record)
    result = pd.DataFrame.from_records(rows)
    if not result.empty:
        result["organic_priority_score"] = (
            pd.to_numeric(result["cluster_utility"], errors="coerce").clip(lower=0).fillna(0.0)
            + 0.7 * pd.to_numeric(result["rescue_by_feature"], errors="coerce").clip(lower=0).fillna(0.0)
            + 0.5 * pd.to_numeric(result["single_ab_utility"], errors="coerce").clip(lower=0).fillna(0.0)
            + 0.3 * pd.to_numeric(result["single_cd_utility"], errors="coerce").clip(lower=0).fillna(0.0)
            + 0.8 * pd.to_numeric(result["single_best_precision_utility"], errors="coerce").clip(lower=0).fillna(0.0)
            + 0.5 * pd.to_numeric(result["single_precision70_recall_utility"], errors="coerce").clip(lower=0).fillna(0.0)
        )
        result.sort_values(["backend", "organic_priority_score", "feature"], ascending=[True, False, True], kind="mergesort", inplace=True)
    return result


def build_pair_nonadditivity(
    summary: pd.DataFrame,
    plan: pd.DataFrame,
    pair_table: pd.DataFrame,
    *,
    role_prefix: str = "selection",
) -> pd.DataFrame:
    if summary.empty or pair_table.empty:
        return pd.DataFrame()
    utility_col = f"{role_prefix}_mean_pr_auc_utility"
    pair_plan = plan[plan["test_type"].eq("correlated_pair_loo")]
    rows: list[dict[str, Any]] = []
    for backend in sorted(summary["backend"].dropna().astype(str).unique()):
        for pair in pair_table.itertuples(index=False):
            a = str(pair.feature_a)
            b = str(pair.feature_b)
            condition_id = f"PAIR_LOO::{min(a,b)}::{max(a,b)}"
            pair_u = _lookup_value(summary, backend, "correlated_pair_loo", condition_id, utility_col)
            a_u = _lookup_value(summary, backend, "single_feature_loo", f"LOO::{a}", utility_col)
            b_u = _lookup_value(summary, backend, "single_feature_loo", f"LOO::{b}", utility_col)
            if not any(np.isfinite(value) for value in (pair_u, a_u, b_u)):
                continue
            rows.append(
                {
                    "backend": backend,
                    "feature_a": a,
                    "feature_b": b,
                    "combined_abs_corr": float(pair.combined_abs_corr),
                    "single_a_utility": a_u,
                    "single_b_utility": b_u,
                    "pair_utility": pair_u,
                    "pair_nonadditivity": pair_u - (a_u + b_u) if all(np.isfinite(v) for v in (pair_u,a_u,b_u)) else np.nan,
                    "joint_unmasking_over_best_single": pair_u - max(a_u,b_u) if all(np.isfinite(v) for v in (pair_u,a_u,b_u)) else np.nan,
                    "mutual_backup_score": pair_u - max(0.0, a_u) - max(0.0, b_u) if all(np.isfinite(v) for v in (pair_u,a_u,b_u)) else np.nan,
                }
            )
    result = pd.DataFrame.from_records(rows)
    if not result.empty:
        result.sort_values(["backend", "joint_unmasking_over_best_single", "combined_abs_corr"], ascending=[True, False, False], kind="mergesort", inplace=True)
    return result


def build_cluster_synergy(
    summary: pd.DataFrame,
    plan: pd.DataFrame,
    *,
    role_prefix: str = "selection",
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    utility_col = f"{role_prefix}_mean_pr_auc_utility"
    cluster_plan = plan[plan["test_type"].eq("correlation_cluster_loo")]
    rows: list[dict[str, Any]] = []
    for backend in sorted(summary["backend"].dropna().astype(str).unique()):
        for cond in cluster_plan.itertuples(index=False):
            members = [value for value in str(cond.dropped_features).split("|") if value]
            cluster_u = _lookup_value(summary, backend, "correlation_cluster_loo", str(cond.condition_id), utility_col)
            singles = np.asarray([
                _lookup_value(summary, backend, "single_feature_loo", f"LOO::{feature}", utility_col)
                for feature in members
            ], dtype=np.float64)
            finite = singles[np.isfinite(singles)]
            rows.append(
                {
                    "backend": backend,
                    "condition_id": str(cond.condition_id),
                    "relation_threshold": float(cond.relation_threshold),
                    "relation_set_hash": str(cond.relation_set_hash),
                    "cluster_size": len(members),
                    "members": "|".join(members),
                    "cluster_utility": cluster_u,
                    "sum_single_utility": float(np.sum(finite)) if len(finite) else np.nan,
                    "max_single_utility": float(np.max(finite)) if len(finite) else np.nan,
                    "mean_single_utility": float(np.mean(finite)) if len(finite) else np.nan,
                    "hidden_cluster_value_vs_sum": cluster_u - float(np.sum(finite)) if len(finite) and np.isfinite(cluster_u) else np.nan,
                    "hidden_cluster_value_vs_best": cluster_u - float(np.max(finite)) if len(finite) and np.isfinite(cluster_u) else np.nan,
                }
            )
    result = pd.DataFrame.from_records(rows)
    if not result.empty:
        result.sort_values(["backend", "hidden_cluster_value_vs_best", "cluster_utility"], ascending=[True, False, False], kind="mergesort", inplace=True)
    return result


def selective_precision_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    *,
    target_precision: float = 0.70,
    minimum_alerts: int = 30,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s) & np.isin(y, [0, 1])
    y = y[valid]
    s = s[valid]
    result = {
        "best_precision_min_alerts": np.nan,
        "best_precision_min_alerts_count": 0,
        "max_recall_at_precision_target": 0.0,
        "precision_at_max_recall_precision_target": np.nan,
        "alerts_at_precision_target": 0,
        "precision_target_reachable": False,
    }
    if len(y) == 0 or int(np.sum(y == 1)) == 0:
        return result
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    tp = np.cumsum(sorted_y == 1)
    n = np.arange(1, len(y) + 1, dtype=np.int64)
    precision = tp / n
    recall = tp / max(1, int(np.sum(y == 1)))
    eligible = n >= max(1, int(minimum_alerts))
    if eligible.any():
        p = precision.copy()
        p[~eligible] = -np.inf
        best_idx = int(np.argmax(p))
        if np.isfinite(p[best_idx]):
            result["best_precision_min_alerts"] = float(precision[best_idx])
            result["best_precision_min_alerts_count"] = int(n[best_idx])
    safe = eligible & (precision >= float(target_precision))
    if safe.any():
        safe_idx = np.flatnonzero(safe)
        # Maximize recall, then alert count. Prefix recall is non-decreasing, so last safe is correct.
        idx = int(safe_idx[-1])
        result["max_recall_at_precision_target"] = float(recall[idx])
        result["precision_at_max_recall_precision_target"] = float(precision[idx])
        result["alerts_at_precision_target"] = int(n[idx])
        result["precision_target_reachable"] = True
    return result
