from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


SCHEMA_VERSION = "crashwatch_surge_tickerwise_correlation_map_v10"


@dataclass(frozen=True)
class HierarchyConfig:
    prior_strength: float = 120.0
    minimum_peer_tickers: int = 2
    weak_effect: float = 0.015
    strong_effect: float = 0.035
    unique_delta: float = 0.025
    amplified_delta: float = 0.020
    reversal_effect: float = 0.020
    minimum_reliability: float = 0.10
    confirmation_margin: float = 0.0
    recent_margin: float = 0.0
    industry_weight: float = 0.35
    bucket_weight: float = 0.35
    market_weight: float = 0.20
    global_weight: float = 0.10
    similarity_min_common: int = 20
    similarity_edge_threshold: float = 0.35
    similarity_top_k: int = 5
    similarity_cluster_threshold: float = 0.30
    similarity_feature_count: int = 120


def _finite_number(value: Any, fallback: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return number if math.isfinite(number) else float(fallback)


def _sign(value: float, tolerance: float = 1e-12) -> int:
    if not math.isfinite(value) or abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def normalize_map_frames(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize raw and transformed maps to a common node identifier."""
    summary_out = summary.copy()
    fold_out = by_fold.copy()
    if "transformed_feature" not in summary_out.columns:
        summary_out["transformed_feature"] = np.nan
    if "transformed_feature" not in fold_out.columns:
        fold_out["transformed_feature"] = np.nan
    summary_out["node_id"] = summary_out["transformed_feature"].where(
        summary_out["transformed_feature"].notna(), summary_out["feature"]
    ).astype(str)
    fold_out["node_id"] = fold_out["transformed_feature"].where(
        fold_out["transformed_feature"].notna(), fold_out["feature"]
    ).astype(str)
    summary_out["source_feature"] = summary_out["feature"].astype(str)
    fold_out["source_feature"] = fold_out["feature"].astype(str)
    for column in ["ticker", "axis", "transform"]:
        summary_out[column] = summary_out[column].astype(str)
        fold_out[column] = fold_out[column].astype(str)
    return summary_out, fold_out


def _selection_sample_summary(
    by_fold: pd.DataFrame,
    selection_fold_ids: Sequence[int],
) -> pd.DataFrame:
    selection = by_fold.loc[by_fold["fold_id"].astype(int).isin({int(v) for v in selection_fold_ids})].copy()
    if selection.empty:
        return pd.DataFrame()
    selection["positive_n"] = pd.to_numeric(selection["positive_n"], errors="coerce").fillna(0.0)
    selection["negative_n"] = pd.to_numeric(selection["negative_n"], errors="coerce").fillna(0.0)
    selection["valid_rows"] = pd.to_numeric(selection["valid_rows"], errors="coerce").fillna(0.0)
    p = selection["positive_n"].to_numpy(dtype=float)
    n = selection["negative_n"].to_numpy(dtype=float)
    selection["effective_n"] = np.where((p > 0) & (n > 0), 2.0 * p * n / np.maximum(p + n, 1e-12), 0.0)
    aggregation: dict[str, tuple[str, str]] = {
        "selection_positive_sum": ("positive_n", "sum"),
        "selection_negative_sum": ("negative_n", "sum"),
        "selection_valid_sum": ("valid_rows", "sum"),
        "selection_effective_n": ("effective_n", "sum"),
        "selection_observed_folds": ("fold_id", "nunique"),
    }
    optional_metrics = [
        "matched_concordance",
        "matched_pair_count",
        "positive_coverage",
        "negative_coverage",
        "smd",
        "ks",
        "wasserstein_iqr",
        "js_divergence",
        "missing_rate_difference",
    ]
    for metric in optional_metrics:
        if metric in selection.columns:
            selection[metric] = pd.to_numeric(selection[metric], errors="coerce")
            aggregation[f"selection_mean_{metric}"] = (metric, "mean")
    grouped = selection.groupby(["ticker", "axis", "node_id", "source_feature", "transform"], as_index=False).agg(**aggregation)
    return grouped


def _weighted_leave_one_out(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    effect_column: str,
    weight_column: str,
    prefix: str,
) -> pd.DataFrame:
    """Weighted leave-one-ticker-out reference within each map stratum."""
    keys = list(group_columns)
    work = frame[keys + ["ticker", effect_column, weight_column]].copy()
    work[effect_column] = pd.to_numeric(work[effect_column], errors="coerce")
    work[weight_column] = pd.to_numeric(work[weight_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    finite = np.isfinite(work[effect_column].to_numpy(dtype=float)) & (work[weight_column].to_numpy(dtype=float) > 0)
    work["_weighted"] = np.where(finite, work[effect_column] * work[weight_column], 0.0)
    work["_valid_weight"] = np.where(finite, work[weight_column], 0.0)
    work["_valid_ticker"] = finite.astype(np.int8)
    totals = work.groupby(keys, as_index=False).agg(
        _total_weight=("_valid_weight", "sum"),
        _total_weighted=("_weighted", "sum"),
        _total_tickers=("_valid_ticker", "sum"),
    )
    merged = work.merge(totals, on=keys, how="left", validate="many_to_one")
    own_weight = merged["_valid_weight"].to_numpy(dtype=float)
    own_weighted = merged["_weighted"].to_numpy(dtype=float)
    denominator = merged["_total_weight"].to_numpy(dtype=float) - own_weight
    numerator = merged["_total_weighted"].to_numpy(dtype=float) - own_weighted
    peer_count = merged["_total_tickers"].to_numpy(dtype=float) - merged["_valid_ticker"].to_numpy(dtype=float)
    effect = np.full(len(denominator), np.nan, dtype=float)
    np.divide(numerator, denominator, out=effect, where=denominator > 0)
    result = merged[keys + ["ticker"]].copy()
    result[f"{prefix}_loo_effect"] = effect
    result[f"{prefix}_peer_weight"] = denominator
    result[f"{prefix}_peer_tickers"] = peer_count.astype(int)
    return result


def _combine_prior(row: pd.Series, config: HierarchyConfig) -> tuple[float, str, int]:
    candidates = [
        ("industry", config.industry_weight),
        ("bucket", config.bucket_weight),
        ("market", config.market_weight),
        ("global", config.global_weight),
    ]
    weighted_sum = 0.0
    total_weight = 0.0
    used: list[str] = []
    peer_total = 0
    for name, configured_weight in candidates:
        effect = _finite_number(row.get(f"{name}_loo_effect"))
        peers = int(_finite_number(row.get(f"{name}_peer_tickers"), 0.0))
        if not math.isfinite(effect):
            continue
        if name != "global" and peers < int(config.minimum_peer_tickers):
            continue
        if name == "global" and peers < 1:
            continue
        weighted_sum += float(configured_weight) * effect
        total_weight += float(configured_weight)
        used.append(name)
        peer_total += peers
    if total_weight <= 0:
        return math.nan, "NONE", 0
    return weighted_sum / total_weight, "+".join(used), peer_total


def _taxonomy(row: pd.Series, config: HierarchyConfig) -> str:
    reliability = _finite_number(row.get("ticker_map_reliability"), 0.0)
    posterior = _finite_number(row.get("posterior_signed_effect"))
    prior = _finite_number(row.get("peer_prior_signed_effect"))
    delta = _finite_number(row.get("ticker_specific_delta"))
    if reliability < config.minimum_reliability or not math.isfinite(posterior):
        return "LOW_EVIDENCE"
    if abs(posterior) < config.weak_effect:
        return "WEAK_OR_NEUTRAL"
    posterior_sign = _sign(posterior)
    prior_sign = _sign(prior)
    if (
        prior_sign != 0
        and posterior_sign != 0
        and prior_sign != posterior_sign
        and abs(prior) >= config.reversal_effect
        and abs(posterior) >= config.reversal_effect
    ):
        return "TICKER_DIRECTION_REVERSAL"
    if (
        math.isfinite(prior)
        and abs(prior) < config.weak_effect
        and abs(posterior) >= config.strong_effect
        and abs(delta) >= config.unique_delta
    ):
        return "TICKER_UNIQUE_DRIVER"
    if math.isfinite(prior) and prior_sign == posterior_sign and abs(delta) >= config.amplified_delta:
        if abs(posterior) > abs(prior):
            return "TICKER_AMPLIFIED_DRIVER"
        return "TICKER_DAMPENED_DRIVER"
    industry_effect = _finite_number(row.get("industry_loo_effect"))
    bucket_effect = _finite_number(row.get("bucket_loo_effect"))
    global_effect = _finite_number(row.get("global_loo_effect"))
    if math.isfinite(industry_effect) and _sign(industry_effect) == posterior_sign and abs(industry_effect) >= config.weak_effect:
        return "INDUSTRY_SHARED_DRIVER"
    if math.isfinite(bucket_effect) and _sign(bucket_effect) == posterior_sign and abs(bucket_effect) >= config.weak_effect:
        return "BUCKET_SHARED_DRIVER"
    if math.isfinite(global_effect) and _sign(global_effect) == posterior_sign and abs(global_effect) >= config.weak_effect:
        return "UNIVERSAL_DRIVER"
    return "MIXED_TICKER_EFFECT"


def build_hierarchical_effect_map(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    ticker_metadata: pd.DataFrame,
    selection_fold_ids: Sequence[int],
    config: HierarchyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create raw, peer-referenced and partially pooled ticker-specific effects.

    No pooled prediction model is trained. The peer references only stabilize the
    descriptive map and are leave-one-ticker-out.
    """
    normalized_summary, normalized_fold = normalize_map_frames(summary, by_fold)
    samples = _selection_sample_summary(normalized_fold, selection_fold_ids)
    result = normalized_summary.merge(
        samples,
        on=["ticker", "axis", "node_id", "source_feature", "transform"],
        how="left",
        validate="one_to_one",
    )
    metadata_columns = [column for column in ["ticker", "name", "market", "bucket", "industry"] if column in ticker_metadata.columns]
    metadata = ticker_metadata[metadata_columns].drop_duplicates("ticker", keep="last").copy()
    result = result.merge(metadata, on="ticker", how="left", validate="many_to_one")
    for column in ["name", "market", "bucket", "industry"]:
        if column not in result.columns:
            result[column] = "UNKNOWN"
        result[column] = result[column].fillna("UNKNOWN").astype(str)
    fixed = pd.to_numeric(result["selection_mean_shrunk_auc"], errors="coerce")
    fallback_fixed = pd.to_numeric(result["selection_mean_fixed_auc"], errors="coerce")
    fixed = fixed.where(np.isfinite(fixed), fallback_fixed)
    direction = pd.to_numeric(result["selection_direction"], errors="coerce").fillna(0.0)
    result["ticker_signed_effect"] = direction * (fixed - 0.5)
    effective_n = pd.to_numeric(result["selection_effective_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
    fold_count = pd.to_numeric(result["selection_fold_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    consistency = pd.to_numeric(result["selection_direction_consistency"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    sample_reliability = effective_n / (effective_n + float(config.prior_strength))
    fold_reliability = np.minimum(fold_count / max(len(selection_fold_ids), 1), 1.0)
    result["ticker_map_reliability"] = sample_reliability * np.sqrt(fold_reliability * consistency)
    evidence = pd.to_numeric(result.get("selection_evidence_score", 0.0), errors="coerce").fillna(0.0)
    result["prior_weight"] = effective_n * (0.25 + 0.75 * consistency) * (1.0 + np.minimum(evidence, 2.0))
    reference_keys = ["axis", "node_id", "source_feature", "transform"]
    global_reference = _weighted_leave_one_out(result, reference_keys, "ticker_signed_effect", "prior_weight", "global")
    result = result.merge(global_reference, on=reference_keys + ["ticker"], how="left", validate="one_to_one")
    for level, column in [("market", "market"), ("bucket", "bucket"), ("industry", "industry")]:
        reference = _weighted_leave_one_out(
            result,
            [*reference_keys, column],
            "ticker_signed_effect",
            "prior_weight",
            level,
        )
        result = result.merge(reference, on=[*reference_keys, column, "ticker"], how="left", validate="one_to_one")
    priors = result.apply(lambda row: _combine_prior(row, config), axis=1)
    result["peer_prior_signed_effect"] = [value[0] for value in priors]
    result["peer_prior_sources"] = [value[1] for value in priors]
    result["peer_reference_count"] = [value[2] for value in priors]
    raw_effect = pd.to_numeric(result["ticker_signed_effect"], errors="coerce").to_numpy(dtype=float)
    prior_effect = pd.to_numeric(result["peer_prior_signed_effect"], errors="coerce").to_numpy(dtype=float)
    reliability = pd.to_numeric(result["ticker_map_reliability"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    posterior = np.where(
        np.isfinite(prior_effect),
        reliability * raw_effect + (1.0 - reliability) * prior_effect,
        raw_effect,
    )
    result["posterior_signed_effect"] = posterior
    result["posterior_direction"] = np.where(np.isfinite(posterior), np.sign(posterior), 0).astype(int)
    result["posterior_oriented_auc"] = 0.5 + np.abs(posterior)
    result["ticker_specific_delta"] = posterior - prior_effect
    result["ticker_specificity_score"] = np.abs(result["ticker_specific_delta"]) * result["ticker_map_reliability"]
    result["confirmation_direction_support"] = pd.to_numeric(
        result.get("confirmation_mean_fixed_auc"), errors="coerce"
    ) >= 0.5 + float(config.confirmation_margin)
    result["recent_direction_support"] = pd.to_numeric(
        result.get("recent_mean_fixed_auc"), errors="coerce"
    ) >= 0.5 + float(config.recent_margin)
    result["ticker_effect_class"] = result.apply(lambda row: _taxonomy(row, config), axis=1)
    axis_weight = result["axis"].map({"AB": 1.0, "TARGET": 0.75, "CD": 0.55}).fillna(0.50)
    matched = pd.to_numeric(result.get("selection_mean_matched_concordance"), errors="coerce").fillna(0.5)
    time_support = (
        result["confirmation_direction_support"].fillna(False).astype(float)
        + result["recent_direction_support"].fillna(False).astype(float)
    ) / 2.0
    result["ticker_driver_score"] = axis_weight * (
        5.0 * np.maximum(np.abs(result["posterior_signed_effect"]) - 0.01, 0.0)
        + 4.0 * np.abs(result["ticker_specific_delta"].fillna(0.0))
        + 0.5 * np.maximum(matched - 0.5, 0.0)
        + 0.35 * time_support
    ) * np.sqrt(np.maximum(result["ticker_map_reliability"], 0.0))
    result["is_ticker_specific"] = result["ticker_effect_class"].isin(
        ["TICKER_UNIQUE_DRIVER", "TICKER_AMPLIFIED_DRIVER", "TICKER_DIRECTION_REVERSAL"]
    )
    result["is_peer_shared"] = result["ticker_effect_class"].isin(
        ["INDUSTRY_SHARED_DRIVER", "BUCKET_SHARED_DRIVER", "UNIVERSAL_DRIVER"]
    )
    result = result.sort_values(
        ["ticker", "axis", "ticker_driver_score", "node_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    reference_columns = [
        *reference_keys,
        "ticker",
        "name",
        "market",
        "bucket",
        "industry",
        "ticker_signed_effect",
        "global_loo_effect",
        "market_loo_effect",
        "bucket_loo_effect",
        "industry_loo_effect",
        "peer_prior_signed_effect",
        "posterior_signed_effect",
        "ticker_specific_delta",
        "ticker_map_reliability",
        "ticker_effect_class",
    ]
    references = result[[column for column in reference_columns if column in result.columns]].copy()
    return result, references


def build_ticker_driver_profiles(
    effect_map: pd.DataFrame,
    *,
    per_axis_count: int = 12,
    unique_count: int = 16,
    shared_count: int = 16,
    reversal_count: int = 8,
) -> tuple[dict[str, Any], pd.DataFrame]:
    profiles: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for ticker, part in effect_map.groupby("ticker", sort=True):
        ticker_profiles: dict[str, list[str]] = {}
        for axis, name in [("TARGET", "SURGE_ASSOCIATION"), ("AB", "PRECISION_SEPARATOR"), ("CD", "MISSED_POSITIVE_RECOVERY")]:
            nodes = (
                part.loc[part["axis"].eq(axis)]
                .sort_values(["ticker_driver_score", "node_id"], ascending=[False, True], kind="mergesort")
                .head(int(per_axis_count))["node_id"].astype(str).tolist()
            )
            ticker_profiles[name] = nodes
        ticker_profiles["TICKER_SPECIFIC"] = (
            part.loc[part["is_ticker_specific"]]
            .sort_values(["ticker_driver_score", "node_id"], ascending=[False, True], kind="mergesort")
            .head(int(unique_count))["node_id"].astype(str).tolist()
        )
        ticker_profiles["PEER_SHARED"] = (
            part.loc[part["is_peer_shared"]]
            .sort_values(["ticker_driver_score", "node_id"], ascending=[False, True], kind="mergesort")
            .head(int(shared_count))["node_id"].astype(str).tolist()
        )
        ticker_profiles["DIRECTION_REVERSAL"] = (
            part.loc[part["ticker_effect_class"].eq("TICKER_DIRECTION_REVERSAL")]
            .sort_values(["ticker_driver_score", "node_id"], ascending=[False, True], kind="mergesort")
            .head(int(reversal_count))["node_id"].astype(str).tolist()
        )
        ticker_profiles["HIERARCHICAL_COMBINED"] = list(
            dict.fromkeys(
                ticker_profiles["PRECISION_SEPARATOR"]
                + ticker_profiles["TICKER_SPECIFIC"]
                + ticker_profiles["SURGE_ASSOCIATION"]
                + ticker_profiles["PEER_SHARED"]
                + ticker_profiles["MISSED_POSITIVE_RECOVERY"]
            )
        )
        profiles[str(ticker)] = ticker_profiles
        for profile_name, nodes in ticker_profiles.items():
            for rank, node_id in enumerate(nodes, start=1):
                rows.append(
                    {
                        "ticker": str(ticker),
                        "profile": profile_name,
                        "rank": int(rank),
                        "node_id": str(node_id),
                    }
                )
    return profiles, pd.DataFrame(rows)


def build_ticker_similarity_map(
    effect_map: pd.DataFrame,
    config: HierarchyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if effect_map.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    useful = effect_map.loc[effect_map["axis"].isin(["AB", "TARGET", "CD"])].copy()
    useful["absolute_effect"] = np.abs(pd.to_numeric(useful["posterior_signed_effect"], errors="coerce"))
    top_nodes = (
        useful.groupby(["axis", "node_id"], as_index=False)["ticker_driver_score"].mean()
        .sort_values(["ticker_driver_score", "axis", "node_id"], ascending=[False, True, True], kind="mergesort")
        .head(int(config.similarity_feature_count))
    )
    useful = useful.merge(top_nodes[["axis", "node_id"]], on=["axis", "node_id"], how="inner")
    useful["signature_node"] = useful["axis"].astype(str) + "::" + useful["node_id"].astype(str)
    pivot = useful.pivot_table(
        index="ticker",
        columns="signature_node",
        values="posterior_signed_effect",
        aggfunc="max",
    )
    tickers = pivot.index.astype(str).tolist()
    matrix = np.eye(len(tickers), dtype=float)
    overlap = np.zeros((len(tickers), len(tickers)), dtype=int)
    agreement = np.eye(len(tickers), dtype=float)
    values = pivot.to_numpy(dtype=float)
    edge_rows: list[dict[str, Any]] = []
    for left in range(len(tickers)):
        for right in range(left + 1, len(tickers)):
            mask = np.isfinite(values[left]) & np.isfinite(values[right])
            common = int(mask.sum())
            overlap[left, right] = overlap[right, left] = common
            if common < int(config.similarity_min_common):
                similarity = math.nan
                direction_agreement = math.nan
            else:
                x = pd.Series(values[left, mask]).rank(method="average").to_numpy(dtype=float)
                y = pd.Series(values[right, mask]).rank(method="average").to_numpy(dtype=float)
                if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
                    similarity = 0.0
                else:
                    similarity = float(np.corrcoef(x, y)[0, 1])
                nonzero = (np.abs(values[left, mask]) > 1e-12) & (np.abs(values[right, mask]) > 1e-12)
                direction_agreement = (
                    float(np.mean(np.sign(values[left, mask][nonzero]) == np.sign(values[right, mask][nonzero])))
                    if nonzero.any()
                    else math.nan
                )
            matrix[left, right] = matrix[right, left] = similarity if math.isfinite(similarity) else 0.0
            agreement[left, right] = agreement[right, left] = direction_agreement if math.isfinite(direction_agreement) else 0.0
            edge_rows.append(
                {
                    "ticker_a": tickers[left],
                    "ticker_b": tickers[right],
                    "map_spearman": similarity,
                    "direction_agreement": direction_agreement,
                    "common_nodes": common,
                }
            )
    all_edges = pd.DataFrame(edge_rows)
    selected_edges_parts: list[pd.DataFrame] = []
    if not all_edges.empty:
        for ticker in tickers:
            candidates = all_edges.loc[
                all_edges["ticker_a"].eq(ticker) | all_edges["ticker_b"].eq(ticker)
            ].copy()
            candidates = candidates.loc[
                pd.to_numeric(candidates["map_spearman"], errors="coerce") >= float(config.similarity_edge_threshold)
            ]
            selected_edges_parts.append(
                candidates.sort_values(
                    ["map_spearman", "common_nodes", "ticker_a", "ticker_b"],
                    ascending=[False, False, True, True],
                    kind="mergesort",
                ).head(int(config.similarity_top_k))
            )
    selected_edges = (
        pd.concat(selected_edges_parts, ignore_index=True).drop_duplicates(["ticker_a", "ticker_b"])
        if selected_edges_parts
        else pd.DataFrame(columns=all_edges.columns)
    )
    if len(tickers) == 1:
        labels = np.ones(1, dtype=int)
    else:
        distance = np.clip((1.0 - np.clip(matrix, -1.0, 1.0)) / 2.0, 0.0, 1.0)
        np.fill_diagonal(distance, 0.0)
        tree = linkage(squareform(distance, checks=False), method="average")
        labels = fcluster(tree, t=(1.0 - float(config.similarity_cluster_threshold)) / 2.0, criterion="distance")
    clusters = pd.DataFrame({"ticker": tickers, "ticker_map_cluster": labels.astype(int)})
    cluster_sizes = clusters.groupby("ticker_map_cluster")["ticker"].size().rename("cluster_size")
    clusters = clusters.merge(cluster_sizes, on="ticker_map_cluster", how="left")
    matrix_frame = pd.DataFrame(matrix, index=tickers, columns=tickers)
    overlap_frame = pd.DataFrame(overlap, index=tickers, columns=tickers)
    return selected_edges, clusters, matrix_frame, overlap_frame


def serialize_profiles(path: Path, profiles: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profiles, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def save_ticker_map_visualizations(
    effect_map: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    output_dir: Path,
    *,
    per_ticker_top_n: int = 20,
    heatmap_node_count: int = 40,
) -> list[str]:
    """Render deterministic diagnostic figures for the ticker-wise maps.

    These figures are descriptive. They do not alter profile membership or model
    selection and therefore cannot leak holdout information back into the map.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    generated: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    if not similarity_matrix.empty:
        figure, axis = plt.subplots(figsize=(max(7.0, len(similarity_matrix) * 0.28), max(6.0, len(similarity_matrix) * 0.25)))
        image = axis.imshow(similarity_matrix.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, aspect="auto")
        tickers = similarity_matrix.index.astype(str).tolist()
        axis.set_xticks(np.arange(len(tickers)))
        axis.set_yticks(np.arange(len(tickers)))
        axis.set_xticklabels(tickers, rotation=90, fontsize=7)
        axis.set_yticklabels(tickers, fontsize=7)
        axis.set_title("Ticker map similarity")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        path = output_dir / "ticker_similarity_heatmap.png"
        figure.savefig(path, dpi=170, bbox_inches="tight")
        plt.close(figure)
        generated.append(str(path.name))

    if not effect_map.empty:
        top_nodes = (
            effect_map.groupby(["axis", "node_id"], as_index=False)["ticker_driver_score"]
            .mean()
            .sort_values(["ticker_driver_score", "axis", "node_id"], ascending=[False, True, True], kind="mergesort")
            .head(int(heatmap_node_count))
        )
        heat = effect_map.merge(top_nodes[["axis", "node_id"]], on=["axis", "node_id"], how="inner")
        heat["signature"] = heat["axis"].astype(str) + "::" + heat["node_id"].astype(str)
        pivot = heat.pivot_table(index="ticker", columns="signature", values="posterior_signed_effect", aggfunc="max")
        if not pivot.empty:
            figure, axis = plt.subplots(
                figsize=(max(10.0, len(pivot.columns) * 0.30), max(5.5, len(pivot.index) * 0.27))
            )
            values = pivot.to_numpy(dtype=float)
            finite = np.abs(values[np.isfinite(values)])
            bound = float(np.quantile(finite, 0.98)) if finite.size else 0.1
            bound = max(bound, 1e-6)
            image = axis.imshow(values, vmin=-bound, vmax=bound, aspect="auto")
            axis.set_xticks(np.arange(len(pivot.columns)))
            axis.set_yticks(np.arange(len(pivot.index)))
            axis.set_xticklabels(pivot.columns.astype(str), rotation=90, fontsize=6)
            axis.set_yticklabels(pivot.index.astype(str), fontsize=7)
            axis.set_title("Ticker-specific posterior driver effects")
            figure.colorbar(image, ax=axis, fraction=0.025, pad=0.02)
            figure.tight_layout()
            path = output_dir / "ticker_driver_effect_heatmap.png"
            figure.savefig(path, dpi=170, bbox_inches="tight")
            plt.close(figure)
            generated.append(str(path.name))

        per_ticker_root = output_dir / "per_ticker"
        for ticker, part in effect_map.groupby("ticker", sort=True):
            selected = (
                part.loc[np.isfinite(pd.to_numeric(part["posterior_signed_effect"], errors="coerce"))]
                .sort_values(["ticker_driver_score", "axis", "node_id"], ascending=[False, True, True], kind="mergesort")
                .head(int(per_ticker_top_n))
                .copy()
            )
            if selected.empty:
                continue
            selected = selected.iloc[::-1]
            labels = selected["axis"].astype(str) + "::" + selected["node_id"].astype(str)
            values = pd.to_numeric(selected["posterior_signed_effect"], errors="coerce").to_numpy(dtype=float)
            figure, axis = plt.subplots(figsize=(9.5, max(5.0, len(selected) * 0.31)))
            axis.barh(np.arange(len(selected)), values)
            axis.axvline(0.0, linewidth=0.8)
            axis.set_yticks(np.arange(len(selected)))
            axis.set_yticklabels(labels, fontsize=7)
            axis.set_xlabel("Posterior signed AUC edge")
            axis.set_title(f"Ticker {ticker}: top hierarchical drivers")
            figure.tight_layout()
            ticker_dir = per_ticker_root / str(ticker)
            ticker_dir.mkdir(parents=True, exist_ok=True)
            path = ticker_dir / "ticker_driver_profile.png"
            figure.savefig(path, dpi=170, bbox_inches="tight")
            plt.close(figure)
            generated.append(str(path.relative_to(output_dir)))
    return generated
