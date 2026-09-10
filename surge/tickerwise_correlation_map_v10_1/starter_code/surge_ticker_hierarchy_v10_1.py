from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

SCHEMA_VERSION = "crashwatch_surge_tickerwise_hierarchy_v10_1"


@dataclass(frozen=True)
class HierarchyLevelAudit:
    level: str
    valid: bool
    reason: str
    ticker_count: int
    non_unknown_count: int
    unique_non_unknown: int
    unknown_ratio: float
    multi_member_coverage: float
    median_group_size: float
    name_identity_ratio: float
    requested_weight: float
    effective_weight: float = 0.0


@dataclass(frozen=True)
class RobustHierarchyConfig:
    prior_strength_grid: tuple[float, ...] = (20.0, 40.0, 80.0, 120.0)
    minimum_peer_tickers: int = 2
    minimum_reliability: float = 0.08
    weak_effect: float = 0.0125
    strong_effect: float = 0.030
    minimum_specific_z: float = 1.25
    robust_specific_z: float = 1.64
    minimum_sensitivity_sign_ratio: float = 0.75
    minimum_sensitivity_class_ratio: float = 0.50
    requested_industry_weight: float = 0.35
    requested_bucket_weight: float = 0.35
    requested_market_weight: float = 0.20
    requested_global_weight: float = 0.10
    industry_max_unknown_ratio: float = 0.35
    industry_max_unique_ratio: float = 0.70
    hierarchy_min_multi_member_coverage: float = 0.50
    precision_min_selection_auc: float = 0.57
    precision_min_selection_min_auc: float = 0.52
    precision_min_direction_consistency: float = 0.80
    precision_min_selection_folds: int = 2
    precision_min_effective_n: float = 12.0
    precision_min_matched_concordance: float = 0.54
    precision_min_specific_z: float = 1.25
    precision_confirm_auc: float = 0.53
    precision_recent_auc: float = 0.53
    similarity_feature_count: int = 180
    similarity_min_node_coverage: float = 0.75
    similarity_min_reliability: float = 0.05
    similarity_edge_threshold: float = 0.30
    similarity_top_k: int = 5
    similarity_cluster_threshold: float = 0.25


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if math.isfinite(number) else float(default)


def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "UNKNOWN"
    text = str(value).strip()
    return text if text else "UNKNOWN"


def _sign(value: float, tolerance: float = 1e-12) -> int:
    if not math.isfinite(value) or abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def normalize_map_frames(summary: pd.DataFrame, by_fold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_out = summary.copy()
    fold_out = by_fold.copy()
    for frame in (summary_out, fold_out):
        if "transformed_feature" not in frame.columns:
            frame["transformed_feature"] = np.nan
        if "transform" not in frame.columns:
            frame["transform"] = "raw"
        frame["node_id"] = frame["transformed_feature"].where(
            frame["transformed_feature"].notna(), frame["feature"]
        ).astype(str)
        frame["source_feature"] = frame["feature"].astype(str)
        for column in ["ticker", "axis", "transform"]:
            if column in frame.columns:
                frame[column] = frame[column].astype(str)
    return summary_out, fold_out


def audit_hierarchy_levels(
    ticker_metadata: pd.DataFrame,
    config: RobustHierarchyConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Audit hierarchy labels before they are allowed to shrink ticker effects.

    V10 accidentally treated a company/legal-name-like column as industry for many
    tickers.  This function rejects levels with too many UNKNOWNs, near one-to-one
    identity with ticker names, or too few tickers in multi-member groups.
    """
    metadata = ticker_metadata.copy()
    metadata["ticker"] = metadata["ticker"].astype(str)
    if "name" not in metadata.columns:
        metadata["name"] = "UNKNOWN"
    requested = {
        "industry": float(config.requested_industry_weight),
        "bucket": float(config.requested_bucket_weight),
        "market": float(config.requested_market_weight),
        "global": float(config.requested_global_weight),
    }
    rows: list[HierarchyLevelAudit] = []
    ticker_count = int(metadata["ticker"].nunique())
    for level in ["industry", "bucket", "market"]:
        if level not in metadata.columns:
            rows.append(
                HierarchyLevelAudit(
                    level, False, "MISSING_COLUMN", ticker_count, 0, 0, 1.0, 0.0, 0.0, 0.0, requested[level]
                )
            )
            continue
        values = metadata[level].map(_safe_str)
        names = metadata["name"].map(_safe_str)
        unknown = values.str.upper().isin({"UNKNOWN", "NAN", "NONE", "NULL", ""})
        non_unknown = values.loc[~unknown]
        group_sizes = non_unknown.value_counts()
        non_unknown_count = int((~unknown).sum())
        unique_non_unknown = int(non_unknown.nunique())
        unknown_ratio = float(unknown.mean()) if len(values) else 1.0
        if non_unknown_count:
            multi_member_values = set(group_sizes.loc[group_sizes >= 2].index.astype(str))
            multi_member_coverage = float(non_unknown.astype(str).isin(multi_member_values).mean())
            median_group_size = float(group_sizes.median()) if len(group_sizes) else 0.0
            name_identity_ratio = float((non_unknown.reset_index(drop=True) == names.loc[~unknown].reset_index(drop=True)).mean())
        else:
            multi_member_coverage = 0.0
            median_group_size = 0.0
            name_identity_ratio = 0.0
        unique_ratio = unique_non_unknown / max(non_unknown_count, 1)
        reasons: list[str] = []
        if level == "industry" and unknown_ratio > float(config.industry_max_unknown_ratio):
            reasons.append("TOO_MANY_UNKNOWN")
        if level == "industry" and unique_ratio > float(config.industry_max_unique_ratio):
            reasons.append("TOO_CLOSE_TO_TICKER_IDENTITY")
        if level == "industry" and name_identity_ratio > 0.50:
            reasons.append("COMPANY_NAME_LIKE")
        if multi_member_coverage < float(config.hierarchy_min_multi_member_coverage):
            reasons.append("INSUFFICIENT_MULTI_MEMBER_COVERAGE")
        if unique_non_unknown < 2:
            reasons.append("INSUFFICIENT_GROUPS")
        valid = not reasons
        rows.append(
            HierarchyLevelAudit(
                level=level,
                valid=valid,
                reason="OK" if valid else "+".join(reasons),
                ticker_count=ticker_count,
                non_unknown_count=non_unknown_count,
                unique_non_unknown=unique_non_unknown,
                unknown_ratio=unknown_ratio,
                multi_member_coverage=multi_member_coverage,
                median_group_size=median_group_size,
                name_identity_ratio=name_identity_ratio,
                requested_weight=requested[level],
            )
        )
    rows.append(
        HierarchyLevelAudit(
            level="global",
            valid=ticker_count >= 2,
            reason="OK" if ticker_count >= 2 else "INSUFFICIENT_TICKERS",
            ticker_count=ticker_count,
            non_unknown_count=ticker_count,
            unique_non_unknown=1,
            unknown_ratio=0.0,
            multi_member_coverage=1.0,
            median_group_size=float(ticker_count),
            name_identity_ratio=0.0,
            requested_weight=requested["global"],
        )
    )
    audit = pd.DataFrame([asdict(row) for row in rows])
    active = audit.loc[audit["valid"] & (audit["requested_weight"] > 0)].copy()
    denominator = float(active["requested_weight"].sum())
    effective: dict[str, float] = {level: 0.0 for level in requested}
    if denominator > 0:
        for row in active.itertuples(index=False):
            effective[str(row.level)] = float(row.requested_weight) / denominator
    audit["effective_weight"] = audit["level"].map(effective).fillna(0.0)
    return audit, effective


def _selection_sample_summary(by_fold: pd.DataFrame, selection_fold_ids: Sequence[int]) -> pd.DataFrame:
    selection = by_fold.loc[
        pd.to_numeric(by_fold["fold_id"], errors="coerce").isin({int(v) for v in selection_fold_ids})
    ].copy()
    if selection.empty:
        return pd.DataFrame()
    for column in ["positive_n", "negative_n", "valid_rows"]:
        selection[column] = pd.to_numeric(selection.get(column), errors="coerce").fillna(0.0)
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
    for metric in [
        "matched_concordance", "matched_pair_count", "positive_coverage", "negative_coverage",
        "smd", "ks", "wasserstein_iqr", "js_divergence", "missing_rate_difference",
    ]:
        if metric in selection.columns:
            selection[metric] = pd.to_numeric(selection[metric], errors="coerce")
            aggregation[f"selection_mean_{metric}"] = (metric, "mean")
    return selection.groupby(
        ["ticker", "axis", "node_id", "source_feature", "transform"], as_index=False
    ).agg(**aggregation)


def auc_standard_error(auc: float, positive_n: float, negative_n: float) -> float:
    """Hanley-McNeil style standard error for an AUC estimate."""
    a = float(auc)
    n1 = float(positive_n)
    n0 = float(negative_n)
    if not (math.isfinite(a) and 0.0 <= a <= 1.0 and n1 >= 2 and n0 >= 2):
        return math.nan
    q1 = a / max(2.0 - a, 1e-12)
    q2 = 2.0 * a * a / max(1.0 + a, 1e-12)
    variance = (
        a * (1.0 - a)
        + (n1 - 1.0) * (q1 - a * a)
        + (n0 - 1.0) * (q2 - a * a)
    ) / max(n1 * n0, 1e-12)
    return math.sqrt(max(variance, 0.0))


def _weighted_leave_one_out_stats(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    effect_column: str,
    weight_column: str,
    prefix: str,
) -> pd.DataFrame:
    keys = list(group_columns)
    work = frame[keys + ["ticker", effect_column, weight_column]].copy()
    effect = pd.to_numeric(work[effect_column], errors="coerce").to_numpy(dtype=float)
    weight = pd.to_numeric(work[weight_column], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(dtype=float)
    finite = np.isfinite(effect) & (weight > 0)
    work["_w"] = np.where(finite, weight, 0.0)
    work["_wx"] = np.where(finite, weight * effect, 0.0)
    work["_wx2"] = np.where(finite, weight * effect * effect, 0.0)
    work["_valid"] = finite.astype(np.int8)
    totals = work.groupby(keys, as_index=False).agg(
        _tw=("_w", "sum"), _twx=("_wx", "sum"), _twx2=("_wx2", "sum"), _tn=("_valid", "sum")
    )
    merged = work.merge(totals, on=keys, how="left", validate="many_to_one")
    own_w = merged["_w"].to_numpy(dtype=float)
    own_wx = merged["_wx"].to_numpy(dtype=float)
    own_wx2 = merged["_wx2"].to_numpy(dtype=float)
    denom = merged["_tw"].to_numpy(dtype=float) - own_w
    numerator = merged["_twx"].to_numpy(dtype=float) - own_wx
    numerator2 = merged["_twx2"].to_numpy(dtype=float) - own_wx2
    peers = merged["_tn"].to_numpy(dtype=float) - merged["_valid"].to_numpy(dtype=float)
    mean = np.full(len(merged), np.nan, dtype=float)
    np.divide(numerator, denom, out=mean, where=denom > 0)
    second = np.full(len(merged), np.nan, dtype=float)
    np.divide(numerator2, denom, out=second, where=denom > 0)
    variance = np.maximum(second - np.square(mean), 0.0)
    sd = np.sqrt(variance)
    # Effective peer count using weight concentration.
    peer_n = np.maximum(peers, 1.0)
    se = sd / np.sqrt(peer_n)
    result = merged[keys + ["ticker"]].copy()
    result[f"{prefix}_loo_effect"] = mean
    result[f"{prefix}_loo_sd"] = sd
    result[f"{prefix}_loo_se"] = se
    result[f"{prefix}_peer_weight"] = denom
    result[f"{prefix}_peer_tickers"] = peers.astype(int)
    return result


def _combine_prior(row: pd.Series, weights: Mapping[str, float], minimum_peer_tickers: int) -> tuple[float, float, str, int]:
    weighted_sum = 0.0
    total_weight = 0.0
    variance_sum = 0.0
    sources: list[str] = []
    peer_total = 0
    for level in ["industry", "bucket", "market", "global"]:
        configured = float(weights.get(level, 0.0))
        if configured <= 0:
            continue
        effect = _finite(row.get(f"{level}_loo_effect"))
        se = _finite(row.get(f"{level}_loo_se"))
        peers = int(_finite(row.get(f"{level}_peer_tickers"), 0.0))
        if not math.isfinite(effect):
            continue
        if level != "global" and peers < int(minimum_peer_tickers):
            continue
        if level == "global" and peers < 1:
            continue
        weighted_sum += configured * effect
        total_weight += configured
        if math.isfinite(se):
            variance_sum += (configured * se) ** 2
        sources.append(level)
        peer_total += peers
    if total_weight <= 0:
        return math.nan, math.nan, "NONE", 0
    return weighted_sum / total_weight, math.sqrt(variance_sum) / total_weight, "+".join(sources), peer_total


def _prepare_base_effect_frame(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    ticker_metadata: pd.DataFrame,
    selection_fold_ids: Sequence[int],
) -> pd.DataFrame:
    normalized_summary, normalized_fold = normalize_map_frames(summary, by_fold)
    samples = _selection_sample_summary(normalized_fold, selection_fold_ids)
    result = normalized_summary.merge(
        samples,
        on=["ticker", "axis", "node_id", "source_feature", "transform"],
        how="left",
        validate="one_to_one",
    )
    metadata_columns = [c for c in ["ticker", "name", "market", "bucket", "industry"] if c in ticker_metadata.columns]
    metadata = ticker_metadata[metadata_columns].drop_duplicates("ticker", keep="last").copy()
    result = result.merge(metadata, on="ticker", how="left", validate="many_to_one")
    for column in ["name", "market", "bucket", "industry"]:
        if column not in result.columns:
            result[column] = "UNKNOWN"
        result[column] = result[column].map(_safe_str)
    # IMPORTANT V10.1 FIX: use the selection fixed AUC, not the already shrunk
    # selection_mean_shrunk_auc.  V10 shrank once at the AUC-map layer and then
    # again in the hierarchy, which can erase real ticker heterogeneity.
    fixed = pd.to_numeric(result["selection_mean_fixed_auc"], errors="coerce")
    direction = pd.to_numeric(result["selection_direction"], errors="coerce").fillna(0.0)
    result["ticker_signed_effect_raw"] = direction * (fixed - 0.5)
    result["ticker_auc_se"] = [
        auc_standard_error(_finite(a), _finite(p, 0.0), _finite(n, 0.0))
        for a, p, n in zip(
            fixed,
            pd.to_numeric(result["selection_positive_sum"], errors="coerce").fillna(0.0),
            pd.to_numeric(result["selection_negative_sum"], errors="coerce").fillna(0.0),
        )
    ]
    effective_n = pd.to_numeric(result["selection_effective_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
    consistency = pd.to_numeric(result["selection_direction_consistency"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    evidence = pd.to_numeric(result.get("selection_evidence_score", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    result["prior_weight"] = effective_n * (0.20 + 0.80 * consistency) * (1.0 + np.minimum(evidence, 2.0))
    result["selection_effective_n_recalc"] = effective_n
    return result


def build_hierarchy_for_strength(
    base: pd.DataFrame,
    selection_fold_ids: Sequence[int],
    prior_strength: float,
    weights: Mapping[str, float],
    config: RobustHierarchyConfig,
) -> pd.DataFrame:
    result = base.copy()
    effective_n = pd.to_numeric(result["selection_effective_n_recalc"], errors="coerce").fillna(0.0).clip(lower=0.0)
    fold_count = pd.to_numeric(result["selection_fold_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    consistency = pd.to_numeric(result["selection_direction_consistency"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    sample_reliability = effective_n / (effective_n + max(float(prior_strength), 1e-12))
    fold_reliability = np.minimum(fold_count / max(len(selection_fold_ids), 1), 1.0)
    result["ticker_map_reliability"] = sample_reliability * np.sqrt(fold_reliability * consistency)
    reference_keys = ["axis", "node_id", "source_feature", "transform"]
    global_ref = _weighted_leave_one_out_stats(result, reference_keys, "ticker_signed_effect_raw", "prior_weight", "global")
    result = result.merge(global_ref, on=reference_keys + ["ticker"], how="left", validate="one_to_one")
    for level, column in [("market", "market"), ("bucket", "bucket"), ("industry", "industry")]:
        ref = _weighted_leave_one_out_stats(
            result, [*reference_keys, column], "ticker_signed_effect_raw", "prior_weight", level
        )
        result = result.merge(ref, on=[*reference_keys, column, "ticker"], how="left", validate="one_to_one")
    priors = result.apply(lambda row: _combine_prior(row, weights, int(config.minimum_peer_tickers)), axis=1)
    result["peer_prior_signed_effect"] = [v[0] for v in priors]
    result["peer_prior_se"] = [v[1] for v in priors]
    result["peer_prior_sources"] = [v[2] for v in priors]
    result["peer_reference_count"] = [v[3] for v in priors]
    raw = pd.to_numeric(result["ticker_signed_effect_raw"], errors="coerce").to_numpy(dtype=float)
    prior = pd.to_numeric(result["peer_prior_signed_effect"], errors="coerce").to_numpy(dtype=float)
    reliability = pd.to_numeric(result["ticker_map_reliability"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    posterior = np.where(np.isfinite(prior), reliability * raw + (1.0 - reliability) * prior, raw)
    result["posterior_signed_effect"] = posterior
    result["ticker_specific_delta_raw"] = raw - prior
    result["ticker_specific_delta_posterior"] = posterior - prior
    raw_se = pd.to_numeric(result["ticker_auc_se"], errors="coerce").to_numpy(dtype=float)
    peer_se = pd.to_numeric(result["peer_prior_se"], errors="coerce").to_numpy(dtype=float)
    denominator = np.sqrt(np.square(raw_se) + np.square(peer_se))
    specific_z = np.full(len(result), np.nan, dtype=float)
    np.divide(raw - prior, denominator, out=specific_z, where=np.isfinite(denominator) & (denominator > 1e-12))
    result["ticker_specific_z"] = specific_z
    result["posterior_oriented_auc"] = 0.5 + np.abs(posterior)
    result["prior_strength"] = float(prior_strength)
    return result


def _stable_ratio(values: Iterable[float], predicate) -> float:
    finite = [float(v) for v in values if math.isfinite(_finite(v))]
    if not finite:
        return 0.0
    return float(np.mean([bool(predicate(v)) for v in finite]))


def aggregate_hierarchy_sensitivity(
    sensitivity_long: pd.DataFrame,
    config: RobustHierarchyConfig,
) -> pd.DataFrame:
    keys = ["ticker", "axis", "node_id", "source_feature", "transform", "name", "market", "bucket", "industry"]
    if sensitivity_long.empty:
        return pd.DataFrame()

    # Every group contains one row per prior strength.  The original V10.1
    # implementation iterated through ~90k tiny DataFrames; expressing the
    # same reductions as vectorized groupby aggregations removes that Python
    # bottleneck without changing the statistical definitions.
    work = sensitivity_long.copy()
    numeric_columns = [
        "prior_strength",
        "ticker_signed_effect_raw",
        "peer_prior_signed_effect",
        "posterior_signed_effect",
        "ticker_specific_delta_raw",
        "ticker_specific_z",
        "ticker_map_reliability",
    ]
    for column in numeric_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work["_specific_z_abs"] = np.abs(work["ticker_specific_z"])
    posterior = work["posterior_signed_effect"].to_numpy(dtype=float)
    delta = work["ticker_specific_delta_raw"].to_numpy(dtype=float)
    posterior_nonzero = np.isfinite(posterior) & (np.abs(posterior) > 1e-12)
    delta_nonzero = np.isfinite(delta) & (np.abs(delta) > 1e-12)
    work["_post_nonzero"] = posterior_nonzero.astype(np.int8)
    work["_post_positive"] = (posterior_nonzero & (posterior > 0)).astype(np.int8)
    work["_post_negative"] = (posterior_nonzero & (posterior < 0)).astype(np.int8)
    work["_delta_nonzero"] = delta_nonzero.astype(np.int8)
    work["_delta_positive"] = (delta_nonzero & (delta > 0)).astype(np.int8)
    work["_delta_negative"] = (delta_nonzero & (delta < 0)).astype(np.int8)
    work["_peer_prior_sources_text"] = work["peer_prior_sources"].astype(str)

    selected_columns = [
        "selection_direction", "selection_fold_count", "selection_mean_fixed_auc", "selection_min_fixed_auc",
        "selection_direction_consistency", "selection_mean_matched_concordance", "selection_effective_n",
        "confirmation_mean_fixed_auc", "recent_mean_fixed_auc", "confirmation_direction_consistency",
        "recent_direction_consistency", "selection_evidence_score",
    ]
    selected_columns = [column for column in selected_columns if column in work.columns]
    aggregations: dict[str, tuple[str, str]] = {
        "sensitivity_strength_count": ("prior_strength", "nunique"),
        "ticker_signed_effect_raw": ("ticker_signed_effect_raw", "median"),
        "peer_prior_signed_effect_median": ("peer_prior_signed_effect", "median"),
        "posterior_signed_effect_median": ("posterior_signed_effect", "median"),
        "posterior_signed_effect_min": ("posterior_signed_effect", "min"),
        "posterior_signed_effect_max": ("posterior_signed_effect", "max"),
        "specific_delta_raw_median": ("ticker_specific_delta_raw", "median"),
        "specific_z_median": ("ticker_specific_z", "median"),
        "specific_z_abs_min": ("_specific_z_abs", "min"),
        "specific_z_abs_max": ("_specific_z_abs", "max"),
        "reliability_median": ("ticker_map_reliability", "median"),
        "_post_nonzero": ("_post_nonzero", "sum"),
        "_post_positive": ("_post_positive", "sum"),
        "_post_negative": ("_post_negative", "sum"),
        "_delta_nonzero": ("_delta_nonzero", "sum"),
        "_delta_positive": ("_delta_positive", "sum"),
        "_delta_negative": ("_delta_negative", "sum"),
        "peer_prior_sources": ("_peer_prior_sources_text", "first"),
    }
    aggregations.update({column: (column, "first") for column in selected_columns})
    out = work.groupby(keys, dropna=False, sort=True, as_index=False).agg(**aggregations)
    out["peer_prior_sources"] = out["peer_prior_sources"].replace("NONE", "")
    post_denom = out.pop("_post_nonzero").to_numpy(dtype=float)
    post_numerator = np.maximum(
        out.pop("_post_positive").to_numpy(dtype=float),
        out.pop("_post_negative").to_numpy(dtype=float),
    )
    delta_denom = out.pop("_delta_nonzero").to_numpy(dtype=float)
    delta_numerator = np.maximum(
        out.pop("_delta_positive").to_numpy(dtype=float),
        out.pop("_delta_negative").to_numpy(dtype=float),
    )
    out["posterior_sign_stability"] = np.divide(
        post_numerator, post_denom, out=np.zeros(len(out), dtype=float), where=post_denom > 0
    )
    out["specific_delta_sign_stability"] = np.divide(
        delta_numerator, delta_denom, out=np.zeros(len(out), dtype=float), where=delta_denom > 0
    )
    ordered = [
        *keys,
        "sensitivity_strength_count",
        "ticker_signed_effect_raw",
        "peer_prior_signed_effect_median",
        "posterior_signed_effect_median",
        "posterior_signed_effect_min",
        "posterior_signed_effect_max",
        "specific_delta_raw_median",
        "specific_z_median",
        "specific_z_abs_min",
        "specific_z_abs_max",
        "reliability_median",
        "posterior_sign_stability",
        "specific_delta_sign_stability",
        "peer_prior_sources",
        *selected_columns,
    ]
    out = out[ordered]
    if out.empty:
        return out
    out["ticker_specific_robust"] = (
        (out["reliability_median"] >= float(config.minimum_reliability))
        & (out["posterior_sign_stability"] >= float(config.minimum_sensitivity_sign_ratio))
        & (out["specific_delta_sign_stability"] >= float(config.minimum_sensitivity_sign_ratio))
        & (np.abs(pd.to_numeric(out["specific_z_median"], errors="coerce")) >= float(config.minimum_specific_z))
        & (np.abs(pd.to_numeric(out["posterior_signed_effect_median"], errors="coerce")) >= float(config.weak_effect))
    )
    out["ticker_specific_strong"] = out["ticker_specific_robust"] & (
        np.abs(pd.to_numeric(out["specific_z_median"], errors="coerce")) >= float(config.robust_specific_z)
    ) & (
        np.abs(pd.to_numeric(out["posterior_signed_effect_median"], errors="coerce")) >= float(config.strong_effect)
    )
    out["direction_reversal_robust"] = (
        out["ticker_specific_robust"]
        & (np.sign(pd.to_numeric(out["posterior_signed_effect_median"], errors="coerce"))
           != np.sign(pd.to_numeric(out["peer_prior_signed_effect_median"], errors="coerce")))
        & (np.abs(pd.to_numeric(out["peer_prior_signed_effect_median"], errors="coerce")) >= float(config.weak_effect))
    )
    out["ticker_specificity_score_v10_1"] = (
        np.abs(pd.to_numeric(out["posterior_signed_effect_median"], errors="coerce")).fillna(0.0) * 2.5
        + np.abs(pd.to_numeric(out["specific_delta_raw_median"], errors="coerce")).fillna(0.0) * 3.5
        + np.minimum(np.abs(pd.to_numeric(out["specific_z_median"], errors="coerce")).fillna(0.0), 4.0) * 0.20
    ) * np.sqrt(pd.to_numeric(out["reliability_median"], errors="coerce").fillna(0.0).clip(lower=0.0))
    return out


def build_precision_separator_map(robust_map: pd.DataFrame, config: RobustHierarchyConfig) -> pd.DataFrame:
    ab = robust_map.loc[robust_map["axis"].astype(str).eq("AB")].copy()
    if ab.empty:
        return ab
    sel_auc = pd.to_numeric(ab["selection_mean_fixed_auc"], errors="coerce")
    sel_min = pd.to_numeric(ab["selection_min_fixed_auc"], errors="coerce")
    sel_cons = pd.to_numeric(ab["selection_direction_consistency"], errors="coerce")
    sel_folds = pd.to_numeric(ab["selection_fold_count"], errors="coerce")
    eff_n = pd.to_numeric(ab.get("selection_effective_n"), errors="coerce")
    matched = pd.to_numeric(ab.get("selection_mean_matched_concordance"), errors="coerce")
    specific_z = np.abs(pd.to_numeric(ab["specific_z_median"], errors="coerce"))
    # Missing matched concordance is not fatal for raw TARGET-style maps, but for
    # AB it is valuable evidence; use 0.5 neutral when unavailable.
    matched_neutral = matched.fillna(0.5)
    ab["precision_separator_selection_candidate"] = (
        (sel_auc >= float(config.precision_min_selection_auc))
        & (sel_min >= float(config.precision_min_selection_min_auc))
        & (sel_cons >= float(config.precision_min_direction_consistency))
        & (sel_folds >= int(config.precision_min_selection_folds))
        & (eff_n >= float(config.precision_min_effective_n))
        & (specific_z >= float(config.precision_min_specific_z))
        & (ab["specific_delta_sign_stability"] >= float(config.minimum_sensitivity_sign_ratio))
    )
    ab["precision_separator_matched_support"] = matched_neutral >= float(config.precision_min_matched_concordance)
    conf = pd.to_numeric(ab.get("confirmation_mean_fixed_auc"), errors="coerce")
    recent = pd.to_numeric(ab.get("recent_mean_fixed_auc"), errors="coerce")
    ab["precision_separator_confirmation_support"] = conf >= float(config.precision_confirm_auc)
    ab["precision_separator_recent_support"] = recent >= float(config.precision_recent_auc)
    ab["precision_separator_confirmed"] = (
        ab["precision_separator_selection_candidate"]
        & ab["precision_separator_matched_support"]
        & ab["precision_separator_confirmation_support"].fillna(False)
    )
    ab["precision_separator_recent_confirmed"] = (
        ab["precision_separator_confirmed"]
        & ab["precision_separator_recent_support"].fillna(False)
    )
    ab["precision_separator_score_v10_1"] = (
        np.maximum(sel_auc - 0.5, 0.0) * 4.0
        + np.maximum(sel_min - 0.5, 0.0) * 2.0
        + np.maximum(matched_neutral - 0.5, 0.0) * 1.5
        + np.minimum(specific_z.fillna(0.0), 4.0) * 0.10
        + ab["ticker_specificity_score_v10_1"].fillna(0.0)
    ) * np.sqrt(pd.to_numeric(ab["reliability_median"], errors="coerce").fillna(0.0).clip(lower=0.0))
    return ab.sort_values(
        ["precision_separator_selection_candidate", "precision_separator_score_v10_1", "ticker", "node_id"],
        ascending=[False, False, True, True], kind="mergesort"
    ).reset_index(drop=True)


def build_fixed_signature_similarity(
    robust_map: pd.DataFrame,
    config: RobustHierarchyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build ticker similarity from the same fixed signature nodes for all tickers.

    V10 selected top nodes globally then required >=20 common finite nodes, but most
    ticker profiles only overlapped on 0-16 nodes.  Here we deliberately choose raw
    nodes with high cross-ticker coverage and use the same ordered signature for
    every ticker.  Missing/low-reliability effects contribute zero after centering.
    """
    if robust_map.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty
    work = robust_map.copy()
    work["posterior"] = pd.to_numeric(work["posterior_signed_effect_median"], errors="coerce")
    work["reliability"] = pd.to_numeric(work["reliability_median"], errors="coerce").fillna(0.0)
    tickers = sorted(work["ticker"].astype(str).unique().tolist())
    ticker_count = max(len(tickers), 1)
    raw = work.loc[work["transform"].astype(str).eq("raw")].copy()
    raw["signature_node"] = raw["axis"].astype(str) + "::" + raw["node_id"].astype(str)
    node_stats = raw.groupby("signature_node", as_index=False).agg(
        finite_tickers=("posterior", lambda s: int(np.isfinite(pd.to_numeric(s, errors="coerce")).sum())),
        reliable_tickers=("reliability", lambda s: int((pd.to_numeric(s, errors="coerce") >= float(config.similarity_min_reliability)).sum())),
        median_abs_effect=("posterior", lambda s: float(np.nanmedian(np.abs(pd.to_numeric(s, errors="coerce"))))),
        median_reliability=("reliability", "median"),
    )
    node_stats["coverage"] = node_stats["finite_tickers"] / ticker_count
    node_stats["reliable_coverage"] = node_stats["reliable_tickers"] / ticker_count
    node_stats["signature_score"] = node_stats["median_abs_effect"].fillna(0.0) * np.sqrt(
        node_stats["median_reliability"].fillna(0.0).clip(lower=0.0)
    )
    eligible = node_stats.loc[node_stats["coverage"] >= float(config.similarity_min_node_coverage)].copy()
    if eligible.empty:
        eligible = node_stats.copy()
    selected = eligible.sort_values(
        ["signature_score", "reliable_coverage", "signature_node"], ascending=[False, False, True], kind="mergesort"
    ).head(int(config.similarity_feature_count)).copy()
    selected_nodes = selected["signature_node"].astype(str).tolist()
    use = raw.loc[raw["signature_node"].isin(selected_nodes)].copy()
    effect_pivot = use.pivot_table(index="ticker", columns="signature_node", values="posterior", aggfunc="mean")
    rel_pivot = use.pivot_table(index="ticker", columns="signature_node", values="reliability", aggfunc="mean")
    effect_pivot = effect_pivot.reindex(index=tickers, columns=selected_nodes)
    rel_pivot = rel_pivot.reindex(index=tickers, columns=selected_nodes).fillna(0.0)
    values = effect_pivot.fillna(0.0).to_numpy(dtype=float)
    reliabilities = rel_pivot.to_numpy(dtype=float)
    similarity = np.eye(len(tickers), dtype=float)
    spearman_matrix = np.eye(len(tickers), dtype=float)
    overlap = np.zeros((len(tickers), len(tickers)), dtype=int)
    edge_rows: list[dict[str, Any]] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            weight = np.sqrt(reliabilities[i] * reliabilities[j])
            active = weight >= float(config.similarity_min_reliability)
            common = int(active.sum())
            overlap[i, j] = overlap[j, i] = common
            if common >= 3:
                x = values[i, active]
                y = values[j, active]
                w = weight[active]
                numerator = float(np.sum(w * x * y))
                denominator = math.sqrt(float(np.sum(w * x * x)) * float(np.sum(w * y * y)))
                cosine = numerator / denominator if denominator > 1e-12 else 0.0
                rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
                ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
                spear = float(np.corrcoef(rx, ry)[0, 1]) if np.std(rx) > 1e-12 and np.std(ry) > 1e-12 else 0.0
                composite = 0.55 * cosine + 0.45 * spear
            else:
                cosine = 0.0
                spear = 0.0
                composite = 0.0
            similarity[i, j] = similarity[j, i] = composite
            spearman_matrix[i, j] = spearman_matrix[j, i] = spear
            edge_rows.append({
                "ticker_a": tickers[i], "ticker_b": tickers[j], "fixed_signature_similarity": composite,
                "weighted_cosine": cosine, "fixed_signature_spearman": spear, "common_reliable_nodes": common,
            })
    all_edges = pd.DataFrame(edge_rows)
    selected_parts: list[pd.DataFrame] = []
    if not all_edges.empty:
        for ticker in tickers:
            part = all_edges.loc[all_edges["ticker_a"].eq(ticker) | all_edges["ticker_b"].eq(ticker)].copy()
            part = part.loc[part["fixed_signature_similarity"] >= float(config.similarity_edge_threshold)]
            selected_parts.append(part.sort_values(
                ["fixed_signature_similarity", "common_reliable_nodes", "ticker_a", "ticker_b"],
                ascending=[False, False, True, True], kind="mergesort"
            ).head(int(config.similarity_top_k)))
    selected_edges = pd.concat(selected_parts, ignore_index=True).drop_duplicates(["ticker_a", "ticker_b"]) if selected_parts else pd.DataFrame(columns=all_edges.columns)
    if len(tickers) <= 1:
        labels = np.ones(len(tickers), dtype=int)
    else:
        distance = np.clip((1.0 - np.clip(similarity, -1.0, 1.0)) / 2.0, 0.0, 1.0)
        np.fill_diagonal(distance, 0.0)
        tree = linkage(squareform(distance, checks=False), method="average")
        labels = fcluster(tree, t=(1.0 - float(config.similarity_cluster_threshold)) / 2.0, criterion="distance")
    clusters = pd.DataFrame({"ticker": tickers, "ticker_map_cluster": labels.astype(int)})
    clusters["cluster_size"] = clusters["ticker_map_cluster"].map(clusters["ticker_map_cluster"].value_counts())
    return (
        selected_edges,
        clusters,
        pd.DataFrame(similarity, index=tickers, columns=tickers),
        pd.DataFrame(overlap, index=tickers, columns=tickers),
        selected,
    )


def build_adaptive_probe_eligibility(
    original: pd.DataFrame,
    *,
    configured_min_rows: int = 60,
    recent_row_fraction: float = 0.85,
    minimum_recent_rows_floor: int = 45,
    minimum_positive: int = 3,
    minimum_negative: int = 20,
) -> pd.DataFrame:
    """Repair V10's fold-7 impossibility without pretending tiny samples are strong.

    If a fold physically contains fewer rows than configured_min_rows for every
    ticker (V10 recent fold had 57 vs min 60), lower only that fold's row threshold
    to max(floor, ceil(fraction * fold_max_rows)).  Positive/negative requirements
    remain explicit and are reported separately.
    """
    out = original.copy()
    fold_max = out.groupby("fold_id")["validation_rows"].max().to_dict()
    required_rows: list[int] = []
    adaptive: list[bool] = []
    eligible: list[bool] = []
    reasons: list[str] = []
    for row in out.itertuples(index=False):
        max_rows = int(fold_max.get(row.fold_id, row.validation_rows))
        if max_rows >= int(configured_min_rows):
            req = int(configured_min_rows)
            was_adaptive = False
        else:
            req = max(int(minimum_recent_rows_floor), int(math.ceil(float(recent_row_fraction) * max_rows)))
            req = min(req, max_rows)
            was_adaptive = True
        pos = int(row.validation_positive)
        neg = int(row.validation_negative)
        n = int(row.validation_rows)
        ok = n >= req and pos >= int(minimum_positive) and neg >= int(minimum_negative)
        reason_parts: list[str] = []
        if n < req:
            reason_parts.append("VALIDATION_ROWS")
        if pos < int(minimum_positive):
            reason_parts.append("VALIDATION_POSITIVE")
        if neg < int(minimum_negative):
            reason_parts.append("VALIDATION_NEGATIVE")
        required_rows.append(req)
        adaptive.append(was_adaptive)
        eligible.append(ok)
        reasons.append("OK" if ok else "+".join(reason_parts))
    out["v10_1_required_validation_rows"] = required_rows
    out["v10_1_adaptive_row_threshold"] = adaptive
    out["v10_1_eligible_model"] = eligible
    out["v10_1_reason"] = reasons
    return out


def build_profiles_v10_1(
    robust_map: pd.DataFrame,
    precision_map: pd.DataFrame,
    *,
    per_axis_count: int = 12,
    specific_count: int = 16,
    precision_count: int = 16,
) -> tuple[dict[str, Any], pd.DataFrame]:
    profiles: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    tickers = sorted(robust_map["ticker"].astype(str).unique().tolist())
    for ticker in tickers:
        part = robust_map.loc[robust_map["ticker"].astype(str).eq(ticker)].copy()
        ppart = precision_map.loc[precision_map["ticker"].astype(str).eq(ticker)].copy() if not precision_map.empty else pd.DataFrame()
        data: dict[str, list[str]] = {}
        data["SURGE_ASSOCIATION_ROBUST"] = (
            part.loc[part["axis"].eq("TARGET")]
            .sort_values(["ticker_specificity_score_v10_1", "node_id"], ascending=[False, True], kind="mergesort")
            .head(int(per_axis_count))["node_id"].astype(str).tolist()
        )
        data["TICKER_SPECIFIC_ROBUST"] = (
            part.loc[part["ticker_specific_robust"]]
            .sort_values(["ticker_specificity_score_v10_1", "node_id"], ascending=[False, True], kind="mergesort")
            .head(int(specific_count))["node_id"].astype(str).tolist()
        )
        data["DIRECTION_REVERSAL_ROBUST"] = (
            part.loc[part["direction_reversal_robust"]]
            .sort_values(["ticker_specificity_score_v10_1", "node_id"], ascending=[False, True], kind="mergesort")
            .head(int(specific_count))["node_id"].astype(str).tolist()
        )
        if not ppart.empty:
            data["PRECISION_SEPARATOR_SELECTION"] = (
                ppart.loc[ppart["precision_separator_selection_candidate"]]
                .sort_values(["precision_separator_score_v10_1", "node_id"], ascending=[False, True], kind="mergesort")
                .head(int(precision_count))["node_id"].astype(str).tolist()
            )
            data["PRECISION_SEPARATOR_CONFIRMED"] = (
                ppart.loc[ppart["precision_separator_confirmed"]]
                .sort_values(["precision_separator_score_v10_1", "node_id"], ascending=[False, True], kind="mergesort")
                .head(int(precision_count))["node_id"].astype(str).tolist()
            )
        else:
            data["PRECISION_SEPARATOR_SELECTION"] = []
            data["PRECISION_SEPARATOR_CONFIRMED"] = []
        data["HIERARCHICAL_COMBINED_V10_1"] = list(dict.fromkeys(
            data["PRECISION_SEPARATOR_SELECTION"]
            + data["TICKER_SPECIFIC_ROBUST"]
            + data["SURGE_ASSOCIATION_ROBUST"]
            + data["DIRECTION_REVERSAL_ROBUST"]
        ))
        profiles[ticker] = data
        for profile_name, nodes in data.items():
            for rank, node in enumerate(nodes, start=1):
                rows.append({"ticker": ticker, "profile": profile_name, "rank": rank, "node_id": node})
    return profiles, pd.DataFrame(rows)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
