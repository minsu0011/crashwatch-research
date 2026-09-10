from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import norm, spearmanr

SCHEMA_VERSION = "crashwatch_surge_tickerwise_hierarchy_v10_2"


@dataclass(frozen=True)
class ConservativeHierarchyConfig:
    prior_strength_grid: tuple[float, ...] = (20.0, 40.0, 80.0, 120.0)
    bucket_weight: float = 0.55
    market_outer_weight: float = 0.30
    global_outer_weight: float = 0.15
    minimum_peer_effective_n: float = 1.5
    minimum_reliability: float = 0.08
    minimum_specific_z: float = 1.25
    robust_specific_z: float = 1.64
    specific_fdr_ticker_axis: float = 0.10
    specific_fdr_global_axis: float = 0.20
    minimum_sensitivity_sign_ratio: float = 0.75
    weak_effect: float = 0.0125
    strong_effect: float = 0.030
    precision_min_selection_auc: float = 0.57
    precision_min_selection_min_auc: float = 0.52
    precision_min_direction_consistency: float = 0.80
    precision_min_selection_folds: int = 2
    precision_min_effective_n: float = 12.0
    precision_min_matched_concordance: float = 0.54
    precision_min_specific_z: float = 1.25
    precision_confirm_auc: float = 0.53
    precision_confirm_min_auc: float = 0.50
    precision_confirm_direction_consistency: float = 1.0
    precision_recent_auc: float = 0.53
    precision_recent_direction_consistency: float = 1.0
    similarity_feature_count: int = 180
    similarity_min_coverage: float = 0.70
    similarity_min_common: int = 20
    similarity_edge_threshold: float = 0.25
    similarity_top_k: int = 5
    similarity_cluster_threshold: float = 0.20


def _finite(value: Any, default: float = math.nan) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if math.isfinite(x) else float(default)


def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "UNKNOWN"
    text = str(value).strip()
    return text or "UNKNOWN"


def normalize_map_frames(summary: pd.DataFrame, by_fold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = summary.copy()
    f = by_fold.copy()
    for frame in (s, f):
        if "transformed_feature" not in frame.columns:
            frame["transformed_feature"] = np.nan
        if "transform" not in frame.columns:
            frame["transform"] = "raw"
        frame["node_id"] = frame["transformed_feature"].where(frame["transformed_feature"].notna(), frame["feature"]).astype(str)
        frame["source_feature"] = frame["feature"].astype(str)
        for col in ["ticker", "axis", "transform"]:
            if col in frame.columns:
                frame[col] = frame[col].astype(str)
    return s, f


def auc_standard_error(auc: float, positive_n: float, negative_n: float) -> float:
    auc = _finite(auc)
    n1 = _finite(positive_n, 0.0)
    n0 = _finite(negative_n, 0.0)
    if not math.isfinite(auc) or n1 <= 1 or n0 <= 1:
        return math.nan
    q1 = auc / max(2.0 - auc, 1e-12)
    q2 = 2.0 * auc * auc / max(1.0 + auc, 1e-12)
    variance = (auc * (1.0 - auc) + (n1 - 1.0) * (q1 - auc * auc) + (n0 - 1.0) * (q2 - auc * auc)) / max(n1 * n0, 1e-12)
    return math.sqrt(max(variance, 0.0))


def _selection_sample_summary(by_fold: pd.DataFrame, selection_folds: Sequence[int]) -> pd.DataFrame:
    part = by_fold.loc[pd.to_numeric(by_fold["fold_id"], errors="coerce").isin({int(x) for x in selection_folds})].copy()
    if part.empty:
        return pd.DataFrame()
    for col in ["positive_n", "negative_n", "valid_rows"]:
        part[col] = pd.to_numeric(part.get(col), errors="coerce").fillna(0.0)
    p = part["positive_n"].to_numpy(float)
    n = part["negative_n"].to_numpy(float)
    part["effective_n"] = np.where((p > 0) & (n > 0), 2.0 * p * n / np.maximum(p + n, 1e-12), 0.0)
    agg: dict[str, tuple[str, str]] = {
        "selection_positive_sum": ("positive_n", "sum"),
        "selection_negative_sum": ("negative_n", "sum"),
        "selection_valid_sum": ("valid_rows", "sum"),
        "selection_effective_n": ("effective_n", "sum"),
        "selection_observed_folds": ("fold_id", "nunique"),
    }
    for metric in ["matched_concordance", "matched_pair_count", "positive_coverage", "negative_coverage"]:
        if metric in part.columns:
            part[metric] = pd.to_numeric(part[metric], errors="coerce")
            agg[f"selection_mean_{metric}"] = (metric, "mean")
    keys = ["ticker", "axis", "node_id", "source_feature", "transform"]
    return part.groupby(keys, as_index=False).agg(**agg)


def prepare_base_effect_frame(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    ticker_metadata: pd.DataFrame,
    selection_folds: Sequence[int],
) -> pd.DataFrame:
    summary, by_fold = normalize_map_frames(summary, by_fold)
    samples = _selection_sample_summary(by_fold, selection_folds)
    keys = ["ticker", "axis", "node_id", "source_feature", "transform"]
    out = summary.merge(samples, on=keys, how="left", validate="one_to_one")
    meta_cols = [c for c in ["ticker", "name", "market", "bucket", "industry"] if c in ticker_metadata.columns]
    meta = ticker_metadata[meta_cols].drop_duplicates("ticker", keep="last")
    out = out.merge(meta, on="ticker", how="left", validate="many_to_one")
    for c in ["name", "market", "bucket", "industry"]:
        if c not in out.columns:
            out[c] = "UNKNOWN"
        out[c] = out[c].map(_safe_str)
    fixed = pd.to_numeric(out["selection_mean_fixed_auc"], errors="coerce")
    direction = pd.to_numeric(out["selection_direction"], errors="coerce").fillna(0.0)
    out["ticker_signed_effect_raw"] = direction * (fixed - 0.5)
    out["ticker_auc_se"] = [
        auc_standard_error(a, p, n)
        for a, p, n in zip(fixed, out["selection_positive_sum"], out["selection_negative_sum"])
    ]
    eff = pd.to_numeric(out["selection_effective_n"], errors="coerce").fillna(0.0).clip(lower=0.0)
    cons = pd.to_numeric(out["selection_direction_consistency"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    evidence = pd.to_numeric(out.get("selection_evidence_score", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    out["prior_weight"] = eff * (0.20 + 0.80 * cons) * (1.0 + np.minimum(evidence, 2.0))
    out["selection_effective_n_recalc"] = eff
    return out


def _moment_columns(frame: pd.DataFrame, effect_col: str, weight_col: str) -> pd.DataFrame:
    out = frame.copy()
    x = pd.to_numeric(out[effect_col], errors="coerce").to_numpy(float)
    w = pd.to_numeric(out[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0).to_numpy(float)
    valid = np.isfinite(x) & (w > 0)
    out["_w"] = np.where(valid, w, 0.0)
    out["_w2"] = np.where(valid, w * w, 0.0)
    out["_wx"] = np.where(valid, w * x, 0.0)
    out["_wx2"] = np.where(valid, w * x * x, 0.0)
    out["_n"] = valid.astype(np.int16)
    return out


def _aggregate_moments(frame: pd.DataFrame, keys: Sequence[str], prefix: str) -> pd.DataFrame:
    return frame.groupby(list(keys), as_index=False).agg(
        **{
            f"{prefix}_w": ("_w", "sum"),
            f"{prefix}_w2": ("_w2", "sum"),
            f"{prefix}_wx": ("_wx", "sum"),
            f"{prefix}_wx2": ("_wx2", "sum"),
            f"{prefix}_n": ("_n", "sum"),
        }
    )


def _ring_stats(w: np.ndarray, w2: np.ndarray, wx: np.ndarray, wx2: np.ndarray, n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.full(len(w), np.nan, float)
    np.divide(wx, w, out=mean, where=w > 0)
    second = np.full(len(w), np.nan, float)
    np.divide(wx2, w, out=second, where=w > 0)
    var = np.maximum(second - mean * mean, 0.0)
    neff = np.zeros(len(w), float)
    np.divide(w * w, w2, out=neff, where=w2 > 0)
    se = np.full(len(w), np.nan, float)
    np.divide(np.sqrt(var), np.sqrt(neff), out=se, where=neff > 1e-12)
    return mean, se, neff, n.astype(float)


def build_disjoint_peer_rings(base: pd.DataFrame) -> pd.DataFrame:
    """Build non-overlapping peer rings: same bucket, same market outside bucket, global outside market.

    This avoids V10.1's nested bucket/market/global double counting. Each ticker can occur in
    at most one ring for a target row.
    """
    keys = ["axis", "node_id", "source_feature", "transform"]
    work = _moment_columns(base, "ticker_signed_effect_raw", "prior_weight")
    global_tot = _aggregate_moments(work, keys, "g")
    market_tot = _aggregate_moments(work, [*keys, "market"], "m")
    bucket_market_tot = _aggregate_moments(work, [*keys, "market", "bucket"], "b")
    own = work[keys + ["ticker", "market", "bucket", "_w", "_w2", "_wx", "_wx2", "_n"]].copy()
    out = own.merge(global_tot, on=keys, how="left", validate="many_to_one")
    out = out.merge(market_tot, on=[*keys, "market"], how="left", validate="many_to_one")
    out = out.merge(bucket_market_tot, on=[*keys, "market", "bucket"], how="left", validate="many_to_one")

    # bucket peers exclude own ticker
    bw = out["b_w"].to_numpy(float) - out["_w"].to_numpy(float)
    bw2 = out["b_w2"].to_numpy(float) - out["_w2"].to_numpy(float)
    bwx = out["b_wx"].to_numpy(float) - out["_wx"].to_numpy(float)
    bwx2 = out["b_wx2"].to_numpy(float) - out["_wx2"].to_numpy(float)
    bn = out["b_n"].to_numpy(float) - out["_n"].to_numpy(float)
    bmean, bse, bneff, bn = _ring_stats(bw, bw2, bwx, bwx2, bn)

    # market ring explicitly excludes the entire own bucket (including own)
    mw = out["m_w"].to_numpy(float) - out["b_w"].to_numpy(float)
    mw2 = out["m_w2"].to_numpy(float) - out["b_w2"].to_numpy(float)
    mwx = out["m_wx"].to_numpy(float) - out["b_wx"].to_numpy(float)
    mwx2 = out["m_wx2"].to_numpy(float) - out["b_wx2"].to_numpy(float)
    mn = out["m_n"].to_numpy(float) - out["b_n"].to_numpy(float)
    mmean, mse, mneff, mn = _ring_stats(mw, mw2, mwx, mwx2, mn)

    # global ring excludes the entire own market
    gw = out["g_w"].to_numpy(float) - out["m_w"].to_numpy(float)
    gw2 = out["g_w2"].to_numpy(float) - out["m_w2"].to_numpy(float)
    gwx = out["g_wx"].to_numpy(float) - out["m_wx"].to_numpy(float)
    gwx2 = out["g_wx2"].to_numpy(float) - out["m_wx2"].to_numpy(float)
    gn = out["g_n"].to_numpy(float) - out["m_n"].to_numpy(float)
    gmean, gse, gneff, gn = _ring_stats(gw, gw2, gwx, gwx2, gn)

    result = out[keys + ["ticker", "market", "bucket"]].copy()
    for prefix, mean, se, neff, count in [
        ("bucket", bmean, bse, bneff, bn),
        ("market_outer", mmean, mse, mneff, mn),
        ("global_outer", gmean, gse, gneff, gn),
    ]:
        result[f"{prefix}_effect"] = mean
        result[f"{prefix}_se"] = se
        result[f"{prefix}_neff"] = neff
        result[f"{prefix}_count"] = count
    return result


def combine_disjoint_prior(row: pd.Series, config: ConservativeHierarchyConfig) -> tuple[float, float, str, float]:
    candidates = [
        ("bucket", config.bucket_weight),
        ("market_outer", config.market_outer_weight),
        ("global_outer", config.global_outer_weight),
    ]
    parts: list[tuple[str, float, float, float]] = []
    for prefix, requested in candidates:
        mean = _finite(row.get(f"{prefix}_effect"))
        se = _finite(row.get(f"{prefix}_se"))
        neff = _finite(row.get(f"{prefix}_neff"), 0.0)
        if requested > 0 and math.isfinite(mean) and math.isfinite(se) and neff >= config.minimum_peer_effective_n:
            parts.append((prefix, float(requested), mean, se))
    if not parts:
        return math.nan, math.nan, "NONE", 0.0
    denom = sum(p[1] for p in parts)
    ws = np.asarray([p[1] / denom for p in parts], float)
    means = np.asarray([p[2] for p in parts], float)
    ses = np.asarray([p[3] for p in parts], float)
    prior = float(np.sum(ws * means))
    sampling_var = float(np.sum((ws * ses) ** 2))
    # Conservative heterogeneity inflation: disagreement among disjoint rings is uncertainty, not signal.
    heterogeneity_var = float(np.sum(ws * (means - prior) ** 2)) if len(parts) > 1 else 0.0
    prior_se = math.sqrt(max(sampling_var + heterogeneity_var, 0.0))
    return prior, prior_se, "+".join(p[0] for p in parts), float(len(parts))


def build_hierarchy_for_strength(
    base: pd.DataFrame,
    rings: pd.DataFrame,
    selection_folds: Sequence[int],
    prior_strength: float,
    config: ConservativeHierarchyConfig,
) -> pd.DataFrame:
    keys = ["axis", "node_id", "source_feature", "transform", "ticker", "market", "bucket"]
    result = base.merge(rings, on=keys, how="left", validate="one_to_one")
    priors = result.apply(lambda r: combine_disjoint_prior(r, config), axis=1)
    result["peer_prior_signed_effect"] = [x[0] for x in priors]
    result["peer_prior_se"] = [x[1] for x in priors]
    result["peer_prior_sources"] = [x[2] for x in priors]
    result["peer_prior_ring_count"] = [x[3] for x in priors]
    eff = pd.to_numeric(result["selection_effective_n_recalc"], errors="coerce").fillna(0.0).clip(lower=0.0)
    fold_count = pd.to_numeric(result["selection_fold_count"], errors="coerce").fillna(0.0).clip(lower=0.0)
    consistency = pd.to_numeric(result["selection_direction_consistency"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    sample_rel = eff / (eff + max(float(prior_strength), 1e-12))
    fold_rel = np.minimum(fold_count / max(len(selection_folds), 1), 1.0)
    result["ticker_map_reliability"] = sample_rel * np.sqrt(fold_rel * consistency)
    raw = pd.to_numeric(result["ticker_signed_effect_raw"], errors="coerce").to_numpy(float)
    prior = pd.to_numeric(result["peer_prior_signed_effect"], errors="coerce").to_numpy(float)
    reliability = result["ticker_map_reliability"].to_numpy(float)
    posterior = np.where(np.isfinite(prior), reliability * raw + (1.0 - reliability) * prior, raw)
    result["posterior_signed_effect"] = posterior
    result["ticker_specific_delta_raw"] = raw - prior
    result["ticker_specific_delta_posterior"] = posterior - prior
    raw_se = pd.to_numeric(result["ticker_auc_se"], errors="coerce").to_numpy(float)
    peer_se = pd.to_numeric(result["peer_prior_se"], errors="coerce").to_numpy(float)
    denom = np.sqrt(raw_se * raw_se + peer_se * peer_se)
    z = np.full(len(result), np.nan, float)
    np.divide(raw - prior, denom, out=z, where=np.isfinite(denom) & (denom > 1e-12))
    result["ticker_specific_z"] = z
    result["prior_strength"] = float(prior_strength)
    return result


def _bh_qvalues(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, float)
    q = np.full(len(p), np.nan, float)
    finite_idx = np.flatnonzero(np.isfinite(p))
    if not len(finite_idx):
        return q
    vals = np.clip(p[finite_idx], 0.0, 1.0)
    order = np.argsort(vals, kind="mergesort")
    ranked = vals[order]
    m = len(ranked)
    raw = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    temp = np.empty(m, float)
    temp[order] = np.minimum(adj, 1.0)
    q[finite_idx] = temp
    return q


def aggregate_hierarchy_sensitivity(long: pd.DataFrame, config: ConservativeHierarchyConfig) -> pd.DataFrame:
    keys = ["ticker", "axis", "node_id", "source_feature", "transform", "name", "market", "bucket", "industry"]
    rows: list[dict[str, Any]] = []
    for key, part in long.groupby(keys, dropna=False, sort=True):
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        raw = pd.to_numeric(part["ticker_signed_effect_raw"], errors="coerce").to_numpy(float)
        prior = pd.to_numeric(part["peer_prior_signed_effect"], errors="coerce").to_numpy(float)
        posterior = pd.to_numeric(part["posterior_signed_effect"], errors="coerce").to_numpy(float)
        delta = pd.to_numeric(part["ticker_specific_delta_raw"], errors="coerce").to_numpy(float)
        z = pd.to_numeric(part["ticker_specific_z"], errors="coerce").to_numpy(float)
        rel = pd.to_numeric(part["ticker_map_reliability"], errors="coerce").to_numpy(float)
        fz = z[np.isfinite(z)]
        fd = delta[np.isfinite(delta)]
        fp = posterior[np.isfinite(posterior)]
        signs = np.sign(fd[np.abs(fd) > 1e-12])
        post_signs = np.sign(fp[np.abs(fp) > 1e-12])
        delta_sign_stability = float(max(np.mean(signs > 0), np.mean(signs < 0))) if len(signs) else 0.0
        post_sign_stability = float(max(np.mean(post_signs > 0), np.mean(post_signs < 0))) if len(post_signs) else 0.0
        if len(fz):
            median_z = float(np.nanmedian(fz))
            conservative_abs_z = float(np.nanmin(np.abs(fz)))
            conservative_z = math.copysign(conservative_abs_z, median_z) if abs(median_z) > 1e-12 else 0.0
        else:
            median_z = conservative_z = conservative_abs_z = math.nan
        first = part.iloc[0]
        rec.update({
            "sensitivity_strength_count": int(part["prior_strength"].nunique()),
            "ticker_signed_effect_raw": float(np.nanmedian(raw)) if np.isfinite(raw).any() else math.nan,
            "peer_prior_signed_effect_median": float(np.nanmedian(prior)) if np.isfinite(prior).any() else math.nan,
            "posterior_signed_effect_median": float(np.nanmedian(fp)) if len(fp) else math.nan,
            "specific_delta_raw_median": float(np.nanmedian(fd)) if len(fd) else math.nan,
            "specific_z_median": median_z,
            "specific_z_conservative": conservative_z,
            "specific_z_abs_min": conservative_abs_z,
            "reliability_median": float(np.nanmedian(rel[np.isfinite(rel)])) if np.isfinite(rel).any() else 0.0,
            "posterior_sign_stability": post_sign_stability,
            "specific_delta_sign_stability": delta_sign_stability,
            "peer_prior_sources": "+".join(sorted(set(part["peer_prior_sources"].astype(str)) - {"NONE"})),
        })
        for col in [
            "selection_direction", "selection_fold_count", "selection_mean_fixed_auc", "selection_min_fixed_auc",
            "selection_direction_consistency", "selection_mean_matched_concordance", "selection_effective_n",
            "confirmation_mean_fixed_auc", "confirmation_min_fixed_auc", "confirmation_direction_consistency",
            "recent_mean_fixed_auc", "recent_min_fixed_auc", "recent_direction_consistency", "selection_evidence_score",
        ]:
            if col in part.columns:
                rec[col] = first.get(col)
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    zc = pd.to_numeric(out["specific_z_conservative"], errors="coerce").to_numpy(float)
    p = np.where(np.isfinite(zc), 2.0 * norm.sf(np.abs(zc)), np.nan)
    out["specific_p_conservative"] = p
    out["specific_q_ticker_axis"] = np.nan
    for _, idx in out.groupby(["ticker", "axis"], sort=False).groups.items():
        ii = np.asarray(list(idx), int)
        out.loc[ii, "specific_q_ticker_axis"] = _bh_qvalues(out.loc[ii, "specific_p_conservative"].to_numpy(float))
    out["specific_q_global_axis"] = np.nan
    for _, idx in out.groupby(["axis"], sort=False).groups.items():
        ii = np.asarray(list(idx), int)
        out.loc[ii, "specific_q_global_axis"] = _bh_qvalues(out.loc[ii, "specific_p_conservative"].to_numpy(float))
    abs_post = np.abs(pd.to_numeric(out["posterior_signed_effect_median"], errors="coerce"))
    out["ticker_specific_robust"] = (
        (out["reliability_median"] >= config.minimum_reliability)
        & (out["posterior_sign_stability"] >= config.minimum_sensitivity_sign_ratio)
        & (out["specific_delta_sign_stability"] >= config.minimum_sensitivity_sign_ratio)
        & (np.abs(pd.to_numeric(out["specific_z_conservative"], errors="coerce")) >= config.minimum_specific_z)
        & (pd.to_numeric(out["specific_q_ticker_axis"], errors="coerce") <= config.specific_fdr_ticker_axis)
        & (pd.to_numeric(out["specific_q_global_axis"], errors="coerce") <= config.specific_fdr_global_axis)
        & (abs_post >= config.weak_effect)
    )
    out["ticker_specific_strong"] = out["ticker_specific_robust"] & (
        np.abs(pd.to_numeric(out["specific_z_conservative"], errors="coerce")) >= config.robust_specific_z
    ) & (abs_post >= config.strong_effect)
    out["ticker_specificity_score_v10_2"] = (
        abs_post.fillna(0.0) * 2.0
        + np.abs(pd.to_numeric(out["specific_delta_raw_median"], errors="coerce")).fillna(0.0) * 3.0
        + np.minimum(np.abs(pd.to_numeric(out["specific_z_conservative"], errors="coerce")).fillna(0.0), 4.0) * 0.20
    ) * np.sqrt(pd.to_numeric(out["reliability_median"], errors="coerce").fillna(0.0).clip(lower=0.0))
    return out


def build_precision_separator_map(robust: pd.DataFrame, config: ConservativeHierarchyConfig) -> pd.DataFrame:
    ab = robust.loc[robust["axis"].astype(str).eq("AB")].copy()
    if ab.empty:
        return ab
    sel_auc = pd.to_numeric(ab["selection_mean_fixed_auc"], errors="coerce")
    sel_min = pd.to_numeric(ab["selection_min_fixed_auc"], errors="coerce")
    sel_cons = pd.to_numeric(ab["selection_direction_consistency"], errors="coerce")
    sel_folds = pd.to_numeric(ab["selection_fold_count"], errors="coerce")
    eff_n = pd.to_numeric(ab.get("selection_effective_n"), errors="coerce")
    matched = pd.to_numeric(ab.get("selection_mean_matched_concordance"), errors="coerce").fillna(0.5)
    z = np.abs(pd.to_numeric(ab["specific_z_conservative"], errors="coerce"))
    q = pd.to_numeric(ab["specific_q_ticker_axis"], errors="coerce")
    ab["precision_separator_selection_candidate"] = (
        (sel_auc >= config.precision_min_selection_auc)
        & (sel_min >= config.precision_min_selection_min_auc)
        & (sel_cons >= config.precision_min_direction_consistency)
        & (sel_folds >= config.precision_min_selection_folds)
        & (eff_n >= config.precision_min_effective_n)
        & (matched >= config.precision_min_matched_concordance)
        & (z >= config.precision_min_specific_z)
        & (q <= config.specific_fdr_ticker_axis)
        & (ab["specific_delta_sign_stability"] >= config.minimum_sensitivity_sign_ratio)
    )
    conf = pd.to_numeric(ab.get("confirmation_mean_fixed_auc"), errors="coerce")
    conf_min = pd.to_numeric(ab.get("confirmation_min_fixed_auc"), errors="coerce")
    conf_cons = pd.to_numeric(ab.get("confirmation_direction_consistency"), errors="coerce")
    recent = pd.to_numeric(ab.get("recent_mean_fixed_auc"), errors="coerce")
    recent_cons = pd.to_numeric(ab.get("recent_direction_consistency"), errors="coerce")
    ab["confirmation_status"] = np.where(
        ~np.isfinite(conf), "UNTESTED",
        np.where((conf >= config.precision_confirm_auc) & (conf_min >= config.precision_confirm_min_auc) & (conf_cons >= config.precision_confirm_direction_consistency), "SUPPORTED", "NOT_SUPPORTED")
    )
    ab["recent_status"] = np.where(
        ~np.isfinite(recent), "UNTESTED",
        np.where((recent >= config.precision_recent_auc) & (recent_cons >= config.precision_recent_direction_consistency), "SUPPORTED", "NOT_SUPPORTED")
    )
    ab["precision_separator_confirmed"] = ab["precision_separator_selection_candidate"] & ab["confirmation_status"].eq("SUPPORTED")
    ab["precision_separator_recent_confirmed"] = ab["precision_separator_confirmed"] & ab["recent_status"].eq("SUPPORTED")
    ab["precision_separator_score_v10_2"] = (
        np.maximum(sel_auc - 0.5, 0.0) * 4.0
        + np.maximum(sel_min - 0.5, 0.0) * 2.0
        + np.maximum(matched - 0.5, 0.0) * 1.5
        + np.minimum(z.fillna(0.0), 4.0) * 0.10
        + ab["ticker_specificity_score_v10_2"].fillna(0.0)
    ) * np.sqrt(pd.to_numeric(ab["reliability_median"], errors="coerce").fillna(0.0).clip(lower=0.0))
    return ab.sort_values(["precision_separator_selection_candidate", "precision_separator_score_v10_2", "ticker", "node_id"], ascending=[False, False, True, True], kind="mergesort").reset_index(drop=True)


def deduplicate_precision_sources(precision: pd.DataFrame, max_per_ticker: int = 24) -> pd.DataFrame:
    if precision.empty:
        return precision.copy()
    part = precision.loc[precision["precision_separator_selection_candidate"]].copy()
    part = part.sort_values(["ticker", "source_feature", "precision_separator_score_v10_2", "node_id"], ascending=[True, True, False, True], kind="mergesort")
    part = part.drop_duplicates(["ticker", "source_feature"], keep="first")
    part = part.sort_values(["ticker", "precision_separator_score_v10_2", "source_feature"], ascending=[True, False, True], kind="mergesort")
    part["source_rank"] = part.groupby("ticker").cumcount() + 1
    return part.loc[part["source_rank"] <= int(max_per_ticker)].reset_index(drop=True)


def build_residual_similarity(
    robust: pd.DataFrame,
    config: ConservativeHierarchyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Similarity uses ticker-specific residual, not posterior, to avoid common-prior collapse."""
    raw = robust.loc[robust["transform"].astype(str).eq("raw")].copy()
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    raw["signature_node"] = raw["axis"].astype(str) + "::" + raw["source_feature"].astype(str)
    node_stats = raw.groupby("signature_node", as_index=False).agg(
        ticker_count=("ticker", "nunique"),
        median_reliability=("reliability_median", "median"),
        median_abs_residual=("specific_delta_raw_median", lambda s: float(np.nanmedian(np.abs(pd.to_numeric(s, errors="coerce")))))
    )
    total_tickers = max(int(raw["ticker"].nunique()), 1)
    node_stats["coverage"] = node_stats["ticker_count"] / total_tickers
    nodes = node_stats.loc[node_stats["coverage"] >= config.similarity_min_coverage].sort_values(
        ["median_abs_residual", "median_reliability", "signature_node"], ascending=[False, False, True], kind="mergesort"
    ).head(config.similarity_feature_count)
    selected = nodes["signature_node"].tolist()
    work = raw.loc[raw["signature_node"].isin(selected)].copy()
    matrix = work.pivot_table(index="ticker", columns="signature_node", values="specific_delta_raw_median", aggfunc="first").sort_index()
    weight = work.pivot_table(index="ticker", columns="signature_node", values="reliability_median", aggfunc="first").reindex_like(matrix)
    tickers = matrix.index.astype(str).tolist()
    sim = np.eye(len(tickers), dtype=float)
    edges: list[dict[str, Any]] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a = matrix.iloc[i].to_numpy(float); b = matrix.iloc[j].to_numpy(float)
            wa = weight.iloc[i].to_numpy(float); wb = weight.iloc[j].to_numpy(float)
            valid = np.isfinite(a) & np.isfinite(b)
            common = int(valid.sum())
            if common < config.similarity_min_common:
                value = math.nan
            else:
                w = np.sqrt(np.clip(wa[valid], 0, None) * np.clip(wb[valid], 0, None))
                if not np.isfinite(w).any() or float(np.nansum(w)) <= 0:
                    w = np.ones(common)
                av = a[valid]; bv = b[valid]
                denom = math.sqrt(float(np.sum(w * av * av)) * float(np.sum(w * bv * bv)))
                cosine = float(np.sum(w * av * bv) / denom) if denom > 1e-12 else 0.0
                rho = spearmanr(av, bv, nan_policy="omit").statistic
                rho = float(rho) if math.isfinite(_finite(rho)) else 0.0
                value = 0.60 * cosine + 0.40 * rho
                sim[i, j] = sim[j, i] = value
                edges.append({"ticker_a": tickers[i], "ticker_b": tickers[j], "residual_similarity": value, "common_nodes": common, "weighted_cosine": cosine, "spearman": rho})
    sim_df = pd.DataFrame(sim, index=tickers, columns=tickers)
    edge_df = pd.DataFrame(edges)
    selected_edges: list[pd.DataFrame] = []
    if not edge_df.empty:
        for ticker in tickers:
            part = edge_df.loc[(edge_df["ticker_a"].eq(ticker)) | (edge_df["ticker_b"].eq(ticker))].copy()
            part = part.loc[part["residual_similarity"] >= config.similarity_edge_threshold]
            selected_edges.append(part.sort_values(["residual_similarity", "common_nodes"], ascending=[False, False]).head(config.similarity_top_k))
    selected_df = pd.concat(selected_edges, ignore_index=True).drop_duplicates(["ticker_a", "ticker_b"]) if selected_edges else pd.DataFrame()
    if len(tickers) >= 2:
        filled = sim_df.to_numpy(float)
        finite_off = filled[np.isfinite(filled) & (~np.eye(len(tickers), dtype=bool))]
        fill = float(np.nanmedian(finite_off)) if len(finite_off) else 0.0
        filled = np.where(np.isfinite(filled), filled, fill)
        distance = np.clip((1.0 - np.clip(filled, -1, 1)) / 2.0, 0.0, 1.0)
        np.fill_diagonal(distance, 0.0)
        tree = linkage(squareform(distance, checks=False), method="average")
        labels = fcluster(tree, t=(1.0 - config.similarity_cluster_threshold) / 2.0, criterion="distance")
        clusters = pd.DataFrame({"ticker": tickers, "ticker_residual_cluster": labels.astype(int)})
    else:
        clusters = pd.DataFrame({"ticker": tickers, "ticker_residual_cluster": [1] * len(tickers)})
    return sim_df, selected_df, clusters, nodes
