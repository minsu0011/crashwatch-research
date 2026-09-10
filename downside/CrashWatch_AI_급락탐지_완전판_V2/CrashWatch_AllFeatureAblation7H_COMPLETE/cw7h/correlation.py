from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_selection import mutual_info_classif

from .data import PreparedData, References
from .folds import FoldSlice
from .utils import atomic_json, canonical_hash

LOGGER = logging.getLogger(__name__)


def _balanced_sample_indices(dates_ns: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    n = len(dates_ns)
    if n <= max_rows:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    unique_dates, inverse = np.unique(dates_ns, return_inverse=True)
    per_date = max(1, max_rows // len(unique_dates))
    selected: list[np.ndarray] = []
    for code in range(len(unique_dates)):
        idx = np.flatnonzero(inverse == code)
        if len(idx) <= per_date:
            selected.append(idx)
        else:
            selected.append(np.sort(rng.choice(idx, size=per_date, replace=False)))
    result = np.concatenate(selected)
    if len(result) < max_rows:
        remaining = np.setdiff1d(np.arange(n, dtype=np.int64), result, assume_unique=False)
        extra = rng.choice(remaining, size=min(max_rows - len(result), len(remaining)), replace=False)
        result = np.concatenate([result, extra])
    if len(result) > max_rows:
        result = rng.choice(result, size=max_rows, replace=False)
    return np.sort(result.astype(np.int64))


def _group_residualize(X: np.ndarray, group_codes: np.ndarray, *, zscore: bool = True) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    codes = np.asarray(group_codes, dtype=np.int32)
    groups = int(codes.max()) + 1 if len(codes) else 0
    out = np.empty_like(X, dtype=np.float32)
    for j in range(X.shape[1]):
        col = X[:, j].astype(np.float64, copy=False)
        valid = np.isfinite(col)
        counts = np.bincount(codes[valid], minlength=groups).astype(np.float64)
        # NumPy 2.x returns an integer bincount for an empty weighted input.
        # Keep every accumulator floating-point so an all-missing feature is
        # residualized to NaN instead of failing during true division.
        sums = np.bincount(codes[valid], weights=col[valid], minlength=groups).astype(np.float64, copy=False)
        means = np.divide(sums, counts, out=np.zeros(groups, dtype=np.float64), where=counts > 0)
        residual = col - means[codes]
        residual[~valid] = np.nan
        if zscore:
            sq = np.bincount(codes[valid], weights=residual[valid] ** 2, minlength=groups).astype(np.float64, copy=False)
            std = np.sqrt(np.divide(sq, np.maximum(counts - 1, 1), out=np.ones(groups, dtype=np.float64), where=counts > 1))
            std[~np.isfinite(std) | (std < 1e-12)] = 1.0
            residual = residual / std[codes]
        out[:, j] = residual.astype(np.float32)
    return out


def _fast_pairwise_pearson(X: np.ndarray, min_periods: int) -> np.ndarray:
    """Exact pairwise-complete Pearson correlation using four BLAS matrix products.

    pandas.DataFrame.corr(method="spearman") ranks every feature pair separately and
    becomes prohibitively slow near 400+ columns.  This routine computes each pair's
    own count, sums, sum-of-squares and cross-product in vectorized form while keeping
    NaNs excluded exactly.
    """
    raw = np.asarray(X, dtype=np.float32)
    valid = np.isfinite(raw)
    # Per-column affine scaling does not change correlation, but prevents overflow
    # and catastrophic cancellation for volume/notional features with large units.
    with np.errstate(invalid="ignore"):
        center = np.nanmean(raw, axis=0, dtype=np.float64)
        scale = np.nanstd(raw, axis=0, dtype=np.float64)
    center[~np.isfinite(center)] = 0.0
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    x = ((raw - center.astype(np.float32)) / scale.astype(np.float32)).astype(np.float32, copy=False)
    x[~valid] = np.nan
    mask = valid.astype(np.float32, copy=False)
    x0 = np.where(valid, x, np.float32(0.0)).astype(np.float32, copy=False)

    # A[i,j] = sum(X_i | X_j observed), B[i,j] = sum(X_i^2 | X_j observed).
    count = mask.T @ mask
    sums = x0.T @ mask
    cross = x0.T @ x0
    squares = (x0 * x0).T @ mask

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        safe_count = np.maximum(count, 1.0)
        covariance = cross - (sums * sums.T) / safe_count
        variance_i = squares - (sums * sums) / safe_count
        variance_j = variance_i.T
        denominator = np.sqrt(np.maximum(variance_i, 0.0) * np.maximum(variance_j, 0.0))
        corr = covariance / denominator

    invalid = (count < float(min_periods)) | ~(denominator > 1e-20)
    corr[invalid] = 0.0
    corr = np.clip(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    corr = ((corr + corr.T) * np.float32(0.5)).astype(np.float32, copy=False)
    np.fill_diagonal(corr, 1.0)
    return corr


def _safe_corr_frame(X: np.ndarray, names: list[str], method: str, min_periods: int) -> pd.DataFrame:
    if method == "pearson":
        values = _fast_pairwise_pearson(X, min_periods)
    elif method == "spearman":
        # Rank each feature once, then apply pairwise Pearson to those global ranks.
        # With unequal missingness this is the stable "global-rank Spearman" variant;
        # it avoids pandas' prohibitively expensive per-pair reranking at 400+ columns.
        ranks = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=names, copy=False).rank(
            axis=0, method="average", na_option="keep", pct=False
        ).to_numpy(dtype=np.float32, copy=False)
        values = _fast_pairwise_pearson(ranks, min_periods)
        del ranks
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    return pd.DataFrame(values, index=names, columns=names)


def _target_corr(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    pearson = np.full(X.shape[1], np.nan, dtype=np.float64)
    spearman = np.full(X.shape[1], np.nan, dtype=np.float64)
    for j in range(X.shape[1]):
        col = X[:, j].astype(np.float64, copy=False)
        valid = np.isfinite(col)
        if valid.sum() < 20 or np.nanstd(col[valid]) < 1e-12 or np.unique(y[valid]).size < 2:
            continue
        pearson[j] = np.corrcoef(col[valid], y[valid])[0, 1]
        ranks = pd.Series(col[valid]).rank(method="average").to_numpy(dtype=np.float64)
        spearman[j] = np.corrcoef(ranks, y[valid])[0, 1]
    return pearson, spearman


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    out = np.zeros_like(values)
    if not finite.any():
        return out
    low, high = np.nanquantile(values[finite], [0.02, 0.98])
    if high <= low:
        out[finite] = 0.5
        return out
    out[finite] = np.clip((values[finite] - low) / (high - low), 0, 1)
    return out


def _cluster_assignments(combined_abs: np.ndarray, thresholds: list[float]) -> dict[float, np.ndarray]:
    distance = np.clip(1.0 - combined_abs, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    z = linkage(condensed, method="average", optimal_ordering=False)
    result: dict[float, np.ndarray] = {}
    for threshold in thresholds:
        result[float(threshold)] = fcluster(z, t=max(1e-9, 1.0 - float(threshold)), criterion="distance").astype(np.int32)
    return result


def run_correlation_audit(
    prepared: PreparedData,
    refs: References,
    folds: list[FoldSlice],
    output_dir: Path,
    config: dict[str, Any],
    *,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    corr_dir = output_dir / "correlation"
    corr_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = corr_dir / "correlation_manifest.json"
    if not force and manifest_path.exists():
        existing = __import__("json").load(manifest_path.open(encoding="utf-8"))
        required = [corr_dir / "feature_correlation_summary.csv", corr_dir / "primary_clusters.csv", corr_dir / "feature_priority.csv"]
        config_hash = canonical_hash(config.get("correlation", {}))
        if (
            existing.get("status") == "complete"
            and existing.get("dataset_signature") == prepared.signature
            and existing.get("correlation_config_hash") == config_hash
            and all(p.exists() for p in required)
        ):
            LOGGER.info("correlation audit cache hit")
            return existing

    X = np.load(prepared.x_path, mmap_mode="r")
    y = np.load(prepared.y_path, mmap_mode="r")
    dates = np.load(prepared.dates_path, mmap_mode="r")
    tickers = np.load(prepared.tickers_path, mmap_mode="r")
    names = prepared.feature_names
    sample_rows = int(config["correlation"].get("sample_rows", 40_000))
    spearman_rows = int(config["correlation"].get("spearman_sample_rows", 15_000))
    mi_rows = int(config["correlation"].get("mi_sample_rows", 20_000))
    seed = int(config["correlation"].get("seed", 17))
    sample_idx = _balanced_sample_indices(dates, sample_rows, seed)
    Xs = np.asarray(X[sample_idx], dtype=np.float32)
    ys = np.asarray(y[sample_idx], dtype=np.uint8)
    ds = np.asarray(dates[sample_idx], dtype=np.int64)
    ts = np.asarray(tickers[sample_idx]).astype(str)
    min_periods = max(100, int(len(sample_idx) * float(config["correlation"].get("min_pair_fraction", 0.10))))

    LOGGER.info("correlation audit: pearson sample=%s features=%s", len(sample_idx), len(names))
    pearson = _safe_corr_frame(Xs, names, "pearson", min_periods)
    spearman_local_idx = _balanced_sample_indices(ds, min(spearman_rows, len(ds)), seed + 29)
    spearman_min_periods = max(100, int(len(spearman_local_idx) * float(config["correlation"].get("min_pair_fraction", 0.10))))
    LOGGER.info("correlation audit: spearman sample=%s", len(spearman_local_idx))
    spearman = _safe_corr_frame(Xs[spearman_local_idx], names, "spearman", spearman_min_periods)
    _, date_codes = np.unique(ds, return_inverse=True)
    _, ticker_codes = np.unique(ts, return_inverse=True)
    within_date_x = _group_residualize(Xs, date_codes, zscore=True)
    within_ticker_x = _group_residualize(Xs, ticker_codes, zscore=True)
    within_date = _safe_corr_frame(within_date_x, names, "pearson", min_periods)
    within_ticker = _safe_corr_frame(within_ticker_x, names, "pearson", min_periods)

    missing = np.isnan(Xs).astype(np.float32)
    missing_std = missing.std(axis=0)
    missing_corr = np.zeros((len(names), len(names)), dtype=np.float64)
    variable_missing = np.flatnonzero(missing_std > 1e-9)
    if len(variable_missing) > 1:
        sub = np.corrcoef(missing[:, variable_missing], rowvar=False)
        sub = np.nan_to_num(sub, nan=0.0)
        missing_corr[np.ix_(variable_missing, variable_missing)] = sub
    np.fill_diagonal(missing_corr, 1.0)

    pearson_values = pearson.to_numpy()
    spearman_values = spearman.to_numpy()
    date_values = within_date.to_numpy()
    ticker_values = within_ticker.to_numpy()
    combined_abs = np.maximum.reduce([
        np.abs(pearson_values), np.abs(spearman_values), np.abs(date_values), np.abs(ticker_values)
    ])
    combined_abs = np.nan_to_num(combined_abs, nan=0.0)
    np.fill_diagonal(combined_abs, 1.0)

    # Leakage-safe clustering basis: all conditional/cluster ablations use a fixed
    # correlation structure learned only from the earliest outer fold's training data.
    # The full-development matrices above remain descriptive and never select model inputs.
    eligible_folds = sorted((f for f in folds if f.eligible), key=lambda f: f.fold_id)
    if not eligible_folds:
        raise RuntimeError("No eligible outer fold for leakage-safe correlation clustering")
    cluster_basis_fold = eligible_folds[0]
    cluster_sample_rows = int(config["correlation"].get("cluster_sample_rows", 20_000))
    cluster_train_dates = np.asarray(dates[cluster_basis_fold.train_start:cluster_basis_fold.train_stop], dtype=np.int64)
    cluster_local_idx = _balanced_sample_indices(
        cluster_train_dates, min(cluster_sample_rows, len(cluster_train_dates)), seed + 701
    )
    cluster_idx = cluster_local_idx + cluster_basis_fold.train_start
    cluster_x = np.asarray(X[cluster_idx], dtype=np.float32)
    cluster_dates = np.asarray(dates[cluster_idx], dtype=np.int64)
    cluster_tickers = np.asarray(tickers[cluster_idx]).astype(str)
    cluster_min_periods = max(100, int(len(cluster_idx) * float(config["correlation"].get("min_pair_fraction", 0.10))))
    cluster_pearson = _safe_corr_frame(cluster_x, names, "pearson", cluster_min_periods).to_numpy(dtype=np.float32)
    cluster_spearman = _safe_corr_frame(cluster_x, names, "spearman", cluster_min_periods).to_numpy(dtype=np.float32)
    _, cluster_date_codes = np.unique(cluster_dates, return_inverse=True)
    _, cluster_ticker_codes = np.unique(cluster_tickers, return_inverse=True)
    cluster_date_residual = _group_residualize(cluster_x, cluster_date_codes, zscore=True)
    cluster_ticker_residual = _group_residualize(cluster_x, cluster_ticker_codes, zscore=True)
    cluster_within_date = _safe_corr_frame(cluster_date_residual, names, "pearson", cluster_min_periods).to_numpy(dtype=np.float32)
    cluster_within_ticker = _safe_corr_frame(cluster_ticker_residual, names, "pearson", cluster_min_periods).to_numpy(dtype=np.float32)
    cluster_basis_abs = np.maximum.reduce([
        np.abs(cluster_pearson), np.abs(cluster_spearman),
        np.abs(cluster_within_date), np.abs(cluster_within_ticker),
    ])
    cluster_basis_abs = np.nan_to_num(cluster_basis_abs, nan=0.0).astype(np.float32, copy=False)
    np.fill_diagonal(cluster_basis_abs, 1.0)

    # Outcome-free representative quality is also estimated only in the same earliest
    # training window, so missingness/coverage changes from future folds cannot leak in.
    cluster_missing_ratio = np.mean(~np.isfinite(cluster_x), axis=0, dtype=np.float64)
    cluster_unique_count = pd.DataFrame(cluster_x, columns=names, copy=False).nunique(dropna=True).to_numpy(dtype=np.float64)
    unique_cluster_tickers = max(1, len(np.unique(cluster_tickers)))
    cluster_coverage = np.zeros(len(names), dtype=np.float64)
    for j in range(len(names)):
        observed = np.isfinite(cluster_x[:, j])
        cluster_coverage[j] = len(np.unique(cluster_tickers[observed])) / unique_cluster_tickers if observed.any() else 0.0

    del cluster_date_residual, cluster_ticker_residual

    # Time-stability audit: Pearson matrices are recomputed inside each outer-training window.
    # This is descriptive and never touches sealed data.
    fold_corr_sample_rows = int(config["correlation"].get("fold_sample_rows", 25_000))
    fold_pearson_stack: list[np.ndarray] = []
    for fold in folds:
        train_dates_local = np.asarray(dates[fold.train_start:fold.train_stop], dtype=np.int64)
        local_idx = _balanced_sample_indices(train_dates_local, min(fold_corr_sample_rows, len(train_dates_local)), seed + 100 + fold.fold_id)
        global_idx = local_idx + fold.train_start
        fold_x = np.asarray(X[global_idx], dtype=np.float32)
        fold_min_periods = max(50, int(len(global_idx) * float(config["correlation"].get("min_pair_fraction", 0.10))))
        fold_corr = _safe_corr_frame(fold_x, names, "pearson", fold_min_periods).to_numpy(dtype=np.float32)
        fold_pearson_stack.append(fold_corr)
        del fold_x, fold_corr
    fold_pearson_array = np.stack(fold_pearson_stack, axis=0) if fold_pearson_stack else np.empty((0, len(names), len(names)), dtype=np.float32)
    np.savez_compressed(corr_dir / "fold_pearson_matrices.npz", fold_ids=np.array([f.fold_id for f in folds], dtype=np.int16), matrices=fold_pearson_array)

    target_pearson, target_spearman = _target_corr(Xs, ys)
    fold_rows: list[dict[str, Any]] = []
    fold_target_values = np.full((len(folds), len(names)), np.nan, dtype=np.float64)
    for fi, fold in enumerate(folds):
        idx = slice(fold.validation_start, fold.validation_stop)
        fp, fs = _target_corr(np.asarray(X[idx], dtype=np.float32), np.asarray(y[idx], dtype=np.uint8))
        fold_target_values[fi] = fp
        for j, feature in enumerate(names):
            fold_rows.append({
                "outer_fold": fold.fold_id,
                "feature": feature,
                "target_pearson": fp[j],
                "target_spearman": fs[j],
            })
    pd.DataFrame(fold_rows).to_csv(corr_dir / "feature_target_correlation_by_fold.csv", index=False)

    mi_idx_local = _balanced_sample_indices(ds, min(mi_rows, len(ds)), seed + 1)
    Xmi = np.asarray(Xs[mi_idx_local], dtype=np.float32)
    ymi = np.asarray(ys[mi_idx_local], dtype=np.uint8)
    medians = np.nanmedian(Xmi, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    nan_rows, nan_cols = np.where(~np.isfinite(Xmi))
    if len(nan_rows):
        Xmi[nan_rows, nan_cols] = medians[nan_cols]
    LOGGER.info("mutual information: rows=%s", len(Xmi))
    mi = mutual_info_classif(
        Xmi,
        ymi,
        discrete_features=False,
        n_neighbors=int(config["correlation"].get("mi_neighbors", 3)),
        random_state=seed,
        n_jobs=int(config["correlation"].get("mi_jobs", -1)),
    )

    audit = refs.audit.set_index("feature").reindex(names)
    missing_ratio = audit["missing_ratio"].fillna(1.0).to_numpy(dtype=np.float64) if "missing_ratio" in audit else np.isnan(Xs).mean(axis=0)
    coverage = audit["ticker_coverage"].fillna(0.0).to_numpy(dtype=np.float64) if "ticker_coverage" in audit else np.ones(len(names))
    unique_count = audit["unique_count"].fillna(0.0).to_numpy(dtype=np.float64) if "unique_count" in audit else np.zeros(len(names))
    fold_abs = np.nanmean(np.abs(fold_target_values), axis=0)
    fold_sign_consistency = np.maximum(
        np.nanmean(fold_target_values > 0, axis=0),
        np.nanmean(fold_target_values < 0, axis=0),
    )
    predictive_quality = (
        0.28 * (1.0 - np.clip(missing_ratio, 0, 1))
        + 0.24 * _normalize01(mi)
        + 0.18 * _normalize01(np.abs(target_pearson))
        + 0.10 * _normalize01(fold_abs)
        + 0.08 * np.clip(fold_sign_consistency, 0, 1)
        + 0.07 * np.clip(coverage, 0, 1)
        + 0.05 * _normalize01(np.log1p(unique_count))
    )
    # Representative selection is deliberately outcome-free. This prevents target leakage
    # when the representative-only model is evaluated on outer validation folds.
    unsupervised_base_quality = (
        0.58 * (1.0 - np.clip(cluster_missing_ratio, 0, 1))
        + 0.22 * np.clip(cluster_coverage, 0, 1)
        + 0.20 * _normalize01(np.log1p(cluster_unique_count))
    )

    thresholds = sorted(set(float(x) for x in config["correlation"].get("cluster_thresholds", [0.80, 0.90, 0.92, 0.95, 0.98])))
    assignments = _cluster_assignments(cluster_basis_abs, thresholds)
    primary_threshold = float(config["correlation"].get("primary_cluster_threshold", 0.92))
    if primary_threshold not in assignments:
        assignments.update(_cluster_assignments(cluster_basis_abs, [primary_threshold]))
    primary = assignments[primary_threshold]

    summary_rows: list[dict[str, Any]] = []
    max_corr = np.zeros(len(names), dtype=np.float64)
    max_corr_peer = [""] * len(names)
    for i, feature in enumerate(names):
        row = combined_abs[i].copy()
        row[i] = -1
        peer = int(np.argmax(row))
        max_corr[i] = row[peer]
        max_corr_peer[i] = names[peer]
        peer_fold_corr = fold_pearson_array[:, i, peer].astype(np.float64) if len(fold_pearson_array) else np.array([], dtype=np.float64)
        peer_fold_abs = np.abs(peer_fold_corr[np.isfinite(peer_fold_corr)])
        summary_rows.append({
            "feature": feature,
            "group": prepared.feature_groups.get(feature, ""),
            "missing_ratio": missing_ratio[i],
            "ticker_coverage": coverage[i],
            "unique_count": unique_count[i],
            "target_pearson": target_pearson[i],
            "target_spearman": target_spearman[i],
            "target_fold_abs_mean": fold_abs[i],
            "target_fold_sign_consistency": fold_sign_consistency[i],
            "mutual_information": mi[i],
            "quality_score": predictive_quality[i],
            "max_abs_correlation": max_corr[i],
            "max_corr_peer": max_corr_peer[i],
            "max_peer_fold_abs_corr_mean": float(np.mean(peer_fold_abs)) if len(peer_fold_abs) else float("nan"),
            "max_peer_fold_abs_corr_min": float(np.min(peer_fold_abs)) if len(peer_fold_abs) else float("nan"),
            "max_peer_fold_primary_threshold_ratio": float(np.mean(peer_fold_abs >= float(config["correlation"].get("primary_cluster_threshold", 0.92)))) if len(peer_fold_abs) else float("nan"),
            "max_missingness_correlation": float(np.max(np.abs(missing_corr[i][np.arange(len(names)) != i]))) if len(names) > 1 else 0.0,
        })
    summary = pd.DataFrame(summary_rows)

    cluster_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    representative_by_cluster: dict[int, str] = {}
    for threshold, labels in assignments.items():
        for cluster_id in np.unique(labels):
            members = np.flatnonzero(labels == cluster_id)
            if len(members) == 1:
                centrality = np.ones(1, dtype=np.float64)
            else:
                sub = cluster_basis_abs[np.ix_(members, members)].copy()
                np.fill_diagonal(sub, np.nan)
                centrality = np.nanmean(sub, axis=1)
                centrality = np.nan_to_num(centrality, nan=0.0)
            representative_scores = 0.75 * unsupervised_base_quality[members] + 0.25 * _normalize01(centrality)
            representative_index = int(members[np.argmax(representative_scores)])
            representative = names[representative_index]
            score_by_index = {int(idx): float(score) for idx, score in zip(members, representative_scores)}
            centrality_by_index = {int(idx): float(value) for idx, value in zip(members, centrality)}
            if abs(threshold - primary_threshold) < 1e-12:
                representative_by_cluster[int(cluster_id)] = representative
            for idx in members:
                row = {
                    "threshold": threshold,
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(members)),
                    "feature": names[idx],
                    "representative": representative,
                    "is_representative": bool(idx == representative_index),
                    "predictive_quality_score": predictive_quality[idx],
                    "representative_quality_score": score_by_index[int(idx)],
                    "within_cluster_centrality": centrality_by_index[int(idx)],
                }
                cluster_rows.append(row)
                if abs(threshold - primary_threshold) < 1e-12:
                    primary_rows.append(row)
    cluster_frame = pd.DataFrame(cluster_rows)
    primary_frame = pd.DataFrame(primary_rows)
    primary_frame.to_csv(corr_dir / "primary_clusters.csv", index=False)
    cluster_frame.to_csv(corr_dir / "all_cluster_assignments.csv", index=False)

    primary_map = primary_frame.set_index("feature")
    summary["primary_cluster_id"] = summary["feature"].map(primary_map["cluster_id"])
    summary["primary_cluster_size"] = summary["feature"].map(primary_map["cluster_size"])
    summary["primary_representative"] = summary["feature"].map(primary_map["representative"])
    summary["is_primary_representative"] = summary["feature"].map(primary_map["is_representative"])
    summary["representative_quality_score"] = summary["feature"].map(primary_map["representative_quality_score"])
    summary["within_cluster_centrality"] = summary["feature"].map(primary_map["within_cluster_centrality"])
    summary.sort_values(["quality_score", "mutual_information"], ascending=False, inplace=True)
    summary.to_csv(corr_dir / "feature_correlation_summary.csv", index=False)
    summary[["feature", "quality_score", "mutual_information", "target_fold_abs_mean", "max_abs_correlation"]].to_csv(corr_dir / "feature_priority.csv", index=False)

    high_threshold = float(config["correlation"].get("edge_output_threshold", 0.70))
    edges: list[dict[str, Any]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if combined_abs[i, j] < high_threshold:
                continue
            fold_values = fold_pearson_array[:, i, j].astype(np.float64) if len(fold_pearson_array) else np.array([], dtype=np.float64)
            fold_values = fold_values[np.isfinite(fold_values)]
            fold_abs_values = np.abs(fold_values)
            global_sign = np.sign(pearson_values[i, j])
            edges.append({
                "feature_a": names[i],
                "feature_b": names[j],
                "pearson": pearson_values[i, j],
                "spearman": spearman_values[i, j],
                "within_date_pearson": date_values[i, j],
                "within_ticker_pearson": ticker_values[i, j],
                "missingness_corr": missing_corr[i, j],
                "combined_abs_corr": combined_abs[i, j],
                "fold_pearson_mean": float(np.mean(fold_values)) if len(fold_values) else float("nan"),
                "fold_abs_pearson_mean": float(np.mean(fold_abs_values)) if len(fold_abs_values) else float("nan"),
                "fold_abs_pearson_min": float(np.min(fold_abs_values)) if len(fold_abs_values) else float("nan"),
                "fold_abs_pearson_max": float(np.max(fold_abs_values)) if len(fold_abs_values) else float("nan"),
                "fold_sign_consistency": float(np.mean(np.sign(fold_values) == global_sign)) if len(fold_values) and global_sign != 0 else float("nan"),
                "fold_ratio_abs_ge_080": float(np.mean(fold_abs_values >= 0.80)) if len(fold_abs_values) else float("nan"),
                "fold_ratio_abs_ge_primary": float(np.mean(fold_abs_values >= primary_threshold)) if len(fold_abs_values) else float("nan"),
            })
    edge_columns = [
        "feature_a", "feature_b", "pearson", "spearman", "within_date_pearson",
        "within_ticker_pearson", "missingness_corr", "combined_abs_corr",
        "fold_pearson_mean", "fold_abs_pearson_mean", "fold_abs_pearson_min",
        "fold_abs_pearson_max", "fold_sign_consistency", "fold_ratio_abs_ge_080",
        "fold_ratio_abs_ge_primary",
    ]
    edge_frame = pd.DataFrame(edges, columns=edge_columns)
    if not edge_frame.empty:
        edge_frame.sort_values("combined_abs_corr", ascending=False, inplace=True)
    edge_frame.to_csv(corr_dir / "high_correlation_pairs.csv", index=False)

    # Gzip matrices keep exact labels without requiring Excel or an additional binary reader.
    for filename, matrix in [
        ("pearson.csv.gz", pearson),
        ("spearman.csv.gz", spearman),
        ("within_date_pearson.csv.gz", within_date),
        ("within_ticker_pearson.csv.gz", within_ticker),
        ("combined_abs_correlation.csv.gz", pd.DataFrame(combined_abs, index=names, columns=names)),
        ("cluster_basis_combined_abs.csv.gz", pd.DataFrame(cluster_basis_abs, index=names, columns=names)),
        ("missingness_correlation.csv.gz", pd.DataFrame(missing_corr, index=names, columns=names)),
    ]:
        matrix.to_csv(corr_dir / filename, compression="gzip")
    np.savez_compressed(
        corr_dir / "correlation_matrices.npz",
        pearson=pearson_values.astype(np.float32),
        spearman=spearman_values.astype(np.float32),
        within_date=date_values.astype(np.float32),
        within_ticker=ticker_values.astype(np.float32),
        missingness=missing_corr.astype(np.float32),
        combined_abs=combined_abs.astype(np.float32),
        cluster_basis_abs=cluster_basis_abs.astype(np.float32),
    )
    primary_representatives = sorted(set(representative_by_cluster.values()), key=names.index)
    atomic_json(primary_representatives, corr_dir / "primary_representatives.json")
    atomic_json({str(k): v for k, v in representative_by_cluster.items()}, corr_dir / "representative_by_cluster.json")
    manifest = {
        "status": "complete",
        "dataset_signature": prepared.signature,
        "feature_count": len(names),
        "sample_rows": int(len(sample_idx)),
        "spearman_sample_rows": int(len(spearman_local_idx)),
        "spearman_method": "global_rank_pairwise_complete_pearson",
        "mi_rows": int(len(Xmi)),
        "cluster_basis_outer_fold": int(cluster_basis_fold.fold_id),
        "cluster_basis_rows": int(len(cluster_idx)),
        "cluster_basis_leakage_safe": True,
        "fold_correlation_sample_rows": fold_corr_sample_rows,
        "min_periods": int(min_periods),
        "thresholds": thresholds,
        "primary_threshold": primary_threshold,
        "primary_cluster_count": int(len(np.unique(primary))),
        "primary_representative_count": len(primary_representatives),
        "high_correlation_edge_count": len(edges),
        "correlation_hash": canonical_hash({"threshold": primary_threshold, "clusters": primary_rows}),
        "correlation_config_hash": canonical_hash(config.get("correlation", {})),
    }
    atomic_json(manifest, manifest_path)
    return manifest
