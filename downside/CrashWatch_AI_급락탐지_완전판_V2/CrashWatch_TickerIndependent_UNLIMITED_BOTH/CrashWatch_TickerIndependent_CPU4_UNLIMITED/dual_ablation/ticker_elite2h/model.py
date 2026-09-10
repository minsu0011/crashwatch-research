from __future__ import annotations

import gc
import json
import math
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..io_utils import atomic_csv, atomic_json, atomic_parquet
from ..refine12h.calibration import (
    SafeCalibrationPolicy,
    apply_calibrator,
    choose_threshold,
    fit_calibrator,
    logit,
    sigmoid,
)
from ..refine12h.models import fit_tree_model, release_tree
from .data import AdaptiveFold, CachedTickerData, apply_training_window


def _safe_auc(function, y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(function(y, p)) if len(np.unique(y)) == 2 else math.nan
    except Exception:
        return math.nan


def _top_precision(y: np.ndarray, p: np.ndarray, fraction: float) -> float:
    if len(y) == 0:
        return math.nan
    count = max(1, int(math.ceil(len(y) * fraction)))
    chosen = np.argsort(p)[-count:]
    return float(np.mean(y[chosen]))


def metric_bundle(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    raw = np.clip(np.asarray(raw, dtype=float), 1e-7, 1 - 1e-7)
    calibrated = np.clip(np.asarray(calibrated, dtype=float), 1e-7, 1 - 1e-7)
    labels = (calibrated >= float(threshold)).astype(np.int8)
    prior = float(y.mean())
    baseline = np.full(len(y), np.clip(prior, 1e-7, 1 - 1e-7), dtype=float)
    raw_pr = _safe_auc(average_precision_score, y, raw)
    raw_roc = _safe_auc(roc_auc_score, y, raw)
    cal_pr = _safe_auc(average_precision_score, y, calibrated)
    cal_roc = _safe_auc(roc_auc_score, y, calibrated)
    brier = float(brier_score_loss(y, calibrated))
    base_brier = float(brier_score_loss(y, baseline))
    ll = float(log_loss(y, calibrated, labels=[0, 1]))
    base_ll = float(log_loss(y, baseline, labels=[0, 1]))
    return {
        "rows": int(len(y)),
        "positives": int(y.sum()),
        "positive_rate": prior,
        "mean_raw_prediction": float(raw.mean()),
        "mean_calibrated_prediction": float(calibrated.mean()),
        "raw_pr_auc": raw_pr,
        "raw_roc_auc": raw_roc,
        "calibrated_pr_auc": cal_pr,
        "calibrated_roc_auc": cal_roc,
        "raw_pr_lift": float(raw_pr / prior) if prior > 0 and np.isfinite(raw_pr) else math.nan,
        "calibrated_pr_lift": float(cal_pr / prior) if prior > 0 and np.isfinite(cal_pr) else math.nan,
        "accuracy": float(accuracy_score(y, labels)),
        "balanced_accuracy": float(balanced_accuracy_score(y, labels)) if len(np.unique(y)) == 2 else math.nan,
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "f1": float(f1_score(y, labels, zero_division=0)),
        "brier": brier,
        "logloss": ll,
        "brier_skill": float(1.0 - brier / base_brier) if base_brier > 0 else math.nan,
        "logloss_skill": float(1.0 - ll / base_ll) if base_ll > 0 else math.nan,
        "top_10pct_precision": _top_precision(y, raw, 0.10),
        "top_20pct_precision": _top_precision(y, raw, 0.20),
        "label_prior_gap": float(abs(prior - calibrated.mean())),
    }


def _standardized_effect(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    positive = y == 1
    negative = ~positive
    finite = np.isfinite(X)
    pos_mean = np.nanmean(np.where(finite & positive[:, None], X, np.nan), axis=0)
    neg_mean = np.nanmean(np.where(finite & negative[:, None], X, np.nan), axis=0)
    scale = np.nanstd(X, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, np.nan)
    return (pos_mean - neg_mean) / scale


def feature_audit(X: np.ndarray, y: np.ndarray, feature_names: list[str], segments: int) -> pd.DataFrame:
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int8)
    finite = np.isfinite(X)
    missing = 1.0 - finite.mean(axis=0)
    unique = np.asarray([len(np.unique(col[np.isfinite(col)])) for col in X.T], dtype=int)
    full_effect = _standardized_effect(X, y)
    base = np.abs(full_effect)

    segment_effects: list[np.ndarray] = []
    for idx in np.array_split(np.arange(len(y)), max(3, int(segments))):
        if len(idx) < 30 or len(np.unique(y[idx])) < 2:
            segment_effects.append(np.full(X.shape[1], np.nan))
        else:
            segment_effects.append(_standardized_effect(X[idx], y[idx]))
    segment_matrix = np.vstack(segment_effects)
    median_sign = np.sign(np.nanmedian(segment_matrix, axis=0))
    sign_agreement = np.nanmean(np.sign(segment_matrix) == median_sign[None, :], axis=0)
    sign_agreement = np.nan_to_num(sign_agreement, nan=0.0)
    temporal_std = np.nanstd(segment_matrix, axis=0)

    tail = max(60, int(len(y) * 0.25))
    recent_idx = np.arange(max(0, len(y) - tail), len(y))
    recent_effect = _standardized_effect(X[recent_idx], y[recent_idx]) if len(np.unique(y[recent_idx])) == 2 else full_effect
    split = max(1, len(y) - tail)
    early_mean = np.nanmean(X[:split], axis=0)
    late_mean = np.nanmean(X[split:], axis=0)
    early_std = np.nanstd(X[:split], axis=0)
    early_std = np.where(np.isfinite(early_std) & (early_std > 1e-8), early_std, 1.0)
    drift = np.abs(late_mean - early_mean) / early_std
    drift = np.nan_to_num(drift, nan=0.0, posinf=10.0, neginf=10.0)

    stable = base * (0.55 + 0.45 * sign_agreement) / (1.0 + 0.18 * drift + 0.15 * np.nan_to_num(temporal_std))
    recent = 0.50 * base + 0.50 * np.abs(recent_effect)
    low_drift = base / (1.0 + 0.45 * drift)
    consensus = (
        0.35 * _rank01(base)
        + 0.25 * _rank01(np.abs(recent_effect))
        + 0.25 * sign_agreement
        + 0.15 * (1.0 - _rank01(drift))
    )
    return pd.DataFrame({
        "local_index": np.arange(X.shape[1], dtype=int),
        "feature": feature_names,
        "missing_ratio": missing,
        "unique_count": unique,
        "signed_effect": full_effect,
        "recent_signed_effect": recent_effect,
        "base_score": base,
        "sign_agreement": sign_agreement,
        "temporal_effect_std": temporal_std,
        "drift_score": drift,
        "standard_score": base,
        "soft_stable_score": stable,
        "recent_relevance_score": recent,
        "low_drift_score": low_drift,
        "consensus_score": consensus,
    })


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=-np.inf)
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order / max(1, len(values) - 1)


def select_features_from_audit(
    X: np.ndarray,
    audit: pd.DataFrame,
    policy: str,
    budget: int,
    plan: dict[str, Any],
) -> np.ndarray:
    score_col = f"{policy}_score"
    if score_col not in audit.columns:
        raise KeyError(score_col)
    eligible = audit.loc[
        audit["missing_ratio"].le(float(plan["screening_max_missing"]))
        & audit["unique_count"].ge(3)
        & pd.to_numeric(audit[score_col], errors="coerce").notna()
    ].sort_values(score_col, ascending=False).head(int(plan["screening_pool_features"]))
    if eligible.empty:
        raise RuntimeError("no eligible ticker features")
    candidates = eligible["local_index"].astype(int).to_numpy()
    sample_idx = np.linspace(0, len(X) - 1, min(1200, len(X)), dtype=int)
    sample = np.asarray(X[sample_idx][:, candidates], dtype=np.float64)
    medians = np.nanmedian(sample, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    sample = np.where(np.isfinite(sample), sample, medians)
    std = np.std(sample, axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    sample = (sample - np.mean(sample, axis=0)) / std
    corr = np.nan_to_num(np.corrcoef(sample, rowvar=False), nan=0.0, posinf=0.0, neginf=0.0)
    threshold = float(plan["screening_redundancy_threshold"])
    keep: list[int] = []
    for pos in range(len(candidates)):
        if any(abs(float(corr[pos, old])) >= threshold for old in keep):
            continue
        keep.append(pos)
        if len(keep) >= int(budget):
            break
    return candidates[np.asarray(keep, dtype=int)].astype(np.int32)


def _score_prediction(y: np.ndarray, p: np.ndarray, stability_penalty: float, worst_weight: float) -> dict[str, float]:
    y = np.asarray(y, dtype=np.int8)
    p = np.asarray(p, dtype=float)
    parts = np.array_split(np.arange(len(y)), 2)
    part_lifts: list[float] = []
    part_rocs: list[float] = []
    for idx in parts:
        if len(idx) < 15 or len(np.unique(y[idx])) < 2:
            continue
        prior = float(y[idx].mean())
        pr = _safe_auc(average_precision_score, y[idx], p[idx])
        roc = _safe_auc(roc_auc_score, y[idx], p[idx])
        if prior > 0 and np.isfinite(pr):
            part_lifts.append(pr / prior)
        if np.isfinite(roc):
            part_rocs.append(roc)
    prior = float(y.mean())
    pr = _safe_auc(average_precision_score, y, p)
    roc = _safe_auc(roc_auc_score, y, p)
    lift = pr / prior if prior > 0 and np.isfinite(pr) else 0.0
    if not part_lifts:
        part_lifts = [lift]
    if not part_rocs:
        part_rocs = [roc if np.isfinite(roc) else 0.5]
    score = (
        lift
        + 0.25 * float(np.nanmean(part_rocs))
        + float(worst_weight) * float(np.nanmin(part_lifts))
        - float(stability_penalty) * float(np.nanstd(part_lifts))
    )
    return {
        "score": float(score),
        "pr_auc": float(pr),
        "pr_lift": float(lift),
        "roc_auc": float(roc),
        "half_pr_lift_mean": float(np.nanmean(part_lifts)),
        "half_pr_lift_std": float(np.nanstd(part_lifts)),
        "half_pr_lift_min": float(np.nanmin(part_lifts)),
    }


def _fit_candidate(
    family: str,
    config: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    threads: int,
):
    """Use a scientifically fixed backend for each execution profile.

    CPU4: every family is forced to CPU and CUDA is hidden by the launcher.
    FULL: LightGBM stays on CPU; XGBoost and CatBoost are forced to CUDA.
    No silent backend fallback is allowed in either profile.
    """
    mode = os.environ.get("CRASHWATCH_BACKEND_MODE", "cpu_only").strip().lower()
    if mode == "cpu_only":
        backend = "cpu"
        force = "cpu"
    elif mode == "hybrid_fixed":
        backend = "cpu" if family == "lightgbm" else "cuda"
        force = backend
    else:
        raise ValueError(f"unsupported CRASHWATCH_BACKEND_MODE: {mode}")
    return fit_tree_model(
        family,
        config,
        np.ascontiguousarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int8),
        seed=seed,
        requested_backend=backend,
        threads=threads,
        force_backend=force,
    )


def _prior_adjustment(
    y_fit: np.ndarray,
    y_cal: np.ndarray,
    dates_cal: np.ndarray,
    base_cal: np.ndarray,
    base_val: np.ndarray,
    plan: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    y_cal = np.asarray(y_cal, dtype=np.int8)
    dates_cal = np.asarray(dates_cal)
    base_cal = np.clip(np.asarray(base_cal, dtype=float), 1e-7, 1 - 1e-7)
    base_val = np.clip(np.asarray(base_val, dtype=float), 1e-7, 1 - 1e-7)
    unique_dates = np.unique(dates_cal)
    split_pos = max(5, min(len(unique_dates) - 5, int(len(unique_dates) * 0.67)))
    fit_dates = set(unique_dates[:split_pos])
    early = np.asarray([value in fit_dates for value in dates_cal], dtype=bool)
    late = ~early
    if early.sum() < 30 or late.sum() < 20 or len(np.unique(y_cal[late])) < 2:
        return base_val.astype(np.float32), {"used": False, "reason": "insufficient_prior_validation"}

    recent_days = int(plan["prior_recent_days"])
    early_unique = np.unique(dates_cal[early])
    recent_dates = set(early_unique[-min(recent_days, len(early_unique)):])
    recent_mask = early & np.asarray([value in recent_dates for value in dates_cal], dtype=bool)
    long_prior = float(np.mean(y_fit))
    strength = float(plan["prior_shrinkage_strength"])
    target_prior = (float(y_cal[recent_mask].sum()) + strength * long_prior) / (float(recent_mask.sum()) + strength)
    predicted_prior = float(np.mean(base_cal[recent_mask]))
    shift = float(logit(np.asarray([target_prior]))[0] - logit(np.asarray([predicted_prior]))[0])
    shifted_late = sigmoid(logit(base_cal[late]) + shift)
    base_loss = float(log_loss(y_cal[late], base_cal[late], labels=[0, 1])) + 0.5 * float(brier_score_loss(y_cal[late], base_cal[late]))
    shifted_loss = float(log_loss(y_cal[late], shifted_late, labels=[0, 1])) + 0.5 * float(brier_score_loss(y_cal[late], shifted_late))
    if shifted_loss >= base_loss - 0.001:
        return base_val.astype(np.float32), {
            "used": False,
            "reason": "no_heldout_gain",
            "base_loss": base_loss,
            "shifted_loss": shifted_loss,
        }

    full_unique = np.unique(dates_cal)
    full_recent_dates = set(full_unique[-min(recent_days, len(full_unique)):])
    full_recent = np.asarray([value in full_recent_dates for value in dates_cal], dtype=bool)
    target_prior = (float(y_cal[full_recent].sum()) + strength * long_prior) / (float(full_recent.sum()) + strength)
    predicted_prior = float(np.mean(base_cal[full_recent]))
    final_shift = float(logit(np.asarray([target_prior]))[0] - logit(np.asarray([predicted_prior]))[0])
    output = sigmoid(logit(base_val) + final_shift)
    return output.astype(np.float32), {
        "used": True,
        "logit_shift": final_shift,
        "target_prior": target_prior,
        "predicted_prior": predicted_prior,
        "heldout_base_loss": base_loss,
        "heldout_shifted_loss": shifted_loss,
    }


def _candidate_config_payload(candidate: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    family = str(candidate["family"])
    config = {key: value for key, value in candidate.items() if key != "family"}
    return family, config



def _rolling_platt_policy(
    y: np.ndarray,
    raw_p: np.ndarray,
    dates: np.ndarray,
    plan: dict[str, Any],
) -> SafeCalibrationPolicy:
    """Select a positive-slope Platt calibrator with rolling-origin OOF checks.

    The final calibrator is fitted only on the most recent selected window. Its
    threshold and safety diagnostics come from earlier rolling holdout blocks,
    so the outer validation period never participates in calibration fitting.
    """
    y = np.asarray(y, dtype=np.int8)
    raw = np.clip(np.asarray(raw_p, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    dates = pd.to_datetime(np.asarray(dates)).to_numpy(dtype="datetime64[ns]")
    unique_dates = np.unique(dates)
    min_rows = int(plan.get("rolling_platt_min_rows", 45))
    min_pos = int(plan.get("rolling_platt_min_positives", 5))
    eval_days = int(plan.get("rolling_platt_eval_days", 20))
    block_count = int(plan.get("rolling_platt_blocks", 3))
    max_roc_drop = float(plan["calibration_max_roc_drop"])
    max_pr_drop = float(plan["calibration_max_pr_drop"])
    min_rank_corr = float(plan["calibration_min_rank_correlation"])

    def fallback(reason: str) -> SafeCalibrationPolicy:
        threshold = choose_threshold(y, raw)
        raw_roc = _safe_auc(roc_auc_score, y, raw)
        raw_pr = _safe_auc(average_precision_score, y, raw)
        return SafeCalibrationPolicy(
            method="none",
            params={"rolling_strategy": "platt_positive", "fallback_reason": reason},
            threshold=float(threshold),
            selection_score=math.inf,
            validation_brier=float(brier_score_loss(y, raw)) if len(y) else math.nan,
            validation_logloss=float(log_loss(y, raw, labels=[0, 1])) if len(y) else math.nan,
            validation_raw_roc_auc=raw_roc,
            validation_calibrated_roc_auc=raw_roc,
            validation_raw_pr_auc=raw_pr,
            validation_calibrated_pr_auc=raw_pr,
            rank_correlation=1.0,
            validation_rows=int(len(y)),
            validation_positives=int(y.sum()),
            fallback_reason=reason,
        )

    if len(y) < min_rows or len(np.unique(y)) < 2 or len(unique_dates) < eval_days * 2:
        return fallback("insufficient rolling calibration history")

    candidates: list[dict[str, Any]] = []
    for window_days in [int(value) for value in plan.get("rolling_platt_windows", [40, 60, 80, 100])]:
        oof = np.full(len(y), np.nan, dtype=np.float64)
        used_blocks = 0
        for block_offset in range(block_count, 0, -1):
            eval_end = len(unique_dates) - (block_offset - 1) * eval_days
            eval_start = eval_end - eval_days
            train_end = eval_start
            train_start = max(0, train_end - window_days)
            if train_end <= train_start or eval_start < 0:
                continue
            train_dates = unique_dates[train_start:train_end]
            eval_dates = unique_dates[eval_start:eval_end]
            train_mask = np.isin(dates, train_dates)
            eval_mask = np.isin(dates, eval_dates)
            y_train = y[train_mask]
            if (
                int(train_mask.sum()) < min_rows
                or int(eval_mask.sum()) < max(10, eval_days // 2)
                or len(np.unique(y_train)) < 2
                or int(y_train.sum()) < min_pos
                or int((1 - y_train).sum()) < min_pos
            ):
                continue
            try:
                params = fit_calibrator("platt_positive", y_train, raw[train_mask])
                oof[eval_mask] = apply_calibrator("platt_positive", params, raw[eval_mask])
                used_blocks += 1
            except Exception:
                continue
        mask = np.isfinite(oof)
        if used_blocks < 2 or int(mask.sum()) < min_rows or len(np.unique(y[mask])) < 2:
            continue
        raw_eval = raw[mask]
        cal_eval = oof[mask]
        raw_roc = _safe_auc(roc_auc_score, y[mask], raw_eval)
        cal_roc = _safe_auc(roc_auc_score, y[mask], cal_eval)
        raw_pr = _safe_auc(average_precision_score, y[mask], raw_eval)
        cal_pr = _safe_auc(average_precision_score, y[mask], cal_eval)
        correlation = spearmanr(raw_eval, cal_eval, nan_policy="omit").statistic
        correlation = float(correlation) if np.isfinite(correlation) else 0.0
        roc_drop = (raw_roc - cal_roc) if np.isfinite(raw_roc) and np.isfinite(cal_roc) else 0.0
        pr_drop = (raw_pr - cal_pr) if np.isfinite(raw_pr) and np.isfinite(cal_pr) else 0.0
        if roc_drop > max_roc_drop or pr_drop > max_pr_drop or correlation < min_rank_corr:
            continue
        brier = float(brier_score_loss(y[mask], cal_eval))
        ll = float(log_loss(y[mask], cal_eval, labels=[0, 1]))
        candidates.append({
            "window_days": window_days,
            "oof_mask": mask,
            "oof_prediction": cal_eval,
            "used_blocks": used_blocks,
            "score": ll + 0.50 * brier,
            "brier": brier,
            "logloss": ll,
            "raw_roc": raw_roc,
            "cal_roc": cal_roc,
            "raw_pr": raw_pr,
            "cal_pr": cal_pr,
            "rank_correlation": correlation,
        })

    if not candidates:
        return fallback("no rolling Platt candidate passed safety checks")
    candidates.sort(key=lambda row: (row["score"], -row["used_blocks"], row["window_days"]))
    selected = candidates[0]
    final_window = int(selected["window_days"])
    fit_dates = unique_dates[-min(final_window, len(unique_dates)):]
    fit_mask = np.isin(dates, fit_dates)
    if len(np.unique(y[fit_mask])) < 2 or int(y[fit_mask].sum()) < min_pos:
        return fallback("latest rolling Platt window has insufficient labels")
    params = fit_calibrator("platt_positive", y[fit_mask], raw[fit_mask])
    params.update({
        "rolling_strategy": "expanding_holdout_recent_window",
        "rolling_window_days": final_window,
        "rolling_eval_days": eval_days,
        "rolling_oof_blocks": int(selected["used_blocks"]),
        "rolling_oof_rows": int(np.sum(selected["oof_mask"])),
    })
    final_pred = apply_calibrator("platt_positive", params, raw)
    final_roc = _safe_auc(roc_auc_score, y, final_pred)
    final_pr = _safe_auc(average_precision_score, y, final_pred)
    base_roc = _safe_auc(roc_auc_score, y, raw)
    base_pr = _safe_auc(average_precision_score, y, raw)
    full_corr = spearmanr(raw, final_pred, nan_policy="omit").statistic
    full_corr = float(full_corr) if np.isfinite(full_corr) else 0.0
    if (
        (np.isfinite(base_roc) and np.isfinite(final_roc) and base_roc - final_roc > max_roc_drop)
        or (np.isfinite(base_pr) and np.isfinite(final_pr) and base_pr - final_pr > max_pr_drop)
        or full_corr < min_rank_corr
        or float(params.get("slope", 0.0)) <= 0.0
    ):
        return fallback("final rolling Platt failed monotonic safety guard")
    threshold = choose_threshold(y[selected["oof_mask"]], selected["oof_prediction"])
    return SafeCalibrationPolicy(
        method="platt_positive",
        params=params,
        threshold=float(threshold),
        selection_score=float(selected["score"]),
        validation_brier=float(selected["brier"]),
        validation_logloss=float(selected["logloss"]),
        validation_raw_roc_auc=float(selected["raw_roc"]),
        validation_calibrated_roc_auc=float(selected["cal_roc"]),
        validation_raw_pr_auc=float(selected["raw_pr"]),
        validation_calibrated_pr_auc=float(selected["cal_pr"]),
        rank_correlation=float(selected["rank_correlation"]),
        validation_rows=int(np.sum(selected["oof_mask"])),
        validation_positives=int(y[selected["oof_mask"]].sum()),
        fallback_reason=None,
    )


def _rank_final_recipes(
    recipe_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    plan: dict[str, Any],
) -> pd.DataFrame:
    """Rank final recipes by outer performance and stability, never alphabetically."""
    keys = ["family", "config_name", "feature_policy", "feature_budget", "training_window_days"]
    metric_columns = [
        "fold", "raw_pr_auc", "raw_pr_lift", "raw_roc_auc", "balanced_accuracy",
        "top_10pct_precision", "brier_skill", "logloss_skill",
    ]
    merged = recipe_frame.merge(fold_frame[[c for c in metric_columns if c in fold_frame.columns]], on="fold", how="left")
    aggregations: dict[str, tuple[str, str]] = {
        "selection_count": ("fold", "count"),
        "outer_pr_auc_mean": ("raw_pr_auc", "mean"),
        "outer_pr_auc_std": ("raw_pr_auc", "std"),
        "outer_pr_auc_min": ("raw_pr_auc", "min"),
        "outer_pr_lift_mean": ("raw_pr_lift", "mean"),
        "outer_roc_auc_mean": ("raw_roc_auc", "mean"),
        "outer_balanced_accuracy_mean": ("balanced_accuracy", "mean"),
        "outer_top10_precision_mean": ("top_10pct_precision", "mean"),
        "outer_brier_skill_mean": ("brier_skill", "mean"),
        "outer_logloss_skill_mean": ("logloss_skill", "mean"),
    }
    if "inner_selection_score" in merged.columns:
        aggregations["inner_selection_score_mean"] = ("inner_selection_score", "mean")
    ranked = merged.groupby(keys, dropna=False).agg(**aggregations).reset_index()
    fold_count = max(1, int(fold_frame["fold"].nunique()))
    ranked["selection_fraction"] = ranked["selection_count"] / fold_count
    ranked["outer_pr_auc_std"] = ranked["outer_pr_auc_std"].fillna(0.18)
    ranked["inner_selection_score_mean"] = pd.to_numeric(
        ranked.get("inner_selection_score_mean", pd.Series(0.0, index=ranked.index)), errors="coerce"
    ).fillna(0.0)
    ranked["final_recipe_score"] = (
        0.90 * ranked["outer_pr_auc_mean"].fillna(0.0)
        + 0.22 * ranked["outer_pr_lift_mean"].fillna(0.0)
        + 0.35 * ranked["outer_roc_auc_mean"].fillna(0.5)
        + 0.12 * ranked["outer_balanced_accuracy_mean"].fillna(0.5)
        + 0.08 * ranked["outer_top10_precision_mean"].fillna(0.0)
        + 0.16 * ranked["outer_pr_auc_min"].fillna(0.0)
        + 0.06 * ranked["inner_selection_score_mean"]
        + 0.05 * ranked["selection_fraction"]
        - 0.55 * ranked["outer_pr_auc_std"]
    )
    # Numeric/stability tie breakers only. Family/config text never determines the winner.
    ranked = ranked.sort_values(
        [
            "final_recipe_score", "selection_count", "outer_pr_auc_std",
            "outer_pr_auc_min", "inner_selection_score_mean", "feature_budget",
        ],
        ascending=[False, False, True, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["selection_rule"] = "outer_stability_score_v2_no_alphabetic_tie_break"
    return ranked


def _error_correlations(
    X: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    feature_names: list[str],
    importance: np.ndarray,
    top_n: int = 40,
) -> pd.DataFrame:
    residual = y.astype(float) - p.astype(float)
    threshold = float(np.quantile(p, 0.80))
    fp = ((p >= threshold) & (y == 0)).astype(float)
    fn = ((p < threshold) & (y == 1)).astype(float)
    order = np.argsort(importance)[::-1][: min(top_n, len(feature_names))] if len(importance) else np.arange(min(top_n, len(feature_names)))
    rows: list[dict[str, Any]] = []
    for idx in order:
        values = np.asarray(X[:, idx], dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < 20 or np.std(values[mask]) < 1e-10:
            continue
        rows.append({
            "feature": feature_names[idx],
            "importance": float(importance[idx]) if idx < len(importance) else math.nan,
            "residual_correlation": float(np.corrcoef(values[mask], residual[mask])[0, 1]),
            "false_positive_correlation": float(np.corrcoef(values[mask], fp[mask])[0, 1]) if np.std(fp[mask]) > 0 else math.nan,
            "false_negative_correlation": float(np.corrcoef(values[mask], fn[mask])[0, 1]) if np.std(fn[mask]) > 0 else math.nan,
            "missing_ratio": float(1.0 - mask.mean()),
        })
    return pd.DataFrame(rows)


def _load_existing_fold(task_dir: Path, fold_id: int) -> tuple[dict[str, Any], pd.DataFrame] | None:
    marker = task_dir / f"fold_{fold_id}_complete.json"
    metric_path = task_dir / f"fold_{fold_id}_metrics.json"
    pred_path = task_dir / f"fold_{fold_id}_predictions.parquet"
    if marker.exists() and metric_path.exists() and pred_path.exists():
        return json.loads(metric_path.read_text(encoding="utf-8")), pd.read_parquet(pred_path)
    return None


def _select_recipe_for_fold(
    data: CachedTickerData,
    fold: AdaptiveFold,
    plan: dict[str, Any],
    threads: int,
    seed: int,
    task_dir: Path,
) -> tuple[list[dict[str, Any]], int, pd.DataFrame, pd.DataFrame]:
    inner_fit = np.asarray(fold.inner_fit_idx, dtype=np.int32)
    inner_val = np.asarray(fold.inner_validation_idx, dtype=np.int32)
    audit = feature_audit(data.X[inner_fit], data.y[inner_fit], data.features, int(plan["stable_sign_segments"]))
    policy_rows: list[dict[str, Any]] = []
    proxy_predictions: dict[tuple[str, int], np.ndarray] = {}
    proxy_config = dict(plan["proxy_xgboost_config"])

    for policy in plan["feature_policies"]:
        for budget in plan["feature_budgets"]:
            selected = select_features_from_audit(data.X[inner_fit], audit, str(policy), int(budget), plan)
            model = _fit_candidate("xgboost", proxy_config, data.X[inner_fit][:, selected], data.y[inner_fit], seed, threads)
            pred = model.predict(data.X[inner_val][:, selected])
            backend = model.actual_backend
            release_tree(model)
            diagnostics = _score_prediction(
                data.y[inner_val], pred,
                float(plan["recipe_stability_penalty"]),
                float(plan["recipe_worst_fold_weight"]),
            )
            row = {
                "policy": str(policy),
                "budget": int(budget),
                "feature_count": int(len(selected)),
                "backend": backend,
                **diagnostics,
            }
            policy_rows.append(row)
            proxy_predictions[(str(policy), int(budget))] = pred
    policy_frame = pd.DataFrame(policy_rows).sort_values("score", ascending=False)
    top_policies = policy_frame.head(2)[["policy", "budget"]].to_dict("records")

    model_rows: list[dict[str, Any]] = []
    model_predictions: dict[int, np.ndarray] = {}
    model_recipes: dict[int, dict[str, Any]] = {}
    candidate_id = 0
    for policy_info in top_policies:
        policy = str(policy_info["policy"])
        budget = int(policy_info["budget"])
        selected = select_features_from_audit(data.X[inner_fit], audit, policy, budget, plan)
        for candidate in plan["model_candidates"]:
            family, config = _candidate_config_payload(dict(candidate))
            model = _fit_candidate(family, config, data.X[inner_fit][:, selected], data.y[inner_fit], seed + candidate_id * 13, threads)
            pred = model.predict(data.X[inner_val][:, selected])
            backend = model.actual_backend
            release_tree(model)
            diagnostics = _score_prediction(
                data.y[inner_val], pred,
                float(plan["recipe_stability_penalty"]),
                float(plan["recipe_worst_fold_weight"]),
            )
            recipe = {
                "candidate_id": candidate_id,
                "family": family,
                "config": config,
                "policy": policy,
                "budget": budget,
                "inner_selection_score": float(diagnostics["score"]),
                "inner_pr_auc": float(diagnostics["pr_auc"]),
                "inner_pr_lift": float(diagnostics["pr_lift"]),
                "inner_roc_auc": float(diagnostics["roc_auc"]),
                "inner_half_pr_lift_std": float(diagnostics["half_pr_lift_std"]),
                "inner_half_pr_lift_min": float(diagnostics["half_pr_lift_min"]),
            }
            model_rows.append({
                "candidate_id": candidate_id,
                "family": family,
                "config_name": str(config["name"]),
                "policy": policy,
                "budget": budget,
                "feature_count": int(len(selected)),
                "backend": backend,
                **diagnostics,
            })
            model_predictions[candidate_id] = pred
            model_recipes[candidate_id] = recipe
            candidate_id += 1
    model_frame = pd.DataFrame(model_rows).sort_values("score", ascending=False)
    best_id = int(model_frame.iloc[0]["candidate_id"])
    members = [model_recipes[best_id]]
    best_pred = model_predictions[best_id]
    best_score = _score_prediction(data.y[inner_val], best_pred, float(plan["recipe_stability_penalty"]), float(plan["recipe_worst_fold_weight"]))["score"]

    for _, row in model_frame.iloc[1:6].iterrows():
        second_id = int(row["candidate_id"])
        if model_recipes[second_id]["family"] == members[0]["family"]:
            continue
        second_pred = model_predictions[second_id]
        correlation = spearmanr(best_pred, second_pred, nan_policy="omit").statistic
        ensemble_pred = 0.5 * best_pred + 0.5 * second_pred
        ensemble_score = _score_prediction(data.y[inner_val], ensemble_pred, float(plan["recipe_stability_penalty"]), float(plan["recipe_worst_fold_weight"]))["score"]
        if np.isfinite(correlation) and correlation < 0.985 and ensemble_score >= best_score + float(plan["minimum_ensemble_gain"]):
            members.append(model_recipes[second_id])
            best_score = ensemble_score
            break

    primary = members[0]
    selected = select_features_from_audit(data.X[inner_fit], audit, primary["policy"], primary["budget"], plan)
    window_rows: list[dict[str, Any]] = []
    for window_days in plan["training_windows"]:
        train_idx = apply_training_window(inner_fit, data.dates, int(window_days))
        model = _fit_candidate(primary["family"], primary["config"], data.X[train_idx][:, selected], data.y[train_idx], seed + int(window_days), threads)
        pred = model.predict(data.X[inner_val][:, selected])
        release_tree(model)
        diagnostics = _score_prediction(data.y[inner_val], pred, float(plan["recipe_stability_penalty"]), float(plan["recipe_worst_fold_weight"]))
        window_rows.append({"window_days": int(window_days), "train_rows": int(len(train_idx)), **diagnostics})
    window_frame = pd.DataFrame(window_rows).sort_values("score", ascending=False)
    selected_window = int(window_frame.iloc[0]["window_days"])

    atomic_csv(policy_frame, task_dir / f"fold_{fold.fold_id}_policy_scout.csv")
    atomic_csv(model_frame, task_dir / f"fold_{fold.fold_id}_model_scout.csv")
    atomic_csv(window_frame, task_dir / f"fold_{fold.fold_id}_window_scout.csv")
    return members, selected_window, policy_frame, model_frame



def _build_final_development_model(
    data: CachedTickerData,
    task_dir: Path,
    plan: dict[str, Any],
    recipe_frame: pd.DataFrame,
    fold_frame: pd.DataFrame,
    threads: int,
) -> dict[str, Any]:
    model_dir = task_dir / "development_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    if recipe_frame.empty:
        return {"status": "skipped", "reason": "no_recipe_history"}
    ranked = _rank_final_recipes(recipe_frame, fold_frame, plan)
    selected_rows = [ranked.iloc[0].to_dict()]
    for _, candidate in ranked.iloc[1:].iterrows():
        if len(selected_rows) >= int(plan["maximum_ensemble_members"]):
            break
        score_gap = float(selected_rows[0]["final_recipe_score"]) - float(candidate["final_recipe_score"])
        if (
            str(candidate["family"]) != str(selected_rows[0]["family"])
            and int(candidate["selection_count"]) >= 2
            and score_gap <= 0.025
        ):
            selected_rows.append(candidate.to_dict())

    config_map = {str(item["name"]): dict(item) for item in plan["model_candidates"]}
    unique_dates = pd.DatetimeIndex(np.asarray(data.dates)).unique().sort_values()
    calibration_days = min(int(plan["calibration_days"]), max(40, len(unique_dates) // 6))
    purge = int(plan["calibration_purge_days"])
    if len(unique_dates) < calibration_days + purge + int(plan["min_fit_days"]):
        return {"status": "skipped", "reason": "insufficient_final_fit_dates"}
    calibration_dates = unique_dates[-calibration_days:]
    fit_dates = unique_dates[: -(calibration_days + purge)]
    raw_dates = np.asarray(data.dates)
    fit_idx_base = np.flatnonzero(np.isin(raw_dates, fit_dates.to_numpy())).astype(np.int32)
    cal_idx = np.flatnonzero(np.isin(raw_dates, calibration_dates.to_numpy())).astype(np.int32)
    member_cal_predictions: list[np.ndarray] = []
    member_cards: list[dict[str, Any]] = []

    for member_index, row in enumerate(selected_rows):
        family = str(row["family"])
        config_name = str(row["config_name"])
        config_source = config_map[config_name]
        config = {key: value for key, value in config_source.items() if key != "family"}
        policy = str(row["feature_policy"])
        budget = int(row["feature_budget"])
        window_days = int(row["training_window_days"])
        fit_idx = apply_training_window(fit_idx_base, data.dates, window_days)
        audit = feature_audit(data.X[fit_idx], data.y[fit_idx], data.features, int(plan["stable_sign_segments"]))
        selected = select_features_from_audit(data.X[fit_idx], audit, policy, budget, plan)
        feature_frame = pd.DataFrame({"feature_index": selected, "feature": [data.features[index] for index in selected]})
        atomic_csv(feature_frame, model_dir / f"member_{member_index}_features.csv")
        seed_predictions: list[np.ndarray] = []
        model_paths: list[str] = []
        backends: list[str] = []
        for seed in plan["seeds"]:
            model = _fit_candidate(
                family, config, data.X[fit_idx][:, selected], data.y[fit_idx],
                int(seed) + member_index * 101, threads,
            )
            seed_predictions.append(model.predict(data.X[cal_idx][:, selected]))
            base_path = model_dir / f"member_{member_index}_seed_{seed}"
            model.save(base_path)
            model_paths.append(str(base_path))
            backends.append(model.actual_backend)
            release_tree(model)
        member_cal_predictions.append(np.mean(seed_predictions, axis=0))
        member_cards.append({
            "member_index": member_index,
            "family": family,
            "config_name": config_name,
            "feature_policy": policy,
            "feature_budget": budget,
            "selected_feature_count": int(len(selected)),
            "training_window_days": window_days,
            "fit_rows": int(len(fit_idx)),
            "model_paths": model_paths,
            "backends": sorted(set(backends)),
        })

    raw_cal = np.mean(member_cal_predictions, axis=0)
    calibration_policy = _rolling_platt_policy(
        data.y[cal_idx], raw_cal, data.dates[cal_idx], plan
    )
    base_cal = apply_calibrator(calibration_policy.method, calibration_policy.params, raw_cal)
    final_cal, prior_info = _prior_adjustment(
        data.y[fit_idx_base], data.y[cal_idx], data.dates[cal_idx], base_cal, base_cal, plan
    )
    threshold = choose_threshold(data.y[cal_idx], final_cal)
    calibration_payload = {
        "base_policy": calibration_policy.payload(),
        "prior_adjustment": prior_info,
        "final_threshold": float(threshold),
    }
    atomic_json(calibration_payload, model_dir / "calibration_policy.json")
    card = {
        "ticker": data.ticker,
        "bucket": data.bucket,
        "status": "development_only",
        "warning": "최종 재학습 뒤 별도 untouched 검증이 없으므로 실전 배포용 모델이 아님",
        "ensemble_members": member_cards,
        "final_recipe_selection_rule": "outer_stability_score_v2_no_alphabetic_tie_break",
        "selected_recipe_rows": selected_rows,
        "pooled_comparison_used_for_storage": False,
        "fit_date_min": str(pd.Timestamp(raw_dates[fit_idx_base].min()).date()),
        "fit_date_max": str(pd.Timestamp(raw_dates[fit_idx_base].max()).date()),
        "calibration_date_min": str(pd.Timestamp(raw_dates[cal_idx].min()).date()),
        "calibration_date_max": str(pd.Timestamp(raw_dates[cal_idx].max()).date()),
        "calibration_rows": int(len(cal_idx)),
        "calibration_positive_rate": float(np.mean(data.y[cal_idx])),
        "calibration_metrics": metric_bundle(data.y[cal_idx], raw_cal, final_cal, threshold),
        "calibration_policy": calibration_payload,
        "calibration_strategy": "rolling_platt_positive_with_rank_safety",
    }
    atomic_json(card, model_dir / "model_card.json")
    atomic_csv(ranked, model_dir / "recipe_selection_frequency.csv")
    return {
        "status": "completed",
        "ensemble_member_count": len(member_cards),
        "model_dir": str(model_dir),
        "calibration_brier_skill": card["calibration_metrics"]["brier_skill"],
        "calibration_logloss_skill": card["calibration_metrics"]["logloss_skill"],
        "selected_family": str(selected_rows[0]["family"]),
        "selected_config_name": str(selected_rows[0]["config_name"]),
        "selected_feature_policy": str(selected_rows[0]["feature_policy"]),
        "selected_recipe_score": float(selected_rows[0]["final_recipe_score"]),
    }

def run_elite_ticker_task(
    data: CachedTickerData,
    result_dir: Path,
    plan: dict[str, Any],
    *,
    threads: int,
    deadline_epoch: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    task_dir = result_dir / "ticker_models" / data.ticker
    task_dir.mkdir(parents=True, exist_ok=True)
    if len(data.folds) < int(plan["minimum_outer_folds"]):
        summary = {
            "ticker": data.ticker,
            "bucket": data.bucket,
            "status": "skipped",
            "reason": "insufficient_feasible_folds",
            "fold_count": len(data.folds),
        }
        atomic_json(summary, task_dir / "task_summary.json")
        return summary

    fold_metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    importance_rows: list[pd.DataFrame] = []
    error_rows: list[pd.DataFrame] = []
    recipe_rows: list[dict[str, Any]] = []

    for fold in data.folds:
        existing = _load_existing_fold(task_dir, fold.fold_id)
        if existing is not None:
            metric_row, pred_frame = existing
            fold_metrics.append(metric_row)
            predictions.append(pred_frame)
            continue
        if (result_dir / "STOP_REQUESTED").exists():
            summary = {
                "ticker": data.ticker,
                "bucket": data.bucket,
                "status": "partial",
                "reason": "manual_stop_before_next_fold",
                "completed_folds": len(fold_metrics),
                "total_folds": len(data.folds),
            }
            atomic_json(summary, task_dir / "task_summary.json")
            return summary

        seed_for_tuning = int(plan["seeds"][0]) + fold.fold_id * 1009
        members, window_days, _, _ = _select_recipe_for_fold(data, fold, plan, threads, seed_for_tuning, task_dir)
        fit_idx = apply_training_window(fold.fit_idx, data.dates, window_days)
        member_cal: list[np.ndarray] = []
        member_val: list[np.ndarray] = []
        combined_importance = np.zeros(len(data.features), dtype=float)
        backend_values: list[str] = []
        member_feature_names: set[str] = set()

        for member_idx, member in enumerate(members):
            audit = feature_audit(data.X[fit_idx], data.y[fit_idx], data.features, int(plan["stable_sign_segments"]))
            selected = select_features_from_audit(data.X[fit_idx], audit, member["policy"], member["budget"], plan)
            selected_names = [data.features[index] for index in selected]
            member_feature_names.update(selected_names)
            seed_cal_predictions: list[np.ndarray] = []
            seed_val_predictions: list[np.ndarray] = []
            seed_importance: list[np.ndarray] = []
            for seed in plan["seeds"]:
                model = _fit_candidate(
                    member["family"], member["config"], data.X[fit_idx][:, selected], data.y[fit_idx],
                    int(seed) + fold.fold_id * 7919 + member_idx * 101, threads,
                )
                seed_cal_predictions.append(model.predict(data.X[fold.calibration_idx][:, selected]))
                seed_val_predictions.append(model.predict(data.X[fold.validation_idx][:, selected]))
                local_importance = model.feature_importance()
                mapped = np.zeros(len(data.features), dtype=float)
                if len(local_importance) == len(selected):
                    mapped[selected] = local_importance
                seed_importance.append(mapped)
                backend_values.append(model.actual_backend)
                release_tree(model)
            member_cal.append(np.mean(seed_cal_predictions, axis=0))
            member_val.append(np.mean(seed_val_predictions, axis=0))
            combined_importance += np.mean(seed_importance, axis=0) / max(1, len(members))

        raw_cal = np.mean(member_cal, axis=0)
        raw_val = np.mean(member_val, axis=0)
        policy = _rolling_platt_policy(
            data.y[fold.calibration_idx], raw_cal, data.dates[fold.calibration_idx], plan
        )
        base_cal = apply_calibrator(policy.method, policy.params, raw_cal)
        base_val = apply_calibrator(policy.method, policy.params, raw_val)
        calibrated_val, prior_info = _prior_adjustment(
            data.y[fit_idx], data.y[fold.calibration_idx], data.dates[fold.calibration_idx],
            base_cal, base_val, plan,
        )
        threshold = policy.threshold
        if prior_info.get("used"):
            calibrated_cal, _ = _prior_adjustment(
                data.y[fit_idx], data.y[fold.calibration_idx], data.dates[fold.calibration_idx],
                base_cal, base_cal, plan,
            )
            threshold = choose_threshold(data.y[fold.calibration_idx], calibrated_cal)
        metrics = metric_bundle(data.y[fold.validation_idx], raw_val, calibrated_val, threshold)
        metric_row = {
            "ticker": data.ticker,
            "bucket": data.bucket,
            "fold": fold.fold_id,
            "validation_days": fold.validation_days,
            "fit_rows": int(len(fit_idx)),
            "calibration_rows": int(len(fold.calibration_idx)),
            "validation_rows": int(len(fold.validation_idx)),
            "fit_date_min": fold.fit_date_min,
            "fit_date_max": fold.fit_date_max,
            "calibration_date_min": fold.calibration_date_min,
            "calibration_date_max": fold.calibration_date_max,
            "validation_date_min": fold.validation_date_min,
            "validation_date_max": fold.validation_date_max,
            "training_window_days": int(window_days),
            "ensemble_members": len(members),
            "model_families": "+".join(member["family"] for member in members),
            "model_configs": "+".join(str(member["config"]["name"]) for member in members),
            "feature_policies": "+".join(member["policy"] for member in members),
            "feature_budgets": "+".join(str(member["budget"]) for member in members),
            "selected_feature_union": int(len(member_feature_names)),
            "backend": "+".join(sorted(set(backend_values))),
            "calibration_method": policy.method,
            "calibration_strategy": "rolling_platt_positive",
            "calibration_window_days": int(policy.params.get("rolling_window_days", 0)) if policy.params else 0,
            "calibration_oof_rows": int(policy.params.get("rolling_oof_rows", 0)) if policy.params else 0,
            "calibration_threshold": float(threshold),
            "prior_adjustment_used": bool(prior_info.get("used", False)),
            "prior_logit_shift": float(prior_info.get("logit_shift", 0.0)),
            **metrics,
        }
        pred_frame = pd.DataFrame({
            "date": pd.to_datetime(np.asarray(data.dates)[fold.validation_idx]),
            "ticker": data.ticker,
            "bucket": data.bucket,
            "fold": fold.fold_id,
            "target": np.asarray(data.y)[fold.validation_idx],
            "raw_prediction": raw_val,
            "calibrated_prediction": calibrated_val,
            "threshold": threshold,
        })
        importance_frame = pd.DataFrame({
            "ticker": data.ticker,
            "fold": fold.fold_id,
            "feature": data.features,
            "importance": combined_importance,
        }).sort_values("importance", ascending=False)
        errors = _error_correlations(
            data.X[fold.validation_idx], data.y[fold.validation_idx], raw_val,
            data.features, combined_importance,
        )
        errors["ticker"] = data.ticker
        errors["fold"] = fold.fold_id
        fold_recipe_rows = []
        for member in members:
            recipe_record = {
                "ticker": data.ticker,
                "fold": fold.fold_id,
                "family": member["family"],
                "config_name": member["config"]["name"],
                "feature_policy": member["policy"],
                "feature_budget": member["budget"],
                "training_window_days": window_days,
                "inner_selection_score": member.get("inner_selection_score"),
                "inner_pr_auc": member.get("inner_pr_auc"),
                "inner_pr_lift": member.get("inner_pr_lift"),
                "inner_roc_auc": member.get("inner_roc_auc"),
                "inner_half_pr_lift_std": member.get("inner_half_pr_lift_std"),
                "inner_half_pr_lift_min": member.get("inner_half_pr_lift_min"),
            }
            recipe_rows.append(recipe_record)
            fold_recipe_rows.append(recipe_record)

        atomic_json(metric_row, task_dir / f"fold_{fold.fold_id}_metrics.json")
        atomic_parquet(pred_frame, task_dir / f"fold_{fold.fold_id}_predictions.parquet")
        atomic_csv(pd.DataFrame(fold_recipe_rows), task_dir / f"fold_{fold.fold_id}_recipe.csv")
        atomic_csv(importance_frame, task_dir / f"fold_{fold.fold_id}_importance.csv")
        atomic_csv(errors, task_dir / f"fold_{fold.fold_id}_error_correlations.csv")
        atomic_json({"completed": True, "timestamp": pd.Timestamp.now().isoformat()}, task_dir / f"fold_{fold.fold_id}_complete.json")
        fold_metrics.append(metric_row)
        predictions.append(pred_frame)
        importance_rows.append(importance_frame)
        error_rows.append(errors)
        gc.collect()

    fold_frame = pd.DataFrame(fold_metrics).sort_values("fold")
    prediction_frame = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    recipe_files = [pd.read_csv(path) for path in sorted(task_dir.glob("fold_*_recipe.csv"))]
    recipe_frame = pd.concat(recipe_files, ignore_index=True) if recipe_files else pd.DataFrame(recipe_rows)
    importance_files = [pd.read_csv(path) for path in sorted(task_dir.glob("fold_*_importance.csv"))]
    error_files = [pd.read_csv(path) for path in sorted(task_dir.glob("fold_*_error_correlations.csv"))]
    importance_all = pd.concat(importance_files, ignore_index=True) if importance_files else pd.DataFrame()
    errors_all = pd.concat(error_files, ignore_index=True) if error_files else pd.DataFrame()
    atomic_csv(fold_frame, task_dir / "fold_metrics.csv")
    atomic_parquet(prediction_frame, task_dir / "outer_predictions.parquet")
    atomic_csv(recipe_frame, task_dir / "recipe_history.csv")
    atomic_csv(importance_all, task_dir / "feature_importance_by_fold.csv")
    atomic_csv(errors_all, task_dir / "error_correlations_by_fold.csv")

    try:
        final_model = _build_final_development_model(data, task_dir, plan, recipe_frame, fold_frame, threads)
    except Exception as exc:
        final_model = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        atomic_json(final_model, task_dir / "development_model_error.json")

    summary = {
        "ticker": data.ticker,
        "bucket": data.bucket,
        "status": "completed",
        "fold_count": int(len(fold_frame)),
        "raw_pr_auc": float(fold_frame["raw_pr_auc"].mean()),
        "raw_pr_auc_std": float(fold_frame["raw_pr_auc"].std(ddof=1)) if len(fold_frame) > 1 else 0.0,
        "raw_pr_auc_min": float(fold_frame["raw_pr_auc"].min()),
        "raw_pr_lift": float(fold_frame["raw_pr_lift"].mean()),
        "raw_roc_auc": float(fold_frame["raw_roc_auc"].mean()),
        "balanced_accuracy": float(fold_frame["balanced_accuracy"].mean()),
        "brier_skill": float(fold_frame["brier_skill"].mean()),
        "logloss_skill": float(fold_frame["logloss_skill"].mean()),
        "top_10pct_precision": float(fold_frame["top_10pct_precision"].mean()),
        "fold_positive_rate_std": float(fold_frame["positive_rate"].std(ddof=1)) if len(fold_frame) > 1 else 0.0,
        "most_used_family": Counter("+".join(fold_frame["model_families"]).split("+")).most_common(1)[0][0],
        "most_used_policy": Counter("+".join(fold_frame["feature_policies"]).split("+")).most_common(1)[0][0],
        "elapsed_seconds": float(time.perf_counter() - started),
        "final_model_status": final_model.get("status"),
        "final_model_dir": final_model.get("model_dir", ""),
        "final_model_calibration_brier_skill": final_model.get("calibration_brier_skill"),
        "final_model_calibration_logloss_skill": final_model.get("calibration_logloss_skill"),
        "final_selected_family": final_model.get("selected_family"),
        "final_selected_config_name": final_model.get("selected_config_name"),
        "final_selected_feature_policy": final_model.get("selected_feature_policy"),
        "final_selected_recipe_score": final_model.get("selected_recipe_score"),
    }
    atomic_json(summary, task_dir / "task_summary.json")
    return summary
