from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from surge_v13_data import TARGET_COLUMN, MOVE_COLUMN, classify_feature, date_cross_sectional_rank, grouped_historical_rank
from surge_v13_models import (
    StageSpec,
    fit_binary_model,
    predict_binary_model,
    make_sample_weights,
    safe_pr_auc,
    safe_roc_auc,
    wilson_lower,
)

SCHEMA = "crashwatch_surge_competingrisk_hardfp_v14"
DOWN_COLUMN = "label_down_3d_5pct"
UP_D1_COLUMN = "label_up_hit_d1"
UP_D2_COLUMN = "label_up_hit_d2"
UP_D3_COLUMN = "label_up_hit_d3"
FIRST_UP_DAY_COLUMN = "first_up_hit_day"
FIRST_DOWN_DAY_COLUMN = "first_down_hit_day"


def add_competing_risk_labels(frame: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
    """Add path labels from V13's past-safe reconstructed D+1..D+3 cumulative returns.

    The official surge label remains the final target. Reconstructed path labels are auxiliary
    targets only. They are never model inputs.
    """
    required = {TARGET_COLUMN, "future_cumret_d1", "future_cumret_d2", "future_cumret_d3"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"V14 competing-risk labels missing columns: {sorted(missing)}")
    out = frame.copy()
    path = np.column_stack([
        pd.to_numeric(out["future_cumret_d1"], errors="coerce").to_numpy(float),
        pd.to_numeric(out["future_cumret_d2"], errors="coerce").to_numpy(float),
        pd.to_numeric(out["future_cumret_d3"], errors="coerce").to_numpy(float),
    ])
    valid = np.all(np.isfinite(path), axis=1)
    up_hits = path >= float(threshold)
    down_hits = path <= -float(threshold)
    out[UP_D1_COLUMN] = np.where(valid, up_hits[:, 0].astype(np.int8), -1)
    out[UP_D2_COLUMN] = np.where(valid, np.any(up_hits[:, :2], axis=1).astype(np.int8), -1)
    # D3 final target uses the official source of truth instead of reconstructed label.
    official = pd.to_numeric(out[TARGET_COLUMN], errors="coerce").to_numpy(float)
    out[UP_D3_COLUMN] = np.where(np.isfinite(official), official.astype(np.int8), -1)
    out[DOWN_COLUMN] = np.where(valid, np.any(down_hits, axis=1).astype(np.int8), -1)

    first_up = np.zeros(len(out), dtype=np.int8)
    first_down = np.zeros(len(out), dtype=np.int8)
    for day in range(3):
        m = valid & (first_up == 0) & up_hits[:, day]
        first_up[m] = day + 1
        m = valid & (first_down == 0) & down_hits[:, day]
        first_down[m] = day + 1
    # Keep the official surge target as source of truth for the small set of
    # reconstruction disagreements. If the official target is positive but the
    # reconstructed path never crosses, assign the conservative latest day (D3)
    # to the auxiliary hazard target instead of silently turning it negative.
    official_pos = valid & np.isfinite(official) & (official == 1) & (first_up == 0)
    first_up[official_pos] = 3
    first_up[~valid] = -1
    first_down[~valid] = -1
    out[FIRST_UP_DAY_COLUMN] = first_up
    out[FIRST_DOWN_DAY_COLUMN] = first_down
    out["label_up_before_down"] = np.where(
        valid,
        ((first_up > 0) & ((first_down == 0) | (first_up <= first_down))).astype(np.int8),
        -1,
    )
    out["label_down_before_up"] = np.where(
        valid,
        ((first_down > 0) & ((first_up == 0) | (first_down < first_up))).astype(np.int8),
        -1,
    )
    return out


def _auc_with_fixed_direction(y: np.ndarray, x: np.ndarray, direction: int, minimum_rows: int) -> tuple[float, int]:
    mask = np.isfinite(y) & np.isfinite(x) & np.isin(y, [0, 1])
    n = int(mask.sum())
    if n < int(minimum_rows) or np.unique(y[mask]).size < 2:
        return float("nan"), n
    auc = safe_roc_auc(y[mask], x[mask])
    if not math.isfinite(auc):
        return float("nan"), n
    return float(auc if direction > 0 else 1.0 - auc), n


def _fit_direction(y: np.ndarray, x: np.ndarray, minimum_rows: int) -> tuple[int, float, int]:
    mask = np.isfinite(y) & np.isfinite(x) & np.isin(y, [0, 1])
    n = int(mask.sum())
    if n < int(minimum_rows) or np.unique(y[mask]).size < 2:
        return 0, float("nan"), n
    auc = safe_roc_auc(y[mask], x[mask])
    if not math.isfinite(auc):
        return 0, float("nan"), n
    direction = 1 if auc >= 0.5 else -1
    oriented = auc if direction > 0 else 1.0 - auc
    return direction, float(oriented), n


def crossfit_rank_features(
    discovery: pd.DataFrame,
    features: Sequence[str],
    *,
    target_column: str,
    folds: Sequence[int] = (0, 1, 2),
    mask: Sequence[bool] | None = None,
    allowed_families: set[str] | None = None,
    minimum_total_rows: int = 240,
    minimum_holdout_rows: int = 60,
    minimum_coverage: float = 0.50,
    minimum_fold_coverage: float = 0.40,
    minimum_worst_auc: float = 0.49,
) -> pd.DataFrame:
    """Fold-held-out univariate ranking.

    Feature direction is fitted on the other discovery folds and evaluated on the held-out fold.
    This removes V13's same-fold feature-selection optimism. Sparse features are explicitly gated.
    """
    if "fold_id" not in discovery:
        raise ValueError("crossfit_rank_features requires fold_id")
    y_all = pd.to_numeric(discovery[target_column], errors="coerce").to_numpy(float)
    fold_values = pd.to_numeric(discovery["fold_id"], errors="coerce").to_numpy(float)
    base_mask = np.isfinite(y_all) & np.isin(y_all, [0, 1])
    if mask is not None:
        base_mask &= np.asarray(mask, dtype=bool)
    folds = [int(f) for f in folds]
    rows: list[dict[str, Any]] = []
    denom_total = max(int(base_mask.sum()), 1)
    fold_denoms = {f: max(int(np.sum(base_mask & (fold_values == f))), 1) for f in folds}

    for feature in features:
        family = classify_feature(str(feature))
        if family == "metadata":
            continue
        if allowed_families is not None and family not in allowed_families:
            continue
        x = pd.to_numeric(discovery[feature], errors="coerce").to_numpy(float)
        available = base_mask & np.isfinite(x)
        n_total = int(available.sum())
        coverage = n_total / denom_total
        if n_total < int(minimum_total_rows) or coverage < float(minimum_coverage):
            continue
        fold_coverage: list[float] = []
        held_auc: list[float] = []
        held_n: list[int] = []
        directions: list[int] = []
        train_auc: list[float] = []
        valid = True
        for f in folds:
            valmask = base_mask & (fold_values == f)
            trmask = base_mask & np.isin(fold_values, [g for g in folds if g != f])
            fcov = int(np.sum(valmask & np.isfinite(x))) / fold_denoms[f]
            fold_coverage.append(float(fcov))
            if fcov < float(minimum_fold_coverage):
                valid = False
                break
            direction, tr_auc, _ = _fit_direction(y_all[trmask], x[trmask], max(minimum_holdout_rows, 80))
            if direction == 0:
                valid = False
                break
            va_auc, va_n = _auc_with_fixed_direction(y_all[valmask], x[valmask], direction, minimum_holdout_rows)
            if not math.isfinite(va_auc):
                valid = False
                break
            directions.append(direction)
            train_auc.append(tr_auc)
            held_auc.append(va_auc)
            held_n.append(va_n)
        if not valid or len(held_auc) != len(folds):
            continue
        median_auc = float(np.median(held_auc))
        worst_auc = float(np.min(held_auc))
        if worst_auc < float(minimum_worst_auc):
            continue
        consistency = float(max(directions.count(1), directions.count(-1)) / len(directions))
        support_shrink = math.sqrt(n_total / (n_total + 500.0))
        # Reward out-of-fold effect, punish fragile fold spread and sparse support.
        effect = max(median_auc - 0.5, 0.0)
        worst_effect = max(worst_auc - 0.5, -0.05)
        spread = float(np.std(held_auc))
        score = support_shrink * (2.0 * effect + 1.25 * worst_effect + 0.30 * max(consistency - 2/3, 0.0) - 0.75 * spread)
        rows.append({
            "feature": str(feature), "family": family, "rows": n_total, "coverage": coverage,
            "minimum_fold_coverage": float(min(fold_coverage)), "direction_consistency": consistency,
            "median_holdout_auc": median_auc, "worst_holdout_auc": worst_auc,
            "mean_holdout_auc": float(np.mean(held_auc)), "holdout_auc_std": spread,
            "mean_train_oriented_auc": float(np.mean(train_auc)), "minimum_holdout_rows": int(min(held_n)),
            "crossfit_score": float(score),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["crossfit_score", "worst_holdout_auc", "coverage", "feature"],
        ascending=[False, False, False, True], kind="mergesort",
    ).reset_index(drop=True)


def feature_stem(feature: str) -> str:
    s = str(feature)
    if "__" in s:
        s = s.split("__", 1)[0]
    for suffix in ["_adj"]:
        s = s.replace(suffix, "")
    return s


def deduplicate_ranked_features(
    discovery: pd.DataFrame,
    ranking: pd.DataFrame,
    *,
    max_features: int,
    corr_threshold: float = 0.965,
    max_per_stem: int = 2,
    sample_rows: int = 8000,
) -> tuple[list[str], pd.DataFrame]:
    if ranking.empty or max_features <= 0:
        return [], pd.DataFrame(columns=["feature", "kept", "reason"])
    candidates = ranking["feature"].astype(str).tolist()
    sample = discovery[candidates].copy()
    if len(sample) > sample_rows:
        # deterministic temporal spread, not random sampling
        idx = np.linspace(0, len(sample) - 1, sample_rows).astype(int)
        sample = sample.iloc[idx].copy()
    selected: list[str] = []
    stem_count: dict[str, int] = {}
    decisions: list[dict[str, Any]] = []
    cached: dict[str, np.ndarray] = {}

    def vec(col: str) -> np.ndarray:
        if col not in cached:
            a = pd.to_numeric(sample[col], errors="coerce").to_numpy(float)
            med = float(np.nanmedian(a)) if np.isfinite(a).any() else 0.0
            a = np.where(np.isfinite(a), a, med)
            sd = float(np.std(a))
            cached[col] = (a - float(np.mean(a))) / (sd if sd > 1e-8 else 1.0)
        return cached[col]

    for feature in candidates:
        if len(selected) >= int(max_features):
            decisions.append({"feature": feature, "kept": False, "reason": "MAX_FEATURES"})
            continue
        stem = feature_stem(feature)
        if stem_count.get(stem, 0) >= int(max_per_stem):
            decisions.append({"feature": feature, "kept": False, "reason": "STEM_LIMIT"})
            continue
        v = vec(feature)
        reject = None
        for prior in selected:
            p = vec(prior)
            corr = float(np.dot(v, p) / max(len(v), 1))
            if abs(corr) >= float(corr_threshold):
                reject = f"CORRELATED:{prior}:{corr:.4f}"
                break
        if reject:
            decisions.append({"feature": feature, "kept": False, "reason": reject})
            continue
        selected.append(feature)
        stem_count[stem] = stem_count.get(stem, 0) + 1
        decisions.append({"feature": feature, "kept": True, "reason": "KEPT"})
    return selected, pd.DataFrame(decisions)


def magnitude_balancing_weights(
    train: pd.DataFrame,
    *,
    target_column: str,
    magnitude_column: str = "future_abs_excursion_3d",
    bins: int = 8,
    half_life_days: float = 756.0,
) -> np.ndarray:
    """Keep all large-move rows and balance classes within outcome-magnitude strata.

    This replaces V13's one-positive/one-negative duplication. Future magnitude is used only
    for training weights, never as a feature or inference input.
    """
    y = pd.to_numeric(train[target_column], errors="raise").to_numpy(np.int8)
    mag = pd.to_numeric(train[magnitude_column], errors="coerce")
    base = make_sample_weights(train, y, half_life_days=half_life_days)
    finite = mag.notna()
    if finite.sum() < 20:
        return base
    try:
        q = pd.qcut(mag.rank(method="first"), q=min(int(bins), int(finite.sum())), labels=False, duplicates="drop")
    except Exception:
        return base
    extra = np.ones(len(train), dtype=float)
    tmp = pd.DataFrame({"bin": q, "y": y})
    for _, idx in tmp.groupby("bin", dropna=True).groups.items():
        ii = np.asarray(list(idx), dtype=int)
        yy = y[ii]
        n0 = max(int(np.sum(yy == 0)), 1)
        n1 = max(int(np.sum(yy == 1)), 1)
        target_n = (n0 + n1) / 2.0
        extra[ii[yy == 0]] *= target_n / n0
        extra[ii[yy == 1]] *= target_n / n1
    w = base * extra
    w = np.clip(w, 0.05, np.nanpercentile(w, 99.5) if np.isfinite(w).any() else 10.0)
    return w / max(float(np.mean(w)), 1e-8)


@dataclass
class NonNegativeStacker:
    intercept: float
    weights: np.ndarray
    columns: list[str]
    medians: np.ndarray
    means: np.ndarray
    scales: np.ndarray


def _softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def fit_nonnegative_stacker(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    target_column: str = TARGET_COLUMN,
    sample_weight: Sequence[float] | None = None,
    l2: float = 0.20,
) -> NonNegativeStacker:
    cols = list(columns)
    x = frame[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    med = np.nanmedian(x, axis=0)
    med = np.where(np.isfinite(med), med, 0.5)
    x = np.where(np.isfinite(x), x, med)
    mean = np.mean(x, axis=0)
    scale = np.std(x, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    z = (x - mean) / scale
    y = pd.to_numeric(frame[target_column], errors="raise").to_numpy(float)
    if sample_weight is None:
        sw = np.ones(len(frame), dtype=float)
    else:
        sw = np.asarray(sample_weight, dtype=float)
    sw = sw / max(float(np.mean(sw)), 1e-8)

    def loss(theta: np.ndarray) -> tuple[float, np.ndarray]:
        b = theta[0]
        w = _softplus(theta[1:])
        lin = b + z @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(lin, -35, 35)))
        eps = 1e-9
        val = -np.mean(sw * (y * np.log(p + eps) + (1-y) * np.log(1-p + eps))) + float(l2) * float(np.sum(w*w))
        # analytic gradient through softplus
        err = sw * (p - y) / len(y)
        gb = float(np.sum(err))
        gw = z.T @ err + 2.0 * float(l2) * w
        sigmoid_theta = 1.0 / (1.0 + np.exp(-np.clip(theta[1:], -35, 35)))
        gt = gw * sigmoid_theta
        return float(val), np.r_[gb, gt]

    init = np.r_[math.log(max(float(np.mean(y)), 1e-3) / max(1.0-float(np.mean(y)), 1e-3)), np.full(len(cols), -1.5)]
    res = minimize(lambda t: loss(t), init, jac=True, method="L-BFGS-B", options={"maxiter": 1200, "ftol": 1e-10})
    theta = res.x
    return NonNegativeStacker(float(theta[0]), _softplus(theta[1:]), cols, med, mean, scale)


def predict_nonnegative_stacker(model: NonNegativeStacker, frame: pd.DataFrame) -> np.ndarray:
    x = frame[model.columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x = np.where(np.isfinite(x), x, model.medians)
    z = (x - model.means) / model.scales
    lin = model.intercept + z @ model.weights
    return np.clip(1.0 / (1.0 + np.exp(-np.clip(lin, -35, 35))), 1e-8, 1-1e-8)


def build_ranked_policy_score(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    probability_column: str,
    raw_weight: float = 0.45,
    historical_weight: float = 0.40,
    date_weight: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cur = current[["date", "ticker", probability_column]].copy()
    ref = reference[["ticker", probability_column]].copy() if not reference.empty else pd.DataFrame(columns=["ticker", probability_column])
    hist = grouped_historical_rank(ref, cur, score_column=probability_column, group_column="ticker", minimum_group_rows=30) if not ref.empty else np.full(len(cur), np.nan)
    if not ref.empty:
        glob = np.sort(pd.to_numeric(ref[probability_column], errors="coerce").dropna().to_numpy(float))
        vals = pd.to_numeric(cur[probability_column], errors="coerce").to_numpy(float)
        gr = np.searchsorted(glob, vals, side="right") / max(len(glob), 1)
        hist = np.where(np.isfinite(hist), hist, gr)
    else:
        hist = pd.to_numeric(cur[probability_column], errors="coerce").fillna(0.5).to_numpy(float)
    date_rank = date_cross_sectional_rank(cur, probability_column)
    raw = pd.to_numeric(cur[probability_column], errors="coerce").fillna(0.5).to_numpy(float)
    score = raw_weight*raw + historical_weight*np.where(np.isfinite(hist), hist, 0.5) + date_weight*np.where(np.isfinite(date_rank), date_rank, 0.5)
    return np.clip(score, 0, 1), np.clip(hist, 0, 1), np.clip(date_rank, 0, 1)


def hardfp_candidate_mask(frame: pd.DataFrame, score_column: str, quantile: float = 0.75) -> np.ndarray:
    work = frame[["ticker", "fold_id", score_column]].copy()
    ranks = work.groupby(["ticker", "fold_id"], sort=False)[score_column].rank(pct=True, method="average")
    return (ranks >= float(quantile)).to_numpy(bool)


def select_threshold(
    predictions: pd.DataFrame,
    *,
    folds: Sequence[int],
    score_column: str,
    target_column: str = TARGET_COLUMN,
    minimum_alerts: int = 30,
    target_precision: float = 0.70,
) -> dict[str, Any]:
    subset = predictions.loc[predictions["fold_id"].isin([int(x) for x in folds])].copy()
    values = np.sort(pd.to_numeric(subset[score_column], errors="coerce").dropna().unique())
    if not len(values):
        raise ValueError("No score values for threshold selection")
    # evaluate only meaningful upper-tail cut points
    qs = np.linspace(0.50, 0.999, 500)
    candidates = np.unique(np.quantile(values, qs))
    best = None
    safe = None
    for threshold in candidates:
        fold_rows = []
        valid = True
        for f in folds:
            g = subset.loc[subset["fold_id"].eq(int(f))]
            alert = pd.to_numeric(g[score_column], errors="coerce").to_numpy(float) >= threshold
            n = int(alert.sum())
            tp = int(pd.to_numeric(g.loc[alert, target_column], errors="coerce").eq(1).sum()) if n else 0
            precision = tp / n if n else 0.0
            fold_rows.append((f, n, tp, precision, wilson_lower(tp, n)))
            if n < int(minimum_alerts):
                valid = False
        min_p = min(x[3] for x in fold_rows)
        mean_p = float(np.mean([x[3] for x in fold_rows]))
        min_w = min(x[4] for x in fold_rows)
        key = (min_p, min_w, mean_p, threshold)
        record = {"threshold": float(threshold), "fold_rows": fold_rows, "minimum_precision": min_p, "mean_precision": mean_p, "minimum_wilson": min_w, "valid_alerts": valid}
        if valid and min_p >= float(target_precision):
            if safe is None or key > safe[0]: safe = (key, record)
        if valid and (best is None or key > best[0]): best = (key, record)
    chosen = safe[1] if safe is not None else (best[1] if best is not None else None)
    if chosen is None:
        # fallback threshold producing at least minimum alerts globally
        threshold = float(np.quantile(values, max(0.0, 1 - minimum_alerts / max(len(values),1))))
        chosen = {"threshold": threshold, "fold_rows": [], "minimum_precision": 0.0, "mean_precision": 0.0, "minimum_wilson": 0.0, "valid_alerts": False}
    chosen["safe"] = bool(safe is not None)
    chosen["reason"] = "SAFE" if safe is not None else "DIAGNOSTIC_BEST_MIN_PRECISION"
    return chosen


def policy_metrics(predictions: pd.DataFrame, threshold: float, *, score_column: str = "policy_score") -> pd.DataFrame:
    rows = []
    for fold, g in predictions.groupby("fold_id", sort=True):
        score = pd.to_numeric(g[score_column], errors="coerce").to_numpy(float)
        y = pd.to_numeric(g[TARGET_COLUMN], errors="coerce").to_numpy(float)
        alert = score >= float(threshold)
        n = int(alert.sum()); tp = int(np.sum((y == 1) & alert))
        rows.append({
            "fold_id": int(fold), "rows": int(len(g)), "positive": int(np.sum(y==1)), "base_rate": float(np.mean(y)),
            "alerts": n, "true_positive": tp, "precision": float(tp/n if n else 0.0),
            "wilson_lower_95": wilson_lower(tp,n), "alert_rate": float(n/max(len(g),1)),
            "pr_auc": safe_pr_auc(y, score), "roc_auc": safe_roc_auc(y, score),
        })
    return pd.DataFrame(rows)


def oracle_topk(predictions: pd.DataFrame, *, minimum_alerts: int = 30, score_column: str = "policy_score") -> pd.DataFrame:
    rows=[]
    for fold,g in predictions.groupby("fold_id",sort=True):
        s=pd.to_numeric(g[score_column],errors="coerce").to_numpy(float); y=pd.to_numeric(g[TARGET_COLUMN],errors="coerce").to_numpy(int)
        order=np.argsort(-s,kind="mergesort"); yy=y[order]; ss=s[order]
        if len(g)<minimum_alerts: continue
        cum=np.cumsum(yy==1); ks=np.arange(1,len(g)+1); p=cum/ks; p[:minimum_alerts-1]=-1
        k=int(np.argmax(p))+1
        rows.append({"fold_id":int(fold),"rows":len(g),"best_k":k,"best_precision":float(p[k-1]),"true_positive":int(cum[k-1]),"score_cutoff":float(ss[k-1])})
    return pd.DataFrame(rows)
