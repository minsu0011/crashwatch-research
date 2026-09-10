from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None

from surge_precision_gate_v12 import (
    EPS,
    apply_frozen_threshold,
    combine_base_and_gate,
    safe_pr_auc,
    safe_roc_auc,
    select_frozen_dev_threshold,
)

SCHEMA = "crashwatch_surge_precision_gate_v12_7h"


@dataclass(frozen=True)
class TrialConfig:
    family_id: int
    backend: str
    candidate_quantile: float
    feature_k: int
    alpha: float
    negative_weight: float
    include_ticker_onehot: bool
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    num_leaves: int
    min_child_samples: int

    def key(self) -> str:
        raw = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ModelBundle7H:
    backend: str
    model: Any
    feature_columns: list[str]
    ticker_categories: list[str]
    ticker_onehot: bool
    config: TrialConfig
    gpu_used: bool


def _weighted_choice(rng: np.random.Generator, items: Sequence[Any], probs: Sequence[float]) -> Any:
    p = np.asarray(probs, dtype=float)
    p = p / p.sum()
    return items[int(rng.choice(len(items), p=p))]


def config_for_family(family_id: int, base_seed: int = 1701) -> TrialConfig:
    """Deterministic architecture generator; seed variants are handled separately."""
    rng = np.random.default_rng(int(base_seed) + int(family_id) * 10007)
    backend = _weighted_choice(rng, ["xgb_gpu", "lgbm", "logit"], [0.58, 0.32, 0.10])
    q = float(rng.choice([0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]))
    feature_k = int(rng.choice([0, 16, 32, 48, 64, 96, 128, 192, 256, 439]))
    if backend == "logit":
        feature_k = int(rng.choice([0, 16, 32, 64, 96]))
    alpha = float(rng.choice([0.35, 0.50, 0.65, 0.75, 0.85, 1.00]))
    neg = float(rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0]))
    depth = int(rng.choice([2, 3, 4, 5, 6]))
    lr = float(rng.choice([0.0125, 0.02, 0.03, 0.045, 0.065]))
    n_est = int(rng.choice([350, 550, 800, 1100, 1500, 2100]))
    return TrialConfig(
        family_id=int(family_id),
        backend=str(backend),
        candidate_quantile=q,
        feature_k=feature_k,
        alpha=alpha,
        negative_weight=neg,
        include_ticker_onehot=bool(rng.random() < 0.75),
        n_estimators=n_est,
        learning_rate=lr,
        max_depth=depth,
        min_child_weight=float(rng.choice([2, 5, 10, 20, 40])),
        subsample=float(rng.choice([0.60, 0.72, 0.84, 0.94, 1.00])),
        colsample_bytree=float(rng.choice([0.40, 0.55, 0.70, 0.85, 1.00])),
        reg_alpha=float(rng.choice([0.0, 0.25, 0.75, 1.5, 3.0, 6.0])),
        reg_lambda=float(rng.choice([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])),
        num_leaves=int(rng.choice([7, 11, 15, 23, 31])),
        min_child_samples=int(rng.choice([15, 25, 40, 70, 110])),
    )


def _auc_orientation(y: np.ndarray, x: np.ndarray) -> tuple[float, int, int]:
    mask = np.isfinite(x) & np.isin(y, [0, 1])
    if mask.sum() < 30 or np.unique(y[mask]).size < 2:
        return float("nan"), 0, int(mask.sum())
    try:
        auc = float(roc_auc_score(y[mask], x[mask]))
    except Exception:
        return float("nan"), 0, int(mask.sum())
    return auc, (1 if auc >= 0.5 else -1), int(mask.sum())


def rank_hard_fp_features(
    discovery: pd.DataFrame,
    raw_features: Sequence[str],
    *,
    candidate_quantile: float,
    fold_column: str = "fold_id",
    target_column: str = "target",
    rank_column: str = "base_hist_rank",
    minimum_rows: int = 40,
) -> pd.DataFrame:
    """Discovery-only A/B ranking among high-base-score candidates.

    This does not use development/confirmation/recent labels.  A feature receives a high
    score only when its A-vs-B direction is reasonably stable across discovery folds.
    """
    work = discovery.loc[pd.to_numeric(discovery[rank_column], errors="coerce") >= float(candidate_quantile)].copy()
    rows: list[dict[str, Any]] = []
    y_all = pd.to_numeric(work[target_column], errors="coerce").to_numpy(np.int8)
    folds = sorted(pd.to_numeric(work[fold_column], errors="coerce").dropna().astype(int).unique())
    for feature in raw_features:
        if feature not in work.columns:
            continue
        x_all = pd.to_numeric(work[feature], errors="coerce").to_numpy(float)
        auc_all, sign_all, n_all = _auc_orientation(y_all, x_all)
        if n_all < int(minimum_rows) or not math.isfinite(auc_all):
            continue
        fold_aucs: list[float] = []
        fold_signs: list[int] = []
        fold_ns: list[int] = []
        for fold in folds:
            mask = pd.to_numeric(work[fold_column], errors="coerce").to_numpy(float) == int(fold)
            auc_f, sign_f, n_f = _auc_orientation(y_all[mask], x_all[mask])
            if math.isfinite(auc_f):
                fold_aucs.append(float(auc_f))
                fold_signs.append(int(sign_f))
                fold_ns.append(int(n_f))
        if not fold_aucs:
            continue
        consistency = float(np.mean(np.asarray(fold_signs) == int(sign_all))) if sign_all else 0.0
        oriented = [a if sign_all > 0 else 1.0 - a for a in fold_aucs]
        global_effect = abs(float(auc_all) - 0.5) * 2.0
        median_effect = max(float(np.median(oriented)) - 0.5, 0.0) * 2.0
        worst_effect = max(float(np.min(oriented)) - 0.5, 0.0) * 2.0
        score = global_effect * (0.45 + 0.55 * consistency) + 0.35 * median_effect + 0.20 * worst_effect
        rows.append({
            "feature": str(feature),
            "auc_all": float(auc_all),
            "direction": int(sign_all),
            "rows": int(n_all),
            "folds_evaluable": int(len(fold_aucs)),
            "sign_consistency": consistency,
            "median_oriented_auc": float(np.median(oriented)),
            "worst_oriented_auc": float(np.min(oriented)),
            "score": float(score),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["score", "sign_consistency", "feature"], ascending=[False, False, True], kind="mergesort"
    ).reset_index(drop=True)


def base_gate_columns(meta: pd.DataFrame) -> list[str]:
    preferred = [
        "base_logit", "base_hist_rank", "ab_evidence_count", "ab_weighted_mean", "ab_mean",
        "ab_min", "ab_max", "ab_std", "ab_favorable_fraction", "ab_adverse_fraction",
        "ab_strong_adverse_fraction", "ticker_prior_logit", "ticker_prior_prob",
    ]
    slots = sorted([c for c in meta.columns if c.startswith("ab_slot_")])
    return [c for c in preferred if c in meta.columns] + slots


def add_discovery_ticker_prior(
    discovery: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    ticker_column: str = "ticker",
    target_column: str = "target",
    strength: float = 80.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = discovery.copy()
    e = evaluation.copy()
    y = pd.to_numeric(d[target_column], errors="coerce")
    global_rate = float(y.mean())
    stats = d.groupby(ticker_column)[target_column].agg(["sum", "count"])
    prior = (stats["sum"] + float(strength) * global_rate) / (stats["count"] + float(strength))
    prior = prior.clip(EPS, 1.0 - EPS)
    for frame in (d, e):
        p = frame[ticker_column].astype(str).map(prior).fillna(global_rate).astype(float).clip(EPS, 1.0 - EPS)
        frame["ticker_prior_prob"] = p.to_numpy(float)
        frame["ticker_prior_logit"] = np.log(p.to_numpy(float) / (1.0 - p.to_numpy(float)))
    return d, e


def matrix_columns(
    meta: pd.DataFrame,
    selected_raw_features: Sequence[str],
    *,
    include_ticker_onehot: bool,
    ticker_categories: Sequence[str],
) -> list[str]:
    cols = base_gate_columns(meta) + [str(x) for x in selected_raw_features if str(x) in meta.columns]
    # Ticker one-hot is generated separately and therefore not listed here.
    seen: set[str] = set()
    dedup: list[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            dedup.append(c)
    return dedup


def build_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    ticker_categories: Sequence[str],
    include_ticker_onehot: bool,
) -> np.ndarray:
    x = frame[list(feature_columns)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    if include_ticker_onehot:
        tick = frame["ticker"].astype(str)
        one = np.column_stack([(tick == str(t)).to_numpy(np.float32) for t in ticker_categories])
        x = np.concatenate([x, one], axis=1)
    return x


def _fit_xgb(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    config: TrialConfig,
    *,
    seed: int,
    cpu_threads: int,
    use_gpu: bool,
) -> tuple[Any, bool]:
    if xgb is None:
        raise RuntimeError("xgboost is not installed")
    params = dict(
        n_estimators=int(config.n_estimators),
        learning_rate=float(config.learning_rate),
        max_depth=int(config.max_depth),
        min_child_weight=float(config.min_child_weight),
        subsample=float(config.subsample),
        colsample_bytree=float(config.colsample_bytree),
        reg_alpha=float(config.reg_alpha),
        reg_lambda=float(config.reg_lambda),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=int(seed),
        n_jobs=max(1, int(cpu_threads)),
        verbosity=0,
    )
    if use_gpu:
        params["device"] = "cuda"
    else:
        params["device"] = "cpu"
    model = xgb.XGBClassifier(**params)
    model.fit(x, y, sample_weight=w, verbose=False)
    return model, bool(use_gpu)


def _fit_lgbm(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    config: TrialConfig,
    *,
    seed: int,
    cpu_threads: int,
) -> tuple[Any, bool]:
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")
    model = lgb.LGBMClassifier(
        n_estimators=int(config.n_estimators),
        learning_rate=float(config.learning_rate),
        num_leaves=int(config.num_leaves),
        max_depth=int(config.max_depth),
        min_child_samples=int(config.min_child_samples),
        subsample=float(config.subsample),
        colsample_bytree=float(config.colsample_bytree),
        reg_alpha=float(config.reg_alpha),
        reg_lambda=float(config.reg_lambda),
        random_state=int(seed),
        n_jobs=max(1, int(cpu_threads)),
        verbosity=-1,
    )
    model.fit(x, y, sample_weight=w)
    return model, False


def _fit_logit(
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    seed: int,
) -> tuple[Any, bool]:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=0.20, solver="liblinear", max_iter=3000, random_state=int(seed))),
    ])
    model.fit(x, y, logit__sample_weight=w)
    return model, False


def fit_trial_model(
    discovery_meta: pd.DataFrame,
    selected_raw_features: Sequence[str],
    config: TrialConfig,
    *,
    seed: int,
    cpu_threads: int,
    gpu_available: bool,
    require_gpu_for_xgb: bool = False,
) -> ModelBundle7H:
    eligible = pd.to_numeric(discovery_meta["base_hist_rank"], errors="coerce") >= float(config.candidate_quantile)
    train = discovery_meta.loc[eligible & discovery_meta["target"].isin([0, 1])].copy()
    if train.empty or train["target"].nunique() < 2:
        raise ValueError("candidate training rows have fewer than two classes")
    ticker_categories = sorted(discovery_meta["ticker"].astype(str).unique())
    columns = matrix_columns(
        discovery_meta,
        selected_raw_features,
        include_ticker_onehot=bool(config.include_ticker_onehot),
        ticker_categories=ticker_categories,
    )
    x = build_matrix(train, columns, ticker_categories=ticker_categories, include_ticker_onehot=config.include_ticker_onehot)
    y = train["target"].to_numpy(np.int8)
    w = np.where(y == 0, float(config.negative_weight), 1.0).astype(np.float32)
    if config.backend == "xgb_gpu":
        if require_gpu_for_xgb and not gpu_available:
            raise RuntimeError("xgb_gpu requested but CUDA preflight failed")
        try:
            model, gpu_used = _fit_xgb(x, y, w, config, seed=seed, cpu_threads=cpu_threads, use_gpu=gpu_available)
        except Exception:
            if require_gpu_for_xgb:
                raise
            model, gpu_used = _fit_xgb(x, y, w, config, seed=seed, cpu_threads=cpu_threads, use_gpu=False)
    elif config.backend == "lgbm":
        model, gpu_used = _fit_lgbm(x, y, w, config, seed=seed, cpu_threads=cpu_threads)
    elif config.backend == "logit":
        model, gpu_used = _fit_logit(x, y, w, seed=seed)
    else:
        raise ValueError(config.backend)
    return ModelBundle7H(
        backend=config.backend,
        model=model,
        feature_columns=columns,
        ticker_categories=ticker_categories,
        ticker_onehot=bool(config.include_ticker_onehot),
        config=config,
        gpu_used=bool(gpu_used),
    )


def predict_trial_model(bundle: ModelBundle7H, frame: pd.DataFrame) -> np.ndarray:
    x = build_matrix(
        frame,
        bundle.feature_columns,
        ticker_categories=bundle.ticker_categories,
        include_ticker_onehot=bundle.ticker_onehot,
    )
    p = np.asarray(bundle.model.predict_proba(x)[:, 1], dtype=float)
    return np.clip(p, EPS, 1.0 - EPS)


def build_prediction_frame(
    evaluation_meta: pd.DataFrame,
    gate_prob: np.ndarray,
    config: TrialConfig,
    *,
    minimum_evidence_count: int,
) -> pd.DataFrame:
    out = evaluation_meta[["ticker", "fold_id", "row_index", "source_row_id", "target", "base_score_raw"]].copy()
    out["gate_prob"] = np.asarray(gate_prob, dtype=float)
    out["base_hist_rank"] = pd.to_numeric(evaluation_meta["base_hist_rank"], errors="coerce").to_numpy(float)
    out["ab_evidence_count"] = pd.to_numeric(evaluation_meta["ab_evidence_count"], errors="coerce").fillna(0).to_numpy(float)
    out["candidate_eligible"] = (
        (out["base_hist_rank"] >= float(config.candidate_quantile))
        & (out["ab_evidence_count"] >= int(minimum_evidence_count))
    )
    out["score"] = combine_base_and_gate(out["base_score_raw"], out["gate_prob"], float(config.alpha))
    return out


def fold_metrics(pred: pd.DataFrame, folds: Iterable[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in folds:
        g = pred.loc[pred["fold_id"].eq(int(fold))].copy()
        if g.empty:
            continue
        y = g["target"].to_numpy(np.int8)
        s = g["score"].to_numpy(float)
        b = g["base_score_raw"].to_numpy(float)
        rows.append({
            "fold_id": int(fold),
            "rows": int(len(g)),
            "positives": int((y == 1).sum()),
            "base_rate": float(np.mean(y)),
            "pr_auc": safe_pr_auc(y, s),
            "base_pr_auc": safe_pr_auc(y, b),
            "pr_auc_delta": safe_pr_auc(y, s) - safe_pr_auc(y, b),
            "roc_auc": safe_roc_auc(y, s),
            "base_roc_auc": safe_roc_auc(y, b),
        })
    return pd.DataFrame(rows)



def fast_select_frozen_dev_threshold(
    predictions: pd.DataFrame,
    *,
    dev_folds: Sequence[int],
    score_column: str = "score",
    eligible_column: str = "candidate_eligible",
    target_column: str = "target",
    minimum_alerts: int = 30,
    target_precision: float = 0.70,
) -> dict[str, Any]:
    """Vectorized equivalent of the V12 frozen threshold scan.

    The same single threshold must have >= minimum_alerts in every development fold.
    """
    part = predictions.loc[predictions["fold_id"].isin([int(x) for x in dev_folds])].copy()
    eligible_scores = pd.to_numeric(part.loc[part[eligible_column].astype(bool), score_column], errors="coerce")
    eligible_scores = eligible_scores[np.isfinite(eligible_scores)]
    if eligible_scores.empty:
        return {"safe": False, "threshold": float("inf"), "reason": "NO_ELIGIBLE_SCORES", "per_fold": []}
    unique = np.unique(eligible_scores.to_numpy(float))
    if unique.size > 3000:
        unique = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, 3001)))
    thresholds = np.sort(unique)[::-1]
    fold_tables: dict[int, dict[str, np.ndarray | int]] = {}
    for fold in dev_folds:
        g = part.loc[part["fold_id"].eq(int(fold)) & part[eligible_column].astype(bool)].copy()
        s = pd.to_numeric(g[score_column], errors="coerce").to_numpy(float)
        y = pd.to_numeric(g[target_column], errors="coerce").to_numpy(np.int8)
        valid = np.isfinite(s) & np.isin(y, [0,1])
        s = s[valid]; y = y[valid]
        order = np.argsort(-s, kind="mergesort")
        s = s[order]; y = y[order]
        fold_tables[int(fold)] = {
            "scores": s,
            "cum_tp": np.cumsum(y == 1).astype(int),
            "positives": int((y == 1).sum()),
        }
    best_safe = None
    best_practical = None
    for thr in thresholds:
        per=[]; precisions=[]; recalls=[]; alerts=[]; valid_support=True
        for fold in dev_folds:
            tab=fold_tables[int(fold)]
            scores=np.asarray(tab["scores"],dtype=float)
            n=int(np.searchsorted(-scores, -float(thr), side="right")) if scores.size else 0
            tp=int(np.asarray(tab["cum_tp"])[n-1]) if n>0 else 0
            fp=n-tp
            precision=float(tp/n) if n>0 else float("nan")
            positives=int(tab["positives"]); recall=float(tp/positives) if positives>0 else float("nan")
            per.append({"fold_id":int(fold),"alerts":n,"tp":tp,"fp":fp,"precision":precision,"recall":recall})
            if n < int(minimum_alerts) or not math.isfinite(precision):
                valid_support=False; break
            precisions.append(precision); recalls.append(recall); alerts.append(n)
        if not valid_support:
            continue
        key=(min(precisions),float(np.mean(precisions)),min(recalls),float(np.mean(recalls)),-float(np.mean(alerts)))
        item=(key,float(thr),per)
        if best_practical is None or key>best_practical[0]: best_practical=item
        if all(p>=float(target_precision) for p in precisions):
            if best_safe is None or key>best_safe[0]: best_safe=item
    chosen=best_safe if best_safe is not None else best_practical
    if chosen is None:
        return {"safe":False,"threshold":float("inf"),"reason":"NO_THRESHOLD_WITH_MIN_ALERTS_EACH_DEV_FOLD","per_fold":[]}
    key,thr,per=chosen
    return {
        "safe":best_safe is not None,"threshold":thr,
        "reason":"SAFE_DEV_POLICY" if best_safe is not None else "DIAGNOSTIC_ONLY_DEV_GATE_FAIL",
        "min_dev_precision":float(key[0]),"mean_dev_precision":float(key[1]),
        "min_dev_recall":float(key[2]),"mean_dev_recall":float(key[3]),"per_fold":per,
    }

def dev_summary(
    pred: pd.DataFrame,
    *,
    dev_folds: Sequence[int],
    minimum_alerts: int,
    target_precision: float,
) -> dict[str, Any]:
    threshold = fast_select_frozen_dev_threshold(
        pred,
        dev_folds=[int(x) for x in dev_folds],
        minimum_alerts=int(minimum_alerts),
        target_precision=float(target_precision),
        score_column="score",
    )
    metrics = fold_metrics(pred, dev_folds)
    policy = apply_frozen_threshold(
        pred.rename(columns={"score": "v12_score"}),
        float(threshold["threshold"]),
        folds=[int(x) for x in dev_folds],
        score_column="v12_score",
    )
    if policy.empty:
        min_p = mean_p = 0.0
    else:
        min_p = float(policy["precision"].min())
        mean_p = float(policy["precision"].mean())
    return {
        "dev_safe": bool(threshold["safe"]),
        "threshold": float(threshold["threshold"]),
        "threshold_reason": str(threshold["reason"]),
        "min_dev_precision": min_p,
        "mean_dev_precision": mean_p,
        "min_dev_pr_auc_delta": float(metrics["pr_auc_delta"].min()) if not metrics.empty else float("nan"),
        "mean_dev_pr_auc_delta": float(metrics["pr_auc_delta"].mean()) if not metrics.empty else float("nan"),
        "dev_policy": policy.to_dict(orient="records"),
        "dev_metrics": metrics.to_dict(orient="records"),
    }


def family_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    safe = 1.0 if bool(row.get("dev_safe", False)) else 0.0
    min_p = float(row.get("min_dev_precision", 0.0) or 0.0)
    mean_p = float(row.get("mean_dev_precision", 0.0) or 0.0)
    min_d = float(row.get("min_dev_pr_auc_delta", -999.0))
    mean_d = float(row.get("mean_dev_pr_auc_delta", -999.0))
    # Penalize giant raw feature sets slightly when metrics are tied.
    k = -float(row.get("feature_k", 9999)) / 10000.0
    return safe, min_p, mean_p, min_d, mean_d, k


def stratified_bootstrap_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Bootstrap discovery rows within ticker x target strata."""
    parts: list[np.ndarray] = []
    for _, grp in frame.groupby(["ticker", "target"], sort=False):
        idx = grp.index.to_numpy()
        if idx.size:
            parts.append(rng.choice(idx, size=idx.size, replace=True))
    if not parts:
        return frame.index.to_numpy()
    out = np.concatenate(parts)
    rng.shuffle(out)
    return out


def gpu_preflight(seed: int = 17) -> dict[str, Any]:
    if xgb is None:
        return {"available": False, "reason": "xgboost_not_installed"}
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2048, 24)).astype(np.float32)
    y = (x[:, 0] + 0.3 * x[:, 1] > 0).astype(np.int8)
    try:
        model = xgb.XGBClassifier(
            n_estimators=8, max_depth=2, learning_rate=0.1,
            tree_method="hist", device="cuda", eval_metric="logloss",
            n_jobs=2, verbosity=0, random_state=seed,
        )
        model.fit(x, y, verbose=False)
        _ = model.predict_proba(x[:32])
        return {"available": True, "reason": "cuda_xgboost_fit_pass"}
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def ensemble_prediction(preds: Sequence[pd.DataFrame], config: TrialConfig) -> pd.DataFrame:
    """Average seed-level gate probabilities, then apply one fixed base/gate blend."""
    if not preds:
        raise ValueError("no predictions")
    base = preds[0].copy()
    gates = np.vstack([p["gate_prob"].to_numpy(float) for p in preds])
    gate = np.nanmean(gates, axis=0)
    base["gate_prob"] = gate
    base["score"] = combine_base_and_gate(base["base_score_raw"], gate, float(config.alpha))
    return base
