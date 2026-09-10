from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None
try:
    import xgboost as xgb  # type: ignore
except Exception:  # pragma: no cover
    xgb = None

from surge_v13_data import (
    ABSpec,
    DIRECTION_COLUMN,
    MOVE_COLUMN,
    TARGET_COLUMN,
    classify_feature,
    date_cross_sectional_rank,
    grouped_historical_rank,
)

EPS = 1e-8
SCHEMA = "crashwatch_surge_magnitude_direction_v13"


@dataclass(frozen=True)
class StageSpec:
    backend: str
    feature_k: int
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: float = 5.0
    subsample: float = 0.82
    colsample_bytree: float = 0.72
    reg_alpha: float = 0.5
    reg_lambda: float = 4.0
    num_leaves: int = 31
    min_child_samples: int = 30


@dataclass(frozen=True)
class V13Config:
    config_id: str
    move: StageSpec
    direction: StageSpec
    include_ab: bool
    include_v11_lag0: bool
    include_v11_self: bool
    base_rank_blend: float
    recency_half_life_days: float
    match_negative_reuse: int
    ticker_onehot: bool = True
    market_bucket_onehot: bool = True

    def key(self) -> str:
        raw = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ModelBundle:
    backend: str
    model: Any
    feature_columns: list[str]
    medians: np.ndarray
    centers: np.ndarray | None
    scales: np.ndarray | None
    ticker_categories: list[str]
    market_categories: list[str]
    bucket_categories: list[str]
    ticker_onehot: bool
    market_bucket_onehot: bool
    gpu_used: bool


@dataclass
class PolicySelection:
    threshold: float
    safe_on_discovery: bool
    reason: str
    minimum_precision: float
    mean_precision: float
    minimum_wilson_lower: float
    minimum_alerts_observed: int
    fold_metrics: pd.DataFrame


def predefined_configs(*, fast: bool = False) -> list[V13Config]:
    """Small, predeclared architecture set; no random thousands-family search."""
    xgb_move = dict(backend="xgb", n_estimators=220 if fast else 1100, learning_rate=0.04, max_depth=5)
    xgb_dir = dict(backend="xgb", n_estimators=180 if fast else 850, learning_rate=0.035, max_depth=4)
    lgb_move = dict(backend="lgbm", n_estimators=220 if fast else 1000, learning_rate=0.035, max_depth=-1, num_leaves=31)
    lgb_dir = dict(backend="lgbm", n_estimators=180 if fast else 800, learning_rate=0.035, max_depth=-1, num_leaves=23)
    logit_dir = dict(backend="logit", n_estimators=1, learning_rate=0.0, max_depth=0)
    return [
        V13Config("C01_XGB_64_48_AB_NET_SELF", StageSpec(feature_k=64, **xgb_move), StageSpec(feature_k=48, **xgb_dir), True, True, True, 0.00, 756, 3),
        V13Config("C02_XGB_128_64_AB_NET_SELF", StageSpec(feature_k=128, **xgb_move), StageSpec(feature_k=64, **xgb_dir), True, True, True, 0.00, 756, 3),
        V13Config("C03_XGB_192_96_AB_NET_SELF", StageSpec(feature_k=192, **xgb_move), StageSpec(feature_k=96, **xgb_dir), True, True, True, 0.00, 504, 3),
        V13Config("C04_XGB_128_64_AB_ONLY", StageSpec(feature_k=128, **xgb_move), StageSpec(feature_k=64, **xgb_dir), True, False, False, 0.00, 756, 3),
        V13Config("C05_XGB_128_64_NET_SELF_ONLY", StageSpec(feature_k=128, **xgb_move), StageSpec(feature_k=64, **xgb_dir), False, True, True, 0.00, 756, 3),
        V13Config("C06_XGB_128_64_AB_NET_BASE10", StageSpec(feature_k=128, **xgb_move), StageSpec(feature_k=64, **xgb_dir), True, True, True, 0.10, 756, 3),
        V13Config("C07_LGB_128_64_AB_NET_SELF", StageSpec(feature_k=128, **lgb_move), StageSpec(feature_k=64, **lgb_dir), True, True, True, 0.00, 756, 3),
        V13Config("C08_XGB_LOGIT64_AB_NET_SELF", StageSpec(feature_k=128, **xgb_move), StageSpec(feature_k=64, **logit_dir), True, True, True, 0.00, 756, 3),
        V13Config("C09_LGB_192_96_AB_NET_SELF", StageSpec(feature_k=192, **lgb_move), StageSpec(feature_k=96, **lgb_dir), True, True, True, 0.00, 504, 3),
        V13Config("C10_XGB_256_128_AB_NET_SELF", StageSpec(feature_k=256, **xgb_move), StageSpec(feature_k=128, **xgb_dir), True, True, True, 0.00, 504, 2),
    ]


def safe_pr_auc(y: Sequence[int], score: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=float)
    ss = np.asarray(score, dtype=float)
    mask = np.isfinite(yy) & np.isfinite(ss) & np.isin(yy, [0, 1])
    if mask.sum() < 2 or np.unique(yy[mask]).size < 2:
        return float("nan")
    try:
        return float(average_precision_score(yy[mask], ss[mask]))
    except Exception:
        return float("nan")


def safe_roc_auc(y: Sequence[int], score: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=float)
    ss = np.asarray(score, dtype=float)
    mask = np.isfinite(yy) & np.isfinite(ss) & np.isin(yy, [0, 1])
    if mask.sum() < 2 or np.unique(yy[mask]).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(yy[mask], ss[mask]))
    except Exception:
        return float("nan")


def wilson_lower(successes: int, trials: int, z: float = 1.96) -> float:
    if trials <= 0:
        return 0.0
    p = successes / trials
    denom = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
    return float(max((center - margin) / denom, 0.0))


def _oriented_auc(y: np.ndarray, x: np.ndarray, minimum_rows: int = 30) -> tuple[float, int, int]:
    mask = np.isfinite(y) & np.isfinite(x) & np.isin(y, [0, 1])
    if mask.sum() < int(minimum_rows) or np.unique(y[mask]).size < 2:
        return float("nan"), 0, int(mask.sum())
    auc = safe_roc_auc(y[mask], x[mask])
    if not math.isfinite(auc):
        return float("nan"), 0, int(mask.sum())
    return float(auc), (1 if auc >= 0.5 else -1), int(mask.sum())


def rank_move_features(
    discovery_validation: pd.DataFrame,
    raw_features: Sequence[str],
    *,
    fold_column: str = "fold_id",
    minimum_rows: int = 80,
) -> pd.DataFrame:
    """Rank symmetric large-move features using discovery folds only."""
    y = pd.to_numeric(discovery_validation[MOVE_COLUMN], errors="coerce").to_numpy(float)
    folds = sorted(pd.to_numeric(discovery_validation[fold_column], errors="coerce").dropna().astype(int).unique())
    fold_values = pd.to_numeric(discovery_validation[fold_column], errors="coerce").to_numpy(float)
    rows: list[dict[str, Any]] = []
    for feature in raw_features:
        family = classify_feature(feature)
        if family in {"metadata", "direction"}:
            continue
        x = pd.to_numeric(discovery_validation[feature], errors="coerce").to_numpy(float)
        auc, sign, n = _oriented_auc(y, x, minimum_rows)
        if not math.isfinite(auc):
            continue
        oriented_folds: list[float] = []
        signs: list[int] = []
        for fold in folds:
            mask = fold_values == fold
            fa, fs, _ = _oriented_auc(y[mask], x[mask], max(25, minimum_rows // 3))
            if math.isfinite(fa):
                oriented_folds.append(fa if sign > 0 else 1.0 - fa)
                signs.append(fs)
        if not oriented_folds:
            continue
        consistency = float(np.mean(np.asarray(signs) == sign))
        effect = abs(auc - 0.5) * 2.0
        median_effect = max(float(np.median(oriented_folds)) - 0.5, 0.0) * 2.0
        worst_effect = max(float(np.min(oriented_folds)) - 0.5, 0.0) * 2.0
        family_bonus = 0.08 if family == "magnitude" else 0.0
        score = effect * (0.45 + 0.55 * consistency) + 0.35 * median_effect + 0.25 * worst_effect + family_bonus
        rows.append({
            "feature": feature, "family": family, "global_auc": auc, "direction": sign,
            "rows": n, "folds_evaluable": len(oriented_folds), "sign_consistency": consistency,
            "median_oriented_auc": float(np.median(oriented_folds)),
            "worst_oriented_auc": float(np.min(oriented_folds)), "move_score": float(score),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["move_score", "sign_consistency", "feature"], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)


def rank_direction_features(
    discovery_validation: pd.DataFrame,
    raw_features: Sequence[str],
    *,
    fold_column: str = "fold_id",
    minimum_rows: int = 50,
    magnitude_penalty: float = 0.80,
) -> pd.DataFrame:
    """Rank direction features while penalizing symmetric-magnitude shortcuts."""
    all_move = pd.to_numeric(discovery_validation[MOVE_COLUMN], errors="coerce").to_numpy(float)
    large = all_move == 1
    y_dir = pd.to_numeric(discovery_validation[DIRECTION_COLUMN], errors="coerce").to_numpy(float)
    fold_values = pd.to_numeric(discovery_validation[fold_column], errors="coerce").to_numpy(float)
    folds = sorted(pd.Series(fold_values[large]).dropna().astype(int).unique())
    rows: list[dict[str, Any]] = []
    for feature in raw_features:
        family = classify_feature(feature)
        if family in {"metadata", "magnitude"}:
            continue
        x = pd.to_numeric(discovery_validation[feature], errors="coerce").to_numpy(float)
        dir_auc, sign, n = _oriented_auc(y_dir[large], x[large], minimum_rows)
        if not math.isfinite(dir_auc):
            continue
        move_auc, _, _ = _oriented_auc(all_move, x, max(minimum_rows, 80))
        move_effect = abs(move_auc - 0.5) * 2.0 if math.isfinite(move_auc) else 0.0
        oriented_folds: list[float] = []
        signs: list[int] = []
        for fold in folds:
            mask = large & (fold_values == fold)
            fa, fs, _ = _oriented_auc(y_dir[mask], x[mask], max(18, minimum_rows // 3))
            if math.isfinite(fa):
                oriented_folds.append(fa if sign > 0 else 1.0 - fa)
                signs.append(fs)
        if not oriented_folds:
            continue
        consistency = float(np.mean(np.asarray(signs) == sign))
        dir_effect = abs(dir_auc - 0.5) * 2.0
        median_effect = max(float(np.median(oriented_folds)) - 0.5, 0.0) * 2.0
        worst_effect = max(float(np.min(oriented_folds)) - 0.5, 0.0) * 2.0
        family_penalty = 0.10 if family == "mixed" else 0.0
        score = dir_effect * (0.35 + 0.65 * consistency) + 0.45 * median_effect + 0.30 * worst_effect - magnitude_penalty * move_effect - family_penalty
        rows.append({
            "feature": feature, "family": family, "direction_auc": dir_auc, "direction": sign,
            "rows_large_move": n, "folds_evaluable": len(oriented_folds), "sign_consistency": consistency,
            "median_oriented_auc": float(np.median(oriented_folds)),
            "worst_oriented_auc": float(np.min(oriented_folds)), "move_auc": move_auc,
            "move_effect_penalty": move_effect, "direction_purity_score": float(score),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["direction_purity_score", "sign_consistency", "feature"], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)


def select_ranked_features(ranking: pd.DataFrame, score_column: str, k: int, *, minimum_consistency: float = 2.0 / 3.0) -> list[str]:
    if ranking.empty or int(k) <= 0:
        return []
    stable = ranking.loc[(pd.to_numeric(ranking["sign_consistency"], errors="coerce") >= float(minimum_consistency)) & (pd.to_numeric(ranking["folds_evaluable"], errors="coerce") >= 2)].copy()
    if score_column in stable:
        positive = stable.loc[pd.to_numeric(stable[score_column], errors="coerce") > 0].copy()
        if len(positive) >= min(int(k), 8):
            stable = positive
    if stable.empty:
        stable = ranking.copy()
    return stable.head(min(int(k), len(stable)))["feature"].astype(str).tolist()



def discover_ab_specs_v13(
    discovery_validation: pd.DataFrame,
    raw_features: Sequence[str],
    *,
    candidate_quantile: float = 0.65,
    max_features_per_ticker: int = 8,
    minimum_candidates: int = 24,
    minimum_class: int = 6,
) -> tuple[dict[str, list[ABSpec]], pd.DataFrame]:
    """Rebuild V10.2-style A/B separators using folds 0-2 only.

    A/B is defined inside each ticker-fold among high V10.2 base-score rows.
    Symmetric magnitude families are excluded so the evidence block cannot simply
    rediscover the V12 volatility shortcut.  No V10.2 fold-3/4 candidate identity
    is used by the primary V13 model.
    """
    required = {"ticker", "fold_id", "base_score_raw", TARGET_COLUMN}
    missing = required - set(discovery_validation.columns)
    if missing:
        raise ValueError(f"Discovery A/B map missing columns: {sorted(missing)}")
    work = discovery_validation.copy()
    work["ticker"] = work["ticker"].astype(str)
    work["base_score_raw"] = pd.to_numeric(work["base_score_raw"], errors="coerce")
    work["__base_rank"] = work.groupby(["ticker", "fold_id"], sort=False)["base_score_raw"].rank(pct=True, method="average")
    work = work.loc[work["__base_rank"] >= float(candidate_quantile)].copy()
    rows: list[dict[str, Any]] = []
    specs: dict[str, list[ABSpec]] = {}
    for ticker, ticker_rows in work.groupby("ticker", sort=True):
        y = pd.to_numeric(ticker_rows[TARGET_COLUMN], errors="coerce").to_numpy(float)
        if len(ticker_rows) < int(minimum_candidates) or np.sum(y == 1) < int(minimum_class) or np.sum(y == 0) < int(minimum_class):
            continue
        fold_values = pd.to_numeric(ticker_rows["fold_id"], errors="coerce").to_numpy(float)
        folds = sorted(pd.Series(fold_values).dropna().astype(int).unique())
        ticker_feature_rows: list[dict[str, Any]] = []
        for feature in raw_features:
            family = classify_feature(feature)
            if family in {"metadata", "magnitude"}:
                continue
            x = pd.to_numeric(ticker_rows[feature], errors="coerce").to_numpy(float)
            auc, sign, n = _oriented_auc(y, x, max(18, minimum_candidates // 2))
            if not math.isfinite(auc):
                continue
            oriented: list[float] = []
            signs: list[int] = []
            fold_ns: list[int] = []
            for fold in folds:
                mask = fold_values == fold
                fa, fs, fn = _oriented_auc(y[mask], x[mask], max(8, minimum_class * 2))
                if math.isfinite(fa):
                    oriented.append(fa if sign > 0 else 1.0 - fa)
                    signs.append(fs)
                    fold_ns.append(fn)
            if len(oriented) < 2:
                continue
            consistency = float(np.mean(np.asarray(signs) == sign))
            global_effect = abs(auc - 0.5) * 2.0
            median_effect = max(float(np.median(oriented)) - 0.5, 0.0) * 2.0
            worst_effect = max(float(np.min(oriented)) - 0.5, 0.0) * 2.0
            score = global_effect * (0.35 + 0.65 * consistency) + 0.45 * median_effect + 0.30 * worst_effect
            ticker_feature_rows.append({
                "ticker": str(ticker), "feature": str(feature), "family": family,
                "direction": int(sign), "candidate_rows": int(n),
                "candidate_positive": int(np.sum(y == 1)), "candidate_negative": int(np.sum(y == 0)),
                "folds_evaluable": int(len(oriented)), "sign_consistency": consistency,
                "global_auc": float(auc), "median_oriented_auc": float(np.median(oriented)),
                "worst_oriented_auc": float(np.min(oriented)), "ab_score": float(score),
            })
        if not ticker_feature_rows:
            continue
        ranked = pd.DataFrame(ticker_feature_rows).sort_values(
            ["ab_score", "sign_consistency", "feature"], ascending=[False, False, True], kind="mergesort"
        )
        stable = ranked.loc[(ranked["sign_consistency"] >= 2.0 / 3.0) & (ranked["ab_score"] > 0)].copy()
        if stable.empty:
            stable = ranked.head(int(max_features_per_ticker))
        chosen = stable.head(int(max_features_per_ticker))
        ticker_specs: list[ABSpec] = []
        for rank, (_, row) in enumerate(chosen.iterrows(), start=1):
            ticker_specs.append(ABSpec(
                ticker=str(ticker), node_id=str(row["feature"]), source_feature=str(row["feature"]),
                direction=int(row["direction"]), weight=max(float(row["ab_score"]), 0.05), rank=rank,
            ))
        if ticker_specs:
            specs[str(ticker)] = ticker_specs
        rows.extend(ticker_feature_rows)
    ranking = pd.DataFrame(rows)
    if not ranking.empty:
        ranking = ranking.sort_values(["ticker", "ab_score", "feature"], ascending=[True, False, True], kind="mergesort").reset_index(drop=True)
        chosen_keys = {(s.ticker, s.node_id) for values in specs.values() for s in values}
        ranking["selected_for_v13_ab"] = [(str(t), str(f)) in chosen_keys for t, f in zip(ranking["ticker"], ranking["feature"])]
    return specs, ranking

def _normalize_categories(series: pd.Series) -> list[str]:
    return sorted(series.astype("string").fillna("UNKNOWN").astype(str).unique().tolist())


def _matrix(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    medians: np.ndarray | None,
    ticker_categories: Sequence[str],
    market_categories: Sequence[str],
    bucket_categories: Sequence[str],
    ticker_onehot: bool,
    market_bucket_onehot: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if columns:
        x = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    else:
        x = np.empty((len(frame), 0), dtype=np.float32)
    if medians is None:
        med = np.nanmedian(x, axis=0) if x.shape[1] else np.empty(0, dtype=np.float32)
        med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    else:
        med = np.asarray(medians, dtype=np.float32)
    if x.shape[1]:
        bad = ~np.isfinite(x)
        if bad.any():
            x[bad] = np.take(med, np.nonzero(bad)[1])
    extra: list[np.ndarray] = []
    if ticker_onehot:
        tick = frame["ticker"].astype(str).to_numpy()
        extra.append(np.column_stack([(tick == c).astype(np.float32) for c in ticker_categories]))
    if market_bucket_onehot:
        market = frame.get("market", pd.Series("UNKNOWN", index=frame.index)).astype(str).to_numpy()
        bucket = frame.get("bucket", pd.Series("UNKNOWN", index=frame.index)).astype(str).to_numpy()
        extra.append(np.column_stack([(market == c).astype(np.float32) for c in market_categories]))
        extra.append(np.column_stack([(bucket == c).astype(np.float32) for c in bucket_categories]))
    if extra:
        x = np.concatenate([x, *extra], axis=1)
    return x.astype(np.float32, copy=False), med


def make_sample_weights(
    frame: pd.DataFrame,
    target: Sequence[int],
    *,
    half_life_days: float,
    ticker_balance: bool = True,
) -> np.ndarray:
    y = np.asarray(target, dtype=np.int8)
    dates = pd.to_datetime(frame["date"], errors="coerce")
    age = (dates.max() - dates).dt.days.to_numpy(float)
    recency = np.power(0.5, np.maximum(age, 0.0) / max(float(half_life_days), 30.0))
    pos = max(int(np.sum(y == 1)), 1)
    neg = max(int(np.sum(y == 0)), 1)
    class_w = np.where(y == 1, len(y) / (2.0 * pos), len(y) / (2.0 * neg))
    weights = recency * class_w
    if ticker_balance:
        tick = frame["ticker"].astype(str)
        counts = tick.value_counts()
        tw = tick.map((counts.max() / counts).pow(0.5)).to_numpy(float)
        weights *= tw
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 1.0)
    return weights / max(float(np.mean(weights)), EPS)


def fit_binary_model(
    train: pd.DataFrame,
    features: Sequence[str],
    target_column: str,
    spec: StageSpec,
    *,
    sample_weight: Sequence[float] | None,
    seed: int,
    cpu_threads: int,
    use_gpu: bool,
    ticker_onehot: bool,
    market_bucket_onehot: bool,
) -> ModelBundle:
    columns = [str(c) for c in features if str(c) in train.columns]
    ticker_categories = _normalize_categories(train["ticker"])
    market_categories = _normalize_categories(train.get("market", pd.Series("UNKNOWN", index=train.index)))
    bucket_categories = _normalize_categories(train.get("bucket", pd.Series("UNKNOWN", index=train.index)))
    xmat, medians = _matrix(
        train, columns, medians=None, ticker_categories=ticker_categories,
        market_categories=market_categories, bucket_categories=bucket_categories,
        ticker_onehot=ticker_onehot, market_bucket_onehot=market_bucket_onehot,
    )
    y = pd.to_numeric(train[target_column], errors="raise").to_numpy(np.int8)
    if np.unique(y).size < 2:
        raise RuntimeError(f"{target_column} has one class in training data")
    w = np.ones(len(train), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    backend = spec.backend.lower()
    centers: np.ndarray | None = None
    scales: np.ndarray | None = None
    gpu_used = False
    if backend == "xgb":
        if xgb is None:
            raise RuntimeError("xgboost is not installed")
        model = xgb.XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            device="cuda" if use_gpu else "cpu", n_estimators=int(spec.n_estimators),
            learning_rate=float(spec.learning_rate), max_depth=int(spec.max_depth),
            min_child_weight=float(spec.min_child_weight), subsample=float(spec.subsample),
            colsample_bytree=float(spec.colsample_bytree), reg_alpha=float(spec.reg_alpha),
            reg_lambda=float(spec.reg_lambda), max_bin=256, random_state=int(seed),
            n_jobs=int(cpu_threads), verbosity=0,
        )
        model.fit(xmat, y, sample_weight=w)
        gpu_used = bool(use_gpu)
    elif backend == "lgbm":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed")
        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=int(spec.n_estimators), learning_rate=float(spec.learning_rate),
            num_leaves=int(spec.num_leaves), max_depth=int(spec.max_depth), min_child_samples=int(spec.min_child_samples),
            subsample=float(spec.subsample), subsample_freq=1, colsample_bytree=float(spec.colsample_bytree),
            reg_alpha=float(spec.reg_alpha), reg_lambda=float(spec.reg_lambda), random_state=int(seed),
            n_jobs=int(cpu_threads), verbosity=-1,
        )
        model.fit(xmat, y, sample_weight=w)
    elif backend == "logit":
        # ``medians`` covers only raw numeric columns, while ``xmat`` also contains
        # ticker/market/bucket one-hot columns. Keep a separate full-width center
        # vector so C08 cannot fail by broadcasting when categoricals are enabled.
        centers = np.median(xmat, axis=0).astype(np.float32)
        centers[~np.isfinite(centers)] = 0.0
        scales = np.std(xmat, axis=0, ddof=0).astype(np.float32)
        scales[~np.isfinite(scales) | (scales < 1e-6)] = 1.0
        xstd = (xmat - centers) / scales
        model = LogisticRegression(C=0.35, max_iter=3000, class_weight=None, solver="lbfgs", random_state=int(seed))
        model.fit(xstd, y, sample_weight=w)
    else:
        raise ValueError(f"Unknown backend: {spec.backend}")
    return ModelBundle(
        backend=backend, model=model, feature_columns=columns, medians=medians, centers=centers, scales=scales,
        ticker_categories=ticker_categories, market_categories=market_categories,
        bucket_categories=bucket_categories, ticker_onehot=ticker_onehot,
        market_bucket_onehot=market_bucket_onehot, gpu_used=gpu_used,
    )


def predict_binary_model(bundle: ModelBundle, frame: pd.DataFrame) -> np.ndarray:
    xmat, _ = _matrix(
        frame, bundle.feature_columns, medians=bundle.medians,
        ticker_categories=bundle.ticker_categories, market_categories=bundle.market_categories,
        bucket_categories=bundle.bucket_categories, ticker_onehot=bundle.ticker_onehot,
        market_bucket_onehot=bundle.market_bucket_onehot,
    )
    if bundle.backend == "logit":
        if bundle.centers is None or bundle.scales is None:
            raise RuntimeError("logit bundle missing centers/scales")
        xmat = (xmat - bundle.centers) / bundle.scales
    pred = np.asarray(bundle.model.predict_proba(xmat)[:, 1], dtype=float)
    return np.clip(pred, EPS, 1.0 - EPS)


def match_direction_training_rows(
    train: pd.DataFrame,
    *,
    max_negative_reuse: int = 3,
    seed: int = 17,
    neighbor_window: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match upward large moves to similarly sized non-surge large moves.

    Matching variables are training outcomes used only to select/weight training
    examples; they are never model inputs. Hierarchy is ticker -> bucket -> market
    -> global. This deliberately prevents Stage 2 from winning via volatility size.
    """
    work = train.loc[pd.to_numeric(train[MOVE_COLUMN], errors="coerce").eq(1)].copy()
    work = work.loc[pd.to_numeric(work[DIRECTION_COLUMN], errors="coerce").isin([0, 1])].copy()
    work["__row_pos"] = np.arange(len(work), dtype=np.int64)
    work["__mag"] = pd.to_numeric(work["future_abs_excursion_3d"], errors="coerce")
    work["__date_ord"] = pd.to_datetime(work["date"], errors="raise").map(pd.Timestamp.toordinal).astype(float)
    work = work.loc[np.isfinite(work["__mag"])].copy()
    positives = work.loc[pd.to_numeric(work[DIRECTION_COLUMN], errors="coerce").eq(1)].copy()
    negatives = work.loc[pd.to_numeric(work[DIRECTION_COLUMN], errors="coerce").eq(0)].copy()
    if positives.empty or negatives.empty:
        raise RuntimeError("Direction matching requires both upward and non-surge large moves")
    neg_indices = negatives.index.to_numpy()
    neg_mag = negatives["__mag"].to_numpy(float)
    neg_date = negatives["__date_ord"].to_numpy(float)
    usage = {idx: 0 for idx in neg_indices.tolist()}

    pools: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    def add_pool(level: str, key: str, subset: pd.DataFrame) -> None:
        if subset.empty:
            return
        order = np.argsort(subset["__mag"].to_numpy(float), kind="mergesort")
        pools[(level, key)] = (subset["__mag"].to_numpy(float)[order], subset.index.to_numpy()[order])
    for ticker, grp in negatives.groupby("ticker", sort=False):
        add_pool("ticker", str(ticker), grp)
    for bucket, grp in negatives.groupby("bucket", sort=False):
        add_pool("bucket", str(bucket), grp)
    for market, grp in negatives.groupby("market", sort=False):
        add_pool("market", str(market), grp)
    add_pool("global", "ALL", negatives)

    rng = np.random.default_rng(int(seed))
    positive_order = positives.index.to_numpy().copy()
    rng.shuffle(positive_order)
    pairs: list[dict[str, Any]] = []
    selected: list[Any] = []
    for pidx in positive_order:
        prow = positives.loc[pidx]
        keys = [("ticker", str(prow["ticker"])), ("bucket", str(prow["bucket"])), ("market", str(prow["market"])), ("global", "ALL")]
        chosen = None
        chosen_level = None
        chosen_cost = float("inf")
        pmag = float(prow["__mag"])
        pdate = float(prow["__date_ord"])
        for level, key in keys:
            pool = pools.get((level, key))
            if pool is None:
                continue
            mags, ids = pool
            center = int(np.searchsorted(mags, pmag))
            lo = max(0, center - int(neighbor_window))
            hi = min(len(ids), center + int(neighbor_window) + 1)
            for nidx in ids[lo:hi]:
                if usage.get(nidx, 0) >= int(max_negative_reuse):
                    continue
                nrow = negatives.loc[nidx]
                mag_cost = abs(float(nrow["__mag"]) - pmag) / 0.01
                time_cost = abs(float(nrow["__date_ord"]) - pdate) / 252.0
                cost = mag_cost + 0.20 * time_cost + 0.05 * usage.get(nidx, 0)
                if cost < chosen_cost:
                    chosen, chosen_level, chosen_cost = nidx, level, cost
            if chosen is not None:
                break
        if chosen is None:
            continue
        usage[chosen] = usage.get(chosen, 0) + 1
        selected.extend([pidx, chosen])
        pairs.append({
            "positive_index": int(pidx) if isinstance(pidx, (int, np.integer)) else str(pidx),
            "negative_index": int(chosen) if isinstance(chosen, (int, np.integer)) else str(chosen),
            "positive_ticker": str(prow["ticker"]), "negative_ticker": str(negatives.loc[chosen, "ticker"]),
            "match_level": chosen_level, "positive_abs_excursion": pmag,
            "negative_abs_excursion": float(negatives.loc[chosen, "__mag"]),
            "abs_excursion_difference": abs(float(negatives.loc[chosen, "__mag"]) - pmag),
            "calendar_days_difference": abs(float(negatives.loc[chosen, "__date_ord"]) - pdate),
            "negative_reuse_after_match": usage[chosen], "match_cost": chosen_cost,
        })
    if not selected:
        raise RuntimeError("Direction matching produced no pairs")
    matched = work.loc[selected].copy()
    matched["direction_match_weight"] = 1.0
    matched = matched.drop(columns=["__row_pos", "__mag", "__date_ord"], errors="ignore")
    manifest = pd.DataFrame(pairs)
    return matched, manifest


def combine_stage_probabilities(p_move: Sequence[float], p_up: Sequence[float]) -> np.ndarray:
    pm = np.clip(np.asarray(p_move, dtype=float), EPS, 1.0 - EPS)
    pu = np.clip(np.asarray(p_up, dtype=float), EPS, 1.0 - EPS)
    return np.clip(pm * pu, EPS, 1.0 - EPS)


def build_policy_score(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    train_stage_score: Sequence[float],
    validation_stage_score: Sequence[float],
    base_past_rank: Sequence[float] | None,
    base_rank_blend: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_local = train[["date", "ticker"]].copy()
    valid_local = validation[["date", "ticker"]].copy()
    train_local["stage_score"] = np.asarray(train_stage_score, dtype=float)
    valid_local["stage_score"] = np.asarray(validation_stage_score, dtype=float)
    hist = grouped_historical_rank(train_local, valid_local, score_column="stage_score", group_column="ticker", minimum_group_rows=30)
    global_hist = np.searchsorted(
        np.sort(train_local["stage_score"].dropna().to_numpy(float)),
        valid_local["stage_score"].to_numpy(float), side="right",
    ) / max(train_local["stage_score"].notna().sum(), 1)
    hist = np.where(np.isfinite(hist), hist, global_hist)
    date_rank = date_cross_sectional_rank(valid_local, "stage_score")
    stage_policy = 0.75 * hist + 0.25 * np.where(np.isfinite(date_rank), date_rank, 0.5)
    blend = float(np.clip(base_rank_blend, 0.0, 0.5))
    if base_past_rank is None or blend <= 0:
        final = stage_policy
    else:
        br = np.asarray(base_past_rank, dtype=float)
        br = np.where(np.isfinite(br), br, 0.5)
        final = (1.0 - blend) * stage_policy + blend * br
    return np.asarray(final, dtype=float), np.asarray(hist, dtype=float), np.asarray(date_rank, dtype=float)


def recalibrate_policy_from_prior_oof(
    prior_oof: pd.DataFrame,
    current: pd.DataFrame,
    *,
    base_rank_blend: float,
    fallback_historical_rank: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    """Calibrate a fold using only predictions from earlier OOF folds.

    V12 exposed a severe candidate-rate drift when a single discovery CDF was
    reused far into the future.  V13 first creates a past-only provisional rank
    from each outer fold's training data, then replaces it as soon as genuine
    earlier OOF predictions exist for the same frozen config/seed.  No target
    from ``current`` is read here.

    The first requested fold has no earlier OOF reference.  In that one case the
    caller-provided train-only rank is retained.  All subsequent folds use prior
    OOF score distributions, with ticker-specific ranking and a global fallback.
    """
    required = {"date", "ticker", "stage_probability", "base_past_rank"}
    missing = required - set(current.columns)
    if missing:
        raise ValueError(f"Current prediction frame missing policy columns: {sorted(missing)}")

    cur = current[["date", "ticker", "stage_probability", "base_past_rank"]].copy().reset_index(drop=True)
    cur["stage_probability"] = pd.to_numeric(cur["stage_probability"], errors="coerce")
    reference = prior_oof.copy()
    reference_ok = (
        not reference.empty
        and {"ticker", "stage_probability"}.issubset(reference.columns)
        and pd.to_numeric(reference["stage_probability"], errors="coerce").notna().sum() >= 30
    )

    if reference_ok:
        reference = reference[["ticker", "stage_probability"]].copy()
        reference["stage_probability"] = pd.to_numeric(reference["stage_probability"], errors="coerce")
        hist = grouped_historical_rank(
            reference,
            cur,
            score_column="stage_probability",
            group_column="ticker",
            minimum_group_rows=30,
        )
        global_ref = np.sort(reference["stage_probability"].dropna().to_numpy(float))
        current_score = cur["stage_probability"].to_numpy(float)
        global_rank = np.full(len(cur), np.nan, dtype=float)
        good = np.isfinite(current_score)
        if len(global_ref):
            global_rank[good] = np.searchsorted(global_ref, current_score[good], side="right") / float(len(global_ref))
        hist = np.where(np.isfinite(hist), hist, global_rank)
        source = "EARLIER_OOF"
    else:
        if fallback_historical_rank is None:
            # Raw probability is bounded and past-trained.  This path is mainly a
            # defensive fallback for custom fold subsets that start after fold 0.
            hist = np.clip(cur["stage_probability"].to_numpy(float), 0.0, 1.0)
            source = "RAW_PROBABILITY_FALLBACK"
        else:
            hist = np.asarray(fallback_historical_rank, dtype=float)
            if len(hist) != len(cur):
                raise ValueError("fallback_historical_rank length mismatch")
            hist = np.where(np.isfinite(hist), hist, np.clip(cur["stage_probability"].to_numpy(float), 0.0, 1.0))
            source = "OUTER_TRAIN_REFERENCE_FIRST_FOLD"

    date_rank = date_cross_sectional_rank(cur, "stage_probability")
    hist = np.where(np.isfinite(hist), hist, 0.5)
    date_rank = np.where(np.isfinite(date_rank), date_rank, 0.5)
    stage_policy = 0.75 * hist + 0.25 * date_rank
    blend = float(np.clip(base_rank_blend, 0.0, 0.5))
    if blend > 0:
        base_rank = pd.to_numeric(cur["base_past_rank"], errors="coerce").to_numpy(float)
        base_rank = np.where(np.isfinite(base_rank), base_rank, 0.5)
        final = (1.0 - blend) * stage_policy + blend * base_rank
    else:
        final = stage_policy
    return (
        np.clip(np.asarray(final, dtype=float), 0.0, 1.0),
        np.clip(np.asarray(hist, dtype=float), 0.0, 1.0),
        np.clip(np.asarray(date_rank, dtype=float), 0.0, 1.0),
        source,
    )


def fold_score_metrics(predictions: pd.DataFrame, score_column: str = "policy_score") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, grp in predictions.groupby("fold_id", sort=True):
        y = pd.to_numeric(grp[TARGET_COLUMN], errors="coerce").to_numpy(float)
        score = pd.to_numeric(grp[score_column], errors="coerce").to_numpy(float)
        rows.append({
            "fold_id": int(fold), "rows": int(len(grp)), "positive": int(np.nansum(y == 1)),
            "base_rate": float(np.nanmean(y)), "pr_auc": safe_pr_auc(y, score), "roc_auc": safe_roc_auc(y, score),
            "move_pr_auc": safe_pr_auc(pd.to_numeric(grp[MOVE_COLUMN], errors="coerce"), grp["p_move"]),
            "direction_pr_auc_on_large_move": safe_pr_auc(
                pd.to_numeric(grp.loc[pd.to_numeric(grp[MOVE_COLUMN], errors="coerce").eq(1), DIRECTION_COLUMN], errors="coerce"),
                pd.to_numeric(grp.loc[pd.to_numeric(grp[MOVE_COLUMN], errors="coerce").eq(1), "p_up_given_move"], errors="coerce"),
            ),
        })
    return pd.DataFrame(rows)


def apply_policy(predictions: pd.DataFrame, threshold: float, score_column: str = "policy_score") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, grp in predictions.groupby("fold_id", sort=True):
        score = pd.to_numeric(grp[score_column], errors="coerce").to_numpy(float)
        y = pd.to_numeric(grp[TARGET_COLUMN], errors="coerce").to_numpy(np.int8)
        alert = np.isfinite(score) & (score >= float(threshold))
        n = int(alert.sum())
        tp = int(np.sum(y[alert] == 1)) if n else 0
        precision = tp / n if n else float("nan")
        rows.append({
            "fold_id": int(fold), "threshold": float(threshold), "rows": int(len(grp)),
            "alerts": n, "true_positive": tp, "precision": precision,
            "wilson_lower_95": wilson_lower(tp, n), "alert_rate": n / max(len(grp), 1),
            "base_rate": float(np.mean(y == 1)), "pr_auc": safe_pr_auc(y, score), "roc_auc": safe_roc_auc(y, score),
        })
    return pd.DataFrame(rows)


def select_frozen_policy(
    discovery_predictions: pd.DataFrame,
    *,
    discovery_folds: Sequence[int],
    minimum_alerts: int,
    target_precision: float,
    score_column: str = "policy_score",
) -> PolicySelection:
    work = discovery_predictions.loc[discovery_predictions["fold_id"].isin([int(x) for x in discovery_folds])].copy()
    values = pd.to_numeric(work[score_column], errors="coerce").dropna().to_numpy(float)
    if not len(values):
        raise RuntimeError("No finite discovery policy scores")
    quantiles = np.unique(np.concatenate([np.linspace(0.40, 0.995, 240), np.array([0.997, 0.999])]))
    thresholds = np.unique(np.quantile(values, quantiles))
    candidates: list[tuple[bool, float, float, float, int, float, pd.DataFrame]] = []
    for threshold in thresholds:
        metrics = apply_policy(work, float(threshold), score_column)
        metrics = metrics.loc[metrics["fold_id"].isin(discovery_folds)].copy()
        if len(metrics) != len(set(discovery_folds)):
            continue
        alerts_ok = bool((metrics["alerts"] >= int(minimum_alerts)).all())
        precision_values = pd.to_numeric(metrics["precision"], errors="coerce")
        min_precision = float(precision_values.min()) if precision_values.notna().all() else 0.0
        mean_precision = float(precision_values.mean()) if precision_values.notna().any() else 0.0
        min_wilson = float(pd.to_numeric(metrics["wilson_lower_95"], errors="coerce").min())
        min_alert = int(metrics["alerts"].min())
        safe = bool(alerts_ok and min_precision >= float(target_precision))
        candidates.append((safe, min_wilson, min_precision, mean_precision, min_alert, float(threshold), metrics))
    if not candidates:
        raise RuntimeError("No frozen policy candidate produced complete discovery metrics")
    safe_candidates = [x for x in candidates if x[0]]
    if safe_candidates:
        chosen = max(safe_candidates, key=lambda x: (x[1], x[2], x[3], x[5]))
        reason = "DISCOVERY_SAFE_THRESHOLD"
    else:
        eligible = [x for x in candidates if x[4] >= int(minimum_alerts)]
        chosen = max(eligible or candidates, key=lambda x: (x[2], x[1], x[3], x[4], x[5]))
        reason = "DIAGNOSTIC_BEST_MIN_PRECISION_NO_SAFE_THRESHOLD"
    return PolicySelection(
        threshold=chosen[5], safe_on_discovery=chosen[0], reason=reason,
        minimum_precision=chosen[2], mean_precision=chosen[3], minimum_wilson_lower=chosen[1],
        minimum_alerts_observed=chosen[4], fold_metrics=chosen[6].reset_index(drop=True),
    )


def oracle_top_k(predictions: pd.DataFrame, *, minimum_alerts: int, score_column: str = "policy_score") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, grp in predictions.groupby("fold_id", sort=True):
        part = grp.loc[pd.to_numeric(grp[score_column], errors="coerce").notna()].copy()
        part = part.sort_values(score_column, ascending=False, kind="mergesort")
        y = pd.to_numeric(part[TARGET_COLUMN], errors="coerce").to_numpy(np.int8)
        if len(y) < int(minimum_alerts):
            rows.append({"fold_id": int(fold), "rows": len(y), "best_k": 0, "best_precision": float("nan")})
            continue
        cumulative = np.cumsum(y == 1)
        ks = np.arange(1, len(y) + 1)
        precision = cumulative / ks
        allowed = ks >= int(minimum_alerts)
        idx = int(np.argmax(np.where(allowed, precision, -1.0)))
        rows.append({
            "fold_id": int(fold), "rows": int(len(y)), "best_k": int(ks[idx]),
            "best_precision": float(precision[idx]), "true_positive": int(cumulative[idx]),
            "score_cutoff": float(part.iloc[idx][score_column]),
        })
    return pd.DataFrame(rows)


def past_oof_base_rank(
    base_oof: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    fold_id: int,
    score_column: str = "base_score_raw",
) -> np.ndarray:
    """Rank V10.2 base score using only earlier OOF folds, never current-fold CDF."""
    prior = base_oof.loc[pd.to_numeric(base_oof["fold_id"], errors="coerce") < int(fold_id)].copy()
    if prior.empty:
        return np.full(len(validation), 0.5, dtype=float)
    eval_local = validation[["ticker"]].copy()
    source = validation[score_column] if score_column in validation.columns else pd.Series(np.nan, index=validation.index)
    eval_local[score_column] = pd.to_numeric(source, errors="coerce").to_numpy(float)
    ranks = grouped_historical_rank(prior, eval_local, score_column=score_column, group_column="ticker", minimum_group_rows=30)
    return np.where(np.isfinite(ranks), ranks, 0.5)


def gpu_preflight(seed: int = 17) -> dict[str, Any]:
    if xgb is None:
        return {"available": False, "reason": "xgboost_not_installed"}
    xmat = np.asarray([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.asarray([0, 0, 1, 1], dtype=np.int8)
    try:
        model = xgb.XGBClassifier(n_estimators=4, max_depth=2, tree_method="hist", device="cuda", random_state=int(seed), verbosity=0)
        model.fit(xmat, y)
        pred = model.predict_proba(xmat)[:, 1]
        return {"available": bool(np.isfinite(pred).all()), "reason": "ok", "prediction_mean": float(np.mean(pred))}
    except Exception as exc:  # pragma: no cover - hardware dependent
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
