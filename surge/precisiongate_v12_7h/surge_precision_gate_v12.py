from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

try:  # optional at import time; required only for backend='lgbm'
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover
    lgb = None


EPS = 1e-6
V12_SCHEMA = "crashwatch_surge_precision_gate_v12"


@dataclass(frozen=True)
class SeparatorSpec:
    ticker: str
    node_id: str
    source_feature: str
    direction: int
    weight: float
    rank: int


@dataclass
class RobustScale:
    median: float
    scale: float


@dataclass
class TickerEvidenceState:
    node_scales: dict[str, RobustScale]
    base_scores_sorted: np.ndarray
    specs: list[SeparatorSpec]


@dataclass
class EvidenceTransformer:
    states: dict[str, TickerEvidenceState]
    max_slots: int


@dataclass
class GateModelBundle:
    backend: str
    global_model: Any
    ticker_models: dict[str, Any]
    ticker_train_n: dict[str, int]
    ticker_train_pos: dict[str, int]
    ticker_train_neg: dict[str, int]
    prior_strength: float
    feature_columns: list[str]
    candidate_quantile: float
    negative_weight: float
    global_prior: float


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"1", "true", "yes", "y"})


def safe_pr_auc(y: Sequence[int], score: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=int)
    ss = np.asarray(score, dtype=float)
    mask = np.isfinite(ss) & np.isin(yy, [0, 1])
    if mask.sum() == 0 or np.unique(yy[mask]).size < 2:
        return float("nan")
    return float(average_precision_score(yy[mask], ss[mask]))


def safe_roc_auc(y: Sequence[int], score: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=int)
    ss = np.asarray(score, dtype=float)
    mask = np.isfinite(ss) & np.isin(yy, [0, 1])
    if mask.sum() == 0 or np.unique(yy[mask]).size < 2:
        return float("nan")
    return float(roc_auc_score(yy[mask], ss[mask]))


def wilson_lower_bound(successes: int, trials: int, z: float = 1.959963984540054) -> float:
    if trials <= 0:
        return float("nan")
    p = successes / trials
    denom = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    spread = z * math.sqrt((p * (1.0 - p) / trials) + (z * z / (4.0 * trials * trials)))
    return float((centre - spread) / denom)


def build_discovery_separator_specs(
    discovery_precision: pd.DataFrame,
    available_columns: Iterable[str],
    *,
    max_features_per_ticker: int = 12,
    require_candidate: bool = True,
) -> dict[str, list[SeparatorSpec]]:
    """Build ticker-specific A/B specs using discovery-only V10.2 map.

    This function intentionally refuses to infer directions from V10.2 confirmation/recent
    compact results.  The input should be probe_discovery_precision_map_v10_2.csv, which
    was rebuilt on folds 0-2 only by V10.2.
    """
    required = {
        "ticker", "axis", "node_id", "source_feature", "selection_direction",
        "precision_separator_score_v10_2",
    }
    missing = required - set(discovery_precision.columns)
    if missing:
        raise ValueError(f"discovery precision map missing columns: {sorted(missing)}")
    part = discovery_precision.copy()
    part["ticker"] = part["ticker"].astype(str).str.zfill(6)
    part = part.loc[part["axis"].astype(str).eq("AB")].copy()
    if require_candidate:
        if "precision_separator_selection_candidate" not in part.columns:
            raise ValueError("discovery precision map lacks precision_separator_selection_candidate")
        part = part.loc[_as_bool(part["precision_separator_selection_candidate"])].copy()
    part["selection_direction"] = pd.to_numeric(part["selection_direction"], errors="coerce")
    part["precision_separator_score_v10_2"] = pd.to_numeric(
        part["precision_separator_score_v10_2"], errors="coerce"
    )
    part = part.loc[part["selection_direction"].isin([-1, 1])].copy()
    available = set(str(x) for x in available_columns)
    part = part.loc[part["node_id"].astype(str).isin(available)].copy()
    if part.empty:
        return {}
    # One transformed representative per source feature prevents correlated transform clones
    # from dominating the gate, consistent with V10.2 source-dedup philosophy.
    part = part.sort_values(
        ["ticker", "source_feature", "precision_separator_score_v10_2", "node_id"],
        ascending=[True, True, False, True], kind="mergesort"
    ).drop_duplicates(["ticker", "source_feature"], keep="first")
    part = part.sort_values(
        ["ticker", "precision_separator_score_v10_2", "source_feature"],
        ascending=[True, False, True], kind="mergesort"
    )
    out: dict[str, list[SeparatorSpec]] = {}
    for ticker, grp in part.groupby("ticker", sort=True):
        rows = grp.head(int(max_features_per_ticker))
        specs: list[SeparatorSpec] = []
        for rank, (_, row) in enumerate(rows.iterrows(), start=1):
            score = float(row["precision_separator_score_v10_2"])
            specs.append(
                SeparatorSpec(
                    ticker=str(ticker),
                    node_id=str(row["node_id"]),
                    source_feature=str(row["source_feature"]),
                    direction=int(row["selection_direction"]),
                    weight=max(score, 0.05),
                    rank=rank,
                )
            )
        if specs:
            out[str(ticker)] = specs
    return out


def _robust_scale(values: np.ndarray) -> RobustScale:
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return RobustScale(0.0, 1.0)
    med = float(np.median(x))
    q25, q75 = np.quantile(x, [0.25, 0.75])
    scale = float((q75 - q25) / 1.349)
    if not math.isfinite(scale) or scale < 1e-6:
        scale = float(np.std(x)) if x.size > 1 else 1.0
    if not math.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    return RobustScale(med, scale)


def fit_evidence_transformer(
    frame: pd.DataFrame,
    specs_by_ticker: Mapping[str, Sequence[SeparatorSpec]],
    *,
    ticker_column: str = "ticker",
    base_score_column: str = "base_score_raw",
    max_slots: int = 12,
) -> EvidenceTransformer:
    states: dict[str, TickerEvidenceState] = {}
    tick = frame[ticker_column].astype(str).str.zfill(6)
    for ticker, specs0 in specs_by_ticker.items():
        specs = list(specs0)[: int(max_slots)]
        rows = frame.loc[tick.eq(str(ticker))]
        node_scales: dict[str, RobustScale] = {}
        for spec in specs:
            node_scales[spec.node_id] = _robust_scale(
                pd.to_numeric(rows.get(spec.node_id, pd.Series(dtype=float)), errors="coerce").to_numpy(float)
            )
        base = pd.to_numeric(rows[base_score_column], errors="coerce").to_numpy(float)
        base = np.sort(base[np.isfinite(base)])
        states[str(ticker)] = TickerEvidenceState(node_scales=node_scales, base_scores_sorted=base, specs=specs)
    return EvidenceTransformer(states=states, max_slots=int(max_slots))


def empirical_rank(sorted_reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.asarray(sorted_reference, dtype=float)
    x = np.asarray(values, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    valid = np.isfinite(x)
    if ref.size == 0:
        return out
    out[valid] = np.searchsorted(ref, x[valid], side="right") / float(ref.size)
    return out


def transform_evidence(
    frame: pd.DataFrame,
    transformer: EvidenceTransformer,
    *,
    ticker_column: str = "ticker",
    base_score_column: str = "base_score_raw",
) -> pd.DataFrame:
    n = len(frame)
    result = pd.DataFrame(index=frame.index)
    base = np.clip(pd.to_numeric(frame[base_score_column], errors="coerce").to_numpy(float), EPS, 1.0 - EPS)
    result["base_prob"] = base
    result["base_logit"] = logit(base)
    result["base_hist_rank"] = np.nan
    result["ab_evidence_count"] = 0.0
    for slot in range(1, transformer.max_slots + 1):
        result[f"ab_slot_{slot:02d}"] = 0.0
        result[f"ab_slot_present_{slot:02d}"] = 0.0
    aggregate_names = [
        "ab_weighted_mean", "ab_mean", "ab_min", "ab_max", "ab_std",
        "ab_favorable_fraction", "ab_adverse_fraction", "ab_strong_adverse_fraction",
    ]
    for name in aggregate_names:
        result[name] = 0.0

    tick = frame[ticker_column].astype(str).str.zfill(6)
    for ticker, state in transformer.states.items():
        pos = np.flatnonzero(tick.to_numpy() == str(ticker))
        if pos.size == 0:
            continue
        subset = frame.iloc[pos]
        b = pd.to_numeric(subset[base_score_column], errors="coerce").to_numpy(float)
        result.iloc[pos, result.columns.get_loc("base_hist_rank")] = empirical_rank(state.base_scores_sorted, b)
        oriented_cols: list[np.ndarray] = []
        weights: list[float] = []
        for slot, spec in enumerate(state.specs, start=1):
            values = pd.to_numeric(subset[spec.node_id], errors="coerce").to_numpy(float)
            scale = state.node_scales[spec.node_id]
            z = (values - scale.median) / scale.scale
            z = np.clip(z, -6.0, 6.0) * float(spec.direction)
            present = np.isfinite(z)
            z = np.where(present, z, 0.0)
            result.iloc[pos, result.columns.get_loc(f"ab_slot_{slot:02d}")] = z
            result.iloc[pos, result.columns.get_loc(f"ab_slot_present_{slot:02d}")] = present.astype(float)
            oriented_cols.append(z)
            weights.append(float(spec.weight))
        if not oriented_cols:
            continue
        mat = np.vstack(oriented_cols).T
        present = np.vstack([
            result.iloc[pos][f"ab_slot_present_{slot:02d}"].to_numpy(float)
            for slot in range(1, len(oriented_cols) + 1)
        ]).T
        count = present.sum(axis=1)
        weight_arr = np.asarray(weights, dtype=float)
        weighted_denom = (present * weight_arr[None, :]).sum(axis=1)
        weighted_num = (mat * present * weight_arr[None, :]).sum(axis=1)
        weighted_mean = np.divide(weighted_num, weighted_denom, out=np.zeros_like(weighted_num), where=weighted_denom > 0)
        safe_count = np.maximum(count, 1.0)
        mean = (mat * present).sum(axis=1) / safe_count
        diff = (mat - mean[:, None]) * present
        std = np.sqrt((diff * diff).sum(axis=1) / safe_count)
        masked_min = np.where(present > 0, mat, np.inf).min(axis=1)
        masked_max = np.where(present > 0, mat, -np.inf).max(axis=1)
        masked_min[~np.isfinite(masked_min)] = 0.0
        masked_max[~np.isfinite(masked_max)] = 0.0
        favorable = ((mat > 0.5) * present).sum(axis=1) / safe_count
        adverse = ((mat < -0.5) * present).sum(axis=1) / safe_count
        strong_adverse = ((mat < -1.5) * present).sum(axis=1) / safe_count
        values_map = {
            "ab_evidence_count": count,
            "ab_weighted_mean": weighted_mean,
            "ab_mean": mean,
            "ab_min": masked_min,
            "ab_max": masked_max,
            "ab_std": std,
            "ab_favorable_fraction": favorable,
            "ab_adverse_fraction": adverse,
            "ab_strong_adverse_fraction": strong_adverse,
        }
        for name, vals in values_map.items():
            result.iloc[pos, result.columns.get_loc(name)] = vals
    result["base_hist_rank"] = pd.to_numeric(result["base_hist_rank"], errors="coerce")
    return result


def gate_feature_columns(max_slots: int) -> list[str]:
    cols = [
        "base_logit", "base_hist_rank", "ab_evidence_count", "ab_weighted_mean", "ab_mean",
        "ab_min", "ab_max", "ab_std", "ab_favorable_fraction", "ab_adverse_fraction",
        "ab_strong_adverse_fraction",
    ]
    for slot in range(1, int(max_slots) + 1):
        cols.extend([f"ab_slot_{slot:02d}", f"ab_slot_present_{slot:02d}"])
    return cols


def _prepare_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    x = frame[list(columns)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    x[~np.isfinite(x)] = 0.0
    return x


def _fit_model(backend: str, x: np.ndarray, y: np.ndarray, weights: np.ndarray, seed: int) -> Any:
    if np.unique(y).size < 2:
        return ("constant", float(np.mean(y)) if len(y) else 0.0)
    if backend == "logit":
        model = LogisticRegression(
            C=0.25, solver="liblinear", max_iter=2500,
            random_state=int(seed),
        )
        model.fit(x, y, sample_weight=weights)
        return model
    if backend == "lgbm":
        if lgb is None:
            raise RuntimeError("lightgbm is not installed")
        model = lgb.LGBMClassifier(
            n_estimators=140,
            learning_rate=0.035,
            num_leaves=7,
            max_depth=3,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.80,
            reg_alpha=1.0,
            reg_lambda=4.0,
            random_state=int(seed),
            n_jobs=4,
            verbosity=-1,
        )
        model.fit(x, y, sample_weight=weights)
        return model
    raise ValueError(f"unknown backend={backend}")


def _predict_model(model: Any, x: np.ndarray) -> np.ndarray:
    if isinstance(model, tuple) and model and model[0] == "constant":
        return np.full(len(x), float(model[1]), dtype=float)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def fit_gate_bundle(
    meta: pd.DataFrame,
    *,
    backend: str,
    candidate_quantile: float,
    prior_strength: float = 60.0,
    negative_weight: float = 2.0,
    min_ticker_rows: int = 30,
    min_ticker_class: int = 5,
    max_slots: int = 12,
    ticker_column: str = "ticker",
    target_column: str = "target",
    seed: int = 17,
) -> GateModelBundle:
    feature_cols = gate_feature_columns(max_slots)
    work = meta.copy()
    work["candidate"] = pd.to_numeric(work["base_hist_rank"], errors="coerce") >= float(candidate_quantile)
    work = work.loc[work["candidate"] & work[target_column].isin([0, 1])].copy()
    if work.empty:
        raise ValueError("no candidate rows for gate training")
    y = work[target_column].to_numpy(np.int8)
    x = _prepare_matrix(work, feature_cols)
    weights = np.where(y == 0, float(negative_weight), 1.0)
    global_model = _fit_model(backend, x, y, weights, seed)
    ticker_models: dict[str, Any] = {}
    ticker_train_n: dict[str, int] = {}
    ticker_train_pos: dict[str, int] = {}
    ticker_train_neg: dict[str, int] = {}
    for ticker, grp in work.groupby(ticker_column, sort=True):
        yy = grp[target_column].to_numpy(np.int8)
        pos = int((yy == 1).sum())
        neg = int((yy == 0).sum())
        ticker_train_n[str(ticker)] = int(len(grp))
        ticker_train_pos[str(ticker)] = pos
        ticker_train_neg[str(ticker)] = neg
        if len(grp) < int(min_ticker_rows) or min(pos, neg) < int(min_ticker_class):
            continue
        xx = _prepare_matrix(grp, feature_cols)
        ww = np.where(yy == 0, float(negative_weight), 1.0)
        ticker_models[str(ticker)] = _fit_model(backend, xx, yy, ww, seed + 1009 + len(ticker_models))
    return GateModelBundle(
        backend=backend,
        global_model=global_model,
        ticker_models=ticker_models,
        ticker_train_n=ticker_train_n,
        ticker_train_pos=ticker_train_pos,
        ticker_train_neg=ticker_train_neg,
        prior_strength=float(prior_strength),
        feature_columns=feature_cols,
        candidate_quantile=float(candidate_quantile),
        negative_weight=float(negative_weight),
        global_prior=float(np.mean(y)),
    )


def predict_gate_bundle(
    bundle: GateModelBundle,
    meta: pd.DataFrame,
    *,
    ticker_column: str = "ticker",
) -> pd.DataFrame:
    x = _prepare_matrix(meta, bundle.feature_columns)
    global_prob = np.clip(_predict_model(bundle.global_model, x), EPS, 1.0 - EPS)
    final = global_prob.copy()
    source = np.array(["GLOBAL"] * len(meta), dtype=object)
    tick = meta[ticker_column].astype(str).to_numpy()
    for ticker, model in bundle.ticker_models.items():
        pos = np.flatnonzero(tick == str(ticker))
        if pos.size == 0:
            continue
        ticker_prob = np.clip(_predict_model(model, x[pos]), EPS, 1.0 - EPS)
        n = float(bundle.ticker_train_n.get(str(ticker), 0))
        shrink = n / (n + float(bundle.prior_strength))
        combined_logit = (1.0 - shrink) * logit(global_prob[pos]) + shrink * logit(ticker_prob)
        final[pos] = expit(combined_logit)
        source[pos] = "SHRUNK_TICKER"
    eligible = pd.to_numeric(meta["base_hist_rank"], errors="coerce").to_numpy(float) >= bundle.candidate_quantile
    return pd.DataFrame(
        {
            "gate_prob": final,
            "gate_global_prob": global_prob,
            "candidate_eligible": eligible,
            "gate_source": source,
        },
        index=meta.index,
    )


def combine_base_and_gate(base_prob: Sequence[float], gate_prob: Sequence[float], alpha: float) -> np.ndarray:
    base = np.clip(np.asarray(base_prob, dtype=float), EPS, 1.0 - EPS)
    gate = np.clip(np.asarray(gate_prob, dtype=float), EPS, 1.0 - EPS)
    return expit((1.0 - float(alpha)) * logit(base) + float(alpha) * logit(gate))


def evaluate_scores(
    frame: pd.DataFrame,
    *,
    score_column: str,
    base_score_column: str = "base_score_raw",
    target_column: str = "target",
    fold_column: str = "fold_id",
    policy_eligible_column: str = "candidate_eligible",
    minimum_alerts: int = 30,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_id, grp in frame.groupby(fold_column, sort=True):
        y = grp[target_column].to_numpy(np.int8)
        s = pd.to_numeric(grp[score_column], errors="coerce").to_numpy(float)
        b = pd.to_numeric(grp[base_score_column], errors="coerce").to_numpy(float)
        eligible = grp[policy_eligible_column].astype(bool).to_numpy() if policy_eligible_column in grp else np.ones(len(grp), bool)
        best = best_practical_precision(y[eligible], s[eligible], minimum_alerts=minimum_alerts)
        rows.append(
            {
                "fold_id": int(fold_id),
                "rows": int(len(grp)),
                "positives": int((y == 1).sum()),
                "base_rate": float(np.mean(y)) if len(y) else float("nan"),
                "pr_auc": safe_pr_auc(y, s),
                "base_pr_auc": safe_pr_auc(y, b),
                "roc_auc": safe_roc_auc(y, s),
                "base_roc_auc": safe_roc_auc(y, b),
                "pr_auc_lift": safe_pr_auc(y, s) / max(safe_pr_auc(y, b), EPS) if math.isfinite(safe_pr_auc(y, b)) else float("nan"),
                **{f"best_{k}": v for k, v in best.items()},
            }
        )
    return pd.DataFrame(rows)


def best_practical_precision(y: Sequence[int], score: Sequence[float], *, minimum_alerts: int = 30) -> dict[str, Any]:
    yy = np.asarray(y, dtype=np.int8)
    ss = np.asarray(score, dtype=float)
    mask = np.isfinite(ss) & np.isin(yy, [0, 1])
    yy = yy[mask]
    ss = ss[mask]
    if len(yy) < int(minimum_alerts):
        return {"threshold": float("inf"), "alerts": 0, "tp": 0, "fp": 0, "precision": float("nan"), "recall": 0.0, "wilson_lcb": float("nan")}
    order = np.argsort(-ss, kind="mergesort")
    ysort = yy[order]
    ssort = ss[order]
    tp = np.cumsum(ysort == 1)
    alerts = np.arange(1, len(ysort) + 1)
    precision = tp / alerts
    positives = max(int((ysort == 1).sum()), 1)
    recall = tp / positives
    eligible = np.flatnonzero(alerts >= int(minimum_alerts))
    if eligible.size == 0:
        return {"threshold": float("inf"), "alerts": 0, "tp": 0, "fp": 0, "precision": float("nan"), "recall": 0.0, "wilson_lcb": float("nan")}
    # Prefer precision first, then recall, then fewer alerts for the same precision/recall.
    idx = max(eligible.tolist(), key=lambda i: (precision[i], recall[i], -alerts[i]))
    return {
        "threshold": float(ssort[idx]),
        "alerts": int(alerts[idx]),
        "tp": int(tp[idx]),
        "fp": int(alerts[idx] - tp[idx]),
        "precision": float(precision[idx]),
        "recall": float(recall[idx]),
        "wilson_lcb": wilson_lower_bound(int(tp[idx]), int(alerts[idx])),
    }


def _threshold_metrics(y: np.ndarray, score: np.ndarray, eligible: np.ndarray, threshold: float) -> dict[str, Any]:
    mask = eligible & np.isfinite(score) & (score >= float(threshold))
    alerts = int(mask.sum())
    if alerts == 0:
        return {"alerts": 0, "tp": 0, "fp": 0, "precision": float("nan"), "recall": 0.0, "wilson_lcb": float("nan")}
    tp = int((y[mask] == 1).sum())
    fp = alerts - tp
    positives = max(int((y == 1).sum()), 1)
    return {
        "alerts": alerts,
        "tp": tp,
        "fp": fp,
        "precision": float(tp / alerts),
        "recall": float(tp / positives),
        "wilson_lcb": wilson_lower_bound(tp, alerts),
    }


def select_frozen_dev_threshold(
    predictions: pd.DataFrame,
    *,
    dev_folds: Sequence[int] = (3, 4),
    score_column: str = "v12_score",
    target_column: str = "target",
    eligible_column: str = "candidate_eligible",
    minimum_alerts: int = 30,
    target_precision: float = 0.70,
) -> dict[str, Any]:
    part = predictions.loc[predictions["fold_id"].isin([int(x) for x in dev_folds])].copy()
    if part.empty:
        raise ValueError("no development predictions")
    candidate_scores = pd.to_numeric(part.loc[part[eligible_column].astype(bool), score_column], errors="coerce")
    candidate_scores = candidate_scores[np.isfinite(candidate_scores)]
    if candidate_scores.empty:
        return {"safe": False, "threshold": float("inf"), "reason": "NO_ELIGIBLE_SCORES", "per_fold": []}
    # Evaluate every unique score plus a compact quantile grid for deterministic behavior.
    unique = np.unique(candidate_scores.to_numpy(float))
    if unique.size > 3000:
        unique = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, 3001)))
    thresholds = np.sort(unique)[::-1]
    safe_candidates: list[tuple[tuple[float, ...], float, list[dict[str, Any]]]] = []
    practical_candidates: list[tuple[tuple[float, ...], float, list[dict[str, Any]]]] = []
    for threshold in thresholds:
        fold_metrics: list[dict[str, Any]] = []
        valid_support = True
        precisions: list[float] = []
        recalls: list[float] = []
        alerts_list: list[int] = []
        for fold_id in dev_folds:
            grp = part.loc[part["fold_id"].eq(int(fold_id))]
            y = grp[target_column].to_numpy(np.int8)
            s = pd.to_numeric(grp[score_column], errors="coerce").to_numpy(float)
            e = grp[eligible_column].astype(bool).to_numpy()
            m = _threshold_metrics(y, s, e, float(threshold))
            fold_metrics.append({"fold_id": int(fold_id), **m})
            if m["alerts"] < int(minimum_alerts) or not math.isfinite(float(m["precision"])):
                valid_support = False
            else:
                precisions.append(float(m["precision"]))
                recalls.append(float(m["recall"]))
                alerts_list.append(int(m["alerts"]))
        if not valid_support:
            continue
        key = (min(precisions), float(np.mean(precisions)), min(recalls), float(np.mean(recalls)), -float(np.mean(alerts_list)))
        practical_candidates.append((key, float(threshold), fold_metrics))
        if all(p >= float(target_precision) for p in precisions):
            safe_candidates.append((key, float(threshold), fold_metrics))
    chosen_pool = safe_candidates if safe_candidates else practical_candidates
    if not chosen_pool:
        return {"safe": False, "threshold": float("inf"), "reason": "NO_THRESHOLD_WITH_MIN_ALERTS_EACH_DEV_FOLD", "per_fold": []}
    key, threshold, per_fold = max(chosen_pool, key=lambda item: item[0])
    return {
        "safe": bool(safe_candidates),
        "threshold": float(threshold),
        "reason": "SAFE_DEV_POLICY" if safe_candidates else "DIAGNOSTIC_ONLY_DEV_GATE_FAIL",
        "min_dev_precision": float(key[0]),
        "mean_dev_precision": float(key[1]),
        "min_dev_recall": float(key[2]),
        "mean_dev_recall": float(key[3]),
        "per_fold": per_fold,
    }


def apply_frozen_threshold(
    predictions: pd.DataFrame,
    threshold: float,
    *,
    folds: Sequence[int],
    score_column: str = "v12_score",
    target_column: str = "target",
    eligible_column: str = "candidate_eligible",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold_id in folds:
        grp = predictions.loc[predictions["fold_id"].eq(int(fold_id))]
        y = grp[target_column].to_numpy(np.int8)
        s = pd.to_numeric(grp[score_column], errors="coerce").to_numpy(float)
        e = grp[eligible_column].astype(bool).to_numpy()
        rows.append({"fold_id": int(fold_id), **_threshold_metrics(y, s, e, threshold)})
    return pd.DataFrame(rows)


def config_sort_key(summary: Mapping[str, Any]) -> tuple[float, ...]:
    """Lexicographic selection: safe policy first, then precision, then ranking lift."""
    def finite(value: Any, fallback: float = -1.0) -> float:
        try:
            x = float(value)
        except Exception:
            return fallback
        return x if math.isfinite(x) else fallback
    return (
        1.0 if bool(summary.get("dev_safe", False)) else 0.0,
        finite(summary.get("min_dev_precision", -1.0)),
        finite(summary.get("mean_dev_precision", -1.0)),
        finite(summary.get("min_dev_pr_auc_lift", -1.0)),
        finite(summary.get("mean_dev_pr_auc_lift", -1.0)),
    )


def serialize_specs(specs: Mapping[str, Sequence[SeparatorSpec]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ticker, items in sorted(specs.items()):
        for spec in items:
            rows.append(dataclasses.asdict(spec))
    return rows
