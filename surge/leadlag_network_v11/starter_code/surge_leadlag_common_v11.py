from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


SCHEMA_VERSION = "crashwatch_surge_leadlag_network_v11"


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    role: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    purge_start: pd.Timestamp
    purge_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def utc_now() -> str:
    return pd.Timestamp.now(tz="UTC").isoformat()


def normalize_ticker(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_folds(path: Path) -> list[FoldSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds: list[FoldSpec] = []
    for item in payload:
        folds.append(
            FoldSpec(
                fold_id=int(item["fold_id"]),
                role=str(item.get("fold_role") or role_for_fold(int(item["fold_id"]))),
                train_start=pd.Timestamp(item["train_start"]),
                train_end=pd.Timestamp(item["train_end"]),
                purge_start=pd.Timestamp(item["purge_start"]),
                purge_end=pd.Timestamp(item["purge_end"]),
                validation_start=pd.Timestamp(item["validation_start"]),
                validation_end=pd.Timestamp(item["validation_end"]),
            )
        )
    return folds


def role_for_fold(fold_id: int) -> str:
    if int(fold_id) <= 2:
        return "discovery"
    if int(fold_id) <= 4:
        return "development"
    if int(fold_id) <= 6:
        return "confirmation"
    return "recent_audit"


def lag_align(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    """Return A[t], B[t+lag]. Positive lag means A leads B."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError("lag_align requires equal-length calendar-aligned arrays")
    if lag > 0:
        return x[:-lag], y[lag:]
    if lag < 0:
        k = -lag
        return x[k:], y[:-k]
    return x, y


def finite_pair(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    return x[valid], y[valid]


def safe_corr(x: np.ndarray, y: np.ndarray, *, method: str = "pearson", minimum: int = 20) -> tuple[float, float, int]:
    x, y = finite_pair(x, y)
    n = int(len(x))
    if n < int(minimum) or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return math.nan, math.nan, n
    if method == "spearman":
        value, p_value = stats.spearmanr(x, y)
    else:
        value, p_value = stats.pearsonr(x, y)
    return float(value), float(p_value), n


def safe_auc(y: np.ndarray, score: np.ndarray) -> float:
    y, score = finite_pair(y, score)
    if len(y) < 2 or len(np.unique(y)) < 2:
        return math.nan
    return float(roc_auc_score(y.astype(int), score))


def sign_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    x, y = finite_pair(x, y)
    n = int(len(x))
    if n == 0:
        return {key: math.nan for key in ["sign_agreement", "up_up_probability", "down_down_probability", "opposite_probability", "phi"]} | {"observations": 0}
    up_x, up_y = x > 0, y > 0
    down_x, down_y = x < 0, y < 0
    nonzero = (x != 0) & (y != 0)
    agreement = float(np.mean(np.sign(x[nonzero]) == np.sign(y[nonzero]))) if nonzero.any() else math.nan
    phi, _, _ = safe_corr(up_x.astype(float), up_y.astype(float), minimum=10)
    return {
        "observations": n,
        "sign_agreement": agreement,
        "up_up_probability": float(np.mean(up_x & up_y)),
        "down_down_probability": float(np.mean(down_x & down_y)),
        "opposite_probability": float(np.mean((up_x & down_y) | (down_x & up_y))),
        "phi": phi,
    }


def tail_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    x, y = finite_pair(x, y)
    n = int(len(x))
    if n < 20:
        return {"observations": n, "upper_tail_dependence": math.nan, "lower_tail_dependence": math.nan,
                "joint_up_2pct": math.nan, "joint_down_2pct": math.nan, "joint_up_5pct": math.nan}
    qx_hi, qy_hi = np.quantile(x, 0.90), np.quantile(y, 0.90)
    qx_lo, qy_lo = np.quantile(x, 0.10), np.quantile(y, 0.10)
    p_x_hi = max(float(np.mean(x >= qx_hi)), 1e-12)
    p_x_lo = max(float(np.mean(x <= qx_lo)), 1e-12)
    return {
        "observations": n,
        "upper_tail_dependence": float(np.mean((x >= qx_hi) & (y >= qy_hi)) / p_x_hi),
        "lower_tail_dependence": float(np.mean((x <= qx_lo) & (y <= qy_lo)) / p_x_lo),
        "joint_up_2pct": float(np.mean((x >= 0.02) & (y >= 0.02))),
        "joint_down_2pct": float(np.mean((x <= -0.02) & (y <= -0.02))),
        "joint_up_5pct": float(np.mean((x >= 0.05) & (y >= 0.05))),
    }


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if not len(valid_positions):
        return result
    valid = values[valid_positions]
    order = np.argsort(valid, kind="mergesort")
    ranked = valid[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    result[valid_positions] = restored
    return result


def fisher_combine(p_values: Iterable[float]) -> float:
    values = np.asarray([value for value in p_values if math.isfinite(value) and value > 0], dtype=float)
    if not len(values):
        return math.nan
    statistic = float(-2.0 * np.sum(np.log(np.clip(values, 1e-300, 1.0))))
    return float(stats.chi2.sf(statistic, 2 * len(values)))


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    return float(np.average(value_array[valid], weights=weight_array[valid])) if valid.any() else math.nan


def percentile_rank(values: Sequence[float]) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    return series.rank(method="average", pct=True, na_option="keep").to_numpy(float)


def quantile_mutual_information(x: np.ndarray, y: np.ndarray, *, bins: int = 8, minimum: int = 30) -> tuple[float, int]:
    x, y = finite_pair(x, y)
    n = len(x)
    if n < minimum:
        return math.nan, n
    try:
        x_bin = pd.qcut(x, q=min(bins, max(2, n // 10)), labels=False, duplicates="drop")
        y_bin = pd.qcut(y, q=min(bins, max(2, n // 10)), labels=False, duplicates="drop")
    except ValueError:
        return 0.0, n
    table = pd.crosstab(x_bin, y_bin).to_numpy(float)
    if not table.size or table.sum() <= 0:
        return 0.0, n
    joint = table / table.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    valid = joint > 0
    mi = float(np.sum(joint[valid] * np.log(joint[valid] / expected[valid])))
    return mi, n


def binary_mutual_information(x: np.ndarray, y: np.ndarray, minimum: int = 30) -> tuple[float, int]:
    x, y = finite_pair(x, y)
    if len(x) < minimum:
        return math.nan, len(x)
    table = np.zeros((2, 2), dtype=float)
    xb, yb = (x > 0).astype(int), (y > 0).astype(int)
    for a, b in zip(xb, yb):
        table[a, b] += 1
    joint = table / table.sum()
    expected = joint.sum(axis=1, keepdims=True) @ joint.sum(axis=0, keepdims=True)
    valid = joint > 0
    return float(np.sum(joint[valid] * np.log(joint[valid] / expected[valid]))), len(x)


def linear_r2(design: np.ndarray, target: np.ndarray) -> tuple[float, float, int, int]:
    design = np.asarray(design, dtype=float)
    target = np.asarray(target, dtype=float)
    valid = np.isfinite(target) & np.all(np.isfinite(design), axis=1)
    x = design[valid]
    y = target[valid]
    if len(y) <= x.shape[1] + 5 or np.nanstd(y) <= 1e-12:
        return math.nan, math.nan, len(y), x.shape[1]
    x = np.column_stack([np.ones(len(x)), x])
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ coefficients
    sse = float(np.sum(residual ** 2))
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / total if total > 0 else math.nan
    return r2, sse, len(y), x.shape[1]


def granger_incremental(target: np.ndarray, leader: np.ndarray, order: int) -> dict[str, float | int]:
    target = np.asarray(target, dtype=float)
    leader = np.asarray(leader, dtype=float)
    if len(target) <= order + 10:
        return {"observations": 0, "baseline_r2": math.nan, "augmented_r2": math.nan,
                "incremental_r2": math.nan, "f_statistic": math.nan, "p_value": math.nan}
    y = target[order:]
    own = np.column_stack([target[order - lag : -lag] for lag in range(1, order + 1)])
    external = np.column_stack([leader[order - lag : -lag] for lag in range(1, order + 1)])
    valid = np.isfinite(y) & np.all(np.isfinite(own), axis=1) & np.all(np.isfinite(external), axis=1)
    y, own, external = y[valid], own[valid], external[valid]
    if len(y) <= 2 * order + 10:
        return {"observations": len(y), "baseline_r2": math.nan, "augmented_r2": math.nan,
                "incremental_r2": math.nan, "f_statistic": math.nan, "p_value": math.nan}
    base_r2, base_sse, n, _ = linear_r2(own, y)
    aug_r2, aug_sse, _, aug_parameters = linear_r2(np.column_stack([own, external]), y)
    denominator_df = n - aug_parameters
    if not math.isfinite(base_sse) or not math.isfinite(aug_sse) or aug_sse <= 0 or denominator_df <= 0:
        f_stat, p_value = math.nan, math.nan
    else:
        f_stat = max(((base_sse - aug_sse) / order) / (aug_sse / denominator_df), 0.0)
        p_value = float(stats.f.sf(f_stat, order, denominator_df))
    return {"observations": n, "baseline_r2": base_r2, "augmented_r2": aug_r2,
            "incremental_r2": aug_r2 - base_r2 if math.isfinite(base_r2) and math.isfinite(aug_r2) else math.nan,
            "f_statistic": f_stat, "p_value": p_value}


def binary_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(y) & np.isfinite(score)
    y, score = y[valid].astype(int), score[valid]
    rows = int(len(y))
    positives = int(np.sum(y == 1))
    negatives = rows - positives
    if rows == 0 or positives == 0 or negatives == 0:
        return {"rows": rows, "positives": positives, "negatives": negatives, "base_rate": math.nan,
                "pr_auc": math.nan, "roc_auc": math.nan, "pr_auc_lift": math.nan}
    base = positives / rows
    pr = float(average_precision_score(y, score))
    return {"rows": rows, "positives": positives, "negatives": negatives, "base_rate": base,
            "pr_auc": pr, "roc_auc": float(roc_auc_score(y, score)), "pr_auc_lift": pr / base if base > 0 else math.nan}


def best_precision_policy(y: np.ndarray, score: np.ndarray, *, target_precision: float = 0.70, minimum_alerts: int = 30) -> dict[str, Any]:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(y) & np.isfinite(score)
    y, score = y[valid].astype(int), score[valid]
    if len(y) == 0:
        return {"gate_pass": False, "threshold": math.inf, "alerts": 0, "precision": math.nan, "recall": 0.0,
                "best_practical_threshold": math.inf, "best_practical_alerts": 0, "best_practical_precision": math.nan}
    order = np.argsort(-score, kind="mergesort")
    sorted_y, sorted_score = y[order], score[order]
    tp = np.cumsum(sorted_y)
    alerts = np.arange(1, len(y) + 1)
    precision = tp / alerts
    recall = tp / max(int(np.sum(y)), 1)
    practical = np.flatnonzero(alerts >= int(minimum_alerts))
    if len(practical):
        best_index = practical[np.argmax(precision[practical])]
    else:
        best_index = int(np.argmax(precision))
    passing = np.flatnonzero((alerts >= int(minimum_alerts)) & (precision >= float(target_precision)))
    gate_index = passing[np.argmax(recall[passing])] if len(passing) else None
    return {
        "gate_pass": gate_index is not None,
        "threshold": float(sorted_score[gate_index]) if gate_index is not None else math.inf,
        "alerts": int(alerts[gate_index]) if gate_index is not None else 0,
        "precision": float(precision[gate_index]) if gate_index is not None else math.nan,
        "recall": float(recall[gate_index]) if gate_index is not None else 0.0,
        "best_practical_threshold": float(sorted_score[best_index]),
        "best_practical_alerts": int(alerts[best_index]),
        "best_practical_precision": float(precision[best_index]),
        "best_practical_recall": float(recall[best_index]),
    }


def folds_to_dicts(folds: Sequence[FoldSpec]) -> list[dict[str, Any]]:
    return [asdict(fold) for fold in folds]
