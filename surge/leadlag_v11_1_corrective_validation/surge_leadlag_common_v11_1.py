from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

SCHEMA_VERSION = "crashwatch_surge_leadlag_v11_1_corrective_validation"


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


def role_for_fold(fold_id: int) -> str:
    fold_id = int(fold_id)
    if fold_id <= 2:
        return "discovery"
    if fold_id <= 4:
        return "development"
    if fold_id <= 6:
        return "confirmation"
    return "recent_audit"


def load_folds(path: Path) -> list[FoldSpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    folds: list[FoldSpec] = []
    for item in payload:
        fold_id = int(item["fold_id"])
        folds.append(
            FoldSpec(
                fold_id=fold_id,
                role=role_for_fold(fold_id),
                train_start=pd.Timestamp(item["train_start"]),
                train_end=pd.Timestamp(item["train_end"]),
                purge_start=pd.Timestamp(item["purge_start"]),
                purge_end=pd.Timestamp(item["purge_end"]),
                validation_start=pd.Timestamp(item["validation_start"]),
                validation_end=pd.Timestamp(item["validation_end"]),
            )
        )
    return folds


def normalize_ticker(value: object) -> str:
    if pd.isna(value):
        return "UNKNOWN"
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int_seed(*parts: object, modulus: int = 2**32 - 1) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:16], 16) % modulus


def bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    valid_positions = np.flatnonzero(np.isfinite(values))
    if len(valid_positions) == 0:
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


def weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return float(np.average(x[valid], weights=w[valid])) if valid.any() else math.nan


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


def finite_corr(x: np.ndarray, y: np.ndarray, minimum: int = 20) -> tuple[float, int]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    n = int(len(x))
    if n < int(minimum):
        return math.nan, n
    xc = x - x.mean()
    yc = y - y.mean()
    denom = float(np.sqrt(np.sum(xc * xc) * np.sum(yc * yc)))
    if denom <= 1e-15:
        return math.nan, n
    return float(np.sum(xc * yc) / denom), n


def rowwise_corr_against_vector(x: np.ndarray, y_matrix: np.ndarray, minimum: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Fast row-wise Pearson correlation between one x vector and P permuted y rows."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y_matrix, dtype=float)
    if y.ndim != 2 or y.shape[1] != len(x):
        raise ValueError("y_matrix must have shape [permutations, len(x)]")
    finite_x = np.isfinite(x)
    x0 = np.where(finite_x, x, 0.0)
    mask = np.isfinite(y) & finite_x[None, :]
    n = mask.sum(axis=1).astype(float)
    y0 = np.where(mask, y, 0.0)
    mx = mask.astype(float)
    sum_x = mx @ x0
    sum_y = y0.sum(axis=1)
    sum_x2 = mx @ (x0 * x0)
    sum_y2 = (y0 * y0).sum(axis=1)
    sum_xy = y0 @ x0
    with np.errstate(invalid="ignore", divide="ignore"):
        cov_num = sum_xy - (sum_x * sum_y / n)
        var_x = sum_x2 - (sum_x * sum_x / n)
        var_y = sum_y2 - (sum_y * sum_y / n)
        denom = np.sqrt(np.maximum(var_x, 0.0) * np.maximum(var_y, 0.0))
        corr = cov_num / denom
    bad = (n < minimum) | ~np.isfinite(corr) | (denom <= 1e-15)
    corr[bad] = np.nan
    return corr, n.astype(int)


def binary_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(y, dtype=float)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(y) & np.isfinite(score)
    y = y[valid].astype(int)
    score = score[valid]
    rows = int(len(y))
    positives = int(np.sum(y == 1))
    negatives = rows - positives
    if rows == 0 or positives == 0 or negatives == 0:
        return {
            "rows": rows,
            "positives": positives,
            "negatives": negatives,
            "base_rate": positives / rows if rows else math.nan,
            "pr_auc": math.nan,
            "roc_auc": math.nan,
            "pr_auc_lift": math.nan,
        }
    base = positives / rows
    pr_auc = float(average_precision_score(y, score))
    return {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "base_rate": base,
        "pr_auc": pr_auc,
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc_lift": pr_auc / base if base > 0 else math.nan,
    }


def alignment_offsets(lag: int, target_horizon: int = 3) -> list[tuple[int, int]]:
    """
    For A -> B lag=h and B forecast origin t, map B D+k to A[t+k-h].
    Only offsets <=0 are legal at forecast time t.
    Returns [(k, leader_offset), ...].
    """
    lag = int(lag)
    if lag <= 0:
        raise ValueError("directed lag must be positive")
    result: list[tuple[int, int]] = []
    for k in range(1, target_horizon + 1):
        offset = k - lag
        if offset <= 0:
            result.append((k, offset))
    return result


def select_frozen_portfolio_threshold(
    predictions: pd.DataFrame,
    *,
    target_precision: float,
    minimum_alerts_per_fold: int,
    development_folds: Sequence[int] = (3, 4),
) -> dict[str, Any]:
    """Choose a single threshold using development folds only, then freeze it."""
    dev = predictions.loc[predictions["fold_id"].isin(list(development_folds))].copy()
    dev = dev.loc[np.isfinite(dev["score"]) & dev["target"].isin([0, 1])]
    if dev.empty or set(development_folds) - set(dev["fold_id"].unique()):
        return {"threshold": math.nan, "development_gate_pass": False, "selection_status": "MISSING_DEVELOPMENT_FOLD"}
    candidates = np.unique(dev["score"].to_numpy(float))
    candidates = np.sort(candidates)[::-1]
    records: list[dict[str, Any]] = []
    for threshold in candidates:
        fold_rows = []
        for fold_id in development_folds:
            part = dev.loc[dev["fold_id"].eq(fold_id)]
            mask = part["score"].to_numpy(float) >= threshold
            alerts = int(mask.sum())
            precision = float(part.loc[mask, "target"].mean()) if alerts else math.nan
            fold_rows.append((fold_id, alerts, precision))
        support_ok = all(alerts >= minimum_alerts_per_fold for _, alerts, _ in fold_rows)
        precision_ok = support_ok and all(math.isfinite(precision) and precision >= target_precision for _, _, precision in fold_rows)
        min_precision = min((precision for _, alerts, precision in fold_rows if alerts >= minimum_alerts_per_fold and math.isfinite(precision)), default=-1.0)
        pooled_mask = dev["score"].to_numpy(float) >= threshold
        pooled_precision = float(dev.loc[pooled_mask, "target"].mean()) if pooled_mask.any() else math.nan
        pooled_alerts = int(pooled_mask.sum())
        records.append({
            "threshold": float(threshold),
            "support_ok": support_ok,
            "precision_ok": precision_ok,
            "min_precision": min_precision,
            "pooled_precision": pooled_precision,
            "pooled_alerts": pooled_alerts,
            "fold_rows": fold_rows,
        })
    passing = [row for row in records if row["precision_ok"]]
    if passing:
        # Prefer the lowest threshold that still passes, i.e. maximum coverage.
        chosen = sorted(passing, key=lambda r: (-r["pooled_alerts"], -r["pooled_precision"], r["threshold"]))[0]
        status = "DEV_GATE_PASS"
        gate = True
    else:
        supported = [row for row in records if row["support_ok"]]
        if supported:
            chosen = sorted(supported, key=lambda r: (-r["min_precision"], -r["pooled_precision"], -r["pooled_alerts"], -r["threshold"]))[0]
            status = "DEV_GATE_FAIL_DIAGNOSTIC_THRESHOLD"
        else:
            chosen = records[0]
            status = "DEV_INSUFFICIENT_ALERT_SUPPORT"
        gate = False
    return {
        "threshold": float(chosen["threshold"]),
        "development_gate_pass": bool(gate),
        "selection_status": status,
        "development_min_precision": float(chosen["min_precision"]) if math.isfinite(chosen["min_precision"]) else math.nan,
        "development_pooled_precision": float(chosen["pooled_precision"]) if math.isfinite(chosen["pooled_precision"]) else math.nan,
        "development_pooled_alerts": int(chosen["pooled_alerts"]),
        "development_fold_detail": [
            {"fold_id": int(fid), "alerts": int(alerts), "precision": float(precision) if math.isfinite(precision) else math.nan}
            for fid, alerts, precision in chosen["fold_rows"]
        ],
    }
