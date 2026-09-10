from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, log_loss

_EPS = 1e-7


def _clip(p: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=np.float64), _EPS, 1.0 - _EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _clip(p)
    return np.log(p / (1.0 - p))


@dataclass(frozen=True)
class CalibrationPolicy:
    method: str
    params: dict[str, Any]
    threshold: float
    selection_score: float
    validation_brier: float
    validation_logloss: float
    validation_rows: int
    validation_positives: int

    def payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "params": self.params,
            "threshold": self.threshold,
            "selection_score": self.selection_score,
            "validation_brier": self.validation_brier,
            "validation_logloss": self.validation_logloss,
            "validation_rows": self.validation_rows,
            "validation_positives": self.validation_positives,
        }


def fit_calibrator(method: str, y: np.ndarray, raw_p: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = _clip(raw_p)
    if method == "none":
        return {}
    if len(np.unique(y)) < 2:
        return {}
    if method == "sigmoid":
        x = _logit(p).reshape(-1, 1)
        model = LogisticRegression(C=1_000.0, solver="lbfgs", max_iter=2_000)
        model.fit(x, y)
        return {"coef": model.coef_[0].astype(float).tolist(), "intercept": float(model.intercept_[0])}
    if method == "beta":
        x = np.column_stack([np.log(p), np.log1p(-p)])
        model = LogisticRegression(C=1_000.0, solver="lbfgs", max_iter=2_000)
        model.fit(x, y)
        return {"coef": model.coef_[0].astype(float).tolist(), "intercept": float(model.intercept_[0])}
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(p, y)
        return {
            "x_thresholds": np.asarray(model.X_thresholds_, dtype=float).tolist(),
            "y_thresholds": np.asarray(model.y_thresholds_, dtype=float).tolist(),
        }
    raise ValueError(f"지원하지 않는 calibration method: {method}")


def apply_calibrator(method: str, params: dict[str, Any], raw_p: np.ndarray) -> np.ndarray:
    p = _clip(raw_p)
    if method == "none" or not params:
        return p.astype(np.float32)
    if method == "sigmoid":
        coef = float(params["coef"][0])
        intercept = float(params["intercept"])
        z = coef * _logit(p) + intercept
        return (1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))).astype(np.float32)
    if method == "beta":
        coef = np.asarray(params["coef"], dtype=float)
        intercept = float(params["intercept"])
        x = np.column_stack([np.log(p), np.log1p(-p)])
        z = x @ coef + intercept
        return (1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))).astype(np.float32)
    if method == "isotonic":
        x = np.asarray(params["x_thresholds"], dtype=float)
        y = np.asarray(params["y_thresholds"], dtype=float)
        if len(x) == 0:
            return p.astype(np.float32)
        return np.interp(p, x, y, left=y[0], right=y[-1]).astype(np.float32)
    raise ValueError(f"지원하지 않는 calibration method: {method}")


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    p = _clip(p)
    if len(np.unique(y)) < 2:
        return 0.5
    candidates = np.unique(np.concatenate([
        np.linspace(0.03, 0.97, 189),
        np.quantile(p, np.linspace(0.05, 0.95, 37)),
    ]))
    best = (float("-inf"), 0.5)
    for threshold in candidates:
        pred = (p >= threshold).astype(np.int8)
        if pred.min() == pred.max():
            continue
        score = float(balanced_accuracy_score(y, pred))
        positive_rate = float(pred.mean())
        # 같은 점수라면 지나치게 많은 경고를 내지 않는 임계값을 선택한다.
        tie_break = -abs(positive_rate - min(0.35, max(0.08, float(y.mean()))))
        candidate = (score + 1e-6 * tie_break, float(threshold))
        if candidate > best:
            best = candidate
    return float(best[1])


def select_train_only_policy(
    y: np.ndarray,
    raw_p: np.ndarray,
    dates: np.ndarray,
    methods: tuple[str, ...] = ("none", "sigmoid", "beta", "isotonic"),
) -> CalibrationPolicy:
    y = np.asarray(y, dtype=np.int8)
    raw_p = _clip(raw_p)
    dates = np.asarray(dates)
    unique_dates = np.unique(dates)
    if len(unique_dates) < 12 or len(np.unique(y)) < 2:
        return CalibrationPolicy("none", {}, 0.5, math.inf, math.nan, math.nan, len(y), int(y.sum()))

    split_pos = max(6, int(len(unique_dates) * 0.67))
    split_pos = min(split_pos, len(unique_dates) - 5)
    fit_dates = set(unique_dates[:split_pos])
    fit_mask = np.asarray([d in fit_dates for d in dates], dtype=bool)
    eval_mask = ~fit_mask
    if fit_mask.sum() < 100 or eval_mask.sum() < 50 or len(np.unique(y[fit_mask])) < 2 or len(np.unique(y[eval_mask])) < 2:
        fit_mask = np.arange(len(y)) < max(1, int(len(y) * 0.67))
        eval_mask = ~fit_mask

    candidates: list[tuple[float, str, dict[str, Any], float, float]] = []
    for method in methods:
        if method == "isotonic" and (int(y[fit_mask].sum()) < 20 or int((1 - y[fit_mask]).sum()) < 20):
            continue
        try:
            params = fit_calibrator(method, y[fit_mask], raw_p[fit_mask])
            pred = apply_calibrator(method, params, raw_p[eval_mask])
            brier = float(brier_score_loss(y[eval_mask], pred))
            ll = float(log_loss(y[eval_mask], pred, labels=[0, 1]))
            score = ll + 0.50 * brier
            candidates.append((score, method, params, brier, ll))
        except Exception:
            continue
    if not candidates:
        candidates = [(math.inf, "none", {}, math.nan, math.nan)]
    candidates.sort(key=lambda x: (x[0], x[1]))
    score, method, _, brier, ll = candidates[0]
    final_params = fit_calibrator(method, y, raw_p)
    # 임계값은 시간상 뒤쪽 검증 구간에서만 고른다.
    threshold_pred = apply_calibrator(method, final_params, raw_p[eval_mask])
    threshold = choose_threshold(y[eval_mask], threshold_pred)
    return CalibrationPolicy(
        method=method,
        params=final_params,
        threshold=threshold,
        selection_score=float(score),
        validation_brier=brier,
        validation_logloss=ll,
        validation_rows=int(eval_mask.sum()),
        validation_positives=int(y[eval_mask].sum()),
    )
