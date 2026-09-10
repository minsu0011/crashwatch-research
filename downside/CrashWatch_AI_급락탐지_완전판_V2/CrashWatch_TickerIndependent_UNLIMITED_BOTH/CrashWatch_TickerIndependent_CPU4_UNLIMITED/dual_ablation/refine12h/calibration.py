from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

_EPS = 1e-7


def clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), _EPS, 1.0 - _EPS)


def logit(values: np.ndarray) -> np.ndarray:
    p = clip_probability(values)
    return np.log(p / (1.0 - p))


def sigmoid(values: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def _weighted_binary_loss(theta: np.ndarray, x: np.ndarray, y: np.ndarray, l2: float) -> tuple[float, np.ndarray]:
    z = x @ theta
    p = sigmoid(z)
    loss = -np.mean(y * np.log(clip_probability(p)) + (1.0 - y) * np.log(clip_probability(1.0 - p)))
    if l2 > 0:
        loss += 0.5 * l2 * float(np.sum(theta[1:] ** 2))
    gradient = (x.T @ (p - y)) / max(1, len(y))
    if l2 > 0:
        gradient[1:] += l2 * theta[1:]
    return float(loss), gradient.astype(np.float64)


def _fit_bounded_logistic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    positive_indices: Iterable[int] = (),
    l2: float = 1e-4,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    theta0 = np.zeros(x.shape[1], dtype=np.float64)
    theta0[0] = logit(np.asarray([np.clip(y.mean(), 1e-4, 1 - 1e-4)]))[0]
    bounds: list[tuple[float | None, float | None]] = [(None, None)] * x.shape[1]
    for index in positive_indices:
        bounds[index] = (0.0, None)
        theta0[index] = 1.0

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        return _weighted_binary_loss(theta, x, y, l2)

    result = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"bounded logistic fit failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def _rank_preserving_tie_break(raw: np.ndarray, calibrated: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Preserve raw ordering inside isotonic plateaus without changing useful probabilities."""
    raw = clip_probability(raw)
    calibrated = clip_probability(calibrated)
    order = np.argsort(raw, kind="mergesort")
    adjusted = calibrated[order].copy()
    ranks = np.linspace(0.0, epsilon, len(adjusted), dtype=np.float64)
    adjusted = np.maximum.accumulate(adjusted + ranks)
    adjusted = np.minimum(adjusted, 1.0 - _EPS)
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output


def _standardize_regime(regime: np.ndarray | None) -> tuple[np.ndarray | None, np.ndarray, np.ndarray]:
    if regime is None:
        return None, np.asarray([], dtype=float), np.asarray([], dtype=float)
    values = np.asarray(regime, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    medians = np.nanmedian(values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    values = np.where(np.isfinite(values), values, medians)
    scales = np.nanstd(values, axis=0)
    scales = np.where(np.isfinite(scales) & (scales > 1e-8), scales, 1.0)
    return (values - medians) / scales, medians, scales


@dataclass(frozen=True)
class SafeCalibrationPolicy:
    method: str
    params: dict[str, Any]
    threshold: float
    selection_score: float
    validation_brier: float
    validation_logloss: float
    validation_raw_roc_auc: float
    validation_calibrated_roc_auc: float
    validation_raw_pr_auc: float
    validation_calibrated_pr_auc: float
    rank_correlation: float
    validation_rows: int
    validation_positives: int
    fallback_reason: str | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def fit_calibrator(
    method: str,
    y: np.ndarray,
    raw_p: np.ndarray,
    regime: np.ndarray | None = None,
    regime_names: list[str] | None = None,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.int8)
    p = clip_probability(raw_p)
    if method == "none" or len(np.unique(y)) < 2:
        return {}
    if method == "platt_positive":
        x = np.column_stack([np.ones(len(p)), logit(p)])
        theta = _fit_bounded_logistic(x, y, positive_indices=(1,))
        return {"intercept": float(theta[0]), "slope": float(theta[1])}
    if method == "beta_positive":
        # z = c + a*log(p) + b*(-log(1-p)); a,b>=0 is strictly increasing.
        x = np.column_stack([np.ones(len(p)), np.log(p), -np.log1p(-p)])
        theta = _fit_bounded_logistic(x, y, positive_indices=(1, 2))
        return {"intercept": float(theta[0]), "a": float(theta[1]), "b": float(theta[2])}
    if method == "isotonic_safe":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0, increasing=True)
        model.fit(p, y)
        return {
            "x_thresholds": np.asarray(model.X_thresholds_, dtype=float).tolist(),
            "y_thresholds": np.asarray(model.y_thresholds_, dtype=float).tolist(),
            "tie_break_epsilon": 1e-8,
        }
    if method == "regime_logistic":
        scaled, medians, scales = _standardize_regime(regime)
        if scaled is None or scaled.shape[1] == 0:
            raise ValueError("regime_logistic requires regime features")
        x = np.column_stack([np.ones(len(p)), logit(p), scaled])
        theta = _fit_bounded_logistic(x, y, positive_indices=(1,), l2=5e-4)
        return {
            "intercept": float(theta[0]),
            "raw_slope": float(theta[1]),
            "regime_coef": theta[2:].astype(float).tolist(),
            "regime_medians": medians.astype(float).tolist(),
            "regime_scales": scales.astype(float).tolist(),
            "regime_names": list(regime_names or [f"regime_{i}" for i in range(scaled.shape[1])]),
        }
    raise ValueError(f"unsupported calibration method: {method}")


def apply_calibrator(
    method: str,
    params: dict[str, Any],
    raw_p: np.ndarray,
    regime: np.ndarray | None = None,
) -> np.ndarray:
    p = clip_probability(raw_p)
    if method == "none" or not params:
        return p.astype(np.float32)
    if method == "platt_positive":
        output = sigmoid(float(params["intercept"]) + float(params["slope"]) * logit(p))
    elif method == "beta_positive":
        z = (
            float(params["intercept"])
            + float(params["a"]) * np.log(p)
            + float(params["b"]) * (-np.log1p(-p))
        )
        output = sigmoid(z)
    elif method == "isotonic_safe":
        x = np.asarray(params.get("x_thresholds", []), dtype=float)
        y = np.asarray(params.get("y_thresholds", []), dtype=float)
        output = p if len(x) == 0 else np.interp(p, x, y, left=y[0], right=y[-1])
        output = _rank_preserving_tie_break(p, output, float(params.get("tie_break_epsilon", 1e-8)))
    elif method == "regime_logistic":
        values = np.asarray(regime, dtype=np.float64) if regime is not None else np.empty((len(p), 0))
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        medians = np.asarray(params["regime_medians"], dtype=float)
        scales = np.asarray(params["regime_scales"], dtype=float)
        if values.shape[1] != len(medians):
            raise ValueError(f"regime feature count mismatch: {values.shape[1]} != {len(medians)}")
        values = np.where(np.isfinite(values), values, medians)
        scaled = (values - medians) / scales
        coef = np.asarray(params["regime_coef"], dtype=float)
        z = float(params["intercept"]) + float(params["raw_slope"]) * logit(p) + scaled @ coef
        output = sigmoid(z)
    else:
        raise ValueError(f"unsupported calibration method: {method}")
    return clip_probability(output).astype(np.float32)


def choose_threshold(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    p = clip_probability(p)
    if len(np.unique(y)) < 2:
        return 0.5
    candidates = np.unique(np.concatenate([np.linspace(0.02, 0.98, 193), np.quantile(p, np.linspace(0.02, 0.98, 49))]))
    target_alert_rate = min(0.35, max(0.08, float(y.mean())))
    best_key = (float("-inf"), float("-inf"))
    best_threshold = 0.5
    for threshold in candidates:
        labels = (p >= threshold).astype(np.int8)
        if labels.min() == labels.max():
            continue
        balanced = float(balanced_accuracy_score(y, labels))
        rate_penalty = -abs(float(labels.mean()) - target_alert_rate)
        key = (balanced, rate_penalty)
        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
    return best_threshold


def _safe_metric(function, y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(function(y, p)) if len(np.unique(y)) == 2 else math.nan
    except Exception:
        return math.nan


def _candidate_diagnostics(y: np.ndarray, raw: np.ndarray, calibrated: np.ndarray) -> dict[str, float]:
    raw_roc = _safe_metric(roc_auc_score, y, raw)
    cal_roc = _safe_metric(roc_auc_score, y, calibrated)
    raw_pr = _safe_metric(average_precision_score, y, raw)
    cal_pr = _safe_metric(average_precision_score, y, calibrated)
    finite = np.isfinite(raw) & np.isfinite(calibrated)
    raw_finite = raw[finite]
    calibrated_finite = calibrated[finite]
    if (
        len(raw_finite) < 2
        or np.unique(raw_finite).size < 2
        or np.unique(calibrated_finite).size < 2
    ):
        correlation = 1.0 if np.allclose(raw_finite, calibrated_finite) else 0.0
    else:
        correlation = spearmanr(raw_finite, calibrated_finite).statistic
    return {
        "raw_roc": raw_roc,
        "cal_roc": cal_roc,
        "raw_pr": raw_pr,
        "cal_pr": cal_pr,
        "rank_correlation": float(correlation) if np.isfinite(correlation) else 0.0,
    }


def select_safe_policy(
    y: np.ndarray,
    raw_p: np.ndarray,
    dates: np.ndarray,
    *,
    regime: np.ndarray | None = None,
    regime_names: list[str] | None = None,
    methods: tuple[str, ...] = ("none", "platt_positive", "beta_positive", "isotonic_safe", "regime_logistic"),
    forced_method: str | None = None,
    forced_threshold: float | None = None,
    max_roc_drop: float = 0.005,
    max_pr_drop: float = 0.010,
    min_rank_correlation: float = 0.995,
) -> SafeCalibrationPolicy:
    y = np.asarray(y, dtype=np.int8)
    raw_p = clip_probability(raw_p)
    dates = np.asarray(dates)
    if len(y) != len(raw_p) or len(y) != len(dates):
        raise ValueError("calibration arrays must have equal length")
    if len(np.unique(y)) < 2 or len(y) < 80:
        threshold = float(forced_threshold if forced_threshold is not None else choose_threshold(y, raw_p))
        return SafeCalibrationPolicy(
            "none", {}, threshold, math.inf, math.nan, math.nan, math.nan, math.nan, math.nan, math.nan,
            1.0, len(y), int(y.sum()), "insufficient calibration labels",
        )

    unique_dates = np.unique(dates)
    split_pos = min(len(unique_dates) - 5, max(6, int(len(unique_dates) * 0.67)))
    fit_dates = set(unique_dates[:split_pos])
    fit_mask = np.asarray([value in fit_dates for value in dates], dtype=bool)
    eval_mask = ~fit_mask
    if fit_mask.sum() < 100 or eval_mask.sum() < 50 or len(np.unique(y[fit_mask])) < 2 or len(np.unique(y[eval_mask])) < 2:
        cut = max(1, int(len(y) * 0.67))
        fit_mask = np.arange(len(y)) < cut
        eval_mask = ~fit_mask

    candidate_methods = (forced_method,) if forced_method else methods
    candidates: list[dict[str, Any]] = []
    for method in candidate_methods:
        if method == "regime_logistic" and regime is None:
            continue
        if method == "isotonic_safe" and min(int(y[fit_mask].sum()), int((1 - y[fit_mask]).sum())) < 20:
            continue
        try:
            fit_regime = None if regime is None else np.asarray(regime)[fit_mask]
            eval_regime = None if regime is None else np.asarray(regime)[eval_mask]
            params = fit_calibrator(method, y[fit_mask], raw_p[fit_mask], fit_regime, regime_names)
            pred = apply_calibrator(method, params, raw_p[eval_mask], eval_regime)
            diagnostics = _candidate_diagnostics(y[eval_mask], raw_p[eval_mask], pred)
            roc_drop = diagnostics["raw_roc"] - diagnostics["cal_roc"] if np.isfinite(diagnostics["raw_roc"]) else 0.0
            pr_drop = diagnostics["raw_pr"] - diagnostics["cal_pr"] if np.isfinite(diagnostics["raw_pr"]) else 0.0
            safe = (
                roc_drop <= max_roc_drop
                and pr_drop <= max_pr_drop
                and diagnostics["rank_correlation"] >= min_rank_correlation
            )
            if not safe and method != "none":
                continue
            brier = float(brier_score_loss(y[eval_mask], pred))
            ll = float(log_loss(y[eval_mask], pred, labels=[0, 1]))
            # Ranking degradation receives a large penalty even within tolerance.
            score = ll + 0.50 * brier + 2.0 * max(0.0, roc_drop) + max(0.0, pr_drop)
            candidates.append({"score": score, "method": method, "brier": brier, "logloss": ll, "fit_params": params, "eval_prediction": pred, **diagnostics})
        except Exception:
            continue

    fallback_reason = None
    if not candidates:
        candidates = [{
            "score": math.inf, "method": "none", "brier": math.nan, "logloss": math.nan,
            "fit_params": {}, "eval_prediction": raw_p[eval_mask],
            **_candidate_diagnostics(y[eval_mask], raw_p[eval_mask], raw_p[eval_mask]),
        }]
        fallback_reason = "all candidate calibrators failed safety guards"
    candidates.sort(key=lambda row: (row["score"], row["method"]))
    selected = candidates[0]
    method = str(selected["method"])
    final_params = fit_calibrator(method, y, raw_p, regime, regime_names)
    final_pred = apply_calibrator(method, final_params, raw_p, regime)
    final_diag = _candidate_diagnostics(y, raw_p, final_pred)
    if (
        method != "none"
        and (
            final_diag["raw_roc"] - final_diag["cal_roc"] > max_roc_drop
            or final_diag["raw_pr"] - final_diag["cal_pr"] > max_pr_drop
            or final_diag["rank_correlation"] < min_rank_correlation
        )
    ):
        fallback_reason = f"final {method} failed full-calibration safety guard"
        method = "none"
        final_params = {}
        final_pred = raw_p
        final_diag = _candidate_diagnostics(y, raw_p, final_pred)

    # Threshold is selected on the held-out tail using parameters fitted only on the earlier calibration segment.
    if forced_threshold is not None:
        threshold = float(forced_threshold)
    elif method == selected["method"]:
        threshold = float(choose_threshold(y[eval_mask], np.asarray(selected["eval_prediction"], dtype=float)))
    else:
        threshold = float(choose_threshold(y[eval_mask], raw_p[eval_mask]))
    return SafeCalibrationPolicy(
        method=method,
        params=final_params,
        threshold=threshold,
        selection_score=float(selected["score"]),
        validation_brier=float(selected["brier"]),
        validation_logloss=float(selected["logloss"]),
        validation_raw_roc_auc=float(selected["raw_roc"]),
        validation_calibrated_roc_auc=float(selected["cal_roc"]),
        validation_raw_pr_auc=float(selected["raw_pr"]),
        validation_calibrated_pr_auc=float(selected["cal_pr"]),
        rank_correlation=float(selected["rank_correlation"]),
        validation_rows=int(eval_mask.sum()),
        validation_positives=int(y[eval_mask].sum()),
        fallback_reason=fallback_reason,
    )
