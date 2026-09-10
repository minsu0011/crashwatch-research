from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


V6_SCHEMA = "crashwatch_surge_precision70_v6"
EPSILON = 1e-9


@dataclass(frozen=True)
class PrecisionPolicy:
    """Frozen selective-alert policy.

    The policy never imposes a daily count/fraction cap. Alerts are emitted only
    when a calibrated selective score clears a global or scope-specific
    threshold chosen on selection-fold OOF data.
    """

    kind: str = "global_threshold"
    target_precision: float = 0.70
    threshold: float | None = None
    scope_column: str | None = None
    scope_thresholds: dict[str, float] | None = None
    fallback_threshold: float | None = None
    minimum_alerts: int = 30
    minimum_alert_days: int = 10
    minimum_precision_lcb: float = 0.60
    confidence_level: float = 0.95
    required_fold_pass_rate: float = 1.0
    achieved_fold_pass_rate: float | None = None
    achieved_worst_precision: float | None = None
    achieved_worst_precision_lcb: float | None = None
    achieved_mean_recall: float | None = None
    achieved_alerts: int | None = None
    gate_pass: bool = False
    source: str = "selection_forward_oof"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PrecisionPolicy":
        allowed = {field.name for field in dataclasses.fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass(frozen=True)
class CalibratorSpec:
    kind: str
    parameters: dict[str, Any]
    source_score_kind: str = "unit_interval"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parameters": self.parameters,
            "source_score_kind": self.source_score_kind,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CalibratorSpec":
        return cls(
            kind=str(payload.get("kind", "identity")),
            parameters=dict(payload.get("parameters", {})),
            source_score_kind=str(payload.get("source_score_kind", "unit_interval")),
        )


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def sigmoid(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -40.0, 40.0)))


def logit(values: np.ndarray | Sequence[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    return np.log(clipped / (1.0 - clipped))


def rank_normalize(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.full(len(array), np.nan, dtype=np.float64)
    valid = np.isfinite(array)
    if not valid.any():
        return result
    result[valid] = pd.Series(array[valid]).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    return np.clip(result, 1e-7, 1.0 - 1e-7)


def datewise_rank_normalize(
    values: np.ndarray | Sequence[float],
    dates: np.ndarray | Sequence[Any],
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    parsed_dates = pd.to_datetime(pd.Series(dates), errors="coerce")
    frame = pd.DataFrame({"score": array, "date": parsed_dates})
    result = frame.groupby("date", sort=False, dropna=False)["score"].rank(
        method="average", pct=True
    )
    return np.clip(result.to_numpy(dtype=np.float64), 1e-7, 1.0 - 1e-7)


def rank_before_seed_average(
    predictions: Sequence[np.ndarray],
    dates: np.ndarray | Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize every seed inside each date before averaging.

    Returns the seed-rank mean and seed-rank standard deviation. This prevents a
    seed with a wider raw score scale from dominating a ranking ensemble.
    """

    if not predictions:
        raise ValueError("seed prediction이 비어 있습니다")
    rows = [datewise_rank_normalize(values, dates) for values in predictions]
    matrix = np.vstack(rows)
    return np.nanmean(matrix, axis=0), np.nanstd(matrix, axis=0)


def ensure_unit_interval(
    values: np.ndarray | Sequence[float],
    dates: np.ndarray | Sequence[Any] | None = None,
    strategy: str = "auto",
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    valid = array[np.isfinite(array)]
    if not len(valid):
        return np.full(len(array), 0.5, dtype=np.float64)
    if strategy == "date_rank":
        if dates is None:
            raise ValueError("date_rank 변환에는 dates가 필요합니다")
        return datewise_rank_normalize(array, dates)
    if strategy == "rank":
        return rank_normalize(array)
    if strategy == "sigmoid":
        return sigmoid(array)
    if strategy == "auto":
        if float(np.nanmin(valid)) >= 0.0 and float(np.nanmax(valid)) <= 1.0:
            return np.clip(array, 1e-7, 1.0 - 1e-7)
        if dates is not None:
            return datewise_rank_normalize(array, dates)
        return rank_normalize(array)
    raise ValueError(f"지원하지 않는 score 변환: {strategy}")


def wilson_lower_bound(
    successes: int,
    trials: int,
    confidence_level: float = 0.95,
) -> float:
    if trials <= 0:
        return float("nan")
    alpha = 1.0 - float(confidence_level)
    z = float(norm.ppf(1.0 - alpha))  # one-sided lower bound
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * trials)) / trials
    )
    return float(max(0.0, (centre - radius) / denominator))


def beta_posterior_lower_bound(
    successes: int,
    failures: int,
    confidence_level: float = 0.95,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> float:
    if successes < 0 or failures < 0:
        raise ValueError("성공/실패 수는 음수가 될 수 없습니다")
    quantile = 1.0 - float(confidence_level)
    return float(
        beta_distribution.ppf(
            quantile,
            prior_alpha + successes,
            prior_beta + failures,
        )
    )


def binary_ranking_metrics(target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(score, dtype=np.float64)
    valid = np.isin(y, [0, 1]) & np.isfinite(s)
    y = y[valid]
    s = s[valid]
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    result: dict[str, Any] = {
        "rows": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "positive_rate": safe_divide(positives, len(y)),
        "score_mean": float(np.mean(s)) if len(s) else float("nan"),
    }
    if len(np.unique(y)) < 2:
        result.update({"pr_auc": float("nan"), "roc_auc": float("nan")})
    else:
        result.update(
            {
                "pr_auc": float(average_precision_score(y, s)),
                "roc_auc": float(roc_auc_score(y, s)),
            }
        )
    result["pr_auc_lift"] = safe_divide(result["pr_auc"], result["positive_rate"])
    return result


def event_episode_metrics(
    target: np.ndarray,
    alert: np.ndarray,
    dates: np.ndarray | Sequence[Any],
    tickers: np.ndarray | Sequence[Any] | None,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.int8)
    selected = np.asarray(alert, dtype=bool)
    if tickers is None:
        return {
            "event_count": float("nan"),
            "captured_events": float("nan"),
            "event_recall": float("nan"),
            "alerts_per_captured_event": float("nan"),
        }
    frame = pd.DataFrame(
        {
            "target": y,
            "alert": selected,
            "date": pd.to_datetime(pd.Series(dates), errors="coerce"),
            "ticker": pd.Series(tickers, dtype="string"),
        }
    ).dropna(subset=["date", "ticker"])
    event_count = 0
    captured = 0
    for _, part in frame.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        values = part["target"].to_numpy(dtype=np.int8)
        alerts = part["alert"].to_numpy(dtype=bool)
        start = 0
        while start < len(values):
            if values[start] != 1:
                start += 1
                continue
            end = start + 1
            while end < len(values) and values[end] == 1:
                end += 1
            event_count += 1
            captured += int(bool(np.any(alerts[start:end])))
            start = end
    total_alerts = int(selected.sum())
    return {
        "event_count": int(event_count),
        "captured_events": int(captured),
        "event_recall": safe_divide(captured, event_count),
        "alerts_per_captured_event": safe_divide(total_alerts, captured),
    }


def evaluate_alerts(
    target: np.ndarray | Sequence[int],
    score: np.ndarray | Sequence[float],
    alert: np.ndarray | Sequence[bool],
    dates: np.ndarray | Sequence[Any],
    tickers: np.ndarray | Sequence[Any] | None = None,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(score, dtype=np.float64)
    selected = np.asarray(alert, dtype=bool)
    parsed_dates = pd.to_datetime(pd.Series(dates), errors="coerce")
    valid = np.isin(y, [0, 1]) & np.isfinite(s) & parsed_dates.notna().to_numpy()
    y = y[valid]
    s = s[valid]
    selected = selected[valid]
    d = parsed_dates.loc[valid].to_numpy(dtype="datetime64[ns]")
    t = np.asarray(tickers, dtype=object)[valid] if tickers is not None else None

    tp = int(np.sum(selected & (y == 1)))
    fp = int(np.sum(selected & (y == 0)))
    fn = int(np.sum((~selected) & (y == 1)))
    tn = int(np.sum((~selected) & (y == 0)))
    alerts = tp + fp
    positives = tp + fn
    negatives = fp + tn
    precision = safe_divide(tp, alerts)
    result: dict[str, Any] = {
        "rows": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "alerts": alerts,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "positive_rate": safe_divide(positives, len(y)),
        "precision": precision,
        "recall": safe_divide(tp, positives),
        "alert_rate": safe_divide(alerts, len(y)),
        "specificity": safe_divide(tn, negatives),
        "false_positive_rate": safe_divide(fp, negatives),
        "precision_lift": safe_divide(precision, safe_divide(positives, len(y))),
        "precision_wilson_lcb": wilson_lower_bound(tp, alerts, confidence_level),
        "precision_beta_lcb": beta_posterior_lower_bound(tp, fp, confidence_level),
        "date_count": int(pd.Series(d).nunique()),
        "alert_days": int(pd.Series(d[selected]).nunique()) if alerts else 0,
        "alerts_per_day_mean": safe_divide(alerts, int(pd.Series(d).nunique())),
        "score_mean": float(np.mean(s)) if len(s) else float("nan"),
        "alert_score_min": float(np.min(s[selected])) if alerts else float("nan"),
        "alert_score_mean": float(np.mean(s[selected])) if alerts else float("nan"),
    }
    result.update(event_episode_metrics(y, selected, d, t))
    result.update({f"ranking_{key}": value for key, value in binary_ranking_metrics(y, s).items()})
    return result


def date_block_bootstrap_precision_lcb(
    target: np.ndarray | Sequence[int],
    alert: np.ndarray | Sequence[bool],
    dates: np.ndarray | Sequence[Any],
    samples: int = 500,
    confidence_level: float = 0.95,
    seed: int = 20260811,
) -> float:
    """Date-block bootstrap lower bound using vectorized daily counts.

    Resampling dates, rather than individual rows, preserves the cross-sectional
    dependence among stocks observed on the same trading day.
    """

    y = np.asarray(target, dtype=np.int8)
    selected = np.asarray(alert, dtype=bool)
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
    valid = np.isin(y, [0, 1]) & parsed.notna().to_numpy()
    if not valid.any():
        return float("nan")
    frame = pd.DataFrame(
        {
            "date": parsed.loc[valid].to_numpy(dtype="datetime64[ns]"),
            "tp": (selected[valid] & (y[valid] == 1)).astype(np.int32),
            "alerts": selected[valid].astype(np.int32),
        }
    )
    daily = frame.groupby("date", sort=True)[["tp", "alerts"]].sum()
    if daily.empty:
        return float("nan")
    tp = daily["tp"].to_numpy(dtype=np.int64)
    alerts = daily["alerts"].to_numpy(dtype=np.int64)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(daily), size=(int(samples), len(daily)), endpoint=False)
    sampled_tp = tp[draw].sum(axis=1)
    sampled_alerts = alerts[draw].sum(axis=1)
    valid_samples = sampled_alerts > 0
    if not valid_samples.any():
        return float("nan")
    precision = sampled_tp[valid_samples] / sampled_alerts[valid_samples]
    return float(np.quantile(precision.astype(np.float64), 1.0 - confidence_level))


def _raw_calibration_input(values: np.ndarray, source_score_kind: str) -> np.ndarray:
    if source_score_kind == "unit_interval":
        return np.clip(values, 1e-7, 1.0 - 1e-7)
    if source_score_kind == "raw_logit":
        return sigmoid(values)
    if source_score_kind == "rank":
        return rank_normalize(values)
    raise ValueError(f"지원하지 않는 source_score_kind: {source_score_kind}")


def apply_calibrator(spec: CalibratorSpec | Mapping[str, Any], values: np.ndarray) -> np.ndarray:
    if not isinstance(spec, CalibratorSpec):
        spec = CalibratorSpec.from_dict(spec)
    base = _raw_calibration_input(np.asarray(values, dtype=np.float64), spec.source_score_kind)
    kind = spec.kind
    parameters = spec.parameters
    if kind == "identity":
        return np.clip(base, 1e-7, 1.0 - 1e-7)
    if kind == "platt":
        transformed = float(parameters["coefficient"]) * logit(base) + float(parameters["intercept"])
        return np.clip(sigmoid(transformed), 1e-7, 1.0 - 1e-7)
    if kind == "beta":
        features = np.column_stack([np.log(base), -np.log1p(-base)])
        coefficients = np.asarray(parameters["coefficients"], dtype=np.float64)
        transformed = features @ coefficients + float(parameters["intercept"])
        return np.clip(sigmoid(transformed), 1e-7, 1.0 - 1e-7)
    if kind == "isotonic":
        x = np.asarray(parameters["x_thresholds"], dtype=np.float64)
        y = np.asarray(parameters["y_thresholds"], dtype=np.float64)
        return np.clip(np.interp(base, x, y, left=y[0], right=y[-1]), 1e-7, 1.0 - 1e-7)
    raise ValueError(f"지원하지 않는 calibrator: {kind}")


def _fit_single_calibrator(
    kind: str,
    target: np.ndarray,
    base_score: np.ndarray,
    source_score_kind: str,
) -> CalibratorSpec:
    y = np.asarray(target, dtype=np.int8)
    p = _raw_calibration_input(np.asarray(base_score, dtype=np.float64), source_score_kind)
    if kind == "identity":
        return CalibratorSpec("identity", {}, source_score_kind)
    if kind == "platt":
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        model.fit(logit(p).reshape(-1, 1), y)
        return CalibratorSpec(
            "platt",
            {
                "coefficient": float(model.coef_[0, 0]),
                "intercept": float(model.intercept_[0]),
            },
            source_score_kind,
        )
    if kind == "beta":
        x = np.column_stack([np.log(p), -np.log1p(-p)])
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)
        model.fit(x, y)
        return CalibratorSpec(
            "beta",
            {
                "coefficients": [float(value) for value in model.coef_[0]],
                "intercept": float(model.intercept_[0]),
            },
            source_score_kind,
        )
    if kind == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
        model.fit(p, y)
        return CalibratorSpec(
            "isotonic",
            {
                "x_thresholds": [float(value) for value in model.X_thresholds_],
                "y_thresholds": [float(value) for value in model.y_thresholds_],
            },
            source_score_kind,
        )
    raise ValueError(kind)


def fit_best_calibrator(
    target: np.ndarray | Sequence[int],
    raw_score: np.ndarray | Sequence[float],
    dates: np.ndarray | Sequence[Any],
    source_score_kind: str = "unit_interval",
    minimum_rows: int = 300,
    candidates: Sequence[str] = ("identity", "platt", "beta", "isotonic"),
) -> tuple[CalibratorSpec, pd.DataFrame]:
    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(raw_score, dtype=np.float64)
    d = pd.to_datetime(pd.Series(dates), errors="coerce")
    valid = np.isin(y, [0, 1]) & np.isfinite(s) & d.notna().to_numpy()
    y = y[valid]
    s = s[valid]
    d = d.loc[valid].reset_index(drop=True)
    if len(y) < minimum_rows or len(np.unique(y)) < 2:
        spec = CalibratorSpec("identity", {}, source_score_kind)
        return spec, pd.DataFrame([{"kind": "identity", "brier": float("nan"), "logloss": float("nan")}])

    unique_dates = pd.DatetimeIndex(d.unique()).sort_values()
    cut_position = min(len(unique_dates) - 1, max(1, int(math.floor(len(unique_dates) * 0.70))))
    cut_date = unique_dates[cut_position]
    train_mask = (d < cut_date).to_numpy()
    valid_mask = ~train_mask
    if int(train_mask.sum()) < minimum_rows // 2 or int(valid_mask.sum()) < max(50, minimum_rows // 5):
        order = np.argsort(d.to_numpy(dtype="datetime64[ns]"), kind="mergesort")
        cut = max(1, int(len(order) * 0.70))
        train_mask = np.zeros(len(y), dtype=bool)
        train_mask[order[:cut]] = True
        valid_mask = ~train_mask

    records: list[dict[str, Any]] = []
    valid_specs: list[tuple[CalibratorSpec, tuple[float, float, int]]] = []
    for index, kind in enumerate(candidates):
        try:
            spec = _fit_single_calibrator(kind, y[train_mask], s[train_mask], source_score_kind)
            prediction = apply_calibrator(spec, s[valid_mask])
            brier = float(brier_score_loss(y[valid_mask], prediction))
            loss = float(log_loss(y[valid_mask], prediction, labels=[0, 1]))
            records.append({"kind": kind, "brier": brier, "logloss": loss, "status": "SUCCESS"})
            valid_specs.append((spec, (brier, loss, index)))
        except Exception as exc:
            records.append(
                {
                    "kind": kind,
                    "brier": float("nan"),
                    "logloss": float("nan"),
                    "status": f"FAILED:{type(exc).__name__}",
                }
            )
    if not valid_specs:
        chosen_kind = "identity"
    else:
        chosen_kind = min(valid_specs, key=lambda item: item[1])[0].kind
    final_spec = _fit_single_calibrator(chosen_kind, y, s, source_score_kind)
    return final_spec, pd.DataFrame(records)


def threshold_grid(
    score: np.ndarray | Sequence[float],
    maximum_candidates: int = 500,
    include_probability_grid: bool = True,
) -> np.ndarray:
    """Return threshold candidates without an implicit alert-rate cap.

    Earlier versions sampled only the upper half of a large score distribution,
    which silently limited alerts to roughly 50% of rows. V6 deliberately has
    no alert-count/rate cap, so the grid spans the full empirical distribution
    and contains explicit alert-all and alert-none sentinels.
    """

    values = np.asarray(score, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.asarray([], dtype=np.float64)
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    unique = np.unique(values)
    candidate_limit = max(32, int(maximum_candidates))
    if len(unique) <= candidate_limit:
        candidates = unique
    else:
        quantiles = np.unique(
            np.concatenate(
                [
                    np.linspace(0.0, 0.50, 126),
                    np.linspace(0.505, 0.90, 80),
                    np.linspace(0.905, 0.99, 86),
                    np.linspace(0.991, 0.9999, 90),
                    np.asarray([1.0]),
                ]
            )
        )
        if len(quantiles) > candidate_limit:
            positions = np.linspace(0, len(quantiles) - 1, candidate_limit).round().astype(int)
            quantiles = quantiles[np.unique(positions)]
        candidates = np.quantile(values, quantiles)
    if include_probability_grid and minimum >= 0.0 and maximum <= 1.0:
        probability_points = np.concatenate(
            [
                np.linspace(0.01, 0.49, 49),
                np.linspace(0.50, 0.90, 81),
                np.linspace(0.905, 0.99, 18),
            ]
        )
        candidates = np.concatenate([candidates, probability_points])
    candidates = np.concatenate(
        [
            np.asarray(candidates, dtype=np.float64),
            np.asarray(
                [
                    np.nextafter(minimum, -np.inf),
                    minimum,
                    maximum,
                    np.nextafter(maximum, np.inf),
                ],
                dtype=np.float64,
            ),
        ]
    )
    candidates = np.unique(candidates)
    if len(candidates) > candidate_limit:
        positions = np.linspace(0, len(candidates) - 1, candidate_limit).round().astype(int)
        candidates = candidates[np.unique(positions)]
    return candidates[::-1]


def apply_precision_policy(
    policy: PrecisionPolicy | Mapping[str, Any],
    score: np.ndarray | Sequence[float],
    scopes: np.ndarray | Sequence[Any] | None = None,
) -> np.ndarray:
    if not isinstance(policy, PrecisionPolicy):
        policy = PrecisionPolicy.from_dict(policy)
    s = np.asarray(score, dtype=np.float64)
    if policy.kind == "global_threshold":
        threshold = policy.threshold if policy.threshold is not None else float("inf")
        return np.isfinite(s) & (s >= float(threshold))
    if policy.kind == "scope_threshold":
        if scopes is None:
            raise ValueError("scope_threshold policy에는 scope 값이 필요합니다")
        values = np.asarray(scopes, dtype=object)
        thresholds = policy.scope_thresholds or {}
        fallback = (
            policy.fallback_threshold
            if policy.fallback_threshold is not None
            else policy.threshold
            if policy.threshold is not None
            else float("inf")
        )
        result = np.zeros(len(s), dtype=bool)
        for index, (value, local_score) in enumerate(zip(values, s)):
            threshold = thresholds.get(str(value), fallback)
            result[index] = bool(np.isfinite(local_score) and local_score >= float(threshold))
        return result
    raise ValueError(f"지원하지 않는 precision policy: {policy.kind}")


def _candidate_passes(
    metrics: Mapping[str, Any],
    target_precision: float,
    minimum_precision_lcb: float,
    minimum_alerts: int,
    minimum_alert_days: int,
) -> bool:
    return bool(
        int(metrics.get("alerts", 0)) >= int(minimum_alerts)
        and int(metrics.get("alert_days", 0)) >= int(minimum_alert_days)
        and float(metrics.get("precision", float("nan"))) >= float(target_precision)
        and float(metrics.get("precision_wilson_lcb", float("nan"))) >= float(minimum_precision_lcb)
    )


def fast_alert_metrics(
    target: np.ndarray | Sequence[int],
    score: np.ndarray | Sequence[float],
    alert: np.ndarray | Sequence[bool],
    dates: np.ndarray | Sequence[Any],
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Cheap threshold metrics used during policy search.

    Event reconstruction and ranking metrics do not vary in a useful way across
    threshold candidates and are intentionally omitted here. The selected policy
    is re-evaluated with :func:`evaluate_alerts` afterward.
    """

    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(score, dtype=np.float64)
    selected = np.asarray(alert, dtype=bool)
    parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
    valid = np.isin(y, [0, 1]) & np.isfinite(s) & parsed.notna().to_numpy()
    y = y[valid]
    selected = selected[valid]
    d = parsed.loc[valid].to_numpy(dtype="datetime64[ns]")
    tp = int(np.sum(selected & (y == 1)))
    fp = int(np.sum(selected & (y == 0)))
    fn = int(np.sum((~selected) & (y == 1)))
    tn = int(np.sum((~selected) & (y == 0)))
    alerts = tp + fp
    positives = tp + fn
    negatives = fp + tn
    precision = safe_divide(tp, alerts)
    base_rate = safe_divide(positives, len(y))
    return {
        "rows": int(len(y)),
        "positives": positives,
        "negatives": negatives,
        "alerts": alerts,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "positive_rate": base_rate,
        "precision": precision,
        "recall": safe_divide(tp, positives),
        "alert_rate": safe_divide(alerts, len(y)),
        "specificity": safe_divide(tn, negatives),
        "false_positive_rate": safe_divide(fp, negatives),
        "precision_lift": safe_divide(precision, base_rate),
        "precision_wilson_lcb": wilson_lower_bound(tp, alerts, confidence_level),
        "precision_beta_lcb": beta_posterior_lower_bound(tp, fp, confidence_level),
        "date_count": int(pd.Series(d).nunique()),
        "alert_days": int(pd.Series(d[selected]).nunique()) if alerts else 0,
        "alerts_per_day_mean": safe_divide(alerts, int(pd.Series(d).nunique())),
    }


def evaluate_threshold_by_fold(
    frame: pd.DataFrame,
    threshold: float,
    confidence_level: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for fold_id, part in frame.groupby("fold_id", sort=True):
        alert = part["score"].to_numpy(dtype=np.float64) >= float(threshold)
        metrics = fast_alert_metrics(
            part["target"].to_numpy(dtype=np.int8),
            part["score"].to_numpy(dtype=np.float64),
            alert,
            part["date"].to_numpy(dtype="datetime64[ns]"),
            confidence_level,
        )
        records.append({"fold_id": int(fold_id), "threshold": float(threshold), **metrics})
    fold_frame = pd.DataFrame(records)
    pooled_alert = frame["score"].to_numpy(dtype=np.float64) >= float(threshold)
    pooled = fast_alert_metrics(
        frame["target"].to_numpy(dtype=np.int8),
        frame["score"].to_numpy(dtype=np.float64),
        pooled_alert,
        frame["date"].to_numpy(dtype="datetime64[ns]"),
        confidence_level,
    )
    return fold_frame, pooled


def select_global_precision_policy(
    frame: pd.DataFrame,
    target_precision: float = 0.70,
    minimum_precision_lcb: float = 0.60,
    minimum_alerts_per_fold: int = 30,
    minimum_alert_days_per_fold: int = 10,
    confidence_level: float = 0.95,
    required_fold_pass_rate: float = 1.0,
    source: str = "selection_forward_oof",
    maximum_threshold_candidates: int = 500,
) -> tuple[PrecisionPolicy, pd.DataFrame, pd.DataFrame]:
    required = {"fold_id", "date", "target", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"precision policy frame 컬럼 누락: {sorted(missing)}")
    candidates = threshold_grid(frame["score"].to_numpy(dtype=np.float64), maximum_threshold_candidates)
    search_records: list[dict[str, Any]] = []
    fold_records: list[pd.DataFrame] = []
    best: tuple[tuple[float, ...], float, pd.DataFrame, dict[str, Any]] | None = None
    for threshold in candidates:
        per_fold, pooled = evaluate_threshold_by_fold(frame, float(threshold), confidence_level)
        passes = per_fold.apply(
            lambda row: _candidate_passes(
                row,
                target_precision,
                minimum_precision_lcb,
                minimum_alerts_per_fold,
                minimum_alert_days_per_fold,
            ),
            axis=1,
        )
        pass_rate = float(passes.mean()) if len(passes) else 0.0
        gate_pass = bool(
            pass_rate + 1e-12 >= required_fold_pass_rate
            and _candidate_passes(
                pooled,
                target_precision,
                minimum_precision_lcb,
                minimum_alerts_per_fold * max(1, int(frame["fold_id"].nunique())),
                minimum_alert_days_per_fold,
            )
        )
        record = {
            "threshold": float(threshold),
            "fold_count": int(len(per_fold)),
            "fold_pass_count": int(passes.sum()),
            "fold_pass_rate": pass_rate,
            "gate_pass": gate_pass,
            "worst_precision": float(per_fold["precision"].min()) if len(per_fold) else float("nan"),
            "worst_precision_lcb": float(per_fold["precision_wilson_lcb"].min()) if len(per_fold) else float("nan"),
            "mean_precision": float(per_fold["precision"].mean()) if len(per_fold) else float("nan"),
            "mean_recall": float(per_fold["recall"].mean()) if len(per_fold) else float("nan"),
            "pooled_precision": float(pooled["precision"]),
            "pooled_recall": float(pooled["recall"]),
            "pooled_alerts": int(pooled["alerts"]),
            "pooled_alert_rate": float(pooled["alert_rate"]),
            "pooled_precision_lcb": float(pooled["precision_wilson_lcb"]),
        }
        search_records.append(record)
        local_fold = per_fold.copy()
        local_fold["fold_gate_pass"] = passes.to_numpy(dtype=bool)
        fold_records.append(local_fold)
        objective = (
            1.0 if gate_pass else 0.0,
            pass_rate,
            float(pooled["recall"]) if np.isfinite(pooled["recall"]) else -1.0,
            float(pooled["alerts"]),
            float(record["worst_precision_lcb"]) if np.isfinite(record["worst_precision_lcb"]) else -1.0,
            -float(threshold),
        )
        if best is None or objective > best[0]:
            best = (objective, float(threshold), per_fold, pooled)
    if best is None:
        raise RuntimeError("precision threshold 후보가 없습니다")
    _, threshold, per_fold, pooled = best
    passes = per_fold.apply(
        lambda row: _candidate_passes(
            row,
            target_precision,
            minimum_precision_lcb,
            minimum_alerts_per_fold,
            minimum_alert_days_per_fold,
        ),
        axis=1,
    )
    pass_rate = float(passes.mean()) if len(passes) else 0.0
    gate_pass = bool(
        pass_rate + 1e-12 >= required_fold_pass_rate
        and float(pooled["precision"]) >= target_precision
        and float(pooled["precision_wilson_lcb"]) >= minimum_precision_lcb
    )
    policy = PrecisionPolicy(
        kind="global_threshold",
        target_precision=float(target_precision),
        threshold=float(threshold),
        minimum_alerts=int(minimum_alerts_per_fold),
        minimum_alert_days=int(minimum_alert_days_per_fold),
        minimum_precision_lcb=float(minimum_precision_lcb),
        confidence_level=float(confidence_level),
        required_fold_pass_rate=float(required_fold_pass_rate),
        achieved_fold_pass_rate=pass_rate,
        achieved_worst_precision=float(per_fold["precision"].min()) if len(per_fold) else float("nan"),
        achieved_worst_precision_lcb=float(per_fold["precision_wilson_lcb"].min()) if len(per_fold) else float("nan"),
        achieved_mean_recall=float(per_fold["recall"].mean()) if len(per_fold) else float("nan"),
        achieved_alerts=int(pooled["alerts"]),
        gate_pass=gate_pass,
        source=source,
    )
    search_frame = pd.DataFrame(search_records).sort_values(
        ["gate_pass", "fold_pass_rate", "pooled_recall", "pooled_alerts", "worst_precision_lcb"],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    )
    all_folds = pd.concat(fold_records, ignore_index=True) if fold_records else pd.DataFrame()
    selected_folds = all_folds.loc[np.isclose(all_folds["threshold"], threshold)].copy()
    return policy, search_frame.reset_index(drop=True), selected_folds.reset_index(drop=True)


def select_scope_precision_policy(
    frame: pd.DataFrame,
    scope_column: str,
    target_precision: float = 0.70,
    minimum_precision_lcb: float = 0.60,
    minimum_alerts_per_scope: int = 20,
    minimum_alert_days_per_scope: int = 8,
    confidence_level: float = 0.95,
    required_fold_pass_rate: float = 1.0,
    source: str = "selection_scope_oof",
    maximum_threshold_candidates: int = 500,
) -> tuple[PrecisionPolicy, pd.DataFrame]:
    """Select scope thresholds without discarding temporal fold stability."""

    required = {"fold_id", "date", "target", "score", scope_column}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"scope precision frame 컬럼 누락: {sorted(missing)}")

    fold_count = max(1, int(frame["fold_id"].nunique()))
    fallback_min_alerts = max(5, int(math.ceil(minimum_alerts_per_scope / fold_count)))
    fallback_min_days = max(2, int(math.ceil(minimum_alert_days_per_scope / fold_count)))
    fallback, _, _ = select_global_precision_policy(
        frame,
        target_precision=target_precision,
        minimum_precision_lcb=minimum_precision_lcb,
        minimum_alerts_per_fold=fallback_min_alerts,
        minimum_alert_days_per_fold=fallback_min_days,
        confidence_level=confidence_level,
        required_fold_pass_rate=required_fold_pass_rate,
        source=source + "_fallback",
        maximum_threshold_candidates=maximum_threshold_candidates,
    )

    thresholds: dict[str, float] = {}
    records: list[dict[str, Any]] = []
    local_min_alerts = max(3, int(math.ceil(minimum_alerts_per_scope / fold_count)))
    local_min_days = max(2, int(math.ceil(minimum_alert_days_per_scope / fold_count)))
    for scope, part in frame.groupby(scope_column, dropna=False, sort=True):
        label = str(scope)
        if len(part) < max(100, minimum_alerts_per_scope * 3) or int(part["target"].sum()) < 10:
            records.append(
                {
                    "record_type": "scope_summary",
                    "scope": label,
                    "status": "FALLBACK_INSUFFICIENT_SUPPORT",
                    "rows": int(len(part)),
                    "positives": int(part["target"].sum()),
                }
            )
            continue
        try:
            local_policy, local_search, local_folds = select_global_precision_policy(
                part,
                target_precision=target_precision,
                minimum_precision_lcb=minimum_precision_lcb,
                minimum_alerts_per_fold=local_min_alerts,
                minimum_alert_days_per_fold=local_min_days,
                confidence_level=confidence_level,
                required_fold_pass_rate=required_fold_pass_rate,
                source=source + f"_{label}",
                maximum_threshold_candidates=maximum_threshold_candidates,
            )
        except Exception as exc:
            records.append(
                {
                    "record_type": "scope_summary",
                    "scope": label,
                    "status": f"FALLBACK_POLICY_ERROR:{type(exc).__name__}",
                }
            )
            continue
        if local_policy.gate_pass and local_policy.threshold is not None:
            thresholds[label] = float(local_policy.threshold)
            best_row = local_search.iloc[0]
            records.append(
                {
                    "record_type": "scope_summary",
                    "scope": label,
                    "status": "SCOPE_THRESHOLD",
                    "threshold": float(local_policy.threshold),
                    "fold_pass_rate": float(local_policy.achieved_fold_pass_rate or 0.0),
                    "precision": float(best_row["pooled_precision"]),
                    "recall": float(best_row["pooled_recall"]),
                    "alerts": int(best_row["pooled_alerts"]),
                }
            )
            for row in local_folds.to_dict(orient="records"):
                records.append(
                    {
                        "record_type": "scope_fold",
                        "scope": label,
                        "status": "PASS" if bool(row.get("fold_gate_pass", False)) else "FAIL",
                        **row,
                    }
                )
        else:
            records.append(
                {
                    "record_type": "scope_summary",
                    "scope": label,
                    "status": "FALLBACK_NO_STABLE_PRECISION_GATE",
                    "fold_pass_rate": float(local_policy.achieved_fold_pass_rate or 0.0),
                }
            )

    provisional = PrecisionPolicy(
        kind="scope_threshold",
        target_precision=target_precision,
        threshold=fallback.threshold,
        scope_column=scope_column,
        scope_thresholds=thresholds,
        fallback_threshold=fallback.threshold,
        minimum_alerts=minimum_alerts_per_scope,
        minimum_alert_days=minimum_alert_days_per_scope,
        minimum_precision_lcb=minimum_precision_lcb,
        confidence_level=confidence_level,
        required_fold_pass_rate=required_fold_pass_rate,
        source=source,
    )
    scopes = frame[scope_column].astype("string").fillna("__MISSING__").to_numpy(dtype=object)
    score = frame["score"].to_numpy(dtype=np.float64)
    alert = apply_precision_policy(provisional, score, scopes)

    combined_fold_records: list[dict[str, Any]] = []
    combined_passes: list[bool] = []
    annotated = frame.copy()
    annotated["__alert"] = alert
    for fold_id, part in annotated.groupby("fold_id", sort=True):
        metrics = fast_alert_metrics(
            part["target"].to_numpy(dtype=np.int8),
            part["score"].to_numpy(dtype=np.float64),
            part["__alert"].to_numpy(dtype=bool),
            part["date"].to_numpy(dtype="datetime64[ns]"),
            confidence_level,
        )
        passed = _candidate_passes(
            metrics,
            target_precision,
            minimum_precision_lcb,
            fallback_min_alerts,
            fallback_min_days,
        )
        combined_passes.append(bool(passed))
        combined_fold_records.append(
            {
                "record_type": "combined_fold",
                "scope": "__COMBINED__",
                "status": "PASS" if passed else "FAIL",
                "fold_id": int(fold_id),
                **metrics,
            }
        )

    pooled = evaluate_alerts(
        frame["target"].to_numpy(dtype=np.int8),
        score,
        alert,
        frame["date"].to_numpy(dtype="datetime64[ns]"),
        frame["ticker"].to_numpy(dtype=object) if "ticker" in frame else None,
        confidence_level,
    )
    pass_rate = float(np.mean(combined_passes)) if combined_passes else 0.0
    pooled_pass = _candidate_passes(
        pooled,
        target_precision,
        minimum_precision_lcb,
        minimum_alerts_per_scope,
        minimum_alert_days_per_scope,
    )
    gate_pass = bool(pass_rate + 1e-12 >= required_fold_pass_rate and pooled_pass)
    policy = dataclasses.replace(
        provisional,
        achieved_fold_pass_rate=pass_rate,
        achieved_worst_precision=(
            float(min(record["precision"] for record in combined_fold_records))
            if combined_fold_records
            else float("nan")
        ),
        achieved_worst_precision_lcb=(
            float(min(record["precision_wilson_lcb"] for record in combined_fold_records))
            if combined_fold_records
            else float("nan")
        ),
        achieved_mean_recall=(
            float(np.mean([record["recall"] for record in combined_fold_records]))
            if combined_fold_records
            else float("nan")
        ),
        achieved_alerts=int(pooled["alerts"]),
        gate_pass=gate_pass,
    )
    records.extend(combined_fold_records)
    records.append(
        {
            "record_type": "combined_pooled",
            "scope": "__COMBINED__",
            "status": "PASS" if gate_pass else "FAIL",
            "fold_pass_rate": pass_rate,
            **pooled,
        }
    )
    return policy, pd.DataFrame(records)


def build_precision_curve(
    target: np.ndarray | Sequence[int],
    score: np.ndarray | Sequence[float],
    dates: np.ndarray | Sequence[Any],
    tickers: np.ndarray | Sequence[Any] | None = None,
    maximum_candidates: int = 300,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(score, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for threshold in threshold_grid(s, maximum_candidates):
        metrics = fast_alert_metrics(y, s, s >= threshold, dates, confidence_level)
        records.append({"threshold": float(threshold), **metrics})
    return pd.DataFrame(records).sort_values("threshold", ascending=False, kind="mergesort")


def maximum_recall_at_precision(
    curve: pd.DataFrame,
    target_precision: float,
    minimum_alerts: int = 1,
    minimum_alert_days: int = 1,
    minimum_precision_lcb: float | None = None,
) -> dict[str, Any]:
    """Return maximum Recall under point-precision and support constraints."""

    precision = pd.to_numeric(curve.get("precision"), errors="coerce")
    alerts = pd.to_numeric(curve.get("alerts"), errors="coerce")
    alert_days = pd.to_numeric(curve.get("alert_days"), errors="coerce")
    point_mask = (
        (precision >= target_precision)
        & (alerts >= minimum_alerts)
        & (alert_days >= minimum_alert_days)
    )
    gate_mask = point_mask.copy()
    if minimum_precision_lcb is not None and "precision_wilson_lcb" in curve:
        gate_mask &= (
            pd.to_numeric(curve["precision_wilson_lcb"], errors="coerce")
            >= float(minimum_precision_lcb)
        )
    point_eligible = curve.loc[point_mask]
    eligible = curve.loc[gate_mask]
    if eligible.empty:
        best = curve.sort_values(
            ["precision", "recall", "alerts"],
            ascending=[False, False, False],
            kind="mergesort",
        ).head(1)
        point_best = point_eligible.sort_values(
            ["recall", "alerts"],
            ascending=[False, False],
            kind="mergesort",
        ).head(1)
        return {
            "gate_feasible": False,
            "point_precision_feasible": bool(len(point_eligible)),
            "maximum_recall": 0.0,
            "point_precision_maximum_recall": (
                float(point_best.iloc[0]["recall"]) if len(point_best) else 0.0
            ),
            "threshold": float(best.iloc[0]["threshold"]) if len(best) else float("nan"),
            "best_available_precision": (
                float(best.iloc[0]["precision"]) if len(best) else float("nan")
            ),
            "best_available_precision_lcb": (
                float(best.iloc[0].get("precision_wilson_lcb", float("nan")))
                if len(best)
                else float("nan")
            ),
            "alerts": int(best.iloc[0]["alerts"]) if len(best) else 0,
        }
    row = eligible.sort_values(
        ["recall", "alerts", "precision_wilson_lcb"],
        ascending=[False, False, False],
        kind="mergesort",
    ).iloc[0]
    return {
        "gate_feasible": True,
        "point_precision_feasible": True,
        "maximum_recall": float(row["recall"]),
        "point_precision_maximum_recall": float(row["recall"]),
        "threshold": float(row["threshold"]),
        "best_available_precision": float(row["precision"]),
        "best_available_precision_lcb": float(
            row.get("precision_wilson_lcb", float("nan"))
        ),
        "alerts": int(row["alerts"]),
        "alert_days": int(row.get("alert_days", 0)),
        "alert_rate": float(row["alert_rate"]),
    }


def forward_training_folds(available_folds: Sequence[int], heldout_fold: int) -> list[int]:
    return [int(value) for value in sorted(set(int(v) for v in available_folds)) if int(value) < int(heldout_fold)]


def delayed_maturity_mask(
    history_dates: np.ndarray | Sequence[Any],
    current_date: Any,
    all_trading_dates: Sequence[Any],
    horizon_days: int = 3,
) -> np.ndarray:
    parsed_history = pd.to_datetime(pd.Series(history_dates), errors="coerce").to_numpy(dtype="datetime64[ns]")
    trading = pd.DatetimeIndex(pd.to_datetime(pd.Series(all_trading_dates), errors="coerce").dropna().unique()).sort_values()
    current = pd.Timestamp(current_date)
    position = int(trading.searchsorted(current, side="left"))
    mature_position = position - int(horizon_days)
    if mature_position < 0:
        return np.zeros(len(parsed_history), dtype=bool)
    mature_date = np.datetime64(trading[mature_position])
    return parsed_history <= mature_date



def select_online_precision_threshold_fast(
    frame: pd.DataFrame,
    target_precision: float = 0.70,
    minimum_precision_lcb: float = 0.60,
    minimum_alerts: int = 30,
    minimum_alert_days: int = 10,
    confidence_level: float = 0.95,
    maximum_threshold_candidates: int = 200,
) -> tuple[float, bool, dict[str, Any]]:
    """Select a single-history threshold without pandas groupby per candidate.

    This path is used by delayed online simulation, where repeatedly invoking
    the multi-fold policy search can dominate runtime.  There is no alert-count
    cap: candidates span the full score distribution.  If no statistically safe
    threshold exists, the function returns ``inf`` and therefore abstains.
    """

    required = {"date", "target", "score"}
    missing = required - set(frame.columns)
    if missing:
        raise KeyError(f"online precision frame 컬럼 누락: {sorted(missing)}")
    y = pd.to_numeric(frame["target"], errors="coerce").to_numpy(dtype=np.float64)
    s = pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=np.float64)
    parsed = pd.to_datetime(frame["date"], errors="coerce")
    valid = np.isin(y, [0.0, 1.0]) & np.isfinite(s) & parsed.notna().to_numpy()
    y = y[valid].astype(np.int8, copy=False)
    s = s[valid]
    if not len(y):
        return float("inf"), False, {"status": "NO_VALID_ROWS", "alerts": 0}
    date_codes, unique_dates = pd.factorize(parsed.loc[valid], sort=True)
    candidates = threshold_grid(s, maximum_threshold_candidates)
    positives = int(np.sum(y == 1))
    rows = int(len(y))
    best_gate: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
    best_any: tuple[tuple[float, ...], float, dict[str, Any]] | None = None
    for threshold in candidates:
        selected = s >= float(threshold)
        alerts = int(np.sum(selected))
        tp = int(np.sum(selected & (y == 1)))
        fp = alerts - tp
        precision = safe_divide(tp, alerts)
        recall = safe_divide(tp, positives)
        alert_days = int(np.unique(date_codes[selected]).size) if alerts else 0
        lcb = wilson_lower_bound(tp, alerts, confidence_level)
        gate = bool(
            alerts >= int(minimum_alerts)
            and alert_days >= int(minimum_alert_days)
            and np.isfinite(precision)
            and precision + 1e-12 >= float(target_precision)
            and lcb + 1e-12 >= float(minimum_precision_lcb)
        )
        metrics = {
            "status": "PASS" if gate else "FAIL",
            "threshold": float(threshold),
            "rows": rows,
            "date_count": int(len(unique_dates)),
            "positives": positives,
            "alerts": alerts,
            "true_positives": tp,
            "false_positives": fp,
            "precision": precision,
            "recall": recall,
            "alert_rate": safe_divide(alerts, rows),
            "alert_days": alert_days,
            "precision_wilson_lcb": lcb,
            "gate_pass": gate,
        }
        objective = (
            recall if np.isfinite(recall) else -1.0,
            float(alerts),
            lcb if np.isfinite(lcb) else -1.0,
            precision if np.isfinite(precision) else -1.0,
            -float(threshold),
        )
        if best_any is None or objective > best_any[0]:
            best_any = (objective, float(threshold), metrics)
        if gate and (best_gate is None or objective > best_gate[0]):
            best_gate = (objective, float(threshold), metrics)
    if best_gate is None:
        diagnostics = dict(best_any[2]) if best_any is not None else {"alerts": 0}
        diagnostics["status"] = "NO_SAFE_THRESHOLD"
        diagnostics["gate_pass"] = False
        return float("inf"), False, diagnostics
    return float(best_gate[1]), True, dict(best_gate[2])

def simulate_delayed_online_threshold(
    historical_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    target_precision: float = 0.70,
    minimum_precision_lcb: float = 0.60,
    minimum_alerts: int = 30,
    minimum_alert_days: int = 10,
    confidence_level: float = 0.95,
    horizon_days: int = 3,
    lookback_days: int = 504,
    update_every_days: int = 5,
    maximum_threshold_candidates: int = 200,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Simulate an online threshold with a three-trading-day label delay.

    Validation labels are appended to the calibration history only after their
    horizon has matured. The current or future validation target is never used
    to choose the threshold applied on the current date.
    """

    history = historical_frame.copy()
    validation = validation_frame.copy().sort_values("date", kind="mergesort")
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    validation["date"] = pd.to_datetime(validation["date"], errors="coerce")
    all_dates = pd.DatetimeIndex(
        pd.concat([history["date"], validation["date"]], ignore_index=True).dropna().unique()
    ).sort_values()
    output_alert = np.zeros(len(validation), dtype=bool)
    audit: list[dict[str, Any]] = []
    pending = validation[["date", "target", "score", *(["ticker"] if "ticker" in validation else [])]].copy()
    latest_threshold = float("inf")
    unique_validation_dates = pd.DatetimeIndex(validation["date"].unique()).sort_values()
    for date_index, current_date in enumerate(unique_validation_dates):
        maturity = delayed_maturity_mask(pending["date"], current_date, all_dates, horizon_days)
        mature_rows = pending.loc[maturity & (pending["date"] < current_date)]
        if len(mature_rows):
            history = pd.concat([history, mature_rows], ignore_index=True)
            pending = pending.loc[~(maturity & (pending["date"] < current_date))].copy()
        if date_index % max(1, int(update_every_days)) == 0:
            history_dates = pd.DatetimeIndex(history["date"].dropna().unique()).sort_values()
            if len(history_dates) > lookback_days:
                cutoff = history_dates[-int(lookback_days)]
                calibration = history.loc[history["date"] >= cutoff].copy()
            else:
                calibration = history.copy()
            calibration["fold_id"] = 0
            if len(calibration) >= minimum_alerts * 3 and int(calibration["target"].sum()) >= 10:
                latest_threshold, gate, threshold_metrics = select_online_precision_threshold_fast(
                    calibration,
                    target_precision=target_precision,
                    minimum_precision_lcb=minimum_precision_lcb,
                    minimum_alerts=minimum_alerts,
                    minimum_alert_days=minimum_alert_days,
                    confidence_level=confidence_level,
                    maximum_threshold_candidates=maximum_threshold_candidates,
                )
            else:
                latest_threshold = float("inf")
                gate = False
                threshold_metrics = {"status": "INSUFFICIENT_HISTORY", "alerts": 0}
        day_mask = validation["date"].eq(current_date).to_numpy()
        output_alert[day_mask] = validation.loc[day_mask, "score"].to_numpy(dtype=np.float64) >= latest_threshold
        audit.append(
            {
                "date": pd.Timestamp(current_date),
                "threshold": float(latest_threshold),
                "history_rows": int(len(history)),
                "history_latest_date": history["date"].max(),
                "gate_available": bool(gate),
                "threshold_status": str(threshold_metrics.get("status", "UNKNOWN")),
                "calibration_precision": threshold_metrics.get("precision"),
                "calibration_recall": threshold_metrics.get("recall"),
                "calibration_alerts": threshold_metrics.get("alerts"),
                "alerts": int(output_alert[day_mask].sum()),
            }
        )
    return output_alert, pd.DataFrame(audit)


def _infer_signal_role(signal: str, family_map: Mapping[str, str]) -> str:
    family = str(family_map.get(signal, "unknown")).lower()
    name = str(signal).lower()
    if name.endswith("__seed_agreement") or "seed_agreement" in family:
        return "agreement"
    if "crash" in name or "crash" in family:
        return "crash"
    if "strong7" in name or "strong10" in name or "strong" in family:
        return "strong_surge"
    if "direction" in name or "two_stage" in name or "direction" in family:
        return "direction_up"
    return "surge"


def build_meta_feature_frame(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    date_column: str = "date",
    scope_columns: Sequence[str] = ("market", "bucket"),
    signal_roles: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build high-precision meta features with correct signal semantics.

    Surge predictions, crash probabilities and seed-agreement diagnostics are
    separated. Agreement and crash-risk values are never averaged as if they
    were surge probabilities.
    """

    missing = [column for column in signal_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"meta signal 누락: {missing[:20]}")
    roles = {
        str(column): str(
            (signal_roles or {}).get(
                column,
                _infer_signal_role(str(column), family_map),
            )
        )
        for column in signal_columns
    }

    output = pd.DataFrame(index=frame.index)
    values_by_signal: dict[str, np.ndarray] = {}
    for column in signal_columns:
        values = np.clip(
            pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64),
            1e-7,
            1.0 - 1e-7,
        )
        output[f"signal__{column}"] = values
        output[f"rank__{column}"] = datewise_rank_normalize(values, frame[date_column])
        values_by_signal[str(column)] = values

    positive_columns = [
        str(column)
        for column in signal_columns
        if roles[str(column)] in {"surge", "strong_surge", "direction_up"}
    ]
    crash_columns = [
        str(column) for column in signal_columns if roles[str(column)] == "crash"
    ]
    agreement_columns = [
        str(column) for column in signal_columns if roles[str(column)] == "agreement"
    ]
    strong_columns = [
        str(column)
        for column in signal_columns
        if roles[str(column)] == "strong_surge"
    ]
    direction_columns = [
        str(column)
        for column in signal_columns
        if roles[str(column)] == "direction_up"
    ]
    if not positive_columns:
        raise ValueError("surge-oriented meta signal이 없습니다")

    positive_matrix = np.column_stack(
        [values_by_signal[column] for column in positive_columns]
    )
    output["ensemble_mean"] = np.nanmean(positive_matrix, axis=1)
    output["ensemble_median"] = np.nanmedian(positive_matrix, axis=1)
    output["ensemble_std"] = np.nanstd(positive_matrix, axis=1)
    output["ensemble_min"] = np.nanmin(positive_matrix, axis=1)
    output["ensemble_max"] = np.nanmax(positive_matrix, axis=1)
    output["ensemble_q25"] = np.nanquantile(positive_matrix, 0.25, axis=1)
    output["ensemble_q75"] = np.nanquantile(positive_matrix, 0.75, axis=1)
    output["ensemble_q90"] = np.nanquantile(positive_matrix, 0.90, axis=1)
    output["consensus_ge_050"] = np.mean(positive_matrix >= 0.50, axis=1)
    output["consensus_ge_060"] = np.mean(positive_matrix >= 0.60, axis=1)
    output["consensus_ge_070"] = np.mean(positive_matrix >= 0.70, axis=1)
    output["consensus_ge_080"] = np.mean(positive_matrix >= 0.80, axis=1)
    output["uncertainty_lcb_z1"] = (
        output["ensemble_mean"] - output["ensemble_std"]
    )
    output["uncertainty_lcb_z2"] = (
        output["ensemble_mean"] - 2.0 * output["ensemble_std"]
    )

    families: dict[str, list[str]] = {}
    for signal in signal_columns:
        families.setdefault(
            str(family_map.get(signal, "unknown")),
            [],
        ).append(str(signal))
    positive_family_mean_columns: list[str] = []
    for family, columns in sorted(families.items()):
        values = np.column_stack([values_by_signal[column] for column in columns])
        safe_name = family.replace(" ", "_")
        mean_column = f"family_mean__{safe_name}"
        output[mean_column] = np.nanmean(values, axis=1)
        output[f"family_std__{safe_name}"] = np.nanstd(values, axis=1)
        output[f"family_max__{safe_name}"] = np.nanmax(values, axis=1)
        family_roles = {roles[column] for column in columns}
        if (
            family_roles & {"surge", "strong_surge", "direction_up"}
            and not family_roles & {"crash", "agreement"}
        ):
            positive_family_mean_columns.append(mean_column)

    if agreement_columns:
        agreement_matrix = np.column_stack(
            [values_by_signal[column] for column in agreement_columns]
        )
        output["seed_agreement_mean"] = np.nanmean(agreement_matrix, axis=1)
        output["seed_agreement_min"] = np.nanmin(agreement_matrix, axis=1)
        output["seed_agreement_std"] = np.nanstd(agreement_matrix, axis=1)
    else:
        output["seed_agreement_mean"] = 1.0
        output["seed_agreement_min"] = 1.0
        output["seed_agreement_std"] = 0.0

    if strong_columns:
        strong_matrix = np.column_stack(
            [values_by_signal[column] for column in strong_columns]
        )
        output["strong_surge_mean"] = np.nanmean(strong_matrix, axis=1)
        output["strong_surge_max"] = np.nanmax(strong_matrix, axis=1)
    else:
        output["strong_surge_mean"] = output["ensemble_mean"]
        output["strong_surge_max"] = output["ensemble_max"]
    output["d3_minus_strong_margin"] = (
        output["ensemble_mean"] - output["strong_surge_mean"]
    )

    if crash_columns:
        crash_matrix = np.column_stack(
            [values_by_signal[column] for column in crash_columns]
        )
        output["crash_risk_mean"] = np.nanmean(crash_matrix, axis=1)
        output["crash_risk_max"] = np.nanmax(crash_matrix, axis=1)
    else:
        output["crash_risk_mean"] = 0.0
        output["crash_risk_max"] = 0.0
    output["crash_safety_mean"] = 1.0 - output["crash_risk_mean"]
    output["crash_safety_min"] = 1.0 - output["crash_risk_max"]
    output["surge_safety_margin"] = (
        output["ensemble_mean"] - output["crash_risk_mean"]
    )
    output["surge_safety_product"] = (
        output["ensemble_mean"] * output["crash_safety_mean"]
    )
    output["conservative_q25_safety"] = (
        output["ensemble_q25"] * output["crash_safety_min"]
    )
    output["consensus_lcb_safety"] = (
        np.clip(output["uncertainty_lcb_z1"], 0.0, 1.0)
        * output["crash_safety_mean"]
        * output["seed_agreement_mean"]
    )
    output["strong_safety_product"] = (
        output["strong_surge_mean"]
        * output["crash_safety_mean"]
        * output["seed_agreement_mean"]
    )
    if direction_columns:
        direction_matrix = np.column_stack(
            [values_by_signal[column] for column in direction_columns]
        )
        output["direction_up_mean"] = np.nanmean(direction_matrix, axis=1)

    categories: dict[str, list[str]] = {}
    for scope in scope_columns:
        if scope not in frame.columns:
            continue
        values = frame[scope].astype("string").fillna("__MISSING__")
        unique = sorted(str(value) for value in values.unique())
        categories[scope] = unique
        for value in unique:
            output[f"scope__{scope}__{value}"] = values.eq(value).to_numpy(
                dtype=np.uint8
            )

    output = (
        output.replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
    )
    spec = {
        "feature_columns": list(output.columns),
        "signal_columns": [str(value) for value in signal_columns],
        "signal_roles": roles,
        "positive_signal_columns": positive_columns,
        "crash_signal_columns": crash_columns,
        "agreement_signal_columns": agreement_columns,
        "positive_family_mean_columns": positive_family_mean_columns,
        "family_map": {str(key): str(value) for key, value in family_map.items()},
        "scope_categories": categories,
        "date_column": date_column,
    }
    return output, spec


def apply_meta_feature_spec(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
) -> pd.DataFrame:
    generated, _ = build_meta_feature_frame(
        frame,
        [str(value) for value in spec["signal_columns"]],
        {str(key): str(value) for key, value in spec["family_map"].items()},
        date_column=str(spec.get("date_column", "date")),
        scope_columns=list(spec.get("scope_categories", {}).keys()),
        signal_roles={
            str(key): str(value)
            for key, value in spec.get("signal_roles", {}).items()
        },
    )
    for column in spec["feature_columns"]:
        if column not in generated:
            generated[column] = 0.0
    return generated[
        [str(value) for value in spec["feature_columns"]]
    ].astype(np.float32)


def build_scope_metrics(
    frame: pd.DataFrame,
    scope_columns: Sequence[str],
    confidence_level: float = 0.95,
    minimum_rows: int = 30,
    minimum_alerts: int = 5,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for role, role_part in frame.groupby("fold_role", sort=False):
        for column in scope_columns:
            if column not in role_part.columns:
                continue
            for value, part in role_part.groupby(column, dropna=False, sort=True):
                if len(part) < minimum_rows or int(part["alert"].sum()) < minimum_alerts:
                    continue
                metrics = evaluate_alerts(
                    part["target"].to_numpy(dtype=np.int8),
                    part["score"].to_numpy(dtype=np.float64),
                    part["alert"].to_numpy(dtype=bool),
                    part["date"].to_numpy(dtype="datetime64[ns]"),
                    part["ticker"].to_numpy(dtype=object) if "ticker" in part else None,
                    confidence_level,
                )
                records.append(
                    {
                        "fold_role": str(role),
                        "scope_column": column,
                        "scope_value": str(value),
                        **metrics,
                    }
                )
    return pd.DataFrame(records)
