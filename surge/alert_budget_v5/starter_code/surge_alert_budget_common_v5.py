from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from surge_model_zoo_common import (
    _selected_metrics,
    datewise_rank_normalize,
    safe_binary_metrics,
)


V5_SCHEMA = "crashwatch_surge_alert_budget_v5"


@dataclass(frozen=True)
class DailyBudgetPolicy:
    """A frozen date-wise alert budget selected from selection-fold OOF only."""

    kind: str = "daily_budget"
    daily_fraction: float = 0.40
    max_alerts_per_day: int | None = None
    target_recall: float = 0.70
    selection_recall_buffer: float = 0.0
    minimum_precision_lift: float = 1.10
    maximum_alert_rate: float = 0.40
    required_fold_pass_rate: float = 1.0
    allocation_column: str | None = None
    allocation_biases: dict[str, float] | None = None
    achieved_fold_pass_rate: float | None = None
    achieved_worst_recall: float | None = None
    achieved_mean_recall: float | None = None
    achieved_worst_precision_lift: float | None = None
    achieved_mean_alert_rate: float | None = None
    gate_pass: bool = False
    source: str = "selection_crossfit"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DailyBudgetPolicy":
        allowed = {field.name for field in dataclasses.fields(cls)}
        values = {key: value for key, value in payload.items() if key in allowed}
        return cls(**values)


def _as_datetime64(values: Sequence[Any] | np.ndarray) -> np.ndarray:
    return pd.to_datetime(pd.Series(values), errors="coerce").to_numpy(dtype="datetime64[ns]")


def _coerce_group_values(values: Sequence[Any] | np.ndarray | None, length: int) -> np.ndarray:
    if values is None:
        return np.full(length, "__ALL__", dtype=object)
    series = pd.Series(values, dtype="string").fillna("__MISSING__").str.strip()
    series = series.replace("", "__MISSING__")
    return series.to_numpy(dtype=object)


def apply_allocation_biases(
    scores: np.ndarray,
    groups: Sequence[Any] | np.ndarray | None,
    biases: Mapping[str, float] | None,
) -> np.ndarray:
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    if not biases:
        return adjusted
    values = _coerce_group_values(groups, len(adjusted))
    for group, bias in biases.items():
        adjusted[values == str(group)] += float(bias)
    return adjusted


def daily_budget_selection(
    scores: np.ndarray,
    dates: Sequence[Any] | np.ndarray,
    fraction: float,
    max_alerts_per_day: int | None = None,
    groups: Sequence[Any] | np.ndarray | None = None,
    allocation_biases: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Select the highest-scoring rows independently on each trading date.

    The count is min(floor(available_rows * fraction), max_alerts_per_day).
    ``floor`` is deliberate: the requested fraction is a hard upper bound, not
    a soft target that may be exceeded by integer rounding.  On a date whose
    universe is too small for even one alert under the cap, zero rows are
    selected.  This keeps the operational burden invariant to calibration and
    prevents an absolute-score shift from selecting nearly the whole universe.
    """

    if not 0 < float(fraction) <= 1.0:
        raise ValueError("fraction은 (0, 1] 범위여야 합니다")
    if max_alerts_per_day is not None and int(max_alerts_per_day) < 1:
        raise ValueError("max_alerts_per_day는 1 이상이어야 합니다")

    raw_scores = np.asarray(scores, dtype=np.float64)
    date_values = _as_datetime64(dates)
    adjusted = apply_allocation_biases(raw_scores, groups, allocation_biases)
    selected = np.zeros(len(raw_scores), dtype=bool)
    valid = np.isfinite(adjusted) & ~pd.isna(date_values)
    if not valid.any():
        return selected

    valid_indices = np.flatnonzero(valid)
    date_codes, unique_dates = pd.factorize(date_values[valid], sort=True)
    for code in range(len(unique_dates)):
        local = valid_indices[date_codes == code]
        count = int(math.floor(len(local) * float(fraction) + 1e-12))
        if max_alerts_per_day is not None:
            count = min(count, int(max_alerts_per_day))
        count = min(max(0, count), len(local))
        if count == 0:
            continue
        # mergesort plus original row order makes tie handling deterministic.
        order = np.argsort(-adjusted[local], kind="mergesort")[:count]
        selected[local[order]] = True
    return selected


def event_episode_metrics(
    target: np.ndarray,
    selected: np.ndarray,
    dates: Sequence[Any] | np.ndarray,
    tickers: Sequence[Any] | np.ndarray | None,
) -> dict[str, float | int]:
    """Collapse consecutive positive lead rows into ticker-level surge episodes."""

    y = np.asarray(target, dtype=np.int8)
    alert = np.asarray(selected, dtype=bool)
    d = _as_datetime64(dates)
    if tickers is None:
        return {
            "event_count": 0,
            "captured_events": 0,
            "event_recall": float("nan"),
            "alerts_per_captured_event": float("nan"),
        }
    ticker_values = _coerce_group_values(tickers, len(y))
    frame = pd.DataFrame(
        {
            "ticker": ticker_values,
            "date": d,
            "target": y,
            "alert": alert,
            "row_order": np.arange(len(y), dtype=np.int64),
        }
    )
    frame = frame.loc[frame["date"].notna()].sort_values(
        ["ticker", "date", "row_order"], kind="mergesort"
    )
    event_count = 0
    captured = 0
    for _, part in frame.groupby("ticker", sort=False):
        positives = part["target"].to_numpy(dtype=np.int8) == 1
        alerts = part["alert"].to_numpy(dtype=bool)
        in_event = False
        event_alert = False
        for is_positive, is_alert in zip(positives, alerts, strict=True):
            if is_positive:
                if not in_event:
                    in_event = True
                    event_alert = False
                    event_count += 1
                event_alert = event_alert or bool(is_alert)
            elif in_event:
                captured += int(event_alert)
                in_event = False
                event_alert = False
        if in_event:
            captured += int(event_alert)
    return {
        "event_count": int(event_count),
        "captured_events": int(captured),
        "event_recall": float(captured / event_count) if event_count else float("nan"),
        "alerts_per_captured_event": (
            float(np.sum(alert) / captured) if captured else float("nan")
        ),
    }


def evaluate_daily_budget(
    target: np.ndarray,
    scores: np.ndarray,
    dates: Sequence[Any] | np.ndarray,
    fraction: float,
    max_alerts_per_day: int | None = None,
    groups: Sequence[Any] | np.ndarray | None = None,
    allocation_biases: Mapping[str, float] | None = None,
    tickers: Sequence[Any] | np.ndarray | None = None,
) -> dict[str, float | int]:
    selected = daily_budget_selection(
        scores,
        dates,
        fraction,
        max_alerts_per_day=max_alerts_per_day,
        groups=groups,
        allocation_biases=allocation_biases,
    )
    metrics = dict(_selected_metrics(target, selected))
    date_values = pd.to_datetime(pd.Series(dates), errors="coerce")
    counts = pd.Series(selected.astype(np.int8)).groupby(date_values, sort=False).sum()
    metrics.update(
        {
            "daily_fraction": float(fraction),
            "max_alerts_per_day": int(max_alerts_per_day) if max_alerts_per_day is not None else None,
            "date_count": int(counts.size),
            "alerts_per_day_mean": float(counts.mean()) if len(counts) else float("nan"),
            "alerts_per_day_median": float(counts.median()) if len(counts) else float("nan"),
            "alerts_per_day_p95": float(counts.quantile(0.95)) if len(counts) else float("nan"),
            "alerts_per_day_max": int(counts.max()) if len(counts) else 0,
        }
    )
    metrics.update(event_episode_metrics(target, selected, dates, tickers))
    return metrics


def apply_daily_budget_policy(
    policy: DailyBudgetPolicy,
    scores: np.ndarray,
    dates: Sequence[Any] | np.ndarray,
    groups: Sequence[Any] | np.ndarray | None = None,
) -> np.ndarray:
    if policy.kind != "daily_budget":
        raise ValueError(f"지원하지 않는 V5 정책: {policy.kind}")
    return daily_budget_selection(
        scores,
        dates,
        policy.daily_fraction,
        max_alerts_per_day=policy.max_alerts_per_day,
        groups=groups,
        allocation_biases=policy.allocation_biases,
    )


def _metric_is_finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def _fold_pass(
    metrics: Mapping[str, Any],
    target_recall: float,
    minimum_precision_lift: float,
    maximum_alert_rate: float,
) -> bool:
    return bool(
        _metric_is_finite(metrics.get("recall"))
        and float(metrics["recall"]) >= float(target_recall)
        and _metric_is_finite(metrics.get("lift"))
        and float(metrics["lift"]) >= float(minimum_precision_lift)
        and _metric_is_finite(metrics.get("alert_rate"))
        and float(metrics["alert_rate"]) <= float(maximum_alert_rate) + 1e-12
    )


def summarize_budget_fold_rows(
    fold_rows: pd.DataFrame,
    target_recall: float,
    minimum_precision_lift: float,
    maximum_alert_rate: float,
    required_fold_pass_rate: float,
) -> dict[str, Any]:
    if fold_rows.empty:
        raise ValueError("fold_rows가 비어 있습니다")
    recalls = pd.to_numeric(fold_rows["recall"], errors="coerce").to_numpy(dtype=np.float64)
    lifts = pd.to_numeric(fold_rows["lift"], errors="coerce").to_numpy(dtype=np.float64)
    alert_rates = pd.to_numeric(fold_rows["alert_rate"], errors="coerce").to_numpy(dtype=np.float64)
    passes = np.asarray(
        [
            _fold_pass(row, target_recall, minimum_precision_lift, maximum_alert_rate)
            for row in fold_rows.to_dict(orient="records")
        ],
        dtype=bool,
    )
    valid_recall = recalls[np.isfinite(recalls)]
    valid_lift = lifts[np.isfinite(lifts)]
    valid_alert = alert_rates[np.isfinite(alert_rates)]
    fold_pass_rate = float(np.mean(passes)) if len(passes) else 0.0
    return {
        "fold_count": int(len(fold_rows)),
        "fold_pass_count": int(passes.sum()),
        "fold_pass_rate": fold_pass_rate,
        "worst_recall": float(np.min(valid_recall)) if len(valid_recall) else float("nan"),
        "mean_recall": float(np.mean(valid_recall)) if len(valid_recall) else float("nan"),
        "median_recall": float(np.median(valid_recall)) if len(valid_recall) else float("nan"),
        "worst_precision_lift": float(np.min(valid_lift)) if len(valid_lift) else float("nan"),
        "mean_precision_lift": float(np.mean(valid_lift)) if len(valid_lift) else float("nan"),
        "maximum_fold_alert_rate": float(np.max(valid_alert)) if len(valid_alert) else float("nan"),
        "mean_alert_rate": float(np.mean(valid_alert)) if len(valid_alert) else float("nan"),
        "gate_pass": bool(fold_pass_rate + 1e-12 >= float(required_fold_pass_rate)),
    }


def select_minimax_daily_budget_policy(
    frame: pd.DataFrame,
    fraction_grid: Sequence[float],
    target_recall: float,
    selection_recall_buffer: float,
    minimum_precision_lift: float,
    maximum_alert_rate: float,
    required_fold_pass_rate: float,
    max_alerts_per_day: int | None = None,
    score_column: str = "score",
    fold_column: str = "fold_id",
    target_column: str = "target",
    date_column: str = "date",
    ticker_column: str = "ticker",
    allocation_column: str | None = None,
    allocation_biases: Mapping[str, float] | None = None,
    source: str = "selection_crossfit",
) -> tuple[DailyBudgetPolicy, pd.DataFrame, pd.DataFrame]:
    """Choose the smallest budget satisfying a minimax selection-fold gate.

    If no budget is feasible under the hard cap, the best diagnostic candidate
    is returned with gate_pass=False.  The function never silently relaxes the
    user-specified maximum alert rate.
    """

    frame = frame.reset_index(drop=True).copy()
    required_columns = {fold_column, target_column, date_column, score_column}
    missing = required_columns - set(frame.columns)
    if missing:
        raise KeyError(f"budget policy 입력 컬럼 누락: {sorted(missing)}")
    fractions = sorted(
        {
            float(value)
            for value in fraction_grid
            if 0 < float(value) <= min(1.0, float(maximum_alert_rate) + 1e-12)
        }
    )
    if not fractions:
        raise ValueError("maximum_alert_rate 이하의 daily fraction 후보가 없습니다")
    gate_recall = min(0.999, float(target_recall) + float(selection_recall_buffer))
    fold_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    group_values = frame[allocation_column].to_numpy(dtype=object) if allocation_column and allocation_column in frame else None
    ticker_values_all = frame[ticker_column].to_numpy(dtype=object) if ticker_column in frame else None

    for fraction in fractions:
        local_rows: list[dict[str, Any]] = []
        for fold_id, part in frame.groupby(fold_column, sort=True):
            indices = part.index.to_numpy(dtype=np.int64)
            local_groups = group_values[indices] if group_values is not None else None
            local_tickers = ticker_values_all[indices] if ticker_values_all is not None else None
            metrics = evaluate_daily_budget(
                part[target_column].to_numpy(dtype=np.int8),
                part[score_column].to_numpy(dtype=np.float64),
                part[date_column].to_numpy(dtype="datetime64[ns]"),
                fraction,
                max_alerts_per_day=max_alerts_per_day,
                groups=local_groups,
                allocation_biases=allocation_biases,
                tickers=local_tickers,
            )
            record = {
                "fraction": float(fraction),
                "fold_id": int(fold_id),
                **metrics,
            }
            record["fold_gate_pass"] = _fold_pass(
                record,
                gate_recall,
                minimum_precision_lift,
                maximum_alert_rate,
            )
            fold_records.append(record)
            local_rows.append(record)
        local_frame = pd.DataFrame(local_rows)
        summary = summarize_budget_fold_rows(
            local_frame,
            gate_recall,
            minimum_precision_lift,
            maximum_alert_rate,
            required_fold_pass_rate,
        )
        summary_records.append(
            {
                "fraction": float(fraction),
                "gate_target_recall": gate_recall,
                **summary,
            }
        )

    summary_frame = pd.DataFrame(summary_records)
    fold_frame = pd.DataFrame(fold_records)
    feasible = summary_frame.loc[summary_frame["gate_pass"].eq(True)].copy()
    if not feasible.empty:
        feasible = feasible.sort_values(
            [
                "mean_alert_rate",
                "maximum_fold_alert_rate",
                "worst_recall",
                "worst_precision_lift",
                "mean_recall",
            ],
            ascending=[True, True, False, False, False],
            kind="mergesort",
        )
        chosen = feasible.iloc[0]
    else:
        # Best effort under the non-negotiable alert cap.  Pass count and the
        # worst fold are prioritized over attractive means.
        diagnostic = summary_frame.sort_values(
            [
                "fold_pass_rate",
                "worst_recall",
                "worst_precision_lift",
                "mean_recall",
                "mean_precision_lift",
                "mean_alert_rate",
            ],
            ascending=[False, False, False, False, False, True],
            kind="mergesort",
        )
        chosen = diagnostic.iloc[0]

    policy = DailyBudgetPolicy(
        kind="daily_budget",
        daily_fraction=float(chosen["fraction"]),
        max_alerts_per_day=int(max_alerts_per_day) if max_alerts_per_day is not None else None,
        target_recall=float(target_recall),
        selection_recall_buffer=float(selection_recall_buffer),
        minimum_precision_lift=float(minimum_precision_lift),
        maximum_alert_rate=float(maximum_alert_rate),
        required_fold_pass_rate=float(required_fold_pass_rate),
        allocation_column=allocation_column,
        allocation_biases={str(key): float(value) for key, value in (allocation_biases or {}).items()} or None,
        achieved_fold_pass_rate=float(chosen["fold_pass_rate"]),
        achieved_worst_recall=float(chosen["worst_recall"]),
        achieved_mean_recall=float(chosen["mean_recall"]),
        achieved_worst_precision_lift=float(chosen["worst_precision_lift"]),
        achieved_mean_alert_rate=float(chosen["mean_alert_rate"]),
        gate_pass=bool(chosen["gate_pass"]),
        source=source,
    )
    return policy, summary_frame, fold_frame


def datewise_rank_signal_frame(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    date_column: str = "date",
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    dates = frame[date_column].to_numpy(dtype="datetime64[ns]")
    for column in signal_columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        ranked = datewise_rank_normalize(values, dates)
        ranked[~np.isfinite(ranked)] = 0.5
        result[column] = ranked
    return result


def family_rank_matrix(
    ranked_signals: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
) -> tuple[np.ndarray, list[str], dict[str, list[str]]]:
    members: dict[str, list[str]] = {}
    for column in signal_columns:
        family = str(family_map.get(column, "unknown"))
        members.setdefault(family, []).append(str(column))
    families = sorted(members)
    columns: list[np.ndarray] = []
    for family in families:
        matrix = ranked_signals[members[family]].to_numpy(dtype=np.float64)
        columns.append(np.nanmean(matrix, axis=1))
    return np.column_stack(columns), families, members


def generate_simplex_candidates(
    dimension: int,
    random_samples: int,
    seed: int,
) -> list[np.ndarray]:
    if dimension < 1:
        raise ValueError("simplex dimension은 1 이상이어야 합니다")
    candidates: list[np.ndarray] = [np.full(dimension, 1.0 / dimension, dtype=np.float64)]
    for index in range(dimension):
        one = np.zeros(dimension, dtype=np.float64)
        one[index] = 1.0
        candidates.append(one)
    for left in range(dimension):
        for right in range(left + 1, dimension):
            pair = np.zeros(dimension, dtype=np.float64)
            pair[left] = 0.5
            pair[right] = 0.5
            candidates.append(pair)
    rng = np.random.default_rng(int(seed))
    for _ in range(max(0, int(random_samples))):
        candidates.append(rng.dirichlet(np.ones(dimension, dtype=np.float64)))
    unique: dict[tuple[float, ...], np.ndarray] = {}
    for candidate in candidates:
        normalized = np.asarray(candidate, dtype=np.float64)
        normalized = np.clip(normalized, 0.0, None)
        total = float(normalized.sum())
        if total <= 0:
            continue
        normalized /= total
        key = tuple(np.round(normalized, 10).tolist())
        unique[key] = normalized
    return list(unique.values())


def candidate_allocation_biases(
    groups: Sequence[Any] | np.ndarray | None,
    bias_grid: Sequence[float],
    maximum_groups: int = 2,
) -> list[dict[str, float] | None]:
    if groups is None:
        return [None]
    values = pd.Series(groups, dtype="string").fillna("__MISSING__").str.strip()
    counts = values.value_counts()
    if len(counts) < 2 or len(counts) > int(maximum_groups):
        return [None]
    ordered = sorted(counts.index.astype(str).tolist())
    baseline = ordered[0]
    others = ordered[1:]
    if len(others) != 1:
        return [None]
    candidate: list[dict[str, float] | None] = [None]
    for bias in sorted({float(value) for value in bias_grid}):
        if abs(bias) < 1e-15:
            continue
        candidate.append({baseline: 0.0, others[0]: float(bias)})
    return candidate


def _weight_search_objective(summary: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        1.0 if bool(summary.get("gate_pass")) else 0.0,
        float(summary.get("fold_pass_rate", 0.0)),
        float(summary.get("worst_recall", -1.0)),
        float(summary.get("worst_precision_lift", -1.0)),
        float(summary.get("mean_recall", -1.0)),
        float(summary.get("mean_precision_lift", -1.0)),
        -float(summary.get("mean_alert_rate", 2.0)),
    )


def fit_optimized_family_rank_spec(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    fraction_grid: Sequence[float],
    target_recall: float,
    selection_recall_buffer: float,
    minimum_precision_lift: float,
    maximum_alert_rate: float,
    required_fold_pass_rate: float,
    max_alerts_per_day: int | None,
    random_weight_samples: int,
    seed: int,
    allocation_column: str | None = None,
    allocation_bias_grid: Sequence[float] = (-0.10, -0.05, 0.0, 0.05, 0.10),
) -> tuple[dict[str, Any], DailyBudgetPolicy, pd.DataFrame]:
    """Fit family weights under a hard alert budget.

    Weight search and allocation-bias search are intentionally separated.  A
    nested weight×bias search would repeatedly sort every trading date tens of
    thousands of times and adds little statistical value.
    """

    frame = frame.reset_index(drop=True).copy()
    ranked = datewise_rank_signal_frame(frame, signal_columns)
    family_matrix, families, members = family_rank_matrix(ranked, signal_columns, family_map)
    weight_candidates = generate_simplex_candidates(len(families), random_weight_samples, seed)
    base = frame[["fold_id", "date", "target"]].copy()
    if "ticker" in frame.columns:
        base["ticker"] = frame["ticker"].to_numpy(dtype=object)
    if allocation_column and allocation_column in frame.columns:
        base[allocation_column] = frame[allocation_column].to_numpy(dtype=object)

    search_rows: list[dict[str, Any]] = []
    best_weight: tuple[tuple[float, ...], np.ndarray, DailyBudgetPolicy] | None = None
    for candidate_index, weights in enumerate(weight_candidates):
        local = base.copy()
        local["score"] = family_matrix @ weights
        policy, summary, _ = select_minimax_daily_budget_policy(
            local,
            fraction_grid,
            target_recall,
            selection_recall_buffer,
            minimum_precision_lift,
            maximum_alert_rate,
            required_fold_pass_rate,
            max_alerts_per_day=max_alerts_per_day,
            allocation_column=None,
            allocation_biases=None,
            source="optimized_family_rank_weight_search",
        )
        chosen_summary = summary.loc[np.isclose(summary["fraction"], policy.daily_fraction)].iloc[0].to_dict()
        search_rows.append(
            {
                "search_stage": "weight",
                "candidate_index": int(candidate_index),
                "weights": weights.tolist(),
                "allocation_biases": None,
                **chosen_summary,
            }
        )
        objective = _weight_search_objective(chosen_summary)
        if best_weight is None or objective > best_weight[0]:
            best_weight = (objective, weights, policy)
    if best_weight is None:
        raise RuntimeError("optimized family rank weight 후보를 만들지 못했습니다")

    selected_weights = best_weight[1]
    score = family_matrix @ selected_weights
    group_values = frame[allocation_column].to_numpy(dtype=object) if allocation_column and allocation_column in frame else None
    bias_candidates = candidate_allocation_biases(group_values, allocation_bias_grid)
    best_bias: tuple[tuple[float, ...], dict[str, float] | None, DailyBudgetPolicy] | None = None
    for bias_index, biases in enumerate(bias_candidates):
        local = base.copy()
        local["score"] = score
        policy, summary, _ = select_minimax_daily_budget_policy(
            local,
            fraction_grid,
            target_recall,
            selection_recall_buffer,
            minimum_precision_lift,
            maximum_alert_rate,
            required_fold_pass_rate,
            max_alerts_per_day=max_alerts_per_day,
            allocation_column=allocation_column if biases else None,
            allocation_biases=biases,
            source="optimized_family_rank_bias_search",
        )
        chosen_summary = summary.loc[np.isclose(summary["fraction"], policy.daily_fraction)].iloc[0].to_dict()
        search_rows.append(
            {
                "search_stage": "bias",
                "candidate_index": int(bias_index),
                "weights": selected_weights.tolist(),
                "allocation_biases": biases,
                **chosen_summary,
            }
        )
        objective = _weight_search_objective(chosen_summary)
        if best_bias is None or objective > best_bias[0]:
            best_bias = (objective, biases, policy)
    if best_bias is None:
        raise RuntimeError("optimized family rank bias 후보를 만들지 못했습니다")

    spec = {
        "method": "optimized_family_rank",
        "signals": [str(value) for value in signal_columns],
        "families": families,
        "family_members": members,
        "weights": {
            family: float(weight)
            for family, weight in zip(families, selected_weights, strict=True)
        },
        "allocation_column": allocation_column,
        "allocation_biases": best_bias[1],
    }
    return spec, best_bias[2], pd.DataFrame(search_rows)

def fit_equal_rank_spec(
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    method: str,
    allocation_column: str | None = None,
    allocation_biases: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    if method not in {"equal_recipe_rank", "equal_family_rank"}:
        raise ValueError(method)
    members: dict[str, list[str]] = {}
    for signal in signal_columns:
        members.setdefault(str(family_map.get(signal, "unknown")), []).append(str(signal))
    return {
        "method": method,
        "signals": [str(value) for value in signal_columns],
        "family_members": members,
        "allocation_column": allocation_column,
        "allocation_biases": {str(k): float(v) for k, v in (allocation_biases or {}).items()} or None,
    }


def apply_rank_ensemble_spec(
    spec: Mapping[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    method = str(spec["method"])
    signals = [str(value) for value in spec["signals"]]
    ranked = datewise_rank_signal_frame(frame, signals)
    if method == "equal_recipe_rank":
        return np.nanmean(ranked[signals].to_numpy(dtype=np.float64), axis=1)
    members = {
        str(family): [str(value) for value in values]
        for family, values in dict(spec.get("family_members", {})).items()
    }
    if method == "equal_family_rank":
        family_values = [
            np.nanmean(ranked[values].to_numpy(dtype=np.float64), axis=1)
            for _, values in sorted(members.items())
        ]
        return np.nanmean(np.column_stack(family_values), axis=1)
    if method == "optimized_family_rank":
        weight_map = {str(key): float(value) for key, value in dict(spec["weights"]).items()}
        result = np.zeros(len(frame), dtype=np.float64)
        total = 0.0
        for family, values in sorted(members.items()):
            weight = float(weight_map.get(family, 0.0))
            if weight <= 0:
                continue
            result += weight * np.nanmean(ranked[values].to_numpy(dtype=np.float64), axis=1)
            total += weight
        return result / total if total > 0 else np.nanmean(ranked[signals].to_numpy(dtype=np.float64), axis=1)
    raise ValueError(f"지원하지 않는 rank ensemble method: {method}")


def build_meta_feature_matrix(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    market_column: str | None = "market",
    market_categories: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    ranked = datewise_rank_signal_frame(frame, signal_columns)
    matrix = ranked[list(signal_columns)].to_numpy(dtype=np.float64)
    features: list[np.ndarray] = []
    names: list[str] = []
    for index, signal in enumerate(signal_columns):
        features.append(matrix[:, index])
        names.append(f"rank__{signal}")
    features.extend(
        [
            np.mean(matrix, axis=1),
            np.std(matrix, axis=1),
            np.min(matrix, axis=1),
            np.max(matrix, axis=1),
            np.max(matrix, axis=1) - np.min(matrix, axis=1),
            np.mean(matrix >= 0.80, axis=1),
            np.mean(matrix >= 0.90, axis=1),
        ]
    )
    names.extend(
        [
            "rank_mean",
            "rank_std",
            "rank_min",
            "rank_max",
            "rank_range",
            "consensus_ge_080",
            "consensus_ge_090",
        ]
    )
    members: dict[str, list[int]] = {}
    for index, signal in enumerate(signal_columns):
        members.setdefault(str(family_map.get(signal, "unknown")), []).append(index)
    for family, indices in sorted(members.items()):
        features.append(np.mean(matrix[:, indices], axis=1))
        names.append(f"family_rank__{family}")

    categories: list[str] = []
    if market_column and market_column in frame.columns:
        market = pd.Series(frame[market_column], dtype="string").fillna("__MISSING__").str.strip()
        categories = sorted(set(str(value) for value in (market_categories or market.unique().tolist())))
        for category in categories:
            features.append(market.eq(category).to_numpy(dtype=np.float64))
            names.append(f"market__{category}")
    x = np.column_stack(features).astype(np.float32)
    x[~np.isfinite(x)] = 0.5
    return x, names, categories


def fit_hard_negative_meta_lgb(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    seed: int,
    iterations: int = 160,
    hard_negative_pool_fraction: float = 0.50,
    hard_negative_multiplier: float = 3.0,
    positive_weight_mode: str = "sqrt_balance",
    market_column: str | None = "market",
    market_categories: Sequence[str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    import lightgbm as lgb

    x, feature_names, categories = build_meta_feature_matrix(
        frame,
        signal_columns,
        family_map,
        market_column=market_column,
        market_categories=market_categories,
    )
    y = frame["target"].to_numpy(dtype=np.int8)
    base_score = np.mean(x[:, : len(signal_columns)], axis=1)
    hard_pool = daily_budget_selection(
        base_score,
        frame["date"].to_numpy(dtype="datetime64[ns]"),
        hard_negative_pool_fraction,
    )
    weights = np.ones(len(y), dtype=np.float64)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives > 0 and positive_weight_mode == "sqrt_balance":
        weights[y == 1] *= math.sqrt(max(1.0, negatives / positives))
    elif positives > 0 and positive_weight_mode == "balanced":
        weights[y == 1] *= max(1.0, negatives / positives)
    weights[(y == 0) & hard_pool] *= float(hard_negative_multiplier)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": 7,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.3,
        "lambda_l2": 3.0,
        "verbosity": -1,
        "force_col_wise": True,
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(seed),
        "num_threads": 4,
    }
    dataset = lgb.Dataset(x, label=y, weight=weights, feature_name=feature_names, free_raw_data=True)
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=int(iterations),
        callbacks=[lgb.log_evaluation(period=0)],
    )
    spec = {
        "method": "hard_negative_meta_lgb",
        "signals": [str(value) for value in signal_columns],
        "family_map": {str(k): str(v) for k, v in family_map.items()},
        "feature_names": feature_names,
        "market_column": market_column,
        "market_categories": categories,
        "iterations": int(iterations),
        "hard_negative_pool_fraction": float(hard_negative_pool_fraction),
        "hard_negative_multiplier": float(hard_negative_multiplier),
        "positive_weight_mode": positive_weight_mode,
    }
    return booster, spec


def predict_hard_negative_meta_lgb(
    booster: Any,
    spec: Mapping[str, Any],
    frame: pd.DataFrame,
) -> np.ndarray:
    signal_columns = [str(value) for value in spec["signals"]]
    family_map = {str(k): str(v) for k, v in dict(spec["family_map"]).items()}
    x, feature_names, _ = build_meta_feature_matrix(
        frame,
        signal_columns,
        family_map,
        market_column=spec.get("market_column"),
        market_categories=spec.get("market_categories"),
    )
    expected = [str(value) for value in spec["feature_names"]]
    if feature_names != expected:
        raise ValueError("hard-negative meta feature schema mismatch")
    prediction = np.asarray(booster.predict(x), dtype=np.float64)
    return datewise_rank_normalize(prediction, frame["date"].to_numpy(dtype="datetime64[ns]"))


def summarize_method_crossfit(
    metrics: pd.DataFrame,
    target_recall: float,
    minimum_precision_lift: float,
    maximum_alert_rate: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method, part in metrics.groupby("method", sort=False):
        recalls = pd.to_numeric(part["recall"], errors="coerce").to_numpy(dtype=np.float64)
        lifts = pd.to_numeric(part["lift"], errors="coerce").to_numpy(dtype=np.float64)
        alerts = pd.to_numeric(part["alert_rate"], errors="coerce").to_numpy(dtype=np.float64)
        pr_lift_series = part["pr_auc_lift"] if "pr_auc_lift" in part.columns else pd.Series(np.nan, index=part.index)
        pr_lifts = pd.to_numeric(pr_lift_series, errors="coerce").to_numpy(dtype=np.float64)
        passes = np.asarray(
            [
                _fold_pass(row, target_recall, minimum_precision_lift, maximum_alert_rate)
                for row in part.to_dict(orient="records")
            ],
            dtype=bool,
        )
        record = {
            "method": str(method),
            "fold_count": int(len(part)),
            "fold_pass_count": int(passes.sum()),
            "fold_pass_rate": float(np.mean(passes)) if len(passes) else 0.0,
            "worst_recall": float(np.nanmin(recalls)) if np.isfinite(recalls).any() else float("nan"),
            "mean_recall": float(np.nanmean(recalls)) if np.isfinite(recalls).any() else float("nan"),
            "worst_precision_lift": float(np.nanmin(lifts)) if np.isfinite(lifts).any() else float("nan"),
            "mean_precision_lift": float(np.nanmean(lifts)) if np.isfinite(lifts).any() else float("nan"),
            "maximum_alert_rate": float(np.nanmax(alerts)) if np.isfinite(alerts).any() else float("nan"),
            "mean_alert_rate": float(np.nanmean(alerts)) if np.isfinite(alerts).any() else float("nan"),
            "mean_pr_auc_lift": float(np.nanmean(pr_lifts)) if np.isfinite(pr_lifts).any() else float("nan"),
        }
        records.append(record)
    summary = pd.DataFrame(records)
    if summary.empty:
        return summary
    summary = summary.sort_values(
        [
            "fold_pass_rate",
            "worst_recall",
            "worst_precision_lift",
            "maximum_alert_rate",
            "mean_pr_auc_lift",
        ],
        ascending=[False, False, False, True, False],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "selection_rank", np.arange(1, len(summary) + 1, dtype=np.int64))
    return summary


def select_best_method(summary: pd.DataFrame) -> str:
    if summary.empty:
        raise ValueError("method summary가 비어 있습니다")
    return str(summary.iloc[0]["method"])


def attach_ranking_metrics(metrics: dict[str, Any], target: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    """Attach metrics that remain valid for an uncalibrated ranking score.

    V5 deliberately operates on within-date ranks and ranker margins.  They are
    not probabilities, so Brier score, log loss, and ECE would be numerically
    computable but semantically invalid.
    """

    values = np.asarray(score, dtype=np.float64)
    monotonic = 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))
    ranking = safe_binary_metrics(target, monotonic)
    for key in ("pr_auc", "pr_auc_lift", "roc_auc"):
        metrics[key] = ranking.get(key)
    metrics["score_mean"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    metrics["score_is_calibrated_probability"] = False
    return metrics


def role_aggregate(frame: pd.DataFrame, role_column: str = "fold_role") -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    numeric_columns = [
        column
        for column in frame.columns
        if column not in {
            "fold_id",
            role_column,
            "method",
            "policy_kind",
            "score_is_calibrated_probability",
        }
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    for role, part in frame.groupby(role_column, sort=False):
        record: dict[str, Any] = {role_column: role, "fold_count": int(part["fold_id"].nunique())}
        for column in numeric_columns:
            values = pd.to_numeric(part[column], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            if len(values):
                record[column] = float(np.mean(values))
        records.append(record)
    return pd.DataFrame(records)


def build_scope_metrics(
    scored: pd.DataFrame,
    scope_columns: Sequence[str],
    minimum_rows: int,
    minimum_positives: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for role, role_part in scored.groupby("fold_role", sort=False):
        for column in scope_columns:
            if column not in role_part.columns:
                continue
            for value, part in role_part.groupby(column, dropna=False, sort=False):
                y = part["target"].to_numpy(dtype=np.int8)
                if len(part) < int(minimum_rows) or int(np.sum(y == 1)) < int(minimum_positives):
                    continue
                metrics = _selected_metrics(y, part["alert"].to_numpy(dtype=bool))
                raw_score = part["score"].to_numpy(dtype=np.float64)
                monotonic = 1.0 / (1.0 + np.exp(-np.clip(raw_score, -30.0, 30.0)))
                ranking = safe_binary_metrics(y, monotonic)
                records.append(
                    {
                        "fold_role": role,
                        "scope_column": column,
                        "scope_value": "__MISSING__" if pd.isna(value) else str(value),
                        **metrics,
                        "pr_auc": ranking.get("pr_auc"),
                        "pr_auc_lift": ranking.get("pr_auc_lift"),
                        "roc_auc": ranking.get("roc_auc"),
                    }
                )
    return pd.DataFrame(records)


def save_meta_model(booster: Any, path: Path) -> dict[str, Any]:
    from surge_model_zoo_common import sha256_file

    path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(path))
    return {
        "format": "lightgbm_text",
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def load_meta_model(path: Path) -> Any:
    import lightgbm as lgb

    return lgb.Booster(model_file=str(path))
