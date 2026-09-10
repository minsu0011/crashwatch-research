from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

from .regimes import REGIMES
from .scoring import metrics_by_regime


SUBMODELS = ("P2_LGB", "P2_XGB", "P7_LGB", "P7_XGB")
REGIME_KO = {
    "CRASH_STRESS": "급락 스트레스",
    "REBOUND": "급락 후 반등",
    "BULL_LOW_VOL": "상승·저변동",
    "BULL_HIGH_VOL": "강상승·고변동",
    "SIDEWAYS_LOW_VOL": "횡보·저변동",
    "SIDEWAYS_HIGH_VOL": "강횡보·고변동",
    "BEAR_LOW_VOL": "하락·저변동",
    "BEAR_HIGH_VOL": "강하락·고변동",
}


def canonical_hash(value: Any, length: int = 24) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _finite(value: float | int | None) -> float | None:
    if value is None:
        return None
    out = float(value)
    return out if np.isfinite(out) else None


def _daily_top_fraction(frame: pd.DataFrame, prediction: str, fraction: float = 0.03) -> pd.DataFrame:
    parts = []
    for _, block in frame.groupby("date", sort=False):
        count = max(1, int(math.ceil(len(block) * float(fraction))))
        parts.append(block.nlargest(count, prediction, keep="first"))
    return pd.concat(parts, ignore_index=False) if parts else frame.iloc[:0].copy()


def model_metrics(frame: pd.DataFrame, prediction: str) -> dict[str, Any]:
    y = frame["target"].to_numpy(dtype=np.uint8)
    p = np.clip(frame[prediction].to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
    rows = int(len(y))
    positives = int(y.sum())
    positive_rate = float(y.mean()) if rows else float("nan")
    two_class = bool(rows and np.unique(y).size == 2)
    pr_auc = float(average_precision_score(y, p)) if two_class else float("nan")
    roc_auc = float(roc_auc_score(y, p)) if two_class else float("nan")
    brier = float(brier_score_loss(y, p)) if rows else float("nan")
    loss = float(log_loss(y, p, labels=[0, 1])) if rows else float("nan")
    alerts = _daily_top_fraction(frame.assign(_gate_prediction=p), "_gate_prediction") if rows else frame.iloc[:0]
    alert_count = int(len(alerts))
    alert_positives = int(alerts["target"].sum()) if alert_count else 0
    alert_precision = float(alert_positives / alert_count) if alert_count else float("nan")
    alert_recall = float(alert_positives / positives) if positives else float("nan")
    return {
        "rows": rows,
        "positives": positives,
        "positive_rate": positive_rate,
        "dates": int(frame["date"].nunique()) if rows else 0,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
        "logloss": loss,
        "mean_prediction": float(p.mean()) if rows else float("nan"),
        "pr_auc_lift": float(pr_auc / positive_rate) if two_class and positive_rate > 0 else float("nan"),
        "alert_count": alert_count,
        "alert_positives": alert_positives,
        "alert_precision": alert_precision,
        "alert_recall": alert_recall,
        "alert_precision_lift": (
            float(alert_precision / positive_rate)
            if np.isfinite(alert_precision) and np.isfinite(positive_rate) and positive_rate > 0
            else float("nan")
        ),
    }


def raw_utility(metrics: dict[str, Any]) -> float:
    lift = float(metrics.get("pr_auc_lift", float("nan")))
    if not np.isfinite(lift):
        return float("nan")
    roc = float(metrics.get("roc_auc", float("nan")))
    top_lift = float(metrics.get("alert_precision_lift", float("nan")))
    roc = roc if np.isfinite(roc) else 0.5
    top_lift = top_lift if np.isfinite(top_lift) else 1.0
    return float(
        math.log(max(lift, 0.05))
        + 0.25 * (2.0 * roc - 1.0)
        + 0.15 * math.log(max(top_lift, 0.05))
    )


def performance_table(frame: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime in REGIMES:
        part = frame[frame["regime"] == regime]
        for model in SUBMODELS:
            metrics = model_metrics(part, model)
            rows.append({
                "scope": scope,
                "regime": regime,
                "regime_ko": REGIME_KO[regime],
                "submodel": model,
                **metrics,
                "raw_utility": raw_utility(metrics),
            })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class GateFit:
    mapping: dict[str, str]
    regime_table: pd.DataFrame
    cluster_table: pd.DataFrame
    method: str
    shrink_rows: int
    n_clusters: int
    fallback_model: str


def fit_gate(
    frame: pd.DataFrame,
    *,
    method: str,
    shrink_rows: int,
    n_clusters: int,
    random_state: int = 20260808,
) -> GateFit:
    if method not in {"hard", "cluster"}:
        raise ValueError(f"unsupported gate method: {method}")
    if frame.empty:
        raise ValueError("cannot fit gate on an empty frame")

    global_metrics = {model: model_metrics(frame, model) for model in SUBMODELS}
    global_utility = {model: raw_utility(global_metrics[model]) for model in SUBMODELS}
    fallback_model = max(
        SUBMODELS,
        key=lambda name: global_utility[name] if np.isfinite(global_utility[name]) else -np.inf,
    )
    rows: list[dict[str, Any]] = []
    for regime in REGIMES:
        part = frame[frame["regime"] == regime]
        row: dict[str, Any] = {
            "regime": regime,
            "regime_ko": REGIME_KO[regime],
            "rows": int(len(part)),
            "positives": int(part["target"].sum()),
            "dates": int(part["date"].nunique()),
        }
        for model in SUBMODELS:
            metrics = model_metrics(part, model)
            value = raw_utility(metrics)
            if not np.isfinite(value):
                value = global_utility[model]
            alpha = len(part) / (len(part) + max(0, int(shrink_rows))) if len(part) else 0.0
            shrunk = alpha * value + (1.0 - alpha) * global_utility[model]
            row[f"utility_{model}"] = float(shrunk)
            row[f"raw_utility_{model}"] = _finite(raw_utility(metrics))
            row[f"pr_auc_lift_{model}"] = _finite(metrics["pr_auc_lift"])
            row[f"roc_auc_{model}"] = _finite(metrics["roc_auc"])
            row[f"alert_precision_lift_{model}"] = _finite(metrics["alert_precision_lift"])
        rows.append(row)

    table = pd.DataFrame(rows)
    utility_columns = [f"utility_{model}" for model in SUBMODELS]
    matrix = table[utility_columns].to_numpy(dtype=float)
    matrix = matrix - matrix.mean(axis=1, keepdims=True)
    clusters = np.zeros(len(table), dtype=int)
    effective_clusters = 1
    if method == "cluster":
        effective_clusters = max(1, min(int(n_clusters), len(table)))
        clusters = KMeans(
            n_clusters=effective_clusters,
            random_state=int(random_state),
            n_init=50,
        ).fit_predict(matrix)
    table["cluster"] = clusters

    hard_mapping = {
        str(row.regime): max(SUBMODELS, key=lambda model: float(getattr(row, f"utility_{model}")))
        for row in table.itertuples(index=False)
    }
    cluster_rows: list[dict[str, Any]] = []
    cluster_mapping: dict[str, str] = {}
    for cluster in sorted(table["cluster"].unique()):
        regimes = table.loc[table["cluster"] == cluster, "regime"].tolist()
        part = frame[frame["regime"].isin(regimes)]
        utilities = {model: raw_utility(model_metrics(part, model)) for model in SUBMODELS}
        winner = max(
            SUBMODELS,
            key=lambda model: utilities[model] if np.isfinite(utilities[model]) else -np.inf,
        )
        cluster_rows.append({
            "cluster": int(cluster),
            "regimes": ",".join(regimes),
            "rows": int(len(part)),
            "positives": int(part["target"].sum()),
            "dates": int(part["date"].nunique()),
            "winner": winner,
            **{f"utility_{model}": _finite(utilities[model]) for model in SUBMODELS},
        })
        for regime in regimes:
            cluster_mapping[regime] = winner

    mapping = hard_mapping if method == "hard" else cluster_mapping
    table["hard_winner"] = table["regime"].map(hard_mapping)
    table["selected_model"] = table["regime"].map(mapping)
    table["support_status"] = np.select(
        [table["dates"] < 8, table["positives"] < 10],
        ["LOW_DATE_SUPPORT", "LOW_POSITIVE_SUPPORT"],
        default="SUPPORTED",
    )
    return GateFit(
        mapping=mapping,
        regime_table=table,
        cluster_table=pd.DataFrame(cluster_rows),
        method=method,
        shrink_rows=int(shrink_rows),
        n_clusters=effective_clusters,
        fallback_model=fallback_model,
    )


def apply_gate(frame: pd.DataFrame, gate: GateFit | dict[str, str]) -> pd.DataFrame:
    mapping = gate.mapping if isinstance(gate, GateFit) else gate
    out = frame.copy()
    fallback = "P2_XGB"
    if isinstance(gate, GateFit):
        fallback = gate.fallback_model
    selected = out["regime"].map(mapping).fillna(fallback)
    out["selected_submodel"] = selected
    prediction = np.empty(len(out), dtype=float)
    for model in SUBMODELS:
        mask = selected.eq(model).to_numpy()
        prediction[mask] = out.loc[mask, model].to_numpy(dtype=float)
    out["prediction"] = np.clip(prediction, 1e-6, 1.0 - 1e-6)
    return out


def evaluate_prediction(frame: pd.DataFrame, name: str, prediction: str = "prediction") -> tuple[dict[str, Any], pd.DataFrame]:
    evaluated = frame.copy()
    if prediction != "prediction":
        evaluated["prediction"] = evaluated[prediction].to_numpy(dtype=float)
    score, regimes = metrics_by_regime(evaluated)
    row = {
        "candidate": name,
        "selection_score": float(score["selection_score"]),
        **{f"overall_{key}": value for key, value in score["overall"].items()},
        "regime_pr_lift_geometric_mean": score.get("regime_pr_lift_geometric_mean"),
        "worst_regime_pr_lift": score.get("worst_regime_pr_lift"),
        "top3_precision_lift": score.get("top3_precision_lift"),
    }
    regimes.insert(0, "candidate", name)
    return row, regimes


def submodel_label_table(gate: GateFit) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for regime_row in gate.regime_table.itertuples(index=False):
        utilities = {model: float(getattr(regime_row, f"utility_{model}")) for model in SUBMODELS}
        best = max(utilities.values())
        for model in SUBMODELS:
            gap = best - utilities[model]
            if regime_row.support_status != "SUPPORTED":
                label = "UNCERTAIN_SUPPORT"
            elif gap <= 1e-12:
                label = "STRONG_WINNER"
            elif gap <= 0.03:
                label = "COMPETITIVE"
            else:
                label = "WEAK"
            rows.append({
                "regime": regime_row.regime,
                "regime_ko": regime_row.regime_ko,
                "cluster": int(regime_row.cluster),
                "submodel": model,
                "profile": model.split("_", 1)[0],
                "algorithm": model.split("_", 1)[1],
                "rows": int(regime_row.rows),
                "positives": int(regime_row.positives),
                "dates": int(regime_row.dates),
                "support_status": regime_row.support_status,
                "utility": utilities[model],
                "utility_gap_to_best": float(gap),
                "label": label,
                "is_hard_winner": bool(model == regime_row.hard_winner),
                "is_frozen_gate_model": bool(model == regime_row.selected_model),
            })
    return pd.DataFrame(rows)

