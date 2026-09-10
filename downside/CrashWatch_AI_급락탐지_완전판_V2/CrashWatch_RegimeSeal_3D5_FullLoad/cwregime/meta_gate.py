from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .regimes import REGIMES


MARKET_META_FEATURES = (
    "ret1",
    "ret_5",
    "ret_20",
    "ret_60",
    "vol_20",
    "vol_rank_252",
    "drawdown_60",
)


def _daily_top_precision(y: np.ndarray, prediction: np.ndarray, fraction: float = 0.03) -> float:
    k = max(1, int(math.ceil(len(y) * fraction)))
    selected = np.argsort(prediction, kind="mergesort")[-k:]
    return float(np.mean(y[selected])) if len(selected) else float("nan")


def _rank_utility(y: np.ndarray, prediction: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y, dtype=np.uint8)
    prediction = np.asarray(prediction, dtype=np.float64)
    if np.unique(y).size == 2:
        ap = float(average_precision_score(y, prediction))
        roc = float(roc_auc_score(y, prediction))
    else:
        ap = 0.0 if int(y.sum()) == 0 else 1.0
        roc = 0.5
    top = _daily_top_precision(y, prediction)
    return float(0.50 * ap + 0.25 * roc + 0.25 * top)


def _prediction_summaries(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    return {
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_std": float(values.std(ddof=0)),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_q10": float(quantiles[0]),
        f"{prefix}_q25": float(quantiles[1]),
        f"{prefix}_q50": float(quantiles[2]),
        f"{prefix}_q75": float(quantiles[3]),
        f"{prefix}_q90": float(quantiles[4]),
        f"{prefix}_q95": float(quantiles[5]),
    }


def build_daily_meta_frame(wide: pd.DataFrame, regime_calendar: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = {"date", "temporal_block", "regime", "target", "P2_LGB", "P2_XGB", "P7_LGB"}
    missing = required - set(wide.columns)
    if missing:
        raise ValueError(f"OOF frame is missing columns: {sorted(missing)}")
    frame = wide.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["P2_LOCKED"] = 0.1 * frame["P2_LGB"].to_numpy(dtype=float) + 0.9 * frame["P2_XGB"].to_numpy(dtype=float)
    calendar = regime_calendar.copy()
    calendar["date"] = pd.to_datetime(calendar["date"]).dt.normalize()
    calendar = calendar[["date", *MARKET_META_FEATURES]].drop_duplicates("date")
    rows: list[dict[str, Any]] = []
    for date, part in frame.groupby("date", sort=True):
        regimes = part["regime"].unique()
        blocks = part["temporal_block"].unique()
        if len(regimes) != 1 or len(blocks) != 1:
            raise ValueError(f"date {date} does not have one regime/block")
        y = part["target"].to_numpy(dtype=np.uint8)
        p2 = part["P2_LOCKED"].to_numpy(dtype=float)
        p7 = part["P7_LGB"].to_numpy(dtype=float)
        delta = p7 - p2
        p2_utility = _rank_utility(y, p2)
        p7_utility = _rank_utility(y, p7)
        rank_p2 = pd.Series(p2).rank(method="average").to_numpy(dtype=float)
        rank_p7 = pd.Series(p7).rank(method="average").to_numpy(dtype=float)
        correlation = float(np.corrcoef(rank_p2, rank_p7)[0, 1]) if len(p2) > 1 else 1.0
        k = max(1, int(math.ceil(len(y) * 0.03)))
        top_p2 = set(np.argsort(p2, kind="mergesort")[-k:].tolist())
        top_p7 = set(np.argsort(p7, kind="mergesort")[-k:].tolist())
        row: dict[str, Any] = {
            "date": date,
            "temporal_block": str(blocks[0]),
            "regime": str(regimes[0]),
            "rows": int(len(part)),
            "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
            "p2_rank_utility": p2_utility,
            "p7_rank_utility": p7_utility,
            "p7_utility_delta": float(p7_utility - p2_utility),
            "p7_win": int(p7_utility > p2_utility + 1e-12),
            "rank_correlation": correlation,
            "top3_overlap": float(len(top_p2 & top_p7) / max(1, k)),
            "prediction_disagreement_mean_abs": float(np.mean(np.abs(delta))),
            "prediction_disagreement_q90_abs": float(np.quantile(np.abs(delta), 0.90)),
            **_prediction_summaries(p2, "p2"),
            **_prediction_summaries(p7, "p7"),
            **_prediction_summaries(delta, "p7_minus_p2"),
        }
        rows.append(row)
    daily = pd.DataFrame(rows).merge(calendar, on="date", how="left", validate="one_to_one")
    for regime in REGIMES:
        daily[f"regime_{regime}"] = daily["regime"].eq(regime).astype(np.float32)
    forbidden = {
        "date", "temporal_block", "regime", "rows", "positives", "positive_rate",
        "p2_rank_utility", "p7_rank_utility", "p7_utility_delta", "p7_win",
    }
    feature_names = [name for name in daily.columns if name not in forbidden]
    target_leaks = [name for name in feature_names if "utility" in name or name in {"positives", "positive_rate", "p7_win"}]
    if target_leaks:
        raise RuntimeError(f"meta feature leakage detected: {target_leaks}")
    return daily, feature_names


def meta_candidate_grid() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = [{
        "model": "none", "parameter": 0.0, "threshold": 1e9, "candidate_id": "NO_OVERRIDE",
    }]
    for alpha in (0.1, 1.0, 10.0, 100.0):
        for threshold in (0.0, 0.005, 0.01, 0.02):
            candidates.append({"model": "ridge", "parameter": alpha, "threshold": threshold})
    for c_value in (0.1, 1.0, 10.0):
        for threshold in (0.55, 0.65, 0.75, 0.85):
            candidates.append({"model": "logistic", "parameter": c_value, "threshold": threshold})
    for leaves in (3, 5):
        for l2 in (1.0, 10.0):
            for threshold in (0.0, 0.005, 0.01, 0.02):
                candidates.append({
                    "model": "hist", "parameter": leaves, "l2": l2, "threshold": threshold,
                })
    for item in candidates:
        if "candidate_id" not in item:
            payload = json.dumps(item, sort_keys=True, separators=(",", ":"))
            item["candidate_id"] = hashlib.sha256(payload.encode()).hexdigest()[:16]
    return candidates


def build_meta_estimator(candidate: dict[str, Any]):
    model = str(candidate["model"])
    if model == "none":
        return None
    if model == "ridge":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=float(candidate["parameter"]))),
        ])
    if model == "logistic":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                C=float(candidate["parameter"]), solver="lbfgs", max_iter=2000,
                random_state=20260808,
            )),
        ])
    if model == "hist":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_iter=150,
                max_leaf_nodes=int(candidate["parameter"]),
                max_depth=2,
                min_samples_leaf=15,
                l2_regularization=float(candidate["l2"]),
                random_state=20260808,
            )),
        ])
    raise ValueError(f"unsupported meta model: {model}")


def fit_predict_meta_gate(
    train_daily: pd.DataFrame,
    validation_daily: pd.DataFrame,
    feature_names: list[str],
    candidate: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Any]:
    if candidate["model"] == "none":
        score = np.full(len(validation_daily), -np.inf, dtype=np.float64)
        return np.zeros(len(validation_daily), dtype=bool), score, None
    train = train_daily[train_daily["positives"] > 0].copy()
    if len(train) < 40:
        raise ValueError("insufficient positive-support dates for meta gate")
    x_train = train[feature_names].replace([np.inf, -np.inf], np.nan)
    x_validation = validation_daily[feature_names].replace([np.inf, -np.inf], np.nan)
    weights = np.sqrt(np.maximum(1.0, train["positives"].to_numpy(dtype=float)))
    estimator = build_meta_estimator(candidate)
    if candidate["model"] == "logistic":
        y = train["p7_win"].to_numpy(dtype=np.uint8)
        if np.unique(y).size != 2:
            return np.zeros(len(validation_daily), dtype=bool), np.zeros(len(validation_daily)), None
        estimator.fit(x_train, y, model__sample_weight=weights)
        score = estimator.predict_proba(x_validation)[:, 1]
    else:
        y = train["p7_utility_delta"].to_numpy(dtype=np.float64)
        estimator.fit(x_train, y, model__sample_weight=weights)
        score = estimator.predict(x_validation)
    override = score > float(candidate["threshold"])
    return np.asarray(override, dtype=bool), np.asarray(score, dtype=np.float64), estimator


def apply_date_override(
    row_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    override: np.ndarray,
) -> pd.DataFrame:
    dates = pd.to_datetime(daily_frame["date"]).dt.normalize()
    mapping = dict(zip(dates, np.asarray(override, dtype=bool), strict=True))
    out = row_frame.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    selected = out["date"].map(mapping)
    if selected.isna().any():
        raise RuntimeError("meta gate has no selection for some validation dates")
    use_p7 = selected.to_numpy(dtype=bool)
    p2 = 0.1 * out["P2_LGB"].to_numpy(dtype=float) + 0.9 * out["P2_XGB"].to_numpy(dtype=float)
    p7 = out["P7_LGB"].to_numpy(dtype=float)
    out["selected_submodel"] = np.where(use_p7, "P7_LGB", "P2_LOCKED")
    out["prediction"] = np.where(use_p7, p7, p2)
    out["meta_override"] = use_p7
    return out
