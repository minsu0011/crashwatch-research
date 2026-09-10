from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .expert_router import REGIME_KO_V2, route_prediction_columns
from .gating import model_metrics, raw_utility
from .regimes import REGIMES


PAIR_EXPERTS = ("P2_XGB", "P7_LGB")


@dataclass(frozen=True)
class PairRouteFit:
    mapping: dict[str, str]
    table: pd.DataFrame
    fallback_model: str
    shrink_rows: int
    p7_margin: float
    min_dates: int
    min_positives: int
    base_expert: str
    challenger_expert: str


def fit_conservative_pair_route(
    frame: pd.DataFrame,
    *,
    shrink_rows: int,
    p7_margin: float,
    min_dates: int = 8,
    min_positives: int = 10,
    base_expert: str = "P2_XGB",
    challenger_expert: str = "P7_LGB",
) -> PairRouteFit:
    if frame.empty:
        raise ValueError("cannot fit pair route on empty data")
    experts = (str(base_expert), str(challenger_expert))
    missing = [expert for expert in experts if expert not in frame.columns]
    if missing:
        raise ValueError(f"pair route is missing expert columns: {missing}")
    global_utility = {
        expert: raw_utility(model_metrics(frame, expert))
        for expert in experts
    }
    fallback = str(base_expert)
    rows: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for regime in REGIMES:
        part = frame[frame["regime"] == regime]
        dates = int(part["date"].nunique())
        positives = int(part["target"].sum())
        support = dates >= int(min_dates) and positives >= int(min_positives)
        alpha = len(part) / (len(part) + max(0, int(shrink_rows))) if len(part) else 0.0
        utilities: dict[str, float] = {}
        raw: dict[str, float | None] = {}
        for expert in experts:
            value = raw_utility(model_metrics(part, expert))
            raw[expert] = float(value) if np.isfinite(value) else None
            local = float(value) if np.isfinite(value) else float(global_utility[expert])
            utilities[expert] = float(alpha * local + (1.0 - alpha) * global_utility[expert])
        p7_advantage = utilities[challenger_expert] - utilities[base_expert]
        selected = challenger_expert if support and p7_advantage > float(p7_margin) else fallback
        mapping[regime] = selected
        rows.append({
            "regime": regime,
            "regime_ko": REGIME_KO_V2[regime],
            "rows": int(len(part)),
            "dates": dates,
            "positives": positives,
            "support_pass": bool(support),
            "base_expert": base_expert,
            "challenger_expert": challenger_expert,
            "raw_utility_base": raw[base_expert],
            "raw_utility_challenger": raw[challenger_expert],
            "shrunk_utility_base": utilities[base_expert],
            "shrunk_utility_challenger": utilities[challenger_expert],
            "challenger_advantage": float(p7_advantage),
            "p7_margin_required": float(p7_margin),
            "selected_expert": selected,
        })
    return PairRouteFit(
        mapping=mapping,
        table=pd.DataFrame(rows),
        fallback_model=fallback,
        shrink_rows=int(shrink_rows),
        p7_margin=float(p7_margin),
        min_dates=int(min_dates),
        min_positives=int(min_positives),
        base_expert=str(base_expert),
        challenger_expert=str(challenger_expert),
    )


def _logit(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=np.float64), eps, 1.0 - eps)
    return np.log(values) - np.log1p(-values)


def calibration_matrix(
    prediction: np.ndarray,
    regime: pd.Series | np.ndarray,
    method: str,
) -> np.ndarray:
    base = _logit(prediction).reshape(-1, 1)
    if method == "platt_global":
        return base
    if method == "platt_regime":
        values = np.asarray(regime, dtype=str)
        one_hot = np.column_stack([(values == name).astype(np.float64) for name in REGIMES])
        return np.column_stack([base, one_hot])
    raise ValueError(f"unsupported calibration method: {method}")


@dataclass
class TrainOnlyCalibrator:
    method: str
    c_value: float
    model: LogisticRegression | None

    @classmethod
    def fit(
        cls,
        prediction: np.ndarray,
        target: np.ndarray,
        regime: pd.Series | np.ndarray,
        *,
        method: str,
        c_value: float,
    ) -> "TrainOnlyCalibrator":
        if method == "none":
            return cls(method="none", c_value=float(c_value), model=None)
        y = np.asarray(target, dtype=np.uint8)
        if np.unique(y).size != 2:
            raise ValueError("calibration training data must contain both classes")
        matrix = calibration_matrix(prediction, regime, method)
        model = LogisticRegression(
            C=float(c_value),
            solver="lbfgs",
            max_iter=1000,
            random_state=20260808,
        )
        model.fit(matrix, y)
        return cls(method=method, c_value=float(c_value), model=model)

    def predict(self, prediction: np.ndarray, regime: pd.Series | np.ndarray) -> np.ndarray:
        raw = np.asarray(prediction, dtype=np.float64)
        if self.method == "none":
            return np.clip(raw, 1e-7, 1.0 - 1e-7)
        if self.model is None:
            raise RuntimeError("calibrator model is missing")
        matrix = calibration_matrix(raw, regime, self.method)
        return np.clip(self.model.predict_proba(matrix)[:, 1], 1e-7, 1.0 - 1e-7)

    def to_artifact(self) -> dict[str, Any]:
        if self.method == "none":
            return {"method": "none", "c_value": self.c_value}
        if self.model is None:
            raise RuntimeError("calibrator model is missing")
        return {
            "method": self.method,
            "c_value": self.c_value,
            "regime_order": list(REGIMES) if self.method == "platt_regime" else [],
            "intercept": self.model.intercept_.astype(float).tolist(),
            "coefficients": self.model.coef_.astype(float).tolist(),
            "classes": self.model.classes_.astype(int).tolist(),
        }


def apply_calibration_artifact(
    prediction: np.ndarray,
    regime: pd.Series | np.ndarray,
    artifact: dict[str, Any],
) -> np.ndarray:
    method = str(artifact.get("method", "none"))
    raw = np.asarray(prediction, dtype=np.float64)
    if method == "none":
        return np.clip(raw, 1e-7, 1.0 - 1e-7)
    matrix = calibration_matrix(raw, regime, method)
    coef = np.asarray(artifact["coefficients"], dtype=np.float64)
    intercept = np.asarray(artifact["intercept"], dtype=np.float64)
    score = matrix @ coef[0] + intercept[0]
    score = np.clip(score, -35.0, 35.0)
    return np.clip(1.0 / (1.0 + np.exp(-score)), 1e-7, 1.0 - 1e-7)


def fit_and_apply_candidate(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    shrink_rows: int,
    p7_margin: float,
    calibration_method: str,
    calibration_c: float,
    min_dates: int = 8,
    min_positives: int = 10,
) -> tuple[pd.DataFrame, PairRouteFit, TrainOnlyCalibrator]:
    route = fit_conservative_pair_route(
        train,
        shrink_rows=shrink_rows,
        p7_margin=p7_margin,
        min_dates=min_dates,
        min_positives=min_positives,
    )
    routed_train = route_prediction_columns(train, route.mapping)
    calibrator = TrainOnlyCalibrator.fit(
        routed_train["prediction"].to_numpy(dtype=float),
        routed_train["target"].to_numpy(dtype=np.uint8),
        routed_train["regime"],
        method=calibration_method,
        c_value=calibration_c,
    )
    routed_validation = route_prediction_columns(validation, route.mapping)
    routed_validation["raw_prediction"] = routed_validation["prediction"].to_numpy(dtype=float)
    routed_validation["prediction"] = calibrator.predict(
        routed_validation["raw_prediction"].to_numpy(dtype=float),
        routed_validation["regime"],
    )
    return routed_validation, route, calibrator


def candidate_id(candidate: dict[str, Any]) -> str:
    payload = json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
