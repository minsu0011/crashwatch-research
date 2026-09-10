from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_estimators: int = 800
    max_depth: int = 7
    learning_rate: float = 0.035
    subsample: float = 0.90
    colsample_bytree: float = 0.78
    min_child_weight: float = 5.0
    reg_alpha: float = 0.15
    reg_lambda: float = 1.5
    max_bin: int = 256
    grow_policy: str = "depthwise"
    max_leaves: int = 0

    def payload(self) -> dict:
        return asdict(self)


CANDIDATES = [
    ModelConfig("depth6_fast", n_estimators=650, max_depth=6, learning_rate=0.045, colsample_bytree=0.80),
    ModelConfig("depth7_balanced", n_estimators=850, max_depth=7, learning_rate=0.032, colsample_bytree=0.76),
    ModelConfig("depth8_slow", n_estimators=1000, max_depth=8, learning_rate=0.025, min_child_weight=7.0, colsample_bytree=0.72),
    ModelConfig("lossguide64", n_estimators=900, max_depth=0, max_leaves=64, grow_policy="lossguide", learning_rate=0.03, colsample_bytree=0.76),
]


def fit_xgb(x_train: np.ndarray, y_train: np.ndarray, seed: int, config: ModelConfig, *, threads: int = 16, prefer_gpu: bool = True):
    from xgboost import XGBClassifier

    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    scale_pos_weight = max(1.0, negatives / max(1, positives))
    params = dict(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        max_bin=config.max_bin,
        grow_policy=config.grow_policy,
        max_leaves=config.max_leaves,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=threads,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cuda" if prefer_gpu else "cpu",
        verbosity=0,
        validate_parameters=True,
    )
    try:
        model = XGBClassifier(**params)
        model.fit(x_train, y_train)
        return model, "xgboost_cuda" if prefer_gpu else "xgboost_cpu"
    except Exception as exc:
        if not prefer_gpu:
            raise
        LOGGER.warning("CUDA 학습 실패, CPU로 재시도: %s", exc)
        params["device"] = "cpu"
        model = XGBClassifier(**params)
        model.fit(x_train, y_train)
        return model, "xgboost_cpu_fallback"
