from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass
class FittedModel:
    model: object
    backend: str

    def predict_proba(self, x):
        values = self.model.predict_proba(x)
        return values[:, 1]


def fit_model(x_train, y_train, seed: int, *, prefer_gpu: bool = True, backend: str = "auto") -> FittedModel:
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    scale_pos_weight = max(1.0, negatives / max(1, positives))
    force_hist = backend == "hist" or os.getenv("CRASHWATCH_FORCE_HIST", "0") == "1"
    if not force_hist:
        try:
            from xgboost import XGBClassifier

            common = dict(
                n_estimators=450, max_depth=6, learning_rate=0.04,
                subsample=0.85, colsample_bytree=0.72,
                min_child_weight=5, reg_alpha=0.15, reg_lambda=1.2,
                objective="binary:logistic", eval_metric="logloss",
                random_state=seed, n_jobs=-1, scale_pos_weight=scale_pos_weight,
            )
            if prefer_gpu:
                try:
                    model = XGBClassifier(**common, tree_method="hist", device="cuda")
                    model.fit(x_train, y_train)
                    return FittedModel(model, "xgboost_cuda")
                except Exception as exc:  # noqa: BLE001
                    LOGGER.info("XGBoost CUDA fallback: %s", exc)
            model = XGBClassifier(**common, tree_method="hist", device="cpu")
            model.fit(x_train, y_train)
            return FittedModel(model, "xgboost_cpu")
        except Exception as exc:  # noqa: BLE001
            LOGGER.info("XGBoost fallback: %s", exc)

    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        learning_rate=0.065, max_iter=140, max_leaf_nodes=31,
        min_samples_leaf=25, l2_regularization=1.0, random_state=seed,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    return FittedModel(model, "hist_gradient_boosting")
