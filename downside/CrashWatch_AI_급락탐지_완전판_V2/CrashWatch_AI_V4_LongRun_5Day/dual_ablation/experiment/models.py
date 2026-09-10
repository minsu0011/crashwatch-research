from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_model_settings(prefer_gpu: bool = True) -> dict[str, Any]:
    """환경변수 기반의 재현 가능한 모델 설정.

    장기 실행에서는 GPU를 한계까지 밀지 않도록 트리 수와 CPU 스레드를
    외부 오케스트레이터가 통제한다. 캐시 키에도 이 설정이 포함된다.
    """
    cpu_count = os.cpu_count() or 8
    default_jobs = min(8, max(1, cpu_count // 2))
    return {
        "prefer_gpu": bool(prefer_gpu),
        "n_estimators": max(50, _env_int("CRASHWATCH_XGB_N_ESTIMATORS", 420)),
        "max_depth": max(2, _env_int("CRASHWATCH_XGB_MAX_DEPTH", 6)),
        "learning_rate": _env_float("CRASHWATCH_XGB_LEARNING_RATE", 0.04),
        "subsample": _env_float("CRASHWATCH_XGB_SUBSAMPLE", 0.85),
        "colsample_bytree": _env_float("CRASHWATCH_XGB_COLSAMPLE", 0.72),
        "min_child_weight": _env_float("CRASHWATCH_XGB_MIN_CHILD_WEIGHT", 5.0),
        "reg_alpha": _env_float("CRASHWATCH_XGB_REG_ALPHA", 0.15),
        "reg_lambda": _env_float("CRASHWATCH_XGB_REG_LAMBDA", 1.2),
        "max_bin": max(64, _env_int("CRASHWATCH_XGB_MAX_BIN", 256)),
        "n_jobs": max(1, min(cpu_count, _env_int("CRASHWATCH_XGB_N_JOBS", default_jobs))),
        "gpu_device": os.getenv("CRASHWATCH_GPU_DEVICE", "cuda"),
        "force_hist": os.getenv("CRASHWATCH_FORCE_HIST", "0") == "1",
        "hist_max_iter": max(50, _env_int("CRASHWATCH_HIST_MAX_ITER", 140)),
    }


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
    settings = get_model_settings(prefer_gpu=prefer_gpu)
    force_hist = backend == "hist" or bool(settings["force_hist"])

    if not force_hist:
        try:
            from xgboost import XGBClassifier

            common = dict(
                n_estimators=settings["n_estimators"],
                max_depth=settings["max_depth"],
                learning_rate=settings["learning_rate"],
                subsample=settings["subsample"],
                colsample_bytree=settings["colsample_bytree"],
                min_child_weight=settings["min_child_weight"],
                reg_alpha=settings["reg_alpha"],
                reg_lambda=settings["reg_lambda"],
                max_bin=settings["max_bin"],
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                n_jobs=settings["n_jobs"],
                scale_pos_weight=scale_pos_weight,
                tree_method="hist",
                verbosity=0,
            )
            if settings["prefer_gpu"]:
                try:
                    model = XGBClassifier(**common, device=settings["gpu_device"])
                    model.fit(x_train, y_train)
                    return FittedModel(model, "xgboost_cuda")
                except Exception as exc:  # noqa: BLE001
                    LOGGER.warning("XGBoost CUDA fallback: %s", exc)
            model = XGBClassifier(**common, device="cpu")
            model.fit(x_train, y_train)
            return FittedModel(model, "xgboost_cpu")
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("XGBoost fallback: %s", exc)

    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        learning_rate=0.065,
        max_iter=settings["hist_max_iter"],
        max_leaf_nodes=31,
        min_samples_leaf=25,
        l2_regularization=1.0,
        random_state=seed,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)
    return FittedModel(model, "hist_gradient_boosting")
