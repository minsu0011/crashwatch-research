from __future__ import annotations

import gc
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


TREE_CONFIGS: dict[str, list[dict[str, Any]]] = {
    "xgboost": [
        {
            "name": "xgb_depth7",
            "n_estimators": 1000,
            "max_depth": 7,
            "learning_rate": 0.028,
            "subsample": 0.90,
            "colsample_bytree": 0.78,
            "min_child_weight": 5.0,
            "reg_alpha": 0.15,
            "reg_lambda": 1.6,
            "max_bin": 512,
            "grow_policy": "depthwise",
        },
        {
            "name": "xgb_lossguide96",
            "n_estimators": 1100,
            "max_depth": 0,
            "max_leaves": 96,
            "learning_rate": 0.025,
            "subsample": 0.90,
            "colsample_bytree": 0.72,
            "min_child_weight": 7.0,
            "reg_alpha": 0.20,
            "reg_lambda": 1.8,
            "max_bin": 512,
            "grow_policy": "lossguide",
        },
    ],
    "lightgbm": [
        {
            "name": "lgbm_leaf63",
            "n_estimators": 1200,
            "learning_rate": 0.025,
            "num_leaves": 63,
            "max_depth": -1,
            "min_child_samples": 40,
            "subsample": 0.90,
            "colsample_bytree": 0.78,
            "reg_alpha": 0.15,
            "reg_lambda": 1.5,
            "max_bin": 255,
        },
        {
            "name": "lgbm_leaf127",
            "n_estimators": 1000,
            "learning_rate": 0.028,
            "num_leaves": 127,
            "max_depth": -1,
            "min_child_samples": 55,
            "subsample": 0.88,
            "colsample_bytree": 0.72,
            "reg_alpha": 0.25,
            "reg_lambda": 1.8,
            "max_bin": 255,
        },
    ],
    "catboost": [
        {
            "name": "cat_depth8",
            "iterations": 1100,
            "depth": 8,
            "learning_rate": 0.035,
            "l2_leaf_reg": 5.0,
            "random_strength": 0.5,
            "border_count": 254,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.5,
        },
        {
            "name": "cat_depth10",
            "iterations": 900,
            "depth": 10,
            "learning_rate": 0.030,
            "l2_leaf_reg": 7.0,
            "random_strength": 0.8,
            "border_count": 254,
            "bootstrap_type": "Bayesian",
            "bagging_temperature": 0.8,
        },
    ],
}


@dataclass
class FittedTree:
    family: str
    model: Any
    actual_backend: str
    config: dict[str, Any]

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = self.model.predict_proba(matrix)
        return np.asarray(values[:, 1], dtype=np.float32)

    def feature_importance(self) -> np.ndarray:
        if hasattr(self.model, "feature_importances_"):
            return np.asarray(self.model.feature_importances_, dtype=float)
        if hasattr(self.model, "get_feature_importance"):
            return np.asarray(self.model.get_feature_importance(), dtype=float)
        return np.asarray([], dtype=float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.family == "xgboost":
            self.model.save_model(path.with_suffix(".json"))
        elif self.family == "lightgbm":
            # LightGBM on Windows can fail to open a Unicode destination path.
            target = path.with_suffix(".txt")
            with tempfile.TemporaryDirectory(prefix="crashwatch_lgbm_") as temp_dir:
                temporary = Path(temp_dir) / "model.txt"
                self.model.booster_.save_model(str(temporary))
                shutil.copyfile(temporary, target)
        elif self.family == "catboost":
            self.model.save_model(str(path.with_suffix(".cbm")))
        else:
            raise ValueError(self.family)
        path.with_suffix(".meta.json").write_text(
            json.dumps({"family": self.family, "backend": self.actual_backend, "config": self.config}, indent=2),
            encoding="utf-8",
        )


def _scale_pos_weight(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int8)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    return max(1.0, negatives / max(1, positives))


def fit_tree_model(
    family: str,
    config: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    requested_backend: str,
    threads: int,
    eval_set: tuple[np.ndarray, np.ndarray] | None = None,
    force_backend: str | None = None,
) -> FittedTree:
    """Fit one tree model. GPU failures fall back to CPU and are recorded."""
    family = family.lower()
    backend = force_backend or requested_backend
    cfg = {key: value for key, value in config.items() if key != "name"}
    positive_weight_multiplier = float(cfg.pop("positive_weight_multiplier", 1.0))
    scale_pos_weight = _scale_pos_weight(y_train) * max(0.25, positive_weight_multiplier)
    if family == "xgboost":
        from xgboost import XGBClassifier

        params = dict(
            **cfg,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            n_jobs=threads,
            scale_pos_weight=scale_pos_weight,
            tree_method="hist",
            device="cuda" if backend == "cuda" else "cpu",
            verbosity=0,
            validate_parameters=True,
        )
        try:
            model = XGBClassifier(**params)
            fit_kwargs: dict[str, Any] = {}
            if eval_set is not None:
                fit_kwargs["eval_set"] = [eval_set]
                fit_kwargs["verbose"] = False
            model.fit(x_train, y_train, **fit_kwargs)
            return FittedTree(family, model, "cuda" if backend == "cuda" else "cpu", config)
        except Exception as exc:
            if backend != "cuda" or force_backend == "cuda":
                raise
            LOGGER.warning("XGBoost CUDA failed; CPU fallback: %s", exc)
            params["device"] = "cpu"
            model = XGBClassifier(**params)
            model.fit(x_train, y_train)
            return FittedTree(family, model, "cpu_fallback", config)

    if family == "lightgbm":
        from lightgbm import LGBMClassifier

        params = dict(
            **cfg,
            objective="binary",
            random_state=seed,
            n_jobs=threads,
            scale_pos_weight=scale_pos_weight,
            verbosity=-1,
            device_type="gpu" if backend == "cuda" else "cpu",
        )
        if backend == "cuda":
            params.update(gpu_use_dp=False)
        try:
            model = LGBMClassifier(**params)
            fit_kwargs = {}
            if eval_set is not None:
                fit_kwargs["eval_set"] = [eval_set]
                fit_kwargs["callbacks"] = []
            model.fit(x_train, y_train, **fit_kwargs)
            return FittedTree(family, model, "gpu_opencl" if backend == "cuda" else "cpu", config)
        except Exception as exc:
            if backend != "cuda" or force_backend == "cuda":
                raise
            LOGGER.warning("LightGBM GPU failed; CPU fallback: %s", exc)
            params["device_type"] = "cpu"
            params.pop("gpu_use_dp", None)
            model = LGBMClassifier(**params)
            model.fit(x_train, y_train)
            return FittedTree(family, model, "cpu_fallback", config)

    if family == "catboost":
        from catboost import CatBoostClassifier

        params = dict(
            **cfg,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=seed,
            thread_count=threads,
            verbose=False,
            allow_writing_files=False,
            scale_pos_weight=scale_pos_weight,
            task_type="GPU" if backend == "cuda" else "CPU",
            devices="0" if backend == "cuda" else None,
        )
        params = {key: value for key, value in params.items() if value is not None}
        try:
            model = CatBoostClassifier(**params)
            fit_kwargs = {}
            if eval_set is not None:
                fit_kwargs["eval_set"] = eval_set
                fit_kwargs["use_best_model"] = True
                fit_kwargs["early_stopping_rounds"] = 80
            model.fit(x_train, y_train, **fit_kwargs)
            return FittedTree(family, model, "cuda" if backend == "cuda" else "cpu", config)
        except Exception as exc:
            if backend != "cuda" or force_backend == "cuda":
                raise
            LOGGER.warning("CatBoost GPU failed; CPU fallback: %s", exc)
            params["task_type"] = "CPU"
            params.pop("devices", None)
            model = CatBoostClassifier(**params)
            model.fit(x_train, y_train)
            return FittedTree(family, model, "cpu_fallback", config)

    raise ValueError(f"unsupported tree family: {family}")


def release_tree(model: FittedTree | None) -> None:
    if model is not None:
        del model.model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
