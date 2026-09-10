from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _model_info(model_format: str, path: Path, iterations: int) -> dict[str, Any]:
    return {
        "format": model_format,
        "path": str(path),
        "iterations": int(iterations),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def save_lightgbm_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    params: Mapping[str, Any],
    iterations: int,
    path: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    path.parent.mkdir(parents=True, exist_ok=True)
    train_set = lgb.Dataset(x_train, label=y_train, weight=weights, free_raw_data=True)
    booster = lgb.train(
        dict(params),
        train_set,
        num_boost_round=int(iterations),
        callbacks=[lgb.log_evaluation(period=0)],
    )
    booster.save_model(str(path), num_iteration=int(iterations))
    return _model_info("lightgbm_text", path, int(iterations))


def save_xgboost_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    params: Mapping[str, Any],
    iterations: int,
    path: Path,
) -> dict[str, Any]:
    import xgboost as xgb

    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = dict(params)
    max_bin = int(resolved.get("max_bin", 256))
    matrix = xgb.QuantileDMatrix(x_train, label=y_train, weight=weights, max_bin=max_bin)
    booster = xgb.train(resolved, matrix, num_boost_round=int(iterations), verbose_eval=False)
    booster.save_model(path)
    return _model_info("xgboost_json", path, int(iterations))


def save_catboost_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    params: Mapping[str, Any],
    iterations: int,
    path: Path,
) -> dict[str, Any]:
    from catboost import CatBoostClassifier, Pool

    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = dict(params)
    resolved["iterations"] = int(iterations)
    model = CatBoostClassifier(**resolved)
    model.fit(Pool(x_train, label=y_train, weight=weights), verbose=False)
    model.save_model(str(path), format="cbm")
    return _model_info("catboost_cbm", path, int(iterations))


def save_extra_trees_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    params: Mapping[str, Any],
    seed: int,
    threads: int,
    path: Path,
) -> dict[str, Any]:
    import joblib
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer

    path.parent.mkdir(parents=True, exist_ok=True)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    clean = imputer.fit_transform(x_train)
    resolved = dict(params)
    model = ExtraTreesClassifier(
        **resolved,
        random_state=int(seed),
        n_jobs=int(threads),
        bootstrap=False,
        criterion="log_loss",
    )
    model.fit(clean, y_train, sample_weight=weights)
    joblib.dump({"imputer": imputer, "model": model}, path, compress=3)
    return _model_info(
        "joblib_extra_trees",
        path,
        int(resolved.get("n_estimators", 0)),
    )


def _load_lightgbm(path: Path, x: np.ndarray) -> np.ndarray:
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(path))
    return np.asarray(booster.predict(x), dtype=np.float64)


def _load_xgboost(path: Path, x: np.ndarray) -> np.ndarray:
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(path)
    return np.asarray(booster.inplace_predict(x), dtype=np.float64)


def _load_catboost(path: Path, x: np.ndarray) -> np.ndarray:
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(path), format="cbm")
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def _load_extra_trees(path: Path, x: np.ndarray) -> np.ndarray:
    import joblib

    payload = joblib.load(path)
    clean = payload["imputer"].transform(x)
    return np.asarray(payload["model"].predict_proba(clean)[:, 1], dtype=np.float64)


def predict_single_saved_model(record: Mapping[str, Any], x: np.ndarray, registry_root: Path) -> np.ndarray:
    model_format = str(record["format"])
    if model_format == "constant_probability":
        return np.full(len(x), float(record["probability"]), dtype=np.float64)
    raw_path = Path(str(record["path"]))
    path = raw_path if raw_path.is_absolute() else registry_root / raw_path
    if not path.exists():
        raise FileNotFoundError(path)
    expected_sha = record.get("sha256")
    if isinstance(expected_sha, str) and sha256_file(path) != expected_sha:
        raise ValueError(f"production model SHA-256 mismatch: {path}")
    if model_format == "lightgbm_text":
        return _load_lightgbm(path, x)
    if model_format == "xgboost_json":
        return _load_xgboost(path, x)
    if model_format == "catboost_cbm":
        return _load_catboost(path, x)
    if model_format == "joblib_extra_trees":
        return _load_extra_trees(path, x)
    raise ValueError(f"지원하지 않는 production model format: {model_format}")


def predict_recipe_record(
    record: Mapping[str, Any],
    frame_values: Mapping[str, np.ndarray],
    registry_root: Path,
) -> np.ndarray:
    model_type = str(record.get("model_type", "single"))
    primary_features = [str(value) for value in record["features"]]
    primary = np.column_stack([np.asarray(frame_values[feature], dtype=np.float32) for feature in primary_features])
    if model_type == "single":
        return predict_single_saved_model(record["model"], primary, registry_root)
    if model_type == "two_stage":
        stage1 = predict_single_saved_model(record["large_move_model"], primary, registry_root)
        direction_features = [str(value) for value in record["direction_features"]]
        direction = np.column_stack([np.asarray(frame_values[feature], dtype=np.float32) for feature in direction_features])
        stage2 = predict_single_saved_model(record["direction_model"], direction, registry_root)
        return np.clip(stage1 * stage2, 1e-7, 1.0 - 1e-7)
    raise ValueError(f"지원하지 않는 production model type: {model_type}")


def relativize_registry_paths(payload: Any, root: Path) -> Any:
    """Convert model paths below root into portable POSIX relative paths."""
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "path" and isinstance(value, str):
                path = Path(value)
                try:
                    result[key] = str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
                except Exception:
                    result[key] = value
            else:
                result[key] = relativize_registry_paths(value, root)
        return result
    if isinstance(payload, list):
        return [relativize_registry_paths(value, root) for value in payload]
    return payload


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("schema") != "crashwatch_surge_production_model_registry_v4":
        raise ValueError(f"지원하지 않는 production registry schema: {payload.get('schema')}")
    return payload
