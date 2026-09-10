from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .utils import canonical_json_hash

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "crashwatch_confirmatory_3h_v3",
    "dataset_path": "AUTO",
    "output_dir": "crashwatch_ai_data/confirmatory_3h_v3",
    "cache_dir": "crashwatch_ai_data/shared_cache/confirmatory_3h_v3",
    "target_column": "label_abs_crash_20",
    "date_column": "AUTO",
    "ticker_column": "AUTO",
    "bucket_column": "AUTO",
    "common_start_date": "2021-04-16",
    "common_period_mode": "features_only_same_dates",
    "sealed_path_tokens": ["sealed", "untouched", "holdout"],
    "purge_days": 20,
    "validation_days": 60,
    "min_train_days": 500,
    "inner_validation_days": 60,
    "inner_purge_days": 20,
    "core_seeds": [17, 43, 101, 211, 503],
    "optional_seeds": [17, 101, 503],
    "core_conditions": ["B0", "A1", "A2", "A3"],
    "etf_conditions": ["A4", "A5", "A6"],
    "battery_conditions": ["S1", "S2", "S3"],
    "recent_fold_order": [7, 6, 5, 4, 3, 2, 1, 0],
    "workers": 4,
    "workers_with_gpu_audit": 3,
    "threads_per_worker": 4,
    "gpu_audit_threads": 2,
    "enable_gpu_audit": True,
    "gpu_audit_profiles": ["common_period"],
    "gpu_audit_folds": [4, 5, 6, 7],
    "gpu_audit_seeds": [17, 503],
    "gpu_audit_conditions": ["B0", "A1", "A2", "A3"],
    "soft_runtime_minutes": 165,
    "hard_runtime_minutes": 178,
    "resource_poll_seconds": 5,
    "minimum_free_ram_gb": 3.5,
    "minimum_free_disk_gb": 8.0,
    "matrix_chunk_rows": 8192,
    "save_predictions": True,
    "prediction_compression": "zstd",
    "strong_dataset_hash": False,
    "strict_reference_counts": True,
    "lightgbm": {
        "objective": "binary",
        "learning_rate": 0.028,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 55,
        "feature_fraction": 0.72,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.25,
        "lambda_l2": 1.8,
        "max_bin": 255,
        "max_rounds": 1200,
        "early_stopping_rounds": 80,
        "deterministic": True,
        "force_col_wise": True
    },
    "xgboost_gpu": {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "learning_rate": 0.025,
        "max_depth": 0,
        "max_leaves": 96,
        "grow_policy": "lossguide",
        "subsample": 0.90,
        "colsample_bytree": 0.72,
        "min_child_weight": 7.0,
        "reg_alpha": 0.20,
        "reg_lambda": 1.8,
        "max_bin": 512,
        "num_boost_round": 700
    },
    "decision_thresholds": {
        "global_min_mean_loss": 0.005,
        "global_min_positive_fold_ratio": 0.75,
        "global_worst_fold_floor": -0.005,
        "joint_incremental_min": 0.003,
        "etf_incremental_min": 0.001,
        "battery_min_mean_loss": 0.010
    }
}

@dataclass(frozen=True)
class RuntimePaths:
    package_root: Path
    project_root: Path
    reference_dir: Path
    output_dir: Path
    cache_dir: Path
    config_path: Path


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | None) -> tuple[dict[str, Any], Path]:
    if path is None:
        package_root = Path(__file__).resolve().parents[1]
        path = package_root / "confirmatory_3h_config.json"
    path = path.resolve()
    override: dict[str, Any] = {}
    if path.exists():
        override = json.loads(path.read_text(encoding="utf-8"))
    config = deep_merge(DEFAULT_CONFIG, override)
    config["config_hash"] = canonical_json_hash(config)
    return config, path


def resolve_paths(config: dict[str, Any], config_path: Path, project_root_arg: str | None) -> RuntimePaths:
    package_root = Path(__file__).resolve().parents[1]
    project_root = Path(project_root_arg).expanduser().resolve() if project_root_arg else package_root.parent.resolve()
    output_cfg = Path(str(config["output_dir"]))
    output_dir = output_cfg if output_cfg.is_absolute() else project_root / output_cfg
    cache_cfg = Path(str(config["cache_dir"]))
    cache_dir = cache_cfg if cache_cfg.is_absolute() else project_root / cache_cfg
    return RuntimePaths(package_root, project_root, package_root / "reference", output_dir, cache_dir, config_path)
