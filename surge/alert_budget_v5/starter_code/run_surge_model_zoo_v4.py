from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from surge_model_zoo_common import (
    DATE_COLUMN,
    TARGET_COLUMN,
    TARGET_VALID_COLUMN,
    TICKER_COLUMN,
    FileLock,
    FoldSpec,
    RunStatus,
    ThresholdPolicy,
    _selected_metrics,
    aggregate_metric_records,
    apply_calibrator,
    apply_threshold_policy,
    apply_training_window,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    atomic_write_text,
    build_inner_windows,
    build_sample_weights,
    choose_threshold_policy,
    compute_output_inventory,
    datewise_rank_normalize,
    daily_fraction_selection,
    derive_forward_targets,
    deterministic_seed,
    evaluate_prediction_metrics,
    evaluate_threshold_policy,
    fit_calibrator,
    hash_strings,
    join_source_and_target,
    load_folds,
    load_json,
    log,
    maximum_recall_at_alert_rate,
    parse_bool_series,
    payload_checksum_is_valid,
    rank_normalize,
    read_table,
    role_for_fold,
    safe_binary_metrics,
    sha256_bytes,
    sha256_file,
    stable_json_bytes,
    table_columns,
    theoretical_minimum_alert_rate,
    threshold_for_target_recall,
    utc_now,
    validate_folds,
    with_payload_checksum,
)
from surge_model_zoo_deployment import (
    relativize_registry_paths,
    save_catboost_model,
    save_extra_trees_model,
    save_lightgbm_model,
    save_xgboost_model,
)


DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "metric": "average_precision",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": -1,
    "min_data_in_leaf": 80,
    "lambda_l1": 0.2,
    "lambda_l2": 2.0,
    "max_bin": 255,
    "min_gain_to_split": 1e-12,
    "feature_pre_filter": False,
    "verbosity": -1,
    "force_col_wise": True,
    "deterministic": False,
}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "learning_rate": 0.03,
    "grow_policy": "lossguide",
    "max_depth": 0,
    "max_leaves": 96,
    "min_child_weight": 8.0,
    "subsample": 0.82,
    "colsample_bytree": 0.80,
    "colsample_bynode": 0.88,
    "reg_alpha": 0.25,
    "reg_lambda": 2.2,
    "max_bin": 256,
    "verbosity": 0,
}

DEFAULT_CAT_PARAMS: dict[str, Any] = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.035,
    "depth": 8,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.8,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.82,
    "border_count": 128,
    "verbose": False,
    "allow_writing_files": False,
}

DEFAULT_EXTRA_PARAMS: dict[str, Any] = {
    "n_estimators": 700,
    "max_features": 0.65,
    "min_samples_leaf": 5,
    "max_depth": None,
}


@dataclass(frozen=True)
class Recipe:
    name: str
    family: str
    profile: str
    target_variant: str
    train_policy: str
    positive_weight_mode: str
    params: dict[str, Any]
    time_decay_half_life_days: float | None = None
    rolling_days: int | None = None
    direction_profile: str | None = None
    direction_params: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Recipe":
        return cls(
            name=str(record["name"]),
            family=str(record["family"]),
            profile=str(record["profile"]),
            target_variant=str(record.get("target_variant", "surge_d3")),
            train_policy=str(record.get("train_policy", "expanding")),
            positive_weight_mode=str(record.get("positive_weight_mode", "none")),
            params=dict(record.get("params", {})),
            time_decay_half_life_days=(
                float(record["time_decay_half_life_days"])
                if record.get("time_decay_half_life_days") is not None
                else None
            ),
            rolling_days=int(record["rolling_days"]) if record.get("rolling_days") is not None else None,
            direction_profile=str(record["direction_profile"]) if record.get("direction_profile") else None,
            direction_params=dict(record.get("direction_params", {})) if record.get("direction_params") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "profile": self.profile,
            "target_variant": self.target_variant,
            "train_policy": self.train_policy,
            "positive_weight_mode": self.positive_weight_mode,
            "params": self.params,
            "time_decay_half_life_days": self.time_decay_half_life_days,
            "rolling_days": self.rolling_days,
            "direction_profile": self.direction_profile,
            "direction_params": self.direction_params,
        }


@dataclass
class DataBundle:
    matrix_path: Path
    metadata_path: Path
    features: list[str]
    feature_to_index: dict[str, int]
    frame: pd.DataFrame
    targets: dict[str, np.ndarray]
    dates: np.ndarray
    dataset_sha256: str
    target_sha256: str
    cache_identity_hash: str


@dataclass(frozen=True)
class FitResult:
    prediction: np.ndarray
    best_iteration: int | dict[str, int]


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("정수 목록이 비어 있습니다")
    return list(dict.fromkeys(values))


def parse_str_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def resolve_paths(args: argparse.Namespace) -> None:
    root = args.package_root.resolve()
    args.package_root = root
    if args.dataset is None:
        args.dataset = root / "data" / "training_dataset_finance11h.parquet"
    if args.target_sidecar is None:
        args.target_sidecar = root / "data" / "surge_target_3d5.parquet"
    if args.folds is None:
        candidates = [
            root / "outputs" / "surge_correlation_map_complete" / "walk_forward_folds.json",
            root / "references" / "feature_metadata" / "outer_walk_forward_folds.json",
        ]
        args.folds = next((path for path in candidates if path.exists()), candidates[-1])
    if args.profiles is None:
        candidates = [
            root / "outputs" / "surge_pre_model_gate_v3" / "surge_feature_profiles_corrected.json",
            root / "pre_model_gate" / "surge_feature_profiles_corrected.json",
        ]
        args.profiles = next((path for path in candidates if path.exists()), candidates[0])
    if args.recipes is None:
        args.recipes = Path(__file__).resolve().parent / "default_model_zoo_recipes.json"
    if args.output is None:
        args.output = root / "outputs" / "surge_model_zoo_v4"
    args.dataset = Path(args.dataset).resolve()
    args.target_sidecar = Path(args.target_sidecar).resolve()
    args.folds = Path(args.folds).resolve()
    args.profiles = Path(args.profiles).resolve()
    args.recipes = Path(args.recipes).resolve()
    args.output = Path(args.output).resolve()


def require_inputs(args: argparse.Namespace) -> None:
    paths = {
        "dataset": args.dataset,
        "target_sidecar": args.target_sidecar,
        "folds": args.folds,
        "profiles": args.profiles,
        "recipes": args.recipes,
    }
    missing = {name: str(path) for name, path in paths.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(json.dumps(missing, ensure_ascii=False, indent=2))


def load_profiles(profile_path: Path, recipe_payload: Mapping[str, Any]) -> dict[str, list[str]]:
    payload = load_json(profile_path)
    raw_profiles = payload.get("profiles", payload)
    if not isinstance(raw_profiles, dict):
        raise ValueError("profile JSON 형식이 잘못되었습니다")
    profiles: dict[str, list[str]] = {}
    for name, record in raw_profiles.items():
        if isinstance(record, dict):
            features = record.get("features", [])
        else:
            features = record
        if isinstance(features, list):
            profiles[str(name)] = [str(value) for value in features]
    custom = recipe_payload.get("custom_profiles", {})
    for name, record in custom.items():
        if "base" in record:
            base = str(record["base"])
            if base not in profiles:
                raise KeyError(f"custom profile base 누락: {base}")
            values = list(profiles[base])
            values.extend(str(value) for value in record.get("add", []))
            drop = {str(value) for value in record.get("drop", [])}
            profiles[str(name)] = [value for value in dict.fromkeys(values) if value not in drop]
        elif "union" in record:
            values: list[str] = []
            for base in record["union"]:
                if str(base) not in profiles:
                    raise KeyError(f"custom profile union 누락: {base}")
                values.extend(profiles[str(base)])
            profiles[str(name)] = list(dict.fromkeys(values))
        else:
            raise ValueError(f"custom profile 정의 오류: {name}")
    if "P0_ALL_VALID" not in profiles:
        raise KeyError("P0_ALL_VALID profile이 필요합니다")
    return profiles


def load_recipes(path: Path, quick: bool, allowed_families: Sequence[str] | None) -> tuple[dict[str, Any], list[Recipe]]:
    payload = load_json(path)
    recipes = [Recipe.from_dict(record) for record in payload.get("recipes", [])]
    if quick:
        quick_names = {str(value) for value in payload.get("quick_recipe_names", [])}
        recipes = [recipe for recipe in recipes if recipe.name in quick_names]
    if allowed_families:
        allowed = {str(value) for value in allowed_families}
        recipes = [recipe for recipe in recipes if recipe.family in allowed]
    names = [recipe.name for recipe in recipes]
    if len(names) != len(set(names)):
        raise ValueError("recipe name이 중복되었습니다")
    if not recipes:
        raise ValueError("실행할 recipe가 없습니다")
    return payload, recipes


def _xgboost_cuda_smoke(xgb: Any) -> tuple[bool, str | None]:
    """Run a minimal CUDA train so a CUDA build is not mistaken for a usable GPU."""
    try:
        x = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.3, 0.7], [0.8, 0.2]],
            dtype=np.float32,
        )
        y = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.uint8)
        matrix = xgb.QuantileDMatrix(x, label=y, max_bin=16)
        xgb.train(
            {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "tree_method": "hist",
                "device": "cuda",
                "max_depth": 2,
                "max_bin": 16,
                "verbosity": 0,
            },
            matrix,
            num_boost_round=2,
            verbose_eval=False,
        )
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _catboost_gpu_smoke() -> tuple[bool, str | None]:
    """Run a minimal CatBoost GPU fit to verify driver and device availability."""
    try:
        from catboost import CatBoostClassifier

        x = np.asarray(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.3, 0.7], [0.8, 0.2]],
            dtype=np.float32,
        )
        y = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.uint8)
        model = CatBoostClassifier(
            iterations=2,
            depth=2,
            learning_rate=0.1,
            loss_function="Logloss",
            task_type="GPU",
            devices="0",
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(x, y, verbose=False)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def preflight_families(recipes: Sequence[Recipe], device: str, allow_cpu_fallback: bool) -> dict[str, Any]:
    families = {recipe.family for recipe in recipes}
    audit: dict[str, Any] = {
        "requested_device": device,
        "allow_cpu_fallback": bool(allow_cpu_fallback),
        "nvidia_smi_present": bool(shutil.which("nvidia-smi")),
    }
    if "lightgbm" in families:
        import lightgbm as lgb

        audit["lightgbm"] = {"version": str(lgb.__version__), "device": "cpu"}
    if {"xgboost", "two_stage_xgboost"} & families:
        import xgboost as xgb

        build_info = xgb.build_info() if hasattr(xgb, "build_info") else {}
        use_cuda = bool(build_info.get("USE_CUDA", False))
        gpu_candidate = bool(use_cuda and shutil.which("nvidia-smi"))
        resolved = device
        if device == "auto":
            resolved = "cuda" if gpu_candidate else "cpu"
        fallback_reason: str | None = None
        smoke_passed: bool | None = None
        if resolved == "cuda":
            if not use_cuda:
                fallback_reason = "XGBoost CUDA 빌드가 아닙니다"
                smoke_passed = False
            elif not shutil.which("nvidia-smi"):
                fallback_reason = "nvidia-smi를 찾지 못했습니다"
                smoke_passed = False
            else:
                smoke_passed, fallback_reason = _xgboost_cuda_smoke(xgb)
            if smoke_passed is not True:
                if device == "auto" or allow_cpu_fallback:
                    resolved = "cpu"
                else:
                    raise RuntimeError(f"XGBoost CUDA preflight 실패: {fallback_reason}")
        audit["xgboost"] = {
            "version": str(xgb.__version__),
            "build_info": build_info,
            "resolved_device": resolved,
            "cuda_smoke_passed": smoke_passed,
            "fallback_reason": fallback_reason if resolved == "cpu" else None,
        }
    if "catboost" in families:
        import catboost

        gpu_candidate = bool(shutil.which("nvidia-smi"))
        requested = device
        resolved = "GPU" if (device == "cuda" or (device == "auto" and gpu_candidate)) else "CPU"
        fallback_reason: str | None = None
        smoke_passed: bool | None = None
        if resolved == "GPU":
            if not gpu_candidate:
                smoke_passed = False
                fallback_reason = "nvidia-smi를 찾지 못했습니다"
            else:
                smoke_passed, fallback_reason = _catboost_gpu_smoke()
            if smoke_passed is not True:
                if requested == "auto" or allow_cpu_fallback:
                    resolved = "CPU"
                else:
                    raise RuntimeError(f"CatBoost GPU preflight 실패: {fallback_reason}")
        audit["catboost"] = {
            "version": str(catboost.__version__),
            "resolved_task_type": resolved,
            "gpu_smoke_passed": smoke_passed,
            "fallback_reason": fallback_reason if resolved == "CPU" else None,
        }
    if "extra_trees" in families:
        import sklearn

        audit["extra_trees"] = {"sklearn_version": str(sklearn.__version__), "device": "cpu"}
    return audit


def filter_available_recipes(
    recipes: Sequence[Recipe],
    profiles: Mapping[str, Sequence[str]],
    dataset_columns: Sequence[str],
) -> tuple[list[Recipe], list[dict[str, Any]]]:
    columns = set(dataset_columns)
    usable: list[Recipe] = []
    skipped: list[dict[str, Any]] = []
    for recipe in recipes:
        if recipe.profile not in profiles:
            skipped.append({"recipe": recipe.name, "reason": f"profile 누락: {recipe.profile}"})
            continue
        if recipe.direction_profile and recipe.direction_profile not in profiles:
            skipped.append({"recipe": recipe.name, "reason": f"direction profile 누락: {recipe.direction_profile}"})
            continue
        required = list(profiles[recipe.profile])
        if recipe.direction_profile:
            required.extend(profiles[recipe.direction_profile])
        missing = [feature for feature in required if feature not in columns]
        if missing:
            skipped.append({"recipe": recipe.name, "reason": f"dataset 피처 누락 {len(missing)}개", "missing": missing})
            continue
        usable.append(recipe)
    return usable, skipped


def _matrix_cache_is_valid(metadata_path: Path, identity_hash: str) -> bool:
    if not metadata_path.exists():
        return False
    try:
        metadata = load_json(metadata_path)
    except Exception:
        return False
    if metadata.get("identity_hash") != identity_hash or not payload_checksum_is_valid(metadata):
        return False
    for key in ["matrix_path", "frame_path", "targets_path", "target_audit_path"]:
        if key not in metadata or f"{key}_sha256" not in metadata:
            return False
        path = Path(metadata[key])
        if not path.exists() or sha256_file(path) != metadata[f"{key}_sha256"]:
            return False
    return True


def prepare_data_bundle(
    args: argparse.Namespace,
    profiles: Mapping[str, Sequence[str]],
    recipes: Sequence[Recipe],
    folds: Sequence[FoldSpec],
) -> DataBundle:
    cache_dir = args.output / "matrix_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = cache_dir / "feature_matrix.npy"
    frame_path = cache_dir / "metadata.pkl"
    targets_path = cache_dir / "targets.npz"
    metadata_path = cache_dir / "CACHE_MANIFEST.json"
    target_audit_path = args.output / "TARGET_REAUDIT.json"

    full_features = list(profiles["P0_ALL_VALID"])
    needed_features: list[str] = []
    for recipe in recipes:
        needed_features.extend(profiles[recipe.profile])
        if recipe.direction_profile:
            needed_features.extend(profiles[recipe.direction_profile])
    needed_features = list(dict.fromkeys(needed_features))
    missing_from_full = [value for value in needed_features if value not in full_features]
    full_features.extend(missing_from_full)

    dataset_hash = sha256_file(args.dataset)
    target_hash = sha256_file(args.target_sidecar)
    identity = {
        "schema": "surge_model_zoo_matrix_cache_v4",
        "dataset_sha256": dataset_hash,
        "target_sha256": target_hash,
        "features": full_features,
        "feature_hash": hash_strings(full_features),
        "date_column": args.date_column,
        "ticker_column": args.ticker_column,
        "target_column": args.target_column,
        "target_valid_column": args.target_valid_column,
        "return_column": args.return_column,
        "scope_columns": args.scope_columns,
        "folds": [fold.to_dict() for fold in folds],
    }
    identity_hash = sha256_bytes(stable_json_bytes(identity))
    if args.resume and _matrix_cache_is_valid(metadata_path, identity_hash):
        manifest = load_json(metadata_path)
        frame = pd.read_pickle(frame_path)
        arrays = np.load(targets_path, allow_pickle=False)
        targets = {name: np.asarray(arrays[name]) for name in arrays.files if name.startswith("target__")}
        targets = {name.removeprefix("target__"): values for name, values in targets.items()}
        dates = np.asarray(arrays["dates"], dtype=np.int64).astype("datetime64[ns]")
        return DataBundle(
            matrix_path=matrix_path,
            metadata_path=metadata_path,
            features=full_features,
            feature_to_index={feature: index for index, feature in enumerate(full_features)},
            frame=frame,
            targets=targets,
            dates=dates,
            dataset_sha256=dataset_hash,
            target_sha256=target_hash,
            cache_identity_hash=identity_hash,
        )

    dataset_columns = table_columns(args.dataset)
    required_source = [
        "source_row_id",
        args.date_column,
        args.ticker_column,
        "sealed_do_not_train_or_tune",
        args.return_column,
        *args.scope_columns,
        *full_features,
    ]
    required_source = [value for value in dict.fromkeys(required_source) if value in dataset_columns]
    if args.return_column not in required_source:
        raise KeyError(f"2-stage/target 감사에 필요한 return column 누락: {args.return_column}")
    source = read_table(args.dataset, columns=required_source)
    sidecar_columns = table_columns(args.target_sidecar)
    required_sidecar = [
        "source_row_id",
        args.date_column,
        args.ticker_column,
        args.target_column,
        args.target_valid_column,
        "first_hit_day",
        "best_forward_return_3d",
    ]
    required_sidecar = [value for value in required_sidecar if value in sidecar_columns]
    sidecar = read_table(args.target_sidecar, columns=required_sidecar)
    joined = join_source_and_target(
        source,
        sidecar,
        target_column=args.target_column,
        target_valid_column=args.target_valid_column,
        date_column=args.date_column,
        ticker_column=args.ticker_column,
    )
    if "sealed_do_not_train_or_tune" in joined.columns:
        sealed = pd.to_numeric(joined["sealed_do_not_train_or_tune"], errors="coerce").fillna(0)
        if sealed.ne(0).any():
            raise ValueError(f"sealed 행이 존재합니다: {int(sealed.ne(0).sum())}")
    joined[args.date_column] = pd.to_datetime(joined[args.date_column], errors="coerce")
    joined[args.ticker_column] = joined[args.ticker_column].astype("string").str.strip()

    derived = derive_forward_targets(
        joined,
        return_column=args.return_column,
        ticker_column=args.ticker_column,
        date_column=args.date_column,
        threshold=0.05,
    )
    side_valid = parse_bool_series(joined[args.target_valid_column])
    derived_forward_valid = derived["forward_valid"].eq(1.0)
    validity_mismatch = int(np.sum(side_valid.to_numpy(dtype=bool) != derived_forward_valid.to_numpy(dtype=bool)))
    side_label = pd.to_numeric(joined[args.target_column], errors="coerce")
    common_valid = side_valid & derived_forward_valid & joined[args.date_column].notna()
    valid_label_mask = side_label.isin([0, 1])
    missing_valid_labels = int(np.sum(common_valid & ~valid_label_mask))
    common_comparable = common_valid & valid_label_mask
    label_mismatch = int(
        np.sum(
            side_label.loc[common_comparable].to_numpy(dtype=np.uint8)
            != derived.loc[common_comparable.to_numpy(), "surge_d3"].to_numpy(dtype=np.uint8)
        )
    )

    first_hit_mismatch: int | None = None
    if "first_hit_day" in joined.columns:
        side_hit = pd.to_numeric(joined["first_hit_day"], errors="coerce").to_numpy(dtype=np.float64)
        # The production Surge 3D5 sidecar stores 0 as the no-hit sentinel,
        # while derive_forward_targets uses NaN for the same state.  Normalize
        # the serialized sentinel before the independent target comparison.
        side_hit[side_hit == 0.0] = np.nan
        derived_hit = derived["first_hit_day"].to_numpy(dtype=np.float64)
        hit_equal = (np.isnan(side_hit) & np.isnan(derived_hit)) | np.isclose(
            side_hit, derived_hit, atol=1e-12, rtol=0.0, equal_nan=True
        )
        first_hit_mismatch = int(np.sum(common_valid.to_numpy(dtype=bool) & ~hit_equal))

    best_forward_mismatch: int | None = None
    best_forward_max_abs_error: float | None = None
    if "best_forward_return_3d" in joined.columns:
        side_best = pd.to_numeric(joined["best_forward_return_3d"], errors="coerce").to_numpy(dtype=np.float64)
        derived_best = derived["best_forward_return_3d"].to_numpy(dtype=np.float64)
        comparable = common_valid.to_numpy(dtype=bool) & np.isfinite(side_best) & np.isfinite(derived_best)
        best_nan_equal = (np.isnan(side_best) & np.isnan(derived_best))
        best_close = np.isclose(side_best, derived_best, atol=1e-7, rtol=1e-7, equal_nan=True)
        best_forward_mismatch = int(np.sum(common_valid.to_numpy(dtype=bool) & ~(best_nan_equal | best_close)))
        if comparable.any():
            best_forward_max_abs_error = float(np.max(np.abs(side_best[comparable] - derived_best[comparable])))

    target_audit = with_payload_checksum(
        {
            "schema": "crashwatch_surge_model_zoo_target_reaudit_v4",
            "dataset_sha256": dataset_hash,
            "target_sha256": target_hash,
            "rows": int(len(joined)),
            "sidecar_valid_rows": int(side_valid.sum()),
            "derived_valid_rows": int(derived_forward_valid.sum()),
            "validity_mismatch_rows": validity_mismatch,
            "missing_label_on_common_valid_rows": missing_valid_labels,
            "label_mismatch_rows": label_mismatch,
            "first_hit_day_mismatch_rows": first_hit_mismatch,
            "best_forward_return_mismatch_rows": best_forward_mismatch,
            "best_forward_return_max_abs_error": best_forward_max_abs_error,
            "threshold": 0.05,
            "horizon_trading_days": 3,
            "inclusive_absolute_tolerance": 1e-12,
            "status": "PASS" if all(
                value in {0, None}
                for value in [
                    validity_mismatch,
                    missing_valid_labels,
                    label_mismatch,
                    first_hit_mismatch,
                    best_forward_mismatch,
                ]
            ) else "FAIL",
        }
    )
    atomic_write_json(target_audit_path, target_audit)
    if target_audit["status"] != "PASS":
        raise ValueError(
            "Surge target 독립 재감사 실패: "
            f"valid={validity_mismatch}, missing_label={missing_valid_labels}, "
            f"label={label_mismatch}, first_hit={first_hit_mismatch}, best_forward={best_forward_mismatch}"
        )

    valid = common_comparable
    data = joined.loc[valid].copy().reset_index(drop=True)
    derived_valid = derived.loc[valid.to_numpy()].reset_index(drop=True)
    target_d3 = pd.to_numeric(data[args.target_column], errors="coerce").to_numpy(dtype=np.uint8)
    derived_d3 = derived_valid["surge_d3"].to_numpy(dtype=np.uint8)
    if "first_hit_day" in data.columns:
        hit = pd.to_numeric(data["first_hit_day"], errors="coerce").to_numpy(dtype=np.float64)
        surge_d1 = ((target_d3 == 1) & np.isfinite(hit) & (hit <= 1)).astype(np.uint8)
        surge_d2 = ((target_d3 == 1) & np.isfinite(hit) & (hit <= 2)).astype(np.uint8)
    else:
        surge_d1 = derived_valid["surge_d1"].to_numpy(dtype=np.uint8)
        surge_d2 = derived_valid["surge_d2"].to_numpy(dtype=np.uint8)
    crash_d3 = derived_valid["crash_d3"].to_numpy(dtype=np.uint8)
    large_move = ((target_d3 == 1) | (crash_d3 == 1)).astype(np.uint8)
    direction_valid = ((target_d3 != crash_d3) & (large_move == 1)).astype(np.uint8)
    direction_up = target_d3.copy()

    frame_columns = [
        "source_row_id",
        args.date_column,
        args.ticker_column,
        *args.scope_columns,
    ]
    frame_columns = [column for column in dict.fromkeys(frame_columns) if column in data.columns]
    frame = data[frame_columns].copy()
    frame["row_index"] = np.arange(len(frame), dtype=np.int64)
    validate_folds(folds, frame[args.date_column], minimum_purge_days=args.minimum_purge_trading_days)

    log(f"matrix 생성: {len(data):,}행 × {len(full_features):,}피처")
    matrix = np.lib.format.open_memmap(
        matrix_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(data), len(full_features)),
    )
    block_size = max(1, int(args.cache_column_block))
    for start in range(0, len(full_features), block_size):
        block_features = full_features[start : start + block_size]
        block = data[block_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
        block[~np.isfinite(block)] = np.nan
        matrix[:, start : start + len(block_features)] = block
        matrix.flush()
    del matrix, source, sidecar, joined, data, derived, derived_valid
    gc.collect()

    frame.to_pickle(frame_path)
    dates = pd.to_datetime(frame[args.date_column], errors="coerce").to_numpy(dtype="datetime64[ns]")
    atomic_write_npz(
        targets_path,
        dates=dates.astype(np.int64),
        target__surge_d1=surge_d1,
        target__surge_d2=surge_d2,
        target__surge_d3=target_d3,
        target__crash_d3=crash_d3,
        target__large_move=large_move,
        target__direction_valid=direction_valid,
        target__direction_up=direction_up,
    )
    payload = with_payload_checksum(
        {
            "identity": identity,
            "identity_hash": identity_hash,
            "rows": int(len(frame)),
            "feature_count": int(len(full_features)),
            "matrix_path": str(matrix_path),
            "matrix_path_sha256": sha256_file(matrix_path),
            "frame_path": str(frame_path),
            "frame_path_sha256": sha256_file(frame_path),
            "targets_path": str(targets_path),
            "targets_path_sha256": sha256_file(targets_path),
            "target_audit_path": str(target_audit_path),
            "target_audit_path_sha256": sha256_file(target_audit_path),
            "target_counts": {
                name: int(np.sum(values == 1))
                for name, values in {
                    "surge_d1": surge_d1,
                    "surge_d2": surge_d2,
                    "surge_d3": target_d3,
                    "crash_d3": crash_d3,
                    "large_move": large_move,
                    "direction_valid": direction_valid,
                }.items()
            },
        }
    )
    atomic_write_json(metadata_path, payload)
    return DataBundle(
        matrix_path=matrix_path,
        metadata_path=metadata_path,
        features=full_features,
        feature_to_index={feature: index for index, feature in enumerate(full_features)},
        frame=frame,
        targets={
            "surge_d1": surge_d1,
            "surge_d2": surge_d2,
            "surge_d3": target_d3,
            "crash_d3": crash_d3,
            "large_move": large_move,
            "direction_valid": direction_valid,
            "direction_up": direction_up,
        },
        dates=dates,
        dataset_sha256=dataset_hash,
        target_sha256=target_hash,
        cache_identity_hash=identity_hash,
    )


def fold_indices(bundle: DataBundle, fold: FoldSpec) -> tuple[np.ndarray, np.ndarray]:
    dates = bundle.dates
    train = np.flatnonzero(
        (dates >= np.datetime64(fold.train_start)) & (dates <= np.datetime64(fold.train_end))
    )
    valid = np.flatnonzero(
        (dates >= np.datetime64(fold.validation_start)) & (dates <= np.datetime64(fold.validation_end))
    )
    if not len(train) or not len(valid):
        raise ValueError(f"fold {fold.fold_id}: 빈 train/validation")
    return train, valid


def resolve_model_device(family: str, preflight: Mapping[str, Any]) -> str:
    if family in {"xgboost", "two_stage_xgboost"}:
        return str(preflight.get("xgboost", {}).get("resolved_device", "cpu"))
    if family == "catboost":
        return str(preflight.get("catboost", {}).get("resolved_task_type", "CPU"))
    return "cpu"


def _prepare_extra_matrix(x_train: np.ndarray, x_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.impute import SimpleImputer

    imputer = SimpleImputer(strategy="median", add_indicator=True)
    return imputer.fit_transform(x_train), imputer.transform(x_valid)


def fit_predict_lightgbm(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray | None,
    params: Mapping[str, Any],
    seed: int,
    threads: int,
    max_rounds: int,
    early_stopping_rounds: int | None,
    fixed_iterations: int | None,
) -> FitResult:
    import lightgbm as lgb

    resolved = dict(DEFAULT_LGB_PARAMS)
    resolved.update(params)
    resolved.update(
        {
            "num_threads": int(threads),
            "seed": int(seed),
            "feature_fraction_seed": int(seed),
            "bagging_seed": int(seed),
            "data_random_seed": int(seed),
        }
    )
    train_set = lgb.Dataset(x_train, label=y_train, weight=weights, free_raw_data=True)
    if fixed_iterations is not None:
        booster = lgb.train(
            resolved,
            train_set,
            num_boost_round=int(fixed_iterations),
            callbacks=[lgb.log_evaluation(period=0)],
        )
        prediction = booster.predict(x_valid, num_iteration=int(fixed_iterations))
        return FitResult(np.asarray(prediction, dtype=np.float64), int(fixed_iterations))
    if y_valid is None:
        raise ValueError("early stopping에는 y_valid이 필요합니다")
    valid_set = lgb.Dataset(x_valid, label=y_valid, reference=train_set, free_raw_data=True)
    callbacks = [lgb.log_evaluation(period=0)]
    if early_stopping_rounds:
        callbacks.append(lgb.early_stopping(int(early_stopping_rounds), verbose=False))
    booster = lgb.train(
        resolved,
        train_set,
        num_boost_round=int(max_rounds),
        valid_sets=[valid_set],
        callbacks=callbacks,
    )
    best = int(booster.best_iteration or max_rounds)
    prediction = booster.predict(x_valid, num_iteration=best)
    return FitResult(np.asarray(prediction, dtype=np.float64), best)


def fit_predict_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray | None,
    params: Mapping[str, Any],
    seed: int,
    threads: int,
    device: str,
    max_rounds: int,
    early_stopping_rounds: int | None,
    fixed_iterations: int | None,
    early_stopping_x: np.ndarray | None = None,
    early_stopping_y: np.ndarray | None = None,
) -> FitResult:
    import xgboost as xgb

    resolved = dict(DEFAULT_XGB_PARAMS)
    resolved.update(params)
    resolved.update({"seed": int(seed), "nthread": int(threads), "device": device})
    max_bin = int(resolved.get("max_bin", 256))
    dtrain = xgb.QuantileDMatrix(x_train, label=y_train, weight=weights, max_bin=max_bin)
    dpredict = xgb.QuantileDMatrix(x_valid, ref=dtrain, max_bin=max_bin)
    if fixed_iterations is not None:
        booster = xgb.train(resolved, dtrain, num_boost_round=int(fixed_iterations), verbose_eval=False)
        prediction = booster.predict(dpredict, iteration_range=(0, int(fixed_iterations)))
        return FitResult(np.asarray(prediction, dtype=np.float64), int(fixed_iterations))

    eval_x = early_stopping_x if early_stopping_x is not None else x_valid
    eval_y = early_stopping_y if early_stopping_y is not None else y_valid
    if eval_y is None:
        raise ValueError("XGBoost early stopping에는 validation label이 필요합니다")
    deval = xgb.QuantileDMatrix(eval_x, label=eval_y, ref=dtrain, max_bin=max_bin)
    booster = xgb.train(
        resolved,
        dtrain,
        num_boost_round=int(max_rounds),
        evals=[(deval, "validation")],
        early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds else None,
        verbose_eval=False,
    )
    best = int((booster.best_iteration + 1) if booster.best_iteration is not None else max_rounds)
    prediction = booster.predict(dpredict, iteration_range=(0, best))
    return FitResult(np.asarray(prediction, dtype=np.float64), best)


def fit_predict_catboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray | None,
    params: Mapping[str, Any],
    seed: int,
    threads: int,
    task_type: str,
    max_rounds: int,
    early_stopping_rounds: int | None,
    fixed_iterations: int | None,
) -> FitResult:
    from catboost import CatBoostClassifier, Pool

    resolved = dict(DEFAULT_CAT_PARAMS)
    resolved.update(params)
    iterations = int(fixed_iterations if fixed_iterations is not None else max_rounds)
    resolved.update(
        {
            "iterations": iterations,
            "random_seed": int(seed),
            "thread_count": int(threads),
            "task_type": task_type,
        }
    )
    train_pool = Pool(x_train, label=y_train, weight=weights)
    valid_pool = Pool(x_valid, label=y_valid) if y_valid is not None else None
    model = CatBoostClassifier(**resolved)
    if fixed_iterations is None and valid_pool is not None:
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds else None,
            verbose=False,
        )
        best = int(model.get_best_iteration() + 1) if model.get_best_iteration() >= 0 else iterations
    else:
        model.fit(train_pool, verbose=False)
        best = iterations
    prediction = model.predict_proba(x_valid)[:, 1]
    return FitResult(np.asarray(prediction, dtype=np.float64), best)


def fit_predict_extra_trees(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_valid: np.ndarray,
    params: Mapping[str, Any],
    seed: int,
    threads: int,
) -> FitResult:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.impute import SimpleImputer

    resolved = dict(DEFAULT_EXTRA_PARAMS)
    resolved.update(params)
    imputer = SimpleImputer(strategy="median", add_indicator=True)
    x_train_clean = imputer.fit_transform(x_train)
    x_valid_clean = imputer.transform(x_valid)
    model = ExtraTreesClassifier(
        **resolved,
        random_state=int(seed),
        n_jobs=int(threads),
        bootstrap=False,
        criterion="log_loss",
    )
    model.fit(x_train_clean, y_train, sample_weight=weights)
    prediction = model.predict_proba(x_valid_clean)[:, 1]
    return FitResult(np.asarray(prediction, dtype=np.float64), int(resolved["n_estimators"]))


def fit_predict_base_family(
    family: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray | None,
    params: Mapping[str, Any],
    seed: int,
    args: argparse.Namespace,
    device: str,
    fixed_iterations: int | None,
) -> FitResult:
    if family == "lightgbm":
        return fit_predict_lightgbm(
            x_train,
            y_train,
            weights,
            x_valid,
            y_valid,
            params,
            seed,
            args.threads_per_model,
            args.max_tuning_rounds,
            args.early_stopping_rounds,
            fixed_iterations,
        )
    if family == "xgboost":
        return fit_predict_xgboost(
            x_train,
            y_train,
            weights,
            x_valid,
            y_valid,
            params,
            seed,
            args.xgboost_threads,
            device,
            args.max_tuning_rounds,
            args.early_stopping_rounds,
            fixed_iterations,
        )
    if family == "catboost":
        return fit_predict_catboost(
            x_train,
            y_train,
            weights,
            x_valid,
            y_valid,
            params,
            seed,
            args.threads_per_model,
            device,
            args.max_tuning_rounds,
            args.early_stopping_rounds,
            fixed_iterations,
        )
    if family == "extra_trees":
        return fit_predict_extra_trees(
            x_train,
            y_train,
            weights,
            x_valid,
            params,
            seed,
            args.threads_per_model,
        )
    raise ValueError(f"지원하지 않는 family: {family}")


def _feature_indices(recipe: Recipe, profiles: Mapping[str, Sequence[str]], bundle: DataBundle) -> tuple[np.ndarray, np.ndarray | None]:
    primary = np.asarray([bundle.feature_to_index[value] for value in profiles[recipe.profile]], dtype=np.int64)
    direction = None
    if recipe.direction_profile:
        direction = np.asarray(
            [bundle.feature_to_index[value] for value in profiles[recipe.direction_profile]],
            dtype=np.int64,
        )
    return primary, direction


def fit_predict_recipe(
    recipe: Recipe,
    matrix: np.ndarray,
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: str,
    fixed_iterations: int | dict[str, int] | None,
) -> FitResult:
    primary_indices, direction_indices = _feature_indices(recipe, profiles, bundle)
    target_name = recipe.target_variant if recipe.target_variant != "two_stage" else "surge_d3"
    train_indices = apply_training_window(
        train_indices,
        bundle.dates,
        recipe.train_policy,
        recipe.rolling_days,
    )
    if len(train_indices) < 100:
        raise ValueError(f"{recipe.name}: 학습 행이 너무 적습니다")

    if recipe.family != "two_stage_xgboost":
        y_all = bundle.targets[target_name]
        y_train = y_all[train_indices]
        y_valid = y_all[valid_indices]
        weights = build_sample_weights(
            y_all,
            bundle.dates,
            train_indices,
            recipe.positive_weight_mode,
            recipe.time_decay_half_life_days,
        )
        x_train = np.asarray(matrix[np.ix_(train_indices, primary_indices)], dtype=np.float32)
        x_valid = np.asarray(matrix[np.ix_(valid_indices, primary_indices)], dtype=np.float32)
        iteration = int(fixed_iterations) if isinstance(fixed_iterations, int) else None
        return fit_predict_base_family(
            recipe.family,
            x_train,
            y_train,
            weights,
            x_valid,
            y_valid,
            recipe.params,
            seed,
            args,
            device,
            iteration,
        )

    if direction_indices is None:
        direction_indices = primary_indices
    fixed = fixed_iterations if isinstance(fixed_iterations, dict) else {}
    large = bundle.targets["large_move"]
    x_train_large = np.asarray(matrix[np.ix_(train_indices, primary_indices)], dtype=np.float32)
    x_valid_large = np.asarray(matrix[np.ix_(valid_indices, primary_indices)], dtype=np.float32)
    weight_large = build_sample_weights(
        large,
        bundle.dates,
        train_indices,
        recipe.positive_weight_mode,
        recipe.time_decay_half_life_days,
    )
    stage1 = fit_predict_xgboost(
        x_train_large,
        large[train_indices],
        weight_large,
        x_valid_large,
        large[valid_indices],
        recipe.params,
        seed,
        args.xgboost_threads,
        device,
        args.max_tuning_rounds,
        args.early_stopping_rounds,
        int(fixed["large"]) if "large" in fixed else None,
    )

    direction_valid = bundle.targets["direction_valid"].astype(bool)
    direction_train = train_indices[direction_valid[train_indices]]
    direction_validation_mask = direction_valid[valid_indices]
    if len(direction_train) < 200 or int(np.sum(direction_validation_mask)) < 20:
        direction_probability = np.full(len(valid_indices), float(np.mean(bundle.targets["surge_d3"][train_indices])), dtype=np.float64)
        direction_iteration = 1
    else:
        x_train_direction = np.asarray(matrix[np.ix_(direction_train, direction_indices)], dtype=np.float32)
        x_valid_direction = np.asarray(matrix[np.ix_(valid_indices, direction_indices)], dtype=np.float32)
        direction_eval_indices = valid_indices[direction_validation_mask]
        x_eval_direction = np.asarray(matrix[np.ix_(direction_eval_indices, direction_indices)], dtype=np.float32)
        direction_target = bundle.targets["direction_up"]
        direction_weights = build_sample_weights(
            direction_target,
            bundle.dates,
            direction_train,
            "sqrt_balance",
            recipe.time_decay_half_life_days,
        )
        direction_params = recipe.direction_params or recipe.params
        stage2 = fit_predict_xgboost(
            x_train_direction,
            direction_target[direction_train],
            direction_weights,
            x_valid_direction,
            None,
            direction_params,
            deterministic_seed(seed, "direction"),
            args.xgboost_threads,
            device,
            args.max_tuning_rounds,
            args.early_stopping_rounds,
            int(fixed["direction"]) if "direction" in fixed else None,
            early_stopping_x=x_eval_direction,
            early_stopping_y=direction_target[direction_eval_indices],
        )
        direction_probability = stage2.prediction
        direction_iteration = int(stage2.best_iteration)
    prediction = np.clip(stage1.prediction * direction_probability, 1e-7, 1.0 - 1e-7)
    return FitResult(
        prediction=prediction,
        best_iteration={"large": int(stage1.best_iteration), "direction": int(direction_iteration)},
    )


def _median_iteration(values: Sequence[int | dict[str, int]], minimum: int, maximum: int) -> int | dict[str, int]:
    if not values:
        return minimum
    if isinstance(values[0], dict):
        keys = sorted({key for value in values if isinstance(value, dict) for key in value})
        return {
            key: max(minimum, min(maximum, int(np.median([int(value[key]) for value in values if isinstance(value, dict) and key in value]))))
            for key in keys
        }
    return max(minimum, min(maximum, int(np.median([int(value) for value in values]))))


def task_paths(output: Path, recipe: Recipe, fold_id: int, seed: int) -> tuple[Path, Path]:
    directory = output / "task_cache" / recipe.name / f"seed_{seed}"
    return directory / f"fold_{fold_id}.json", directory / f"fold_{fold_id}.npz"


def task_is_valid(result_path: Path, prediction_path: Path, identity_hash: str) -> bool:
    if not result_path.exists() or not prediction_path.exists():
        return False
    try:
        payload = load_json(result_path)
    except Exception:
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("identity_hash") == identity_hash
        and payload_checksum_is_valid(payload)
        and payload.get("prediction_sha256") == sha256_file(prediction_path)
    )


def run_recipe_task(
    recipe: Recipe,
    fold: FoldSpec,
    seed: int,
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    result_path, prediction_path = task_paths(args.output, recipe, fold.fold_id, seed)
    identity = {
        "schema": "surge_model_zoo_task_v4",
        "dataset_sha256": bundle.dataset_sha256,
        "target_sha256": bundle.target_sha256,
        "matrix_cache_identity": bundle.cache_identity_hash,
        "recipe": recipe.to_dict(),
        "profile_features": list(profiles[recipe.profile]),
        "direction_features": list(profiles[recipe.direction_profile]) if recipe.direction_profile else None,
        "fold": fold.to_dict(),
        "seed": int(seed),
        "target_recall": float(args.target_recall),
        "threshold_recall_buffer": float(args.threshold_recall_buffer),
        "inner_windows": int(args.inner_windows),
        "inner_validation_days": int(args.inner_validation_days),
        "inner_purge_days": int(args.inner_purge_days),
        "inner_step_days": int(args.inner_step_days),
        "minimum_inner_train_days": int(args.minimum_inner_train_days),
        "code_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "common": sha256_file(Path(__file__).resolve().parent / "surge_model_zoo_common.py"),
        },
    }
    identity_hash = sha256_bytes(stable_json_bytes(identity))
    if args.resume and task_is_valid(result_path, prediction_path, identity_hash):
        return load_json(result_path)

    started = time.monotonic()
    matrix = np.load(bundle.matrix_path, mmap_mode="r")
    outer_train, outer_valid = fold_indices(bundle, fold)
    inner_windows = build_inner_windows(
        bundle.dates,
        outer_train,
        validation_days=args.inner_validation_days,
        purge_days=args.inner_purge_days,
        windows=args.inner_windows,
        step_days=args.inner_step_days,
        minimum_train_days=args.minimum_inner_train_days,
    )
    if not inner_windows:
        raise RuntimeError(f"{recipe.name} fold={fold.fold_id}: inner window를 만들 수 없습니다")
    device = resolve_model_device(recipe.family, preflight)
    oof_indices: list[np.ndarray] = []
    oof_raw: list[np.ndarray] = []
    best_iterations: list[int | dict[str, int]] = []
    for window_index, (inner_train, inner_valid) in enumerate(inner_windows):
        window_seed = deterministic_seed(seed, recipe.name, fold.fold_id, window_index)
        fit = fit_predict_recipe(
            recipe,
            matrix,
            bundle,
            profiles,
            inner_train,
            inner_valid,
            window_seed,
            args,
            device,
            fixed_iterations=None,
        )
        oof_indices.append(inner_valid)
        oof_raw.append(fit.prediction)
        best_iterations.append(fit.best_iteration)
        del fit
        gc.collect()
    oof_index = np.concatenate(oof_indices)
    oof_prediction = np.concatenate(oof_raw)
    order = np.argsort(oof_index, kind="mergesort")
    oof_index = oof_index[order]
    oof_prediction = oof_prediction[order]
    unique_mask = ~pd.Series(oof_index).duplicated(keep="last").to_numpy()
    oof_index = oof_index[unique_mask]
    oof_prediction = oof_prediction[unique_mask]
    y_oof = bundle.targets["surge_d3"][oof_index]
    dates_oof = bundle.dates[oof_index]
    calibrator = fit_calibrator(y_oof, oof_prediction, dates_oof, minimum_rows=args.minimum_calibration_rows)
    calibrated_oof = apply_calibrator(calibrator, oof_prediction)
    policy_target = min(0.995, args.target_recall + args.threshold_recall_buffer)
    policy = choose_threshold_policy(
        y_oof,
        calibrated_oof,
        dates_oof,
        policy_target,
        args.daily_fraction_grid,
        minimum_precision_lift=args.minimum_oof_precision_lift,
    )
    effective_iteration = _median_iteration(
        best_iterations,
        minimum=args.minimum_iterations,
        maximum=args.maximum_iterations,
    )
    final_seed = deterministic_seed(seed, recipe.name, fold.fold_id, "final")
    final_fit = fit_predict_recipe(
        recipe,
        matrix,
        bundle,
        profiles,
        outer_train,
        outer_valid,
        final_seed,
        args,
        device,
        fixed_iterations=effective_iteration,
    )
    raw_validation = final_fit.prediction
    calibrated_validation = apply_calibrator(calibrator, raw_validation)
    y_validation = bundle.targets["surge_d3"][outer_valid]
    dates_validation = bundle.dates[outer_valid]
    metrics = evaluate_prediction_metrics(
        y_validation,
        calibrated_validation,
        dates_validation,
        args.target_recall,
        top_fractions=args.top_fractions,
    )
    operating = evaluate_threshold_policy(policy, y_validation, calibrated_validation, dates_validation)
    for key, value in operating.items():
        metrics[f"transferred_{key}"] = value
    oof_operating = evaluate_threshold_policy(policy, y_oof, calibrated_oof, dates_oof)
    atomic_write_npz(
        prediction_path,
        validation_indices=outer_valid.astype(np.int64),
        target=y_validation.astype(np.uint8),
        dates=dates_validation.astype("datetime64[ns]").astype(np.int64),
        raw_prediction=raw_validation.astype(np.float32),
        calibrated_prediction=calibrated_validation.astype(np.float32),
    )
    payload = with_payload_checksum(
        {
            "status": "completed",
            "identity": identity,
            "identity_hash": identity_hash,
            "recipe": recipe.to_dict(),
            "fold_id": int(fold.fold_id),
            "fold_role": role_for_fold(fold.fold_id, roles),
            "seed": int(seed),
            "device": device,
            "train_rows": int(len(outer_train)),
            "validation_rows": int(len(outer_valid)),
            "inner_oof_rows": int(len(oof_index)),
            "best_iterations_by_window": best_iterations,
            "effective_iteration": effective_iteration,
            "calibrator": calibrator,
            "threshold_policy": policy.to_dict(),
            "inner_oof_operating": oof_operating,
            "metrics": metrics,
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "elapsed_seconds": time.monotonic() - started,
            "completed_at": utc_now(),
        }
    )
    atomic_write_json(result_path, payload)
    return payload


def flatten_task_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "recipe": payload["recipe"]["name"],
        "family": payload["recipe"]["family"],
        "profile": payload["recipe"]["profile"],
        "target_variant": payload["recipe"]["target_variant"],
        "train_policy": payload["recipe"]["train_policy"],
        "fold_id": int(payload["fold_id"]),
        "fold_role": payload["fold_role"],
        "seed": int(payload["seed"]),
        "device": payload["device"],
        "train_rows": int(payload["train_rows"]),
        "validation_rows": int(payload["validation_rows"]),
        "inner_oof_rows": int(payload["inner_oof_rows"]),
        "elapsed_seconds": float(payload["elapsed_seconds"]),
    }
    for key, value in payload.get("metrics", {}).items():
        if isinstance(value, (int, float)) or value is None:
            record[key] = value
    return record


def execute_tasks(
    recipes: Sequence[Recipe],
    folds: Sequence[FoldSpec],
    seeds: Sequence[int],
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    phase: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    total = len(recipes) * len(folds) * len(seeds)
    completed = 0
    for recipe in recipes:
        for seed in seeds:
            for fold in folds:
                completed += 1
                log(f"[{phase}] {completed}/{total} {recipe.name} seed={seed} fold={fold.fold_id}")
                try:
                    payload = run_recipe_task(
                        recipe,
                        fold,
                        seed,
                        bundle,
                        profiles,
                        roles,
                        args,
                        preflight,
                    )
                    records.append(flatten_task_result(payload))
                except Exception as exc:
                    failure = {
                        "recipe": recipe.name,
                        "family": recipe.family,
                        "fold_id": fold.fold_id,
                        "fold_role": role_for_fold(fold.fold_id, roles),
                        "seed": seed,
                        "status": "failed",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    atomic_write_json(
                        args.output / "failed_tasks" / recipe.name / f"seed_{seed}_fold_{fold.fold_id}.json",
                        failure,
                    )
                    if not args.allow_partial:
                        raise
                    log(f"실패 허용: {recipe.name} seed={seed} fold={fold.fold_id}: {exc}")
    frame = pd.DataFrame(records)
    atomic_write_csv(args.output / f"{phase}_seed_fold_metrics.csv", frame)
    return frame


def _percentile_rank(series: pd.Series, ascending: bool) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="average", pct=True, ascending=ascending).fillna(0.0)


def summarize_screen(metrics: pd.DataFrame, target_recall: float) -> pd.DataFrame:
    selection = metrics.loc[metrics["fold_role"].eq("selection")].copy()
    if selection.empty:
        raise ValueError("selection screening 결과가 없습니다")
    numeric_columns = [
        "pr_auc",
        "pr_auc_lift",
        "roc_auc",
        "daily_top_3pct_lift",
        "transferred_recall",
        "transferred_precision",
        "transferred_lift",
        "transferred_alert_rate",
        "oracle_r70_precision",
        "oracle_r70_lift",
        "oracle_r70_alert_rate",
    ]
    records: list[dict[str, Any]] = []
    for recipe, part in selection.groupby("recipe", sort=False):
        first = part.iloc[0]
        record: dict[str, Any] = {
            "recipe": recipe,
            "family": first["family"],
            "profile": first["profile"],
            "target_variant": first["target_variant"],
            "train_policy": first["train_policy"],
            "task_count": int(len(part)),
            "fold_count": int(part["fold_id"].nunique()),
            "seed_count": int(part["seed"].nunique()),
        }
        for column in numeric_columns:
            values = pd.to_numeric(part.get(column), errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            record[f"mean_{column}"] = float(np.mean(values)) if len(values) else float("nan")
            record[f"std_{column}"] = float(np.std(values, ddof=0)) if len(values) else float("nan")
            record[f"min_{column}"] = float(np.min(values)) if len(values) else float("nan")
        recalls = pd.to_numeric(part["transferred_recall"], errors="coerce").to_numpy(dtype=np.float64)
        recalls = recalls[np.isfinite(recalls)]
        record["recall70_pass_rate"] = float(np.mean(recalls >= target_recall)) if len(recalls) else 0.0
        record["recall70_mean_shortfall"] = float(np.mean(np.maximum(0.0, target_recall - recalls))) if len(recalls) else float("nan")
        records.append(record)
    summary = pd.DataFrame(records)
    summary["rank_recall_pass"] = _percentile_rank(summary["recall70_pass_rate"], ascending=True)
    summary["rank_worst_recall"] = _percentile_rank(summary["min_transferred_recall"], ascending=True)
    summary["rank_pr_lift"] = _percentile_rank(summary["mean_pr_auc_lift"], ascending=True)
    summary["rank_precision_lift"] = _percentile_rank(summary["mean_transferred_lift"], ascending=True)
    summary["rank_alert_efficiency"] = _percentile_rank(summary["mean_transferred_alert_rate"], ascending=False)
    summary["rank_top3"] = _percentile_rank(summary["mean_daily_top_3pct_lift"], ascending=True)
    summary["rank_seed_stability"] = _percentile_rank(summary["std_pr_auc_lift"], ascending=False)
    summary["selection_score"] = (
        0.25 * summary["rank_recall_pass"]
        + 0.15 * summary["rank_worst_recall"]
        + 0.20 * summary["rank_pr_lift"]
        + 0.20 * summary["rank_precision_lift"]
        + 0.10 * summary["rank_alert_efficiency"]
        + 0.05 * summary["rank_top3"]
        + 0.05 * summary["rank_seed_stability"]
    )
    return summary.sort_values(["selection_score", "mean_pr_auc_lift"], ascending=[False, False]).reset_index(drop=True)


def load_recipe_prediction(output: Path, recipe: str, fold_id: int, seed: int) -> dict[str, np.ndarray]:
    path = output / "task_cache" / recipe / f"seed_{seed}" / f"fold_{fold_id}.npz"
    arrays = np.load(path, allow_pickle=False)
    return {name: np.asarray(arrays[name]) for name in arrays.files}


def screen_prediction_correlation(
    output: Path,
    recipes: Sequence[str],
    folds: Sequence[FoldSpec],
    seeds: Sequence[int],
) -> pd.DataFrame:
    columns: dict[str, list[np.ndarray]] = defaultdict(list)
    reference_indices: list[np.ndarray] = []
    first_recipe = True
    for fold in folds:
        fold_reference = None
        for recipe in recipes:
            per_seed = []
            indices = None
            for seed in seeds:
                arrays = load_recipe_prediction(output, recipe, fold.fold_id, seed)
                per_seed.append(arrays["calibrated_prediction"].astype(np.float64))
                indices = arrays["validation_indices"].astype(np.int64)
            columns[recipe].append(np.mean(np.vstack(per_seed), axis=0))
            if fold_reference is None:
                fold_reference = indices
        if first_recipe and fold_reference is not None:
            reference_indices.append(fold_reference)
    frame = pd.DataFrame({name: np.concatenate(values) for name, values in columns.items()})
    return frame.corr(method="spearman")


def select_finalists(
    screen_summary: pd.DataFrame,
    correlation: pd.DataFrame,
    top_k: int,
    maximum_correlation: float,
) -> list[str]:
    selected: list[str] = []
    used_families: set[str] = set()
    for _, row in screen_summary.iterrows():
        if len(selected) >= top_k:
            break
        recipe = str(row["recipe"])
        family = str(row["family"])
        if family in used_families:
            continue
        if any(abs(float(correlation.loc[recipe, existing])) > maximum_correlation for existing in selected):
            continue
        selected.append(recipe)
        used_families.add(family)
    for _, row in screen_summary.iterrows():
        if len(selected) >= top_k:
            break
        recipe = str(row["recipe"])
        if recipe in selected:
            continue
        if any(abs(float(correlation.loc[recipe, existing])) > maximum_correlation for existing in selected):
            continue
        selected.append(recipe)
    if len(selected) < min(top_k, len(screen_summary)):
        for recipe in screen_summary["recipe"].astype(str):
            if recipe not in selected:
                selected.append(recipe)
            if len(selected) >= top_k:
                break
    return selected


def summarize_finalists(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    role_summary = aggregate_metric_records(metrics, ["recipe", "family", "profile", "target_variant", "fold_role"])
    seed_records: list[dict[str, Any]] = []
    for keys, part in metrics.groupby(["recipe", "fold_role", "seed"], sort=False):
        recipe, role, seed = keys
        seed_records.append(
            {
                "recipe": recipe,
                "fold_role": role,
                "seed": int(seed),
                "fold_count": int(part["fold_id"].nunique()),
                "pr_auc": float(pd.to_numeric(part["pr_auc"], errors="coerce").mean()),
                "pr_auc_lift": float(pd.to_numeric(part["pr_auc_lift"], errors="coerce").mean()),
                "roc_auc": float(pd.to_numeric(part["roc_auc"], errors="coerce").mean()),
                "transferred_recall": float(pd.to_numeric(part["transferred_recall"], errors="coerce").mean()),
                "transferred_precision": float(pd.to_numeric(part["transferred_precision"], errors="coerce").mean()),
                "transferred_alert_rate": float(pd.to_numeric(part["transferred_alert_rate"], errors="coerce").mean()),
            }
        )
    seed_frame = pd.DataFrame(seed_records)
    stability = aggregate_metric_records(seed_frame, ["recipe", "fold_role"])
    return role_summary, stability


def compute_seed_prediction_diversity(
    output: Path,
    recipe_names: Sequence[str],
    folds: Sequence[FoldSpec],
    seeds: Sequence[int],
) -> pd.DataFrame:
    """Measure whether nominally different seeds actually produce different predictions."""
    records: list[dict[str, Any]] = []
    unique_seeds = list(dict.fromkeys(int(value) for value in seeds))
    if len(unique_seeds) < 2:
        return pd.DataFrame(
            columns=[
                "recipe",
                "fold_id",
                "seed_a",
                "seed_b",
                "rows",
                "mean_abs_difference",
                "max_abs_difference",
                "pearson_correlation",
                "spearman_correlation",
                "identical_within_1e12",
            ]
        )
    for recipe in recipe_names:
        for fold in folds:
            cache: dict[int, np.ndarray] = {}
            for seed in unique_seeds:
                arrays = load_recipe_prediction(output, recipe, fold.fold_id, seed)
                cache[seed] = arrays["raw_prediction"].astype(np.float64)
            for left_index, seed_a in enumerate(unique_seeds):
                for seed_b in unique_seeds[left_index + 1 :]:
                    first = cache[seed_a]
                    second = cache[seed_b]
                    if len(first) != len(second):
                        raise ValueError(f"seed prediction 길이 불일치: {recipe} fold={fold.fold_id}")
                    valid = np.isfinite(first) & np.isfinite(second)
                    difference = np.abs(first[valid] - second[valid])
                    if int(valid.sum()) >= 2 and np.std(first[valid]) > 0 and np.std(second[valid]) > 0:
                        pearson = float(np.corrcoef(first[valid], second[valid])[0, 1])
                        spearman = float(pd.Series(first[valid]).corr(pd.Series(second[valid]), method="spearman"))
                    else:
                        pearson = float("nan")
                        spearman = float("nan")
                    maximum = float(np.max(difference)) if len(difference) else float("nan")
                    records.append(
                        {
                            "recipe": recipe,
                            "fold_id": int(fold.fold_id),
                            "seed_a": int(seed_a),
                            "seed_b": int(seed_b),
                            "rows": int(valid.sum()),
                            "mean_abs_difference": float(np.mean(difference)) if len(difference) else float("nan"),
                            "max_abs_difference": maximum,
                            "pearson_correlation": pearson,
                            "spearman_correlation": spearman,
                            "identical_within_1e12": bool(np.isfinite(maximum) and maximum <= 1e-12),
                        }
                    )
    return pd.DataFrame(records)


def _production_iteration_from_tasks(
    output: Path,
    recipe: Recipe,
    seed: int,
    folds: Sequence[FoldSpec],
    minimum: int,
    maximum: int,
) -> int | dict[str, int]:
    values: list[int | dict[str, int]] = []
    for fold in folds:
        result_path, _ = task_paths(output, recipe, fold.fold_id, seed)
        payload = load_json(result_path)
        value = payload.get("effective_iteration")
        if isinstance(value, dict):
            values.append({str(key): int(item) for key, item in value.items()})
        elif value is not None:
            values.append(int(value))
    if not values:
        raise RuntimeError(f"production iteration 근거가 없습니다: {recipe.name} seed={seed}")
    return _median_iteration(values, minimum=minimum, maximum=maximum)


def _resolved_production_params(
    recipe: Recipe,
    seed: int,
    args: argparse.Namespace,
    device: str,
    direction: bool = False,
) -> dict[str, Any]:
    family = "xgboost" if recipe.family == "two_stage_xgboost" else recipe.family
    source = recipe.direction_params if direction and recipe.direction_params else recipe.params
    if family == "lightgbm":
        resolved = dict(DEFAULT_LGB_PARAMS)
        resolved.update(source)
        resolved.update(
            {
                "num_threads": int(args.threads_per_model),
                "seed": int(seed),
                "feature_fraction_seed": int(seed),
                "bagging_seed": int(seed),
                "data_random_seed": int(seed),
            }
        )
        return resolved
    if family == "xgboost":
        resolved = dict(DEFAULT_XGB_PARAMS)
        resolved.update(source)
        resolved.update({"seed": int(seed), "nthread": int(args.xgboost_threads), "device": device})
        return resolved
    if family == "catboost":
        resolved = dict(DEFAULT_CAT_PARAMS)
        resolved.update(source)
        resolved.update(
            {
                "random_seed": int(seed),
                "thread_count": int(args.threads_per_model),
                "task_type": device,
            }
        )
        return resolved
    if family == "extra_trees":
        resolved = dict(DEFAULT_EXTRA_PARAMS)
        resolved.update(source)
        return resolved
    raise ValueError(f"production params 미지원 family: {recipe.family}")


def _iter_model_artifacts(value: Any) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if "format" in value and str(value.get("format")) != "constant_probability":
            records.append(value)
        for item in value.values():
            records.extend(_iter_model_artifacts(item))
    elif isinstance(value, list):
        for item in value:
            records.extend(_iter_model_artifacts(item))
    return records


def _production_registry_is_valid(path: Path, identity_hash: str, output_root: Path) -> bool:
    if not path.exists():
        return False
    try:
        payload = load_json(path)
    except Exception:
        return False
    if payload.get("schema") != "crashwatch_surge_production_model_registry_v4":
        return False
    if payload.get("identity_hash") != identity_hash:
        return False
    if not payload_checksum_is_valid(payload):
        return False
    for record in _iter_model_artifacts(payload.get("models", [])):
        raw_path = Path(str(record.get("path", "")))
        model_path = raw_path if raw_path.is_absolute() else output_root / raw_path
        if not model_path.exists():
            return False
        if record.get("sha256") != sha256_file(model_path):
            return False
        if int(record.get("bytes", -1)) != int(model_path.stat().st_size):
            return False
    return True


def train_production_models(
    finalists: Sequence[Recipe],
    folds: Sequence[FoldSpec],
    seeds: Sequence[int],
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    """Train serializable final models through the last target-valid date."""
    model_root = args.output / "production_models"
    model_root.mkdir(parents=True, exist_ok=True)
    registry_path = args.output / "PRODUCTION_MODEL_REGISTRY.json"
    iteration_map: dict[str, Any] = {}
    for recipe in finalists:
        iteration_map[recipe.name] = {}
        for seed in seeds:
            iteration_map[recipe.name][str(int(seed))] = _production_iteration_from_tasks(
                args.output,
                recipe,
                int(seed),
                folds,
                args.minimum_iterations,
                args.maximum_iterations,
            )
    identity = {
        "schema": "crashwatch_surge_production_training_identity_v4",
        "dataset_sha256": bundle.dataset_sha256,
        "target_sha256": bundle.target_sha256,
        "matrix_cache_identity": bundle.cache_identity_hash,
        "feature_universe_hash": hash_strings(bundle.features),
        "training_cutoff": str(pd.Timestamp(bundle.dates.max()).date()),
        "recipes": [recipe.to_dict() for recipe in finalists],
        "profiles": {
            recipe.profile: list(profiles[recipe.profile])
            for recipe in finalists
        },
        "direction_profiles": {
            recipe.direction_profile: list(profiles[recipe.direction_profile])
            for recipe in finalists
            if recipe.direction_profile
        },
        "seeds": [int(value) for value in seeds],
        "iterations": iteration_map,
        "code_sha256": {
            "runner": sha256_file(Path(__file__).resolve()),
            "common": sha256_file(Path(__file__).resolve().parent / "surge_model_zoo_common.py"),
            "deployment": sha256_file(Path(__file__).resolve().parent / "surge_model_zoo_deployment.py"),
        },
    }
    identity_hash = sha256_bytes(stable_json_bytes(identity))
    if args.resume and _production_registry_is_valid(registry_path, identity_hash, args.output):
        return load_json(registry_path)

    matrix = np.load(bundle.matrix_path, mmap_mode="r")
    all_indices = np.arange(len(bundle.frame), dtype=np.int64)
    recipe_records: list[dict[str, Any]] = []
    for recipe in finalists:
        primary_indices, direction_indices = _feature_indices(recipe, profiles, bundle)
        primary_features = [str(value) for value in profiles[recipe.profile]]
        for seed in seeds:
            train_indices = apply_training_window(
                all_indices,
                bundle.dates,
                recipe.train_policy,
                recipe.rolling_days,
            )
            iterations = iteration_map[recipe.name][str(int(seed))]
            device = resolve_model_device(recipe.family, preflight)
            directory = model_root / recipe.name / f"seed_{int(seed)}"
            directory.mkdir(parents=True, exist_ok=True)
            base_record: dict[str, Any] = {
                "recipe": recipe.name,
                "family": recipe.family,
                "profile": recipe.profile,
                "target_variant": recipe.target_variant,
                "train_policy": recipe.train_policy,
                "rolling_days": recipe.rolling_days,
                "time_decay_half_life_days": recipe.time_decay_half_life_days,
                "positive_weight_mode": recipe.positive_weight_mode,
                "seed": int(seed),
                "device": device,
                "features": primary_features,
                "feature_hash": hash_strings(primary_features),
                "train_rows": int(len(train_indices)),
                "train_date_min": str(pd.Timestamp(bundle.dates[train_indices].min()).date()),
                "train_date_max": str(pd.Timestamp(bundle.dates[train_indices].max()).date()),
                "effective_iteration": iterations,
            }
            if recipe.family != "two_stage_xgboost":
                target_name = recipe.target_variant
                target = bundle.targets[target_name]
                weights = build_sample_weights(
                    target,
                    bundle.dates,
                    train_indices,
                    recipe.positive_weight_mode,
                    recipe.time_decay_half_life_days,
                )
                x_train = np.asarray(matrix[np.ix_(train_indices, primary_indices)], dtype=np.float32)
                y_train = target[train_indices]
                resolved = _resolved_production_params(recipe, int(seed), args, device)
                if recipe.family == "lightgbm":
                    model_info = save_lightgbm_model(
                        x_train,
                        y_train,
                        weights,
                        resolved,
                        int(iterations),
                        directory / "model.txt",
                    )
                elif recipe.family == "xgboost":
                    model_info = save_xgboost_model(
                        x_train,
                        y_train,
                        weights,
                        resolved,
                        int(iterations),
                        directory / "model.json",
                    )
                elif recipe.family == "catboost":
                    model_info = save_catboost_model(
                        x_train,
                        y_train,
                        weights,
                        resolved,
                        int(iterations),
                        directory / "model.cbm",
                    )
                elif recipe.family == "extra_trees":
                    model_info = save_extra_trees_model(
                        x_train,
                        y_train,
                        weights,
                        resolved,
                        int(seed),
                        int(args.threads_per_model),
                        directory / "model.joblib",
                    )
                else:
                    raise ValueError(f"production model 미지원 family: {recipe.family}")
                base_record.update(
                    {
                        "model_type": "single",
                        "model": model_info,
                        "target_positive_rows": int(np.sum(y_train == 1)),
                        "resolved_params": resolved,
                    }
                )
                recipe_records.append(base_record)
                del x_train, y_train, weights
                gc.collect()
                continue

            if not isinstance(iterations, dict):
                iterations = {"large": int(iterations), "direction": int(iterations)}
            large_target = bundle.targets["large_move"]
            large_weights = build_sample_weights(
                large_target,
                bundle.dates,
                train_indices,
                recipe.positive_weight_mode,
                recipe.time_decay_half_life_days,
            )
            x_large = np.asarray(matrix[np.ix_(train_indices, primary_indices)], dtype=np.float32)
            large_params = _resolved_production_params(recipe, int(seed), args, device, direction=False)
            large_model = save_xgboost_model(
                x_large,
                large_target[train_indices],
                large_weights,
                large_params,
                int(iterations.get("large", args.minimum_iterations)),
                directory / "large_move_model.json",
            )
            if direction_indices is None:
                direction_indices = primary_indices
            direction_features = [str(value) for value in profiles[recipe.direction_profile or recipe.profile]]
            direction_valid = bundle.targets["direction_valid"].astype(bool)
            direction_train = train_indices[direction_valid[train_indices]]
            direction_target = bundle.targets["direction_up"]
            if len(direction_train) < 200 or len(np.unique(direction_target[direction_train])) < 2:
                direction_model: dict[str, Any] = {
                    "format": "constant_probability",
                    "probability": float(np.mean(direction_target[direction_train])) if len(direction_train) else 0.5,
                    "iterations": 0,
                }
                direction_params: dict[str, Any] = {}
            else:
                direction_weights = build_sample_weights(
                    direction_target,
                    bundle.dates,
                    direction_train,
                    "sqrt_balance",
                    recipe.time_decay_half_life_days,
                )
                x_direction = np.asarray(matrix[np.ix_(direction_train, direction_indices)], dtype=np.float32)
                direction_params = _resolved_production_params(
                    recipe,
                    deterministic_seed(seed, "direction"),
                    args,
                    device,
                    direction=True,
                )
                direction_model = save_xgboost_model(
                    x_direction,
                    direction_target[direction_train],
                    direction_weights,
                    direction_params,
                    int(iterations.get("direction", args.minimum_iterations)),
                    directory / "direction_model.json",
                )
                del x_direction, direction_weights
            base_record.update(
                {
                    "model_type": "two_stage",
                    "direction_profile": recipe.direction_profile,
                    "direction_features": direction_features,
                    "direction_feature_hash": hash_strings(direction_features),
                    "large_move_model": large_model,
                    "direction_model": direction_model,
                    "large_move_positive_rows": int(np.sum(large_target[train_indices] == 1)),
                    "direction_train_rows": int(len(direction_train)),
                    "large_move_resolved_params": large_params,
                    "direction_resolved_params": direction_params,
                }
            )
            recipe_records.append(base_record)
            del x_large, large_weights
            gc.collect()
    payload = {
        "schema": "crashwatch_surge_production_model_registry_v4",
        "created_at": utc_now(),
        "identity": identity,
        "identity_hash": identity_hash,
        "dataset_sha256": bundle.dataset_sha256,
        "target_sha256": bundle.target_sha256,
        "training_cutoff": str(pd.Timestamp(bundle.dates.max()).date()),
        "feature_universe_hash": hash_strings(bundle.features),
        "recipes": [recipe.name for recipe in finalists],
        "seeds": [int(value) for value in seeds],
        "models": recipe_records,
    }
    portable = relativize_registry_paths(payload, args.output)
    portable = with_payload_checksum(portable)
    atomic_write_json(registry_path, portable)
    return portable


def build_recipe_fold_predictions(
    output: Path,
    recipe_names: Sequence[str],
    folds: Sequence[FoldSpec],
    seeds: Sequence[int],
    bundle: DataBundle,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold in folds:
        _, validation_indices = fold_indices(bundle, fold)
        frame = pd.DataFrame(
            {
                "validation_index": validation_indices,
                "fold_id": int(fold.fold_id),
                "date": bundle.dates[validation_indices],
                "target": bundle.targets["surge_d3"][validation_indices],
            }
        )
        for recipe in recipe_names:
            seed_predictions: list[np.ndarray] = []
            for seed in seeds:
                arrays = load_recipe_prediction(output, recipe, fold.fold_id, seed)
                indices = arrays["validation_indices"].astype(np.int64)
                if not np.array_equal(indices, validation_indices):
                    raise ValueError(f"prediction index mismatch: {recipe} fold={fold.fold_id} seed={seed}")
                seed_predictions.append(arrays["raw_prediction"].astype(np.float64))
            frame[recipe] = np.mean(np.vstack(seed_predictions), axis=0)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def _safe_logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp / np.sum(exp)


def fit_ensemble_spec(method: str, prediction_frame: pd.DataFrame, recipe_names: Sequence[str]) -> dict[str, Any]:
    x = prediction_frame[list(recipe_names)].to_numpy(dtype=np.float64)
    y = prediction_frame["target"].to_numpy(dtype=np.int8)
    dates = prediction_frame["date"].to_numpy(dtype="datetime64[ns]")
    medians = np.nanmedian(x, axis=0)
    missing = ~np.isfinite(x)
    if missing.any():
        x[missing] = np.take(medians, np.where(missing)[1])
    if method == "equal_probability":
        return {"method": method, "recipes": list(recipe_names), "medians": medians.tolist()}
    if method == "equal_date_rank":
        return {"method": method, "recipes": list(recipe_names), "medians": medians.tolist()}
    logits = _safe_logit(x)
    if method == "nonnegative_logit":
        from scipy.optimize import minimize

        initial = np.zeros(len(recipe_names), dtype=np.float64)

        def objective(theta: np.ndarray) -> float:
            weights = _softmax(theta)
            score = 1.0 / (1.0 + np.exp(-np.clip(logits @ weights, -35.0, 35.0)))
            return float(-np.mean(y * np.log(score + 1e-9) + (1 - y) * np.log(1 - score + 1e-9)))

        result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": 500})
        weights = _softmax(np.asarray(result.x if result.success else initial, dtype=np.float64))
        return {
            "method": method,
            "recipes": list(recipe_names),
            "medians": medians.tolist(),
            "weights": weights.tolist(),
            "optimization_success": bool(result.success),
            "optimization_message": str(result.message),
        }
    if method == "logistic_stacker":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(C=0.5, solver="lbfgs", max_iter=2000, class_weight=None)
        model.fit(logits, y)
        return {
            "method": method,
            "recipes": list(recipe_names),
            "medians": medians.tolist(),
            "coefficients": model.coef_[0].astype(float).tolist(),
            "intercept": float(model.intercept_[0]),
        }
    raise ValueError(f"지원하지 않는 ensemble method: {method}")


def apply_ensemble_spec(spec: Mapping[str, Any], prediction_frame: pd.DataFrame) -> np.ndarray:
    recipes = [str(value) for value in spec["recipes"]]
    x = prediction_frame[recipes].to_numpy(dtype=np.float64)
    medians = np.asarray(spec["medians"], dtype=np.float64)
    missing = ~np.isfinite(x)
    if missing.any():
        x[missing] = np.take(medians, np.where(missing)[1])
    method = str(spec["method"])
    if method == "equal_probability":
        return np.mean(x, axis=1)
    if method == "equal_date_rank":
        dates = prediction_frame["date"].to_numpy(dtype="datetime64[ns]")
        ranked = np.column_stack([datewise_rank_normalize(x[:, column], dates) for column in range(x.shape[1])])
        return np.nanmean(ranked, axis=1)
    logits = _safe_logit(x)
    if method == "nonnegative_logit":
        weights = np.asarray(spec["weights"], dtype=np.float64)
        return 1.0 / (1.0 + np.exp(-np.clip(logits @ weights, -35.0, 35.0)))
    if method == "logistic_stacker":
        coefficients = np.asarray(spec["coefficients"], dtype=np.float64)
        values = logits @ coefficients + float(spec["intercept"])
        return 1.0 / (1.0 + np.exp(-np.clip(values, -35.0, 35.0)))
    raise ValueError(f"지원하지 않는 ensemble method: {method}")


def crossfit_ensemble_methods(
    prediction_frame: pd.DataFrame,
    recipe_names: Sequence[str],
    selection_fold_ids: Sequence[int],
    methods: Sequence[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    final_specs: dict[str, dict[str, Any]] = {}
    selection = prediction_frame.loc[prediction_frame["fold_id"].isin(selection_fold_ids)].copy()
    for method in methods:
        for held_fold in selection_fold_ids:
            train = selection.loc[selection["fold_id"].ne(int(held_fold))].copy()
            valid = selection.loc[selection["fold_id"].eq(int(held_fold))].copy()
            spec = fit_ensemble_spec(method, train, recipe_names)
            train_raw = apply_ensemble_spec(spec, train)
            valid_raw = apply_ensemble_spec(spec, valid)
            calibrator = fit_calibrator(
                train["target"].to_numpy(dtype=np.int8),
                train_raw,
                train["date"].to_numpy(dtype="datetime64[ns]"),
                minimum_rows=args.minimum_calibration_rows,
            )
            train_score = apply_calibrator(calibrator, train_raw)
            valid_score = apply_calibrator(calibrator, valid_raw)
            policy = choose_threshold_policy(
                train["target"].to_numpy(dtype=np.int8),
                train_score,
                train["date"].to_numpy(dtype="datetime64[ns]"),
                min(0.995, args.target_recall + args.threshold_recall_buffer),
                args.daily_fraction_grid,
                minimum_precision_lift=args.minimum_oof_precision_lift,
            )
            metrics = evaluate_prediction_metrics(
                valid["target"].to_numpy(dtype=np.int8),
                valid_score,
                valid["date"].to_numpy(dtype="datetime64[ns]"),
                args.target_recall,
                top_fractions=args.top_fractions,
            )
            operating = evaluate_threshold_policy(
                policy,
                valid["target"].to_numpy(dtype=np.int8),
                valid_score,
                valid["date"].to_numpy(dtype="datetime64[ns]"),
            )
            record = {"method": method, "held_fold": int(held_fold), **metrics}
            record.update({f"transferred_{key}": value for key, value in operating.items()})
            records.append(record)
        final_specs[method] = fit_ensemble_spec(method, selection, recipe_names)
    return pd.DataFrame(records), final_specs


def choose_ensemble_method(crossfit: pd.DataFrame, target_recall: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for method, part in crossfit.groupby("method", sort=False):
        recalls = pd.to_numeric(part["transferred_recall"], errors="coerce")
        record = {
            "method": method,
            "fold_count": int(len(part)),
            "recall70_pass_rate": float((recalls >= target_recall).mean()),
            "mean_recall": float(recalls.mean()),
            "worst_recall": float(recalls.min()),
            "mean_precision": float(pd.to_numeric(part["transferred_precision"], errors="coerce").mean()),
            "mean_precision_lift": float(pd.to_numeric(part["transferred_lift"], errors="coerce").mean()),
            "mean_alert_rate": float(pd.to_numeric(part["transferred_alert_rate"], errors="coerce").mean()),
            "mean_pr_auc_lift": float(pd.to_numeric(part["pr_auc_lift"], errors="coerce").mean()),
            "mean_roc_auc": float(pd.to_numeric(part["roc_auc"], errors="coerce").mean()),
            "mean_daily_top3_lift": float(pd.to_numeric(part["daily_top_3pct_lift"], errors="coerce").mean()),
        }
        records.append(record)
    summary = pd.DataFrame(records)
    summary["rank_pass"] = _percentile_rank(summary["recall70_pass_rate"], ascending=True)
    summary["rank_worst"] = _percentile_rank(summary["worst_recall"], ascending=True)
    summary["rank_precision_lift"] = _percentile_rank(summary["mean_precision_lift"], ascending=True)
    summary["rank_pr_lift"] = _percentile_rank(summary["mean_pr_auc_lift"], ascending=True)
    summary["rank_alert"] = _percentile_rank(summary["mean_alert_rate"], ascending=False)
    summary["ensemble_selection_score"] = (
        0.30 * summary["rank_pass"]
        + 0.20 * summary["rank_worst"]
        + 0.20 * summary["rank_precision_lift"]
        + 0.20 * summary["rank_pr_lift"]
        + 0.10 * summary["rank_alert"]
    )
    return summary.sort_values("ensemble_selection_score", ascending=False).reset_index(drop=True)


def evaluate_frozen_ensemble(
    prediction_frame: pd.DataFrame,
    spec: Mapping[str, Any],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], ThresholdPolicy, dict[str, Any]]:
    selection_ids = [int(value) for value in roles["selection"]]
    selection = prediction_frame.loc[prediction_frame["fold_id"].isin(selection_ids)].copy()
    selection_raw = apply_ensemble_spec(spec, selection)
    calibrator = fit_calibrator(
        selection["target"].to_numpy(dtype=np.int8),
        selection_raw,
        selection["date"].to_numpy(dtype="datetime64[ns]"),
        minimum_rows=args.minimum_calibration_rows,
    )
    selection_score = apply_calibrator(calibrator, selection_raw)
    policy = choose_threshold_policy(
        selection["target"].to_numpy(dtype=np.int8),
        selection_score,
        selection["date"].to_numpy(dtype="datetime64[ns]"),
        min(0.995, args.target_recall + args.threshold_recall_buffer),
        args.daily_fraction_grid,
        minimum_precision_lift=args.minimum_oof_precision_lift,
    )
    rows: list[dict[str, Any]] = []
    scored_parts: list[pd.DataFrame] = []
    for fold_id, part in prediction_frame.groupby("fold_id", sort=True):
        raw = apply_ensemble_spec(spec, part)
        score = apply_calibrator(calibrator, raw)
        metrics = evaluate_prediction_metrics(
            part["target"].to_numpy(dtype=np.int8),
            score,
            part["date"].to_numpy(dtype="datetime64[ns]"),
            args.target_recall,
            top_fractions=args.top_fractions,
        )
        operating = evaluate_threshold_policy(
            policy,
            part["target"].to_numpy(dtype=np.int8),
            score,
            part["date"].to_numpy(dtype="datetime64[ns]"),
        )
        row = {
            "fold_id": int(fold_id),
            "fold_role": role_for_fold(int(fold_id), roles),
            **metrics,
            **{f"transferred_{key}": value for key, value in operating.items()},
        }
        rows.append(row)
        scored = part[["validation_index", "fold_id", "date", "target"]].copy()
        scored["ensemble_raw"] = raw
        scored["ensemble_score"] = score
        scored["alert"] = apply_threshold_policy(policy, score, scored["date"].to_numpy(dtype="datetime64[ns]"))
        scored_parts.append(scored)
    return pd.DataFrame(rows), calibrator, policy, {"scored": pd.concat(scored_parts, ignore_index=True)}


def build_recall_alert_tradeoff(
    scored: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build diagnostic alert-budget curves without using them to refit the frozen policy."""
    curve_records: list[dict[str, Any]] = []
    summary_records: list[dict[str, Any]] = []
    for role, fold_ids in roles.items():
        part = scored.loc[scored["fold_id"].isin([int(value) for value in fold_ids])].copy()
        if part.empty:
            continue
        target = part["target"].to_numpy(dtype=np.int8)
        score = part["ensemble_score"].to_numpy(dtype=np.float64)
        dates = part["date"].to_numpy(dtype="datetime64[ns]")
        positive_rate = float(np.mean(target == 1)) if len(target) else float("nan")
        frozen_metrics = _selected_metrics(target, part["alert"].to_numpy(dtype=bool))
        threshold, oracle_metrics = threshold_for_target_recall(target, score, args.target_recall)
        minimum_daily_fraction: float | None = None
        minimum_daily_metrics: dict[str, Any] | None = None
        for fraction in sorted({float(value) for value in args.daily_fraction_grid if 0 < float(value) <= 1.0}):
            selected = daily_fraction_selection(score, dates, fraction)
            metrics = _selected_metrics(target, selected)
            curve_records.append(
                {
                    "fold_role": role,
                    "policy_kind": "daily_fraction_diagnostic",
                    "configured_fraction": float(fraction),
                    **metrics,
                    "target_recall": float(args.target_recall),
                    "recall_pass": bool(
                        np.isfinite(metrics["recall"]) and float(metrics["recall"]) >= args.target_recall
                    ),
                }
            )
            if (
                minimum_daily_fraction is None
                and np.isfinite(metrics["recall"])
                and float(metrics["recall"]) >= args.target_recall
            ):
                minimum_daily_fraction = float(fraction)
                minimum_daily_metrics = metrics
        summary_records.append(
            {
                "fold_role": role,
                "rows": int(len(part)),
                "positives": int(np.sum(target == 1)),
                "positive_rate": positive_rate,
                "target_recall": float(args.target_recall),
                "perfect_ranking_minimum_alert_rate": theoretical_minimum_alert_rate(
                    positive_rate, args.target_recall
                ),
                "diagnostic_global_threshold": float(threshold),
                "diagnostic_global_alert_rate_at_target_recall": float(oracle_metrics["alert_rate"]),
                "diagnostic_global_precision_at_target_recall": float(oracle_metrics["precision"]),
                "diagnostic_global_lift_at_target_recall": float(oracle_metrics["lift"]),
                "minimum_daily_fraction_grid_value_for_target_recall": minimum_daily_fraction,
                "minimum_daily_fraction_actual_alert_rate": (
                    float(minimum_daily_metrics["alert_rate"]) if minimum_daily_metrics else float("nan")
                ),
                "minimum_daily_fraction_precision": (
                    float(minimum_daily_metrics["precision"]) if minimum_daily_metrics else float("nan")
                ),
                "minimum_daily_fraction_lift": (
                    float(minimum_daily_metrics["lift"]) if minimum_daily_metrics else float("nan")
                ),
                "frozen_policy_recall": float(frozen_metrics["recall"]),
                "frozen_policy_precision": float(frozen_metrics["precision"]),
                "frozen_policy_lift": float(frozen_metrics["lift"]),
                "frozen_policy_alert_rate": float(frozen_metrics["alert_rate"]),
                "daily_top3_theoretical_maximum_recall": maximum_recall_at_alert_rate(
                    positive_rate,
                    _selected_metrics(target, daily_fraction_selection(score, dates, 0.03))["alert_rate"],
                ),
                "diagnostic_only_warning": (
                    "role 내부 target을 보고 계산한 oracle/curve이므로 모델 선택이나 배포 threshold에 사용 금지"
                ),
            }
        )
    return pd.DataFrame(curve_records), pd.DataFrame(summary_records)


def aggregate_role_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    return aggregate_metric_records(fold_metrics, ["fold_role"])


def scope_metrics(
    scored: pd.DataFrame,
    bundle: DataBundle,
    roles: Mapping[str, Sequence[int]],
    scope_columns: Sequence[str],
    minimum_rows: int,
) -> pd.DataFrame:
    metadata = bundle.frame.set_index("row_index", drop=False)
    records: list[dict[str, Any]] = []
    for role, fold_ids in roles.items():
        role_part = scored.loc[scored["fold_id"].isin([int(value) for value in fold_ids])].copy()
        if role_part.empty:
            continue
        role_part = role_part.join(metadata, on="validation_index", rsuffix="__meta")
        for scope in scope_columns:
            if scope not in role_part.columns:
                continue
            for value, part in role_part.groupby(scope, dropna=False, sort=False):
                if len(part) < minimum_rows:
                    continue
                metrics = _selected_metrics(
                    part["target"].to_numpy(dtype=np.int8),
                    part["alert"].to_numpy(dtype=bool),
                )
                ranking = safe_binary_metrics(
                    part["target"].to_numpy(dtype=np.int8),
                    part["ensemble_score"].to_numpy(dtype=np.float64),
                )
                records.append(
                    {
                        "fold_role": role,
                        "scope_type": scope,
                        "scope_value": str(value),
                        **metrics,
                        "pr_auc": ranking["pr_auc"],
                        "pr_auc_lift": ranking["pr_auc_lift"],
                        "roc_auc": ranking["roc_auc"],
                    }
                )
    return pd.DataFrame(records)


def build_performance_gap_report(
    role_summary: pd.DataFrame,
    crossfit_summary: pd.DataFrame,
    seed_stability: pd.DataFrame,
    seed_prediction_diversity: pd.DataFrame,
    scope_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    gaps: list[dict[str, Any]] = []
    roles_payload: dict[str, Any] = {}
    for _, row in role_summary.iterrows():
        role = str(row["fold_role"])
        positive_rate = float(row.get("mean_positive_rate", float("nan")))
        recall = float(row.get("mean_transferred_recall", float("nan")))
        precision = float(row.get("mean_transferred_precision", float("nan")))
        lift = float(row.get("mean_transferred_lift", float("nan")))
        alert_rate = float(row.get("mean_transferred_alert_rate", float("nan")))
        pr_auc = float(row.get("mean_pr_auc", float("nan")))
        pr_auc_lift = float(row.get("mean_pr_auc_lift", float("nan")))
        roc_auc = float(row.get("mean_roc_auc", float("nan")))
        top3_lift = float(row.get("mean_daily_top_3pct_lift", float("nan")))
        top3_alert_rate = float(row.get("mean_daily_top_3pct_alert_rate", float("nan")))
        minimum_alert = theoretical_minimum_alert_rate(positive_rate, args.target_recall)
        top3_maximum_recall = maximum_recall_at_alert_rate(positive_rate, top3_alert_rate)
        alert_efficiency_ratio = (
            float(alert_rate / minimum_alert)
            if np.isfinite(alert_rate) and np.isfinite(minimum_alert) and minimum_alert > 0
            else float("nan")
        )
        roles_payload[role] = {
            "positive_rate": positive_rate,
            "target_recall": args.target_recall,
            "theoretical_minimum_alert_rate": minimum_alert,
            "observed_recall": recall,
            "observed_precision": precision,
            "observed_precision_lift": lift,
            "observed_alert_rate": alert_rate,
            "alert_efficiency_ratio_to_perfect_ranking": alert_efficiency_ratio,
            "pr_auc": pr_auc,
            "pr_auc_lift": pr_auc_lift,
            "roc_auc": roc_auc,
            "daily_top3_lift": top3_lift,
            "daily_top3_alert_rate": top3_alert_rate,
            "daily_top3_theoretical_maximum_recall": top3_maximum_recall,
            "recall_pass": bool(np.isfinite(recall) and recall >= args.target_recall),
            "alert_rate_pass": bool(np.isfinite(alert_rate) and alert_rate <= args.max_alert_rate),
            "lift_pass": bool(np.isfinite(lift) and lift >= args.minimum_holdout_precision_lift),
        }
        if not np.isfinite(recall) or recall < args.target_recall:
            gaps.append({"severity": "CRITICAL", "area": f"{role}_recall", "message": f"{role} 재현율이 70% 목표 미달: {recall:.4f}"})
        if np.isfinite(alert_rate) and alert_rate > args.max_alert_rate:
            gaps.append({"severity": "HIGH", "area": f"{role}_alert_burden", "message": f"{role} alert rate가 상한 초과: {alert_rate:.4f}"})
        if np.isfinite(lift) and lift < args.minimum_holdout_precision_lift:
            gaps.append({"severity": "HIGH", "area": f"{role}_precision_lift", "message": f"{role} precision lift가 낮음: {lift:.4f}"})
        if role in {"confirmation", "recent_audit"} and np.isfinite(roc_auc) and roc_auc < 0.65:
            gaps.append({"severity": "MEDIUM", "area": f"{role}_ranking_discrimination", "message": f"{role} ROC-AUC가 0.65 미만: {roc_auc:.4f}"})
        if role in {"confirmation", "recent_audit"} and np.isfinite(pr_auc_lift) and pr_auc_lift < 1.50:
            gaps.append({"severity": "MEDIUM", "area": f"{role}_pr_auc_lift", "message": f"{role} PR-AUC/base-rate lift가 1.50 미만: {pr_auc_lift:.4f}"})
        if np.isfinite(alert_efficiency_ratio) and alert_efficiency_ratio > 4.0:
            gaps.append({"severity": "MEDIUM", "area": f"{role}_alert_efficiency", "message": f"{role} alert rate가 완전순위 이론 최소치의 {alert_efficiency_ratio:.2f}배"})

    top3_impossible_roles = [
        role
        for role, values in roles_payload.items()
        if np.isfinite(values.get("daily_top3_theoretical_maximum_recall", np.nan))
        and float(values["daily_top3_theoretical_maximum_recall"]) < args.target_recall
    ]
    if top3_impossible_roles:
        gaps.append(
            {
                "severity": "INFO",
                "area": "top3_kpi_mismatch",
                "message": "Daily Top-3% 경보 예산은 완전한 순위에서도 recall 70%가 불가능한 구간: "
                + ", ".join(top3_impossible_roles),
            }
        )

    if not crossfit_summary.empty:
        best = crossfit_summary.iloc[0]
        if float(best["recall70_pass_rate"]) < 1.0:
            gaps.append({"severity": "HIGH", "area": "selection_threshold_transfer", "message": "selection cross-fit 모든 fold에서 70% recall을 유지하지 못함"})
    if not seed_stability.empty:
        seed_std_columns = [column for column in seed_stability.columns if column.startswith("std_pr_auc_lift")]
        if seed_std_columns:
            maximum = float(pd.to_numeric(seed_stability[seed_std_columns[0]], errors="coerce").max())
            if maximum > 0.03:
                gaps.append({"severity": "MEDIUM", "area": "seed_variance", "message": f"일부 후보의 seed PR-lift 표준편차가 큼: {maximum:.4f}"})
    deterministic_recipes: list[str] = []
    if not seed_prediction_diversity.empty:
        grouped = seed_prediction_diversity.groupby("recipe", sort=False)["identical_within_1e12"].all()
        deterministic_recipes = grouped[grouped].index.astype(str).tolist()
        if deterministic_recipes:
            gaps.append(
                {
                    "severity": "HIGH",
                    "area": "ineffective_seed_test",
                    "message": "서로 다른 seed 예측이 완전히 동일한 recipe: " + ", ".join(deterministic_recipes),
                }
            )

    weak_scopes: list[dict[str, Any]] = []
    if not scope_frame.empty:
        eligible = scope_frame.loc[scope_frame["positives"].ge(args.minimum_scope_positives)].copy()
        eligible.sort_values(["recall", "lift"], ascending=[True, True], inplace=True)
        weak_scopes = eligible.head(20).to_dict(orient="records")
        for record in weak_scopes[:5]:
            if np.isfinite(record.get("recall", np.nan)) and float(record["recall"]) < args.target_recall - 0.15:
                gaps.append(
                    {
                        "severity": "MEDIUM",
                        "area": "weak_scope",
                        "message": f"{record['scope_type']}={record['scope_value']} recall={record['recall']:.4f}",
                    }
                )

    confirmation = roles_payload.get("confirmation", {})
    recent = roles_payload.get("recent_audit", {})
    development_pass = bool(
        confirmation.get("recall_pass", False)
        and recent.get("recall_pass", False)
        and confirmation.get("alert_rate_pass", False)
        and recent.get("alert_rate_pass", False)
        and confirmation.get("lift_pass", False)
        and recent.get("lift_pass", False)
    )
    report = {
        "schema": "crashwatch_surge_performance_gap_v4",
        "created_at": utc_now(),
        "target": "3거래일 내 +5% 상승",
        "target_recall": args.target_recall,
        "success_definition": {
            "recall": f">= {args.target_recall}",
            "precision_lift": f">= {args.minimum_holdout_precision_lift}",
            "alert_rate": f"<= {args.max_alert_rate}",
            "warning": "recall만 보면 모든 행을 alert해 100%를 만들 수 있으므로 세 조건을 함께 사용",
        },
        "roles": roles_payload,
        "development_audit_pass": development_pass,
        "gaps": gaps,
        "weak_scopes": weak_scopes,
        "deterministic_seed_recipes": deterministic_recipes,
        "holdout_warning": "기존 confirmation/recent를 반복 검토했으므로 최종 성능 확정에는 2026-06-22 이후 신규 holdout이 필요함",
    }

    lines = [
        "# Surge Model Zoo V4 성능 부족 구간 보고서",
        "",
        f"목표: 3거래일 이내 +5% 상승 이벤트 recall {args.target_recall:.0%} 이상.",
        "",
        "Recall 단독 목표는 모든 행을 양성으로 예측하면 달성되므로, precision lift와 alert rate를 함께 판정한다.",
        "",
        "## 역할별 결과",
        "",
        "| 구간 | 양성률 | PR-AUC lift | ROC-AUC | 이론 최소 alert | recall | precision lift | alert rate | Top-3% 최대 recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role, values in roles_payload.items():
        lines.append(
            f"| {role} | {values['positive_rate']:.4f} | {values['pr_auc_lift']:.4f} | "
            f"{values['roc_auc']:.4f} | {values['theoretical_minimum_alert_rate']:.4f} | "
            f"{values['observed_recall']:.4f} | {values['observed_precision_lift']:.4f} | "
            f"{values['observed_alert_rate']:.4f} | {values['daily_top3_theoretical_maximum_recall']:.4f} |"
        )
    lines.extend(["", "## 남은 병목", ""])
    if gaps:
        for gap in gaps:
            lines.append(f"- [{gap['severity']}] {gap['area']}: {gap['message']}")
    else:
        lines.append("- 설정된 development gate 기준의 명시적 병목은 발견되지 않았다.")
    lines.extend(
        [
            "",
            "## 판정",
            "",
            f"Development audit pass: **{development_pass}**",
            "",
            "이 판정은 기존 개발 구간에 대한 것이다. 최종 배포 성능은 2026-06-22 이후 신규 데이터에서 한 번 더 검증해야 한다.",
        ]
    )
    return report, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch Surge 3D5 고성능 모델 zoo·다중 seed·OOF ensemble V4")
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--recipes", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--date-column", default=DATE_COLUMN)
    parser.add_argument("--ticker-column", default=TICKER_COLUMN)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--target-valid-column", default=TARGET_VALID_COLUMN)
    parser.add_argument("--return-column", default="t_price_ret_1")
    parser.add_argument("--scope-columns", default="ticker,industry_name,market,bucket")
    parser.add_argument("--selection-folds", default="0,1,2,3,4")
    parser.add_argument("--confirmation-folds", default="5,6")
    parser.add_argument("--recent-folds", default="7")
    parser.add_argument("--screen-seeds", default="17,29")
    parser.add_argument("--final-seeds", default="17,29,41,73,101")
    parser.add_argument("--families", default="lightgbm,xgboost,catboost,extra_trees,two_stage_xgboost")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--finalist-count", type=int, default=8)
    parser.add_argument("--maximum-finalist-correlation", type=float, default=0.995)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-production-training", action="store_true")
    parser.add_argument("--threads-per-model", type=int, default=4)
    parser.add_argument("--xgboost-threads", type=int, default=4)
    parser.add_argument("--target-recall", type=float, default=0.70)
    parser.add_argument("--threshold-recall-buffer", type=float, default=0.05)
    parser.add_argument("--minimum-oof-precision-lift", type=float, default=1.05)
    parser.add_argument("--minimum-holdout-precision-lift", type=float, default=1.10)
    parser.add_argument("--max-alert-rate", type=float, default=0.65)
    parser.add_argument("--daily-fraction-grid", default="0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.225,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.80,0.90,1.0")
    parser.add_argument("--top-fractions", default="0.03,0.05,0.10,0.20,0.30,0.50")
    parser.add_argument("--inner-windows", type=int, default=3)
    parser.add_argument("--inner-validation-days", type=int, default=60)
    parser.add_argument("--inner-purge-days", type=int, default=20)
    parser.add_argument("--inner-step-days", type=int, default=120)
    parser.add_argument("--minimum-inner-train-days", type=int, default=500)
    parser.add_argument("--minimum-purge-trading-days", type=int, default=3)
    parser.add_argument("--max-tuning-rounds", type=int, default=600)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--minimum-iterations", type=int, default=10)
    parser.add_argument("--maximum-iterations", type=int, default=400)
    parser.add_argument("--minimum-calibration-rows", type=int, default=400)
    parser.add_argument("--cache-column-block", type=int, default=64)
    parser.add_argument("--minimum-scope-rows", type=int, default=60)
    parser.add_argument("--minimum-scope-positives", type=int, default=10)
    args = parser.parse_args()

    if not 0 < args.target_recall < 1:
        raise ValueError("target recall은 (0,1) 범위여야 합니다")
    args.scope_columns = parse_str_list(args.scope_columns)
    args.selection_folds = parse_int_list(args.selection_folds)
    args.confirmation_folds = parse_int_list(args.confirmation_folds)
    args.recent_folds = parse_int_list(args.recent_folds)
    args.screen_seeds = parse_int_list(args.screen_seeds)
    args.final_seeds = parse_int_list(args.final_seeds)
    args.families = parse_str_list(args.families)
    args.daily_fraction_grid = [float(value) for value in parse_str_list(args.daily_fraction_grid)]
    args.top_fractions = [float(value) for value in parse_str_list(args.top_fractions)]
    resolve_paths(args)
    require_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)

    status = RunStatus(args.output)
    lock_path = args.output / ".surge_model_zoo_v4.lock"
    with FileLock(lock_path):
        try:
            status.stage("inputs", "RUNNING")
            fold_specs = load_folds(args.folds)
            roles = {
                "selection": args.selection_folds,
                "confirmation": args.confirmation_folds,
                "recent_audit": args.recent_folds,
            }
            all_role_ids = [value for values in roles.values() for value in values]
            if len(all_role_ids) != len(set(all_role_ids)):
                raise ValueError("fold role이 중복 지정되었습니다")
            recipe_payload, raw_recipes = load_recipes(args.recipes, args.quick, args.families)
            profiles = load_profiles(args.profiles, recipe_payload)
            dataset_columns = table_columns(args.dataset)
            recipes, skipped = filter_available_recipes(raw_recipes, profiles, dataset_columns)
            if not recipes:
                raise ValueError("사용 가능한 recipe가 없습니다")
            atomic_write_json(args.output / "SKIPPED_RECIPES.json", skipped)
            preflight = preflight_families(recipes, args.device, args.allow_cpu_fallback)
            atomic_write_json(args.output / "BACKEND_PREFLIGHT.json", preflight)
            status.stage("inputs", "SUCCESS", recipes=len(recipes), skipped=len(skipped))

            status.stage("data_cache", "RUNNING")
            bundle = prepare_data_bundle(args, profiles, recipes, fold_specs)
            status.stage("data_cache", "SUCCESS", rows=len(bundle.frame), features=len(bundle.features))

            selection_folds = [fold for fold in fold_specs if fold.fold_id in set(args.selection_folds)]
            status.stage("screen", "RUNNING")
            screen_metrics = execute_tasks(
                recipes,
                selection_folds,
                args.screen_seeds,
                bundle,
                profiles,
                roles,
                args,
                preflight,
                phase="screen",
            )
            screen_summary = summarize_screen(screen_metrics, args.target_recall)
            atomic_write_csv(args.output / "screen_candidate_summary.csv", screen_summary)
            correlation = screen_prediction_correlation(
                args.output,
                screen_summary["recipe"].astype(str).tolist(),
                selection_folds,
                args.screen_seeds,
            )
            atomic_write_csv(args.output / "screen_prediction_correlation.csv", correlation.reset_index(names="recipe"))
            finalist_names = select_finalists(
                screen_summary,
                correlation,
                top_k=args.finalist_count,
                maximum_correlation=args.maximum_finalist_correlation,
            )
            recipe_by_name = {recipe.name: recipe for recipe in recipes}
            finalists = [recipe_by_name[name] for name in finalist_names]
            atomic_write_json(
                args.output / "FINALIST_REGISTRY.json",
                {
                    "selection_only": True,
                    "finalists": [recipe.to_dict() for recipe in finalists],
                    "screen_seeds": args.screen_seeds,
                    "selection_folds": args.selection_folds,
                },
            )
            status.stage("screen", "SUCCESS", finalists=finalist_names)

            status.stage("finalist_multiseed", "RUNNING")
            finalist_metrics = execute_tasks(
                finalists,
                fold_specs,
                args.final_seeds,
                bundle,
                profiles,
                roles,
                args,
                preflight,
                phase="finalist",
            )
            role_summary, seed_stability = summarize_finalists(finalist_metrics)
            atomic_write_csv(args.output / "finalist_role_summary.csv", role_summary)
            atomic_write_csv(args.output / "seed_stability.csv", seed_stability)
            seed_prediction_diversity = compute_seed_prediction_diversity(
                args.output,
                finalist_names,
                fold_specs,
                args.final_seeds,
            )
            atomic_write_csv(args.output / "seed_prediction_diversity.csv", seed_prediction_diversity)
            status.stage("finalist_multiseed", "SUCCESS")

            status.stage("ensemble", "RUNNING")
            prediction_frame = build_recipe_fold_predictions(
                args.output,
                finalist_names,
                fold_specs,
                args.final_seeds,
                bundle,
            )
            ensemble_methods = ["equal_probability", "equal_date_rank", "nonnegative_logit", "logistic_stacker"]
            crossfit, final_specs = crossfit_ensemble_methods(
                prediction_frame,
                finalist_names,
                args.selection_folds,
                ensemble_methods,
                args,
            )
            atomic_write_csv(args.output / "ensemble_crossfit_metrics.csv", crossfit)
            ensemble_summary = choose_ensemble_method(crossfit, args.target_recall)
            atomic_write_csv(args.output / "ensemble_method_summary.csv", ensemble_summary)
            best_method = str(ensemble_summary.iloc[0]["method"])
            final_spec = final_specs[best_method]
            fold_metrics, calibrator, threshold_policy, scored_payload = evaluate_frozen_ensemble(
                prediction_frame,
                final_spec,
                roles,
                args,
            )
            atomic_write_csv(args.output / "frozen_ensemble_metrics_by_fold.csv", fold_metrics)
            role_metrics = aggregate_role_metrics(fold_metrics)
            atomic_write_csv(args.output / "frozen_ensemble_metrics_by_role.csv", role_metrics)
            scored = scored_payload["scored"]
            atomic_write_npz(
                args.output / "frozen_ensemble_predictions.npz",
                validation_index=scored["validation_index"].to_numpy(dtype=np.int64),
                fold_id=scored["fold_id"].to_numpy(dtype=np.int16),
                dates=scored["date"].to_numpy(dtype="datetime64[ns]").astype(np.int64),
                target=scored["target"].to_numpy(dtype=np.uint8),
                score=scored["ensemble_score"].to_numpy(dtype=np.float32),
                alert=scored["alert"].to_numpy(dtype=np.uint8),
            )
            tradeoff_curve, tradeoff_summary = build_recall_alert_tradeoff(scored, roles, args)
            atomic_write_csv(args.output / "recall_alert_tradeoff_by_role.csv", tradeoff_curve)
            atomic_write_csv(args.output / "recall70_alert_budget_summary.csv", tradeoff_summary)
            scope_frame = scope_metrics(
                scored,
                bundle,
                roles,
                args.scope_columns,
                args.minimum_scope_rows,
            )
            atomic_write_csv(args.output / "recall70_scope_metrics.csv", scope_frame)
            ensemble_bundle = with_payload_checksum(
                {
                    "schema": "crashwatch_surge_ensemble_freeze_v4",
                    "method": best_method,
                    "spec": final_spec,
                    "calibrator": calibrator,
                    "threshold_policy": threshold_policy.to_dict(),
                    "recipes": finalist_names,
                    "seeds": args.final_seeds,
                    "selection_crossfit_summary": ensemble_summary.to_dict(orient="records"),
                }
            )
            atomic_write_json(args.output / "ENSEMBLE_FREEZE.json", ensemble_bundle)
            status.stage("ensemble", "SUCCESS", method=best_method)

            production_registry: dict[str, Any] | None = None
            if not args.skip_production_training:
                status.stage("production_training", "RUNNING")
                production_registry = train_production_models(
                    finalists,
                    fold_specs,
                    args.final_seeds,
                    bundle,
                    profiles,
                    args,
                    preflight,
                )
                status.stage(
                    "production_training",
                    "SUCCESS",
                    model_count=len(production_registry.get("models", [])),
                )
            else:
                status.stage("production_training", "SKIPPED")

            status.stage("diagnostics", "RUNNING")
            gap_report, gap_markdown = build_performance_gap_report(
                role_metrics,
                ensemble_summary,
                seed_stability,
                seed_prediction_diversity,
                scope_frame,
                args,
            )
            atomic_write_json(args.output / "PERFORMANCE_GAP_REPORT.json", gap_report)
            atomic_write_text(args.output / "PERFORMANCE_GAP_REPORT_KO.md", gap_markdown)
            final_recommendation = {
                "schema": "crashwatch_surge_final_recommendation_v4",
                "created_at": utc_now(),
                "target": args.target_column,
                "target_recall": args.target_recall,
                "best_ensemble_method": best_method,
                "finalists": finalist_names,
                "final_seeds": args.final_seeds,
                "threshold_policy": threshold_policy.to_dict(),
                "production_model_registry": (
                    "PRODUCTION_MODEL_REGISTRY.json" if production_registry is not None else None
                ),
                "development_audit_pass": gap_report["development_audit_pass"],
                "role_metrics": gap_report["roles"],
                "next_gate": "2026-06-22 이후 신규 미래 holdout에서 frozen ensemble을 1회 평가",
            }
            atomic_write_json(args.output / "FINAL_RECOMMENDATION.json", final_recommendation)
            manifest = {
                "schema": "crashwatch_surge_model_freeze_manifest_v4",
                "created_at": utc_now(),
                "dataset_path": str(args.dataset),
                "dataset_sha256": bundle.dataset_sha256,
                "target_path": str(args.target_sidecar),
                "target_sha256": bundle.target_sha256,
                "folds_path": str(args.folds),
                "folds_sha256": sha256_file(args.folds),
                "profiles_path": str(args.profiles),
                "profiles_sha256": sha256_file(args.profiles),
                "recipes_path": str(args.recipes),
                "recipes_sha256": sha256_file(args.recipes),
                "feature_hash": hash_strings(bundle.features),
                "selection_only_finalist_choice": True,
                "ensemble": ensemble_bundle,
                "production_training_skipped": bool(args.skip_production_training),
                "production_model_registry": (
                    "PRODUCTION_MODEL_REGISTRY.json" if production_registry is not None else None
                ),
                "success_definition": gap_report["success_definition"],
                "holdout_warning": gap_report["holdout_warning"],
            }
            dynamic_names = {"MODEL_FREEZE_MANIFEST.json", "RUN_STATUS.json", "VERIFICATION_REPORT.json"}
            output_paths = [
                path
                for path in args.output.rglob("*")
                if path.is_file()
                and path.name not in dynamic_names
                and not path.name.startswith(".")
                and not path.name.endswith(".tmp")
            ]
            manifest["output_inventory"] = compute_output_inventory(args.output, output_paths)
            atomic_write_json(args.output / "MODEL_FREEZE_MANIFEST.json", manifest)
            status.stage("diagnostics", "SUCCESS")
            status.success(
                best_method=best_method,
                finalists=finalist_names,
                development_audit_pass=gap_report["development_audit_pass"],
            )

            print("=" * 72)
            print("CrashWatch Surge Model Zoo V4")
            print(f"Finalists             : {', '.join(finalist_names)}")
            print(f"Ensemble              : {best_method}")
            print(f"Threshold policy      : {threshold_policy.kind}")
            print(f"Target recall         : {args.target_recall:.1%}")
            print(f"Development gate pass : {gap_report['development_audit_pass']}")
            print(f"Output                 : {args.output}")
            print("=" * 72)
        except BaseException as exc:
            status.failure(exc)
            raise


if __name__ == "__main__":
    main()
