from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from run_surge_model_zoo_v4 import (
    DataBundle,
    Recipe as V4Recipe,
    build_recipe_fold_predictions,
    fold_indices,
    load_profiles,
    prepare_data_bundle,
)
from surge_alert_budget_common_v5 import (
    DailyBudgetPolicy,
    apply_allocation_biases,
    apply_rank_ensemble_spec,
    daily_budget_selection,
    attach_ranking_metrics,
    build_scope_metrics,
    candidate_allocation_biases,
    evaluate_daily_budget,
    fit_equal_rank_spec,
    fit_hard_negative_meta_lgb,
    fit_optimized_family_rank_spec,
    load_meta_model,
    predict_hard_negative_meta_lgb,
    role_aggregate,
    save_meta_model,
    select_best_method,
    select_minimax_daily_budget_policy,
    summarize_method_crossfit,
)
from surge_model_zoo_common import (
    DATE_COLUMN,
    TARGET_COLUMN,
    TARGET_VALID_COLUMN,
    TICKER_COLUMN,
    FileLock,
    FoldSpec,
    RunStatus,
    _selected_metrics,
    apply_training_window,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_npz,
    atomic_write_text,
    build_inner_windows,
    compute_output_inventory,
    deterministic_seed,
    hash_strings,
    load_folds,
    load_json,
    log,
    payload_checksum_is_valid,
    role_for_fold,
    safe_binary_metrics,
    sha256_bytes,
    sha256_file,
    stable_json_bytes,
    utc_now,
    validate_folds,
    with_payload_checksum,
)
from surge_model_zoo_deployment import relativize_registry_paths


RUNNER_SCHEMA = "crashwatch_surge_alert_budget_runner_v5"


@dataclass(frozen=True)
class RankRecipe:
    name: str
    family: str
    profile: str
    train_policy: str
    relevance_mode: str
    episode_balance: bool
    params: dict[str, Any]
    time_decay_half_life_days: float | None = None
    rolling_days: int | None = None

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "RankRecipe":
        family = str(record["family"])
        if family not in {"lightgbm_ranker", "xgboost_ranker"}:
            raise ValueError(f"지원하지 않는 rank family: {family}")
        relevance_mode = str(record.get("relevance_mode", "hit_day"))
        if relevance_mode not in {"binary", "hit_day"}:
            raise ValueError(f"지원하지 않는 relevance_mode: {relevance_mode}")
        return cls(
            name=str(record["name"]),
            family=family,
            profile=str(record["profile"]),
            train_policy=str(record.get("train_policy", "expanding")),
            relevance_mode=relevance_mode,
            episode_balance=bool(record.get("episode_balance", False)),
            params=dict(record.get("params", {})),
            time_decay_half_life_days=(
                float(record["time_decay_half_life_days"])
                if record.get("time_decay_half_life_days") is not None
                else None
            ),
            rolling_days=int(record["rolling_days"]) if record.get("rolling_days") is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "profile": self.profile,
            "train_policy": self.train_policy,
            "relevance_mode": self.relevance_mode,
            "episode_balance": self.episode_balance,
            "params": self.params,
            "time_decay_half_life_days": self.time_decay_half_life_days,
            "rolling_days": self.rolling_days,
        }


@dataclass(frozen=True)
class RankFitResult:
    prediction: np.ndarray
    best_iteration: int


@dataclass(frozen=True)
class MethodFit:
    method: str
    spec: dict[str, Any]
    policy: DailyBudgetPolicy
    booster: Any | None = None


def parse_int_list(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("정수 목록이 비어 있습니다")
    return list(dict.fromkeys(values))


def parse_float_list(text: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("실수 목록이 비어 있습니다")
    return list(dict.fromkeys(values))


def parse_str_list(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def resolve_paths(args: argparse.Namespace) -> None:
    root = args.package_root.resolve()
    args.package_root = root
    args.v4_output = (args.v4_output or root / "outputs" / "surge_model_zoo_v4").resolve()
    args.output = (args.output or root / "outputs" / "surge_alert_budget_v5").resolve()
    args.dataset = (args.dataset or root / "data" / "training_dataset_finance11h.parquet").resolve()
    args.target_sidecar = (args.target_sidecar or root / "data" / "surge_target_3d5.parquet").resolve()
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
    if args.rank_recipes is None:
        args.rank_recipes = Path(__file__).resolve().parent / "default_alert_budget_recipes_v5.json"
    args.folds = Path(args.folds).resolve()
    args.profiles = Path(args.profiles).resolve()
    args.rank_recipes = Path(args.rank_recipes).resolve()


def require_inputs(args: argparse.Namespace) -> None:
    required = {
        "dataset": args.dataset,
        "target_sidecar": args.target_sidecar,
        "folds": args.folds,
        "profiles": args.profiles,
        "rank_recipes": args.rank_recipes,
        "v4_finalist_registry": args.v4_output / "FINALIST_REGISTRY.json",
        "v4_ensemble_freeze": args.v4_output / "ENSEMBLE_FREEZE.json",
        "v4_production_registry": args.v4_output / "PRODUCTION_MODEL_REGISTRY.json",
    }
    missing = {key: str(path) for key, path in required.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(json.dumps(missing, ensure_ascii=False, indent=2))


def load_rank_recipes(path: Path, quick: bool) -> tuple[dict[str, Any], list[RankRecipe]]:
    payload = load_json(path)
    records = payload.get("recipes", [])
    recipes = [RankRecipe.from_dict(record) for record in records]
    if quick:
        names = {str(value) for value in payload.get("quick_recipe_names", [])}
        recipes = [recipe for recipe in recipes if recipe.name in names]
    if not recipes:
        raise ValueError("실행할 V5 rank recipe가 없습니다")
    recipe_names = [recipe.name for recipe in recipes]
    if len(recipe_names) != len(set(recipe_names)):
        raise ValueError("V5 rank recipe name이 중복되었습니다")
    return payload, recipes


def load_v4_registry(args: argparse.Namespace) -> tuple[list[V4Recipe], list[str], list[int]]:
    finalist_payload = load_json(args.v4_output / "FINALIST_REGISTRY.json")
    freeze = load_json(args.v4_output / "ENSEMBLE_FREEZE.json")
    if not payload_checksum_is_valid(freeze):
        raise ValueError("V4 ENSEMBLE_FREEZE payload checksum mismatch")
    recipes = [V4Recipe.from_dict(record) for record in finalist_payload.get("finalists", [])]
    recipe_names = [str(value) for value in freeze.get("recipes", [])]
    seeds = [int(value) for value in freeze.get("seeds", [])]
    by_name = {recipe.name: recipe for recipe in recipes}
    missing = [name for name in recipe_names if name not in by_name]
    if missing:
        raise KeyError(f"V4 finalist registry recipe 누락: {missing}")
    if not seeds:
        raise ValueError("V4 final seed가 비어 있습니다")
    return [by_name[name] for name in recipe_names], recipe_names, seeds


def _load_existing_v4_bundle(
    args: argparse.Namespace,
    required_features: Sequence[str],
) -> DataBundle | None:
    cache = args.v4_output / "matrix_cache"
    manifest_path = cache / "CACHE_MANIFEST.json"
    matrix_path = cache / "feature_matrix.npy"
    frame_path = cache / "metadata.pkl"
    targets_path = cache / "targets.npz"
    if not all(path.exists() for path in [manifest_path, matrix_path, frame_path, targets_path]):
        return None
    try:
        manifest = load_json(manifest_path)
        if not payload_checksum_is_valid(manifest):
            return None
        identity = manifest.get("identity", {})
        if identity.get("dataset_sha256") != sha256_file(args.dataset):
            return None
        if identity.get("target_sha256") != sha256_file(args.target_sidecar):
            return None
        features = [str(value) for value in identity.get("features", [])]
        if any(feature not in features for feature in required_features):
            return None
        if manifest.get("matrix_path_sha256") != sha256_file(matrix_path):
            return None
        if manifest.get("frame_path_sha256") != sha256_file(frame_path):
            return None
        if manifest.get("targets_path_sha256") != sha256_file(targets_path):
            return None
        frame = pd.read_pickle(frame_path)
        arrays = np.load(targets_path, allow_pickle=False)
        targets = {
            name.removeprefix("target__"): np.asarray(arrays[name])
            for name in arrays.files
            if name.startswith("target__")
        }
        dates = np.asarray(arrays["dates"], dtype=np.int64).astype("datetime64[ns]")
        log("V4 matrix cache 재사용")
        return DataBundle(
            matrix_path=matrix_path,
            metadata_path=manifest_path,
            features=features,
            feature_to_index={feature: index for index, feature in enumerate(features)},
            frame=frame,
            targets=targets,
            dates=dates,
            dataset_sha256=str(identity["dataset_sha256"]),
            target_sha256=str(identity["target_sha256"]),
            cache_identity_hash=str(manifest.get("identity_hash", "")),
        )
    except Exception as exc:
        log(f"V4 matrix cache 재사용 실패, V5 cache 재생성: {type(exc).__name__}: {exc}")
        return None


def prepare_bundle(
    args: argparse.Namespace,
    profiles: Mapping[str, Sequence[str]],
    v4_recipes: Sequence[V4Recipe],
    rank_recipes: Sequence[RankRecipe],
    folds: Sequence[FoldSpec],
) -> DataBundle:
    required_features: list[str] = []
    for recipe in v4_recipes:
        required_features.extend(profiles[recipe.profile])
        if recipe.direction_profile:
            required_features.extend(profiles[recipe.direction_profile])
    for recipe in rank_recipes:
        required_features.extend(profiles[recipe.profile])
    required_features = list(dict.fromkeys(required_features))
    if args.reuse_v4_matrix_cache:
        existing = _load_existing_v4_bundle(args, required_features)
        if existing is not None:
            return existing
    compatibility_recipes = list(v4_recipes)
    for recipe in rank_recipes:
        compatibility_recipes.append(
            V4Recipe(
                name=f"v5_compat__{recipe.name}",
                family="lightgbm",
                profile=recipe.profile,
                target_variant="surge_d3",
                train_policy=recipe.train_policy,
                positive_weight_mode="none",
                params={},
                time_decay_half_life_days=recipe.time_decay_half_life_days,
                rolling_days=recipe.rolling_days,
            )
        )
    return prepare_data_bundle(args, profiles, compatibility_recipes, folds)


def preflight_rankers(args: argparse.Namespace, recipes: Sequence[RankRecipe]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    families = {recipe.family for recipe in recipes}
    if "lightgbm_ranker" in families:
        try:
            import lightgbm as lgb

            report["lightgbm_ranker"] = {"available": True, "version": lgb.__version__, "device": "cpu"}
        except Exception as exc:
            report["lightgbm_ranker"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if "xgboost_ranker" in families:
        try:
            import xgboost as xgb

            resolved_device = "cpu"
            if args.device in {"auto", "cuda"}:
                try:
                    x = np.random.default_rng(11).normal(size=(40, 3)).astype(np.float32)
                    y = np.tile(np.asarray([0, 1, 0, 2], dtype=np.float32), 10)
                    qid = np.repeat(np.arange(10), 4)
                    matrix = xgb.DMatrix(x, label=y, qid=qid)
                    xgb.train(
                        {
                            "objective": "rank:pairwise",
                            "tree_method": "hist",
                            "device": "cuda",
                            "max_depth": 2,
                            "verbosity": 0,
                        },
                        matrix,
                        num_boost_round=2,
                        verbose_eval=False,
                    )
                    resolved_device = "cuda"
                except Exception:
                    if args.device == "cuda" and not args.allow_cpu_fallback:
                        raise
            report["xgboost_ranker"] = {
                "available": True,
                "version": xgb.__version__,
                "device": resolved_device,
            }
        except Exception as exc:
            report["xgboost_ranker"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    unavailable = [family for family in families if not report.get(family, {}).get("available")]
    if unavailable:
        raise RuntimeError(f"ranker backend 사용 불가: {unavailable}\n{json.dumps(report, ensure_ascii=False, indent=2)}")
    atomic_write_json(args.output / "RANKER_BACKEND_PREFLIGHT.json", with_payload_checksum(report))
    return report


def _rank_labels(bundle: DataBundle, indices: np.ndarray, mode: str) -> np.ndarray:
    if mode == "binary":
        return bundle.targets["surge_d3"][indices].astype(np.float32)
    d1 = bundle.targets["surge_d1"][indices].astype(bool)
    d2 = bundle.targets["surge_d2"][indices].astype(bool)
    d3 = bundle.targets["surge_d3"][indices].astype(bool)
    labels = np.zeros(len(indices), dtype=np.float32)
    labels[d3] = 1.0
    labels[d2] = 2.0
    labels[d1] = 3.0
    return labels


def _episode_balance_weights(bundle: DataBundle) -> np.ndarray:
    target = bundle.targets["surge_d3"].astype(np.int8)
    ticker = bundle.frame[TICKER_COLUMN].astype("string").fillna("__MISSING__").to_numpy(dtype=object)
    date = bundle.dates
    order = np.lexsort((date.astype(np.int64), ticker.astype(str)))
    weights = np.ones(len(target), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and ticker[order[end]] == ticker[order[start]]:
            end += 1
        local = order[start:end]
        run: list[int] = []
        for index in local:
            if target[index] == 1:
                run.append(int(index))
            elif run:
                value = 1.0 / len(run)
                weights[np.asarray(run, dtype=np.int64)] = value
                run = []
        if run:
            value = 1.0 / len(run)
            weights[np.asarray(run, dtype=np.int64)] = value
        start = end
    return weights


def _rank_row_weights(
    bundle: DataBundle,
    indices: np.ndarray,
    recipe: RankRecipe,
    episode_weights: np.ndarray,
) -> np.ndarray:
    weights = np.ones(len(indices), dtype=np.float64)
    if recipe.episode_balance:
        weights *= episode_weights[indices]
    if recipe.time_decay_half_life_days is not None:
        dates = pd.to_datetime(pd.Series(bundle.dates[indices]), errors="coerce")
        latest = dates.max()
        age_days = (latest - dates).dt.days.to_numpy(dtype=np.float64)
        weights *= np.power(0.5, age_days / float(recipe.time_decay_half_life_days))
    mean = float(np.mean(weights)) if len(weights) else 1.0
    if mean > 0:
        weights /= mean
    return weights


def _sort_rank_indices(indices: np.ndarray, bundle: DataBundle) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(indices, dtype=np.int64)
    dates = bundle.dates[raw]
    ticker = bundle.frame.iloc[raw][TICKER_COLUMN].astype("string").fillna("").to_numpy(dtype=object)
    order = np.lexsort((ticker.astype(str), dates.astype(np.int64)))
    sorted_indices = raw[order]
    sorted_dates = bundle.dates[sorted_indices]
    _, group_sizes = np.unique(sorted_dates, return_counts=True)
    return sorted_indices, group_sizes.astype(np.uint32), order.astype(np.int64)


def _fit_lightgbm_ranker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    weights: np.ndarray,
    train_groups: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray | None,
    valid_groups: np.ndarray | None,
    params: Mapping[str, Any],
    seed: int,
    threads: int,
    max_rounds: int,
    early_stopping_rounds: int | None,
    fixed_iterations: int | None,
) -> RankFitResult:
    import lightgbm as lgb

    resolved: dict[str, Any] = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10, 20],
        "label_gain": [0, 1, 2, 4],
        "learning_rate": 0.025,
        "num_leaves": 63,
        "min_data_in_leaf": 70,
        "max_bin": 255,
        "verbosity": -1,
        "force_col_wise": True,
        "deterministic": False,
    }
    resolved.update(dict(params))
    resolved.update(
        {
            "num_threads": int(threads),
            "seed": int(seed),
            "feature_fraction_seed": int(seed),
            "bagging_seed": int(seed),
            "data_random_seed": int(seed),
        }
    )
    train_set = lgb.Dataset(
        x_train,
        label=y_train,
        weight=weights,
        group=train_groups,
        free_raw_data=True,
    )
    callbacks = [lgb.log_evaluation(period=0)]
    if fixed_iterations is not None:
        booster = lgb.train(resolved, train_set, num_boost_round=int(fixed_iterations), callbacks=callbacks)
        prediction = booster.predict(x_valid, num_iteration=int(fixed_iterations))
        return RankFitResult(np.asarray(prediction, dtype=np.float64), int(fixed_iterations))
    if y_valid is None or valid_groups is None:
        raise ValueError("LightGBM ranker early stopping validation이 없습니다")
    valid_set = lgb.Dataset(x_valid, label=y_valid, group=valid_groups, reference=train_set, free_raw_data=True)
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
    return RankFitResult(np.asarray(prediction, dtype=np.float64), best)


def _fit_xgboost_ranker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    group_weights: np.ndarray,
    train_groups: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray | None,
    valid_groups: np.ndarray | None,
    params: Mapping[str, Any],
    seed: int,
    threads: int,
    device: str,
    max_rounds: int,
    early_stopping_rounds: int | None,
    fixed_iterations: int | None,
) -> RankFitResult:
    import xgboost as xgb

    resolved: dict[str, Any] = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@20",
        "tree_method": "hist",
        "learning_rate": 0.028,
        "grow_policy": "depthwise",
        "max_depth": 7,
        "min_child_weight": 8.0,
        "subsample": 0.82,
        "colsample_bytree": 0.80,
        "colsample_bynode": 0.88,
        "reg_alpha": 0.25,
        "reg_lambda": 2.5,
        "max_bin": 256,
        "verbosity": 0,
    }
    resolved.update(dict(params))
    resolved.update({"seed": int(seed), "nthread": int(threads), "device": device})
    train_matrix = xgb.DMatrix(x_train, label=y_train)
    train_matrix.set_group(train_groups)
    if len(group_weights) == len(train_groups):
        train_matrix.set_weight(group_weights)
    valid_matrix = xgb.DMatrix(x_valid, label=y_valid if y_valid is not None else None)
    if valid_groups is not None:
        valid_matrix.set_group(valid_groups)
    if fixed_iterations is not None:
        booster = xgb.train(resolved, train_matrix, num_boost_round=int(fixed_iterations), verbose_eval=False)
        prediction = booster.predict(valid_matrix, iteration_range=(0, int(fixed_iterations)))
        return RankFitResult(np.asarray(prediction, dtype=np.float64), int(fixed_iterations))
    if y_valid is None or valid_groups is None:
        raise ValueError("XGBoost ranker early stopping validation이 없습니다")
    booster = xgb.train(
        resolved,
        train_matrix,
        num_boost_round=int(max_rounds),
        evals=[(valid_matrix, "validation")],
        early_stopping_rounds=int(early_stopping_rounds) if early_stopping_rounds else None,
        verbose_eval=False,
    )
    best = int((booster.best_iteration + 1) if booster.best_iteration is not None else max_rounds)
    prediction = booster.predict(valid_matrix, iteration_range=(0, best))
    return RankFitResult(np.asarray(prediction, dtype=np.float64), best)


def fit_predict_rank_recipe(
    recipe: RankRecipe,
    matrix: np.ndarray,
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    episode_weights: np.ndarray,
    fixed_iterations: int | None,
) -> RankFitResult:
    train_indices = apply_training_window(
        train_indices,
        bundle.dates,
        recipe.train_policy,
        recipe.rolling_days,
    )
    if len(train_indices) < 200:
        raise ValueError(f"{recipe.name}: ranker 학습 행이 너무 적습니다")
    feature_indices = np.asarray(
        [bundle.feature_to_index[feature] for feature in profiles[recipe.profile]],
        dtype=np.int64,
    )
    sorted_train, train_groups, _ = _sort_rank_indices(train_indices, bundle)
    sorted_valid, valid_groups, valid_order = _sort_rank_indices(valid_indices, bundle)
    x_train = np.asarray(matrix[np.ix_(sorted_train, feature_indices)], dtype=np.float32)
    x_valid = np.asarray(matrix[np.ix_(sorted_valid, feature_indices)], dtype=np.float32)
    y_train = _rank_labels(bundle, sorted_train, recipe.relevance_mode)
    y_valid = _rank_labels(bundle, sorted_valid, recipe.relevance_mode)
    row_weights = _rank_row_weights(bundle, sorted_train, recipe, episode_weights)

    if recipe.family == "lightgbm_ranker":
        fit = _fit_lightgbm_ranker(
            x_train,
            y_train,
            row_weights,
            train_groups,
            x_valid,
            y_valid,
            valid_groups,
            recipe.params,
            seed,
            args.threads_per_model,
            args.max_tuning_rounds,
            args.early_stopping_rounds,
            fixed_iterations,
        )
    else:
        offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(train_groups, dtype=np.int64)])
        group_weights = np.asarray(
            [float(np.mean(row_weights[offsets[i] : offsets[i + 1]])) for i in range(len(train_groups))],
            dtype=np.float64,
        )
        device = str(preflight["xgboost_ranker"]["device"])
        fit = _fit_xgboost_ranker(
            x_train,
            y_train,
            group_weights,
            train_groups,
            x_valid,
            y_valid,
            valid_groups,
            recipe.params,
            seed,
            args.xgboost_threads,
            device,
            args.max_tuning_rounds,
            args.early_stopping_rounds,
            fixed_iterations,
        )
    prediction_original = np.empty(len(valid_indices), dtype=np.float64)
    prediction_original[valid_order] = fit.prediction
    return RankFitResult(prediction_original, fit.best_iteration)


def _median_iteration(values: Sequence[int], minimum: int, maximum: int) -> int:
    if not values:
        return int(minimum)
    return max(int(minimum), min(int(maximum), int(np.median(np.asarray(values, dtype=np.int64)))))


def rank_task_paths(output: Path, recipe: RankRecipe, fold_id: int, seed: int) -> tuple[Path, Path]:
    directory = output / "rank_task_cache" / recipe.name / f"seed_{seed}"
    return directory / f"fold_{fold_id}.json", directory / f"fold_{fold_id}.npz"


def _task_valid(result_path: Path, prediction_path: Path, identity_hash: str) -> bool:
    if not result_path.exists() or not prediction_path.exists():
        return False
    try:
        payload = load_json(result_path)
    except Exception:
        return False
    return bool(
        payload.get("status") == "completed"
        and payload.get("identity_hash") == identity_hash
        and payload_checksum_is_valid(payload)
        and payload.get("prediction_sha256") == sha256_file(prediction_path)
    )


def run_rank_task(
    recipe: RankRecipe,
    fold: FoldSpec,
    seed: int,
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    episode_weights: np.ndarray,
) -> dict[str, Any]:
    result_path, prediction_path = rank_task_paths(args.output, recipe, fold.fold_id, seed)
    identity = {
        "schema": "crashwatch_surge_rank_task_v5",
        "dataset_sha256": bundle.dataset_sha256,
        "target_sha256": bundle.target_sha256,
        "matrix_cache_identity": bundle.cache_identity_hash,
        "recipe": recipe.to_dict(),
        "features": list(profiles[recipe.profile]),
        "fold": fold.to_dict(),
        "seed": int(seed),
        "inner_windows": int(args.inner_windows),
        "inner_validation_days": int(args.inner_validation_days),
        "inner_purge_days": int(args.inner_purge_days),
        "inner_step_days": int(args.inner_step_days),
        "minimum_inner_train_days": int(args.minimum_inner_train_days),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "budget_common_sha256": sha256_file(Path(__file__).resolve().parent / "surge_alert_budget_common_v5.py"),
    }
    identity_hash = sha256_bytes(stable_json_bytes(identity))
    if args.resume and _task_valid(result_path, prediction_path, identity_hash):
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
        raise RuntimeError(f"{recipe.name} fold={fold.fold_id}: inner window 생성 실패")
    best_iterations: list[int] = []
    for window_id, (inner_train, inner_valid) in enumerate(inner_windows):
        fit = fit_predict_rank_recipe(
            recipe,
            matrix,
            bundle,
            profiles,
            inner_train,
            inner_valid,
            deterministic_seed(seed, recipe.name, fold.fold_id, window_id),
            args,
            preflight,
            episode_weights,
            fixed_iterations=None,
        )
        best_iterations.append(int(fit.best_iteration))
        del fit
        gc.collect()
    effective_iteration = _median_iteration(best_iterations, args.minimum_iterations, args.maximum_iterations)
    final_fit = fit_predict_rank_recipe(
        recipe,
        matrix,
        bundle,
        profiles,
        outer_train,
        outer_valid,
        deterministic_seed(seed, recipe.name, fold.fold_id, "final"),
        args,
        preflight,
        episode_weights,
        fixed_iterations=effective_iteration,
    )
    target = bundle.targets["surge_d3"][outer_valid]
    dates = bundle.dates[outer_valid]
    ranking_all = safe_binary_metrics(
        target,
        1.0 / (1.0 + np.exp(-np.clip(final_fit.prediction, -30.0, 30.0))),
    )
    ranking = {
        "pr_auc": ranking_all.get("pr_auc"),
        "pr_auc_lift": ranking_all.get("pr_auc_lift"),
        "roc_auc": ranking_all.get("roc_auc"),
        "raw_score_mean": float(np.nanmean(final_fit.prediction)),
        "score_is_calibrated_probability": False,
    }
    budget_metrics: dict[str, Any] = {}
    for fraction in args.diagnostic_fractions:
        local = evaluate_daily_budget(target, final_fit.prediction, dates, fraction)
        label = f"daily_{int(round(fraction * 100))}pct"
        budget_metrics[f"{label}_recall"] = local["recall"]
        budget_metrics[f"{label}_lift"] = local["lift"]
        budget_metrics[f"{label}_alert_rate"] = local["alert_rate"]
    atomic_write_npz(
        prediction_path,
        validation_indices=outer_valid.astype(np.int64),
        target=target.astype(np.uint8),
        dates=dates.astype("datetime64[ns]").astype(np.int64),
        raw_prediction=final_fit.prediction.astype(np.float32),
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
            "best_iterations_by_window": best_iterations,
            "effective_iteration": int(effective_iteration),
            "ranking_metrics": ranking,
            "budget_metrics": budget_metrics,
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "elapsed_seconds": float(time.monotonic() - started),
            "completed_at": utc_now(),
        }
    )
    atomic_write_json(result_path, payload)
    return payload


def execute_rank_tasks(
    recipes: Sequence[RankRecipe],
    folds: Sequence[FoldSpec],
    seeds: Sequence[int],
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
) -> pd.DataFrame:
    episode_weights = _episode_balance_weights(bundle)
    total = len(recipes) * len(folds) * len(seeds)
    completed = 0
    progress_lock = threading.Lock()

    def execute_recipe_group(group_recipes: Sequence[RankRecipe]) -> list[dict[str, Any]]:
        nonlocal completed
        group_records: list[dict[str, Any]] = []
        for recipe in group_recipes:
            for seed in seeds:
                for fold in folds:
                    payload = run_rank_task(
                        recipe,
                        fold,
                        seed,
                        bundle,
                        profiles,
                        roles,
                        args,
                        preflight,
                        episode_weights,
                    )
                    row = {
                        "recipe": recipe.name,
                        "family": recipe.family,
                        "profile": recipe.profile,
                        "train_policy": recipe.train_policy,
                        "relevance_mode": recipe.relevance_mode,
                        "fold_id": int(fold.fold_id),
                        "fold_role": role_for_fold(fold.fold_id, roles),
                        "seed": int(seed),
                        "effective_iteration": int(payload["effective_iteration"]),
                        "elapsed_seconds": float(payload["elapsed_seconds"]),
                    }
                    row.update({str(k): v for k, v in payload.get("ranking_metrics", {}).items()})
                    row.update({str(k): v for k, v in payload.get("budget_metrics", {}).items()})
                    group_records.append(row)
                    with progress_lock:
                        completed += 1
                        current = completed
                    if current == 1 or current % 10 == 0 or current == total:
                        log(f"rank task {current}/{total}: {recipe.name} seed={seed} fold={fold.fold_id}")
        return group_records

    records: list[dict[str, Any]] = []
    if args.family_parallel:
        family_order = list(dict.fromkeys(recipe.family for recipe in recipes))
        family_groups = [
            [recipe for recipe in recipes if recipe.family == family]
            for family in family_order
        ]
        log(
            "family-parallel rank execution: "
            + ", ".join(f"{family}={len(group)} recipes" for family, group in zip(family_order, family_groups))
        )
        with ThreadPoolExecutor(max_workers=len(family_groups), thread_name_prefix="rank-family") as executor:
            futures = [executor.submit(execute_recipe_group, group) for group in family_groups]
            for future in as_completed(futures):
                records.extend(future.result())
    else:
        records = execute_recipe_group(recipes)

    frame = pd.DataFrame(records)
    recipe_order = {recipe.name: index for index, recipe in enumerate(recipes)}
    frame["__recipe_order"] = frame["recipe"].map(recipe_order)
    frame.sort_values(["__recipe_order", "seed", "fold_id"], inplace=True, kind="mergesort")
    frame.drop(columns=["__recipe_order"], inplace=True)
    frame.reset_index(drop=True, inplace=True)
    atomic_write_csv(args.output / "ranker_seed_fold_metrics.csv", frame)
    return frame


def load_rank_prediction(output: Path, recipe: str, fold_id: int, seed: int) -> dict[str, np.ndarray]:
    path = output / "rank_task_cache" / recipe / f"seed_{seed}" / f"fold_{fold_id}.npz"
    arrays = np.load(path, allow_pickle=False)
    return {name: np.asarray(arrays[name]) for name in arrays.files}


def build_combined_prediction_frame(
    args: argparse.Namespace,
    bundle: DataBundle,
    folds: Sequence[FoldSpec],
    v4_recipe_names: Sequence[str],
    v4_seeds: Sequence[int],
    rank_recipes: Sequence[RankRecipe],
    rank_seeds: Sequence[int],
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    frame = build_recipe_fold_predictions(
        args.v4_output,
        v4_recipe_names,
        folds,
        v4_seeds,
        bundle,
    )
    finalist_registry = load_json(args.v4_output / "FINALIST_REGISTRY.json")
    v4_family = {
        str(record["name"]): str(record["family"])
        for record in finalist_registry.get("finalists", [])
    }
    family_map = {name: v4_family.get(name, "v4_unknown") for name in v4_recipe_names}
    signal_columns = list(v4_recipe_names)
    by_recipe = {recipe.name: recipe for recipe in rank_recipes}
    for recipe_name, recipe in by_recipe.items():
        parts: list[np.ndarray] = []
        expected_indices: list[np.ndarray] = []
        for fold in folds:
            _, validation_indices = fold_indices(bundle, fold)
            per_seed: list[np.ndarray] = []
            for seed in rank_seeds:
                arrays = load_rank_prediction(args.output, recipe_name, fold.fold_id, seed)
                indices = arrays["validation_indices"].astype(np.int64)
                if not np.array_equal(indices, validation_indices):
                    raise ValueError(f"V5 rank prediction index mismatch: {recipe_name} fold={fold.fold_id} seed={seed}")
                per_seed.append(arrays["raw_prediction"].astype(np.float64))
            parts.append(np.mean(np.vstack(per_seed), axis=0))
            expected_indices.append(validation_indices)
        values = np.concatenate(parts)
        indices = np.concatenate(expected_indices)
        mapping = pd.Series(values, index=indices)
        frame[recipe_name] = frame["validation_index"].map(mapping).to_numpy(dtype=np.float64)
        family_map[recipe_name] = recipe.family
        signal_columns.append(recipe_name)

    metadata_columns = [TICKER_COLUMN, *args.scope_columns]
    for column in metadata_columns:
        if column in bundle.frame.columns:
            frame[column] = bundle.frame.iloc[frame["validation_index"].to_numpy(dtype=np.int64)][column].to_numpy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame, family_map, signal_columns


def _select_equal_policy(
    train_frame: pd.DataFrame,
    spec: Mapping[str, Any],
    args: argparse.Namespace,
) -> DailyBudgetPolicy:
    base_score = apply_rank_ensemble_spec(spec, train_frame)
    group_values = (
        train_frame[args.allocation_column].to_numpy(dtype=object)
        if args.allocation_column and args.allocation_column in train_frame.columns
        else None
    )
    bias_candidates = candidate_allocation_biases(group_values, args.allocation_bias_grid)
    best: tuple[tuple[float, ...], DailyBudgetPolicy] | None = None
    for biases in bias_candidates:
        local = train_frame[["fold_id", "date", "target", TICKER_COLUMN] + ([args.allocation_column] if args.allocation_column in train_frame.columns else [])].copy()
        local["score"] = base_score
        policy, summary, _ = select_minimax_daily_budget_policy(
            local,
            args.daily_fraction_grid,
            args.target_recall,
            args.selection_recall_buffer,
            args.minimum_precision_lift,
            args.max_alert_rate,
            args.required_selection_fold_pass_rate,
            max_alerts_per_day=args.max_alerts_per_day,
            allocation_column=args.allocation_column if args.allocation_column in local.columns else None,
            allocation_biases=biases,
            source="selection_crossfit_equal_rank",
        )
        row = summary.loc[np.isclose(summary["fraction"], policy.daily_fraction)].iloc[0]
        objective = (
            1.0 if policy.gate_pass else 0.0,
            float(row["fold_pass_rate"]),
            float(row["worst_recall"]),
            float(row["worst_precision_lift"]),
            -float(row["mean_alert_rate"]),
        )
        if best is None or objective > best[0]:
            best = (objective, policy)
    if best is None:
        raise RuntimeError("equal rank policy 선택 실패")
    return best[1]


def _nested_meta_oof_scores(
    train_frame: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    scores = np.full(len(train_frame), np.nan, dtype=np.float64)
    fold_values = sorted(train_frame["fold_id"].unique().tolist())
    for heldout in fold_values:
        train_mask = train_frame["fold_id"].ne(heldout).to_numpy()
        valid_mask = train_frame["fold_id"].eq(heldout).to_numpy()
        if int(train_mask.sum()) < 200 or int(valid_mask.sum()) == 0:
            continue
        booster, spec = fit_hard_negative_meta_lgb(
            train_frame.loc[train_mask].reset_index(drop=True),
            signal_columns,
            family_map,
            deterministic_seed(seed, "nested_meta", int(heldout)),
            iterations=args.meta_iterations,
            hard_negative_pool_fraction=args.hard_negative_pool_fraction,
            hard_negative_multiplier=args.hard_negative_multiplier,
            market_column=args.allocation_column if args.allocation_column in train_frame.columns else None,
        )
        scores[valid_mask] = predict_hard_negative_meta_lgb(
            booster,
            spec,
            train_frame.loc[valid_mask].reset_index(drop=True),
        )
    if not np.isfinite(scores).all():
        fallback_spec = fit_equal_rank_spec(signal_columns, family_map, "equal_family_rank")
        fallback = apply_rank_ensemble_spec(fallback_spec, train_frame)
        scores[~np.isfinite(scores)] = fallback[~np.isfinite(scores)]
    return scores


def fit_method(
    method: str,
    train_frame: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    args: argparse.Namespace,
    seed: int,
    persist_meta_path: Path | None = None,
) -> MethodFit:
    if method in {"equal_recipe_rank", "equal_family_rank"}:
        spec = fit_equal_rank_spec(
            signal_columns,
            family_map,
            method,
            allocation_column=args.allocation_column if args.allocation_column in train_frame.columns else None,
        )
        policy = _select_equal_policy(train_frame, spec, args)
        return MethodFit(method, spec, policy, None)
    if method == "optimized_family_rank":
        spec, policy, search = fit_optimized_family_rank_spec(
            train_frame.reset_index(drop=True),
            signal_columns,
            family_map,
            args.daily_fraction_grid,
            args.target_recall,
            args.selection_recall_buffer,
            args.minimum_precision_lift,
            args.max_alert_rate,
            args.required_selection_fold_pass_rate,
            args.max_alerts_per_day,
            args.random_weight_samples,
            seed,
            allocation_column=args.allocation_column if args.allocation_column in train_frame.columns else None,
            allocation_bias_grid=args.allocation_bias_grid,
        )
        if persist_meta_path is not None:
            atomic_write_csv(persist_meta_path.with_suffix(".weight_search.csv"), search)
        return MethodFit(method, spec, policy, None)
    if method == "hard_negative_meta_lgb":
        nested_scores = _nested_meta_oof_scores(train_frame, signal_columns, family_map, args, seed)
        local = train_frame[["fold_id", "date", "target", TICKER_COLUMN] + ([args.allocation_column] if args.allocation_column in train_frame.columns else [])].copy()
        local["score"] = nested_scores
        policy, _, _ = select_minimax_daily_budget_policy(
            local.reset_index(drop=True),
            args.daily_fraction_grid,
            args.target_recall,
            args.selection_recall_buffer,
            args.minimum_precision_lift,
            args.max_alert_rate,
            args.required_selection_fold_pass_rate,
            max_alerts_per_day=args.max_alerts_per_day,
            allocation_column=args.allocation_column if args.allocation_column in local.columns else None,
            source="selection_nested_meta",
        )
        booster, spec = fit_hard_negative_meta_lgb(
            train_frame.reset_index(drop=True),
            signal_columns,
            family_map,
            seed,
            iterations=args.meta_iterations,
            hard_negative_pool_fraction=args.hard_negative_pool_fraction,
            hard_negative_multiplier=args.hard_negative_multiplier,
            market_column=args.allocation_column if args.allocation_column in train_frame.columns else None,
        )
        if persist_meta_path is not None:
            model_record = save_meta_model(booster, persist_meta_path)
            spec["model"] = model_record
        return MethodFit(method, spec, policy, booster)
    raise ValueError(method)


def predict_method(fit: MethodFit, frame: pd.DataFrame) -> np.ndarray:
    if fit.method == "hard_negative_meta_lgb":
        booster = fit.booster
        if booster is None:
            model_record = fit.spec.get("model")
            if not isinstance(model_record, dict):
                raise ValueError("meta model record가 없습니다")
            booster = load_meta_model(Path(str(model_record["path"])))
        return predict_hard_negative_meta_lgb(booster, fit.spec, frame.reset_index(drop=True))
    return apply_rank_ensemble_spec(fit.spec, frame.reset_index(drop=True))


def crossfit_methods(
    prediction_frame: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    selection_fold_ids: Sequence[int],
    methods: Sequence[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_records: list[dict[str, Any]] = []
    score_records: list[pd.DataFrame] = []
    selection_set = {int(value) for value in selection_fold_ids}
    selection = prediction_frame.loc[prediction_frame["fold_id"].isin(selection_set)].copy()
    for method in methods:
        log(f"selection cross-fit method: {method}")
        for heldout in sorted(selection_set):
            train = selection.loc[selection["fold_id"].ne(heldout)].copy().reset_index(drop=True)
            valid = selection.loc[selection["fold_id"].eq(heldout)].copy().reset_index(drop=True)
            fit = fit_method(
                method,
                train,
                signal_columns,
                family_map,
                args,
                deterministic_seed(args.policy_seed, method, heldout),
            )
            raw_score = predict_method(fit, valid)
            groups = valid[args.allocation_column].to_numpy(dtype=object) if fit.policy.allocation_column and fit.policy.allocation_column in valid else None
            # Fold-specific allocation bias is learned only from the other
            # selection folds.  Store the already-adjusted OOF score so final
            # policy fitting cannot accidentally discard or double-apply it.
            score = apply_allocation_biases(raw_score, groups, fit.policy.allocation_biases)
            metrics = evaluate_daily_budget(
                valid["target"].to_numpy(dtype=np.int8),
                score,
                valid["date"].to_numpy(dtype="datetime64[ns]"),
                fit.policy.daily_fraction,
                max_alerts_per_day=fit.policy.max_alerts_per_day,
                groups=None,
                allocation_biases=None,
                tickers=valid[TICKER_COLUMN].to_numpy(dtype=object) if TICKER_COLUMN in valid else None,
            )
            attach_ranking_metrics(metrics, valid["target"].to_numpy(dtype=np.int8), score)
            metrics.update(
                {
                    "method": method,
                    "fold_id": int(heldout),
                    "selected_fraction_from_other_folds": float(fit.policy.daily_fraction),
                    "train_policy_gate_pass": bool(fit.policy.gate_pass),
                }
            )
            metric_records.append(metrics)
            rows = valid[["validation_index", "fold_id", "date", "target"]].copy()
            for column in [TICKER_COLUMN, *args.scope_columns]:
                if column in valid.columns:
                    rows[column] = valid[column].to_numpy()
            rows["method"] = method
            rows["score"] = score
            score_records.append(rows)
    metrics = pd.DataFrame(metric_records)
    scores = pd.concat(score_records, ignore_index=True)
    return metrics, scores


def fit_final_method_and_policy(
    method: str,
    prediction_frame: pd.DataFrame,
    crossfit_scores: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    selection_fold_ids: Sequence[int],
    args: argparse.Namespace,
) -> MethodFit:
    selection = prediction_frame.loc[prediction_frame["fold_id"].isin([int(value) for value in selection_fold_ids])].copy().reset_index(drop=True)
    meta_path = args.output / "production" / "meta" / "hard_negative_meta_lgb.txt" if method == "hard_negative_meta_lgb" else None
    fit = fit_method(
        method,
        selection,
        signal_columns,
        family_map,
        args,
        deterministic_seed(args.policy_seed, method, "final"),
        persist_meta_path=meta_path,
    )
    oof = crossfit_scores.loc[crossfit_scores["method"].eq(method)].copy().reset_index(drop=True)
    crossfit_policy, summary, fold_rows = select_minimax_daily_budget_policy(
        oof,
        args.daily_fraction_grid,
        args.target_recall,
        args.selection_recall_buffer,
        args.minimum_precision_lift,
        args.max_alert_rate,
        args.required_selection_fold_pass_rate,
        max_alerts_per_day=args.max_alerts_per_day,
        # OOF scores already contain the fold-specific allocation adjustment.
        allocation_column=None,
        allocation_biases=None,
        source="selected_method_crossfit_oof",
    )
    policy_values = crossfit_policy.to_dict()
    policy_values["allocation_column"] = fit.policy.allocation_column
    policy_values["allocation_biases"] = fit.policy.allocation_biases
    policy = DailyBudgetPolicy.from_dict(policy_values)
    atomic_write_csv(args.output / "final_policy_fraction_summary.csv", summary)
    atomic_write_csv(args.output / "final_policy_fold_grid.csv", fold_rows)
    return MethodFit(method, fit.spec, policy, fit.booster)


def evaluate_final_candidate(
    fit: MethodFit,
    prediction_frame: pd.DataFrame,
    selected_crossfit_scores: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selection_ids = {int(value) for value in roles["selection"]}
    score_by_index = pd.Series(
        selected_crossfit_scores["score"].to_numpy(dtype=np.float64),
        index=selected_crossfit_scores["validation_index"].to_numpy(dtype=np.int64),
    )
    scored_parts: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    for fold_id, part in prediction_frame.groupby("fold_id", sort=True):
        local = part.copy().reset_index(drop=True)
        groups = local[args.allocation_column].to_numpy(dtype=object) if fit.policy.allocation_column and fit.policy.allocation_column in local else None
        if int(fold_id) in selection_ids:
            # Selection OOF scores were adjusted by a bias trained without the
            # held-out fold.  Applying the full-selection bias again would be
            # both leakage-prone and a double adjustment.
            score = local["validation_index"].map(score_by_index).to_numpy(dtype=np.float64)
            if not np.isfinite(score).all():
                raise ValueError(f"selection crossfit score 누락 fold={fold_id}")
        else:
            raw_score = predict_method(fit, local)
            score = apply_allocation_biases(raw_score, groups, fit.policy.allocation_biases)
        alert = daily_budget_selection(
            score,
            local["date"].to_numpy(dtype="datetime64[ns]"),
            fit.policy.daily_fraction,
            max_alerts_per_day=fit.policy.max_alerts_per_day,
        )
        metrics = evaluate_daily_budget(
            local["target"].to_numpy(dtype=np.int8),
            score,
            local["date"].to_numpy(dtype="datetime64[ns]"),
            fit.policy.daily_fraction,
            max_alerts_per_day=fit.policy.max_alerts_per_day,
            groups=None,
            allocation_biases=None,
            tickers=local[TICKER_COLUMN].to_numpy(dtype=object) if TICKER_COLUMN in local else None,
        )
        attach_ranking_metrics(metrics, local["target"].to_numpy(dtype=np.int8), score)
        fold_records.append(
            {
                "fold_id": int(fold_id),
                "fold_role": role_for_fold(int(fold_id), roles),
                "method": fit.method,
                **metrics,
            }
        )
        scored = local[["validation_index", "fold_id", "date", "target"]].copy()
        for column in [TICKER_COLUMN, *args.scope_columns]:
            if column in local.columns:
                scored[column] = local[column].to_numpy()
        scored["score"] = score
        scored["alert"] = alert
        scored_parts.append(scored)
    fold_frame = pd.DataFrame(fold_records)
    role_frame = role_aggregate(fold_frame)
    scored_frame = pd.concat(scored_parts, ignore_index=True)
    return fold_frame, role_frame, scored_frame


def build_tradeoff_curves(
    scored: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    fractions: Sequence[float],
    args: argparse.Namespace,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for role, fold_ids in roles.items():
        part = scored.loc[scored["fold_id"].isin([int(value) for value in fold_ids])].copy()
        if part.empty:
            continue
        for fraction in fractions:
            # final-candidate score is already on the allocation-adjusted
            # ranking scale for every role.
            metrics = evaluate_daily_budget(
                part["target"].to_numpy(dtype=np.int8),
                part["score"].to_numpy(dtype=np.float64),
                part["date"].to_numpy(dtype="datetime64[ns]"),
                fraction,
                max_alerts_per_day=args.max_alerts_per_day,
                groups=None,
                allocation_biases=None,
                tickers=part[TICKER_COLUMN].to_numpy(dtype=object) if TICKER_COLUMN in part else None,
            )
            records.append({"fold_role": role, "fraction": float(fraction), **metrics})
    return pd.DataFrame(records)


def _production_iteration(rank_metrics: pd.DataFrame, recipe: str, seed: int, args: argparse.Namespace) -> int:
    part = rank_metrics.loc[(rank_metrics["recipe"] == recipe) & (rank_metrics["seed"] == seed)]
    values = pd.to_numeric(part["effective_iteration"], errors="coerce").dropna().to_numpy(dtype=np.int64)
    return _median_iteration(values.tolist(), args.minimum_iterations, args.maximum_iterations)


def _save_ranker_production_model(
    recipe: RankRecipe,
    seed: int,
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    rank_metrics: pd.DataFrame,
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    episode_weights: np.ndarray,
) -> dict[str, Any]:
    import lightgbm as lgb
    import xgboost as xgb

    all_indices = np.arange(len(bundle.dates), dtype=np.int64)
    all_indices = apply_training_window(all_indices, bundle.dates, recipe.train_policy, recipe.rolling_days)
    feature_names = [str(value) for value in profiles[recipe.profile]]
    feature_indices = np.asarray([bundle.feature_to_index[value] for value in feature_names], dtype=np.int64)
    sorted_indices, groups, _ = _sort_rank_indices(all_indices, bundle)
    matrix = np.load(bundle.matrix_path, mmap_mode="r")
    x = np.asarray(matrix[np.ix_(sorted_indices, feature_indices)], dtype=np.float32)
    y = _rank_labels(bundle, sorted_indices, recipe.relevance_mode)
    iterations = _production_iteration(rank_metrics, recipe.name, seed, args)
    model_dir = args.output / "production" / "rankers" / recipe.name / f"seed_{seed}"
    model_dir.mkdir(parents=True, exist_ok=True)
    if recipe.family == "lightgbm_ranker":
        weights = _rank_row_weights(bundle, sorted_indices, recipe, episode_weights)
        params: dict[str, Any] = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [10, 20],
            "label_gain": [0, 1, 2, 4],
            "learning_rate": 0.025,
            "num_leaves": 63,
            "min_data_in_leaf": 70,
            "verbosity": -1,
            "force_col_wise": True,
        }
        params.update(recipe.params)
        params.update(
            {
                "num_threads": int(args.threads_per_model),
                "seed": int(seed),
                "feature_fraction_seed": int(seed),
                "bagging_seed": int(seed),
                "data_random_seed": int(seed),
            }
        )
        dataset = lgb.Dataset(x, label=y, weight=weights, group=groups, free_raw_data=True)
        booster = lgb.train(params, dataset, num_boost_round=iterations, callbacks=[lgb.log_evaluation(period=0)])
        path = model_dir / "model.txt"
        booster.save_model(str(path), num_iteration=iterations)
        model_record = {
            "format": "lightgbm_text",
            "path": str(path),
            "iterations": int(iterations),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    else:
        params = {
            "objective": "rank:pairwise",
            "eval_metric": "ndcg@20",
            "tree_method": "hist",
            "learning_rate": 0.028,
            "max_depth": 7,
            "verbosity": 0,
        }
        params.update(recipe.params)
        params.update(
            {
                "seed": int(seed),
                "nthread": int(args.xgboost_threads),
                "device": str(preflight["xgboost_ranker"]["device"]),
            }
        )
        dtrain = xgb.DMatrix(x, label=y)
        dtrain.set_group(groups)
        booster = xgb.train(params, dtrain, num_boost_round=iterations, verbose_eval=False)
        path = model_dir / "model.json"
        booster.save_model(path)
        model_record = {
            "format": "xgboost_json",
            "path": str(path),
            "iterations": int(iterations),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return {
        "recipe": recipe.name,
        "family": recipe.family,
        "profile": recipe.profile,
        "seed": int(seed),
        "features": feature_names,
        "train_policy": recipe.train_policy,
        "rolling_days": recipe.rolling_days,
        "time_decay_half_life_days": recipe.time_decay_half_life_days,
        "relevance_mode": recipe.relevance_mode,
        "model": model_record,
    }


def train_ranker_production_registry(
    recipes: Sequence[RankRecipe],
    seeds: Sequence[int],
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    rank_metrics: pd.DataFrame,
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    episode_weights = _episode_balance_weights(bundle)
    def train_recipe_group(group_recipes: Sequence[RankRecipe]) -> list[dict[str, Any]]:
        group_records: list[dict[str, Any]] = []
        for recipe in group_recipes:
            for seed in seeds:
                log(f"production ranker: {recipe.name} seed={seed}")
                group_records.append(
                    _save_ranker_production_model(
                        recipe,
                        seed,
                        bundle,
                        profiles,
                        rank_metrics,
                        args,
                        preflight,
                        episode_weights,
                    )
                )
        return group_records

    records: list[dict[str, Any]] = []
    if args.family_parallel:
        family_order = list(dict.fromkeys(recipe.family for recipe in recipes))
        family_groups = [[recipe for recipe in recipes if recipe.family == family] for family in family_order]
        log("family-parallel production training")
        with ThreadPoolExecutor(max_workers=len(family_groups), thread_name_prefix="production-family") as executor:
            futures = [executor.submit(train_recipe_group, group) for group in family_groups]
            for future in as_completed(futures):
                records.extend(future.result())
    else:
        records = train_recipe_group(recipes)
    recipe_order = {recipe.name: index for index, recipe in enumerate(recipes)}
    records.sort(key=lambda record: (recipe_order[str(record["recipe"])], int(record["seed"])))
    payload = with_payload_checksum(
        {
            "schema": "crashwatch_surge_ranker_production_registry_v5",
            "dataset_sha256": bundle.dataset_sha256,
            "target_sha256": bundle.target_sha256,
            "recipes": [recipe.name for recipe in recipes],
            "seeds": [int(value) for value in seeds],
            "models": records,
            "created_at": utc_now(),
        }
    )
    portable = relativize_registry_paths(payload, args.output)
    portable = with_payload_checksum({key: value for key, value in portable.items() if key != "payload_sha256"})
    atomic_write_json(args.output / "V5_RANKER_PRODUCTION_REGISTRY.json", portable)
    return portable


def build_gap_report(
    fold_metrics: pd.DataFrame,
    role_metrics: pd.DataFrame,
    policy: DailyBudgetPolicy,
    tradeoff: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    gaps: list[dict[str, Any]] = []
    for _, row in role_metrics.iterrows():
        role = str(row["fold_role"])
        recall = float(row.get("recall", np.nan))
        lift = float(row.get("lift", np.nan))
        alert = float(row.get("alert_rate", np.nan))
        if not np.isfinite(recall) or recall < args.target_recall:
            gaps.append({"severity": "HIGH", "role": role, "area": "RECALL_SHORTFALL", "value": recall, "required": args.target_recall})
        if not np.isfinite(lift) or lift < args.minimum_precision_lift:
            gaps.append({"severity": "HIGH", "role": role, "area": "PRECISION_LIFT_SHORTFALL", "value": lift, "required": args.minimum_precision_lift})
        if not np.isfinite(alert) or alert > args.max_alert_rate + 1e-12:
            gaps.append({"severity": "HIGH", "role": role, "area": "ALERT_RATE_TOO_HIGH", "value": alert, "required_max": args.max_alert_rate})
    for _, row in fold_metrics.iterrows():
        if float(row.get("recall", np.nan)) < args.target_recall:
            gaps.append({"severity": "MEDIUM", "role": str(row["fold_role"]), "fold_id": int(row["fold_id"]), "area": "FOLD_RECALL_SHORTFALL", "value": float(row["recall"])})
    selection_gate = bool(policy.gate_pass)
    transfer_roles = role_metrics.loc[role_metrics["fold_role"].isin(["confirmation", "recent_audit"])]
    transfer_pass = bool(
        not transfer_roles.empty
        and (pd.to_numeric(transfer_roles["recall"], errors="coerce") >= args.target_recall).all()
        and (pd.to_numeric(transfer_roles["lift"], errors="coerce") >= args.minimum_precision_lift).all()
        and (pd.to_numeric(transfer_roles["alert_rate"], errors="coerce") <= args.max_alert_rate + 1e-12).all()
    )
    status = "READY_FOR_NEW_FUTURE_HOLDOUT" if selection_gate and transfer_pass else "STOP_BUDGET_RECALL_GATE"
    report = with_payload_checksum(
        {
            "schema": "crashwatch_surge_budget_gap_report_v5",
            "status": status,
            "selection_policy_gate_pass": selection_gate,
            "development_transfer_gate_pass": transfer_pass,
            "target_recall": args.target_recall,
            "minimum_precision_lift": args.minimum_precision_lift,
            "maximum_alert_rate": args.max_alert_rate,
            "maximum_alerts_per_day": args.max_alerts_per_day,
            "selected_policy": policy.to_dict(),
            "gaps": gaps,
            "warning": "confirmation/recent는 반복 관찰된 개발 구간이며 최종 주장은 신규 미래 holdout에서만 가능",
        }
    )
    lines = [
        "# Surge Alert Budget V5 성능 보고서",
        "",
        f"최종 상태: **{status}**",
        "",
        f"고정 목표: recall ≥ {args.target_recall:.1%}, precision lift ≥ {args.minimum_precision_lift:.2f}, alert rate ≤ {args.max_alert_rate:.1%}.",
        f"정책: 날짜별 상위 {policy.daily_fraction:.1%}, 일일 최대 {policy.max_alerts_per_day if policy.max_alerts_per_day is not None else '제한 없음'}개.",
        "",
        "## 역할별 결과",
        "",
        "| 구간 | Recall | Precision | Lift | Alert rate | Event recall | PR-AUC lift |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in role_metrics.iterrows():
        lines.append(
            f"| {row['fold_role']} | {float(row.get('recall', np.nan)):.4f} | "
            f"{float(row.get('precision', np.nan)):.4f} | {float(row.get('lift', np.nan)):.4f} | "
            f"{float(row.get('alert_rate', np.nan)):.4f} | {float(row.get('event_recall', np.nan)):.4f} | "
            f"{float(row.get('pr_auc_lift', np.nan)):.4f} |"
        )
    lines.extend(["", "## 남은 부족 구간", ""])
    if gaps:
        for gap in gaps:
            lines.append(f"- [{gap['severity']}] {gap.get('role', '')} {gap['area']}: {gap.get('value')}")
    else:
        lines.append("- 개발 구간 Gate 기준의 부족 항목 없음.")
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "이 정책은 경보량 상한을 넘기지 않는다. Gate가 실패해도 50~60%로 자동 완화하지 않으며, best-effort 후보만 기록한다.",
            "기존 fold 5~7은 반복 관찰된 개발 구간이므로 최종 배포 주장은 2026-06-22 이후 신규 데이터에서 동결 정책을 한 번 적용한 뒤에만 가능하다.",
        ]
    )
    return report, "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch Surge V5 realistic alert-budget ranker")
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--v4-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--rank-recipes", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-v4-matrix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--selection-folds", default="0,1,2,3,4")
    parser.add_argument("--confirmation-folds", default="5,6")
    parser.add_argument("--recent-folds", default="7")
    parser.add_argument("--rank-seeds", default="17,29,41")
    parser.add_argument("--methods", default="equal_recipe_rank,equal_family_rank,optimized_family_rank,hard_negative_meta_lgb")
    parser.add_argument("--target-recall", type=float, default=0.70)
    parser.add_argument("--selection-recall-buffer", type=float, default=0.02)
    parser.add_argument("--minimum-precision-lift", type=float, default=1.10)
    parser.add_argument("--max-alert-rate", type=float, default=0.40)
    parser.add_argument("--max-alerts-per-day", type=int, default=20)
    parser.add_argument("--required-selection-fold-pass-rate", type=float, default=1.0)
    parser.add_argument("--daily-fraction-grid", default="0.15,0.20,0.25,0.30,0.35,0.40")
    parser.add_argument("--diagnostic-fractions", default="0.10,0.20,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    parser.add_argument("--allocation-column", default="market")
    parser.add_argument("--allocation-bias-grid", default="-0.15,-0.10,-0.05,0,0.05,0.10,0.15")
    parser.add_argument("--random-weight-samples", type=int, default=128)
    parser.add_argument("--policy-seed", type=int, default=20260810)
    parser.add_argument("--hard-negative-pool-fraction", type=float, default=0.50)
    parser.add_argument("--hard-negative-multiplier", type=float, default=3.0)
    parser.add_argument("--meta-iterations", type=int, default=160)
    parser.add_argument("--threads-per-model", type=int, default=4)
    parser.add_argument("--xgboost-threads", type=int, default=4)
    parser.add_argument(
        "--family-parallel",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="LightGBM CPU family와 XGBoost GPU family를 동시에 순차 처리",
    )
    parser.add_argument("--inner-windows", type=int, default=3)
    parser.add_argument("--inner-validation-days", type=int, default=60)
    parser.add_argument("--inner-purge-days", type=int, default=20)
    parser.add_argument("--inner-step-days", type=int, default=120)
    parser.add_argument("--minimum-inner-train-days", type=int, default=500)
    parser.add_argument("--minimum-purge-trading-days", type=int, default=3)
    parser.add_argument("--max-tuning-rounds", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--minimum-iterations", type=int, default=10)
    parser.add_argument("--maximum-iterations", type=int, default=350)
    parser.add_argument("--cache-column-block", type=int, default=64)
    parser.add_argument("--date-column", default=DATE_COLUMN)
    parser.add_argument("--ticker-column", default=TICKER_COLUMN)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--target-valid-column", default=TARGET_VALID_COLUMN)
    parser.add_argument("--return-column", default="t_price_ret_1")
    parser.add_argument("--scope-columns", default="industry_name,market,bucket")
    parser.add_argument("--minimum-scope-rows", type=int, default=30)
    parser.add_argument("--minimum-scope-positives", type=int, default=5)
    parser.add_argument("--skip-production-training", action="store_true")
    args = parser.parse_args()

    args.selection_folds = parse_int_list(args.selection_folds)
    args.confirmation_folds = parse_int_list(args.confirmation_folds)
    args.recent_folds = parse_int_list(args.recent_folds)
    args.rank_seeds = parse_int_list(args.rank_seeds)
    args.methods = parse_str_list(args.methods)
    args.daily_fraction_grid = parse_float_list(args.daily_fraction_grid)
    args.diagnostic_fractions = parse_float_list(args.diagnostic_fractions)
    args.allocation_bias_grid = parse_float_list(args.allocation_bias_grid)
    args.scope_columns = parse_str_list(args.scope_columns)
    if not 0 < args.target_recall < 1:
        raise ValueError("target_recall은 (0,1) 범위여야 합니다")
    if not 0 < args.max_alert_rate <= 1:
        raise ValueError("max_alert_rate는 (0,1] 범위여야 합니다")
    if any(value > args.max_alert_rate + 1e-12 for value in args.daily_fraction_grid):
        raise ValueError("daily_fraction_grid에 max_alert_rate 초과값이 있습니다. V5는 경보 상한을 자동 완화하지 않습니다")
    resolve_paths(args)
    require_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)

    status = RunStatus(args.output)
    status.payload["schema"] = RUNNER_SCHEMA
    atomic_write_json(status.path, status.payload)
    with FileLock(args.output / ".surge_alert_budget_v5.lock"):
        try:
            status.stage("inputs", "RUNNING")
            folds = load_folds(args.folds)
            roles = {
                "selection": args.selection_folds,
                "confirmation": args.confirmation_folds,
                "recent_audit": args.recent_folds,
            }
            all_ids = [value for values in roles.values() for value in values]
            if not roles["selection"]:
                raise ValueError("selection fold가 비어 있습니다")
            if len(all_ids) != len(set(all_ids)):
                raise ValueError("fold role이 중복되었습니다")
            available_fold_ids = {int(fold.fold_id) for fold in folds}
            missing_role_ids = sorted(set(all_ids) - available_fold_ids)
            if missing_role_ids:
                raise ValueError(f"fold JSON에 없는 role fold ID: {missing_role_ids}")
            recipe_payload, rank_recipes = load_rank_recipes(args.rank_recipes, args.quick)
            v4_recipes, v4_recipe_names, v4_seeds = load_v4_registry(args)
            profiles = load_profiles(args.profiles, {"custom_profiles": load_json(Path(__file__).resolve().parent / "default_model_zoo_recipes.json").get("custom_profiles", {})})
            missing_profiles = sorted({recipe.profile for recipe in rank_recipes if recipe.profile not in profiles})
            if missing_profiles:
                raise KeyError(f"V5 rank profile 누락: {missing_profiles}")
            status.stage("inputs", "SUCCESS", v4_recipes=len(v4_recipe_names), rank_recipes=len(rank_recipes))

            status.stage("bundle", "RUNNING")
            bundle = prepare_bundle(args, profiles, v4_recipes, rank_recipes, folds)
            validate_folds(folds, pd.Series(bundle.dates), minimum_purge_days=args.minimum_purge_trading_days)
            status.stage("bundle", "SUCCESS", rows=len(bundle.dates), features=len(bundle.features))

            status.stage("preflight", "RUNNING")
            preflight = preflight_rankers(args, rank_recipes)
            status.stage("preflight", "SUCCESS")

            status.stage("rank_tasks", "RUNNING")
            rank_metrics = execute_rank_tasks(
                rank_recipes,
                folds,
                args.rank_seeds,
                bundle,
                profiles,
                roles,
                args,
                preflight,
            )
            status.stage("rank_tasks", "SUCCESS", tasks=len(rank_metrics))

            status.stage("prediction_frame", "RUNNING")
            prediction_frame, family_map, signal_columns = build_combined_prediction_frame(
                args,
                bundle,
                folds,
                v4_recipe_names,
                v4_seeds,
                rank_recipes,
                args.rank_seeds,
            )
            atomic_write_json(
                args.output / "SIGNAL_REGISTRY.json",
                with_payload_checksum(
                    {
                        "v4_output": str(args.v4_output),
                        "signals": signal_columns,
                        "family_map": family_map,
                        "v4_seeds": v4_seeds,
                        "v5_rank_seeds": args.rank_seeds,
                    }
                ),
            )
            status.stage("prediction_frame", "SUCCESS", signals=len(signal_columns))

            status.stage("method_crossfit", "RUNNING")
            crossfit_metrics, crossfit_scores = crossfit_methods(
                prediction_frame,
                signal_columns,
                family_map,
                args.selection_folds,
                args.methods,
                args,
            )
            method_summary = summarize_method_crossfit(
                crossfit_metrics,
                args.target_recall,
                args.minimum_precision_lift,
                args.max_alert_rate,
            )
            selected_method = select_best_method(method_summary)
            atomic_write_csv(args.output / "method_crossfit_metrics.csv", crossfit_metrics)
            atomic_write_csv(args.output / "method_crossfit_summary.csv", method_summary)
            atomic_write_npz(
                args.output / "method_crossfit_scores.npz",
                validation_index=crossfit_scores["validation_index"].to_numpy(dtype=np.int64),
                fold_id=crossfit_scores["fold_id"].to_numpy(dtype=np.int16),
                method=crossfit_scores["method"].astype(str).to_numpy(dtype="U64"),
                score=crossfit_scores["score"].to_numpy(dtype=np.float32),
            )
            status.stage("method_crossfit", "SUCCESS", selected_method=selected_method)

            status.stage("final_freeze", "RUNNING")
            final_fit = fit_final_method_and_policy(
                selected_method,
                prediction_frame,
                crossfit_scores,
                signal_columns,
                family_map,
                args.selection_folds,
                args,
            )
            selected_crossfit = crossfit_scores.loc[crossfit_scores["method"].eq(selected_method)].copy()
            fold_metrics, role_metrics, scored = evaluate_final_candidate(
                final_fit,
                prediction_frame,
                selected_crossfit,
                roles,
                args,
            )
            atomic_write_csv(args.output / "budget_candidate_metrics_by_fold.csv", fold_metrics)
            atomic_write_csv(args.output / "budget_candidate_metrics_by_role.csv", role_metrics)
            atomic_write_npz(
                args.output / "budget_candidate_predictions.npz",
                validation_index=scored["validation_index"].to_numpy(dtype=np.int64),
                fold_id=scored["fold_id"].to_numpy(dtype=np.int16),
                date=pd.to_datetime(scored["date"]).to_numpy(dtype="datetime64[ns]").astype(np.int64),
                target=scored["target"].to_numpy(dtype=np.uint8),
                score=scored["score"].to_numpy(dtype=np.float32),
                alert=scored["alert"].to_numpy(dtype=np.uint8),
            )
            tradeoff = build_tradeoff_curves(scored, roles, args.diagnostic_fractions, args)
            atomic_write_csv(args.output / "alert_budget_tradeoff_by_role.csv", tradeoff)
            scope = build_scope_metrics(scored.assign(fold_role=scored["fold_id"].map(lambda value: role_for_fold(int(value), roles))), args.scope_columns, args.minimum_scope_rows, args.minimum_scope_positives)
            atomic_write_csv(args.output / "alert_budget_scope_metrics.csv", scope)

            freeze_payload: dict[str, Any] = {
                "schema": "crashwatch_surge_alert_budget_freeze_v5",
                "selected_method": selected_method,
                "method_spec": final_fit.spec,
                "daily_budget_policy": final_fit.policy.to_dict(),
                "signals": signal_columns,
                "family_map": family_map,
                "v4_output": str(args.v4_output),
                "v4_recipes": v4_recipe_names,
                "v4_seeds": v4_seeds,
                "v5_rank_recipes": [recipe.name for recipe in rank_recipes],
                "v5_rank_seeds": args.rank_seeds,
                "selection_folds": args.selection_folds,
                "created_at": utc_now(),
            }
            if selected_method == "hard_negative_meta_lgb" and isinstance(final_fit.spec.get("model"), dict):
                freeze_payload["method_spec"] = relativize_registry_paths(final_fit.spec, args.output)
            freeze_payload = with_payload_checksum(freeze_payload)
            atomic_write_json(args.output / "ALERT_BUDGET_FREEZE_V5.json", freeze_payload)
            status.stage("final_freeze", "SUCCESS", method=selected_method, policy=final_fit.policy.to_dict())

            status.stage("production", "RUNNING")
            if not args.skip_production_training:
                train_ranker_production_registry(
                    rank_recipes,
                    args.rank_seeds,
                    bundle,
                    profiles,
                    rank_metrics,
                    args,
                    preflight,
                )
            status.stage("production", "SUCCESS", skipped=bool(args.skip_production_training))

            report, report_text = build_gap_report(
                fold_metrics,
                role_metrics,
                final_fit.policy,
                tradeoff,
                args,
            )
            atomic_write_json(args.output / "BUDGET_PERFORMANCE_GAP_REPORT.json", report)
            atomic_write_text(args.output / "BUDGET_PERFORMANCE_GAP_REPORT_KO.md", report_text)
            recommendation = with_payload_checksum(
                {
                    "schema": "crashwatch_surge_alert_budget_recommendation_v5",
                    "status": report["status"],
                    "selected_method": selected_method,
                    "policy": final_fit.policy.to_dict(),
                    "selection_crossfit_summary": method_summary.to_dict(orient="records"),
                    "role_metrics": role_metrics.to_dict(orient="records"),
                    "hard_constraints": {
                        "target_recall": args.target_recall,
                        "minimum_precision_lift": args.minimum_precision_lift,
                        "maximum_alert_rate": args.max_alert_rate,
                        "maximum_alerts_per_day": args.max_alerts_per_day,
                    },
                    "next_gate": "동결 정책을 2026-06-22 이후 신규 미래 데이터에 한 번만 적용",
                }
            )
            atomic_write_json(args.output / "FINAL_RECOMMENDATION_V5.json", recommendation)

            inventory_paths = [
                path
                for path in args.output.rglob("*")
                if path.is_file()
                and path.name not in {
                    "RUN_STATUS.json",
                    "OUTPUT_INVENTORY.json",
                    "VERIFICATION_REPORT_V5.json",
                    ".surge_alert_budget_v5.lock",
                }
            ]
            inventory = compute_output_inventory(args.output, inventory_paths)
            atomic_write_json(args.output / "OUTPUT_INVENTORY.json", with_payload_checksum({"files": inventory}))
            final_status = str(report["status"])
            status.success(
                selected_method=selected_method,
                gate_status=final_status,
                policy=final_fit.policy.to_dict(),
            )
            print("=" * 72)
            print("CrashWatch Surge Alert Budget V5")
            print("=" * 72)
            print(f"Selected method        : {selected_method}")
            print(f"Daily fraction         : {final_fit.policy.daily_fraction:.1%}")
            print(f"Max alerts per day     : {final_fit.policy.max_alerts_per_day}")
            print(f"Selection gate pass    : {final_fit.policy.gate_pass}")
            print(f"Final gate status      : {final_status}")
            print(f"Output                 : {args.output}")
            print("=" * 72)
        except Exception as exc:
            status.failure(exc)
            raise


if __name__ == "__main__":
    main()
