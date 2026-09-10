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
    _median_iteration,
    fit_predict_base_family,
    fold_indices,
    load_profiles,
    preflight_families,
    prepare_data_bundle,
    resolve_model_device,
)
from surge_model_zoo_common import (
    DATE_COLUMN,
    TARGET_COLUMN,
    TARGET_VALID_COLUMN,
    TICKER_COLUMN,
    FileLock,
    FoldSpec,
    RunStatus,
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
    parse_bool_series,
    payload_checksum_is_valid,
    read_table,
    role_for_fold,
    sha256_bytes,
    sha256_file,
    stable_json_bytes,
    table_columns,
    utc_now,
    validate_folds,
    with_payload_checksum,
)
from surge_model_zoo_deployment import (
    predict_single_saved_model,
    relativize_registry_paths,
    save_catboost_model,
    save_lightgbm_model,
    save_xgboost_model,
)
from surge_precision_common_v6 import (
    CalibratorSpec,
    PrecisionPolicy,
    apply_calibrator,
    apply_meta_feature_spec,
    apply_precision_policy,
    beta_posterior_lower_bound,
    build_meta_feature_frame,
    build_precision_curve,
    build_scope_metrics,
    date_block_bootstrap_precision_lcb,
    datewise_rank_normalize,
    ensure_unit_interval,
    evaluate_alerts,
    fit_best_calibrator,
    forward_training_folds,
    maximum_recall_at_precision,
    rank_before_seed_average,
    select_global_precision_policy,
    select_scope_precision_policy,
    sigmoid,
    simulate_delayed_online_threshold,
)


RUNNER_SCHEMA = "crashwatch_surge_precision70_runner_v6"


@dataclass(frozen=True)
class PrecisionRecipe:
    name: str
    family: str
    profile: str
    target_variant: str
    train_policy: str
    false_positive_cost: float
    params: dict[str, Any]
    time_decay_half_life_days: float | None = None
    rolling_days: int | None = None
    seed_aggregation: str = "probability_mean"

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "PrecisionRecipe":
        family = str(record["family"])
        if family not in {"lightgbm", "xgboost", "catboost"}:
            raise ValueError(f"지원하지 않는 precision family: {family}")
        aggregation = str(record.get("seed_aggregation", "probability_mean"))
        if aggregation not in {"probability_mean", "date_rank_mean"}:
            raise ValueError(f"지원하지 않는 seed aggregation: {aggregation}")
        return cls(
            name=str(record["name"]),
            family=family,
            profile=str(record["profile"]),
            target_variant=str(record.get("target_variant", "surge_d3")),
            train_policy=str(record.get("train_policy", "expanding")),
            false_positive_cost=float(record.get("false_positive_cost", 1.0)),
            params=dict(record.get("params", {})),
            time_decay_half_life_days=(
                float(record["time_decay_half_life_days"])
                if record.get("time_decay_half_life_days") is not None
                else None
            ),
            rolling_days=(
                int(record["rolling_days"])
                if record.get("rolling_days") is not None
                else None
            ),
            seed_aggregation=aggregation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "profile": self.profile,
            "target_variant": self.target_variant,
            "train_policy": self.train_policy,
            "false_positive_cost": self.false_positive_cost,
            "params": self.params,
            "time_decay_half_life_days": self.time_decay_half_life_days,
            "rolling_days": self.rolling_days,
            "seed_aggregation": self.seed_aggregation,
        }


@dataclass
class FittedMethod:
    method: str
    feature_spec: dict[str, Any]
    model_kind: str
    model: Any | None
    model_parameters: dict[str, Any]
    calibrator: CalibratorSpec
    negative_cost: float


@dataclass(frozen=True)
class MethodCandidate:
    name: str
    kind: str
    negative_cost: float = 1.0
    uncertainty_z: float = 1.0


METHOD_CANDIDATES: tuple[MethodCandidate, ...] = (
    MethodCandidate("calibrated_recipe_mean", "recipe_mean"),
    MethodCandidate("calibrated_family_mean", "family_mean"),
    MethodCandidate("surge_safety_product", "safety_product"),
    MethodCandidate("conservative_q25_safety", "conservative_q25"),
    MethodCandidate("strong_surge_safety", "strong_safety"),
    MethodCandidate("consensus_lcb_z1", "consensus_lcb", uncertainty_z=1.0),
    MethodCandidate("consensus_lcb_z2", "consensus_lcb", uncertainty_z=2.0),
    MethodCandidate("logistic_precision_c2", "logistic", negative_cost=2.0),
    MethodCandidate("logistic_precision_c4", "logistic", negative_cost=4.0),
    MethodCandidate("lgb_precision_verifier_c2", "lightgbm", negative_cost=2.0),
    MethodCandidate("lgb_precision_verifier_c4", "lightgbm", negative_cost=4.0),
    MethodCandidate("xgb_precision_verifier_c2", "xgboost", negative_cost=2.0),
    MethodCandidate("xgb_precision_verifier_c4", "xgboost", negative_cost=4.0),
)


def parse_int_list(text: str) -> list[int]:
    values = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("정수 목록이 비어 있습니다")
    return list(dict.fromkeys(values))


def parse_str_list(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def resolve_paths(args: argparse.Namespace) -> None:
    root = args.package_root.resolve()
    args.package_root = root
    args.v4_output = (args.v4_output or root / "outputs" / "surge_model_zoo_v4").resolve()
    args.v5_output = (args.v5_output or root / "outputs" / "surge_alert_budget_v5").resolve()
    args.output = (args.output or root / "outputs" / "surge_precision70_v6").resolve()
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
    if args.precision_recipes is None:
        args.precision_recipes = Path(__file__).resolve().parent / "default_precision_recipes_v6.json"
    args.folds = Path(args.folds).resolve()
    args.profiles = Path(args.profiles).resolve()
    args.precision_recipes = Path(args.precision_recipes).resolve()


def require_inputs(args: argparse.Namespace) -> None:
    required = {
        "dataset": args.dataset,
        "target_sidecar": args.target_sidecar,
        "folds": args.folds,
        "profiles": args.profiles,
        "precision_recipes": args.precision_recipes,
        "v4_finalist_registry": args.v4_output / "FINALIST_REGISTRY.json",
        "v4_ensemble_freeze": args.v4_output / "ENSEMBLE_FREEZE.json",
        "v5_signal_registry": args.v5_output / "SIGNAL_REGISTRY.json",
    }
    missing = {key: str(path) for key, path in required.items() if not path.exists()}
    if missing:
        raise FileNotFoundError(json.dumps(missing, ensure_ascii=False, indent=2))


def load_precision_recipes(
    path: Path,
    quick: bool,
) -> tuple[dict[str, Any], list[PrecisionRecipe]]:
    payload = load_json(path)
    records = payload.get("recipes", [])
    recipes = [PrecisionRecipe.from_dict(record) for record in records]
    if quick:
        quick_names = {str(value) for value in payload.get("quick_recipe_names", [])}
        recipes = [recipe for recipe in recipes if recipe.name in quick_names]
    if not recipes:
        raise ValueError("실행할 V6 precision recipe가 없습니다")
    names = [recipe.name for recipe in recipes]
    if len(names) != len(set(names)):
        raise ValueError("V6 recipe name 중복")
    return payload, recipes


def load_existing_recipe_registry(args: argparse.Namespace) -> tuple[list[V4Recipe], list[str], list[int], dict[str, Any]]:
    finalist = load_json(args.v4_output / "FINALIST_REGISTRY.json")
    freeze = load_json(args.v4_output / "ENSEMBLE_FREEZE.json")
    if not payload_checksum_is_valid(freeze):
        raise ValueError("V4 ENSEMBLE_FREEZE checksum mismatch")
    all_recipes = {str(record["name"]): V4Recipe.from_dict(record) for record in finalist.get("finalists", [])}
    recipe_names = [str(value) for value in freeze.get("recipes", [])]
    seeds = [int(value) for value in freeze.get("seeds", [])]
    missing = [name for name in recipe_names if name not in all_recipes]
    if missing:
        raise KeyError(f"V4 finalist recipe 누락: {missing}")
    signal_registry = load_json(args.v5_output / "SIGNAL_REGISTRY.json")
    if not payload_checksum_is_valid(signal_registry):
        raise ValueError("V5 SIGNAL_REGISTRY checksum mismatch")
    return [all_recipes[name] for name in recipe_names], recipe_names, seeds, signal_registry


def _reuse_v4_matrix_cache(
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
        log(f"V4 matrix cache 재사용 실패: {type(exc).__name__}: {exc}")
        return None


def prepare_bundle(
    args: argparse.Namespace,
    profiles: Mapping[str, Sequence[str]],
    existing_recipes: Sequence[V4Recipe],
    precision_recipes: Sequence[PrecisionRecipe],
    folds: Sequence[FoldSpec],
) -> DataBundle:
    required: list[str] = []
    for recipe in existing_recipes:
        required.extend(profiles[recipe.profile])
        if recipe.direction_profile:
            required.extend(profiles[recipe.direction_profile])
    for recipe in precision_recipes:
        required.extend(profiles[recipe.profile])
    required = list(dict.fromkeys(required))
    if args.reuse_v4_matrix_cache:
        bundle = _reuse_v4_matrix_cache(args, required)
        if bundle is not None:
            log("V4 matrix cache 재사용")
            return bundle
    compatibility = list(existing_recipes)
    for recipe in precision_recipes:
        compatibility.append(
            V4Recipe(
                name=f"v6_compat__{recipe.name}",
                family=recipe.family,
                profile=recipe.profile,
                target_variant="surge_d3",
                train_policy=recipe.train_policy,
                positive_weight_mode="none",
                params=recipe.params,
                time_decay_half_life_days=recipe.time_decay_half_life_days,
                rolling_days=recipe.rolling_days,
            )
        )
    return prepare_data_bundle(args, profiles, compatibility, folds)


def augment_continuous_targets(bundle: DataBundle, args: argparse.Namespace) -> dict[str, Any]:
    columns = table_columns(args.target_sidecar)
    requested = [
        "source_row_id",
        args.date_column,
        args.ticker_column,
        "first_hit_day",
        "best_forward_return_3d",
        args.target_column,
        args.target_valid_column,
    ]
    requested = [value for value in requested if value in columns]
    sidecar = read_table(args.target_sidecar, requested)
    frame = bundle.frame.copy()
    if "source_row_id" in frame.columns and "source_row_id" in sidecar.columns:
        side = sidecar.drop_duplicates("source_row_id", keep="last").set_index("source_row_id")
        ids = frame["source_row_id"].to_numpy()
        best = pd.to_numeric(side.reindex(ids).get("best_forward_return_3d"), errors="coerce").to_numpy(dtype=np.float64)
        hit = pd.to_numeric(side.reindex(ids).get("first_hit_day"), errors="coerce").to_numpy(dtype=np.float64)
    else:
        keys = [args.date_column, args.ticker_column]
        left = frame[keys].copy()
        right = sidecar[keys + [value for value in ["best_forward_return_3d", "first_hit_day"] if value in sidecar]].copy()
        joined = left.merge(right, on=keys, how="left", validate="one_to_one")
        best = pd.to_numeric(joined.get("best_forward_return_3d"), errors="coerce").to_numpy(dtype=np.float64)
        hit = pd.to_numeric(joined.get("first_hit_day"), errors="coerce").to_numpy(dtype=np.float64)
    target_d3 = bundle.targets["surge_d3"].astype(np.uint8)
    bundle.targets["surge_strong7"] = ((target_d3 == 1) & np.isfinite(best) & (best >= 0.07 - 1e-12)).astype(np.uint8)
    bundle.targets["surge_strong10"] = ((target_d3 == 1) & np.isfinite(best) & (best >= 0.10 - 1e-12)).astype(np.uint8)
    bundle.targets["first_hit_day"] = hit
    bundle.targets["best_forward_return_3d"] = best
    return {
        "target_rows": int(len(best)),
        "strong7_count": int(np.sum(bundle.targets["surge_strong7"] == 1)),
        "strong10_count": int(np.sum(bundle.targets["surge_strong10"] == 1)),
        "best_forward_missing": int(np.sum(~np.isfinite(best))),
    }


def _episode_balance_weights(bundle: DataBundle, target_variant: str) -> np.ndarray:
    """Balance consecutive positive rows for the target actually trained by a recipe.

    A single market episode can label several adjacent decision rows.  The previous
    implementation always used ``surge_d3`` runs, which incorrectly reweighted
    crash and strong-surge specialists.  Each target variant now receives its own
    episode map.
    """
    if target_variant not in bundle.targets:
        raise KeyError(f"episode weight target 누락: {target_variant}")
    target = np.asarray(bundle.targets[target_variant], dtype=np.int8)
    tickers = bundle.frame[TICKER_COLUMN].astype("string").to_numpy(dtype=object)
    dates = bundle.dates
    weights = np.ones(len(target), dtype=np.float64)
    order = np.lexsort((dates.astype("datetime64[ns]").astype(np.int64), tickers.astype(str)))
    start = 0
    while start < len(order):
        ticker = tickers[order[start]]
        end = start + 1
        while end < len(order) and tickers[order[end]] == ticker:
            end += 1
        local = order[start:end]
        values = target[local]
        run_start = 0
        while run_start < len(local):
            if values[run_start] != 1:
                run_start += 1
                continue
            run_end = run_start + 1
            while run_end < len(local) and values[run_end] == 1:
                run_end += 1
            weights[local[run_start:run_end]] = 1.0 / max(1, run_end - run_start)
            run_start = run_end
        start = end
    return weights


def build_precision_weights(
    bundle: DataBundle,
    recipe: PrecisionRecipe,
    train_indices: np.ndarray,
    episode_weights: np.ndarray,
) -> np.ndarray:
    target = bundle.targets[recipe.target_variant].astype(np.int8)
    y = target[train_indices]
    weights = np.ones(len(train_indices), dtype=np.float64)
    if recipe.false_positive_cost > 0:
        weights[y == 0] *= float(recipe.false_positive_cost)
    weights *= episode_weights[train_indices]
    if recipe.time_decay_half_life_days is not None and recipe.time_decay_half_life_days > 0:
        dates = pd.to_datetime(pd.Series(bundle.dates[train_indices]), errors="coerce")
        latest = dates.max()
        age_days = (latest - dates).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
        weights *= np.power(0.5, np.maximum(age_days, 0.0) / float(recipe.time_decay_half_life_days))
    mean = float(np.mean(weights)) if len(weights) else 1.0
    if mean > 0:
        weights /= mean
    return weights.astype(np.float32)


def specialist_task_paths(
    output: Path,
    recipe: PrecisionRecipe,
    fold_id: int,
    seed: int,
) -> tuple[Path, Path]:
    directory = output / "specialist_task_cache" / recipe.name / f"seed_{seed}"
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


def fit_precision_recipe(
    recipe: PrecisionRecipe,
    matrix: np.ndarray,
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    seed: int,
    args: argparse.Namespace,
    device: str,
    episode_weights: np.ndarray,
    fixed_iterations: int | None,
) -> tuple[np.ndarray, int]:
    train_indices = apply_training_window(train_indices, bundle.dates, recipe.train_policy, recipe.rolling_days)
    if len(train_indices) < 200:
        raise ValueError(f"{recipe.name}: 학습 행 부족")
    features = profiles[recipe.profile]
    feature_indices = np.asarray([bundle.feature_to_index[value] for value in features], dtype=np.int64)
    x_train = np.asarray(matrix[np.ix_(train_indices, feature_indices)], dtype=np.float32)
    x_valid = np.asarray(matrix[np.ix_(valid_indices, feature_indices)], dtype=np.float32)
    y_all = bundle.targets[recipe.target_variant].astype(np.uint8)
    y_train = y_all[train_indices]
    y_valid = y_all[valid_indices]
    if len(np.unique(y_train)) < 2:
        probability = float(np.mean(y_train)) if len(y_train) else 0.0
        return np.full(len(valid_indices), probability, dtype=np.float64), 1
    weights = build_precision_weights(bundle, recipe, train_indices, episode_weights)
    fit = fit_predict_base_family(
        recipe.family,
        x_train,
        y_train,
        weights,
        x_valid,
        y_valid if len(np.unique(y_valid)) >= 2 else None,
        recipe.params,
        int(seed),
        args,
        device,
        fixed_iterations,
    )
    return np.asarray(fit.prediction, dtype=np.float64), int(fit.best_iteration)


def run_specialist_task(
    recipe: PrecisionRecipe,
    fold: FoldSpec,
    seed: int,
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    episode_weights: np.ndarray,
) -> dict[str, Any]:
    result_path, prediction_path = specialist_task_paths(args.output, recipe, fold.fold_id, seed)
    identity = {
        "schema": "crashwatch_surge_precision_specialist_task_v6",
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
        "common_sha256": sha256_file(Path(__file__).resolve().parent / "surge_precision_common_v6.py"),
    }
    identity_hash = sha256_bytes(stable_json_bytes(identity))
    if args.resume and _task_valid(result_path, prediction_path, identity_hash):
        return load_json(result_path)
    started = time.monotonic()
    matrix = np.load(bundle.matrix_path, mmap_mode="r")
    outer_train, outer_valid = fold_indices(bundle, fold)
    windows = build_inner_windows(
        bundle.dates,
        outer_train,
        validation_days=args.inner_validation_days,
        purge_days=args.inner_purge_days,
        windows=args.inner_windows,
        step_days=args.inner_step_days,
        minimum_train_days=args.minimum_inner_train_days,
    )
    iterations: list[int] = []
    device = resolve_model_device(recipe.family, preflight)
    for window_id, (inner_train, inner_valid) in enumerate(windows):
        try:
            _, best = fit_precision_recipe(
                recipe,
                matrix,
                bundle,
                profiles,
                inner_train,
                inner_valid,
                deterministic_seed(seed, recipe.name, fold.fold_id, window_id),
                args,
                device,
                episode_weights,
                fixed_iterations=None,
            )
            iterations.append(int(best))
        except Exception as exc:
            log(f"inner tuning skip {recipe.name} fold={fold.fold_id} window={window_id}: {type(exc).__name__}")
    effective = int(_median_iteration(iterations, args.minimum_iterations, args.maximum_iterations))
    prediction, _ = fit_precision_recipe(
        recipe,
        matrix,
        bundle,
        profiles,
        outer_train,
        outer_valid,
        deterministic_seed(seed, recipe.name, fold.fold_id, "final"),
        args,
        device,
        episode_weights,
        fixed_iterations=effective,
    )
    target_d3 = bundle.targets["surge_d3"][outer_valid].astype(np.uint8)
    atomic_write_npz(
        prediction_path,
        validation_indices=outer_valid.astype(np.int64),
        dates=bundle.dates[outer_valid].astype("datetime64[ns]").astype(np.int64),
        target=target_d3,
        raw_prediction=prediction.astype(np.float32),
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
            "best_iterations_by_window": iterations,
            "effective_iteration": effective,
            "prediction_path": str(prediction_path),
            "prediction_sha256": sha256_file(prediction_path),
            "elapsed_seconds": float(time.monotonic() - started),
            "completed_at": utc_now(),
        }
    )
    atomic_write_json(result_path, payload)
    return payload


def execute_specialist_tasks(
    recipes: Sequence[PrecisionRecipe],
    folds: Sequence[FoldSpec],
    seeds: Sequence[int],
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
) -> pd.DataFrame:
    episode_weights_by_target = {
        target_variant: _episode_balance_weights(bundle, target_variant)
        for target_variant in sorted({recipe.target_variant for recipe in recipes})
    }
    total = len(recipes) * len(folds) * len(seeds)
    completed = 0
    lock = threading.Lock()

    def run_group(group: Sequence[PrecisionRecipe]) -> list[dict[str, Any]]:
        nonlocal completed
        rows: list[dict[str, Any]] = []
        for recipe in group:
            for seed in seeds:
                for fold in folds:
                    payload = run_specialist_task(
                        recipe,
                        fold,
                        seed,
                        bundle,
                        profiles,
                        roles,
                        args,
                        preflight,
                        episode_weights_by_target[recipe.target_variant],
                    )
                    rows.append(
                        {
                            "recipe": recipe.name,
                            "family": recipe.family,
                            "profile": recipe.profile,
                            "target_variant": recipe.target_variant,
                            "train_policy": recipe.train_policy,
                            "false_positive_cost": recipe.false_positive_cost,
                            "fold_id": int(fold.fold_id),
                            "fold_role": role_for_fold(fold.fold_id, roles),
                            "seed": int(seed),
                            "effective_iteration": int(payload["effective_iteration"]),
                            "elapsed_seconds": float(payload["elapsed_seconds"]),
                        }
                    )
                    with lock:
                        completed += 1
                        current = completed
                    if current == 1 or current % 10 == 0 or current == total:
                        log(f"precision specialist {current}/{total}: {recipe.name} seed={seed} fold={fold.fold_id}")
        return rows

    records: list[dict[str, Any]] = []
    if args.family_parallel:
        families = list(dict.fromkeys(recipe.family for recipe in recipes))
        groups = [[recipe for recipe in recipes if recipe.family == family] for family in families]
        with ThreadPoolExecutor(max_workers=len(groups), thread_name_prefix="precision-family") as executor:
            futures = [executor.submit(run_group, group) for group in groups]
            for future in as_completed(futures):
                records.extend(future.result())
    else:
        records = run_group(recipes)
    result = pd.DataFrame(records).sort_values(["recipe", "seed", "fold_id"], kind="mergesort")
    atomic_write_csv(args.output / "precision_specialist_seed_fold_metrics.csv", result)
    return result


def load_npz_checked(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    arrays = np.load(path, allow_pickle=False)
    return {name: np.asarray(arrays[name]) for name in arrays.files}


def build_base_validation_frame(
    bundle: DataBundle,
    folds: Sequence[FoldSpec],
    roles: Mapping[str, Sequence[int]],
    scope_columns: Sequence[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    target_variants = [
        value
        for value in ["surge_d3", "crash_d3", "surge_strong7", "surge_strong10"]
        if value in bundle.targets
    ]
    for fold in folds:
        _, indices = fold_indices(bundle, fold)
        part = pd.DataFrame(
            {
                "validation_index": indices,
                "fold_id": int(fold.fold_id),
                "fold_role": role_for_fold(fold.fold_id, roles),
                "date": bundle.dates[indices],
                "target": bundle.targets["surge_d3"][indices].astype(np.uint8),
            }
        )
        for target_variant in target_variants:
            part[f"target__{target_variant}"] = (
                bundle.targets[target_variant][indices].astype(np.uint8)
            )
        for column in [TICKER_COLUMN, *scope_columns]:
            if column in bundle.frame.columns:
                part[column] = bundle.frame.iloc[indices][column].to_numpy()
        parts.append(part)
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["fold_id", "validation_index"], kind="mergesort")
        .reset_index(drop=True)
    )


def _v4_task_prediction_path(v4_output: Path, recipe: str, seed: int, fold_id: int) -> Path:
    return v4_output / "task_cache" / recipe / f"seed_{seed}" / f"fold_{fold_id}.npz"


def _v5_rank_prediction_path(v5_output: Path, recipe: str, seed: int, fold_id: int) -> Path:
    return v5_output / "rank_task_cache" / recipe / f"seed_{seed}" / f"fold_{fold_id}.npz"


def build_signal_frame(
    args: argparse.Namespace,
    bundle: DataBundle,
    folds: Sequence[FoldSpec],
    roles: Mapping[str, Sequence[int]],
    v4_recipe_names: Sequence[str],
    v4_seeds: Sequence[int],
    v5_signal_registry: Mapping[str, Any],
    precision_recipes: Sequence[PrecisionRecipe],
    precision_seeds: Sequence[int],
) -> tuple[
    pd.DataFrame,
    dict[str, str],
    list[str],
    pd.DataFrame,
    dict[str, str],
    dict[str, str],
]:
    frame = build_base_validation_frame(
        bundle,
        folds,
        roles,
        args.scope_columns,
    )
    family_map = {
        str(key): str(value)
        for key, value in v5_signal_registry.get("family_map", {}).items()
    }
    signal_columns: list[str] = []
    seed_audit: list[dict[str, Any]] = []
    calibration_target_map: dict[str, str] = {}
    signal_roles: dict[str, str] = {}

    def register_signal(
        signal: str,
        family: str,
        role: str,
        calibration_target: str,
    ) -> None:
        family_map[signal] = family
        signal_roles[signal] = role
        calibration_target_map[signal] = calibration_target
        signal_columns.append(signal)

    for recipe in v4_recipe_names:
        values_by_index: dict[int, float] = {}
        std_by_index: dict[int, float] = {}
        for fold in folds:
            _, expected = fold_indices(bundle, fold)
            predictions: list[np.ndarray] = []
            for seed in v4_seeds:
                arrays = load_npz_checked(
                    _v4_task_prediction_path(
                        args.v4_output,
                        recipe,
                        seed,
                        fold.fold_id,
                    )
                )
                indices = arrays["validation_indices"].astype(np.int64)
                if not np.array_equal(indices, expected):
                    raise ValueError(
                        f"V4 index mismatch: {recipe} seed={seed} fold={fold.fold_id}"
                    )
                key = (
                    "calibrated_prediction"
                    if "calibrated_prediction" in arrays
                    else "raw_prediction"
                )
                predictions.append(arrays[key].astype(np.float64))
            matrix = np.vstack(predictions)
            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
            values_by_index.update(
                {
                    int(index): float(value)
                    for index, value in zip(expected, mean)
                }
            )
            std_by_index.update(
                {
                    int(index): float(value)
                    for index, value in zip(expected, std)
                }
            )
            seed_audit.append(
                {
                    "signal": recipe,
                    "fold_id": int(fold.fold_id),
                    "aggregation": "probability_mean",
                    "seed_count": len(predictions),
                    "mean_seed_std": float(np.mean(std)),
                }
            )
        frame[recipe] = frame["validation_index"].map(values_by_index).to_numpy(
            dtype=np.float64
        )
        agreement_name = f"{recipe}__seed_agreement"
        frame[agreement_name] = np.clip(
            1.0
            - frame["validation_index"].map(std_by_index).to_numpy(dtype=np.float64),
            0.0,
            1.0,
        )
        semantic_family = str(family_map.get(recipe, "v4_unknown"))
        role = (
            "direction_up"
            if "two_stage" in recipe.lower() or "direction" in recipe.lower()
            else "surge"
        )
        register_signal(recipe, semantic_family, role, "surge_d3")
        register_signal(
            agreement_name,
            "seed_agreement",
            "agreement",
            "identity",
        )

    v5_recipe_names = [
        str(value)
        for value in v5_signal_registry.get("signals", [])
        if str(value) not in set(v4_recipe_names)
        and not str(value).endswith("__seed_agreement")
    ]
    v5_seeds = [
        int(value) for value in v5_signal_registry.get("v5_rank_seeds", [])
    ]
    for recipe in v5_recipe_names:
        values_by_index: dict[int, float] = {}
        agreement_by_index: dict[int, float] = {}
        for fold in folds:
            _, expected = fold_indices(bundle, fold)
            predictions: list[np.ndarray] = []
            dates = bundle.dates[expected]
            for seed in v5_seeds:
                arrays = load_npz_checked(
                    _v5_rank_prediction_path(
                        args.v5_output,
                        recipe,
                        seed,
                        fold.fold_id,
                    )
                )
                indices = arrays["validation_indices"].astype(np.int64)
                if not np.array_equal(indices, expected):
                    raise ValueError(
                        f"V5 index mismatch: {recipe} seed={seed} fold={fold.fold_id}"
                    )
                predictions.append(arrays["raw_prediction"].astype(np.float64))
            rank_mean, rank_std = rank_before_seed_average(predictions, dates)
            values_by_index.update(
                {
                    int(index): float(value)
                    for index, value in zip(expected, rank_mean)
                }
            )
            agreement_by_index.update(
                {
                    int(index): float(1.0 - value)
                    for index, value in zip(expected, rank_std)
                }
            )
            seed_audit.append(
                {
                    "signal": recipe,
                    "fold_id": int(fold.fold_id),
                    "aggregation": "date_rank_before_seed_average",
                    "seed_count": len(predictions),
                    "mean_seed_std": float(np.mean(rank_std)),
                }
            )
        frame[recipe] = frame["validation_index"].map(values_by_index).to_numpy(
            dtype=np.float64
        )
        agreement_name = f"{recipe}__seed_agreement"
        frame[agreement_name] = np.clip(
            frame["validation_index"].map(agreement_by_index).to_numpy(
                dtype=np.float64
            ),
            0.0,
            1.0,
        )
        register_signal(
            recipe,
            str(family_map.get(recipe, "v5_ranker")),
            "surge",
            "surge_d3",
        )
        register_signal(
            agreement_name,
            "seed_agreement",
            "agreement",
            "identity",
        )

    for recipe in precision_recipes:
        values_by_index: dict[int, float] = {}
        agreement_by_index: dict[int, float] = {}
        for fold in folds:
            _, expected = fold_indices(bundle, fold)
            dates = bundle.dates[expected]
            predictions: list[np.ndarray] = []
            for seed in precision_seeds:
                _, path = specialist_task_paths(
                    args.output,
                    recipe,
                    fold.fold_id,
                    seed,
                )
                arrays = load_npz_checked(path)
                indices = arrays["validation_indices"].astype(np.int64)
                if not np.array_equal(indices, expected):
                    raise ValueError(
                        f"V6 index mismatch: {recipe.name} seed={seed} fold={fold.fold_id}"
                    )
                predictions.append(arrays["raw_prediction"].astype(np.float64))
            if recipe.seed_aggregation == "date_rank_mean":
                mean, std = rank_before_seed_average(predictions, dates)
                aggregation = "date_rank_before_seed_average"
            else:
                matrix = np.vstack(predictions)
                mean = np.nanmean(matrix, axis=0)
                std = np.nanstd(matrix, axis=0)
                aggregation = "probability_mean"
            values_by_index.update(
                {
                    int(index): float(value)
                    for index, value in zip(expected, mean)
                }
            )
            agreement_by_index.update(
                {
                    int(index): float(1.0 - value)
                    for index, value in zip(expected, std)
                }
            )
            seed_audit.append(
                {
                    "signal": recipe.name,
                    "fold_id": int(fold.fold_id),
                    "aggregation": aggregation,
                    "seed_count": len(predictions),
                    "mean_seed_std": float(np.mean(std)),
                }
            )
        frame[recipe.name] = frame["validation_index"].map(
            values_by_index
        ).to_numpy(dtype=np.float64)
        agreement_name = f"{recipe.name}__seed_agreement"
        frame[agreement_name] = np.clip(
            frame["validation_index"].map(agreement_by_index).to_numpy(
                dtype=np.float64
            ),
            0.0,
            1.0,
        )
        if recipe.target_variant == "crash_d3":
            semantic_family = f"{recipe.family}_crash"
            role = "crash"
            calibration_target = "crash_d3"
        elif recipe.target_variant in {"surge_strong7", "surge_strong10"}:
            semantic_family = f"{recipe.family}_{recipe.target_variant}"
            role = "strong_surge"
            # Map specialist confidence onto the final +5% target so all
            # positive-oriented calibrated signals share the same meaning.
            calibration_target = "surge_d3"
        else:
            semantic_family = recipe.family
            role = "surge"
            calibration_target = "surge_d3"
        register_signal(
            recipe.name,
            semantic_family,
            role,
            calibration_target,
        )
        register_signal(
            agreement_name,
            "seed_agreement",
            "agreement",
            "identity",
        )

    signal_columns = list(dict.fromkeys(signal_columns))
    if frame[signal_columns].isna().any().any():
        missing_signal = frame[signal_columns].isna().sum().sort_values(
            ascending=False
        )
        raise ValueError(
            f"signal frame NaN: "
            f"{missing_signal[missing_signal > 0].head(20).to_dict()}"
        )
    return (
        frame,
        family_map,
        signal_columns,
        pd.DataFrame(seed_audit),
        calibration_target_map,
        signal_roles,
    )



def calibrate_signals_forward(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    selection_folds: Sequence[int],
    args: argparse.Namespace,
    calibration_target_map: Mapping[str, str],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Calibrate every signal against its own semantic target.

    Surge/strong-surge models are mapped to the final D+3 surge target, crash
    models are calibrated to the D+3 crash target, and seed-agreement signals
    remain identity diagnostics.  This prevents a crash probability from being
    accidentally inverted by fitting it directly against the surge label.
    """

    calibrated = frame.copy()
    audit: list[dict[str, Any]] = []
    final_specs: dict[str, Any] = {}
    selection_set = {int(value) for value in selection_folds}
    all_selection = frame.loc[frame["fold_id"].isin(selection_set)].copy()

    def target_column_for(signal: str) -> tuple[str, str]:
        target_name = str(calibration_target_map.get(signal, "surge_d3"))
        if target_name == "identity":
            return target_name, ""
        column = "target" if target_name == "surge_d3" else f"target__{target_name}"
        if column not in frame.columns:
            raise KeyError(
                f"signal calibration target 누락: signal={signal}, "
                f"target={target_name}, column={column}"
            )
        return target_name, column

    for signal in signal_columns:
        if signal not in frame.columns:
            raise KeyError(f"signal calibration 입력 누락: {signal}")
        target_name, target_column = target_column_for(str(signal))
        output = np.full(len(frame), np.nan, dtype=np.float64)

        if target_name == "identity":
            raw = ensure_unit_interval(
                frame[signal].to_numpy(dtype=np.float64),
                frame["date"].to_numpy(dtype="datetime64[ns]"),
                "auto",
            )
            output[:] = raw
            spec = CalibratorSpec("identity", {}, "unit_interval")
            final_specs[str(signal)] = spec.to_dict()
            calibrated[signal] = np.clip(output, 1e-7, 1.0 - 1e-7)
            for fold_id in sorted(frame["fold_id"].unique()):
                audit.append(
                    {
                        "signal": signal,
                        "signal_role": "agreement",
                        "calibration_target": "identity",
                        "fold_id": int(fold_id),
                        "train_folds": "",
                        "calibrator": "identity",
                        "candidate_count": 1,
                    }
                )
            continue

        for fold_id in sorted(frame["fold_id"].unique()):
            valid_mask = frame["fold_id"].eq(fold_id).to_numpy()
            if int(fold_id) in selection_set:
                train_folds = forward_training_folds(selection_folds, int(fold_id))
                if not train_folds:
                    # The first selection fold is a strict temporal warm-up.
                    output[valid_mask] = ensure_unit_interval(
                        frame.loc[valid_mask, signal].to_numpy(dtype=np.float64),
                        frame.loc[valid_mask, "date"].to_numpy(dtype="datetime64[ns]"),
                        "auto",
                    )
                    audit.append(
                        {
                            "signal": signal,
                            "calibration_target": target_name,
                            "fold_id": int(fold_id),
                            "train_folds": "",
                            "calibrator": "warmup_identity",
                            "candidate_count": 1,
                        }
                    )
                    continue
                train_part = frame.loc[frame["fold_id"].isin(train_folds)]
            else:
                train_folds = sorted(selection_set)
                train_part = all_selection

            valid_target = train_part[target_column].isin([0, 1])
            train_part = train_part.loc[valid_target]
            if train_part.empty:
                raise ValueError(
                    f"signal calibration target rows 없음: {signal}/{target_name}"
                )
            raw_train = ensure_unit_interval(
                train_part[signal].to_numpy(dtype=np.float64),
                train_part["date"].to_numpy(dtype="datetime64[ns]"),
                "auto",
            )
            spec, candidates = fit_best_calibrator(
                train_part[target_column].to_numpy(dtype=np.int8),
                raw_train,
                train_part["date"].to_numpy(dtype="datetime64[ns]"),
                source_score_kind="unit_interval",
                minimum_rows=args.minimum_calibration_rows,
            )
            raw_valid = ensure_unit_interval(
                frame.loc[valid_mask, signal].to_numpy(dtype=np.float64),
                frame.loc[valid_mask, "date"].to_numpy(dtype="datetime64[ns]"),
                "auto",
            )
            output[valid_mask] = apply_calibrator(spec, raw_valid)
            audit.append(
                {
                    "signal": signal,
                    "calibration_target": target_name,
                    "fold_id": int(fold_id),
                    "train_folds": ",".join(str(value) for value in train_folds),
                    "calibrator": spec.kind,
                    "candidate_count": int(len(candidates)),
                }
            )

        final_part = all_selection.loc[all_selection[target_column].isin([0, 1])]
        final_raw = ensure_unit_interval(
            final_part[signal].to_numpy(dtype=np.float64),
            final_part["date"].to_numpy(dtype="datetime64[ns]"),
            "auto",
        )
        final_spec, candidates = fit_best_calibrator(
            final_part[target_column].to_numpy(dtype=np.int8),
            final_raw,
            final_part["date"].to_numpy(dtype="datetime64[ns]"),
            source_score_kind="unit_interval",
            minimum_rows=args.minimum_calibration_rows,
        )
        final_specs[str(signal)] = final_spec.to_dict()
        calibrated[signal] = np.clip(output, 1e-7, 1.0 - 1e-7)
        for _, row in candidates.iterrows():
            audit.append(
                {
                    "signal": signal,
                    "calibration_target": target_name,
                    "fold_id": "FINAL_SELECTION",
                    "train_folds": ",".join(str(value) for value in sorted(selection_set)),
                    "calibrator": row["kind"],
                    "candidate_brier": row.get("brier"),
                    "candidate_logloss": row.get("logloss"),
                    "candidate_status": row.get("status"),
                    "selected": bool(str(row["kind"]) == final_spec.kind),
                }
            )
    return calibrated, final_specs, pd.DataFrame(audit)

def _sample_weights_for_meta(target: np.ndarray, negative_cost: float, dates: np.ndarray | None = None, half_life_days: float | None = None) -> np.ndarray:
    y = np.asarray(target, dtype=np.int8)
    weights = np.ones(len(y), dtype=np.float64)
    weights[y == 0] *= float(negative_cost)
    if dates is not None and half_life_days is not None and half_life_days > 0:
        parsed = pd.to_datetime(pd.Series(dates), errors="coerce")
        latest = parsed.max()
        age = (latest - parsed).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
        weights *= np.power(0.5, np.maximum(age, 0.0) / float(half_life_days))
    mean = float(np.mean(weights)) if len(weights) else 1.0
    if mean > 0:
        weights /= mean
    return weights.astype(np.float32)



def fit_method(
    candidate: MethodCandidate,
    train_frame: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    args: argparse.Namespace,
    seed: int,
) -> FittedMethod:
    meta, feature_spec = build_meta_feature_frame(
        train_frame,
        signal_columns,
        family_map,
        date_column="date",
        scope_columns=args.scope_columns,
    )
    target = train_frame["target"].to_numpy(dtype=np.uint8)
    dates = train_frame["date"].to_numpy(dtype="datetime64[ns]")

    def calibrated_simple(
        raw: np.ndarray,
        model_kind: str,
        parameters: Mapping[str, Any] | None = None,
        score_kind: str = "unit_interval",
    ) -> FittedMethod:
        values = np.asarray(raw, dtype=np.float64)
        if score_kind == "rank":
            values = ensure_unit_interval(values, dates, "rank")
        else:
            values = np.clip(values, 1e-7, 1.0 - 1e-7)
        calibrator, _ = fit_best_calibrator(
            target,
            values,
            dates,
            source_score_kind="unit_interval",
            minimum_rows=args.minimum_calibration_rows,
        )
        return FittedMethod(
            candidate.name,
            feature_spec,
            model_kind,
            None,
            dict(parameters or {}),
            calibrator,
            candidate.negative_cost,
        )

    if candidate.kind == "recipe_mean":
        return calibrated_simple(
            meta["ensemble_mean"].to_numpy(dtype=np.float64),
            "recipe_mean",
        )
    if candidate.kind == "family_mean":
        family_columns = [
            str(column)
            for column in feature_spec.get("positive_family_mean_columns", [])
            if str(column) in meta.columns
        ]
        if not family_columns:
            family_columns = ["ensemble_mean"]
        raw = meta[family_columns].mean(axis=1).to_numpy(dtype=np.float64)
        return calibrated_simple(
            raw,
            "family_mean",
            {"family_columns": family_columns},
        )
    if candidate.kind == "safety_product":
        return calibrated_simple(
            meta["surge_safety_product"].to_numpy(dtype=np.float64),
            "safety_product",
        )
    if candidate.kind == "conservative_q25":
        return calibrated_simple(
            meta["conservative_q25_safety"].to_numpy(dtype=np.float64),
            "conservative_q25",
        )
    if candidate.kind == "strong_safety":
        return calibrated_simple(
            meta["strong_safety_product"].to_numpy(dtype=np.float64),
            "strong_safety",
        )
    if candidate.kind == "consensus_lcb":
        raw = (
            meta["ensemble_mean"].to_numpy(dtype=np.float64)
            - candidate.uncertainty_z
            * meta["ensemble_std"].to_numpy(dtype=np.float64)
        )
        return calibrated_simple(
            raw,
            "consensus_lcb",
            {"uncertainty_z": candidate.uncertainty_z},
            score_kind="rank",
        )

    weights = _sample_weights_for_meta(
        target,
        candidate.negative_cost,
        dates,
        args.meta_time_decay_half_life_days,
    )
    x = meta.to_numpy(dtype=np.float32)
    if candidate.kind == "logistic":
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=args.meta_logistic_c,
            solver="lbfgs",
            max_iter=3000,
            random_state=int(seed),
        )
        model.fit(x, target, sample_weight=weights)
        raw = model.predict_proba(x)[:, 1]
        calibrator, _ = fit_best_calibrator(
            target,
            raw,
            dates,
            minimum_rows=args.minimum_calibration_rows,
        )
        return FittedMethod(
            candidate.name,
            feature_spec,
            "logistic",
            model,
            {},
            calibrator,
            candidate.negative_cost,
        )

    unique_dates = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(dates), errors="coerce").unique()
    ).sort_values()
    split = (
        unique_dates[max(1, int(len(unique_dates) * 0.8))]
        if len(unique_dates) > 2
        else unique_dates[-1]
    )
    train_mask = dates < np.datetime64(split)
    valid_mask = ~train_mask
    if int(train_mask.sum()) < 200 or int(valid_mask.sum()) < 50:
        train_mask = np.ones(len(target), dtype=bool)
        valid_mask = np.zeros(len(target), dtype=bool)

    if candidate.kind == "lightgbm":
        import lightgbm as lgb

        params: dict[str, Any] = {
            "objective": "binary",
            "metric": "average_precision",
            "learning_rate": 0.025,
            "num_leaves": 31,
            "min_data_in_leaf": 45,
            "feature_fraction": 0.82,
            "bagging_fraction": 0.82,
            "bagging_freq": 1,
            "lambda_l1": 0.4,
            "lambda_l2": 3.0,
            "verbosity": -1,
            "force_col_wise": True,
            "num_threads": int(args.threads_per_model),
            "seed": int(seed),
            "feature_fraction_seed": int(seed),
            "bagging_seed": int(seed),
        }
        train_set = lgb.Dataset(
            x[train_mask],
            label=target[train_mask],
            weight=weights[train_mask],
            free_raw_data=True,
        )
        callbacks = [lgb.log_evaluation(period=0)]
        if valid_mask.any():
            valid_set = lgb.Dataset(
                x[valid_mask],
                label=target[valid_mask],
                reference=train_set,
                free_raw_data=True,
            )
            callbacks.append(
                lgb.early_stopping(args.meta_early_stopping_rounds, verbose=False)
            )
            model = lgb.train(
                params,
                train_set,
                num_boost_round=args.meta_max_rounds,
                valid_sets=[valid_set],
                callbacks=callbacks,
            )
        else:
            model = lgb.train(
                params,
                train_set,
                num_boost_round=min(160, args.meta_max_rounds),
                callbacks=callbacks,
            )
        iterations = int(model.best_iteration or model.current_iteration())
        raw = model.predict(x, num_iteration=iterations)
        calibrator, _ = fit_best_calibrator(
            target,
            raw,
            dates,
            minimum_rows=args.minimum_calibration_rows,
        )
        return FittedMethod(
            candidate.name,
            feature_spec,
            "lightgbm",
            model,
            {"iterations": iterations},
            calibrator,
            candidate.negative_cost,
        )

    if candidate.kind == "xgboost":
        import xgboost as xgb

        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "tree_method": "hist",
            "device": args.resolved_xgboost_device,
            "learning_rate": 0.025,
            "grow_policy": "lossguide",
            "max_depth": 0,
            "max_leaves": 48,
            "min_child_weight": 8.0,
            "subsample": 0.82,
            "colsample_bytree": 0.80,
            "colsample_bynode": 0.88,
            "reg_alpha": 0.4,
            "reg_lambda": 3.0,
            "max_bin": 256,
            "seed": int(seed),
            "nthread": int(args.xgboost_threads),
            "verbosity": 0,
        }
        dtrain = xgb.QuantileDMatrix(
            x[train_mask],
            label=target[train_mask],
            weight=weights[train_mask],
            max_bin=256,
        )
        if valid_mask.any():
            dvalid = xgb.QuantileDMatrix(
                x[valid_mask],
                label=target[valid_mask],
                ref=dtrain,
                max_bin=256,
            )
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=args.meta_max_rounds,
                evals=[(dvalid, "validation")],
                early_stopping_rounds=args.meta_early_stopping_rounds,
                verbose_eval=False,
            )
        else:
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=min(160, args.meta_max_rounds),
                verbose_eval=False,
            )
        full = xgb.QuantileDMatrix(x, ref=dtrain, max_bin=256)
        iterations = int(
            (model.best_iteration + 1)
            if model.best_iteration is not None
            else min(160, args.meta_max_rounds)
        )
        raw = model.predict(full, iteration_range=(0, iterations))
        calibrator, _ = fit_best_calibrator(
            target,
            raw,
            dates,
            minimum_rows=args.minimum_calibration_rows,
        )
        return FittedMethod(
            candidate.name,
            feature_spec,
            "xgboost",
            model,
            {"iterations": iterations},
            calibrator,
            candidate.negative_cost,
        )
    raise ValueError(candidate.kind)


def predict_method(fit: FittedMethod, frame: pd.DataFrame) -> np.ndarray:
    meta = apply_meta_feature_spec(frame, fit.feature_spec)
    if fit.model_kind == "recipe_mean":
        raw = meta["ensemble_mean"].to_numpy(dtype=np.float64)
    elif fit.model_kind == "family_mean":
        columns = [str(value) for value in fit.model_parameters["family_columns"]]
        raw = meta[columns].mean(axis=1).to_numpy(dtype=np.float64)
    elif fit.model_kind == "safety_product":
        raw = meta["surge_safety_product"].to_numpy(dtype=np.float64)
    elif fit.model_kind == "conservative_q25":
        raw = meta["conservative_q25_safety"].to_numpy(dtype=np.float64)
    elif fit.model_kind == "strong_safety":
        raw = meta["strong_safety_product"].to_numpy(dtype=np.float64)
    elif fit.model_kind == "consensus_lcb":
        z = float(fit.model_parameters["uncertainty_z"])
        raw = (
            meta["ensemble_mean"].to_numpy(dtype=np.float64)
            - z * meta["ensemble_std"].to_numpy(dtype=np.float64)
        )
        raw = ensure_unit_interval(
            raw,
            frame["date"].to_numpy(dtype="datetime64[ns]"),
            "rank",
        )
    elif fit.model_kind == "logistic":
        raw = fit.model.predict_proba(meta.to_numpy(dtype=np.float32))[:, 1]
    elif fit.model_kind == "lightgbm":
        raw = fit.model.predict(
            meta.to_numpy(dtype=np.float32),
            num_iteration=int(fit.model_parameters["iterations"]),
        )
    elif fit.model_kind == "xgboost":
        import xgboost as xgb

        matrix = xgb.DMatrix(meta.to_numpy(dtype=np.float32))
        raw = fit.model.predict(
            matrix,
            iteration_range=(0, int(fit.model_parameters["iterations"])),
        )
    else:
        raise ValueError(fit.model_kind)
    return apply_calibrator(fit.calibrator, np.asarray(raw, dtype=np.float64))


def temporal_oof_method_scores(
    candidate: MethodCandidate,
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    args: argparse.Namespace,
    seed: int,
) -> pd.DataFrame:
    """Generate strictly forward, expanding meta OOF predictions.

    No branch is allowed to predict rows used for fitting.  Short histories use
    adaptive warm-up/block lengths but retain chronological direction.
    """

    unique_dates = pd.DatetimeIndex(
        pd.to_datetime(frame["date"], errors="coerce").unique()
    ).sort_values()
    if len(unique_dates) < 4:
        raise ValueError("temporal meta OOF에는 최소 4개 거래일이 필요합니다")

    configured_initial = max(2, int(args.meta_oof_initial_days))
    configured_block = max(1, int(args.meta_oof_block_days))
    adaptive_initial = min(
        configured_initial,
        max(2, int(math.floor(len(unique_dates) * 0.50))),
    )
    adaptive_initial = min(adaptive_initial, len(unique_dates) - 2)
    remaining = len(unique_dates) - adaptive_initial
    adaptive_block = min(
        configured_block,
        max(5, int(math.ceil(remaining / 4.0))),
    )

    rows: list[pd.DataFrame] = []
    start = int(adaptive_initial)
    block_id = 0
    while start < len(unique_dates):
        end = min(len(unique_dates), start + int(adaptive_block))
        train_dates = unique_dates[:start]
        valid_dates = unique_dates[start:end]
        train = frame.loc[frame["date"].isin(train_dates)].copy()
        valid = frame.loc[frame["date"].isin(valid_dates)].copy()
        if (
            len(train) < args.minimum_meta_train_rows
            or len(valid) == 0
            or train["target"].nunique() < 2
        ):
            start = end
            block_id += 1
            continue
        if pd.Timestamp(train["date"].max()) >= pd.Timestamp(valid["date"].min()):
            raise AssertionError("temporal OOF train/validation chronology violation")
        fit = fit_method(
            candidate,
            train,
            signal_columns,
            family_map,
            args,
            deterministic_seed(seed, candidate.name, block_id),
        )
        part = valid[
            [
                "fold_id",
                "date",
                "target",
                TICKER_COLUMN,
                *[column for column in args.scope_columns if column in valid],
            ]
        ].copy()
        part["score"] = predict_method(fit, valid)
        part["meta_train_end"] = pd.Timestamp(train["date"].max())
        part["meta_valid_start"] = pd.Timestamp(valid["date"].min())
        part["meta_block_id"] = int(block_id)
        rows.append(part)
        start = end
        block_id += 1
    if not rows:
        raise RuntimeError(f"{candidate.name}: strict temporal OOF score 생성 실패")
    return pd.concat(rows, ignore_index=True)

def choose_policy(
    oof: pd.DataFrame,
    policy_kind: str,
    args: argparse.Namespace,
    source: str,
) -> tuple[PrecisionPolicy, pd.DataFrame]:
    if policy_kind == "global_threshold":
        policy, search, folds = select_global_precision_policy(
            oof,
            target_precision=args.selection_target_precision,
            minimum_precision_lcb=args.minimum_precision_lcb,
            minimum_alerts_per_fold=args.minimum_alerts_per_fold,
            minimum_alert_days_per_fold=args.minimum_alert_days_per_fold,
            confidence_level=args.confidence_level,
            required_fold_pass_rate=args.required_selection_fold_pass_rate,
            source=source,
            maximum_threshold_candidates=args.maximum_threshold_candidates,
        )
        search["policy_kind"] = policy_kind
        return policy, search
    scope_policy_columns = {
        "market_threshold": "market",
        "bucket_threshold": "bucket",
        "industry_threshold": "industry_name",
        "scope_threshold": args.scope_threshold_column,
    }
    if policy_kind in scope_policy_columns:
        scope_column = str(scope_policy_columns[policy_kind])
        if scope_column not in oof.columns:
            raise KeyError(f"{policy_kind}에 필요한 scope column 누락: {scope_column}")
        policy, search = select_scope_precision_policy(
            oof,
            scope_column,
            target_precision=args.selection_target_precision,
            minimum_precision_lcb=args.minimum_precision_lcb,
            minimum_alerts_per_scope=args.minimum_alerts_per_scope,
            minimum_alert_days_per_scope=args.minimum_alert_days_per_scope,
            confidence_level=args.confidence_level,
            required_fold_pass_rate=args.required_selection_fold_pass_rate,
            source=source,
            maximum_threshold_candidates=args.maximum_threshold_candidates,
        )
        search["policy_kind"] = policy_kind
        search["scope_column"] = scope_column
        return policy, search
    raise ValueError(policy_kind)



def policy_gate(metrics: Mapping[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        float(metrics.get("precision", float("nan"))) >= args.target_precision
        and float(metrics.get("precision_wilson_lcb", float("nan")))
        >= args.minimum_precision_lcb
        and float(metrics.get("recall", float("nan")))
        >= args.minimum_useful_recall
        and int(metrics.get("alerts", 0)) >= args.minimum_alerts_per_fold
        and int(metrics.get("alert_days", 0))
        >= args.minimum_alert_days_per_fold
    )

def crossfit_method_candidates(
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    selection_folds: Sequence[int],
    methods: Sequence[MethodCandidate],
    policy_kinds: Sequence[str],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_records: list[dict[str, Any]] = []
    score_records: list[pd.DataFrame] = []
    search_records: list[pd.DataFrame] = []
    eligible = [fold for fold in sorted(selection_folds) if forward_training_folds(selection_folds, fold)]
    for candidate in methods:
        for heldout in eligible:
            train_folds = forward_training_folds(selection_folds, heldout)
            train = frame.loc[frame["fold_id"].isin(train_folds)].copy()
            valid = frame.loc[frame["fold_id"].eq(heldout)].copy()
            oof = temporal_oof_method_scores(
                candidate,
                train,
                signal_columns,
                family_map,
                args,
                deterministic_seed(args.policy_seed, candidate.name, heldout, "oof"),
            )
            fit = fit_method(
                candidate,
                train,
                signal_columns,
                family_map,
                args,
                deterministic_seed(args.policy_seed, candidate.name, heldout, "fit"),
            )
            valid_score = predict_method(fit, valid)
            for policy_kind in policy_kinds:
                try:
                    policy, search = choose_policy(
                        oof,
                        policy_kind,
                        args,
                        source=f"forward_selection_fold_{heldout}",
                    )
                except Exception as exc:
                    metric_records.append(
                        {
                            "method": candidate.name,
                            "policy_kind": policy_kind,
                            "fold_id": int(heldout),
                            "status": f"POLICY_FAILED:{type(exc).__name__}",
                            "precision": float("nan"),
                            "recall": float("nan"),
                            "alerts": 0,
                            "gate_pass": False,
                        }
                    )
                    continue
                scopes = (
                    valid[policy.scope_column].to_numpy(dtype=object)
                    if policy.scope_column and policy.scope_column in valid
                    else None
                )
                alert = apply_precision_policy(policy, valid_score, scopes)
                metrics = evaluate_alerts(
                    valid["target"].to_numpy(dtype=np.int8),
                    valid_score,
                    alert,
                    valid["date"].to_numpy(dtype="datetime64[ns]"),
                    valid[TICKER_COLUMN].to_numpy(dtype=object) if TICKER_COLUMN in valid else None,
                    args.confidence_level,
                )
                gate = bool(policy.gate_pass and policy_gate(metrics, args))
                metric_records.append(
                    {
                        "method": candidate.name,
                        "method_kind": candidate.kind,
                        "negative_cost": candidate.negative_cost,
                        "policy_kind": policy_kind,
                        "fold_id": int(heldout),
                        "train_folds": ",".join(str(value) for value in train_folds),
                        "status": "SUCCESS",
                        "policy_threshold": policy.threshold,
                        "policy_gate_on_train_oof": policy.gate_pass,
                        "gate_pass": gate,
                        **metrics,
                    }
                )
                scored = valid[["validation_index", "fold_id", "fold_role", "date", "target", TICKER_COLUMN, *[c for c in args.scope_columns if c in valid]]].copy()
                scored["method"] = candidate.name
                scored["policy_kind"] = policy_kind
                scored["score"] = valid_score
                scored["alert"] = alert.astype(np.uint8)
                score_records.append(scored)
                search = search.copy()
                search["method"] = candidate.name
                search["heldout_fold"] = int(heldout)
                search_records.append(search)
    metrics = pd.DataFrame(metric_records)
    scores = pd.concat(score_records, ignore_index=True) if score_records else pd.DataFrame()
    searches = pd.concat(search_records, ignore_index=True) if search_records else pd.DataFrame()
    return metrics, scores, searches


def build_forward_meta_split_audit(selection_folds: Sequence[int]) -> dict[str, Any]:
    """Record and validate the chronological training folds used by the meta layer."""
    records: list[dict[str, Any]] = []
    ordered = sorted({int(value) for value in selection_folds})
    for heldout in ordered:
        train_folds = forward_training_folds(ordered, heldout)
        records.append(
            {
                "heldout_fold": int(heldout),
                "train_folds": [int(value) for value in train_folds],
                "warmup_only": not bool(train_folds),
                "strictly_past_only": all(int(value) < int(heldout) for value in train_folds),
            }
        )
    passed = all(record["strictly_past_only"] for record in records)
    return with_payload_checksum(
        {
            "schema": "crashwatch_surge_forward_meta_split_audit_v6",
            "status": "PASS" if passed else "FAIL",
            "selection_folds": ordered,
            "records": records,
        }
    )


def summarize_crossfit(metrics: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    successful = metrics.loc[metrics["status"].eq("SUCCESS")].copy()
    for (method, policy_kind), part in successful.groupby(["method", "policy_kind"], sort=False):
        gate_count = int(part["gate_pass"].sum())
        fold_count = int(len(part))
        pass_rate = gate_count / fold_count if fold_count else 0.0
        records.append(
            {
                "method": method,
                "policy_kind": policy_kind,
                "fold_count": fold_count,
                "fold_pass_count": gate_count,
                "fold_pass_rate": pass_rate,
                "worst_precision": float(part["precision"].min()),
                "mean_precision": float(part["precision"].mean()),
                "worst_precision_lcb": float(part["precision_wilson_lcb"].min()),
                "mean_recall": float(part["recall"].mean()),
                "total_alerts": int(part["alerts"].sum()),
                "mean_alert_rate": float(part["alert_rate"].mean()),
                "gate_pass": bool(pass_rate + 1e-12 >= args.required_selection_fold_pass_rate),
            }
        )
    result = pd.DataFrame(records)
    if result.empty:
        return result
    result.sort_values(
        ["gate_pass", "fold_pass_rate", "mean_recall", "total_alerts", "worst_precision_lcb", "mean_precision"],
        ascending=[False, False, False, False, False, False],
        inplace=True,
        kind="mergesort",
    )
    result["selection_rank"] = np.arange(1, len(result) + 1)
    return result.reset_index(drop=True)


def choose_best_candidate(summary: pd.DataFrame) -> tuple[str, str]:
    if summary.empty:
        raise RuntimeError("method crossfit summary가 비어 있습니다")
    row = summary.iloc[0]
    return str(row["method"]), str(row["policy_kind"])


def final_fit_and_policy(
    method_name: str,
    policy_kind: str,
    frame: pd.DataFrame,
    signal_columns: Sequence[str],
    family_map: Mapping[str, str],
    selection_folds: Sequence[int],
    args: argparse.Namespace,
) -> tuple[FittedMethod, PrecisionPolicy, pd.DataFrame, pd.DataFrame]:
    candidate = next(value for value in METHOD_CANDIDATES if value.name == method_name)
    selection = frame.loc[frame["fold_id"].isin(selection_folds)].copy()
    oof = temporal_oof_method_scores(
        candidate,
        selection,
        signal_columns,
        family_map,
        args,
        deterministic_seed(args.policy_seed, method_name, "final_oof"),
    )
    policy, search = choose_policy(oof, policy_kind, args, source="final_selection_temporal_oof")
    fit = fit_method(
        candidate,
        selection,
        signal_columns,
        family_map,
        args,
        deterministic_seed(args.policy_seed, method_name, "final_fit"),
    )
    return fit, policy, oof, search


def evaluate_final_candidate(
    fit: FittedMethod,
    policy: PrecisionPolicy,
    frame: pd.DataFrame,
    crossfit_scores: pd.DataFrame,
    selected_method: str,
    selected_policy_kind: str,
    selection_folds: Sequence[int],
    roles: Mapping[str, Sequence[int]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scored_parts: list[pd.DataFrame] = []
    eligible_selection = [fold for fold in sorted(selection_folds) if forward_training_folds(selection_folds, fold)]
    selected_crossfit = crossfit_scores.loc[
        crossfit_scores["method"].eq(selected_method)
        & crossfit_scores["policy_kind"].eq(selected_policy_kind)
    ].copy()
    for fold_id in sorted(frame["fold_id"].unique()):
        part = frame.loc[frame["fold_id"].eq(fold_id)].copy()
        if int(fold_id) in eligible_selection:
            local = selected_crossfit.loc[selected_crossfit["fold_id"].eq(fold_id)].copy()
            if len(local) != len(part):
                raise ValueError(f"selection crossfit score rows mismatch fold={fold_id}")
            local = local.sort_values("validation_index", kind="mergesort")
            part = part.sort_values("validation_index", kind="mergesort")
            score = local["score"].to_numpy(dtype=np.float64)
            alert = local["alert"].to_numpy(dtype=bool)
            evaluation_role = "selection_forward_eval"
        elif int(fold_id) in set(selection_folds):
            score = predict_method(fit, part)
            scopes = part[policy.scope_column].to_numpy(dtype=object) if policy.scope_column and policy.scope_column in part else None
            alert = apply_precision_policy(policy, score, scopes)
            evaluation_role = "selection_warmup_excluded"
        else:
            score = predict_method(fit, part)
            scopes = part[policy.scope_column].to_numpy(dtype=object) if policy.scope_column and policy.scope_column in part else None
            alert = apply_precision_policy(policy, score, scopes)
            evaluation_role = role_for_fold(int(fold_id), roles)
        scored = part[["validation_index", "fold_id", "fold_role", "date", "target", TICKER_COLUMN, *[c for c in args.scope_columns if c in part]]].copy()
        scored["evaluation_role"] = evaluation_role
        scored["score"] = score
        scored["alert"] = alert.astype(np.uint8)
        scored_parts.append(scored)
    scored_frame = pd.concat(scored_parts, ignore_index=True)

    fold_records: list[dict[str, Any]] = []
    for fold_id, part in scored_frame.groupby("fold_id", sort=True):
        metrics = evaluate_alerts(
            part["target"].to_numpy(dtype=np.int8),
            part["score"].to_numpy(dtype=np.float64),
            part["alert"].to_numpy(dtype=bool),
            part["date"].to_numpy(dtype="datetime64[ns]"),
            part[TICKER_COLUMN].to_numpy(dtype=object),
            args.confidence_level,
        )
        metrics["precision_date_block_bootstrap_lcb"] = date_block_bootstrap_precision_lcb(
            part["target"].to_numpy(dtype=np.int8),
            part["alert"].to_numpy(dtype=bool),
            part["date"].to_numpy(dtype="datetime64[ns]"),
            samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=deterministic_seed(args.policy_seed, "bootstrap", int(fold_id)),
        )
        fold_records.append(
            {
                "fold_id": int(fold_id),
                "fold_role": str(part["fold_role"].iloc[0]),
                "evaluation_role": str(part["evaluation_role"].iloc[0]),
                "gate_pass": policy_gate(metrics, args),
                **metrics,
            }
        )
    fold_metrics = pd.DataFrame(fold_records)

    role_records: list[dict[str, Any]] = []
    for evaluation_role, part in scored_frame.groupby("evaluation_role", sort=False):
        metrics = evaluate_alerts(
            part["target"].to_numpy(dtype=np.int8),
            part["score"].to_numpy(dtype=np.float64),
            part["alert"].to_numpy(dtype=bool),
            part["date"].to_numpy(dtype="datetime64[ns]"),
            part[TICKER_COLUMN].to_numpy(dtype=object),
            args.confidence_level,
        )
        metrics["precision_date_block_bootstrap_lcb"] = date_block_bootstrap_precision_lcb(
            part["target"].to_numpy(dtype=np.int8),
            part["alert"].to_numpy(dtype=bool),
            part["date"].to_numpy(dtype="datetime64[ns]"),
            samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            seed=deterministic_seed(args.policy_seed, "role_bootstrap", evaluation_role),
        )
        role_records.append({"evaluation_role": evaluation_role, "gate_pass": policy_gate(metrics, args), **metrics})
    role_metrics = pd.DataFrame(role_records)

    curves: list[pd.DataFrame] = []
    for evaluation_role, part in scored_frame.groupby("evaluation_role", sort=False):
        curve = build_precision_curve(
            part["target"].to_numpy(dtype=np.int8),
            part["score"].to_numpy(dtype=np.float64),
            part["date"].to_numpy(dtype="datetime64[ns]"),
            part[TICKER_COLUMN].to_numpy(dtype=object),
            maximum_candidates=args.maximum_threshold_candidates,
            confidence_level=args.confidence_level,
        )
        curve.insert(0, "evaluation_role", evaluation_role)
        curves.append(curve)
    precision_curve = pd.concat(curves, ignore_index=True)
    return fold_metrics, role_metrics, scored_frame, precision_curve


def delayed_online_diagnostics(
    scored: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection = scored.loc[scored["fold_role"].eq("selection")].copy()
    records: list[dict[str, Any]] = []
    audits: list[pd.DataFrame] = []
    for role in ["confirmation", "recent_audit"]:
        part = scored.loc[scored["fold_role"].eq(role)].copy()
        if part.empty:
            continue
        alert, audit = simulate_delayed_online_threshold(
            selection[["date", "target", "score", TICKER_COLUMN]],
            part[["date", "target", "score", TICKER_COLUMN]],
            target_precision=args.target_precision,
            minimum_precision_lcb=args.minimum_precision_lcb,
            minimum_alerts=args.minimum_alerts_per_fold,
            minimum_alert_days=args.minimum_alert_days_per_fold,
            confidence_level=args.confidence_level,
            horizon_days=3,
            lookback_days=args.online_lookback_days,
            update_every_days=args.online_update_every_days,
            maximum_threshold_candidates=min(200, int(args.maximum_threshold_candidates)),
        )
        metrics = evaluate_alerts(
            part["target"].to_numpy(dtype=np.int8),
            part["score"].to_numpy(dtype=np.float64),
            alert,
            part["date"].to_numpy(dtype="datetime64[ns]"),
            part[TICKER_COLUMN].to_numpy(dtype=object),
            args.confidence_level,
        )
        records.append({"fold_role": role, "gate_pass": policy_gate(metrics, args), **metrics})
        audit.insert(0, "fold_role", role)
        audits.append(audit)
    return pd.DataFrame(records), pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()


def serialize_method(
    fit: FittedMethod,
    output: Path,
) -> dict[str, Any]:
    model_record: dict[str, Any] | None = None
    model_dir = output / "production" / "meta"
    model_dir.mkdir(parents=True, exist_ok=True)
    if fit.model_kind == "logistic":
        model_record = {
            "format": "sklearn_logistic_parameters",
            "classes": [int(value) for value in fit.model.classes_],
            "coef": fit.model.coef_.astype(float).tolist(),
            "intercept": fit.model.intercept_.astype(float).tolist(),
        }
    elif fit.model_kind == "lightgbm":
        path = model_dir / "precision_meta_lgb.txt"
        fit.model.save_model(str(path), num_iteration=int(fit.model_parameters["iterations"]))
        model_record = {
            "format": "lightgbm_text",
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
            "iterations": int(fit.model_parameters["iterations"]),
        }
    elif fit.model_kind == "xgboost":
        path = model_dir / "precision_meta_xgb.json"
        fit.model.save_model(path)
        model_record = {
            "format": "xgboost_json",
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": int(path.stat().st_size),
            "iterations": int(fit.model_parameters["iterations"]),
        }
    payload = {
        "method": fit.method,
        "model_kind": fit.model_kind,
        "feature_spec": fit.feature_spec,
        "model_parameters": fit.model_parameters,
        "model": model_record,
        "calibrator": fit.calibrator.to_dict(),
        "negative_cost": fit.negative_cost,
    }
    return relativize_registry_paths(payload, output)


def production_iteration(metrics: pd.DataFrame, recipe: str, seed: int, args: argparse.Namespace) -> int:
    part = metrics.loc[(metrics["recipe"] == recipe) & (metrics["seed"] == seed)]
    values = pd.to_numeric(part["effective_iteration"], errors="coerce").dropna().to_numpy(dtype=np.int64)
    if not len(values):
        return args.minimum_iterations
    return int(max(args.minimum_iterations, min(args.maximum_iterations, int(np.median(values)))))


def train_specialist_production_registry(
    recipes: Sequence[PrecisionRecipe],
    seeds: Sequence[int],
    bundle: DataBundle,
    profiles: Mapping[str, Sequence[str]],
    specialist_metrics: pd.DataFrame,
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    episode_weights_by_target = {
        target_variant: _episode_balance_weights(bundle, target_variant)
        for target_variant in sorted({recipe.target_variant for recipe in recipes})
    }
    matrix = np.load(bundle.matrix_path, mmap_mode="r")
    all_indices = np.arange(len(bundle.dates), dtype=np.int64)
    records: list[dict[str, Any]] = []
    for recipe in recipes:
        for seed in seeds:
            train_indices = apply_training_window(all_indices, bundle.dates, recipe.train_policy, recipe.rolling_days)
            features = list(profiles[recipe.profile])
            feature_indices = np.asarray([bundle.feature_to_index[value] for value in features], dtype=np.int64)
            x = np.asarray(matrix[np.ix_(train_indices, feature_indices)], dtype=np.float32)
            target = bundle.targets[recipe.target_variant][train_indices].astype(np.uint8)
            weights = build_precision_weights(
                bundle,
                recipe,
                train_indices,
                episode_weights_by_target[recipe.target_variant],
            )
            iterations = production_iteration(specialist_metrics, recipe.name, seed, args)
            directory = args.output / "production" / "specialists" / recipe.name / f"seed_{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            device = resolve_model_device(recipe.family, preflight)
            if recipe.family == "lightgbm":
                params = {
                    "objective": "binary",
                    "metric": "average_precision",
                    "learning_rate": 0.025,
                    "num_leaves": 63,
                    "min_data_in_leaf": 70,
                    "verbosity": -1,
                    "force_col_wise": True,
                    "num_threads": int(args.threads_per_model),
                    "seed": int(seed),
                    "feature_fraction_seed": int(seed),
                    "bagging_seed": int(seed),
                }
                params.update(recipe.params)
                model = save_lightgbm_model(x, target, weights, params, iterations, directory / "model.txt")
            elif recipe.family == "xgboost":
                params = {
                    "objective": "binary:logistic",
                    "eval_metric": "aucpr",
                    "tree_method": "hist",
                    "device": device,
                    "learning_rate": 0.025,
                    "max_depth": 7,
                    "seed": int(seed),
                    "nthread": int(args.xgboost_threads),
                    "verbosity": 0,
                }
                params.update(recipe.params)
                model = save_xgboost_model(x, target, weights, params, iterations, directory / "model.json")
            else:
                params = {
                    "loss_function": "Logloss",
                    "eval_metric": "AUC",
                    "learning_rate": 0.035,
                    "depth": 8,
                    "verbose": False,
                    "allow_writing_files": False,
                    "task_type": device,
                    "random_seed": int(seed),
                    "thread_count": int(args.threads_per_model),
                }
                params.update(recipe.params)
                model = save_catboost_model(x, target, weights, params, iterations, directory / "model.cbm")
            records.append(
                {
                    "recipe": recipe.name,
                    "family": recipe.family,
                    "profile": recipe.profile,
                    "target_variant": recipe.target_variant,
                    "train_policy": recipe.train_policy,
                    "rolling_days": recipe.rolling_days,
                    "time_decay_half_life_days": recipe.time_decay_half_life_days,
                    "false_positive_cost": recipe.false_positive_cost,
                    "seed_aggregation": recipe.seed_aggregation,
                    "seed": int(seed),
                    "features": features,
                    "model": model,
                }
            )
            log(f"production specialist: {recipe.name} seed={seed}")
    payload = with_payload_checksum(
        {
            "schema": "crashwatch_surge_precision_specialist_registry_v6",
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
    atomic_write_json(args.output / "V6_SPECIALIST_PRODUCTION_REGISTRY.json", portable)
    return portable



def build_gap_report(
    fold_metrics: pd.DataFrame,
    role_metrics: pd.DataFrame,
    curve: pd.DataFrame,
    policy: PrecisionPolicy,
    selected_method: str,
    selected_policy_kind: str,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], str]:
    gaps: list[dict[str, Any]] = []
    feasibility: list[dict[str, Any]] = []
    for role, part in curve.groupby("evaluation_role", sort=False):
        best = maximum_recall_at_precision(
            part,
            args.target_precision,
            minimum_alerts=args.minimum_alerts_per_fold,
            minimum_alert_days=args.minimum_alert_days_per_fold,
            minimum_precision_lcb=args.minimum_precision_lcb,
        )
        feasibility.append({"evaluation_role": role, **best})

    for _, row in role_metrics.iterrows():
        role = str(row["evaluation_role"])
        precision = float(row.get("precision", float("nan")))
        recall = float(row.get("recall", float("nan")))
        lcb = float(row.get("precision_wilson_lcb", float("nan")))
        if not np.isfinite(precision) or precision < args.target_precision:
            gaps.append(
                {
                    "evaluation_role": role,
                    "gap_type": "PRECISION_SHORTFALL",
                    "observed": precision,
                    "required": args.target_precision,
                    "shortfall": (
                        args.target_precision - precision
                        if np.isfinite(precision)
                        else None
                    ),
                }
            )
        if not np.isfinite(lcb) or lcb < args.minimum_precision_lcb:
            gaps.append(
                {
                    "evaluation_role": role,
                    "gap_type": "PRECISION_CONFIDENCE_SHORTFALL",
                    "observed": lcb,
                    "required": args.minimum_precision_lcb,
                }
            )
        if int(row.get("alerts", 0)) < args.minimum_alerts_per_fold:
            gaps.append(
                {
                    "evaluation_role": role,
                    "gap_type": "INSUFFICIENT_ALERT_SUPPORT",
                    "observed": int(row.get("alerts", 0)),
                    "required": args.minimum_alerts_per_fold,
                }
            )
        if int(row.get("alert_days", 0)) < args.minimum_alert_days_per_fold:
            gaps.append(
                {
                    "evaluation_role": role,
                    "gap_type": "INSUFFICIENT_ALERT_DAYS",
                    "observed": int(row.get("alert_days", 0)),
                    "required": args.minimum_alert_days_per_fold,
                }
            )
        if not np.isfinite(recall) or recall < args.minimum_useful_recall:
            gaps.append(
                {
                    "evaluation_role": role,
                    "gap_type": "LOW_COVERAGE_AT_70_PRECISION",
                    "observed": recall,
                    "required": args.minimum_useful_recall,
                }
            )

    fold_gate_by_role: dict[str, dict[str, Any]] = {}
    for evaluation_role, part in fold_metrics.groupby("evaluation_role", sort=False):
        pass_count = int(part["gate_pass"].astype(bool).sum())
        fold_count = int(len(part))
        fold_gate_by_role[str(evaluation_role)] = {
            "fold_count": fold_count,
            "fold_pass_count": pass_count,
            "fold_pass_rate": pass_count / fold_count if fold_count else 0.0,
            "worst_precision": float(part["precision"].min()) if fold_count else float("nan"),
            "worst_recall": float(part["recall"].min()) if fold_count else float("nan"),
        }

    role_gate_map = {
        str(row["evaluation_role"]): bool(row.get("gate_pass", False))
        for _, row in role_metrics.iterrows()
    }
    selection_fold_rate = float(
        fold_gate_by_role.get("selection_forward_eval", {}).get(
            "fold_pass_rate", 0.0
        )
    )
    selection_gate = bool(
        policy.gate_pass
        and role_gate_map.get("selection_forward_eval", False)
        and selection_fold_rate + 1e-12
        >= args.required_selection_fold_pass_rate
    )

    holdout_role_gates: dict[str, bool] = {}
    for role in ("confirmation", "recent_audit"):
        rate = float(fold_gate_by_role.get(role, {}).get("fold_pass_rate", 0.0))
        holdout_role_gates[role] = bool(
            role_gate_map.get(role, False)
            and rate + 1e-12 >= args.required_holdout_fold_pass_rate
        )
    observed_holdout_gate = all(holdout_role_gates.values())
    development_gate = bool(selection_gate and observed_holdout_gate)
    status = (
        "READY_FOR_NEW_FUTURE_HOLDOUT"
        if development_gate
        else "STOP_PRECISION70_GATE"
    )

    report = {
        "schema": "crashwatch_surge_precision70_gap_report_v6",
        "status": status,
        "target_definition": "3거래일 이내 누적수익률 +5% 이상",
        "alert_accuracy_definition": "precision = TP / (TP + FP)",
        "target_precision": args.target_precision,
        "selection_target_precision": args.selection_target_precision,
        "selection_precision_buffer": args.selection_precision_buffer,
        "minimum_useful_recall": args.minimum_useful_recall,
        "minimum_precision_lcb": args.minimum_precision_lcb,
        "no_alert_count_limit": True,
        "selected_method": selected_method,
        "selected_policy_kind": selected_policy_kind,
        "policy": policy.to_dict(),
        "selection_gate_pass": selection_gate,
        "observed_holdout_gate_pass": observed_holdout_gate,
        "development_gate_pass": development_gate,
        "role_gate_map": role_gate_map,
        "fold_gate_by_role": fold_gate_by_role,
        "holdout_role_gate_map": holdout_role_gates,
        "required_selection_fold_pass_rate": args.required_selection_fold_pass_rate,
        "required_holdout_fold_pass_rate": args.required_holdout_fold_pass_rate,
        "gaps": gaps,
        "maximum_recall_at_target_precision": feasibility,
        "role_metrics": role_metrics.to_dict(orient="records"),
    }

    lines = [
        "# CrashWatch Surge Precision 70 V6 — 성능 병목 보고서",
        "",
        f"- 상태: `{status}`",
        "- 경보 정확도 정의: `Precision = TP / (TP + FP)`",
        f"- 최종 목표 Precision: `{args.target_precision:.1%}`",
        f"- Selection 정책 선택 목표: `{args.selection_target_precision:.1%}`",
        f"- 최소 유효 Recall: `{args.minimum_useful_recall:.1%}`",
        "- 일일 경보 개수·비율 상한: 없음",
        (
            f"- 최소 통계 표본: fold당 {args.minimum_alerts_per_fold} alerts, "
            f"{args.minimum_alert_days_per_fold} alert days"
        ),
        "",
        "## 역할별 결과",
        "",
    ]
    for record in role_metrics.to_dict(orient="records"):
        lines.append(
            f"- {record['evaluation_role']}: "
            f"precision={float(record['precision']):.2%}, "
            f"recall={float(record['recall']):.2%}, "
            f"alerts={int(record['alerts'])}, "
            f"Wilson LCB={float(record['precision_wilson_lcb']):.2%}, "
            f"gate={bool(record['gate_pass'])}"
        )
    lines.extend(["", "## Fold 안정성", ""])
    for role, record in fold_gate_by_role.items():
        lines.append(
            f"- {role}: pass={record['fold_pass_count']}/{record['fold_count']} "
            f"({record['fold_pass_rate']:.1%}), "
            f"worst precision={record['worst_precision']:.2%}, "
            f"worst recall={record['worst_recall']:.2%}"
        )
    lines.extend(["", "## 70% Precision에서 가능한 최대 Recall", ""])
    for record in feasibility:
        lines.append(
            f"- {record['evaluation_role']}: "
            f"point-feasible={record.get('point_precision_feasible')}, "
            f"full-gate-feasible={record.get('gate_feasible')}, "
            f"max recall={float(record.get('maximum_recall', 0.0)):.2%}, "
            f"best precision={float(record.get('best_available_precision', float('nan'))):.2%}, "
            f"alerts={int(record.get('alerts', 0))}"
        )
    lines.extend(
        [
            "",
            "## 판정 원칙",
            "",
            "70% Precision을 만들 수 없을 때 threshold를 낮추거나 목표를 완화하지 않는다.",
            "한두 건만 골라 70%를 만드는 착시는 최소 alerts·alert-days·Wilson LCB·최소 Recall Gate로 차단한다.",
            "경보 수 상한은 없으며, 모든 Gate를 만족하는 후보 중 Recall과 경보 커버리지를 최대화한다.",
        ]
    )
    return report, "\n".join(lines) + "\n"

def synthetic_signal_frame(
    seed: int = 11,
    weak: bool = False,
    date_count: int = 240,
    ticker_count: int = 16,
) -> tuple[pd.DataFrame, dict[str, str], list[str]]:
    rng = np.random.default_rng(seed)
    date_count = max(80, int(date_count))
    ticker_count = max(8, int(ticker_count))
    dates = pd.date_range("2019-01-01", periods=date_count, freq="B")
    tickers = [f"T{index:02d}" for index in range(ticker_count)]
    fold_width = max(1, int(math.ceil(date_count / 8)))
    records: list[dict[str, Any]] = []
    for date_index, date in enumerate(dates):
        regime = 0.4 * math.sin(date_index / 55.0)
        for ticker_index, ticker in enumerate(tickers):
            latent = rng.normal() + 0.5 * regime + 0.25 * math.sin(ticker_index)
            if weak:
                probability = sigmoid(np.asarray([latent - 1.25]))[0]
                crash_probability = sigmoid(np.asarray([-latent - 1.25]))[0]
                target = int(rng.random() < probability)
                crash_target = int(rng.random() < crash_probability)
                noise_scale = 1.8
            else:
                # Easy synthetic path validates the READY/freeze branch without
                # pretending to represent real market performance.
                target = int(latent + rng.normal(scale=0.08) > 0.90)
                crash_target = int(-latent + rng.normal(scale=0.08) > 0.90)
                noise_scale = 0.16
            base = sigmoid(np.asarray([latent + rng.normal(scale=noise_scale)]))[0]
            strong = sigmoid(np.asarray([latent - 0.20 + rng.normal(scale=noise_scale)]))[0]
            crash = sigmoid(np.asarray([-latent + rng.normal(scale=noise_scale)]))[0]
            fold_id = min(7, date_index // fold_width)
            role = "selection" if fold_id <= 4 else "confirmation" if fold_id <= 6 else "recent_audit"
            records.append(
                {
                    "validation_index": len(records),
                    "fold_id": fold_id,
                    "fold_role": role,
                    "date": date,
                    "target": target,
                    "target__surge_d3": target,
                    "target__crash_d3": crash_target,
                    TICKER_COLUMN: ticker,
                    "market": "KOSPI" if ticker_index < 16 else "KOSDAQ",
                    "bucket": "A" if ticker_index % 2 == 0 else "B",
                    "base_signal": base,
                    "strong7_signal": strong,
                    "crash_signal": crash,
                    "base_signal__seed_agreement": 0.9 - 0.1 * rng.random(),
                }
            )
    frame = pd.DataFrame(records)
    family = {
        "base_signal": "base",
        "strong7_signal": "strong7",
        "crash_signal": "crash",
        "base_signal__seed_agreement": "seed_agreement",
    }
    return frame, family, list(family)


def run_synthetic_smoke(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    frame, family_map, signals = synthetic_signal_frame(
        seed=args.policy_seed,
        weak=args.synthetic_weak,
        date_count=args.synthetic_date_count,
        ticker_count=args.synthetic_ticker_count,
    )
    calibration_target_map = {
        "base_signal": "surge_d3",
        "strong7_signal": "surge_d3",
        "crash_signal": "crash_d3",
        "base_signal__seed_agreement": "identity",
    }
    signal_roles = {
        "base_signal": "surge",
        "strong7_signal": "strong_surge",
        "crash_signal": "crash",
        "base_signal__seed_agreement": "agreement",
    }
    calibrated, calibrators, audit = calibrate_signals_forward(
        frame,
        signals,
        args.selection_folds,
        args,
        calibration_target_map,
    )
    methods = [candidate for candidate in METHOD_CANDIDATES if candidate.name in args.methods]
    metrics, scores, searches = crossfit_method_candidates(
        calibrated,
        signals,
        family_map,
        args.selection_folds,
        methods,
        args.policy_kinds,
        args,
    )
    atomic_write_json(
        args.output / "FORWARD_META_SPLIT_AUDIT.json",
        build_forward_meta_split_audit(args.selection_folds),
    )
    summary = summarize_crossfit(metrics, args)
    method, policy_kind = choose_best_candidate(summary)
    fit, policy, oof, final_search = final_fit_and_policy(
        method,
        policy_kind,
        calibrated,
        signals,
        family_map,
        args.selection_folds,
        args,
    )
    roles = {"selection": args.selection_folds, "confirmation": args.confirmation_folds, "recent_audit": args.recent_folds}
    folds, role_metrics, scored, curve = evaluate_final_candidate(
        fit,
        policy,
        calibrated,
        scores,
        method,
        policy_kind,
        args.selection_folds,
        roles,
        args,
    )
    report, text = build_gap_report(folds, role_metrics, curve, policy, method, policy_kind, args)
    atomic_write_csv(args.output / "method_forward_crossfit_metrics.csv", metrics)
    atomic_write_csv(args.output / "method_forward_crossfit_summary.csv", summary)
    atomic_write_csv(args.output / "precision_candidate_metrics_by_fold.csv", folds)
    atomic_write_csv(args.output / "precision_candidate_metrics_by_role.csv", role_metrics)
    atomic_write_csv(args.output / "precision_recall_curve_by_role.csv", curve)
    atomic_write_csv(args.output / "signal_calibration_audit.csv", audit)
    atomic_write_json(
        args.output / "SIGNAL_CALIBRATORS_V6.json",
        with_payload_checksum(
            {
                "calibrators": calibrators,
                "calibration_target_map": calibration_target_map,
                "signal_roles": signal_roles,
            }
        ),
    )
    atomic_write_json(args.output / "PRECISION_PERFORMANCE_GAP_REPORT.json", report)
    atomic_write_text(args.output / "PRECISION_PERFORMANCE_GAP_REPORT_KO.md", text)
    freeze = with_payload_checksum(
        {
            "schema": "crashwatch_surge_precision_freeze_v6",
            "selected_method": method,
            "selected_policy_kind": policy_kind,
            "precision_policy": policy.to_dict(),
            "method_spec": serialize_method(fit, args.output),
            "signals": signals,
            "family_map": family_map,
            "signal_calibrators": calibrators,
            "calibration_target_map": calibration_target_map,
            "signal_roles": signal_roles,
            "minimum_useful_recall": args.minimum_useful_recall,
            "required_selection_fold_pass_rate": args.required_selection_fold_pass_rate,
            "required_holdout_fold_pass_rate": args.required_holdout_fold_pass_rate,
            "target_precision": args.target_precision,
            "selection_target_precision": args.selection_target_precision,
            "selection_precision_buffer": args.selection_precision_buffer,
            "no_alert_count_limit": True,
            "synthetic": True,
        }
    )
    atomic_write_json(args.output / "PRECISION_FREEZE_V6.json", freeze)
    atomic_write_json(
        args.output / "SIGNAL_REGISTRY_V6.json",
        with_payload_checksum(
            {
                "signals": signals,
                "family_map": family_map,
                "calibration_target_map": calibration_target_map,
                "signal_roles": signal_roles,
                "synthetic": True,
            }
        ),
    )
    recommendation = with_payload_checksum(
        {
            "schema": "crashwatch_surge_precision_recommendation_v6",
            "status": report["status"],
            "target_precision": args.target_precision,
            "selection_target_precision": args.selection_target_precision,
            "accuracy_definition": "precision = TP / (TP + FP)",
            "no_alert_count_limit": True,
            "selected_method": method,
            "selected_policy_kind": policy_kind,
            "policy": policy.to_dict(),
            "synthetic": True,
        }
    )
    atomic_write_json(args.output / "FINAL_RECOMMENDATION_V6.json", recommendation)
    inventory_paths = [
        path for path in args.output.rglob("*")
        if path.is_file() and path.name not in {"RUN_STATUS.json", "OUTPUT_INVENTORY.json"}
    ]
    atomic_write_json(
        args.output / "OUTPUT_INVENTORY.json",
        with_payload_checksum({"files": compute_output_inventory(args.output, inventory_paths)}),
    )
    atomic_write_json(
        args.output / "RUN_STATUS.json",
        {
            "schema": RUNNER_SCHEMA,
            "status": "SUCCESS",
            "gate_status": report["status"],
            "synthetic": True,
        },
    )
    print(json.dumps({"gate_status": report["status"], "method": method, "policy": policy.to_dict()}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch Surge V6 — precision 70 selective alert system")
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--v4-output", type=Path)
    parser.add_argument("--v5-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--precision-recipes", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--synthetic-weak", action="store_true")
    parser.add_argument("--synthetic-date-count", type=int, default=240)
    parser.add_argument("--synthetic-ticker-count", type=int, default=16)
    parser.add_argument("--reuse-v4-matrix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--selection-folds", default="0,1,2,3,4")
    parser.add_argument("--confirmation-folds", default="5,6")
    parser.add_argument("--recent-folds", default="7")
    parser.add_argument("--precision-seeds", default="17,29,41")
    parser.add_argument(
        "--methods",
        default="calibrated_recipe_mean,calibrated_family_mean,surge_safety_product,conservative_q25_safety,strong_surge_safety,consensus_lcb_z1,consensus_lcb_z2,logistic_precision_c2,logistic_precision_c4,lgb_precision_verifier_c2,lgb_precision_verifier_c4,xgb_precision_verifier_c2,xgb_precision_verifier_c4",
    )
    parser.add_argument("--policy-kinds", default="global_threshold,market_threshold,bucket_threshold")
    parser.add_argument("--target-precision", type=float, default=0.70)
    parser.add_argument("--selection-precision-buffer", type=float, default=0.03)
    parser.add_argument("--minimum-precision-lcb", type=float, default=0.60)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-alerts-per-fold", type=int, default=30)
    parser.add_argument("--minimum-alert-days-per-fold", type=int, default=10)
    parser.add_argument("--minimum-alerts-per-scope", type=int, default=20)
    parser.add_argument("--minimum-alert-days-per-scope", type=int, default=8)
    parser.add_argument("--scope-required-fold-pass-rate", type=float, default=0.75)
    parser.add_argument("--required-selection-fold-pass-rate", type=float, default=1.0)
    parser.add_argument("--required-holdout-fold-pass-rate", type=float, default=1.0)
    parser.add_argument("--minimum-useful-recall", type=float, default=0.05)
    parser.add_argument("--maximum-threshold-candidates", type=int, default=500)
    parser.add_argument("--minimum-calibration-rows", type=int, default=300)
    parser.add_argument("--minimum-meta-train-rows", type=int, default=300)
    parser.add_argument("--scope-threshold-column", default="market")
    parser.add_argument("--scope-columns", default="industry_name,market,bucket")
    parser.add_argument("--minimum-scope-rows", type=int, default=30)
    parser.add_argument("--minimum-scope-alerts", type=int, default=5)
    parser.add_argument("--policy-seed", type=int, default=20260811)
    parser.add_argument("--meta-logistic-c", type=float, default=0.5)
    parser.add_argument("--meta-time-decay-half-life-days", type=float, default=252.0)
    parser.add_argument("--meta-max-rounds", type=int, default=400)
    parser.add_argument("--meta-early-stopping-rounds", type=int, default=50)
    parser.add_argument("--meta-oof-initial-days", type=int, default=180)
    parser.add_argument("--meta-oof-block-days", type=int, default=60)
    parser.add_argument("--online-lookback-days", type=int, default=504)
    parser.add_argument("--online-update-every-days", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--threads-per-model", type=int, default=6)
    parser.add_argument("--xgboost-threads", type=int, default=6)
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
    parser.add_argument("--family-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-production-training", action="store_true")
    parser.add_argument("--train-production-on-gate-fail", action="store_true")
    args = parser.parse_args()

    resolve_paths(args)
    args.selection_folds = parse_int_list(args.selection_folds)
    args.confirmation_folds = parse_int_list(args.confirmation_folds)
    args.recent_folds = parse_int_list(args.recent_folds)
    args.precision_seeds = parse_int_list(args.precision_seeds)
    args.methods = parse_str_list(args.methods)
    args.policy_kinds = parse_str_list(args.policy_kinds)
    args.scope_columns = parse_str_list(args.scope_columns)
    if not (0.5 <= args.target_precision <= 1.0):
        raise ValueError("target precision은 0.5~1.0 범위여야 합니다")
    if not (0.0 <= args.selection_precision_buffer < 0.30):
        raise ValueError("selection precision buffer는 0~0.30 범위여야 합니다")
    args.selection_target_precision = min(0.999999, args.target_precision + args.selection_precision_buffer)
    if any(method not in {candidate.name for candidate in METHOD_CANDIDATES} for method in args.methods):
        unknown = [method for method in args.methods if method not in {candidate.name for candidate in METHOD_CANDIDATES}]
        raise ValueError(f"지원하지 않는 method: {unknown}")
    allowed_policy_kinds = {
        "global_threshold",
        "market_threshold",
        "bucket_threshold",
        "industry_threshold",
        "scope_threshold",
    }
    if any(kind not in allowed_policy_kinds for kind in args.policy_kinds):
        raise ValueError(f"지원하지 않는 policy kind: {args.policy_kinds}; allowed={sorted(allowed_policy_kinds)}")
    if not (0.0 <= args.scope_required_fold_pass_rate <= 1.0):
        raise ValueError("scope-required-fold-pass-rate는 0~1 범위여야 합니다")
    if not (0.0 <= args.required_selection_fold_pass_rate <= 1.0):
        raise ValueError("required-selection-fold-pass-rate는 0~1 범위여야 합니다")
    if not (0.0 <= args.required_holdout_fold_pass_rate <= 1.0):
        raise ValueError("required-holdout-fold-pass-rate는 0~1 범위여야 합니다")
    if not (0.0 <= args.minimum_useful_recall <= 1.0):
        raise ValueError("minimum-useful-recall은 0~1 범위여야 합니다")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.synthetic_smoke:
        args.resolved_xgboost_device = "cpu"
        run_synthetic_smoke(args)
        return

    require_inputs(args)
    status = RunStatus(args.output)
    status.payload["schema"] = RUNNER_SCHEMA
    atomic_write_json(status.path, status.payload)
    roles = {
        "selection": args.selection_folds,
        "confirmation": args.confirmation_folds,
        "recent_audit": args.recent_folds,
    }

    with FileLock(args.output / ".surge_precision70_v6.lock"):
        try:
            status.stage("inputs", "RUNNING")
            recipe_payload, precision_recipes = load_precision_recipes(args.precision_recipes, args.quick)
            profiles = load_profiles(args.profiles, recipe_payload)
            folds = load_folds(args.folds)
            existing_recipes, v4_recipe_names, v4_seeds, v5_signal_registry = load_existing_recipe_registry(args)
            missing_profiles = sorted({recipe.profile for recipe in precision_recipes if recipe.profile not in profiles})
            if missing_profiles:
                raise KeyError(f"precision profile 누락: {missing_profiles}")
            status.stage("inputs", "SUCCESS", precision_recipes=len(precision_recipes), v4_recipes=len(v4_recipe_names))

            status.stage("bundle", "RUNNING")
            bundle = prepare_bundle(args, profiles, existing_recipes, precision_recipes, folds)
            validate_folds(folds, pd.Series(bundle.dates), minimum_purge_days=args.minimum_purge_trading_days)
            target_extension = augment_continuous_targets(bundle, args)
            atomic_write_json(args.output / "V6_TARGET_EXTENSION_AUDIT.json", target_extension)
            status.stage("bundle", "SUCCESS", rows=len(bundle.dates), features=len(bundle.features), **target_extension)

            status.stage("preflight", "RUNNING")
            compatibility = [
                V4Recipe(
                    name=recipe.name,
                    family=recipe.family,
                    profile=recipe.profile,
                    target_variant="surge_d3",
                    train_policy=recipe.train_policy,
                    positive_weight_mode="none",
                    params=recipe.params,
                    time_decay_half_life_days=recipe.time_decay_half_life_days,
                    rolling_days=recipe.rolling_days,
                )
                for recipe in precision_recipes
            ]
            preflight = preflight_families(compatibility, args.device, args.allow_cpu_fallback)
            atomic_write_json(args.output / "V6_BACKEND_PREFLIGHT.json", preflight)
            args.resolved_xgboost_device = str(preflight.get("xgboost", {}).get("resolved_device", "cpu"))
            status.stage("preflight", "SUCCESS", xgboost_device=args.resolved_xgboost_device)

            status.stage("specialist_tasks", "RUNNING")
            specialist_metrics = execute_specialist_tasks(
                precision_recipes,
                folds,
                args.precision_seeds,
                bundle,
                profiles,
                roles,
                args,
                preflight,
            )
            status.stage("specialist_tasks", "SUCCESS", tasks=len(specialist_metrics))

            status.stage("signals", "RUNNING")
            (
                signal_frame,
                family_map,
                signal_columns,
                seed_audit,
                calibration_target_map,
                signal_roles,
            ) = build_signal_frame(
                args,
                bundle,
                folds,
                roles,
                v4_recipe_names,
                v4_seeds,
                v5_signal_registry,
                precision_recipes,
                args.precision_seeds,
            )
            atomic_write_csv(args.output / "seed_aggregation_audit.csv", seed_audit)
            atomic_write_json(
                args.output / "SIGNAL_REGISTRY_V6.json",
                with_payload_checksum(
                    {
                        "signals": signal_columns,
                        "family_map": family_map,
                        "calibration_target_map": calibration_target_map,
                        "signal_roles": signal_roles,
                        "v4_recipes": v4_recipe_names,
                        "v4_seeds": v4_seeds,
                        "v5_rank_seeds": v5_signal_registry.get("v5_rank_seeds", []),
                        "v6_precision_recipes": [recipe.name for recipe in precision_recipes],
                        "v6_precision_seeds": args.precision_seeds,
                    }
                ),
            )
            status.stage("signals", "SUCCESS", signals=len(signal_columns))

            status.stage("calibration", "RUNNING")
            calibrated_frame, signal_calibrators, calibration_audit = calibrate_signals_forward(
                signal_frame,
                signal_columns,
                args.selection_folds,
                args,
                calibration_target_map,
            )
            atomic_write_csv(args.output / "signal_calibration_audit.csv", calibration_audit)
            atomic_write_json(
                args.output / "SIGNAL_CALIBRATORS_V6.json",
                with_payload_checksum(
                    {
                        "calibrators": signal_calibrators,
                        "calibration_target_map": calibration_target_map,
                        "signal_roles": signal_roles,
                    }
                ),
            )
            status.stage("calibration", "SUCCESS")

            status.stage("forward_crossfit", "RUNNING")
            methods = [candidate for candidate in METHOD_CANDIDATES if candidate.name in args.methods]
            crossfit_metrics, crossfit_scores, threshold_search = crossfit_method_candidates(
                calibrated_frame,
                signal_columns,
                family_map,
                args.selection_folds,
                methods,
                args.policy_kinds,
                args,
            )
            atomic_write_json(
                args.output / "FORWARD_META_SPLIT_AUDIT.json",
                build_forward_meta_split_audit(args.selection_folds),
            )
            summary = summarize_crossfit(crossfit_metrics, args)
            selected_method, selected_policy_kind = choose_best_candidate(summary)
            atomic_write_csv(args.output / "method_forward_crossfit_metrics.csv", crossfit_metrics)
            atomic_write_csv(args.output / "method_forward_crossfit_summary.csv", summary)
            atomic_write_csv(args.output / "threshold_search_forward_crossfit.csv", threshold_search)
            status.stage(
                "forward_crossfit",
                "SUCCESS",
                selected_method=selected_method,
                selected_policy_kind=selected_policy_kind,
            )

            status.stage("final_freeze", "RUNNING")
            final_fit, policy, final_oof, final_search = final_fit_and_policy(
                selected_method,
                selected_policy_kind,
                calibrated_frame,
                signal_columns,
                family_map,
                args.selection_folds,
                args,
            )
            fold_metrics, role_metrics, scored, curve = evaluate_final_candidate(
                final_fit,
                policy,
                calibrated_frame,
                crossfit_scores,
                selected_method,
                selected_policy_kind,
                args.selection_folds,
                roles,
                args,
            )
            atomic_write_csv(args.output / "precision_candidate_metrics_by_fold.csv", fold_metrics)
            atomic_write_csv(args.output / "precision_candidate_metrics_by_role.csv", role_metrics)
            atomic_write_csv(args.output / "precision_recall_curve_by_role.csv", curve)
            atomic_write_csv(args.output / "final_threshold_search.csv", final_search)
            atomic_write_npz(
                args.output / "precision_candidate_predictions.npz",
                validation_index=scored["validation_index"].to_numpy(dtype=np.int64),
                fold_id=scored["fold_id"].to_numpy(dtype=np.int16),
                date=pd.to_datetime(scored["date"]).to_numpy(dtype="datetime64[ns]").astype(np.int64),
                target=scored["target"].to_numpy(dtype=np.uint8),
                score=scored["score"].to_numpy(dtype=np.float32),
                alert=scored["alert"].to_numpy(dtype=np.uint8),
            )
            scope = build_scope_metrics(
                scored,
                args.scope_columns,
                args.confidence_level,
                args.minimum_scope_rows,
                args.minimum_scope_alerts,
            )
            atomic_write_csv(args.output / "precision_scope_metrics.csv", scope)
            online_metrics, online_audit = delayed_online_diagnostics(scored, args)
            atomic_write_csv(args.output / "delayed_online_precision_metrics.csv", online_metrics)
            atomic_write_csv(args.output / "delayed_online_threshold_audit.csv", online_audit)

            method_spec = serialize_method(final_fit, args.output)
            selection_reference = scored.loc[
                scored["evaluation_role"].eq("selection_forward_eval"), "score"
            ].to_numpy(dtype=np.float64)
            selection_reference = selection_reference[np.isfinite(selection_reference)]
            reference_quantile_levels = np.linspace(0.0, 1.0, 21)
            reference_quantiles = (
                np.quantile(selection_reference, reference_quantile_levels).astype(np.float64)
                if len(selection_reference)
                else np.asarray([], dtype=np.float64)
            )
            psi_inner_edges = np.unique(reference_quantiles[1:-1]) if len(reference_quantiles) else np.asarray([], dtype=np.float64)
            psi_edges = np.concatenate(([-np.inf], psi_inner_edges, [np.inf]))
            psi_counts = (
                np.histogram(selection_reference, bins=psi_edges)[0].astype(np.float64)
                if len(selection_reference) and len(psi_edges) >= 2
                else np.asarray([], dtype=np.float64)
            )
            psi_proportions = psi_counts / max(1.0, float(np.sum(psi_counts)))
            score_reference = {
                "rows": int(len(selection_reference)),
                "mean": float(np.mean(selection_reference)) if len(selection_reference) else None,
                "std": float(np.std(selection_reference)) if len(selection_reference) else None,
                "quantile_levels": reference_quantile_levels.astype(float).tolist(),
                "quantiles": reference_quantiles.astype(float).tolist(),
                "psi_inner_edges": psi_inner_edges.astype(float).tolist(),
                "psi_reference_proportions": psi_proportions.astype(float).tolist(),
                "psi_warning_threshold": 0.25,
            }
            freeze = with_payload_checksum(
                {
                    "schema": "crashwatch_surge_precision_freeze_v6",
                    "selected_method": selected_method,
                    "selected_policy_kind": selected_policy_kind,
                    "precision_policy": policy.to_dict(),
                    "method_spec": method_spec,
                    "signals": signal_columns,
                    "family_map": family_map,
                    "signal_calibrators": signal_calibrators,
                    "calibration_target_map": calibration_target_map,
                    "signal_roles": signal_roles,
                    "minimum_useful_recall": args.minimum_useful_recall,
                    "minimum_precision_lcb": args.minimum_precision_lcb,
                    "required_selection_fold_pass_rate": args.required_selection_fold_pass_rate,
                    "required_holdout_fold_pass_rate": args.required_holdout_fold_pass_rate,
                    "score_reference": score_reference,
                    "selection_folds": args.selection_folds,
                    "dataset_sha256": bundle.dataset_sha256,
                    "target_sha256": bundle.target_sha256,
                    "training_data_start": pd.Timestamp(np.min(bundle.dates)).isoformat(),
                    "training_data_end": pd.Timestamp(np.max(bundle.dates)).isoformat(),
                    "target_precision": args.target_precision,
                    "selection_target_precision": args.selection_target_precision,
                    "selection_precision_buffer": args.selection_precision_buffer,
                    "no_alert_count_limit": True,
                    "created_at": utc_now(),
                }
            )
            atomic_write_json(args.output / "PRECISION_FREEZE_V6.json", freeze)
            status.stage("final_freeze", "SUCCESS", policy=policy.to_dict())

            report, report_text = build_gap_report(
                fold_metrics,
                role_metrics,
                curve,
                policy,
                selected_method,
                selected_policy_kind,
                args,
            )
            atomic_write_json(args.output / "PRECISION_PERFORMANCE_GAP_REPORT.json", report)
            atomic_write_text(args.output / "PRECISION_PERFORMANCE_GAP_REPORT_KO.md", report_text)
            recommendation = with_payload_checksum(
                {
                    "schema": "crashwatch_surge_precision_recommendation_v6",
                    "status": report["status"],
                    "target_precision": args.target_precision,
                    "selection_target_precision": args.selection_target_precision,
                    "selection_precision_buffer": args.selection_precision_buffer,
                    "minimum_useful_recall": args.minimum_useful_recall,
                    "minimum_precision_lcb": args.minimum_precision_lcb,
                    "required_selection_fold_pass_rate": args.required_selection_fold_pass_rate,
                    "required_holdout_fold_pass_rate": args.required_holdout_fold_pass_rate,
                    "accuracy_definition": "precision = TP / (TP + FP)",
                    "no_alert_count_limit": True,
                    "selected_method": selected_method,
                    "selected_policy_kind": selected_policy_kind,
                    "policy": policy.to_dict(),
                    "selection_crossfit_summary": summary.to_dict(orient="records"),
                    "role_metrics": role_metrics.to_dict(orient="records"),
                    "next_gate": (
                        "모든 구조를 동결한 뒤 2026-06-23 이후 신규 미래 데이터에 한 번만 적용"
                        if report["status"] == "READY_FOR_NEW_FUTURE_HOLDOUT"
                        else "Gate 실패 후보이므로 신규 미래 holdout을 열지 말고 development 데이터에서 병목을 수정"
                    ),
                }
            )
            atomic_write_json(args.output / "FINAL_RECOMMENDATION_V6.json", recommendation)

            status.stage("production", "RUNNING")
            should_train = not args.skip_production_training and (
                report["status"] == "READY_FOR_NEW_FUTURE_HOLDOUT" or args.train_production_on_gate_fail
            )
            if should_train:
                train_specialist_production_registry(
                    precision_recipes,
                    args.precision_seeds,
                    bundle,
                    profiles,
                    specialist_metrics,
                    args,
                    preflight,
                )
            status.stage("production", "SUCCESS", trained=should_train)

            inventory_paths = [
                path
                for path in args.output.rglob("*")
                if path.is_file()
                and path.name
                not in {
                    "RUN_STATUS.json",
                    "OUTPUT_INVENTORY.json",
                    "VERIFICATION_REPORT_V6.json",
                    ".surge_precision70_v6.lock",
                }
            ]
            inventory = compute_output_inventory(args.output, inventory_paths)
            atomic_write_json(args.output / "OUTPUT_INVENTORY.json", with_payload_checksum({"files": inventory}))
            status.success(
                gate_status=report["status"],
                selected_method=selected_method,
                selected_policy_kind=selected_policy_kind,
                no_alert_count_limit=True,
            )
            print("=" * 76)
            print("CrashWatch Surge Precision 70 V6")
            print("=" * 76)
            print(f"Selected method      : {selected_method}")
            print(f"Selected policy      : {selected_policy_kind}")
            print(f"Target precision     : {args.target_precision:.1%}")
            print(f"Selection target     : {args.selection_target_precision:.1%}")
            print("Alert count limit    : NONE")
            print(f"Selection gate       : {policy.gate_pass}")
            print(f"Final status         : {report['status']}")
            print(f"Output               : {args.output}")
            print("=" * 76)
        except Exception as exc:
            status.failure(exc)
            raise


if __name__ == "__main__":
    main()
