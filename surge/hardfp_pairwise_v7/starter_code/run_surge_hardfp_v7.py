from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from run_surge_model_zoo_v4 import (
    DataBundle,
    Recipe,
    fit_predict_base_family,
    fold_indices,
    preflight_families,
    prepare_data_bundle,
    resolve_model_device,
)
from surge_model_zoo_common import (
    DATE_COLUMN,
    TARGET_COLUMN,
    TARGET_VALID_COLUMN,
    TICKER_COLUMN,
    FoldSpec,
    apply_training_window,
    atomic_write_npz,
    load_folds,
    load_json,
    payload_checksum_is_valid,
    sha256_file,
    validate_folds,
)
from surge_hardfp_common_v7 import (
    ErrorGroupConfig,
    PrecisionRule,
    RelativeFeatureSpec,
    add_scope_one_hot,
    apply_rule,
    assign_error_groups,
    atomic_write_csv,
    atomic_write_json,
    binary_metrics,
    build_output_inventory,
    build_pairwise_training_frame,
    build_relative_feature_frame,
    contrast_feature_stats,
    datewise_rank,
    event_structure,
    evaluate_alert_mask,
    fit_lgb_ranker,
    fit_xgb_ranker,
    json_safe,
    method_score,
    predict_ranker,
    rule_gate,
    search_precision_cascade,
    select_error_features,
    select_precision_threshold,
    sha256_bytes,
    stable_json_bytes,
    summarize_error_map,
    with_checksum,
)


RUN_SCHEMA = "crashwatch_surge_hardfp_bottleneck_v7"


@dataclass(frozen=True)
class BaseRecipe:
    name: str
    family: str
    profile: str
    train_policy: str
    params: dict[str, Any]
    rounds: int
    rolling_days: int | None = None
    half_life_days: float | None = None

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "BaseRecipe":
        return cls(
            name=str(record["name"]),
            family=str(record["family"]),
            profile=str(record["profile"]),
            train_policy=str(record.get("train_policy", "expanding")),
            params=dict(record.get("params", {})),
            rounds=int(record.get("rounds", 160)),
            rolling_days=int(record["rolling_days"]) if record.get("rolling_days") is not None else None,
            half_life_days=float(record["half_life_days"]) if record.get("half_life_days") is not None else None,
        )


@dataclass
class MetaModel:
    kind: str
    model: Any
    imputer: Any | None = None
    scaler: Any | None = None
    feature_columns: list[str] | None = None


BASE_METHODS = [
    "base",
    "verifier",
    "pairwise",
    "recent",
    "balanced_blend",
    "hardfp_blend",
    "regime_blend",
]
META_METHODS = ["meta_logistic", "meta_lgb", "meta_xgb"]


def log(message: str) -> None:
    stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def parse_int_list(text: str) -> list[int]:
    values = [int(v.strip()) for v in str(text).split(",") if v.strip()]
    return list(dict.fromkeys(values))


def parse_str_list(text: str) -> list[str]:
    return [v.strip() for v in str(text).split(",") if v.strip()]


def scope_level_mask(values: pd.Series, level: str) -> np.ndarray:
    """Treat missing scope labels as not belonging to any fitted scope."""
    return (
        values.astype("string")
        .eq(str(level))
        .fillna(False)
        .to_numpy(dtype=bool)
    )


def resolve_paths(args: argparse.Namespace) -> None:
    root = args.package_root.resolve()
    args.package_root = root
    args.output = (args.output or root / "outputs" / "surge_hardfp_v7").resolve()
    args.dataset = (args.dataset or root / "data" / "training_dataset_finance11h.parquet").resolve()
    args.target_sidecar = (args.target_sidecar or root / "data" / "surge_target_3d5.parquet").resolve()
    if args.folds is None:
        candidates = [
            root / "outputs" / "surge_correlation_map_complete" / "walk_forward_folds.json",
            root / "references" / "feature_metadata" / "outer_walk_forward_folds.json",
        ]
        args.folds = next((p for p in candidates if p.exists()), candidates[-1])
    if args.profiles is None:
        candidates = [
            root / "outputs" / "surge_pre_model_gate_v3" / "surge_feature_profiles_corrected.json",
            root / "outputs" / "surge_correlation_map_complete" / "surge_feature_profiles.json",
            root / "references" / "feature_metadata" / "profile_manifest.json",
        ]
        args.profiles = next((p for p in candidates if p.exists()), candidates[-1])
    if args.recipes is None:
        args.recipes = Path(__file__).resolve().parent / "default_hardfp_recipes_v7.json"
    args.folds = Path(args.folds).resolve()
    args.profiles = Path(args.profiles).resolve()
    args.recipes = Path(args.recipes).resolve()
    args.output.mkdir(parents=True, exist_ok=True)


def require_inputs(args: argparse.Namespace) -> None:
    paths = [args.dataset, args.target_sidecar, args.folds, args.profiles, args.recipes]
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("필수 입력 누락:\n" + "\n".join(missing))


def load_profiles_flexible(path: Path) -> dict[str, list[str]]:
    payload = load_json(path)
    raw = payload.get("profiles", payload) if isinstance(payload, dict) else payload
    profiles: dict[str, list[str]] = {}
    if isinstance(raw, dict):
        for name, record in raw.items():
            values = record.get("features", []) if isinstance(record, dict) else record
            if isinstance(values, list):
                profiles[str(name)] = [str(v) for v in values]
    if "P0_ALL_VALID" not in profiles:
        for fallback in ["P0_FULL_439", "P0_FULL", "FULL", "P0"]:
            if fallback in profiles:
                profiles["P0_ALL_VALID"] = list(profiles[fallback])
                break
    if "P0_ALL_VALID" not in profiles:
        raise KeyError(f"P0_ALL_VALID profile을 찾지 못했습니다: {path}")
    if "P1_SELECTION_TOP" not in profiles:
        profiles["P1_SELECTION_TOP"] = profiles["P0_ALL_VALID"][: min(100, len(profiles["P0_ALL_VALID"]))]
    if "P7S_UNIVARIATE_TOP3_LIFT" not in profiles:
        for fallback in ["P3S_SELECTION_CLUSTER_REP", "P3_STRICT_CLUSTER_REP", "P1_SELECTION_TOP"]:
            if fallback in profiles:
                profiles["P7S_UNIVARIATE_TOP3_LIFT"] = list(profiles[fallback])[: min(60, len(profiles[fallback]))]
                break
    r2 = list(profiles["P7S_UNIVARIATE_TOP3_LIFT"])
    for feature in ["u_global_copper_adj_ret_5", "t_finshort_value_z_20"]:
        if feature in profiles["P0_ALL_VALID"] and feature not in r2:
            r2.append(feature)
    profiles["P7S_R2"] = r2
    return profiles


def load_base_recipes(path: Path, quick: bool) -> list[BaseRecipe]:
    payload = load_json(path)
    records = payload.get("base_recipes", [])
    recipes = [BaseRecipe.from_dict(r) for r in records]
    if quick:
        quick_names = set(str(v) for v in payload.get("quick_base_recipes", []))
        recipes = [r for r in recipes if r.name in quick_names]
    if not recipes:
        raise ValueError("base recipe가 비어 있습니다")
    return recipes


def _compat_recipe(base: BaseRecipe) -> Recipe:
    return Recipe(
        name=f"v7compat__{base.name}",
        family=base.family,
        profile=base.profile,
        target_variant="surge_d3",
        train_policy=base.train_policy,
        positive_weight_mode="none",
        params=base.params,
        time_decay_half_life_days=base.half_life_days,
        rolling_days=base.rolling_days,
    )


def _reuse_matrix_cache(args: argparse.Namespace, profiles: Mapping[str, Sequence[str]]) -> DataBundle | None:
    candidates = [
        args.package_root / "outputs" / "surge_model_zoo_v4" / "matrix_cache" / "CACHE_MANIFEST.json",
        args.package_root / "outputs" / "surge_precision70_v6" / "matrix_cache" / "CACHE_MANIFEST.json",
        args.package_root / "outputs" / "surge_alert_budget_v5" / "matrix_cache" / "CACHE_MANIFEST.json",
    ]
    for manifest_path in candidates:
        if not manifest_path.exists():
            continue
        try:
            manifest = load_json(manifest_path)
            if not payload_checksum_is_valid(manifest):
                continue
            identity = manifest.get("identity", {})
            if identity.get("dataset_sha256") != sha256_file(args.dataset):
                continue
            if identity.get("target_sha256") != sha256_file(args.target_sidecar):
                continue
            features = [str(v) for v in identity.get("features", [])]
            required = set(profiles["P0_ALL_VALID"])
            if not required.issubset(set(features)):
                continue
            matrix_path = Path(manifest["matrix_path"])
            frame_path = Path(manifest["frame_path"])
            targets_path = Path(manifest["targets_path"])
            if not all(p.exists() for p in [matrix_path, frame_path, targets_path]):
                continue
            if manifest.get("matrix_path_sha256") != sha256_file(matrix_path):
                continue
            frame = pd.read_pickle(frame_path)
            arrays = np.load(targets_path, allow_pickle=False)
            targets = {
                k.removeprefix("target__"): np.asarray(arrays[k])
                for k in arrays.files
                if k.startswith("target__")
            }
            dates = np.asarray(arrays["dates"], dtype=np.int64).astype("datetime64[ns]")
            log(f"기존 matrix cache 재사용: {manifest_path}")
            return DataBundle(
                matrix_path=matrix_path,
                metadata_path=manifest_path,
                features=features,
                feature_to_index={f: i for i, f in enumerate(features)},
                frame=frame,
                targets=targets,
                dates=dates,
                dataset_sha256=str(identity["dataset_sha256"]),
                target_sha256=str(identity["target_sha256"]),
                cache_identity_hash=str(manifest.get("identity_hash", "")),
            )
        except Exception as exc:
            log(f"matrix cache 무시: {manifest_path}: {type(exc).__name__}: {exc}")
    return None


def prepare_bundle(args: argparse.Namespace, profiles: Mapping[str, Sequence[str]], base_recipes: Sequence[BaseRecipe], folds: Sequence[FoldSpec]) -> DataBundle:
    bundle = _reuse_matrix_cache(args, profiles) if args.reuse_matrix_cache else None
    if bundle is not None:
        return bundle
    compatibility = [_compat_recipe(r) for r in base_recipes]
    # P0 전체 피처가 matrix에 포함되도록 compatibility 하나를 강제한다.
    compatibility.append(
        Recipe(
            name="v7compat__p0",
            family="lightgbm",
            profile="P0_ALL_VALID",
            target_variant="surge_d3",
            train_policy="expanding",
            positive_weight_mode="none",
            params={},
        )
    )
    return prepare_data_bundle(args, profiles, compatibility, folds)


def class_and_event_weights(target: np.ndarray, event_weight: np.ndarray, indices: np.ndarray, dates: np.ndarray, half_life: float | None = None) -> np.ndarray:
    y = target[indices].astype(np.uint8)
    w = event_weight[indices].astype(np.float64, copy=True)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives > 0 and negatives > 0:
        pos_weight = min(3.0, math.sqrt(negatives / positives))
        w[y == 1] *= pos_weight
    if half_life is not None and half_life > 0:
        local_dates = pd.to_datetime(pd.Series(dates[indices]), errors="coerce")
        latest = local_dates.max()
        age = (latest - local_dates).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
        w *= np.power(0.5, np.maximum(0.0, age) / half_life)
    mean = float(np.nanmean(w)) if len(w) else 1.0
    if mean > 0:
        w /= mean
    return w.astype(np.float32)


def recipe_feature_indices(recipe: BaseRecipe, profiles: Mapping[str, Sequence[str]], bundle: DataBundle) -> np.ndarray:
    profile = recipe.profile if recipe.profile in profiles else "P1_SELECTION_TOP"
    values = [f for f in profiles[profile] if f in bundle.feature_to_index]
    if not values:
        raise ValueError(f"recipe {recipe.name}에 사용할 feature가 없습니다")
    return np.asarray([bundle.feature_to_index[f] for f in values], dtype=np.int64)


def train_base_oof(
    args: argparse.Namespace,
    bundle: DataBundle,
    folds: Sequence[FoldSpec],
    profiles: Mapping[str, Sequence[str]],
    recipes: Sequence[BaseRecipe],
    event_weight: np.ndarray,
    preflight: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = np.load(bundle.matrix_path, mmap_mode="r")
    y = bundle.targets["surge_d3"].astype(np.uint8)
    all_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    cache_dir = args.output / "base_oof_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        train_idx, valid_idx = fold_indices(bundle, fold)
        fold_frame = bundle.frame.iloc[valid_idx].copy().reset_index(drop=True)
        fold_frame["fold_id"] = int(fold.fold_id)
        fold_frame["target"] = y[valid_idx]
        recipe_scores: dict[str, np.ndarray] = {}
        family_scores: dict[str, list[np.ndarray]] = {}
        for recipe in recipes:
            feature_idx = recipe_feature_indices(recipe, profiles, bundle)
            train_used = apply_training_window(train_idx, bundle.dates, recipe.train_policy, recipe.rolling_days)
            x_train = np.asarray(matrix[np.ix_(train_used, feature_idx)], dtype=np.float32)
            x_valid = np.asarray(matrix[np.ix_(valid_idx, feature_idx)], dtype=np.float32)
            weights = class_and_event_weights(y, event_weight, train_used, bundle.dates, recipe.half_life_days)
            seed_scores: list[np.ndarray] = []
            for seed in args.base_seeds:
                cache = cache_dir / f"fold{fold.fold_id}__{recipe.name}__seed{seed}.npz"
                identity = sha256_bytes(stable_json_bytes({
                    "dataset": bundle.dataset_sha256,
                    "target": bundle.target_sha256,
                    "fold": fold.fold_id,
                    "recipe": recipe.__dict__,
                    "seed": seed,
                    "rows": valid_idx.tolist(),
                }))
                prediction: np.ndarray | None = None
                if args.resume and cache.exists():
                    try:
                        arr = np.load(cache, allow_pickle=False)
                        if str(arr["identity"].item()) == identity and np.array_equal(arr["row_index"], valid_idx):
                            prediction = np.asarray(arr["prediction"], dtype=np.float64)
                    except Exception:
                        prediction = None
                if prediction is None:
                    result = fit_predict_base_family(
                        recipe.family,
                        x_train,
                        y[train_used],
                        weights,
                        x_valid,
                        y[valid_idx],
                        recipe.params,
                        int(seed),
                        args,
                        resolve_model_device(recipe.family, preflight),
                        fixed_iterations=recipe.rounds,
                    )
                    prediction = np.asarray(result.prediction, dtype=np.float64)
                    atomic_write_npz(cache, identity=np.asarray(identity), row_index=valid_idx, prediction=prediction)
                seed_scores.append(datewise_rank(prediction, fold_frame[args.date_column]))
            recipe_score = np.nanmean(np.vstack(seed_scores), axis=0)
            recipe_scores[recipe.name] = recipe_score
            family_scores.setdefault(recipe.family, []).append(recipe_score)
            metrics = binary_metrics(y[valid_idx], recipe_score)
            metric_rows.append({"fold_id": fold.fold_id, "component": recipe.name, "family": recipe.family, **metrics})
            del x_train, x_valid
        family_mean: list[np.ndarray] = []
        for family, values in family_scores.items():
            score = np.nanmean(np.vstack(values), axis=0)
            fold_frame[f"base_family__{family}"] = score
            family_mean.append(score)
        base_score = np.nanmean(np.vstack(family_mean), axis=0)
        fold_frame["base_score"] = base_score
        fold_frame["base_date_rank"] = datewise_rank(base_score, fold_frame[args.date_column])
        metrics = binary_metrics(y[valid_idx], base_score)
        metric_rows.append({"fold_id": fold.fold_id, "component": "base_family_balanced", "family": "ensemble", **metrics})
        all_rows.append(fold_frame)
        log(f"base OOF fold {fold.fold_id} 완료: PR-AUC={metrics['pr_auc']:.4f}")
        gc.collect()
    return pd.concat(all_rows, ignore_index=True), pd.DataFrame(metric_rows)


def build_error_map(
    args: argparse.Namespace,
    bundle: DataBundle,
    base_oof: pd.DataFrame,
    folds: Sequence[FoldSpec],
    profiles: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix = np.load(bundle.matrix_path, mmap_mode="r")
    p0 = [f for f in profiles["P0_ALL_VALID"] if f in bundle.feature_to_index]
    config = ErrorGroupConfig(args.error_top_quantile, args.error_missed_quantile, args.minimum_error_group_rows)
    membership_rows: list[pd.DataFrame] = []
    map_rows: list[dict[str, Any]] = []
    frame_by_row = bundle.frame.reset_index(drop=True)
    for fold_id in args.selection_folds:
        fold = next((f for f in folds if f.fold_id == fold_id), None)
        if fold is None:
            continue
        part = base_oof[base_oof["fold_id"] == fold_id].copy().reset_index(drop=True)
        if part.empty:
            continue
        valid_idx = frame_by_row.index[
            (pd.to_datetime(frame_by_row[args.date_column]) >= fold.validation_start)
            & (pd.to_datetime(frame_by_row[args.date_column]) <= fold.validation_end)
        ].to_numpy(dtype=np.int64)
        # fold_indices와 같은 정렬임을 source_row_id가 있으면 확인한다.
        if len(valid_idx) != len(part):
            _, valid_idx = fold_indices(bundle, fold)
        group_frame = assign_error_groups(part["target"], part["base_score"], part[args.date_column], config)
        membership = part[[c for c in ["source_row_id", args.date_column, args.ticker_column, "fold_id", "target", "base_score", "base_date_rank"] if c in part]].copy()
        membership["error_group"] = group_frame["error_group"].to_numpy(dtype=object)
        membership["row_index"] = valid_idx
        membership_rows.append(membership)
        counts = membership["error_group"].value_counts().to_dict()
        log(f"error groups fold {fold_id}: {counts}")
        raw = np.asarray(matrix[np.ix_(valid_idx, [bundle.feature_to_index[f] for f in p0])], dtype=np.float32)
        raw_df = pd.DataFrame(raw, columns=p0)
        rank_df = raw_df.groupby(pd.to_datetime(part[args.date_column]), sort=False).rank(method="average", pct=True)
        groups = membership["error_group"].to_numpy(dtype=object)
        contrasts = [
            ("A_VS_B", "A_TOP_TRUE_POSITIVE", "B_TOP_FALSE_POSITIVE"),
            ("C_VS_B", "C_LOW_MISSED_POSITIVE", "B_TOP_FALSE_POSITIVE"),
            ("A_VS_C", "A_TOP_TRUE_POSITIVE", "C_LOW_MISSED_POSITIVE"),
        ]
        for j, feature in enumerate(p0):
            raw_col = raw[:, j].astype(np.float64, copy=False)
            rank_col = rank_df.iloc[:, j].to_numpy(dtype=np.float64)
            for contrast, pos_group, neg_group in contrasts:
                pos_count = int(np.sum(groups == pos_group))
                neg_count = int(np.sum(groups == neg_group))
                if min(pos_count, neg_count) < args.minimum_error_group_rows:
                    continue
                map_rows.append(contrast_feature_stats(feature, raw_col, rank_col, groups, pos_group, neg_group, fold_id, contrast))
        del raw, raw_df, rank_df
        gc.collect()
    membership_all = pd.concat(membership_rows, ignore_index=True) if membership_rows else pd.DataFrame()
    map_by_fold = pd.DataFrame(map_rows)
    summary = summarize_error_map(map_by_fold)
    return membership_all, map_by_fold, summary


def summary_for_prior_folds(map_by_fold: pd.DataFrame, prior_folds: Sequence[int]) -> pd.DataFrame:
    if not prior_folds or map_by_fold.empty:
        return pd.DataFrame()
    return summarize_error_map(map_by_fold[map_by_fold["fold_id"].isin([int(v) for v in prior_folds])])


def selected_features_for_fold(
    args: argparse.Namespace,
    fold_id: int,
    map_by_fold: pd.DataFrame,
    profiles: Mapping[str, Sequence[str]],
) -> list[str]:
    prior = [f for f in args.selection_folds if f < fold_id]
    if fold_id not in args.selection_folds:
        prior = list(args.selection_folds)
    summary = summary_for_prior_folds(map_by_fold, prior)
    extras = []
    for profile_name in ["P7S_R2", "P7S_UNIVARIATE_TOP3_LIFT"]:
        extras.extend(profiles.get(profile_name, [])[: args.error_extra_profile_features])
    features = select_error_features(
        summary,
        args.error_feature_count,
        args.minimum_contrast_score,
        extra_features=extras,
    )
    p0set = set(profiles["P0_ALL_VALID"])
    return [f for f in dict.fromkeys(features) if f in p0set][: args.error_feature_count]


def build_relative_for_rows(
    bundle: DataBundle,
    matrix: np.ndarray,
    rows: np.ndarray,
    selected: Sequence[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if not selected:
        return pd.DataFrame(index=np.arange(len(rows)))
    idx = np.asarray([bundle.feature_to_index[f] for f in selected], dtype=np.int64)
    raw = np.asarray(matrix[np.ix_(rows, idx)], dtype=np.float32)
    metadata = bundle.frame.iloc[rows].reset_index(drop=True)
    spec = RelativeFeatureSpec(tuple(selected))
    frame = build_relative_feature_frame(raw, list(selected), metadata, spec, args.date_column, "market", "bucket")
    # scope one-hot은 train/validation category 빈도 누수를 만들 수 있어 여기서는 사용하지 않는다.
    # 시장/버킷 조건부 정보는 동일 날짜 group-rank/delta와 별도 scope model로 처리한다.
    return frame


def align_feature_frames(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = sorted(set(train.columns) | set(valid.columns))
    return train.reindex(columns=columns, fill_value=0.0), valid.reindex(columns=columns, fill_value=0.0)


def fit_binary_custom(
    family: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    weights: np.ndarray,
    x_valid: pd.DataFrame,
    args: argparse.Namespace,
    preflight: Mapping[str, Any],
    seed: int,
    rounds: int,
    params: Mapping[str, Any],
) -> np.ndarray:
    x_train, x_valid = align_feature_frames(x_train, x_valid)
    result = fit_predict_base_family(
        family,
        x_train.to_numpy(dtype=np.float32),
        y_train.astype(np.uint8),
        weights.astype(np.float32),
        x_valid.to_numpy(dtype=np.float32),
        None,
        params,
        seed,
        args,
        resolve_model_device(family, preflight),
        fixed_iterations=rounds,
    )
    return np.asarray(result.prediction, dtype=np.float64)


def train_verifier_components(
    args: argparse.Namespace,
    bundle: DataBundle,
    matrix: np.ndarray,
    membership: pd.DataFrame,
    fold_id: int,
    valid_idx: np.ndarray,
    selected: Sequence[str],
    preflight: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    prior = [f for f in args.selection_folds if f < fold_id]
    if fold_id not in args.selection_folds:
        prior = list(args.selection_folds)
    train_meta = membership[membership["fold_id"].isin(prior)].copy()
    if train_meta.empty or not selected:
        return {}
    train_groups = train_meta["error_group"].astype(str).to_numpy(dtype=object)
    ab_mask = np.isin(train_groups, ["A_TOP_TRUE_POSITIVE", "B_TOP_FALSE_POSITIVE"])
    if int(np.sum(ab_mask)) < args.minimum_verifier_rows or len(np.unique((train_groups[ab_mask] == "A_TOP_TRUE_POSITIVE").astype(int))) < 2:
        return {}
    train_rows = train_meta.loc[ab_mask, "row_index"].to_numpy(dtype=np.int64)
    combined_rows = np.concatenate([train_rows, valid_idx])
    relative = build_relative_for_rows(bundle, matrix, combined_rows, selected, args)
    x_train = relative.iloc[: len(train_rows)].reset_index(drop=True)
    x_valid = relative.iloc[len(train_rows) :].reset_index(drop=True)
    y_train = (train_meta.loc[ab_mask, "error_group"].astype(str).to_numpy() == "A_TOP_TRUE_POSITIVE").astype(np.uint8)
    weights2 = np.where(y_train == 0, args.verifier_false_positive_cost, 1.0).astype(np.float32)
    outputs: dict[str, np.ndarray] = {}
    lgb_params = {"learning_rate": 0.035, "num_leaves": 24, "min_data_in_leaf": 30, "feature_fraction": 0.82, "bagging_fraction": 0.85, "bagging_freq": 1, "lambda_l2": 5.0, "verbosity": -1}
    xgb_params = {"eta": 0.035, "max_depth": 5, "min_child_weight": 8.0, "subsample": 0.82, "colsample_bytree": 0.82, "lambda": 5.0, "alpha": 0.2, "tree_method": "hist"}
    outputs["verifier_lgb"] = fit_binary_custom("lightgbm", x_train, y_train, weights2, x_valid, args, preflight, args.model_seed + 11, args.verifier_rounds, lgb_params)
    if not args.quick:
        outputs["verifier_xgb"] = fit_binary_custom("xgboost", x_train, y_train, weights2, x_valid, args, preflight, args.model_seed + 17, args.verifier_rounds, xgb_params)
    # Pairwise: A/C positive relevance vs hard B negative.
    all_train_rows = train_meta["row_index"].to_numpy(dtype=np.int64)
    pair_combined = np.concatenate([all_train_rows, valid_idx])
    pair_relative = build_relative_for_rows(bundle, matrix, pair_combined, selected, args)
    pair_train = pair_relative.iloc[: len(all_train_rows)].reset_index(drop=True)
    pair_valid = pair_relative.iloc[len(all_train_rows) :].reset_index(drop=True)
    pair_train, pair_valid = align_feature_frames(pair_train, pair_valid)
    x_pair, y_pair, qid, _ = build_pairwise_training_frame(pair_train, train_meta["error_group"].astype(str), train_meta[args.date_column])
    if len(x_pair) >= args.minimum_pairwise_rows and len(np.unique(y_pair)) >= 2:
        lgb_rank = fit_lgb_ranker(x_pair, y_pair, qid, {}, args.pairwise_rounds, args.model_seed + 31, args.threads_per_model)
        outputs["pairwise_lgb"] = datewise_rank(predict_ranker(lgb_rank, "lightgbm", pair_valid.to_numpy(dtype=np.float32)), bundle.frame.iloc[valid_idx][args.date_column])
        if not args.quick:
            xgb_rank = fit_xgb_ranker(x_pair, y_pair, qid, {}, args.pairwise_rounds, args.model_seed + 37, args.xgboost_threads, resolve_model_device("xgboost", preflight))
            outputs["pairwise_xgb"] = datewise_rank(predict_ranker(xgb_rank, "xgboost", pair_valid.to_numpy(dtype=np.float32)), bundle.frame.iloc[valid_idx][args.date_column])
    return outputs


def recent_indices(train_idx: np.ndarray, dates: np.ndarray, rolling_days: int | None) -> np.ndarray:
    if rolling_days is None:
        return train_idx
    return apply_training_window(train_idx, dates, "rolling", rolling_days)


def fit_recent_and_event_components(
    args: argparse.Namespace,
    bundle: DataBundle,
    matrix: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    selected: Sequence[str],
    event_info: pd.DataFrame,
    preflight: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    if not selected:
        return {}
    target = bundle.targets["surge_d3"].astype(np.uint8)
    event_weight = event_info["event_weight"].to_numpy(dtype=np.float64)
    outputs: dict[str, np.ndarray] = {}
    specs = [
        ("recent_lgb_decay126", "lightgbm", None, 126.0),
        ("recent_lgb_roll504", "lightgbm", 504, None),
    ]
    if not args.quick:
        specs.extend([
            ("recent_xgb_decay252", "xgboost", None, 252.0),
            ("recent_xgb_roll756", "xgboost", 756, None),
        ])
    for name, family, rolling, half_life in specs:
        train_used = recent_indices(train_idx, bundle.dates, rolling)
        combined = np.concatenate([train_used, valid_idx])
        relative = build_relative_for_rows(bundle, matrix, combined, selected, args)
        x_train = relative.iloc[: len(train_used)].reset_index(drop=True)
        x_valid = relative.iloc[len(train_used) :].reset_index(drop=True)
        weights = class_and_event_weights(target, event_weight, train_used, bundle.dates, half_life)
        params = (
            {"learning_rate": 0.035, "num_leaves": 31, "min_data_in_leaf": 35, "feature_fraction": 0.80, "bagging_fraction": 0.82, "bagging_freq": 1, "lambda_l2": 5.0, "verbosity": -1}
            if family == "lightgbm"
            else {"eta": 0.035, "max_depth": 5, "min_child_weight": 8.0, "subsample": 0.82, "colsample_bytree": 0.80, "lambda": 5.0, "alpha": 0.2, "tree_method": "hist"}
        )
        pred = fit_binary_custom(family, x_train, target[train_used], weights, x_valid, args, preflight, args.model_seed + len(name), args.recent_rounds, params)
        outputs[name] = datewise_rank(pred, bundle.frame.iloc[valid_idx][args.date_column])
    # Event-start specialist: new rally onset rather than repeated positive windows.
    event_target = event_info["event_start"].to_numpy(dtype=np.uint8)
    event_train = recent_indices(train_idx, bundle.dates, 756)
    if int(event_target[event_train].sum()) >= args.minimum_event_positives:
        combined = np.concatenate([event_train, valid_idx])
        relative = build_relative_for_rows(bundle, matrix, combined, selected, args)
        x_train = relative.iloc[: len(event_train)].reset_index(drop=True)
        x_valid = relative.iloc[len(event_train) :].reset_index(drop=True)
        y_event = event_target[event_train]
        neg = max(1, int(np.sum(y_event == 0)))
        pos = max(1, int(np.sum(y_event == 1)))
        weights = np.where(y_event == 1, min(6.0, math.sqrt(neg / pos)), 1.0).astype(np.float32)
        params = {"learning_rate": 0.035, "num_leaves": 24, "min_data_in_leaf": 35, "feature_fraction": 0.82, "bagging_fraction": 0.85, "bagging_freq": 1, "lambda_l2": 5.0, "verbosity": -1}
        event_pred = fit_binary_custom("lightgbm", x_train, y_event, weights, x_valid, args, preflight, args.model_seed + 101, args.event_rounds, params)
        outputs["event_start_score"] = datewise_rank(event_pred, bundle.frame.iloc[valid_idx][args.date_column])
    return outputs


def fit_scope_conditional_component(
    args: argparse.Namespace,
    bundle: DataBundle,
    matrix: np.ndarray,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    selected: Sequence[str],
    event_info: pd.DataFrame,
    preflight: Mapping[str, Any],
) -> np.ndarray | None:
    if not selected:
        return None
    target = bundle.targets["surge_d3"].astype(np.uint8)
    train_used = recent_indices(train_idx, bundle.dates, 756)
    combined = np.concatenate([train_used, valid_idx])
    relative = build_relative_for_rows(bundle, matrix, combined, selected, args)
    x_train = relative.iloc[: len(train_used)].reset_index(drop=True)
    x_valid = relative.iloc[len(train_used) :].reset_index(drop=True)
    weights = class_and_event_weights(target, event_info["event_weight"].to_numpy(dtype=np.float64), train_used, bundle.dates, 252.0)
    params = {"learning_rate": 0.035, "num_leaves": 24, "min_data_in_leaf": 30, "feature_fraction": 0.82, "bagging_fraction": 0.85, "bagging_freq": 1, "lambda_l2": 5.0, "verbosity": -1}
    global_pred = fit_binary_custom("lightgbm", x_train, target[train_used], weights, x_valid, args, preflight, args.model_seed + 211, args.scope_rounds, params)
    result = global_pred.copy()
    train_meta = bundle.frame.iloc[train_used].reset_index(drop=True)
    valid_meta = bundle.frame.iloc[valid_idx].reset_index(drop=True)
    scopes: list[tuple[str, str]] = []
    for column in ["market", "bucket", "industry_name"]:
        if column not in train_meta.columns:
            continue
        counts = train_meta[column].astype("string").value_counts()
        for level, count in counts.items():
            if int(count) >= args.minimum_scope_train_rows:
                scopes.append((column, str(level)))
    scopes = scopes[: args.max_scope_models]
    for column, level in scopes:
        train_mask = scope_level_mask(train_meta[column], level)
        valid_mask = scope_level_mask(valid_meta[column], level)
        if int(valid_mask.sum()) == 0 or int(train_mask.sum()) < args.minimum_scope_train_rows:
            continue
        y_scope = target[train_used][train_mask]
        if int(y_scope.sum()) < args.minimum_scope_positives or int((1 - y_scope).sum()) < args.minimum_scope_negatives:
            continue
        pred = fit_binary_custom(
            "lightgbm",
            x_train.loc[train_mask].reset_index(drop=True),
            y_scope,
            weights[train_mask],
            x_valid.loc[valid_mask].reset_index(drop=True),
            args,
            preflight,
            args.model_seed + 307 + len(level),
            args.scope_rounds,
            params,
        )
        result[valid_mask] = 0.55 * global_pred[valid_mask] + 0.45 * pred
    return datewise_rank(result, valid_meta[args.date_column])


def build_component_oof(
    args: argparse.Namespace,
    bundle: DataBundle,
    folds: Sequence[FoldSpec],
    profiles: Mapping[str, Sequence[str]],
    base_oof: pd.DataFrame,
    membership: pd.DataFrame,
    map_by_fold: pd.DataFrame,
    event_info: pd.DataFrame,
    preflight: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    matrix = np.load(bundle.matrix_path, mmap_mode="r")
    y = bundle.targets["surge_d3"].astype(np.uint8)
    frames: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    feature_manifest: dict[str, list[str]] = {}
    for fold in folds:
        train_idx, valid_idx = fold_indices(bundle, fold)
        base_part = base_oof[base_oof["fold_id"] == fold.fold_id].copy().reset_index(drop=True)
        if len(base_part) != len(valid_idx):
            raise ValueError(f"fold {fold.fold_id}: base OOF row mismatch")
        selected = selected_features_for_fold(args, fold.fold_id, map_by_fold, profiles)
        feature_manifest[str(fold.fold_id)] = selected
        part = base_part.copy()
        verifier = train_verifier_components(args, bundle, matrix, membership, fold.fold_id, valid_idx, selected, preflight)
        for name, values in verifier.items():
            part[name] = values
        recent = fit_recent_and_event_components(args, bundle, matrix, train_idx, valid_idx, selected, event_info, preflight)
        for name, values in recent.items():
            part[name] = values
        scope_score = fit_scope_conditional_component(args, bundle, matrix, train_idx, valid_idx, selected, event_info, preflight)
        if scope_score is not None:
            part["scope_conditional_score"] = scope_score
        verifier_cols = [c for c in ["verifier_lgb", "verifier_xgb"] if c in part]
        pair_cols = [c for c in ["pairwise_lgb", "pairwise_xgb"] if c in part]
        recent_cols = [c for c in ["recent_lgb_decay126", "recent_lgb_roll504", "recent_xgb_decay252", "recent_xgb_roll756"] if c in part]
        part["verifier_mean"] = part[verifier_cols].mean(axis=1) if verifier_cols else part["base_score"]
        part["pairwise_mean"] = part[pair_cols].mean(axis=1) if pair_cols else part["base_score"]
        part["recent_mean"] = part[recent_cols].mean(axis=1) if recent_cols else part["base_score"]
        if "event_start_score" not in part:
            part["event_start_score"] = part["base_score"]
        if "scope_conditional_score" not in part:
            part["scope_conditional_score"] = part["base_score"]
        part["event_id"] = event_info.iloc[valid_idx]["event_id"].to_numpy(dtype=np.int64)
        part["event_start"] = event_info.iloc[valid_idx]["event_start"].to_numpy(dtype=np.uint8)
        part["selected_error_feature_count"] = len(selected)
        for method in BASE_METHODS:
            part[f"score__{method}"] = method_score(part, method)
            bm = binary_metrics(y[valid_idx], part[f"score__{method}"])
            metrics.append({"fold_id": fold.fold_id, "method": method, "stage": "component", **bm})
        frames.append(part)
        log(f"V7 component fold {fold.fold_id} 완료: selected_features={len(selected)}")
        gc.collect()
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metrics), feature_manifest


def meta_feature_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "base_score",
        "verifier_mean",
        "pairwise_mean",
        "recent_mean",
        "event_start_score",
        "scope_conditional_score",
        "verifier_lgb",
        "verifier_xgb",
        "pairwise_lgb",
        "pairwise_xgb",
        "recent_lgb_decay126",
        "recent_lgb_roll504",
        "recent_xgb_decay252",
        "recent_xgb_roll756",
    ]
    return [c for c in preferred if c in frame.columns]


def enrich_meta_features(frame: pd.DataFrame) -> pd.DataFrame:
    columns = meta_feature_columns(frame)
    out = frame[columns].copy()
    for c in columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    def get(c: str) -> pd.Series:
        return out[c] if c in out else pd.Series(0.5, index=out.index)
    out["interaction__base_verifier"] = get("base_score") * get("verifier_mean")
    out["interaction__base_pair"] = get("base_score") * get("pairwise_mean")
    out["interaction__pair_verifier"] = get("pairwise_mean") * get("verifier_mean")
    out["interaction__recent_verifier"] = get("recent_mean") * get("verifier_mean")
    out["min__base_verifier_pair"] = pd.concat([get("base_score"), get("verifier_mean"), get("pairwise_mean")], axis=1).min(axis=1)
    out["max__base_minus_verifier"] = get("base_score") - get("verifier_mean")
    out["missed_positive_recovery"] = get("pairwise_mean") - get("base_score")
    return out.replace([np.inf, -np.inf], np.nan)


def fit_meta_model(kind: str, train: pd.DataFrame, target: np.ndarray, args: argparse.Namespace, preflight: Mapping[str, Any]) -> MetaModel:
    x = enrich_meta_features(train)
    # 이전 fold에서 아직 생성되지 않은 component는 전체 NaN일 수 있다.
    # 그런 열은 meta 학습에서 제거해 train/predict schema와 imputer 동작을 안정화한다.
    usable = [c for c in x.columns if pd.to_numeric(x[c], errors="coerce").notna().any()]
    x = x[usable].copy()
    columns = list(x.columns)
    if not columns:
        raise ValueError("meta model에 사용할 유효 component feature가 없습니다")
    if kind == "meta_logistic":
        imputer = SimpleImputer(strategy="median", add_indicator=True)
        scaler = StandardScaler()
        xi = imputer.fit_transform(x)
        xs = scaler.fit_transform(xi)
        weights = np.where(target == 0, args.meta_false_positive_cost, 1.0)
        model = LogisticRegression(C=0.35, max_iter=1500, random_state=args.model_seed, solver="lbfgs")
        model.fit(xs, target, sample_weight=weights)
        return MetaModel(kind, model, imputer, scaler, columns)
    y = target.astype(np.uint8)
    weights = np.where(y == 0, args.meta_false_positive_cost, 1.0).astype(np.float32)
    # fit_predict_base_family cannot return the fitted object, so meta tree models are fit directly.
    if kind == "meta_lgb":
        import lightgbm as lgb
        resolved = {
            "objective": "binary",
            "metric": "binary_logloss",
            "learning_rate": 0.025,
            "num_leaves": 15,
            "min_data_in_leaf": 40,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.9,
            "bagging_freq": 1,
            "lambda_l2": 7.0,
            "verbosity": -1,
            "num_threads": args.threads_per_model,
            "seed": args.model_seed,
        }
        model = lgb.train(resolved, lgb.Dataset(x.to_numpy(dtype=np.float32), label=y, weight=weights), num_boost_round=args.meta_rounds, callbacks=[lgb.log_evaluation(period=0)])
        return MetaModel(kind, model, None, None, columns)
    if kind == "meta_xgb":
        import xgboost as xgb
        device = resolve_model_device("xgboost", preflight)
        dtrain = xgb.DMatrix(x.to_numpy(dtype=np.float32), label=y, weight=weights)
        resolved = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "device": device,
            "eta": 0.025,
            "max_depth": 3,
            "min_child_weight": 12,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "lambda": 7.0,
            "alpha": 0.5,
            "seed": args.model_seed,
            "nthread": args.xgboost_threads,
        }
        model = xgb.train(resolved, dtrain, num_boost_round=args.meta_rounds, verbose_eval=False)
        return MetaModel(kind, model, None, None, columns)
    raise ValueError(kind)


def predict_meta_model(model: MetaModel, frame: pd.DataFrame) -> np.ndarray:
    x = enrich_meta_features(frame).reindex(columns=model.feature_columns)
    if model.kind == "meta_logistic":
        xi = model.imputer.transform(x)
        xs = model.scaler.transform(xi)
        return np.asarray(model.model.predict_proba(xs)[:, 1], dtype=np.float64)
    if model.kind == "meta_lgb":
        return np.asarray(model.model.predict(x.to_numpy(dtype=np.float32)), dtype=np.float64)
    if model.kind == "meta_xgb":
        import xgboost as xgb
        return np.asarray(model.model.predict(xgb.DMatrix(x.to_numpy(dtype=np.float32))), dtype=np.float64)
    raise ValueError(model.kind)


def forward_meta_scores(
    args: argparse.Namespace,
    components: pd.DataFrame,
    preflight: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = components.copy()
    audit_rows: list[dict[str, Any]] = []
    selection_set = set(args.selection_folds)
    for kind in META_METHODS:
        result[f"score__{kind}"] = np.nan
    for heldout in args.selection_folds:
        prior = [f for f in args.selection_folds if f < heldout]
        train = result[result["fold_id"].isin(prior)].copy()
        valid_mask = result["fold_id"].eq(heldout)
        valid = result.loc[valid_mask].copy()
        if len(prior) < args.minimum_meta_prior_folds or len(train) < args.minimum_meta_rows or valid.empty:
            audit_rows.append({"heldout_fold": heldout, "train_folds": prior, "status": "WARMUP"})
            continue
        y_train = train["target"].to_numpy(dtype=np.uint8)
        for kind in META_METHODS if not args.quick else ["meta_logistic", "meta_lgb"]:
            model = fit_meta_model(kind, train, y_train, args, preflight)
            pred = predict_meta_model(model, valid)
            result.loc[valid_mask, f"score__{kind}"] = datewise_rank(pred, valid[args.date_column])
        audit_rows.append({"heldout_fold": heldout, "train_folds": prior, "status": "OK", "strictly_past_only": bool(all(f < heldout for f in prior))})
    # Holdout folds: meta is trained on all selection component OOF only.
    train_selection = result[result["fold_id"].isin(args.selection_folds)].copy()
    if len(train_selection) >= args.minimum_meta_rows:
        y_train = train_selection["target"].to_numpy(dtype=np.uint8)
        final_models: dict[str, MetaModel] = {}
        for kind in META_METHODS if not args.quick else ["meta_logistic", "meta_lgb"]:
            final_models[kind] = fit_meta_model(kind, train_selection, y_train, args, preflight)
        holdout_mask = ~result["fold_id"].isin(args.selection_folds)
        holdout = result.loc[holdout_mask].copy()
        for kind, model in final_models.items():
            pred = predict_meta_model(model, holdout)
            result.loc[holdout_mask, f"score__{kind}"] = datewise_rank(pred, holdout[args.date_column])
    return result, pd.DataFrame(audit_rows)


def candidate_methods(components: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    methods = list(BASE_METHODS)
    for method in META_METHODS:
        if f"score__{method}" in components.columns and pd.to_numeric(components[f"score__{method}"], errors="coerce").notna().any():
            methods.append(method)
    return methods


def forward_policy_evaluation(
    args: argparse.Namespace,
    scored: pd.DataFrame,
    methods: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, Any]] = []
    search_rows: list[pd.DataFrame] = []
    eval_folds = [f for f in args.selection_folds if f >= args.policy_start_fold]
    for method in methods:
        score_col = f"score__{method}"
        if score_col not in scored:
            continue
        for heldout in eval_folds:
            train_folds = [f for f in args.selection_folds if f < heldout and f >= args.meta_start_fold]
            train = scored[scored["fold_id"].isin(train_folds)].copy()
            valid = scored[scored["fold_id"].eq(heldout)].copy()
            if train.empty or valid.empty or pd.to_numeric(train[score_col], errors="coerce").notna().sum() < args.minimum_policy_rows:
                continue
            threshold_rule, threshold_search, threshold_diag = select_precision_threshold(
                train,
                "target",
                score_col,
                args.date_column,
                "event_id",
                args.target_precision + args.selection_precision_buffer,
                args.minimum_precision_lcb,
                args.minimum_alerts,
                args.minimum_alert_days,
                args.minimum_recall,
                args.confidence_level,
                args.maximum_threshold_candidates,
            )
            cascade_rule, cascade_search, cascade_diag = search_precision_cascade(
                train,
                "target",
                score_col,
                args.date_column,
                "event_id",
                args.target_precision + args.selection_precision_buffer,
                args.minimum_precision_lcb,
                args.minimum_alerts,
                args.minimum_alert_days,
                args.minimum_recall,
                args.confidence_level,
                args.cascade_quantiles,
            )
            choices = [(threshold_rule, threshold_diag), (cascade_rule, cascade_diag)]
            safe = [item for item in choices if item[0].gate_pass]
            if safe:
                # train에서 안전한 정책 중 recall이 높은 정책. 같은 recall이면 cascade보다 단순 threshold 우선.
                def train_metric(item: tuple[PrecisionRule, dict[str, Any]]) -> tuple[float, float, int]:
                    rule = item[0]
                    alert = apply_rule(train, rule, score_col)
                    m = evaluate_alert_mask(train["target"], alert, train[args.date_column], train["event_id"], args.confidence_level)
                    return (float(m["recall"]), float(m["precision"]), 1 if rule.kind == "threshold" else 0)
                safe.sort(key=train_metric, reverse=True)
                chosen = safe[0][0]
            else:
                # 정책 실패 시 production-like no-alert. diagnostic은 별도 저장.
                chosen = PrecisionRule(kind="no_alert", target_precision=args.target_precision, minimum_lcb=args.minimum_precision_lcb, minimum_alerts=args.minimum_alerts, minimum_alert_days=args.minimum_alert_days, minimum_recall=args.minimum_recall)
            alert = apply_rule(valid, chosen, score_col)
            metrics = evaluate_alert_mask(valid["target"], alert, valid[args.date_column], valid["event_id"], args.confidence_level)
            gate = rule_gate(metrics, PrecisionRule(**{**chosen.__dict__, "target_precision": args.target_precision}))
            bm = binary_metrics(valid["target"], pd.to_numeric(valid[score_col], errors="coerce"))
            metric_rows.append({
                "method": method,
                "heldout_fold": heldout,
                "train_folds": ",".join(str(v) for v in train_folds),
                "policy_kind": chosen.kind,
                "train_policy_gate": chosen.gate_pass,
                "heldout_gate": gate,
                **metrics,
                "pr_auc": bm["pr_auc"],
                "roc_auc": bm["roc_auc"],
                "diagnostic_threshold_best_precision": (threshold_diag.get("metrics", {}) or {}).get("precision"),
                "diagnostic_cascade_best_precision": (cascade_diag.get("metrics", {}) or {}).get("precision"),
            })
            threshold_search["method"] = method
            threshold_search["heldout_fold"] = heldout
            threshold_search["search_kind"] = "threshold"
            cascade_search["method"] = method
            cascade_search["heldout_fold"] = heldout
            cascade_search["search_kind"] = "cascade"
            search_rows.extend([threshold_search, cascade_search])
    metrics = pd.DataFrame(metric_rows)
    searches = pd.concat(search_rows, ignore_index=True) if search_rows else pd.DataFrame()
    return metrics, searches


def summarize_methods(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for method, part in metrics.groupby("method", sort=False):
        precision = pd.to_numeric(part["precision"], errors="coerce")
        recall = pd.to_numeric(part["recall"], errors="coerce")
        practical = pd.concat([
            pd.to_numeric(part["diagnostic_threshold_best_precision"], errors="coerce"),
            pd.to_numeric(part["diagnostic_cascade_best_precision"], errors="coerce"),
        ], axis=1).max(axis=1)
        rows.append({
            "method": method,
            "folds": int(len(part)),
            "heldout_gate_pass_rate": float(part["heldout_gate"].mean()),
            "worst_precision": float(precision.min()) if precision.notna().any() else math.nan,
            "mean_precision": float(precision.mean()) if precision.notna().any() else math.nan,
            "worst_recall": float(recall.min()) if recall.notna().any() else math.nan,
            "mean_recall": float(recall.mean()) if recall.notna().any() else math.nan,
            "mean_pr_auc": float(pd.to_numeric(part["pr_auc"], errors="coerce").mean()),
            "worst_pr_auc": float(pd.to_numeric(part["pr_auc"], errors="coerce").min()),
            "mean_diagnostic_best_precision": float(practical.mean()) if practical.notna().any() else math.nan,
            "worst_diagnostic_best_precision": float(practical.min()) if practical.notna().any() else math.nan,
        })
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["heldout_gate_pass_rate", "worst_precision", "mean_diagnostic_best_precision", "worst_pr_auc", "mean_pr_auc"],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def choose_method(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "balanced_blend"
    return str(summary.iloc[0]["method"])


def final_policy_and_holdout(
    args: argparse.Namespace,
    scored: pd.DataFrame,
    method: str,
) -> tuple[PrecisionRule, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    score_col = f"score__{method}"
    selection = scored[scored["fold_id"].isin(args.selection_folds) & (scored["fold_id"] >= args.meta_start_fold)].copy()
    threshold_rule, threshold_search, threshold_diag = select_precision_threshold(
        selection,
        "target",
        score_col,
        args.date_column,
        "event_id",
        args.target_precision + args.selection_precision_buffer,
        args.minimum_precision_lcb,
        args.minimum_alerts,
        args.minimum_alert_days,
        args.minimum_recall,
        args.confidence_level,
        args.maximum_threshold_candidates,
    )
    cascade_rule, cascade_search, cascade_diag = search_precision_cascade(
        selection,
        "target",
        score_col,
        args.date_column,
        "event_id",
        args.target_precision + args.selection_precision_buffer,
        args.minimum_precision_lcb,
        args.minimum_alerts,
        args.minimum_alert_days,
        args.minimum_recall,
        args.confidence_level,
        args.cascade_quantiles,
    )
    rules = [r for r in [threshold_rule, cascade_rule] if r.gate_pass]
    if rules:
        def selection_key(rule: PrecisionRule) -> tuple[float, float, int]:
            alert = apply_rule(selection, rule, score_col)
            m = evaluate_alert_mask(selection["target"], alert, selection[args.date_column], selection["event_id"], args.confidence_level)
            return float(m["recall"]), float(m["precision"]), 1 if rule.kind == "threshold" else 0
        rules.sort(key=selection_key, reverse=True)
        final_rule = rules[0]
    else:
        final_rule = PrecisionRule(kind="no_alert", score_threshold=math.inf, target_precision=args.target_precision, minimum_lcb=args.minimum_precision_lcb, minimum_alerts=args.minimum_alerts, minimum_alert_days=args.minimum_alert_days, minimum_recall=args.minimum_recall, gate_pass=False)
    role_map = {
        **{f: "selection" for f in args.selection_folds},
        **{f: "confirmation" for f in args.confirmation_folds},
        **{f: "recent" for f in args.recent_folds},
    }
    rows: list[dict[str, Any]] = []
    for fold_id, part in scored.groupby("fold_id", sort=True):
        alert = apply_rule(part, final_rule, score_col)
        metrics = evaluate_alert_mask(part["target"], alert, part[args.date_column], part["event_id"], args.confidence_level)
        gate_rule = PrecisionRule(**{**final_rule.__dict__, "target_precision": args.target_precision})
        bm = binary_metrics(part["target"], pd.to_numeric(part[score_col], errors="coerce"))
        rows.append({"fold_id": int(fold_id), "role": role_map.get(int(fold_id), "other"), "method": method, "policy_kind": final_rule.kind, "gate_pass": rule_gate(metrics, gate_rule), **metrics, "pr_auc": bm["pr_auc"], "roc_auc": bm["roc_auc"]})
    fold_metrics = pd.DataFrame(rows)
    role_metrics = fold_metrics.groupby("role", as_index=False).agg(
        folds=("fold_id", "count"),
        gate_pass_rate=("gate_pass", "mean"),
        mean_precision=("precision", "mean"),
        min_precision=("precision", "min"),
        mean_recall=("recall", "mean"),
        min_recall=("recall", "min"),
        mean_alerts=("alerts", "mean"),
        mean_event_recall=("event_recall", "mean"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_roc_auc=("roc_auc", "mean"),
    )
    diagnostics = {"threshold": threshold_diag, "cascade": cascade_diag}
    search = pd.concat([
        threshold_search.assign(search_kind="threshold"),
        cascade_search.assign(search_kind="cascade"),
    ], ignore_index=True)
    return final_rule, fold_metrics, role_metrics, {"diagnostics": diagnostics, "search": search}


def build_bottleneck_report(
    args: argparse.Namespace,
    error_summary: pd.DataFrame,
    method_summary: pd.DataFrame,
    fold_metrics: pd.DataFrame,
    final_rule: PrecisionRule,
) -> tuple[str, dict[str, Any]]:
    holdout = fold_metrics[fold_metrics["role"].isin(["confirmation", "recent"])] if not fold_metrics.empty else pd.DataFrame()
    gate_pass = bool(not holdout.empty and holdout["gate_pass"].all() and final_rule.gate_pass)
    if gate_pass:
        status = "READY_FOR_NEW_FUTURE_HOLDOUT"
    else:
        status = "STOP_PRECISION70_GATE"
    practical = []
    if not fold_metrics.empty:
        practical = fold_metrics[["fold_id", "role", "precision", "recall", "alerts", "precision_lcb", "pr_auc", "roc_auc", "gate_pass"]].to_dict("records")
    top_features = []
    if not error_summary.empty:
        top = error_summary.sort_values("bottleneck_score", ascending=False).drop_duplicates("feature").head(30)
        top_features = top[["feature", "bottleneck_score", "contrast", "contrast_score"]].to_dict("records")
    payload = with_checksum({
        "schema": "crashwatch_surge_hardfp_gap_report_v7",
        "status": status,
        "target_precision": args.target_precision,
        "final_rule": final_rule.to_dict(),
        "fold_metrics": practical,
        "top_error_map_features": top_features,
        "method_summary": method_summary.head(20).to_dict("records") if not method_summary.empty else [],
    })
    lines = [
        "# CrashWatch Surge Hard-FP V7 병목 보고서",
        "",
        f"최종 상태: **{status}**",
        "",
        f"목표는 경보 Precision {args.target_precision:.0%} 이상이며 경보량 상한은 없습니다. 다만 최소 경보수·경보일·Recall·Wilson 하한은 유지합니다.",
        "",
        "## 이번 버전이 직접 해결하려는 병목",
        "",
        "1. A(상위 점수 TP)와 B(상위 점수 FP)의 피처 차이를 직접 지도화",
        "2. C(낮은 점수 양성)를 B보다 위로 올리는 pairwise ranker",
        "3. 최근 126~756 거래일 중심 recent specialist",
        "4. 연속 양성 run을 event로 묶은 event-start specialist와 event-balanced weight",
        "5. 날짜·시장·버킷 횡단면 rank/z/delta 상대피처",
        "6. 시장/버킷 조건부 LightGBM 및 forward-only meta",
        "",
        "## Fold별 결과",
        "",
    ]
    if fold_metrics.empty:
        lines.append("결과 없음")
    else:
        lines.append(fold_metrics.to_markdown(index=False))
    lines.extend(["", "## 오류집단 지도 상위 피처", ""])
    if top_features:
        lines.append(pd.DataFrame(top_features).to_markdown(index=False))
    else:
        lines.append("상위 피처 없음")
    if status != "READY_FOR_NEW_FUTURE_HOLDOUT":
        lines.extend([
            "",
            "## 판정",
            "",
            "현재 개발 데이터에서 70% Precision을 최소 표본과 시간 전이 Gate까지 만족하지 못했습니다. threshold를 완화하거나 소수 경보로 100% Precision을 만드는 방식은 사용하지 않습니다.",
            "이 상태에서는 신규 미래 holdout을 열지 말고 error-map과 hard-FP pairwise 결과를 다음 신호 개발에 사용해야 합니다.",
        ])
    return "\n".join(lines) + "\n", payload


def write_readme_snapshot(args: argparse.Namespace, selected_features: Mapping[str, Sequence[str]], final_rule: PrecisionRule, status: str) -> None:
    text = f"""# V7 Run Snapshot\n\nstatus: {status}\n\ntarget precision: {args.target_precision:.4f}\n\nfinal rule: `{json.dumps(final_rule.to_dict(), ensure_ascii=False)}`\n\nselection folds: {args.selection_folds}\nconfirmation folds: {args.confirmation_folds}\nrecent folds: {args.recent_folds}\n\nselected error features by fold:\n\n```json\n{json.dumps(selected_features, ensure_ascii=False, indent=2)}\n```\n"""
    (args.output / "RUN_SNAPSHOT_V7.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch Surge V7 hard-FP / missed-positive bottleneck resolver")
    parser.add_argument("--package-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--folds", type=Path)
    parser.add_argument("--profiles", type=Path)
    parser.add_argument("--recipes", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--reuse-matrix-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--base-seeds", default="17,29,41")
    parser.add_argument("--selection-folds", default="0,1,2,3,4")
    parser.add_argument("--confirmation-folds", default="5,6")
    parser.add_argument("--recent-folds", default="7")
    parser.add_argument("--error-top-quantile", type=float, default=0.80)
    parser.add_argument("--error-missed-quantile", type=float, default=0.50)
    parser.add_argument("--minimum-error-group-rows", type=int, default=20)
    parser.add_argument("--error-feature-count", type=int, default=72)
    parser.add_argument("--error-extra-profile-features", type=int, default=20)
    parser.add_argument("--minimum-contrast-score", type=float, default=0.12)
    parser.add_argument("--scope-one-hot-levels", type=int, default=20)
    parser.add_argument("--minimum-verifier-rows", type=int, default=150)
    parser.add_argument("--minimum-pairwise-rows", type=int, default=150)
    parser.add_argument("--verifier-false-positive-cost", type=float, default=2.0)
    parser.add_argument("--verifier-rounds", type=int, default=160)
    parser.add_argument("--pairwise-rounds", type=int, default=180)
    parser.add_argument("--recent-rounds", type=int, default=180)
    parser.add_argument("--event-rounds", type=int, default=140)
    parser.add_argument("--scope-rounds", type=int, default=130)
    parser.add_argument("--minimum-event-positives", type=int, default=40)
    parser.add_argument("--minimum-scope-train-rows", type=int, default=700)
    parser.add_argument("--minimum-scope-positives", type=int, default=40)
    parser.add_argument("--minimum-scope-negatives", type=int, default=100)
    parser.add_argument("--max-scope-models", type=int, default=12)
    parser.add_argument("--minimum-meta-prior-folds", type=int, default=1)
    parser.add_argument("--minimum-meta-rows", type=int, default=500)
    parser.add_argument("--meta-false-positive-cost", type=float, default=2.0)
    parser.add_argument("--meta-rounds", type=int, default=160)
    parser.add_argument("--meta-start-fold", type=int, default=1)
    parser.add_argument("--policy-start-fold", type=int, default=3)
    parser.add_argument("--minimum-policy-rows", type=int, default=600)
    parser.add_argument("--target-precision", type=float, default=0.70)
    parser.add_argument("--selection-precision-buffer", type=float, default=0.03)
    parser.add_argument("--minimum-precision-lcb", type=float, default=0.60)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--minimum-alerts", type=int, default=30)
    parser.add_argument("--minimum-alert-days", type=int, default=10)
    parser.add_argument("--minimum-recall", type=float, default=0.03)
    parser.add_argument("--maximum-threshold-candidates", type=int, default=350)
    parser.add_argument("--cascade-quantiles", default="0.55,0.70,0.80,0.88,0.93")
    parser.add_argument("--model-seed", type=int, default=20260811)
    parser.add_argument("--threads-per-model", type=int, default=6)
    parser.add_argument("--xgboost-threads", type=int, default=6)
    parser.add_argument("--max-tuning-rounds", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--date-column", default=DATE_COLUMN)
    parser.add_argument("--ticker-column", default=TICKER_COLUMN)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--target-valid-column", default=TARGET_VALID_COLUMN)
    parser.add_argument("--return-column", default="t_price_ret_1")
    parser.add_argument("--scope-columns", default="ticker,industry_name,market,bucket")
    parser.add_argument("--minimum-purge-trading-days", type=int, default=3)
    parser.add_argument("--cache-column-block", type=int, default=64)
    args = parser.parse_args()

    resolve_paths(args)
    require_inputs(args)
    args.base_seeds = parse_int_list(args.base_seeds)
    if args.quick:
        args.base_seeds = args.base_seeds[:1]
        args.error_feature_count = min(args.error_feature_count, 24)
        args.minimum_error_group_rows = min(args.minimum_error_group_rows, 8)
        args.minimum_verifier_rows = min(args.minimum_verifier_rows, 50)
        args.minimum_pairwise_rows = min(args.minimum_pairwise_rows, 50)
        args.minimum_meta_rows = min(args.minimum_meta_rows, 120)
        args.minimum_policy_rows = min(args.minimum_policy_rows, 120)
        args.minimum_alerts = min(args.minimum_alerts, 10)
        args.minimum_alert_days = min(args.minimum_alert_days, 4)
        args.verifier_rounds = min(args.verifier_rounds, 60)
        args.pairwise_rounds = min(args.pairwise_rounds, 70)
        args.recent_rounds = min(args.recent_rounds, 70)
        args.event_rounds = min(args.event_rounds, 60)
        args.scope_rounds = min(args.scope_rounds, 60)
        args.meta_rounds = min(args.meta_rounds, 70)
    args.selection_folds = parse_int_list(args.selection_folds)
    args.confirmation_folds = parse_int_list(args.confirmation_folds)
    args.recent_folds = parse_int_list(args.recent_folds)
    args.scope_columns = parse_str_list(args.scope_columns)
    args.cascade_quantiles = [float(v) for v in parse_str_list(args.cascade_quantiles)]

    status_path = args.output / "RUN_STATUS.json"
    started = time.time()
    try:
        profiles = load_profiles_flexible(args.profiles)
        base_recipes = load_base_recipes(args.recipes, args.quick)
        # Missing profile fallback.
        base_recipes = [BaseRecipe(r.name, r.family, r.profile if r.profile in profiles else "P1_SELECTION_TOP", r.train_policy, r.params, r.rounds, r.rolling_days, r.half_life_days) for r in base_recipes]
        folds = load_folds(args.folds)
        roles = set(args.selection_folds + args.confirmation_folds + args.recent_folds)
        missing_folds = sorted(roles - {f.fold_id for f in folds})
        if missing_folds:
            raise ValueError(f"fold 정의 누락: {missing_folds}")
        compatibility = [_compat_recipe(r) for r in base_recipes]
        compatibility.extend([
            Recipe("v7_lgb_aux", "lightgbm", "P1_SELECTION_TOP", "surge_d3", "expanding", "none", {}),
            Recipe("v7_xgb_aux", "xgboost", "P1_SELECTION_TOP", "surge_d3", "expanding", "none", {}),
        ])
        preflight = preflight_families(compatibility, args.device, args.allow_cpu_fallback)
        atomic_write_json(args.output / "V7_BACKEND_PREFLIGHT.json", preflight)
        bundle = prepare_bundle(args, profiles, base_recipes, folds)
        validate_folds(folds, bundle.dates, args.minimum_purge_trading_days)
        target = bundle.targets["surge_d3"].astype(np.uint8)
        events = event_structure(target, bundle.frame[args.ticker_column], bundle.frame[args.date_column])
        event_audit = {
            "rows": len(events),
            "positive_rows": int(target.sum()),
            "events": int(events.loc[events["event_id"] >= 0, "event_id"].nunique()),
            "event_start_rows": int(events["event_start"].sum()),
            "mean_positive_run_weight": float(events.loc[target == 1, "event_weight"].mean()),
        }
        atomic_write_json(args.output / "EVENT_TARGET_AUDIT_V7.json", with_checksum(event_audit))

        base_oof, base_metrics = train_base_oof(args, bundle, folds, profiles, base_recipes, events["event_weight"].to_numpy(), preflight)
        atomic_write_csv(args.output / "base_oof_predictions.csv", base_oof)
        atomic_write_csv(args.output / "base_component_metrics.csv", base_metrics)

        membership, error_by_fold, error_summary = build_error_map(args, bundle, base_oof, folds, profiles)
        atomic_write_csv(args.output / "error_group_membership_oof.csv", membership)
        atomic_write_csv(args.output / "error_contrast_map_by_fold.csv", error_by_fold)
        atomic_write_csv(args.output / "error_contrast_map_summary.csv", error_summary)

        components, component_metrics, selected_manifest = build_component_oof(args, bundle, folds, profiles, base_oof, membership, error_by_fold, events, preflight)
        components, meta_audit = forward_meta_scores(args, components, preflight)
        atomic_write_csv(args.output / "v7_component_oof_predictions.csv", components)
        atomic_write_csv(args.output / "v7_component_metrics.csv", component_metrics)
        atomic_write_json(args.output / "ERROR_FEATURE_MANIFEST_V7.json", with_checksum({"features_by_fold": selected_manifest}))
        atomic_write_csv(args.output / "FORWARD_META_AUDIT_V7.csv", meta_audit)

        methods = candidate_methods(components, args)
        forward_metrics, policy_search = forward_policy_evaluation(args, components, methods)
        method_summary = summarize_methods(forward_metrics)
        atomic_write_csv(args.output / "method_forward_precision_metrics.csv", forward_metrics)
        atomic_write_csv(args.output / "method_forward_precision_summary.csv", method_summary)
        atomic_write_csv(args.output / "forward_policy_search.csv", policy_search)
        chosen_method = choose_method(method_summary)
        log(f"선택 method: {chosen_method}")

        final_rule, fold_metrics, role_metrics, final_search = final_policy_and_holdout(args, components, chosen_method)
        atomic_write_csv(args.output / "final_precision_metrics_by_fold.csv", fold_metrics)
        atomic_write_csv(args.output / "final_precision_metrics_by_role.csv", role_metrics)
        atomic_write_csv(args.output / "final_policy_search.csv", final_search["search"])
        atomic_write_json(args.output / "FINAL_POLICY_V7.json", with_checksum({
            "method": chosen_method,
            "rule": final_rule.to_dict(),
            "diagnostics": final_search["diagnostics"],
        }))

        report_text, report_json = build_bottleneck_report(args, error_summary, method_summary, fold_metrics, final_rule)
        (args.output / "BOTTLENECK_REPORT_V7_KO.md").write_text(report_text, encoding="utf-8")
        atomic_write_json(args.output / "BOTTLENECK_REPORT_V7.json", report_json)
        final_status = str(report_json["status"])
        write_readme_snapshot(args, selected_manifest, final_rule, final_status)
        recommendation = with_checksum({
            "schema": RUN_SCHEMA,
            "status": final_status,
            "chosen_method": chosen_method,
            "final_rule": final_rule.to_dict(),
            "target_precision": args.target_precision,
            "selection_precision_buffer": args.selection_precision_buffer,
            "minimum_precision_lcb": args.minimum_precision_lcb,
            "minimum_alerts": args.minimum_alerts,
            "minimum_alert_days": args.minimum_alert_days,
            "minimum_recall": args.minimum_recall,
            "important_note": "Gate 실패 시 threshold를 완화하지 않고 no-alert로 동결한다. diagnostic practical precision은 별도 결과에만 기록한다.",
        })
        atomic_write_json(args.output / "FINAL_RECOMMENDATION_V7.json", recommendation)
        inventory = build_output_inventory(args.output)
        atomic_write_json(args.output / "OUTPUT_INVENTORY_V7.json", inventory)
        run_status = with_checksum({
            "schema": RUN_SCHEMA,
            "execution_status": "SUCCESS",
            "model_gate_status": final_status,
            "elapsed_seconds": time.time() - started,
            "dataset_sha256": bundle.dataset_sha256,
            "target_sha256": bundle.target_sha256,
            "rows": len(bundle.frame),
            "features": len(bundle.features),
            "chosen_method": chosen_method,
            "final_rule": final_rule.to_dict(),
        })
        atomic_write_json(status_path, run_status)
        print("=" * 72)
        print("CrashWatch Surge Hard-FP V7")
        print("=" * 72)
        print(f"Rows              : {len(bundle.frame):,}")
        print(f"Features          : {len(bundle.features):,}")
        print(f"Chosen method     : {chosen_method}")
        print(f"Policy            : {final_rule.kind}")
        print(f"Gate              : {final_status}")
        print(f"Output            : {args.output}")
        print("=" * 72)
    except Exception as exc:
        error = with_checksum({
            "schema": RUN_SCHEMA,
            "execution_status": "FAILED",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "elapsed_seconds": time.time() - started,
        })
        try:
            atomic_write_json(status_path, error)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
