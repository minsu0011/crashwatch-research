from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..config import ProjectPaths, get_paths, load_baskets
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, feature_hash, finite_float32, normalize_date, normalize_ticker, stable_hash
from .metrics import evaluate_scopes
from .models import fit_model
from .splits import make_walk_forward_folds
from .statistics import paired_delta_table, summarize_deltas

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment: str
    namespace: str
    ablation_mode: str
    target_group: str | None = None
    target_bucket: str | None = None
    target_ticker: str | None = None


def load_catalog(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        raise FileNotFoundError(f"피처 카탈로그가 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_valid_features(df: pd.DataFrame, candidates: Iterable[str], max_missing: float = 0.995, min_non_null: int = 200, min_active_dates: int = 100) -> tuple[list[str], pd.DataFrame]:
    rows = []
    valid = []
    for col in sorted(set(candidates)):
        if col not in df.columns:
            rows.append({"feature": col, "status": "missing_column", "missing_rate": 1.0, "unique_count": 0})
            continue
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing = float(s.isna().mean())
        unique = int(s.nunique(dropna=True))
        non_null = int(s.notna().sum())
        active_dates = int(df.loc[s.notna(), "date"].nunique()) if "date" in df.columns else non_null
        status = "valid" if missing <= max_missing and unique >= 2 and non_null >= min_non_null and active_dates >= min_active_dates else "invalid"
        rows.append({"feature": col, "status": status, "missing_rate": missing, "unique_count": unique, "non_null_count": non_null, "active_dates": active_dates})
        if status == "valid":
            valid.append(col)
    return valid, pd.DataFrame(rows)


def _deterministic_train_sample(block: pd.DataFrame, target: str, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(block) <= max_rows:
        return block
    positives = block.loc[block[target].eq(1)]
    negatives = block.loc[block[target].eq(0)]
    remaining = max(0, max_rows - len(positives))
    if remaining >= len(negatives):
        return block
    key = (
        negatives["ticker"].astype(str) + "|" + negatives["date"].astype(str) + f"|{seed}"
    ).map(lambda x: int(stable_hash(x), 16))
    sampled_neg = negatives.loc[key.nsmallest(remaining).index]
    return pd.concat([positives, sampled_neg], ignore_index=False).sort_index()


def _apply_mask(df: pd.DataFrame, spec: ExperimentSpec, group_features: list[str]) -> pd.DataFrame:
    if spec.ablation_mode not in {"bucket_mask", "ticker_mask"}:
        return df
    out = df.copy()
    if spec.ablation_mode == "bucket_mask":
        mask = out["bucket"].eq(spec.target_bucket)
    else:
        mask = out["ticker"].eq(spec.target_ticker)
    cols = [c for c in group_features if c in out.columns]
    out.loc[mask, cols] = np.nan
    availability = [c for c in cols if c.endswith("_is_available")]
    if availability:
        out.loc[mask, availability] = 0
    return out


def _prediction_path(paths: ProjectPaths, spec: ExperimentSpec, fold: int, seed: int, features: list[str], dataset_signature: str) -> Path:
    payload = {**asdict(spec), "fold": fold, "seed": seed, "feature_hash": feature_hash(features), "dataset_signature": dataset_signature}
    key = stable_hash(payload)
    return paths.cache_dual / spec.namespace / f"{key}.parquet"


def evaluate_spec(
    df: pd.DataFrame,
    target: str,
    base_features: list[str],
    group_features: list[str],
    spec: ExperimentSpec,
    folds: list[dict],
    seeds: list[int],
    baskets: pd.DataFrame,
    paths: ProjectPaths,
    *,
    max_train_rows: int,
    overwrite_cache: bool,
    prefer_gpu: bool,
    dataset_signature: str,
) -> tuple[pd.DataFrame, list[dict]]:
    if spec.ablation_mode in {"global_drop", "ticker_global_drop"}:
        features = [c for c in base_features if c not in set(group_features)]
    else:
        features = list(base_features)
    metric_frames: list[pd.DataFrame] = []
    run_records: list[dict] = []
    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_mask = df["date"].isin(fold["train_dates_index"])
        val_mask = df["date"].isin(fold["validation_dates_index"])
        train_raw = df.loc[train_mask]
        val_raw = df.loc[val_mask]
        for seed in seeds:
            cache = _prediction_path(paths, spec, fold_id, seed, features, dataset_signature)
            if cache.exists() and not overwrite_cache:
                pred = pd.read_parquet(cache)
                backend = "cache"
            else:
                train = _apply_mask(train_raw, spec, group_features)
                val = _apply_mask(val_raw, spec, group_features)
                train = _deterministic_train_sample(train, target, max_train_rows, seed)
                x_train = finite_float32(train, features)
                x_val = finite_float32(val, features)
                y_train = pd.to_numeric(train[target], errors="coerce").astype(int).to_numpy()
                fitted = fit_model(x_train, y_train, seed, prefer_gpu=prefer_gpu)
                probability = fitted.predict_proba(x_val)
                backend = fitted.backend
                pred = val[["date", "ticker", "bucket", target]].copy()
                pred = pred.rename(columns={target: "target"})
                pred["prediction"] = probability
                atomic_parquet(pred, cache)
            metrics = evaluate_scopes(
                pred, baskets, spec.experiment, fold_id, seed,
                target_bucket=spec.target_bucket, target_ticker=spec.target_ticker,
            )
            for key, value in asdict(spec).items():
                metrics[key] = value
            metrics["feature_count"] = len(features)
            metrics["model_backend"] = backend
            metric_frames.append(metrics)
            run_records.append({
                **asdict(spec), "fold": fold_id, "seed": seed, "feature_count": len(features),
                "feature_hash": feature_hash(features), "model_backend": backend, "cache": str(cache),
            })
    return pd.concat(metric_frames, ignore_index=True), run_records


def build_specs(
    universe_catalog: dict[str, list[str]],
    ticker_catalog: dict[str, list[str]],
    baskets: pd.DataFrame,
    modes: set[str],
    groups: set[str] | None = None,
    buckets: set[str] | None = None,
    tickers: set[str] | None = None,
) -> list[ExperimentSpec]:
    specs = [ExperimentSpec("baseline", "baseline", "none")]
    if "universe" in modes:
        for group in universe_catalog:
            if groups and group not in groups:
                continue
            specs.append(ExperimentSpec(f"universe__{group}", "universe", "global_drop", group))
    if "ticker_global" in modes:
        for group in ticker_catalog:
            if groups and group not in groups:
                continue
            specs.append(ExperimentSpec(f"ticker_global__{group}", "ticker", "ticker_global_drop", group))
    if "bucket" in modes:
        for bucket in sorted(baskets["bucket"].unique()):
            if buckets and bucket not in buckets:
                continue
            for group in ticker_catalog:
                if groups and group not in groups:
                    continue
                specs.append(ExperimentSpec(f"bucket__{bucket}__{group}", "ticker", "bucket_mask", group, target_bucket=bucket))
    if "ticker" in modes:
        for ticker in sorted(baskets["ticker"].unique()):
            if tickers and ticker not in tickers:
                continue
            bucket = baskets.loc[baskets["ticker"].eq(ticker), "bucket"].iloc[0]
            for group in ticker_catalog:
                if groups and group not in groups:
                    continue
                specs.append(ExperimentSpec(f"ticker__{ticker}__{group}", "ticker", "ticker_mask", group, target_bucket=bucket, target_ticker=ticker))
    return specs


def run_ablation(
    project: Path | None = None,
    dataset_path: Path | None = None,
    *,
    target: str = "label_abs_crash_20",
    modes: set[str] | None = None,
    groups: set[str] | None = None,
    buckets: set[str] | None = None,
    tickers: set[str] | None = None,
    seeds: list[int] | None = None,
    n_folds: int = 8,
    validation_days: int = 60,
    purge_days: int = 20,
    min_train_days: int = 500,
    max_train_rows: int = 300000,
    overwrite_cache: bool = False,
    prefer_gpu: bool = True,
) -> dict:
    paths = get_paths(project)
    paths.result_dual.mkdir(parents=True, exist_ok=True)
    paths.cache_dual.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_path or paths.data_root / "development" / "training_dataset_dual.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"이원화 학습 데이터가 없습니다: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    df = normalize_date(df)
    df["ticker"] = normalize_ticker(df["ticker"])
    if target not in df.columns:
        raise KeyError(f"target 열이 없습니다: {target}")
    if "sealed_do_not_train_or_tune" in df.columns and pd.to_numeric(df["sealed_do_not_train_or_tune"], errors="coerce").fillna(0).ne(0).any():
        raise ValueError("development 데이터에 sealed 행이 포함되어 있습니다.")
    baskets = load_baskets(paths)
    if "bucket" not in df.columns:
        df = df.merge(baskets[["ticker", "bucket"]], on="ticker", how="left")
    else:
        mapping = baskets.set_index("ticker")["bucket"]
        df["bucket"] = df["bucket"].where(df["bucket"].notna(), df["ticker"].map(mapping))
    df["bucket"] = df["bucket"].fillna("other")
    baskets = baskets.loc[baskets["ticker"].isin(set(df["ticker"]))].reset_index(drop=True)
    dataset_signature = stable_hash({
        "path": str(dataset_path.resolve()), "size": dataset_path.stat().st_size,
        "mtime_ns": dataset_path.stat().st_mtime_ns, "rows": len(df),
        "date_min": str(df["date"].min()), "date_max": str(df["date"].max()),
        "tickers": sorted(df["ticker"].unique().tolist()),
    })

    u_catalog = load_catalog(paths.feature_dual / "feature_catalog_universe.json")
    t_catalog = load_catalog(paths.feature_dual / "feature_catalog_ticker.json")
    all_candidates = sorted(set(sum(u_catalog.values(), []) + sum(t_catalog.values(), [])))
    valid_features, quality = select_valid_features(df, all_candidates)
    quality["namespace"] = quality["feature"].map(lambda c: "universe" if c.startswith("u_") else "ticker")
    atomic_csv(quality, paths.result_dual / "valid_feature_audit.csv")
    valid_set = set(valid_features)
    u_catalog = {g: [c for c in cols if c in valid_set] for g, cols in u_catalog.items()}
    t_catalog = {g: [c for c in cols if c in valid_set] for g, cols in t_catalog.items()}
    empty_groups = [
        {"namespace": namespace, "group": group, "reason": "no_valid_features"}
        for namespace, catalog in [("universe", u_catalog), ("ticker", t_catalog)]
        for group, cols in catalog.items() if not cols
    ]
    atomic_csv(pd.DataFrame(empty_groups), paths.result_dual / "skipped_groups.csv")

    folds = make_walk_forward_folds(df["date"], n_folds, validation_days, purge_days, min_train_days)
    atomic_json([f["metadata"] for f in folds], paths.result_dual / "walk_forward_folds.json")
    modes = modes or {"universe", "ticker_global", "bucket"}
    seeds = seeds or [17, 43, 101]
    specs = build_specs(u_catalog, t_catalog, baskets, modes, groups, buckets, tickers)

    all_metrics: list[pd.DataFrame] = []
    run_records: list[dict] = []
    skipped_records: list[dict] = []
    for spec in specs:
        if spec.experiment == "baseline":
            group_features: list[str] = []
        elif spec.namespace == "universe":
            group_features = u_catalog.get(spec.target_group or "", [])
        else:
            group_features = t_catalog.get(spec.target_group or "", [])
        if spec.experiment != "baseline" and not group_features:
            skipped_records.append({**asdict(spec), "reason": "skipped_no_valid_features"})
            continue
        if spec.ablation_mode == "bucket_mask":
            block = df.loc[df["bucket"].eq(spec.target_bucket)]
            if block["ticker"].nunique() < 5:
                skipped_records.append({**asdict(spec), "reason": "skipped_insufficient_tickers", "rows": len(block), "ticker_count": int(block["ticker"].nunique())})
                continue
            if pd.to_numeric(block[target], errors="coerce").nunique(dropna=True) < 2:
                skipped_records.append({**asdict(spec), "reason": "skipped_single_class", "rows": len(block)})
                continue
        if spec.ablation_mode == "ticker_mask":
            block = df.loc[df["ticker"].eq(spec.target_ticker)]
            if block.empty:
                skipped_records.append({**asdict(spec), "reason": "skipped_no_rows"})
                continue
            if pd.to_numeric(block[target], errors="coerce").nunique(dropna=True) < 2:
                skipped_records.append({**asdict(spec), "reason": "skipped_single_class", "rows": len(block)})
                continue
        LOGGER.info("실험 시작: %s", spec.experiment)
        metrics, records = evaluate_spec(
            df, target, valid_features, group_features, spec, folds, seeds, baskets, paths,
            max_train_rows=max_train_rows, overwrite_cache=overwrite_cache, prefer_gpu=prefer_gpu,
            dataset_signature=dataset_signature,
        )
        all_metrics.append(metrics)
        run_records.extend(records)
        atomic_csv(pd.concat(all_metrics, ignore_index=True), paths.result_dual / "all_metrics_by_fold_seed_scope.csv")
        atomic_json(run_records, paths.result_dual / "run_manifest.json")

    atomic_csv(pd.DataFrame(skipped_records), paths.result_dual / "skipped_experiments.csv")
    if not all_metrics:
        raise RuntimeError("실행 가능한 실험이 없습니다.")
    metrics = pd.concat(all_metrics, ignore_index=True)
    deltas = paired_delta_table(metrics)
    summary = summarize_deltas(deltas)
    atomic_csv(metrics, paths.result_dual / "all_metrics_by_fold_seed_scope.csv")
    atomic_csv(deltas, paths.result_dual / "paired_ablation_deltas.csv")
    atomic_csv(summary, paths.result_dual / "ablation_statistical_summary.csv")

    bucket_matrix_source = summary.loc[(summary["scope_type"].isin(["target_bucket", "bucket"])) & summary["target_group"].notna()]
    if not bucket_matrix_source.empty:
        matrix = bucket_matrix_source.pivot_table(index="scope_value", columns="target_group", values="pr_auc_loss_when_removed_mean", aggfunc="mean")
        matrix.reset_index().to_csv(paths.result_dual / "bucket_group_sensitivity_matrix.csv", index=False, encoding="utf-8-sig")
    ticker_matrix_source = summary.loc[(summary["scope_type"].isin(["target_ticker", "ticker"])) & summary["target_group"].notna()]
    if not ticker_matrix_source.empty:
        matrix = ticker_matrix_source.pivot_table(index="scope_value", columns="target_group", values="pr_auc_loss_when_removed_mean", aggfunc="mean")
        matrix.reset_index().to_csv(paths.result_dual / "ticker_group_sensitivity_matrix.csv", index=False, encoding="utf-8-sig")

    result = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "dataset": str(dataset_path), "rows": len(df), "tickers": int(df["ticker"].nunique()),
        "valid_features": len(valid_features), "universe_groups": len(u_catalog), "ticker_groups": len(t_catalog),
        "planned_experiments": len(specs), "executed_experiments": int(metrics["experiment"].nunique()),
        "skipped_experiments": len(skipped_records), "failed_experiments": 0,
        "folds": len(folds), "seeds": seeds, "modes": sorted(modes),
        "dataset_signature": dataset_signature,
        "target": target, "result_dir": str(paths.result_dual),
    }
    atomic_json(result, paths.result_dual / "run_summary.json")
    return result
