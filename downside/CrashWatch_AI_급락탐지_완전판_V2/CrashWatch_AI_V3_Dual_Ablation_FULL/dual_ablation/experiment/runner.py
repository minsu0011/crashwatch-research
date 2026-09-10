from __future__ import annotations

import gc
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from ..audit import MIN_TICKERS_PER_BUCKET, file_sha256, preflight_validate
from ..config import ProjectPaths, get_paths, load_baskets
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, feature_hash, finite_float32, normalize_date, normalize_ticker, stable_hash
from .metrics import evaluate_scopes
from .models import fit_model
from .splits import make_walk_forward_folds
from .statistics import paired_delta_table, summarize_deltas

LOGGER = logging.getLogger(__name__)
MODEL_PARAMETERS = {
    "n_estimators": 450, "max_depth": 6, "learning_rate": 0.04,
    "subsample": 0.85, "colsample_bytree": 0.72, "min_child_weight": 5,
    "reg_alpha": 0.15, "reg_lambda": 1.2,
}


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
        raise FileNotFoundError(f"feature catalog not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def select_valid_features(
    df: pd.DataFrame, candidates: Iterable[str], max_missing: float = 0.995,
    min_non_null: int = 200, min_active_days: int = 100,
) -> tuple[list[str], pd.DataFrame]:
    rows, valid = [], []
    for col in sorted(set(candidates)):
        if col not in df.columns:
            rows.append({"feature": col, "status": "missing_column", "missing_rate": 1.0,
                         "unique_count": 0, "non_null_count": 0, "active_days": 0})
            continue
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        missing, unique, non_null = float(s.isna().mean()), int(s.nunique(dropna=True)), int(s.notna().sum())
        active_days = int(df.loc[s.notna(), "date"].nunique())
        ok = missing <= max_missing and unique >= 2 and non_null >= min_non_null and active_days >= min_active_days
        status = "valid" if ok else "invalid"
        rows.append({"feature": col, "status": status, "missing_rate": missing,
                     "unique_count": unique, "non_null_count": non_null, "active_days": active_days})
        if ok:
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
    keys = (negatives["ticker"].astype(str) + "|" + negatives["date"].astype(str) + f"|{seed}").map(stable_hash)
    sampled = negatives.assign(_sample_key=keys).sort_values("_sample_key").head(remaining).drop(columns="_sample_key")
    return pd.concat([positives, sampled], ignore_index=False).sort_index()


def _apply_mask(df: pd.DataFrame, spec: ExperimentSpec, group_features: list[str]) -> pd.DataFrame:
    if spec.ablation_mode not in {"bucket_mask", "ticker_mask"}:
        return df
    out = df.copy()
    mask = out["bucket"].eq(spec.target_bucket) if spec.ablation_mode == "bucket_mask" else out["ticker"].eq(spec.target_ticker)
    out.loc[mask, [c for c in group_features if c in out]] = np.nan
    return out


def _fit_calibrator(method: str, probabilities: np.ndarray, target: np.ndarray):
    if method == "none" or len(np.unique(target)) < 2:
        return None
    if method == "sigmoid":
        model = LogisticRegression(random_state=0)
        model.fit(probabilities.reshape(-1, 1), target)
        return model
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(probabilities, target)
        return model
    raise ValueError(f"unknown calibration method: {method}")


def _apply_calibrator(calibrator, probabilities: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return probabilities
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
    return np.asarray(calibrator.predict(probabilities), dtype=float)


def _fit_predict(
    train: pd.DataFrame, validation: pd.DataFrame, features: list[str], target: str,
    seed: int, prefer_gpu: bool, calibration: str,
) -> tuple[np.ndarray, np.ndarray, str]:
    train_dates = pd.DatetimeIndex(train["date"].unique()).sort_values()
    if calibration == "none" or len(train_dates) < 180:
        fit_rows, calibration_rows = train, train.iloc[0:0]
    else:
        calibration_dates = train_dates[-60:]
        fit_rows = train.loc[~train["date"].isin(calibration_dates)]
        calibration_rows = train.loc[train["date"].isin(calibration_dates)]
    fitted = fit_model(finite_float32(fit_rows, features), fit_rows[target].astype(int).to_numpy(), seed, prefer_gpu=prefer_gpu)
    raw_validation = fitted.predict_proba(finite_float32(validation, features))
    calibrator = None
    if not calibration_rows.empty:
        raw_calibration = fitted.predict_proba(finite_float32(calibration_rows, features))
        calibrator = _fit_calibrator(calibration, raw_calibration, calibration_rows[target].astype(int).to_numpy())
    calibrated = np.clip(_apply_calibrator(calibrator, raw_validation), 1e-7, 1 - 1e-7)
    backend = fitted.backend
    del fitted, calibrator
    gc.collect()
    return raw_validation, calibrated, backend


def _prediction_path(
    paths: ProjectPaths, spec: ExperimentSpec, fold: dict, seed: int, features: list[str],
    *, dataset_hash: str, catalog_hash: str, calibration: str,
) -> Path:
    payload = {
        **asdict(spec), "fold": fold["metadata"], "seed": seed,
        "dataset_hash": dataset_hash, "feature_list_hash": feature_hash(features),
        "catalog_hash": catalog_hash, "model_parameters": MODEL_PARAMETERS,
        "calibration": calibration,
    }
    return paths.cache_dual / spec.namespace / f"{stable_hash(payload)}.parquet"


def evaluate_spec(
    df: pd.DataFrame, target: str, base_features: list[str], group_features: list[str],
    spec: ExperimentSpec, folds: list[dict], seeds: list[int], baskets: pd.DataFrame,
    paths: ProjectPaths, *, max_train_rows: int, overwrite_cache: bool, prefer_gpu: bool,
    dataset_hash: str, catalog_hash: str, calibration: str,
) -> tuple[pd.DataFrame, list[dict]]:
    features = [c for c in base_features if c not in set(group_features)] if spec.ablation_mode in {"global_drop", "ticker_global_drop"} else list(base_features)
    if not features:
        raise ValueError(f"experiment has no model features: {spec.experiment}")
    metric_frames, records = [], []
    total_runs = len(folds) * len(seeds)
    completed_runs = 0
    for fold in folds:
        fold_id = int(fold["fold_id"])
        train_raw = df.loc[df["date"].isin(fold["train_dates_index"])]
        val_raw = df.loc[df["date"].isin(fold["validation_dates_index"])]
        if val_raw.empty or val_raw[target].nunique() < 2:
            raise ValueError(f"validation fold is empty or single class: {spec.experiment} fold={fold_id}")
        for seed in seeds:
            started = time.perf_counter()
            cache = _prediction_path(paths, spec, fold, seed, features, dataset_hash=dataset_hash,
                                     catalog_hash=catalog_hash, calibration=calibration)
            if cache.exists() and not overwrite_cache:
                pred, backend = pd.read_parquet(cache), "cache"
            else:
                train = _deterministic_train_sample(_apply_mask(train_raw, spec, group_features), target, max_train_rows, seed)
                validation = _apply_mask(val_raw, spec, group_features)
                raw_probability, probability, backend = _fit_predict(
                    train, validation, features, target, seed, prefer_gpu, calibration,
                )
                pred = validation[["date", "ticker", "bucket", target]].rename(columns={target: "target"}).copy()
                pred["row_id"] = pred["date"].dt.strftime("%Y-%m-%d") + "|" + pred["ticker"]
                pred["prediction_uncalibrated"] = raw_probability
                pred["prediction"] = probability
                atomic_parquet(pred, cache)
            completed_runs += 1
            LOGGER.info(
                "experiment=%s fold=%s seed=%s progress=%s/%s backend=%s elapsed=%.1fs",
                spec.experiment, fold_id, seed, completed_runs, total_runs, backend,
                time.perf_counter() - started,
            )
            metrics = evaluate_scopes(pred, baskets, spec.experiment, fold_id, seed,
                                      target_bucket=spec.target_bucket, target_ticker=spec.target_ticker)
            for key, value in asdict(spec).items():
                metrics[key] = value
            metrics["feature_count"], metrics["model_backend"], metrics["calibration"] = len(features), backend, calibration
            metric_frames.append(metrics)
            records.append({**asdict(spec), "fold": fold_id, "seed": seed, "feature_count": len(features),
                            "feature_hash": feature_hash(features), "model_backend": backend,
                            "calibration": calibration, "cache": str(cache)})
    return pd.concat(metric_frames, ignore_index=True), records


def _load_v2_catalog(paths: ProjectPaths) -> list[str]:
    path = paths.project.parent / "crashwatch_ai_data" / "meta" / "feature_catalog.json"
    if not path.exists():
        return []
    catalog = json.loads(path.read_text(encoding="utf-8"))
    return sorted(set(sum((list(v) for v in catalog.values()), [])))


def _family_comparison(
    df: pd.DataFrame, target: str, families: dict[str, list[str]], folds: list[dict], seeds: list[int],
    baskets: pd.DataFrame, paths: ProjectPaths, *, max_train_rows: int, overwrite_cache: bool,
    prefer_gpu: bool, dataset_hash: str, catalog_hash: str, calibration: str,
) -> tuple[str, str, pd.DataFrame, pd.DataFrame]:
    frames = []
    for family, features in families.items():
        LOGGER.info("model-family comparison started: %s (%s features)", family, len(features))
        spec = ExperimentSpec(f"model_family__{family}", "model_family", "none")
        metrics, _ = evaluate_spec(
            df, target, features, [], spec, folds, seeds, baskets, paths,
            max_train_rows=max_train_rows, overwrite_cache=overwrite_cache, prefer_gpu=prefer_gpu,
            dataset_hash=dataset_hash, catalog_hash=catalog_hash, calibration=calibration,
        )
        metrics["model_family"] = family
        frames.append(metrics)
    all_metrics = pd.concat(frames, ignore_index=True)
    by_fold = all_metrics.loc[all_metrics["scope_type"].eq("all")].copy()
    summary = by_fold.groupby("model_family", as_index=False).agg(
        pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"), brier=("brier", "mean"),
        logloss=("logloss", "mean"), alert_precision=("alert_precision", "mean"),
        alert_recall=("alert_recall", "mean"), top_1_percent_precision=("top_1_percent_precision", "mean"),
        top_3_percent_precision=("top_3_percent_precision", "mean"),
        top_5_percent_precision=("top_5_percent_precision", "mean"),
        pr_auc_fold_std=("pr_auc", "std"), feature_count=("feature_count", "max"),
    )
    ranking = summary.sort_values(
        ["pr_auc", "brier", "pr_auc_fold_std", "feature_count"],
        ascending=[False, True, True, True],
    )
    selected = str(ranking.iloc[0]["model_family"])
    # Model-family comparison and feature-utility ablation answer different
    # questions.  Preserve the overall winner, but guarantee that the
    # ablation reference actually contains V3 features.
    eligible = ranking.loc[ranking["model_family"].isin(["v3_only", "v2_plus_v3"])]
    if eligible.empty:
        raise ValueError("no V3-capable model family is available for feature-utility ablation")
    ablation_reference = str(eligible.iloc[0]["model_family"])
    return selected, ablation_reference, by_fold, summary


def _base_specs(
    universe_catalog: dict[str, list[str]], ticker_catalog: dict[str, list[str]],
    ready_buckets: set[str], modes: set[str], groups: set[str] | None,
) -> tuple[list[ExperimentSpec], list[dict]]:
    specs, skipped = [ExperimentSpec("baseline", "baseline", "none")], []
    if "universe" in modes:
        for group, features in universe_catalog.items():
            if groups and group not in groups:
                continue
            if not features:
                skipped.append({"experiment": f"universe__{group}", "status": "skipped_no_valid_features", "reason": group})
            else:
                specs.append(ExperimentSpec(f"universe__{group}", "universe", "global_drop", group))
    if "ticker_global" in modes:
        for group, features in ticker_catalog.items():
            if groups and group not in groups:
                continue
            if not features:
                skipped.append({"experiment": f"ticker_global__{group}", "status": "skipped_no_valid_features", "reason": group})
            else:
                specs.append(ExperimentSpec(f"ticker_global__{group}", "ticker", "ticker_global_drop", group))
    if "bucket" in modes:
        for bucket in sorted(ready_buckets):
            for group, features in ticker_catalog.items():
                if groups and group not in groups:
                    continue
                if not features:
                    skipped.append({"experiment": f"bucket__{bucket}__{group}", "status": "skipped_no_valid_features", "reason": group})
                else:
                    specs.append(ExperimentSpec(f"bucket__{bucket}__{group}", "ticker", "bucket_mask", group, target_bucket=bucket))
    return specs, skipped


def _target_skip_reason(
    df: pd.DataFrame, target: str, folds: list[dict], spec: ExperimentSpec,
) -> tuple[str, str] | None:
    """Return the required sparse-scope skip status before fitting a mask model."""
    if spec.ablation_mode not in {"bucket_mask", "ticker_mask"}:
        return None
    validation_dates = pd.DatetimeIndex(
        np.concatenate([fold["validation_dates_index"].to_numpy() for fold in folds])
    ).unique()
    block = df.loc[df["date"].isin(validation_dates)]
    if spec.ablation_mode == "bucket_mask":
        block = block.loc[block["bucket"].eq(spec.target_bucket)]
    else:
        block = block.loc[block["ticker"].eq(spec.target_ticker)]
    if block.empty:
        return "skipped_no_rows", "target validation scope has no rows"
    labels = pd.to_numeric(block[target], errors="coerce").dropna()
    if labels.nunique() < 2:
        return "skipped_single_class", "target validation scope contains one class"
    return None


def _write_mode_outputs(metrics: pd.DataFrame, deltas: pd.DataFrame, summary: pd.DataFrame, paths: ProjectPaths) -> None:
    atomic_csv(metrics.loc[metrics["experiment"].eq("baseline")], paths.result_dual / "baseline_metrics_by_fold_scope.csv")
    mappings = {
        "universe": metrics["namespace"].eq("universe"),
        "ticker_global": metrics["ablation_mode"].eq("ticker_global_drop"),
        "bucket": metrics["ablation_mode"].eq("bucket_mask"),
        "ticker": metrics["ablation_mode"].eq("ticker_mask"),
    }
    for name, mask in mappings.items():
        experiments = set(metrics.loc[mask, "experiment"])
        atomic_csv(deltas.loc[deltas["experiment"].isin(experiments)], paths.result_dual / f"{name}_ablation_by_fold.csv")
        atomic_csv(summary.loc[summary["experiment"].isin(experiments)], paths.result_dual / f"{name}_ablation_summary.csv")


def run_ablation(
    project: Path | None = None, dataset_path: Path | None = None, *, target: str = "label_abs_crash_20",
    modes: set[str] | None = None, groups: set[str] | None = None, buckets: set[str] | None = None,
    tickers: set[str] | None = None, seeds: list[int] | None = None, n_folds: int = 8,
    validation_days: int = 60, purge_days: int = 20, min_train_days: int = 500,
    max_train_rows: int = 300000, overwrite_cache: bool = False, prefer_gpu: bool = True,
    calibration: str = "sigmoid", allow_partial_data: bool = False,
) -> dict:
    if purge_days < 20:
        raise ValueError("purge_days must be at least 20 for label_abs_crash_20")
    if n_folds < 4:
        raise ValueError("at least four walk-forward folds are required")
    if calibration not in {"none", "sigmoid", "isotonic"}:
        raise ValueError(f"unknown calibration method: {calibration}")
    paths = get_paths(project)
    paths.result_dual.mkdir(parents=True, exist_ok=True)
    paths.cache_dual.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_path or paths.data_root / "development" / "training_dataset_dual.parquet"
    if not dataset_path.exists():
        raise FileNotFoundError(f"dual training dataset not found: {dataset_path}")
    df = normalize_date(pd.read_parquet(dataset_path))
    df["ticker"] = normalize_ticker(df["ticker"])
    baskets_df = load_baskets(paths)
    if "bucket" not in df:
        df = df.merge(baskets_df[["ticker", "bucket"]], on="ticker", how="left")
    if target not in df:
        raise KeyError(target)

    quality_path = paths.meta_dual / "feature_quality_audit.csv"
    coverage_path = paths.meta_dual / "basket_coverage_report.csv"
    if not quality_path.exists() or not coverage_path.exists():
        raise FileNotFoundError("feature quality and basket coverage audits must be generated first")
    quality, coverage = pd.read_csv(quality_path), pd.read_csv(coverage_path)
    preflight_errors = preflight_validate(df, quality, coverage, allow_partial_data=allow_partial_data, target=target)
    ready_buckets = set(coverage.loc[coverage["status"].eq("ready"), "bucket"])
    if buckets:
        ready_buckets &= buckets
    if len(ready_buckets) == 0 and (modes is None or "bucket" in modes):
        raise ValueError("no ready buckets")

    universe_catalog = load_catalog(paths.feature_dual / "feature_catalog_universe.json")
    ticker_catalog = load_catalog(paths.feature_dual / "feature_catalog_ticker.json")
    valid_audit = set(quality.loc[quality["status"].eq("valid"), "feature"])
    universe_catalog = {group: [c for c in features if c in valid_audit] for group, features in universe_catalog.items()}
    ticker_catalog = {group: [c for c in features if c in valid_audit] for group, features in ticker_catalog.items()}
    v3_candidates = sorted(set(sum(universe_catalog.values(), []) + sum(ticker_catalog.values(), [])))
    v3_features, v3_quality = select_valid_features(df, v3_candidates)
    v2_features, v2_quality = select_valid_features(df, _load_v2_catalog(paths))
    audit = pd.concat([v2_quality.assign(domain="v2"), v3_quality.assign(domain="v3")], ignore_index=True)
    atomic_csv(audit, paths.result_dual / "valid_feature_audit.csv")
    if not v3_features:
        raise ValueError("no valid V3 features")
    families = {"v2_only": v2_features, "v3_only": v3_features,
                "v2_plus_v3": sorted(set(v2_features + v3_features))}
    families = {name: features for name, features in families.items() if features}

    folds = make_walk_forward_folds(df["date"], n_folds, validation_days, purge_days, min_train_days)
    if len(folds) < 4:
        raise ValueError(f"fewer than four valid folds: {len(folds)}")
    atomic_json([fold["metadata"] for fold in folds], paths.result_dual / "walk_forward_folds.json")
    seeds = seeds or [17, 43, 101]
    modes = modes if modes is not None else {"universe", "ticker_global", "bucket", "ticker"}
    dataset_hash = file_sha256(dataset_path)
    catalog_hash = stable_hash({"universe": universe_catalog, "ticker": ticker_catalog})

    selected_family, ablation_family, family_by_fold, family_summary = _family_comparison(
        df, target, families, folds, seeds, baskets_df, paths, max_train_rows=max_train_rows,
        overwrite_cache=overwrite_cache, prefer_gpu=prefer_gpu, dataset_hash=dataset_hash,
        catalog_hash=catalog_hash, calibration=calibration,
    )
    atomic_csv(family_by_fold, paths.result_dual / "model_family_comparison_by_fold.csv")
    atomic_csv(family_summary, paths.result_dual / "model_family_comparison_summary.csv")
    selection = {
        "selected_family": selected_family,
        "ablation_reference_family": ablation_family,
        "ablation_reference_reason": "best-ranked V3-capable family used to measure V3 feature utility",
        "priority": ["PR-AUC descending", "Brier ascending", "fold stability", "feature count"],
        "selected_feature_count": len(families[selected_family]),
        "ablation_reference_feature_count": len(families[ablation_family]),
        "comparison": family_summary.to_dict("records"),
    }
    atomic_json(selection, paths.result_dual / "baseline_selection.json")
    base_features = families[ablation_family]
    base_feature_set = set(base_features)

    specs, skipped = _base_specs(universe_catalog, ticker_catalog, ready_buckets, modes, groups)
    all_metrics, run_records, failed = [], [], []
    for spec in specs:
        LOGGER.info("ablation experiment started: %s", spec.experiment)
        group_features = [] if spec.experiment == "baseline" else (
            universe_catalog.get(spec.target_group or "", []) if spec.namespace == "universe"
            else ticker_catalog.get(spec.target_group or "", [])
        )
        if spec.experiment != "baseline" and not (base_feature_set & set(group_features)):
            skipped.append({
                "experiment": spec.experiment,
                "status": "skipped_no_valid_features",
                "reason": f"group has no feature in ablation baseline family {ablation_family}",
            })
            continue
        sparse_reason = _target_skip_reason(df, target, folds, spec)
        if sparse_reason is not None:
            status, reason = sparse_reason
            skipped.append({"experiment": spec.experiment, "status": status, "reason": reason})
            continue
        try:
            metrics, records = evaluate_spec(
                df, target, base_features, group_features, spec, folds, seeds, baskets_df, paths,
                max_train_rows=max_train_rows, overwrite_cache=overwrite_cache, prefer_gpu=prefer_gpu,
                dataset_hash=dataset_hash, catalog_hash=catalog_hash, calibration=calibration,
            )
            all_metrics.append(metrics)
            run_records.extend(records)
            atomic_csv(pd.concat(all_metrics, ignore_index=True), paths.result_dual / "all_metrics_by_fold_seed_scope.csv")
            atomic_json(run_records, paths.result_dual / "run_manifest.json")
        except (ValueError, AssertionError) as exc:
            failed.append({"experiment": spec.experiment, "status": "failed", "reason": str(exc)})
            if not allow_partial_data:
                raise

    metrics = pd.concat(all_metrics, ignore_index=True)
    deltas = paired_delta_table(metrics)
    summary = summarize_deltas(deltas)

    # Run individual ticker masks only for sensitive bucket/group candidates.
    ticker_specs: list[ExperimentSpec] = []
    if "ticker" in modes:
        candidate_rows = summary.loc[
            summary["experiment"].str.startswith("bucket__")
            & summary["pair_scope_type"].eq("bucket")
            & (summary["scope_value"].astype(str) == summary["target_bucket"].astype(str))
            & ((summary["mean_delta"].abs() >= 0.01) | (summary["fdr_q_value"] <= 0.1))
        ]
        for row in candidate_rows[["target_bucket", "target_group"]].drop_duplicates().itertuples(index=False):
            candidates = baskets_df.loc[baskets_df["bucket"].eq(row.target_bucket), "ticker"]
            for ticker in candidates:
                if tickers and ticker not in tickers:
                    continue
                block = df.loc[df["ticker"].eq(ticker), target]
                if block.empty:
                    skipped.append({"experiment": f"ticker__{ticker}__{row.target_group}", "status": "skipped_no_rows", "reason": ticker})
                elif block.nunique() < 2:
                    skipped.append({"experiment": f"ticker__{ticker}__{row.target_group}", "status": "skipped_single_class", "reason": ticker})
                else:
                    spec = ExperimentSpec(f"ticker__{ticker}__{row.target_group}", "ticker", "ticker_mask", row.target_group, row.target_bucket, ticker)
                    sparse_reason = _target_skip_reason(df, target, folds, spec)
                    if sparse_reason is not None:
                        status, reason = sparse_reason
                        skipped.append({"experiment": spec.experiment, "status": status, "reason": reason})
                    elif not (base_feature_set & set(ticker_catalog.get(row.target_group, []))):
                        skipped.append({
                            "experiment": spec.experiment,
                            "status": "skipped_no_valid_features",
                            "reason": f"group has no feature in ablation baseline family {ablation_family}",
                        })
                    else:
                        ticker_specs.append(spec)
        for spec in ticker_specs:
            try:
                extra, records = evaluate_spec(
                    df, target, base_features, ticker_catalog[spec.target_group or ""], spec, folds, seeds,
                    baskets_df, paths, max_train_rows=max_train_rows, overwrite_cache=overwrite_cache,
                    prefer_gpu=prefer_gpu, dataset_hash=dataset_hash, catalog_hash=catalog_hash,
                    calibration=calibration,
                )
                metrics = pd.concat([metrics, extra], ignore_index=True)
                run_records.extend(records)
                atomic_csv(metrics, paths.result_dual / "all_metrics_by_fold_seed_scope.csv")
                atomic_json(run_records, paths.result_dual / "run_manifest.json")
            except (ValueError, AssertionError) as exc:
                failed.append({"experiment": spec.experiment, "status": "failed", "reason": str(exc)})
                if not allow_partial_data:
                    raise
        deltas = paired_delta_table(metrics)
        summary = summarize_deltas(deltas)

    atomic_csv(metrics, paths.result_dual / "all_metrics_by_fold_seed_scope.csv")
    atomic_csv(deltas, paths.result_dual / "paired_ablation_deltas.csv")
    atomic_csv(summary, paths.result_dual / "ablation_statistical_summary.csv")
    _write_mode_outputs(metrics, deltas, summary, paths)
    skipped_df = pd.DataFrame(skipped + failed, columns=["experiment", "status", "reason"])
    atomic_csv(skipped_df, paths.result_dual / "skipped_experiments.csv")
    atomic_csv(quality, paths.result_dual / "feature_quality_audit.csv")
    atomic_csv(pd.read_csv(paths.meta_dual / "crawl_manifest.csv"), paths.result_dual / "crawl_manifest.csv")
    atomic_csv(coverage, paths.result_dual / "basket_coverage_report.csv")

    bucket_source = summary.loc[
        summary["ablation_mode"].eq("bucket_mask")
        & summary["pair_scope_type"].eq("bucket")
        & summary["target_group"].notna()
        & summary["scope_value"].astype(str).eq(summary["target_bucket"].astype(str))
    ]
    if not bucket_source.empty:
        matrix = bucket_source.pivot_table(index="scope_value", columns="target_group", values="mean_delta", aggfunc="mean")
        bucket_names = baskets_df[["bucket", "bucket_name"]].drop_duplicates("bucket") if "bucket_name" in baskets_df else pd.DataFrame()
        matrix = matrix.reset_index().rename(columns={"scope_value": "bucket"})
        if not bucket_names.empty:
            matrix = matrix.merge(bucket_names, on="bucket", how="left")
        atomic_csv(matrix, paths.result_dual / "bucket_group_sensitivity_matrix.csv")
    ticker_source = summary.loc[
        summary["ablation_mode"].eq("ticker_mask")
        & summary["pair_scope_type"].eq("ticker")
        & summary["target_group"].notna()
        & summary["scope_value"].astype(str).eq(summary["target_ticker"].astype(str))
    ]
    if not ticker_source.empty:
        matrix = ticker_source.pivot_table(index="scope_value", columns="target_group", values="mean_delta", aggfunc="mean")
        ticker_meta = baskets_df[["ticker", "name", "bucket"]].drop_duplicates("ticker")
        matrix = matrix.reset_index().rename(columns={"scope_value": "ticker"}).merge(ticker_meta, on="ticker", how="left")
        atomic_csv(matrix, paths.result_dual / "ticker_group_sensitivity_matrix.csv")

    executed = int(metrics["experiment"].nunique())
    result = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(), "dataset": str(dataset_path),
        "dataset_hash": dataset_hash, "rows": len(df), "tickers": int(df["ticker"].nunique()),
        "target": target, "folds": len(folds), "seeds": seeds, "calibration": calibration,
        "bootstrap_samples": 4000,
        "bootstrap_pvalue_correction": "add_one_monte_carlo",
        "selected_baseline_family": selected_family,
        "ablation_baseline_family": ablation_family,
        "selected_feature_count": len(families[selected_family]),
        "ablation_feature_count": len(base_features),
        "planned_experiments": executed + len(skipped) + len(failed), "executed_experiments": executed,
        "skipped_experiments": len(skipped), "failed_experiments": len(failed),
        "valid_universe_groups": sum(bool(v) for v in universe_catalog.values()),
        "valid_ticker_groups": sum(bool(v) for v in ticker_catalog.values()),
        "valid_v2_features": len(v2_features), "valid_v3_features": len(v3_features),
        "ready_buckets": len(ready_buckets), "ready_tickers": int(df["ticker"].nunique()),
        "preflight_warnings": preflight_errors, "allow_partial_data": allow_partial_data,
        "modes": sorted(modes), "result_dir": str(paths.result_dual),
    }
    atomic_json(result, paths.result_dual / "run_summary.json")
    return result
