from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..config import get_paths, load_baskets
from ..experiment.splits import make_walk_forward_folds
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_date, normalize_ticker, stable_hash
from .model import CANDIDATES, ModelConfig, fit_xgb
from .monitor import ResourceMonitor, prevent_windows_sleep, restore_windows_sleep

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Experiment:
    name: str
    stage: str
    mode: str
    group: str | None = None
    target_bucket: str | None = None
    target_ticker: str | None = None


@dataclass
class TaskResult:
    experiment: str
    stage: str
    mode: str
    group: str | None
    target_bucket: str | None
    target_ticker: str | None
    fold: int
    seed: int
    feature_count: int
    backend: str
    elapsed_seconds: float
    cache_status: str
    prediction_path: str


def _hash_file_metadata(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    payload = {
        "path": str(path.resolve()), "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "sha256": digest.hexdigest(),
    }
    return stable_hash(payload)


def _load_catalog(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _merge_catalogs(*catalogs: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for catalog in catalogs:
        for group, cols in catalog.items():
            out[group] = sorted(set(out.get(group, [])) | set(cols))
    return out


def _feature_quality(df: pd.DataFrame, candidates: Iterable[str], min_ticker_coverage: float = 0.80) -> tuple[list[str], pd.DataFrame]:
    tickers = df["ticker"].astype(str)
    ticker_count = tickers.nunique()
    rows = []
    valid = []
    for col in sorted(set(candidates)):
        if col not in df.columns:
            rows.append({"feature": col, "status": "missing_column", "missing_ratio": 1.0, "ticker_coverage": 0.0})
            continue
        values = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        nonnull_by_ticker = values.notna().groupby(tickers).sum()
        coverage = float((nonnull_by_ticker >= 100).sum() / max(1, ticker_count))
        missing = float(values.isna().mean())
        unique = int(values.nunique(dropna=True))
        nonnull = int(values.notna().sum())
        status = "valid" if missing <= 0.98 and unique >= 2 and nonnull >= 500 and coverage >= min_ticker_coverage else "invalid"
        rows.append({
            "feature": col, "status": status, "missing_ratio": missing,
            "unique_count": unique, "non_null_count": nonnull,
            "ticker_coverage": coverage, "required_ticker_coverage": min_ticker_coverage,
        })
        if status == "valid":
            valid.append(col)
    return valid, pd.DataFrame(rows)


def _metrics(y: np.ndarray, p: np.ndarray, dates: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(y, dtype=np.int8)
    pred_label = (p >= threshold).astype(np.int8)
    unique = np.unique(y)
    result = {
        "rows": int(len(y)), "positives": int(y.sum()), "positive_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y, pred_label)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred_label)) if len(unique) == 2 else np.nan,
        "precision": float(precision_score(y, pred_label, zero_division=0)),
        "recall": float(recall_score(y, pred_label, zero_division=0)),
        "f1": float(f1_score(y, pred_label, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)) if len(unique) == 2 else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(unique) == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "mean_prediction": float(p.mean()),
    }
    for fraction in (0.01, 0.03, 0.05):
        selected = np.zeros(len(y), dtype=bool)
        date_series = pd.Series(dates)
        for _, idx in date_series.groupby(date_series, sort=False).groups.items():
            idx_array = np.asarray(list(idx), dtype=int)
            count = max(1, int(math.ceil(len(idx_array) * fraction)))
            local = idx_array[np.argpartition(p[idx_array], -count)[-count:]]
            selected[local] = True
        tp = int(np.sum(selected & (y == 1)))
        count = int(selected.sum())
        result[f"top_{int(fraction*100)}pct_precision"] = tp / count if count else np.nan
        result[f"top_{int(fraction*100)}pct_recall"] = tp / int(y.sum()) if y.sum() else np.nan
    return result


def _exact_sign_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    values = values[values != 0]
    n = len(values)
    if n == 0:
        return np.nan
    k = int(np.sum(values > 0))
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def _summarize_deltas(delta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["experiment", "stage", "mode", "group", "target_bucket", "target_ticker", "scope_type", "scope_value"]
    metric_cols = ["pr_auc_loss", "roc_auc_loss", "balanced_accuracy_loss", "accuracy_loss", "brier_increase", "logloss_increase", "top_3pct_precision_loss"]
    for key_values, block in delta.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key_values if isinstance(key_values, tuple) else (key_values,)))
        fold = block.groupby("fold", as_index=False)[metric_cols].mean(numeric_only=True)
        row["fold_count"] = int(fold["fold"].nunique())
        row["seed_count"] = int(block["seed"].nunique())
        for col in metric_cols:
            values = pd.to_numeric(fold[col], errors="coerce").dropna().to_numpy()
            row[f"{col}_mean"] = float(np.mean(values)) if len(values) else np.nan
            row[f"{col}_median"] = float(np.median(values)) if len(values) else np.nan
            row[f"{col}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row[f"{col}_positive_fold_ratio"] = float(np.mean(values > 0)) if len(values) else np.nan
            row[f"{col}_sign_p"] = _exact_sign_p(values)
        rows.append(row)
    return pd.DataFrame(rows)


class Finance11HRunner:
    def __init__(
        self,
        project: Path,
        dataset: Path | None,
        *,
        hours: float = 11.0,
        threads: int = 16,
        folds: int = 8,
        seeds: list[int] | None = None,
        validation_days: int = 60,
        purge_days: int = 20,
        prefer_gpu: bool = True,
        overwrite_cache: bool = False,
        cache_namespace: str = "finance11h_v3",
    ) -> None:
        self.project = project.resolve()
        self.paths = get_paths(self.project)
        self.dataset = dataset or self.paths.data_root / "development" / "training_dataset_finance11h.parquet"
        self.hours = hours
        self.threads = threads
        self.n_folds = folds
        self.seeds = seeds or [17, 43]
        self.validation_days = validation_days
        self.purge_days = purge_days
        self.prefer_gpu = prefer_gpu
        self.overwrite_cache = overwrite_cache
        self.cache_namespace = cache_namespace
        self.started = time.monotonic()
        self.deadline = self.started + hours * 3600
        self.reserve_seconds = 12 * 60
        self.result_dir = self.paths.data_root / "ablation_finance11h"
        self.cache_dir = self.result_dir / "prediction_cache" / cache_namespace
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.task_times: list[float] = []
        self.monitor = ResourceMonitor(self.result_dir / "resource_usage.csv", interval_seconds=10, hard_gpu_temp=88, min_available_ram_gb=1.5)

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def can_start(self) -> bool:
        if self.monitor.abort_event.is_set():
            return False
        estimate = np.median(self.task_times[-20:]) if self.task_times else 120.0
        return self.remaining() > self.reserve_seconds + max(180.0, estimate * 1.4)

    def load(self) -> None:
        if not self.dataset.exists():
            raise FileNotFoundError(f"11시간 금융 학습 데이터가 없습니다: {self.dataset}")
        df = normalize_date(pd.read_parquet(self.dataset))
        df["ticker"] = normalize_ticker(df["ticker"])
        if "label_abs_crash_20" not in df.columns:
            raise KeyError("label_abs_crash_20이 없습니다.")
        baskets = load_baskets(self.paths)
        if "bucket" not in df.columns:
            df = df.merge(baskets[["ticker", "bucket"]], on="ticker", how="left")
        df["bucket"] = df["bucket"].fillna("other").astype(str)
        self.baskets = baskets.loc[baskets["ticker"].isin(set(df["ticker"]))].copy()

        base_u = _load_catalog(self.paths.feature_dual / "feature_catalog_universe.json")
        base_t = _load_catalog(self.paths.feature_dual / "feature_catalog_ticker.json")
        fin_u = _load_catalog(self.paths.feature_dual / "feature_catalog_finance11h_universe.json")
        fin_t = _load_catalog(self.paths.feature_dual / "feature_catalog_finance11h_ticker.json")
        self.catalog = _merge_catalogs(base_u, base_t, fin_u, fin_t)
        candidates = sorted(set(sum(self.catalog.values(), [])))
        valid, quality = _feature_quality(df, candidates)
        quality["group"] = quality["feature"].map(lambda c: next((g for g, cols in self.catalog.items() if c in cols), "unmapped"))
        atomic_csv(quality, self.result_dir / "valid_feature_audit.csv")
        valid_set = set(valid)
        self.catalog = {group: [c for c in cols if c in valid_set] for group, cols in self.catalog.items()}
        self.features = sorted(valid_set)

        mandatory = ["t_financial_shorting", "u_financial_shorting", "t_stock_lending", "u_stock_lending"]
        for group in mandatory:
            if len(self.catalog.get(group, [])) < 5:
                raise RuntimeError(f"필수 금융 그룹 {group}의 유효 피처가 5개 미만입니다: {len(self.catalog.get(group, []))}")

        self.df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.X = np.ascontiguousarray(
            self.df[self.features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32),
            dtype=np.float32,
        )
        self.y = pd.to_numeric(self.df["label_abs_crash_20"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
        self.dates = self.df["date"].to_numpy()
        self.tickers = self.df["ticker"].astype(str).to_numpy()
        self.buckets = self.df["bucket"].astype(str).to_numpy()
        self.feature_index = {c: i for i, c in enumerate(self.features)}
        self.folds = make_walk_forward_folds(self.df["date"], self.n_folds, self.validation_days, self.purge_days, 500)
        # Row indices are reused by hundreds of tasks. Precompute them once.
        date_values = self.df["date"].to_numpy()
        self.fold_indices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for fold in self.folds:
            train_idx = np.flatnonzero(np.isin(date_values, pd.DatetimeIndex(fold["train_dates_index"]).to_numpy()))
            val_idx = np.flatnonzero(np.isin(date_values, pd.DatetimeIndex(fold["validation_dates_index"]).to_numpy()))
            self.fold_indices[int(fold["fold_id"])] = (train_idx, val_idx)
        atomic_json([f["metadata"] for f in self.folds], self.result_dir / "walk_forward_folds.json")
        self.dataset_signature = stable_hash({
            "metadata": _hash_file_metadata(self.dataset), "rows": len(self.df),
            "features": self.features, "date_min": str(self.df["date"].min()), "date_max": str(self.df["date"].max()),
        })
        atomic_json({
            "dataset": str(self.dataset), "rows": len(self.df), "tickers": int(self.df["ticker"].nunique()),
            "features": len(self.features), "dataset_signature": self.dataset_signature,
            "mandatory_short_features": len(self.catalog["t_financial_shorting"]),
            "mandatory_market_short_features": len(self.catalog["u_financial_shorting"]),
            "mandatory_lending_features": len(self.catalog["t_stock_lending"]),
            "mandatory_market_lending_features": len(self.catalog["u_stock_lending"]),
        }, self.result_dir / "data_summary.json")

    def _cache_path(self, exp: Experiment, fold: int, seed: int, config: ModelConfig, feature_names: list[str]) -> Path:
        payload = {
            **asdict(exp), "fold": fold, "seed": seed, "config": config.payload(),
            "features": feature_names, "dataset": self.dataset_signature,
            "cache_namespace": self.cache_namespace,
        }
        return self.cache_dir / f"{stable_hash(payload)}.parquet"

    def _indices(self, fold: dict) -> tuple[np.ndarray, np.ndarray]:
        return self.fold_indices[int(fold["fold_id"])]

    def _matrix(self, exp: Experiment, train_idx: np.ndarray, val_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        group_features = self.catalog.get(exp.group or "", [])
        if exp.mode == "global_drop":
            drop = set(group_features)
            names = [c for c in self.features if c not in drop]
            cols = np.asarray([self.feature_index[c] for c in names], dtype=int)
            return np.ascontiguousarray(self.X[train_idx][:, cols]), np.ascontiguousarray(self.X[val_idx][:, cols]), names
        names = self.features
        x_train = np.ascontiguousarray(self.X[train_idx])
        x_val = np.ascontiguousarray(self.X[val_idx])
        if exp.mode in {"bucket_mask", "ticker_mask"}:
            cols = np.asarray([self.feature_index[c] for c in group_features if c in self.feature_index], dtype=int)
            if exp.mode == "bucket_mask":
                train_rows = np.flatnonzero(self.buckets[train_idx] == exp.target_bucket)
                val_rows = np.flatnonzero(self.buckets[val_idx] == exp.target_bucket)
            else:
                train_rows = np.flatnonzero(self.tickers[train_idx] == exp.target_ticker)
                val_rows = np.flatnonzero(self.tickers[val_idx] == exp.target_ticker)
            if len(cols):
                x_train[np.ix_(train_rows, cols)] = np.nan
                x_val[np.ix_(val_rows, cols)] = np.nan
        return x_train, x_val, names

    def _scope_metrics(self, pred: pd.DataFrame, exp: Experiment, fold_id: int, seed: int) -> pd.DataFrame:
        rows = []
        scopes = [("all", "all_validation", pred)]
        # Baseline must expose every comparison scope so target bucket/ticker runs
        # can be paired against exactly the same validation rows.
        if exp.name == "baseline":
            for bucket in sorted(self.baskets["bucket"].unique()):
                scopes.append(("bucket", str(bucket), pred.loc[pred["bucket"].eq(str(bucket))]))
            for ticker in sorted(self.baskets["ticker"].astype(str).unique()):
                scopes.append(("ticker", str(ticker).zfill(6), pred.loc[pred["ticker"].eq(str(ticker).zfill(6))]))
        if exp.target_bucket:
            scopes.append(("target_bucket", exp.target_bucket, pred.loc[pred["bucket"].eq(exp.target_bucket)]))
        if exp.target_ticker:
            scopes.append(("target_ticker", exp.target_ticker, pred.loc[pred["ticker"].eq(exp.target_ticker)]))
        for scope_type, scope_value, block in scopes:
            if block.empty:
                continue
            rows.append({
                "experiment": exp.name, **asdict(exp), "fold": fold_id, "seed": seed,
                "scope_type": scope_type, "scope_value": scope_value,
                **_metrics(block["target"].to_numpy(), block["prediction"].to_numpy(), block["date"].to_numpy()),
            })
        return pd.DataFrame(rows)

    def execute(self, exp: Experiment, fold: dict, seed: int, config: ModelConfig) -> tuple[pd.DataFrame, TaskResult]:
        fold_id = int(fold["fold_id"])
        train_idx, val_idx = self._indices(fold)
        x_train, x_val, names = self._matrix(exp, train_idx, val_idx)
        cache = self._cache_path(exp, fold_id, seed, config, names)
        started = time.perf_counter()
        if cache.exists() and not self.overwrite_cache:
            pred = pd.read_parquet(cache)
            backend = "cache"
            status = "hit"
        else:
            model, backend = fit_xgb(x_train, self.y[train_idx], seed, config, threads=self.threads, prefer_gpu=self.prefer_gpu)
            probability = model.predict_proba(x_val)[:, 1]
            pred = self.df.loc[val_idx, ["date", "ticker", "bucket"]].copy()
            pred["target"] = self.y[val_idx]
            pred["prediction"] = probability.astype(np.float32)
            atomic_parquet(pred, cache)
            status = "miss"
        elapsed = time.perf_counter() - started
        if status == "miss":
            self.task_times.append(elapsed)
        metrics = self._scope_metrics(pred, exp, fold_id, seed)
        result = TaskResult(
            experiment=exp.name, stage=exp.stage, mode=exp.mode, group=exp.group,
            target_bucket=exp.target_bucket, target_ticker=exp.target_ticker,
            fold=fold_id, seed=seed, feature_count=len(names), backend=backend,
            elapsed_seconds=elapsed, cache_status=status, prediction_path=str(cache),
        )
        return metrics, result

    def tune(self) -> ModelConfig:
        rows = []
        tune_folds = self.folds[-2:]
        exp = Experiment("tune_baseline", "tune", "none")
        for config in CANDIDATES:
            for fold in tune_folds:
                if not self.can_start():
                    break
                metrics, record = self.execute(exp, fold, 17, config)
                row = metrics.loc[metrics["scope_type"].eq("all")].iloc[0].to_dict()
                row.update({"config": config.name, "elapsed_seconds": record.elapsed_seconds})
                rows.append(row)
        table = pd.DataFrame(rows)
        atomic_csv(table, self.result_dir / "model_tuning_by_fold.csv")
        if table.empty:
            return CANDIDATES[1]
        summary = table.groupby("config", as_index=False).agg(
            pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"), brier=("brier", "mean"),
            elapsed_seconds=("elapsed_seconds", "sum"),
        )
        summary["score"] = summary["pr_auc"] + 0.35 * summary["roc_auc"] + 0.20 * summary["balanced_accuracy"] - 0.10 * summary["brier"]
        summary = summary.sort_values(["score", "elapsed_seconds"], ascending=[False, True])
        atomic_csv(summary, self.result_dir / "model_tuning_summary.csv")
        selected_name = str(summary.iloc[0]["config"])
        selected = next(c for c in CANDIDATES if c.name == selected_name)
        atomic_json(selected.payload(), self.result_dir / "selected_model_config.json")
        return selected

    def _run_batch(self, experiments: list[Experiment], fold_ids: list[int], seeds: list[int], config: ModelConfig, metrics_frames: list[pd.DataFrame], records: list[dict]) -> None:
        for exp in experiments:
            if exp.name != "baseline" and not self.catalog.get(exp.group or "", []):
                records.append({**asdict(exp), "status": "skipped_no_valid_features"})
                continue
            for fold_id in fold_ids:
                fold = self.folds[fold_id]
                for seed in seeds:
                    if not self.can_start():
                        return
                    LOGGER.info("실행 %s fold=%s seed=%s remaining=%.2fh", exp.name, fold_id, seed, self.remaining()/3600)
                    metrics, result = self.execute(exp, fold, seed, config)
                    metrics_frames.append(metrics)
                    records.append({**asdict(result), "status": "completed"})
                    atomic_csv(pd.concat(metrics_frames, ignore_index=True), self.result_dir / "metrics_incremental.csv")
                    atomic_csv(pd.DataFrame(records), self.result_dir / "task_manifest.csv")
                    atomic_json({
                        "updated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                        "remaining_hours": self.remaining()/3600, "completed_tasks": len([r for r in records if r.get("status") == "completed"]),
                        "median_new_task_seconds": float(np.median(self.task_times)) if self.task_times else None,
                        "temperature_abort": self.monitor.abort_event.is_set(),
                    }, self.result_dir / "run_state.json")

    def run(self) -> dict:
        prevent_windows_sleep()
        self.monitor.start()
        metrics_frames: list[pd.DataFrame] = []
        records: list[dict] = []
        try:
            self.load()
            config = self.tune()

            all_fold_ids = list(range(len(self.folds)))
            baseline = [Experiment("baseline", "baseline", "none")]
            self._run_batch(baseline, all_fold_ids, self.seeds, config, metrics_frames, records)

            mandatory_groups = [
                "t_financial_shorting", "u_financial_shorting",
                "t_stock_lending", "u_stock_lending",
                "t_financial_flow", "u_financial_flow",
                "t_financial_valuation", "t_financial_interaction", "u_financial_market",
                "t_short_selling", "u_aggregate_shorting", "t_investor_flow", "u_aggregate_flow",
                "t_valuation_size", "u_credit_funding", "u_derivatives_risk",
            ]
            global_experiments = [Experiment(f"global__{g}", "financial_global", "global_drop", g) for g in mandatory_groups if self.catalog.get(g)]
            self._run_batch(global_experiments, all_fold_ids, self.seeds, config, metrics_frames, records)

            # Numba-generated finance subclusters isolate transaction, balance and crowding effects.
            short_cols = self.catalog.get("t_financial_shorting", [])
            subclusters = {
                "t_finshort_volume_cluster": [c for c in short_cols if "volume" in c or "value_ratio" in c],
                "t_finshort_balance_cluster": [c for c in short_cols if "balance" in c and not any(k in c for k in ["change", "slope", "acceleration"])],
                "t_finshort_dynamic_cluster": [c for c in short_cols if any(k in c for k in ["change", "slope", "acceleration", "divergence"])],
                "t_finshort_crowding_cluster": [c for c in short_cols if any(k in c for k in ["crowding", "squeeze", "stress", "crash_pressure"])],
            }
            lending_cols = self.catalog.get("t_stock_lending", [])
            subclusters.update({
                "t_lending_transaction_cluster": [c for c in lending_cols if any(k in c for k in ["contract", "net_shares", "net_value"])],
                "t_lending_balance_cluster": [c for c in lending_cols if "balance" in c and not any(k in c for k in ["change", "slope", "acceleration"])],
                "t_lending_dynamic_cluster": [c for c in lending_cols if any(k in c for k in ["change", "slope", "acceleration", "divergence"])],
                "t_lending_crowding_cluster": [c for c in lending_cols if any(k in c for k in ["crowding", "squeeze"])],
            })
            self.catalog.update({k: sorted(set(v)) for k, v in subclusters.items() if v})
            sub_exp = [Experiment(f"subcluster__{g}", "financial_subcluster", "global_drop", g) for g in subclusters if self.catalog.get(g)]
            self._run_batch(sub_exp, all_fold_ids[-6:], [17], config, metrics_frames, records)

            supplemental_groups = [
                "t_limit_stress", "t_tail_dependence", "t_microstructure_proxy", "t_range_volatility",
                "t_volume_price_pressure", "t_network_contagion", "u_etf_pressure", "u_credit_funding",
            ]
            supplemental = [Experiment(f"supplemental__{g}", "supplemental", "global_drop", g) for g in supplemental_groups if self.catalog.get(g)]
            self._run_batch(supplemental, all_fold_ids[-6:], [17], config, metrics_frames, records)

            # Targeted bucket tests are restricted to the strongest mandatory finance groups.
            bucket_groups = [g for g in [
                "t_financial_shorting", "t_stock_lending", "t_financial_flow",
                "t_finshort_crowding_cluster", "t_lending_crowding_cluster", "t_financial_interaction",
            ] if self.catalog.get(g)]
            bucket_exp = [
                Experiment(f"bucket__{bucket}__{group}", "bucket", "bucket_mask", group, target_bucket=bucket)
                for bucket in sorted(self.baskets["bucket"].unique()) for group in bucket_groups
            ]
            self._run_batch(bucket_exp, all_fold_ids[-4:], [17], config, metrics_frames, records)

            # Ticker tests consume remaining time. Financials and short-sensitive high-beta names are first.
            priority_buckets = ["finance", "battery_material", "bio", "semiconductor", "shipbuilding", "defense_nuclear", "automobile", "internet_game"]
            ordered = self.baskets.assign(_priority=self.baskets["bucket"].map({b: i for i, b in enumerate(priority_buckets)}).fillna(99)).sort_values(["_priority", "ticker"])
            ticker_groups = [g for g in [
                "t_financial_shorting", "t_stock_lending", "t_finshort_crowding_cluster",
                "t_lending_crowding_cluster", "t_financial_flow",
            ] if self.catalog.get(g)]
            ticker_exp = [
                Experiment(f"ticker__{ticker}__{group}", "ticker", "ticker_mask", group, target_bucket=bucket, target_ticker=ticker)
                for ticker, bucket in ordered[["ticker", "bucket"]].itertuples(index=False, name=None)
                for group in ticker_groups
            ]
            self._run_batch(ticker_exp, all_fold_ids[-3:], [17], config, metrics_frames, records)

            if not metrics_frames:
                raise RuntimeError("완료된 실험이 없습니다.")
            metrics = pd.concat(metrics_frames, ignore_index=True)
            atomic_csv(metrics, self.result_dir / "all_metrics.csv")
            baseline_metrics = metrics.loc[metrics["experiment"].eq("baseline")].copy()
            metrics = metrics.copy()
            metrics["pair_scope_type"] = metrics["scope_type"].replace({"target_bucket": "bucket", "target_ticker": "ticker"})
            key = ["fold", "seed", "pair_scope_type", "scope_value"]
            base_cols = ["pr_auc", "roc_auc", "balanced_accuracy", "accuracy", "brier", "logloss", "top_3pct_precision"]
            base = baseline_metrics.copy()
            base["pair_scope_type"] = base["scope_type"].replace({"target_bucket": "bucket", "target_ticker": "ticker"})
            base = base[key + base_cols].drop_duplicates(key).rename(columns={c: f"baseline_{c}" for c in base_cols})
            compared = metrics.loc[~metrics["experiment"].isin(["baseline", "tune_baseline"])].merge(base, on=key, how="left", validate="many_to_one")
            missing_pair = compared["baseline_pr_auc"].isna() & compared["pr_auc"].notna()
            if missing_pair.any():
                sample = compared.loc[missing_pair, ["experiment", "fold", "seed", "scope_type", "scope_value"]].head(10)
                raise RuntimeError(f"baseline pairing 실패:\n{sample.to_string(index=False)}")
            compared["pr_auc_loss"] = compared["baseline_pr_auc"] - compared["pr_auc"]
            compared["roc_auc_loss"] = compared["baseline_roc_auc"] - compared["roc_auc"]
            compared["balanced_accuracy_loss"] = compared["baseline_balanced_accuracy"] - compared["balanced_accuracy"]
            compared["accuracy_loss"] = compared["baseline_accuracy"] - compared["accuracy"]
            compared["brier_increase"] = compared["brier"] - compared["baseline_brier"]
            compared["logloss_increase"] = compared["logloss"] - compared["baseline_logloss"]
            compared["top_3pct_precision_loss"] = compared["baseline_top_3pct_precision"] - compared["top_3pct_precision"]
            atomic_csv(compared, self.result_dir / "paired_ablation_deltas.csv")
            summary = _summarize_deltas(compared)
            atomic_csv(summary, self.result_dir / "ablation_summary.csv")

            bucket_source = summary.loc[
                summary["scope_type"].eq("target_bucket")
                & summary["target_bucket"].notna()
                & summary["scope_value"].eq(summary["target_bucket"])
            ]
            if not bucket_source.empty:
                matrix = bucket_source.pivot_table(index="target_bucket", columns="group", values="pr_auc_loss_mean", aggfunc="mean")
                atomic_csv(matrix.reset_index(), self.result_dir / "bucket_finance_sensitivity_matrix.csv")
            ticker_source = summary.loc[
                summary["scope_type"].eq("target_ticker")
                & summary["target_ticker"].notna()
                & summary["scope_value"].astype(str).str.zfill(6).eq(summary["target_ticker"].astype(str).str.zfill(6))
            ]
            if not ticker_source.empty:
                matrix = ticker_source.pivot_table(index="target_ticker", columns="group", values="pr_auc_loss_mean", aggfunc="mean")
                atomic_csv(matrix.reset_index(), self.result_dir / "ticker_finance_sensitivity_matrix.csv")

            baseline_all = baseline_metrics.loc[baseline_metrics["scope_type"].eq("all")]
            means = baseline_all.select_dtypes(include=[np.number]).mean().to_dict()
            goal = {
                "note": "70%는 보장값이 아니라 목표입니다. 불균형 라벨에서는 raw accuracy 단독으로 성공 판정하지 않습니다.",
                "mean_accuracy": means.get("accuracy"),
                "mean_balanced_accuracy": means.get("balanced_accuracy"),
                "mean_roc_auc": means.get("roc_auc"),
                "mean_pr_auc": means.get("pr_auc"),
                "mean_top_3pct_precision": means.get("top_3pct_precision"),
                "raw_accuracy_ge_70": bool(means.get("accuracy", 0) >= 0.70),
                "balanced_accuracy_ge_70": bool(means.get("balanced_accuracy", 0) >= 0.70),
                "roc_auc_ge_70": bool(means.get("roc_auc", 0) >= 0.70),
                "top_3pct_precision_ge_70": bool(means.get("top_3pct_precision", 0) >= 0.70),
            }
            goal["robust_goal_met"] = bool(goal["balanced_accuracy_ge_70"] and goal["roc_auc_ge_70"])
            atomic_json(goal, self.result_dir / "goal_70_report.json")

            result = {
                "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
                "budget_hours": self.hours, "elapsed_hours": (time.monotonic() - self.started) / 3600,
                "deadline_reached": not self.can_start(), "temperature_abort": self.monitor.abort_event.is_set(),
                "completed_tasks": int(sum(r.get("status") == "completed" for r in records)),
                "cache_hits": int(sum(r.get("cache_status") == "hit" for r in records)),
                "new_models": int(sum(r.get("cache_status") == "miss" for r in records)),
                "selected_config": config.payload(), "threads": self.threads,
                "features": len(self.features), "folds": len(self.folds), "seeds": self.seeds,
                "goal_70": goal, "result_dir": str(self.result_dir),
            }
            atomic_json(result, self.result_dir / "run_summary.json")
            return result
        finally:
            self.monitor.stop()
            restore_windows_sleep()
