from __future__ import annotations

import json
import logging
import math
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..config import get_paths, load_baskets
from ..experiment.splits import make_walk_forward_folds
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_date, normalize_ticker, stable_hash
from ..finance11h.model import CANDIDATES, ModelConfig, fit_xgb
from ..finance11h.monitor import ResourceMonitor, prevent_windows_sleep, restore_windows_sleep
from ..finance11h.runner import (
    Experiment,
    _feature_quality,
    _hash_file_metadata,
    _load_catalog,
    _merge_catalogs,
    _metrics,
)
from .calibration import CalibrationPolicy, apply_calibrator, select_train_only_policy
from .control import RunLock

LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "focused_nested_v1"
TUNE_SEED = 1701
CALIBRATION_SEED = 2718


def _bh_qvalues(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    valid = numeric.dropna().sort_values()
    if valid.empty:
        return pd.Series(np.nan, index=values.index)
    m = len(valid)
    ranked = np.empty(m, dtype=float)
    running = 1.0
    arr = valid.to_numpy(dtype=float)
    for i in range(m - 1, -1, -1):
        rank = i + 1
        running = min(running, arr[i] * m / rank)
        ranked[i] = min(1.0, running)
    result = pd.Series(np.nan, index=values.index, dtype=float)
    result.loc[valid.index] = ranked
    return result


def _exact_sign_p(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values != 0]
    n = len(values)
    if n == 0:
        return np.nan
    k = int(np.sum(values > 0))
    smaller = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(smaller + 1)) / (2 ** n)
    return float(min(1.0, 2.0 * tail))


def _model_config_from_payload(payload: dict[str, Any]) -> ModelConfig:
    allowed = set(ModelConfig.__dataclass_fields__)
    return ModelConfig(**{k: payload[k] for k in allowed if k in payload})


def _metric_bundle(y: np.ndarray, raw_p: np.ndarray, calibrated_p: np.ndarray, dates: np.ndarray, threshold: float) -> dict[str, float]:
    result = _metrics(y, calibrated_p, dates, threshold=threshold)
    raw = _metrics(y, raw_p, dates, threshold=0.5)
    for key in ["pr_auc", "roc_auc", "brier", "logloss", "mean_prediction", "top_1pct_precision", "top_3pct_precision", "top_5pct_precision"]:
        result[f"raw_{key}"] = raw.get(key, np.nan)
    prevalence = float(np.mean(y)) if len(y) else np.nan
    if np.isfinite(prevalence) and 0 < prevalence < 1:
        naive_brier = prevalence * (1.0 - prevalence)
        naive_logloss = -(prevalence * math.log(prevalence) + (1.0 - prevalence) * math.log(1.0 - prevalence))
        result["brier_skill"] = 1.0 - result["brier"] / naive_brier if naive_brier > 0 else np.nan
        result["logloss_skill"] = 1.0 - result["logloss"] / naive_logloss if naive_logloss > 0 else np.nan
        result["pr_auc_lift"] = result["pr_auc"] / prevalence
        result["raw_pr_auc_lift"] = result["raw_pr_auc"] / prevalence
    else:
        result.update({"brier_skill": np.nan, "logloss_skill": np.nan, "pr_auc_lift": np.nan, "raw_pr_auc_lift": np.nan})
    result["decision_threshold"] = float(threshold)
    return result


class FocusedNestedRunner:
    def __init__(
        self,
        project: Path,
        dataset: Path | None,
        *,
        profile: str,
        threads: int,
        prefer_gpu: bool,
        hours: float = 0.0,
        cache_namespace: str = "finance_nested_focus_v1",
        outer_folds: int = 8,
        seeds: list[int] | None = None,
        validation_days: int = 60,
        purge_days: int = 20,
        inner_folds: int = 3,
        inner_validation_days: int = 40,
        calibration_days: int = 60,
        calibration_purge_days: int = 20,
        clear_stop_on_start: bool = False,
    ) -> None:
        self.project = project.resolve()
        self.paths = get_paths(self.project)
        self.dataset = dataset or self.paths.data_root / "development" / "training_dataset_finance11h.parquet"
        self.profile = profile
        self.threads = threads
        self.prefer_gpu = prefer_gpu
        self.hours = float(hours)
        self.deadline = time.monotonic() + self.hours * 3600 if self.hours > 0 else math.inf
        self.cache_namespace = cache_namespace
        self.outer_folds_count = outer_folds
        self.seeds = seeds or [17, 43, 101]
        self.validation_days = validation_days
        self.purge_days = purge_days
        self.inner_folds_count = inner_folds
        self.inner_validation_days = inner_validation_days
        self.calibration_days = calibration_days
        self.calibration_purge_days = calibration_purge_days
        self.started = time.monotonic()
        self.run_id = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

        self.result_dir = self.paths.data_root / "ablation_finance_nested_focus"
        self.cache_dir = self.result_dir / "prediction_cache" / cache_namespace
        self.policy_dir = self.result_dir / "nested_policies" / cache_namespace
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.policy_dir.mkdir(parents=True, exist_ok=True)
        self.stop_flag = self.result_dir / "REQUEST_SAFE_STOP.flag"
        self.lock_path = self.result_dir / "RUNNING.lock"
        if clear_stop_on_start:
            self.stop_flag.unlink(missing_ok=True)
        self.monitor = ResourceMonitor(
            self.result_dir / f"resource_usage_{self.run_id}_{self.profile}.csv",
            interval_seconds=10,
            hard_gpu_temp=88 if prefer_gpu else 95,
            min_available_ram_gb=2.0 if profile == "full" else 3.0,
        )
        self.plan = json.loads((self.paths.configs / "focused_nested_plan.json").read_text(encoding="utf-8"))
        self.metrics_path = self.result_dir / "all_metrics_incremental.csv"
        self.manifest_path = self.result_dir / "task_manifest.csv"
        self.policy_manifest_path = self.result_dir / "nested_policy_manifest.csv"
        self.calibration_manifest_path = self.result_dir / "calibration_policy_manifest.csv"
        self.current_stage = "initializing"

    def _global_plan_items(self) -> list[dict[str, Any]]:
        items = list(self.plan["global_groups"])
        extension = self.plan.get("extension", {})
        if extension.get("enabled"):
            items.extend(extension.get("global_groups", []))
        return items

    def _bucket_plan_items(self) -> list[dict[str, Any]]:
        items = list(self.plan["bucket_combinations"])
        extension = self.plan.get("extension", {})
        if extension.get("enabled"):
            for bucket in extension.get("buckets", []):
                for group in extension.get("bucket_groups", []):
                    items.append({
                        "bucket": bucket,
                        "group": group["group"],
                        "components": group.get("components"),
                        "reason": group.get("reason", ""),
                    })
        deduplicated: dict[tuple[str, str], dict[str, Any]] = {}
        for item in items:
            deduplicated.setdefault((item["bucket"], item["group"]), item)
        return list(deduplicated.values())

    def _materialize_composite_groups(self) -> None:
        for item in self._global_plan_items() + self._bucket_plan_items():
            components = item.get("components") or []
            if not components:
                continue
            missing = [group for group in components if not self.catalog.get(group)]
            if missing:
                raise RuntimeError(
                    f"복합 그룹 {item['group']}의 구성 그룹이 없습니다: {missing}"
                )
            self.catalog[item["group"]] = sorted({
                feature
                for group in components
                for feature in self.catalog[group]
            })

    def stop_requested(self) -> bool:
        return self.stop_flag.exists() or self.monitor.abort_event.is_set() or time.monotonic() >= self.deadline

    def _state(self, **extra: Any) -> None:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "profile": self.profile,
            "threads": self.threads,
            "prefer_gpu": self.prefer_gpu,
            "cache_namespace": self.cache_namespace,
            "stage": self.current_stage,
            "elapsed_hours": (time.monotonic() - self.started) / 3600,
            "safe_stop_requested": self.stop_flag.exists(),
            "resource_abort": self.monitor.abort_event.is_set(),
            "deadline_reached": time.monotonic() >= self.deadline,
            "updated_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            **extra,
        }
        atomic_json(payload, self.result_dir / "run_state.json")

    def load(self) -> None:
        if not self.dataset.exists():
            raise FileNotFoundError(f"Finance11H 학습 데이터가 없습니다: {self.dataset}")
        df = normalize_date(pd.read_parquet(self.dataset))
        df["ticker"] = normalize_ticker(df["ticker"])
        if "label_abs_crash_20" not in df.columns:
            raise KeyError("label_abs_crash_20이 없습니다.")
        baskets = load_baskets(self.paths)
        if "bucket" not in df.columns:
            df = df.merge(baskets[["ticker", "bucket"]], on="ticker", how="left")
        df["bucket"] = df["bucket"].fillna("other").astype(str)
        self.baskets = baskets.loc[baskets["ticker"].isin(set(df["ticker"]))].copy()

        required_summary_path = self.paths.raw_dual / "required_data_v3" / "required_data_summary.json"
        if not required_summary_path.exists():
            raise RuntimeError(
                "required_data_summary.json이 없습니다. 먼저 RUN_REQUIRED_DATA_DOWNLOAD.bat과 "
                "CHECK_REQUIRED_DATA.bat을 실행해야 합니다."
            )
        required_summary = json.loads(required_summary_path.read_text(encoding="utf-8"))
        actual_short = required_summary.get("actual_short", {})
        lending = required_summary.get("stock_lending", {})
        if not actual_short.get("is_actual_short") or float(actual_short.get("coverage", 0.0)) < 0.80:
            raise RuntimeError("실제 KRX 공매도 데이터 coverage 80% 기준을 통과하지 못했습니다.")
        if float(lending.get("coverage", 0.0)) < 0.80:
            raise RuntimeError("종목별 대차 데이터 coverage 80% 기준을 통과하지 못했습니다.")
        atomic_json({
            "actual_short": actual_short,
            "stock_lending": lending,
            "investor_flow_note": required_summary.get("investor_flow_foreign_ownership", {}),
            "macro": required_summary.get("macro", {}),
        }, self.result_dir / "focused_data_preflight.json")

        base_u = _load_catalog(self.paths.feature_dual / "feature_catalog_universe.json")
        base_t = _load_catalog(self.paths.feature_dual / "feature_catalog_ticker.json")
        fin_u = _load_catalog(self.paths.feature_dual / "feature_catalog_finance11h_universe.json")
        fin_t = _load_catalog(self.paths.feature_dual / "feature_catalog_finance11h_ticker.json")
        self.catalog = _merge_catalogs(base_u, base_t, fin_u, fin_t)
        short_cols = self.catalog.get("t_financial_shorting", [])
        self.catalog["t_finshort_crowding_cluster"] = sorted({
            c for c in short_cols if any(k in c for k in ["crowding", "squeeze", "stress", "crash_pressure"])
        })
        self._materialize_composite_groups()

        required_groups = (
            [x["group"] for x in self._global_plan_items()]
            + [x["group"] for x in self._bucket_plan_items()]
        )
        missing_groups = [g for g in required_groups if not self.catalog.get(g)]
        if missing_groups:
            raise RuntimeError(f"집중 검증 필수 그룹에 유효 피처가 없습니다: {missing_groups}")
        available_buckets = set(self.baskets["bucket"].astype(str))
        missing_buckets = sorted(
            {x["bucket"] for x in self._bucket_plan_items()} - available_buckets
        )
        if missing_buckets:
            raise RuntimeError(f"집중 검증 대상 업종이 basket에 없습니다: {missing_buckets}")

        candidates = sorted(set(sum(self.catalog.values(), [])))
        valid, quality = _feature_quality(df, candidates)
        quality["group"] = quality["feature"].map(lambda c: next((g for g, cols in self.catalog.items() if c in cols), None))
        atomic_csv(quality, self.result_dir / "valid_feature_audit.csv")
        self.features = valid
        if not self.features:
            raise RuntimeError("유효 피처가 없습니다.")
        self.df = df.reset_index(drop=True)
        self.X = self.df[self.features].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        self.y = pd.to_numeric(self.df["label_abs_crash_20"], errors="raise").to_numpy(dtype=np.int8)
        self.dates = self.df["date"].to_numpy()
        self.tickers = self.df["ticker"].astype(str).to_numpy()
        self.buckets = self.df["bucket"].astype(str).to_numpy()
        self.feature_index = {c: i for i, c in enumerate(self.features)}
        self.outer_folds = make_walk_forward_folds(
            self.df["date"], self.outer_folds_count, self.validation_days, self.purge_days, 500
        )
        self.outer_indices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for fold in self.outer_folds:
            fid = int(fold["fold_id"])
            train_idx = np.flatnonzero(np.isin(self.dates, pd.DatetimeIndex(fold["train_dates_index"]).to_numpy()))
            val_idx = np.flatnonzero(np.isin(self.dates, pd.DatetimeIndex(fold["validation_dates_index"]).to_numpy()))
            self.outer_indices[fid] = (train_idx, val_idx)
        self.dataset_signature = stable_hash({
            "metadata": _hash_file_metadata(self.dataset),
            "rows": len(self.df),
            "features": self.features,
            "date_min": str(self.df["date"].min()),
            "date_max": str(self.df["date"].max()),
            "schema": SCHEMA_VERSION,
        })
        atomic_json([f["metadata"] for f in self.outer_folds], self.result_dir / "outer_walk_forward_folds.json")
        atomic_json({
            "schema_version": SCHEMA_VERSION,
            "dataset": str(self.dataset),
            "dataset_signature": self.dataset_signature,
            "rows": len(self.df),
            "tickers": int(self.df["ticker"].nunique()),
            "features": len(self.features),
            "outer_folds": len(self.outer_folds),
            "outer_seeds": self.seeds,
            "experiments": len(self.experiments()),
            "cache_contract": "profile, threads, CPU affinity and CPU/GPU backend are intentionally excluded from prediction cache keys",
        }, self.result_dir / "data_summary.json")

    def experiments(self) -> list[Experiment]:
        experiments = [Experiment("baseline", "baseline", "none")]
        experiments.extend(
            Experiment(f"global__{item['group']}", "focused_global", "global_drop", item["group"])
            for item in self._global_plan_items()
        )
        experiments.extend(
            Experiment(
                f"bucket__{item['bucket']}__{item['group']}",
                "focused_bucket", "bucket_mask", item["group"], target_bucket=item["bucket"]
            )
            for item in self._bucket_plan_items()
        )
        return experiments

    def core_experiments(self) -> list[Experiment]:
        """원래 요청된 10개 집중 실험을 확장 실험보다 먼저 완주한다."""
        core_names = {"baseline"}
        core_names.update(
            f"global__{item['group']}" for item in self.plan["global_groups"]
        )
        core_names.update(
            f"bucket__{item['bucket']}__{item['group']}"
            for item in self.plan["bucket_combinations"]
        )
        return [exp for exp in self.experiments() if exp.name in core_names]

    def execution_phases(self) -> list[tuple[str, list[Experiment], list[int]]]:
        all_experiments = self.experiments()
        configured_core_seeds = [
            int(seed) for seed in self.plan.get("outer_seeds", [17, 43, 101])
        ]
        core_seeds = [seed for seed in self.seeds if seed in configured_core_seeds]
        if not core_seeds:
            core_seeds = self.seeds[: min(3, len(self.seeds))]
        additional_seeds = [seed for seed in self.seeds if seed not in core_seeds]
        phases: list[tuple[str, list[Experiment], list[int]]] = [
            ("core_confirmation", self.core_experiments(), core_seeds),
            ("extension_confirmation", all_experiments, core_seeds),
        ]
        if additional_seeds:
            phases.append(("seed_stability_extension", all_experiments, additional_seeds))
        return phases

    def _matrix(self, exp: Experiment, train_idx: np.ndarray, val_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        group_features = self.catalog.get(exp.group or "", [])
        if exp.mode == "global_drop":
            drop = set(group_features)
            names = [c for c in self.features if c not in drop]
            cols = np.asarray([self.feature_index[c] for c in names], dtype=np.int32)
            return np.ascontiguousarray(self.X[train_idx][:, cols]), np.ascontiguousarray(self.X[val_idx][:, cols]), names
        names = self.features
        x_train = np.ascontiguousarray(self.X[train_idx])
        x_val = np.ascontiguousarray(self.X[val_idx])
        if exp.mode == "bucket_mask":
            cols = np.asarray([self.feature_index[c] for c in group_features if c in self.feature_index], dtype=np.int32)
            train_rows = np.flatnonzero(self.buckets[train_idx] == exp.target_bucket)
            val_rows = np.flatnonzero(self.buckets[val_idx] == exp.target_bucket)
            if len(cols):
                x_train[np.ix_(train_rows, cols)] = np.nan
                x_val[np.ix_(val_rows, cols)] = np.nan
        return x_train, x_val, names

    def _outer_split_context(self, outer_fold_id: int) -> dict[str, Any]:
        outer_train_idx, _ = self.outer_indices[outer_fold_id]
        train_dates = pd.DatetimeIndex(np.unique(self.dates[outer_train_idx])).sort_values()
        required_tail = self.calibration_purge_days + self.calibration_days
        if len(train_dates) < required_tail + 420:
            raise RuntimeError(f"outer fold {outer_fold_id}: nested tuning용 거래일이 부족합니다.")
        calibration_dates = train_dates[-self.calibration_days:]
        tuning_dates = train_dates[:-required_tail]
        calibration_train_dates = train_dates[:-required_tail]
        min_train = max(220, min(420, int(len(tuning_dates) * 0.45)))
        inner = make_walk_forward_folds(
            pd.Series(tuning_dates), self.inner_folds_count, self.inner_validation_days,
            self.purge_days, min_train_days=min_train, min_train_fraction=0.40,
        )
        inner_indices = []
        for fold in inner:
            train_idx = np.flatnonzero(np.isin(self.dates, pd.DatetimeIndex(fold["train_dates_index"]).to_numpy()))
            val_idx = np.flatnonzero(np.isin(self.dates, pd.DatetimeIndex(fold["validation_dates_index"]).to_numpy()))
            inner_indices.append((train_idx, val_idx, fold["metadata"]))
        calibration_train_idx = np.flatnonzero(np.isin(self.dates, calibration_train_dates.to_numpy()))
        calibration_idx = np.flatnonzero(np.isin(self.dates, calibration_dates.to_numpy()))
        return {
            "inner_indices": inner_indices,
            "calibration_train_idx": calibration_train_idx,
            "calibration_idx": calibration_idx,
            "calibration_dates": calibration_dates,
        }

    def _tuning_cache_path(self, outer_fold: int, inner_fold: int, config: ModelConfig, names: list[str]) -> Path:
        key = stable_hash({
            "schema": SCHEMA_VERSION, "kind": "inner_tune", "dataset": self.dataset_signature,
            "outer_fold": outer_fold, "inner_fold": inner_fold, "seed": TUNE_SEED,
            "config": config.payload(), "features": names,
        })
        return self.cache_dir / "inner_tuning" / f"{key}.parquet"

    def select_outer_config(self, outer_fold_id: int, context: dict[str, Any]) -> ModelConfig:
        path = self.policy_dir / f"outer_{outer_fold_id:02d}_model_policy.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("dataset_signature") == self.dataset_signature and payload.get("schema_version") == SCHEMA_VERSION:
                return _model_config_from_payload(payload["selected_config"])
        rows = []
        baseline = Experiment("nested_tune_baseline", "nested_tuning", "none")
        for config in CANDIDATES:
            for inner_fold_id, (train_idx, val_idx, metadata) in enumerate(context["inner_indices"]):
                if self.stop_requested():
                    raise InterruptedError("safe stop requested during nested tuning")
                x_train, x_val, names = self._matrix(baseline, train_idx, val_idx)
                cache = self._tuning_cache_path(outer_fold_id, inner_fold_id, config, names)
                cache.parent.mkdir(parents=True, exist_ok=True)
                started = time.perf_counter()
                if cache.exists():
                    pred = pd.read_parquet(cache)
                    backend = "cache"
                else:
                    model, backend = fit_xgb(x_train, self.y[train_idx], TUNE_SEED, config, threads=self.threads, prefer_gpu=self.prefer_gpu)
                    raw = model.predict_proba(x_val)[:, 1].astype(np.float32)
                    pred = pd.DataFrame({"date": self.dates[val_idx], "target": self.y[val_idx], "raw_prediction": raw})
                    atomic_parquet(pred, cache)
                y_val = pred["target"].to_numpy(dtype=np.int8)
                p_val = pred["raw_prediction"].to_numpy(dtype=float)
                unique = np.unique(y_val)
                inner_metrics = _metrics(y_val, p_val, pred["date"].to_numpy())
                rows.append({
                    "outer_fold": outer_fold_id, "inner_fold": inner_fold_id, "config": config.name,
                    "backend": backend, "elapsed_seconds": time.perf_counter() - started,
                    "pr_auc": inner_metrics["pr_auc"] if len(unique) == 2 else np.nan,
                    "roc_auc": inner_metrics["roc_auc"] if len(unique) == 2 else np.nan,
                    "balanced_accuracy": inner_metrics["balanced_accuracy"] if len(unique) == 2 else np.nan,
                    "brier": float(brier_score_loss(y_val, p_val)),
                    **{f"inner_{k}": v for k, v in metadata.items()},
                })
                self._state(outer_fold=outer_fold_id, inner_fold=inner_fold_id, config=config.name)
        table = pd.DataFrame(rows)
        existing = pd.read_csv(self.result_dir / "nested_tuning_by_fold.csv") if (self.result_dir / "nested_tuning_by_fold.csv").exists() else pd.DataFrame()
        combined = pd.concat([existing.loc[existing.get("outer_fold", pd.Series(dtype=int)).ne(outer_fold_id)] if not existing.empty else existing, table], ignore_index=True)
        atomic_csv(combined, self.result_dir / "nested_tuning_by_fold.csv")
        summary = table.groupby("config", as_index=False).agg(
            pr_auc=("pr_auc", "mean"), roc_auc=("roc_auc", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"), brier=("brier", "mean"),
            elapsed_seconds=("elapsed_seconds", "sum"), valid_folds=("pr_auc", "count"),
        )
        summary["score"] = summary["pr_auc"] + 0.35 * summary["roc_auc"] + 0.15 * summary["balanced_accuracy"] - 0.15 * summary["brier"]
        summary = summary.sort_values(["score", "elapsed_seconds"], ascending=[False, True])
        selected_name = str(summary.iloc[0]["config"])
        selected = next(c for c in CANDIDATES if c.name == selected_name)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "dataset_signature": self.dataset_signature,
            "outer_fold": outer_fold_id,
            "selected_config": selected.payload(),
            "candidate_summary": summary.to_dict(orient="records"),
            "created_by_profile": self.profile,
        }
        atomic_json(payload, path)
        policy_rows = pd.read_csv(self.policy_manifest_path) if self.policy_manifest_path.exists() else pd.DataFrame()
        new = pd.DataFrame([{
            "outer_fold": outer_fold_id, "selected_config": selected.name, "profile": self.profile,
            "dataset_signature": self.dataset_signature, "policy_path": str(path),
        }])
        policy_rows = (new if policy_rows.empty else pd.concat([policy_rows, new], ignore_index=True))
        policy_rows = policy_rows.drop_duplicates(["outer_fold", "dataset_signature"], keep="last")
        atomic_csv(policy_rows, self.policy_manifest_path)
        return selected

    def _calibration_cache_path(self, exp: Experiment, outer_fold: int, config: ModelConfig, names: list[str]) -> Path:
        key = stable_hash({
            "schema": SCHEMA_VERSION, "kind": "calibration_raw", "dataset": self.dataset_signature,
            "experiment": asdict(exp), "outer_fold": outer_fold, "seed": CALIBRATION_SEED,
            "config": config.payload(), "features": names,
        })
        return self.cache_dir / "calibration" / f"{key}.parquet"

    def calibration_policy(
        self, exp: Experiment, outer_fold: int, config: ModelConfig, context: dict[str, Any],
        *, forced_method: str | None = None, forced_threshold: float | None = None,
    ) -> CalibrationPolicy:
        exp_key = stable_hash(asdict(exp))
        path = self.policy_dir / "calibration" / f"outer_{outer_fold:02d}_{exp_key}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                payload.get("dataset_signature") == self.dataset_signature
                and payload.get("config") == config.payload()
                and payload.get("forced_method") == forced_method
                and payload.get("forced_threshold") == forced_threshold
            ):
                return CalibrationPolicy(**payload["policy"])
        train_idx = context["calibration_train_idx"]
        cal_idx = context["calibration_idx"]
        x_train, x_cal, names = self._matrix(exp, train_idx, cal_idx)
        cache = self._calibration_cache_path(exp, outer_fold, config, names)
        cache.parent.mkdir(parents=True, exist_ok=True)
        if cache.exists():
            pred = pd.read_parquet(cache)
            backend = "cache"
        else:
            model, backend = fit_xgb(x_train, self.y[train_idx], CALIBRATION_SEED, config, threads=self.threads, prefer_gpu=self.prefer_gpu)
            raw = model.predict_proba(x_cal)[:, 1].astype(np.float32)
            pred = self.df.loc[cal_idx, ["date", "ticker", "bucket"]].copy()
            pred["target"] = self.y[cal_idx]
            pred["raw_prediction"] = raw
            atomic_parquet(pred, cache)
        policy = select_train_only_policy(
            pred["target"].to_numpy(dtype=np.int8),
            pred["raw_prediction"].to_numpy(dtype=float),
            pred["date"].to_numpy(),
            methods=(forced_method,) if forced_method else ("none", "sigmoid", "beta", "isotonic"),
        )
        if forced_threshold is not None:
            policy = CalibrationPolicy(
                method=policy.method, params=policy.params, threshold=float(forced_threshold),
                selection_score=policy.selection_score, validation_brier=policy.validation_brier,
                validation_logloss=policy.validation_logloss, validation_rows=policy.validation_rows,
                validation_positives=policy.validation_positives,
            )
        payload = {
            "schema_version": SCHEMA_VERSION, "dataset_signature": self.dataset_signature,
            "outer_fold": outer_fold, "experiment": asdict(exp), "config": config.payload(),
            "policy": policy.payload(), "raw_prediction_path": str(cache), "created_by_profile": self.profile,
            "backend": backend, "forced_method": forced_method, "forced_threshold": forced_threshold,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(payload, path)
        existing = pd.read_csv(self.calibration_manifest_path) if self.calibration_manifest_path.exists() else pd.DataFrame()
        new = pd.DataFrame([{
            "outer_fold": outer_fold, "experiment": exp.name, "method": policy.method,
            "threshold": policy.threshold, "selection_score": policy.selection_score,
            "validation_brier": policy.validation_brier, "validation_logloss": policy.validation_logloss,
            "profile": self.profile, "backend": backend, "forced_method": forced_method,
            "forced_threshold": forced_threshold, "policy_path": str(path),
        }])
        existing = (new if existing.empty else pd.concat([existing, new], ignore_index=True))
        existing = existing.drop_duplicates(["outer_fold", "experiment"], keep="last")
        atomic_csv(existing, self.calibration_manifest_path)
        return policy

    def _outer_cache_path(self, exp: Experiment, outer_fold: int, seed: int, config: ModelConfig, policy: CalibrationPolicy, names: list[str]) -> Path:
        # profile, threads, CPU affinity and backend are deliberately excluded.
        key = stable_hash({
            "schema": SCHEMA_VERSION, "kind": "outer_prediction", "dataset": self.dataset_signature,
            "experiment": asdict(exp), "outer_fold": outer_fold, "seed": seed,
            "config": config.payload(), "calibration": policy.payload(), "features": names,
        })
        return self.cache_dir / "outer" / f"{key}.parquet"

    def _scope_metrics(self, pred: pd.DataFrame, exp: Experiment, outer_fold: int, seed: int, policy: CalibrationPolicy, task_id: str) -> pd.DataFrame:
        scopes = [("all", "all_validation", pred)]
        if exp.name == "baseline":
            for bucket in sorted({x["bucket"] for x in self._bucket_plan_items()}):
                scopes.append(("bucket", bucket, pred.loc[pred["bucket"].eq(bucket)]))
        if exp.target_bucket:
            scopes.append(("target_bucket", exp.target_bucket, pred.loc[pred["bucket"].eq(exp.target_bucket)]))
        rows = []
        for scope_type, scope_value, block in scopes:
            if block.empty:
                continue
            rows.append({
                "task_id": task_id, "dataset_signature": self.dataset_signature,
                "experiment": exp.name, "stage": exp.stage, "mode": exp.mode,
                "group": exp.group, "target_bucket": exp.target_bucket, "outer_fold": outer_fold,
                "seed": seed, "scope_type": scope_type, "scope_value": scope_value,
                "calibration_method": policy.method, "decision_threshold": policy.threshold,
                **_metric_bundle(
                    block["target"].to_numpy(dtype=np.int8),
                    block["raw_prediction"].to_numpy(dtype=float),
                    block["prediction"].to_numpy(dtype=float),
                    block["date"].to_numpy(), policy.threshold,
                ),
            })
        return pd.DataFrame(rows)

    def execute_outer(self, exp: Experiment, outer_fold: int, seed: int, config: ModelConfig, policy: CalibrationPolicy, context: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
        # calibration 구간의 라벨은 모델 학습에 넣지 않는다.
        # tuning pool -> calibration window -> outer purge -> untouched outer validation 순서다.
        train_idx = context["calibration_train_idx"]
        _, val_idx = self.outer_indices[outer_fold]
        x_train, x_val, names = self._matrix(exp, train_idx, val_idx)
        cache = self._outer_cache_path(exp, outer_fold, seed, config, policy, names)
        cache.parent.mkdir(parents=True, exist_ok=True)
        task_id = stable_hash({"path": str(cache), "dataset": self.dataset_signature})
        started = time.perf_counter()
        if cache.exists():
            pred = pd.read_parquet(cache)
            backend, cache_status = "cache", "hit"
        else:
            model, backend = fit_xgb(x_train, self.y[train_idx], seed, config, threads=self.threads, prefer_gpu=self.prefer_gpu)
            raw = model.predict_proba(x_val)[:, 1].astype(np.float32)
            calibrated = apply_calibrator(policy.method, policy.params, raw)
            pred = self.df.loc[val_idx, ["date", "ticker", "bucket"]].copy()
            pred["target"] = self.y[val_idx]
            pred["raw_prediction"] = raw
            pred["prediction"] = calibrated
            atomic_parquet(pred, cache)
            cache_status = "miss"
        metrics = self._scope_metrics(pred, exp, outer_fold, seed, policy, task_id)
        record = {
            "task_id": task_id, "dataset_signature": self.dataset_signature,
            "experiment": exp.name, "stage": exp.stage, "mode": exp.mode,
            "group": exp.group, "target_bucket": exp.target_bucket, "outer_fold": outer_fold,
            "seed": seed, "config": config.name, "calibration_method": policy.method,
            "run_id": self.run_id, "profile": self.profile, "backend": backend, "threads": self.threads,
            "cache_status": cache_status, "feature_count": len(names),
            "elapsed_seconds": time.perf_counter() - started, "prediction_path": str(cache), "status": "completed",
        }
        return metrics, record

    def _load_progress(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        metrics = pd.read_csv(self.metrics_path) if self.metrics_path.exists() else pd.DataFrame()
        manifest = pd.read_csv(self.manifest_path) if self.manifest_path.exists() else pd.DataFrame()
        return metrics, manifest

    def _save_progress(self, metrics: pd.DataFrame, manifest: pd.DataFrame) -> None:
        if not metrics.empty:
            metrics = metrics.drop_duplicates(["task_id", "scope_type", "scope_value"], keep="last")
            atomic_csv(metrics, self.metrics_path)
        if not manifest.empty:
            manifest = manifest.drop_duplicates(["task_id"], keep="last")
            atomic_csv(manifest, self.manifest_path)

    def _write_extension_diagnostics(
        self,
        compared: pd.DataFrame,
        summary: pd.DataFrame,
    ) -> None:
        """추가 model fit 없이 복합 그룹 효용과 seed 수렴성을 진단한다."""
        primary_summary = summary.loc[
            (
                summary["stage"].eq("focused_global")
                & summary["scope_type"].eq("all")
            )
            | (
                summary["stage"].eq("focused_bucket")
                & summary["scope_type"].eq("target_bucket")
            )
        ].copy()

        composite_specs: list[dict[str, Any]] = []
        for item in self._global_plan_items():
            components = item.get("components") or []
            if components:
                composite_specs.append({
                    "experiment": f"global__{item['group']}",
                    "components": [f"global__{group}" for group in components],
                    "scope_type": "all",
                    "scope_value": "all_validation",
                })
        for item in self._bucket_plan_items():
            components = item.get("components") or []
            if components:
                bucket = item["bucket"]
                composite_specs.append({
                    "experiment": f"bucket__{bucket}__{item['group']}",
                    "components": [
                        f"bucket__{bucket}__{group}" for group in components
                    ],
                    "scope_type": "target_bucket",
                    "scope_value": bucket,
                })

        synergy_rows: list[dict[str, Any]] = []
        for spec in composite_specs:
            joint = primary_summary.loc[
                primary_summary["experiment"].eq(spec["experiment"])
                & primary_summary["scope_type"].eq(spec["scope_type"])
                & primary_summary["scope_value"].eq(spec["scope_value"])
            ]
            components = primary_summary.loc[
                primary_summary["experiment"].isin(spec["components"])
                & primary_summary["scope_type"].eq(spec["scope_type"])
                & primary_summary["scope_value"].eq(spec["scope_value"])
            ]
            if joint.empty or len(components) != len(spec["components"]):
                continue
            joint_loss = float(joint.iloc[0]["pr_auc_loss_mean"])
            component_losses = pd.to_numeric(
                components["pr_auc_loss_mean"], errors="coerce"
            ).dropna()
            if component_losses.empty:
                continue
            synergy_rows.append({
                "experiment": spec["experiment"],
                "scope_type": spec["scope_type"],
                "scope_value": spec["scope_value"],
                "components": "|".join(spec["components"]),
                "joint_pr_auc_loss_mean": joint_loss,
                "best_component_pr_auc_loss_mean": float(component_losses.max()),
                "sum_component_pr_auc_loss_mean": float(component_losses.sum()),
                "incremental_over_best_component": (
                    joint_loss - float(component_losses.max())
                ),
                "synergy_vs_additive_sum": (
                    joint_loss - float(component_losses.sum())
                ),
                "joint_positive_fold_ratio": float(
                    joint.iloc[0]["pr_auc_loss_positive_fold_ratio"]
                ),
                "joint_valid_fold_count": int(
                    joint.iloc[0]["valid_pr_auc_loss_fold_count"]
                ),
                "joint_seed_count": int(joint.iloc[0]["seed_count"]),
            })
        synergy_columns = [
            "experiment", "scope_type", "scope_value", "components",
            "joint_pr_auc_loss_mean", "best_component_pr_auc_loss_mean",
            "sum_component_pr_auc_loss_mean", "incremental_over_best_component",
            "synergy_vs_additive_sum", "joint_positive_fold_ratio",
            "joint_valid_fold_count", "joint_seed_count",
        ]
        atomic_csv(
            pd.DataFrame(synergy_rows, columns=synergy_columns),
            self.result_dir / "composite_group_synergy.csv",
        )

        primary_compared = compared.loc[
            (
                compared["stage"].eq("focused_global")
                & compared["scope_type"].eq("all")
            )
            | (
                compared["stage"].eq("focused_bucket")
                & compared["scope_type"].eq("target_bucket")
            )
        ].copy()
        core_seeds = {
            int(seed) for seed in self.plan.get("outer_seeds", [17, 43, 101])
        }
        convergence_rows: list[dict[str, Any]] = []
        for experiment, block in primary_compared.groupby("experiment", sort=False):
            core = block.loc[block["seed"].isin(core_seeds)]
            core_fold = core.groupby("outer_fold")["pr_auc_loss"].mean().dropna()
            all_fold = block.groupby("outer_fold")["pr_auc_loss"].mean().dropna()
            seeds_per_fold = block.groupby("outer_fold")["seed"].nunique()
            if core_fold.empty or all_fold.empty:
                continue
            convergence_rows.append({
                "experiment": experiment,
                "scope_type": block.iloc[0]["scope_type"],
                "scope_value": block.iloc[0]["scope_value"],
                "core_seed_count": int(core["seed"].nunique()),
                "all_seed_count": int(block["seed"].nunique()),
                "core_fold_count": int(len(core_fold)),
                "all_fold_count": int(len(all_fold)),
                "minimum_seeds_per_fold": int(seeds_per_fold.min()),
                "maximum_seeds_per_fold": int(seeds_per_fold.max()),
                "core_pr_auc_loss_mean": float(core_fold.mean()),
                "all_seed_pr_auc_loss_mean": float(all_fold.mean()),
                "mean_shift_after_extra_seeds": float(
                    all_fold.mean() - core_fold.mean()
                ),
                "core_positive_fold_ratio": float((core_fold > 0).mean()),
                "all_seed_positive_fold_ratio": float((all_fold > 0).mean()),
                "positive_fold_ratio_shift": float(
                    (all_fold > 0).mean() - (core_fold > 0).mean()
                ),
                "mean_within_fold_seed_std": float(
                    block.groupby("outer_fold")["pr_auc_loss"].std(ddof=1).mean()
                ),
            })
        convergence_columns = [
            "experiment", "scope_type", "scope_value", "core_seed_count",
            "all_seed_count", "core_fold_count", "all_fold_count",
            "minimum_seeds_per_fold", "maximum_seeds_per_fold",
            "core_pr_auc_loss_mean", "all_seed_pr_auc_loss_mean",
            "mean_shift_after_extra_seeds", "core_positive_fold_ratio",
            "all_seed_positive_fold_ratio", "positive_fold_ratio_shift",
            "mean_within_fold_seed_std",
        ]
        atomic_csv(
            pd.DataFrame(convergence_rows, columns=convergence_columns),
            self.result_dir / "seed_stability_convergence.csv",
        )

    def summarize(self, metrics: pd.DataFrame) -> dict[str, Any]:
        metrics = metrics.drop_duplicates(["task_id", "scope_type", "scope_value"], keep="last").copy()
        atomic_csv(metrics, self.result_dir / "all_metrics.csv")
        baseline = metrics.loc[metrics["experiment"].eq("baseline")].copy()
        metric_cols = [
            "pr_auc", "raw_pr_auc", "roc_auc", "raw_roc_auc", "balanced_accuracy", "accuracy",
            "brier", "raw_brier", "logloss", "raw_logloss", "top_3pct_precision", "raw_top_3pct_precision",
        ]
        base = baseline.copy()
        base["pair_scope_type"] = base["scope_type"].replace({"target_bucket": "bucket"})
        key = ["outer_fold", "seed", "pair_scope_type", "scope_value"]
        base = base.assign(pair_scope_type=base["pair_scope_type"])[key + metric_cols].drop_duplicates(key)
        base = base.rename(columns={c: f"baseline_{c}" for c in metric_cols})
        compared = metrics.loc[~metrics["experiment"].eq("baseline")].copy()
        compared["pair_scope_type"] = compared["scope_type"].replace({"target_bucket": "bucket"})
        compared = compared.merge(base, on=key, how="left", validate="many_to_one")
        if compared["baseline_pr_auc"].isna().any():
            sample = compared.loc[compared["baseline_pr_auc"].isna(), ["experiment", "outer_fold", "seed", "scope_type", "scope_value"]].head()
            raise RuntimeError(f"baseline pairing 실패:\n{sample.to_string(index=False)}")
        compared["pr_auc_loss"] = compared["baseline_pr_auc"] - compared["pr_auc"]
        compared["raw_pr_auc_loss"] = compared["baseline_raw_pr_auc"] - compared["raw_pr_auc"]
        compared["roc_auc_loss"] = compared["baseline_roc_auc"] - compared["roc_auc"]
        compared["balanced_accuracy_loss"] = compared["baseline_balanced_accuracy"] - compared["balanced_accuracy"]
        compared["brier_increase"] = compared["brier"] - compared["baseline_brier"]
        compared["logloss_increase"] = compared["logloss"] - compared["baseline_logloss"]
        compared["top_3pct_precision_loss"] = compared["baseline_top_3pct_precision"] - compared["top_3pct_precision"]
        atomic_csv(compared, self.result_dir / "paired_ablation_deltas.csv")

        keys = ["experiment", "stage", "mode", "group", "target_bucket", "scope_type", "scope_value"]
        rows = []
        delta_cols = ["pr_auc_loss", "raw_pr_auc_loss", "roc_auc_loss", "balanced_accuracy_loss", "brier_increase", "logloss_increase", "top_3pct_precision_loss"]
        for key_values, block in compared.groupby(keys, dropna=False, sort=False):
            row = dict(zip(keys, key_values if isinstance(key_values, tuple) else (key_values,)))
            row["total_outer_fold_count"] = int(block["outer_fold"].nunique())
            row["seed_count"] = int(block["seed"].nunique())
            row["total_rows"] = int(block["rows"].sum()) if "rows" in block else 0
            row["total_positives"] = int(block["positives"].sum()) if "positives" in block else 0
            for col in delta_cols:
                fold_values = block.groupby("outer_fold")[col].mean().dropna().to_numpy(dtype=float)
                row[f"valid_{col}_fold_count"] = int(len(fold_values))
                row[f"{col}_mean"] = float(np.mean(fold_values)) if len(fold_values) else np.nan
                row[f"{col}_median"] = float(np.median(fold_values)) if len(fold_values) else np.nan
                row[f"{col}_std"] = float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else np.nan
                row[f"{col}_positive_fold_ratio"] = float(np.mean(fold_values > 0)) if len(fold_values) else np.nan
                row[f"{col}_sign_p"] = _exact_sign_p(fold_values)
            rows.append(row)
        summary = pd.DataFrame(rows)
        if summary.empty:
            summary = pd.DataFrame(columns=keys + [
                "total_outer_fold_count", "seed_count", "pr_auc_loss_mean",
                "pr_auc_loss_sign_p", "pr_auc_loss_sign_q", "test_family",
            ])
        else:
            summary["test_family"] = np.select(
                [
                    summary["stage"].eq("focused_global"),
                    summary["scope_type"].eq("target_bucket"),
                ],
                ["global_all", "bucket_target"],
                default="bucket_all_validation_secondary",
            )
            summary["pr_auc_loss_sign_q"] = summary.groupby("test_family", group_keys=False)["pr_auc_loss_sign_p"].apply(_bh_qvalues)
            summary["raw_pr_auc_loss_sign_q"] = summary.groupby("test_family", group_keys=False)["raw_pr_auc_loss_sign_p"].apply(_bh_qvalues)
        atomic_csv(summary, self.result_dir / "focused_ablation_summary.csv")
        self._write_extension_diagnostics(compared, summary)

        baseline_all = baseline.loc[baseline["scope_type"].eq("all")]
        means = baseline_all.select_dtypes(include=[np.number]).mean().to_dict()
        goal = {
            "mean_accuracy": means.get("accuracy"),
            "mean_balanced_accuracy": means.get("balanced_accuracy"),
            "mean_roc_auc": means.get("roc_auc"),
            "mean_pr_auc": means.get("pr_auc"),
            "mean_pr_auc_lift": means.get("pr_auc_lift"),
            "mean_top_3pct_precision": means.get("top_3pct_precision"),
            "mean_brier_skill": means.get("brier_skill"),
            "mean_logloss_skill": means.get("logloss_skill"),
        }
        targets = self.plan["success_targets"]
        goal["roc_auc_goal_met"] = bool((goal["mean_roc_auc"] or 0) >= targets["roc_auc"])
        goal["balanced_accuracy_goal_met"] = bool((goal["mean_balanced_accuracy"] or 0) >= targets["balanced_accuracy"])
        goal["pr_auc_lift_goal_met"] = bool((goal["mean_pr_auc_lift"] or 0) >= targets["pr_auc_lift"])
        goal["top_3pct_precision_goal_met"] = bool((goal["mean_top_3pct_precision"] or 0) >= targets["top_3pct_precision"])
        goal["robust_70_goal_met"] = bool(goal["roc_auc_goal_met"] and goal["balanced_accuracy_goal_met"])
        atomic_json(goal, self.result_dir / "goal_report.json")
        return {"goal": goal, "summary_rows": len(summary), "paired_rows": len(compared)}

    def run(self) -> dict[str, Any]:
        prevent_windows_sleep()
        self.monitor.start()
        metrics = pd.DataFrame()
        manifest = pd.DataFrame()
        completed: set[str] = set()
        safely_stopped = False
        try:
            with RunLock(self.lock_path, self.profile):
                self.current_stage = "load"
                self._state()
                self.load()
                metrics, manifest = self._load_progress()
                if not metrics.empty:
                    if "dataset_signature" not in metrics.columns:
                        metrics = pd.DataFrame()
                    else:
                        metrics = metrics.loc[metrics["dataset_signature"].eq(self.dataset_signature)].copy()
                if not manifest.empty:
                    if "dataset_signature" not in manifest.columns:
                        manifest = pd.DataFrame()
                    else:
                        manifest = manifest.loc[manifest["dataset_signature"].eq(self.dataset_signature)].copy()
                completed = set(
                    manifest.loc[manifest["status"].eq("completed"), "task_id"].astype(str)
                ) if not manifest.empty else set()
                total_planned = len(self.outer_folds) * len(self.experiments()) * len(self.seeds)
                for phase_name, phase_experiments, phase_seeds in self.execution_phases():
                    if safely_stopped:
                        break
                    for outer_fold in range(len(self.outer_folds)):
                        if self.stop_requested():
                            safely_stopped = True
                            break
                        self.current_stage = "nested_tuning"
                        self._state(
                            execution_phase=phase_name,
                            outer_fold=outer_fold,
                            phase_seed_count=len(phase_seeds),
                            phase_experiment_count=len(phase_experiments),
                            total_planned_outer_tasks=total_planned,
                        )
                        context = self._outer_split_context(outer_fold)
                        try:
                            config = self.select_outer_config(outer_fold, context)
                        except InterruptedError:
                            safely_stopped = True
                            break
                        baseline_policy: CalibrationPolicy | None = None
                        for exp in phase_experiments:
                            if self.stop_requested():
                                safely_stopped = True
                                break
                            self.current_stage = "train_only_calibration"
                            self._state(
                                execution_phase=phase_name,
                                outer_fold=outer_fold,
                                experiment=exp.name,
                                selected_config=config.name,
                            )
                            policy = self.calibration_policy(
                                exp, outer_fold, config, context,
                                forced_method=baseline_policy.method if baseline_policy is not None else None,
                                forced_threshold=baseline_policy.threshold if baseline_policy is not None else None,
                            )
                            if exp.name == "baseline":
                                baseline_policy = policy
                            for seed in phase_seeds:
                                if self.stop_requested():
                                    safely_stopped = True
                                    break
                                # Calculate the same cache-derived task id before executing so completed tasks can be skipped.
                                train_idx = context["calibration_train_idx"]
                                _, val_idx = self.outer_indices[outer_fold]
                                _, _, names = self._matrix(exp, train_idx[:1], val_idx[:1])
                                cache = self._outer_cache_path(exp, outer_fold, seed, config, policy, names)
                                task_id = stable_hash({"path": str(cache), "dataset": self.dataset_signature})
                                if task_id in completed and not metrics.loc[metrics.get("task_id", pd.Series(dtype=str)).astype(str).eq(task_id)].empty:
                                    continue
                                self.current_stage = "outer_evaluation"
                                self._state(
                                    execution_phase=phase_name,
                                    outer_fold=outer_fold,
                                    experiment=exp.name,
                                    seed=seed,
                                    selected_config=config.name,
                                    calibration=policy.method,
                                )
                                task_metrics, record = self.execute_outer(exp, outer_fold, seed, config, policy, context)
                                metrics = pd.concat([metrics, task_metrics], ignore_index=True)
                                manifest = pd.concat([manifest, pd.DataFrame([record])], ignore_index=True)
                                completed.add(record["task_id"])
                                self._save_progress(metrics, manifest)
                            if safely_stopped:
                                break
                self.current_stage = "summarize"
                summary_result = self.summarize(metrics) if not metrics.empty else {"goal": {}, "summary_rows": 0, "paired_rows": 0}
                current_manifest = manifest.drop_duplicates("task_id", keep="last") if not manifest.empty else manifest
                this_run_manifest = (
                    manifest.loc[manifest.get("run_id", pd.Series(dtype=str)).astype(str).eq(self.run_id)].copy()
                    if not manifest.empty and "run_id" in manifest.columns else pd.DataFrame()
                )
                backend_counts = current_manifest["backend"].value_counts(dropna=False).to_dict() if not current_manifest.empty else {}
                profile_counts = current_manifest["profile"].value_counts(dropna=False).to_dict() if not current_manifest.empty else {}
                result = {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "profile": self.profile,
                    "threads": self.threads,
                    "prefer_gpu": self.prefer_gpu,
                    "cache_namespace": self.cache_namespace,
                    "dataset_signature": self.dataset_signature,
                    "elapsed_hours": (time.monotonic() - self.started) / 3600,
                    "safe_stop": safely_stopped,
                    "temperature_or_ram_abort": self.monitor.abort_event.is_set(),
                    "completed_outer_tasks": int(len(completed)),
                    "planned_outer_tasks": total_planned,
                    "progress_ratio": float(len(completed) / total_planned) if total_planned else 0.0,
                    "new_outer_models_this_run": int(this_run_manifest["cache_status"].eq("miss").sum()) if not this_run_manifest.empty else 0,
                    "outer_cache_hits_this_run": int(this_run_manifest["cache_status"].eq("hit").sum()) if not this_run_manifest.empty else 0,
                    "dataset_backend_counts": backend_counts,
                    "dataset_task_creation_profile_counts": profile_counts,
                    "mixed_backend_results": bool(len([k for k in backend_counts if k != "cache"]) > 1 or len(profile_counts) > 1),
                    "result_dir": str(self.result_dir),
                    **summary_result,
                }
                atomic_json(result, self.result_dir / "run_summary.json")
                self._state(**result)
                return result
        finally:
            self.monitor.stop()
            restore_windows_sleep()
