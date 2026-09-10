from __future__ import annotations

import gc
import json
import math
import multiprocessing as mp
import os
import subprocess
import threading
import time
import traceback
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import get_paths
from ..experiment.splits import make_walk_forward_folds
from ..finance_nested.runner import _metric_bundle
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, stable_hash
from .calibration import SafeCalibrationPolicy, apply_calibrator, select_safe_policy
from .data import RefineDataBundle, load_refine_data
from .deep_models import build_window_indices, fit_deep_model, predict_deep_model
from .models import TREE_CONFIGS, fit_tree_model, release_tree
from .reaggregate import reaggregate_base12h_raw
from .registry import RefineTaskRegistry, TaskClaim
from .reporting import (
    compile_ensembles,
    compile_model_summary,
    create_result_package,
    write_bottleneck_report,
    write_result_brief,
)

SCHEMA_VERSION = "refine12h_v1"
TUNE_SEED = 1701
CALIBRATION_SEED = 2718
FINAL_SEED = 20260801


def _trim_process_memory() -> None:
    """Return released dataframe pages to the OS after supervisor preflight."""
    gc.collect()
    if os.name == "nt":
        try:
            import ctypes

            handle = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.EmptyWorkingSet(handle)
        except Exception:
            pass


def _apply_affinity(profile: str, threads: int, worker_index: int, worker_count: int) -> dict[str, Any]:
    import psutil

    process = psutil.Process()
    available = process.cpu_affinity() if hasattr(process, "cpu_affinity") else list(range(psutil.cpu_count() or threads))
    if profile == "pubg":
        # A spawned Windows child inherits the supervisor affinity. Do not run
        # [::2] again on an already-limited four-CPU mask (4 -> 2 -> 1).
        if len(available) <= threads:
            selected = available[:threads]
        else:
            physical_first = available[::2] or available
            selected = physical_first[:threads] or available[:threads]
        if hasattr(process, "cpu_affinity"):
            process.cpu_affinity(selected)
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            process.nice(10)
    else:
        chunks = np.array_split(np.asarray(available, dtype=int), max(1, worker_count))
        selected = chunks[min(worker_index, len(chunks) - 1)].tolist() or available
        if hasattr(process, "cpu_affinity"):
            process.cpu_affinity(selected)
        if os.name == "nt":
            process.nice(psutil.HIGH_PRIORITY_CLASS)
        else:
            try:
                process.nice(-5)
            except Exception:
                pass
    return {"affinity": selected, "threads": threads, "profile": profile}


def _safe_score(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if len(np.unique(y)) < 2:
        return math.nan, math.nan
    return float(average_precision_score(y, p)), float(roc_auc_score(y, p))


def _inner_splits(bundle: RefineDataBundle, fit_idx: np.ndarray, plan: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    dates = pd.DatetimeIndex(np.unique(bundle.dates[fit_idx])).sort_values()
    try:
        folds = make_walk_forward_folds(
            dates,
            n_folds=int(plan["inner_folds"]),
            validation_days=int(plan["inner_validation_days"]),
            purge_days=int(plan["purge_days"]),
            min_train_days=min(350, max(180, len(dates) // 2)),
            min_train_fraction=0.45,
        )
        output = []
        for fold in folds:
            train_idx = np.flatnonzero(np.isin(bundle.dates, pd.DatetimeIndex(fold["train_dates_index"]).to_numpy()))
            valid_idx = np.flatnonzero(np.isin(bundle.dates, pd.DatetimeIndex(fold["validation_dates_index"]).to_numpy()))
            train_idx = np.intersect1d(train_idx, fit_idx, assume_unique=False)
            valid_idx = np.intersect1d(valid_idx, fit_idx, assume_unique=False)
            if len(train_idx) and len(valid_idx):
                output.append((train_idx, valid_idx))
        if output:
            return output
    except Exception:
        pass
    validation_days = min(int(plan["inner_validation_days"]), max(20, len(dates) // 5))
    purge_days = min(int(plan["purge_days"]), max(5, len(dates) // 20))
    valid_dates = dates[-validation_days:]
    train_dates = dates[: -(validation_days + purge_days)]
    return [(
        np.intersect1d(fit_idx, np.flatnonzero(np.isin(bundle.dates, train_dates.to_numpy()))),
        np.intersect1d(fit_idx, np.flatnonzero(np.isin(bundle.dates, valid_dates.to_numpy()))),
    )]


def _policy_path(result_dir: Path, family: str, profile: str, fold: int, backend: str, dataset_signature: str) -> Path:
    key = stable_hash({"schema": SCHEMA_VERSION, "family": family, "profile": profile, "fold": fold, "backend": backend, "dataset": dataset_signature})
    return result_dir / "tuning_policies" / family / profile / f"{key}.json"




def _acquire_policy_lock(path: Path, timeout: float = 900.0) -> bool:
    deadline = time.time() + timeout
    path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                owner_text = path.read_text(encoding="utf-8").strip()
                owner_pid = int(owner_text)
                import psutil

                if not psutil.pid_exists(owner_pid):
                    path.unlink(missing_ok=True)
                    continue
            except (OSError, ValueError):
                pass
            if time.time() > deadline:
                try:
                    if time.time() - path.stat().st_mtime > timeout:
                        path.unlink(missing_ok=True)
                        continue
                except OSError:
                    pass
                return False
            time.sleep(0.5)

def _select_tree_config(
    bundle: RefineDataBundle,
    result_dir: Path,
    plan: dict[str, Any],
    *,
    family: str,
    profile: str,
    fold: int,
    requested_backend: str,
    threads: int,
) -> tuple[dict[str, Any], str, pd.DataFrame]:
    path = _policy_path(result_dir, family, profile, fold, requested_backend, bundle.dataset_signature)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["selected_config"], payload["effective_backend"], pd.DataFrame(payload.get("candidates", []))
    lock_path = path.with_suffix(".lock")
    owner = _acquire_policy_lock(lock_path)
    if not owner and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["selected_config"], payload["effective_backend"], pd.DataFrame(payload.get("candidates", []))
    if path.exists():
        lock_path.unlink(missing_ok=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["selected_config"], payload["effective_backend"], pd.DataFrame(payload.get("candidates", []))
    context = bundle.fold_indices(fold, profile)
    x_profile, _ = bundle.matrix(np.arange(len(bundle.df), dtype=np.int32), profile)
    splits = _inner_splits(bundle, context["fit_idx"], plan)
    candidate_rows = []
    effective_request = requested_backend
    for config in TREE_CONFIGS[family]:
        pr_values: list[float] = []
        roc_values: list[float] = []
        actual_backends: list[str] = []
        for train_idx, valid_idx in splits:
            model = fit_tree_model(
                family, config, x_profile[train_idx], bundle.y[train_idx], seed=TUNE_SEED,
                requested_backend=effective_request, threads=threads,
                eval_set=(x_profile[valid_idx], bundle.y[valid_idx]),
                force_backend="cpu" if effective_request == "cpu" else None,
            )
            raw = model.predict(x_profile[valid_idx])
            pr, roc = _safe_score(bundle.y[valid_idx], raw)
            pr_values.append(pr)
            roc_values.append(roc)
            actual_backends.append(model.actual_backend)
            if "cpu" in model.actual_backend:
                effective_request = "cpu"
            release_tree(model)
        pr_mean = float(np.nanmean(pr_values))
        roc_mean = float(np.nanmean(roc_values))
        pr_std = float(np.nanstd(pr_values, ddof=1)) if len(pr_values) > 1 else 0.0
        score = pr_mean + 0.50 * roc_mean - 0.20 * pr_std
        candidate_rows.append({
            "family": family, "feature_profile": profile, "outer_fold": fold,
            "config_name": config["name"], "score": score, "raw_pr_auc": pr_mean,
            "raw_roc_auc": roc_mean, "raw_pr_auc_std": pr_std,
            "actual_backends": ",".join(sorted(set(actual_backends))),
        })
    candidates = pd.DataFrame(candidate_rows).sort_values(["score", "config_name"], ascending=[False, True])
    selected_name = str(candidates.iloc[0]["config_name"])
    selected = next(config for config in TREE_CONFIGS[family] if config["name"] == selected_name)
    payload = {
        "schema": SCHEMA_VERSION, "dataset_signature": bundle.dataset_signature,
        "family": family, "feature_profile": profile, "outer_fold": fold,
        "requested_backend": requested_backend, "effective_backend": effective_request,
        "selected_config": selected, "candidates": candidate_rows,
    }
    atomic_json(payload, path)
    lock_path.unlink(missing_ok=True)
    return selected, effective_request, candidates


def _prediction_frame(bundle: RefineDataBundle, rows: np.ndarray, raw: np.ndarray, calibrated: np.ndarray, metadata: dict[str, Any]) -> pd.DataFrame:
    frame = bundle.df.loc[rows, ["date", "ticker", "bucket"]].copy()
    frame["target"] = bundle.y[rows]
    frame["raw_prediction"] = raw.astype(np.float32)
    frame["prediction"] = calibrated.astype(np.float32)
    for key, value in metadata.items():
        frame[key] = value
    return frame


class RefineWorker:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.project = Path(payload["project"])
        self.dataset = Path(payload["dataset"]) if payload.get("dataset") else None
        self.result_dir = Path(payload["result_dir"])
        self.profile = str(payload["profile"])
        self.worker_index = int(payload["worker_index"])
        self.worker_count = int(payload["worker_count"])
        self.threads = int(payload["threads"])
        self.requested_backend = str(payload["backend"])
        self.deadline_epoch = float(payload["deadline_epoch"])
        self.stop_flag = Path(payload["stop_flag"])
        self.registry = RefineTaskRegistry(Path(payload["registry_path"]))
        self.plan = json.loads((self.project / "configs" / "refine12h_plan.json").read_text(encoding="utf-8"))
        self.bundle: RefineDataBundle | None = None
        self.windows: np.ndarray | None = None

    def stop_requested(self) -> bool:
        return self.stop_flag.exists() or time.time() >= self.deadline_epoch

    def load(self) -> None:
        _apply_affinity(self.profile, self.threads, self.worker_index, self.worker_count)
        self.bundle = load_refine_data(self.project, self.dataset, self.result_dir, self.plan)

    def _profile_complete(self, task_id: str, profile: str) -> bool:
        return (self.result_dir / "task_records" / f"{task_id}__{profile}.json").exists()

    def _run_tree_profile(self, claim: TaskClaim, profile: str, force_backend: str | None) -> tuple[str, str]:
        assert self.bundle is not None
        bundle = self.bundle
        record_path = self.result_dir / "task_records" / f"{claim.task_id}__{profile}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            return str(record.get("actual_backend", claim.backend)), str(record.get("config_name", "cached"))
        started = time.perf_counter()
        context = bundle.fold_indices(claim.outer_fold, profile)
        x_profile, feature_names = bundle.matrix(np.arange(len(bundle.df), dtype=np.int32), profile)
        requested = "cpu" if force_backend and "cpu" in force_backend else claim.backend
        config, tuning_backend, candidates = _select_tree_config(
            bundle, self.result_dir, self.plan, family=claim.family, profile=profile,
            fold=claim.outer_fold, requested_backend=requested, threads=claim.threads,
        )
        force = "cpu" if tuning_backend == "cpu" else None
        calibration_key = stable_hash({
            "schema": SCHEMA_VERSION, "dataset": bundle.dataset_signature, "family": claim.family,
            "profile": profile, "fold": claim.outer_fold, "backend": tuning_backend,
            "config": config["name"], "calibration_seed": CALIBRATION_SEED,
        })
        calibration_cache = self.result_dir / "shared_calibration" / claim.family / profile / f"{calibration_key}.parquet"
        calibration_policy_path = self.result_dir / "shared_calibration" / claim.family / profile / f"{calibration_key}.json"
        calibration_lock = calibration_policy_path.with_suffix(".lock")
        if calibration_cache.exists() and calibration_policy_path.exists():
            cached_cal = pd.read_parquet(calibration_cache)
            raw_cal = cached_cal["raw_prediction"].to_numpy(dtype=np.float32)
            policy = SafeCalibrationPolicy(**json.loads(calibration_policy_path.read_text(encoding="utf-8"))["policy"])
            actual_backend = str(json.loads(calibration_policy_path.read_text(encoding="utf-8")).get("actual_backend", tuning_backend))
            if "cpu" in actual_backend:
                force = "cpu"
        else:
            owner = _acquire_policy_lock(calibration_lock)
            if not owner and calibration_cache.exists() and calibration_policy_path.exists():
                cached_cal = pd.read_parquet(calibration_cache)
                raw_cal = cached_cal["raw_prediction"].to_numpy(dtype=np.float32)
                cached_policy = json.loads(calibration_policy_path.read_text(encoding="utf-8"))
                policy = SafeCalibrationPolicy(**cached_policy["policy"])
                actual_backend = str(cached_policy.get("actual_backend", tuning_backend))
                if "cpu" in actual_backend:
                    force = "cpu"
            elif calibration_cache.exists() and calibration_policy_path.exists():
                calibration_lock.unlink(missing_ok=True)
                cached_cal = pd.read_parquet(calibration_cache)
                raw_cal = cached_cal["raw_prediction"].to_numpy(dtype=np.float32)
                cached_policy = json.loads(calibration_policy_path.read_text(encoding="utf-8"))
                policy = SafeCalibrationPolicy(**cached_policy["policy"])
                actual_backend = str(cached_policy.get("actual_backend", tuning_backend))
                if "cpu" in actual_backend:
                    force = "cpu"
            else:
                calibration_model = fit_tree_model(
                    claim.family, config, x_profile[context["fit_idx"]], bundle.y[context["fit_idx"]],
                    seed=CALIBRATION_SEED, requested_backend=tuning_backend, threads=claim.threads, force_backend=force,
                )
                actual_backend = calibration_model.actual_backend
                if "cpu" in actual_backend:
                    force = "cpu"
                raw_cal = calibration_model.predict(x_profile[context["calibration_idx"]])
                release_tree(calibration_model)
                regime_cal = bundle.regime_matrix(context["calibration_idx"])
                policy = select_safe_policy(
                    bundle.y[context["calibration_idx"]], raw_cal, bundle.dates[context["calibration_idx"]],
                    regime=regime_cal, regime_names=bundle.regime_features,
                    methods=tuple(self.plan["calibration_methods"]),
                    max_roc_drop=float(self.plan["calibration_max_roc_drop"]),
                    max_pr_drop=float(self.plan["calibration_max_pr_drop"]),
                    min_rank_correlation=float(self.plan["calibration_min_rank_correlation"]),
                )
                cached_cal = bundle.df.loc[context["calibration_idx"], ["date", "ticker", "bucket"]].copy()
                cached_cal["target"] = bundle.y[context["calibration_idx"]]
                cached_cal["raw_prediction"] = raw_cal
                atomic_parquet(cached_cal, calibration_cache)
                atomic_json({"policy": policy.payload(), "actual_backend": actual_backend}, calibration_policy_path)
                calibration_lock.unlink(missing_ok=True)
        regime_cal = bundle.regime_matrix(context["calibration_idx"])
        outer_model = fit_tree_model(
            claim.family, config, x_profile[context["fit_idx"]], bundle.y[context["fit_idx"]],
            seed=claim.seed, requested_backend="cpu" if force == "cpu" else tuning_backend,
            threads=claim.threads, force_backend=force,
        )
        actual_backend = outer_model.actual_backend
        raw_outer = outer_model.predict(x_profile[context["validation_idx"]])
        importance = outer_model.feature_importance()
        release_tree(outer_model)
        regime_outer = bundle.regime_matrix(context["validation_idx"])
        calibrated_outer = apply_calibrator(policy.method, policy.params, raw_outer, regime_outer)
        calibrated_cal = apply_calibrator(policy.method, policy.params, raw_cal, regime_cal)
        metadata = {
            "task_id": claim.task_id, "family": claim.family, "feature_profile": profile,
            "outer_fold": claim.outer_fold, "seed": claim.seed, "actual_backend": actual_backend,
            "config_name": config["name"], "calibration_method": policy.method,
        }
        outer_frame = _prediction_frame(bundle, context["validation_idx"], raw_outer, calibrated_outer, metadata)
        cal_frame = _prediction_frame(bundle, context["calibration_idx"], raw_cal, calibrated_cal, metadata)
        outer_path = self.result_dir / "predictions" / "outer" / claim.family / profile / f"{claim.task_id}.parquet"
        cal_path = self.result_dir / "predictions" / "calibration" / claim.family / profile / f"{claim.task_id}.parquet"
        atomic_parquet(outer_frame, outer_path)
        atomic_parquet(cal_frame, cal_path)
        values = _metric_bundle(
            outer_frame["target"].to_numpy(dtype=np.int8), raw_outer, calibrated_outer,
            outer_frame["date"].to_numpy(), policy.threshold,
        )
        metrics = pd.DataFrame([{
            **metadata, "rows": len(outer_frame), "feature_count": len(feature_names),
            "decision_threshold": policy.threshold, "calibration_fallback_reason": policy.fallback_reason,
            "elapsed_seconds": time.perf_counter() - started, **values,
        }])
        metrics_path = self.result_dir / "model_metrics" / claim.family / profile / f"{claim.task_id}.parquet"
        atomic_parquet(metrics, metrics_path)
        audit_path = self.result_dir / "calibration_audits" / claim.family / profile / f"{claim.task_id}.json"
        atomic_json({
            **metadata, "policy": policy.payload(), "raw_roc_auc": values.get("raw_roc_auc"),
            "calibrated_roc_auc": values.get("roc_auc"), "raw_pr_auc": values.get("raw_pr_auc"),
            "calibrated_pr_auc": values.get("pr_auc"),
        }, audit_path)
        if len(importance) == len(feature_names):
            importance_frame = pd.DataFrame({"feature": feature_names, "importance": importance})
            importance_frame["task_id"] = claim.task_id
            importance_frame["family"] = claim.family
            importance_frame["feature_profile"] = profile
            atomic_parquet(importance_frame, self.result_dir / "feature_importance" / claim.family / profile / f"{claim.task_id}.parquet")
        atomic_json({
            **metadata, "status": "completed", "outer_prediction": str(outer_path),
            "calibration_prediction": str(cal_path), "metrics_path": str(metrics_path),
            "feature_count": len(feature_names), "config_name": config["name"],
            "tuning_candidates": candidates.to_dict(orient="records"),
        }, record_path)
        del x_profile
        gc.collect()
        return actual_backend, config["name"]

    def run_tree_task(self, claim: TaskClaim) -> tuple[str, str]:
        effective_backend: str | None = None
        records = []
        for profile in ("full_reduced", "common_period"):
            backend, config_name = self._run_tree_profile(claim, profile, effective_backend)
            effective_backend = backend
            records.append({"profile": profile, "backend": backend, "config": config_name})
            self.registry.heartbeat(claim.task_id)
            if self.stop_requested():
                raise InterruptedError("safe stop after completed feature profile")
        task_record = self.result_dir / "task_records" / f"{claim.task_id}.json"
        atomic_json({"task_id": claim.task_id, "stage": "tree", "records": records, "status": "completed"}, task_record)
        return str(task_record), str(task_record)

    def _deep_inner_rows(self, fit_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self.bundle is not None
        dates = pd.DatetimeIndex(np.unique(self.bundle.dates[fit_idx])).sort_values()
        validation_days = min(40, max(20, len(dates) // 6))
        purge_days = min(20, max(5, len(dates) // 25))
        validation_dates = dates[-validation_days:]
        train_dates = dates[: -(validation_days + purge_days)]
        return (
            np.intersect1d(fit_idx, np.flatnonzero(np.isin(self.bundle.dates, train_dates.to_numpy()))),
            np.intersect1d(fit_idx, np.flatnonzero(np.isin(self.bundle.dates, validation_dates.to_numpy()))),
        )

    def _run_deep_profile(self, claim: TaskClaim, profile: str) -> str:
        assert self.bundle is not None
        bundle = self.bundle
        record_path = self.result_dir / "task_records" / f"{claim.task_id}__{profile}.json"
        if record_path.exists():
            return str(record_path)
        started = time.perf_counter()
        if self.windows is None:
            self.windows = build_window_indices(bundle.tickers, bundle.dates, int(self.plan["deep_sequence_length"]))
        context = bundle.fold_indices(claim.outer_fold, profile)
        matrix, feature_names = bundle.matrix(np.arange(len(bundle.df), dtype=np.int32), profile)
        if profile == "common_period":
            start_date = np.datetime64(bundle.common_start_date)
            for key in ("fit_idx", "calibration_idx", "validation_idx"):
                rows = context[key]
                valid = np.all(self.windows[rows] >= 0, axis=1)
                first_rows = np.where(valid, self.windows[rows, 0], 0)
                valid &= bundle.dates[first_rows] >= start_date
                context[key] = rows[valid]
        deep_train, deep_valid = self._deep_inner_rows(context["fit_idx"])
        fitted = fit_deep_model(
            claim.family, matrix, bundle.y, self.windows, deep_train, deep_valid,
            seed=claim.seed, sequence_length=int(self.plan["deep_sequence_length"]),
            max_features=int(self.plan["deep_max_features"]), max_epochs=int(self.plan["deep_max_epochs"]),
            patience=int(self.plan["deep_patience"]), threads=claim.threads,
            memory_fraction=float(self.plan["gpu_memory_fraction"]),
        )
        cal_rows, raw_cal = predict_deep_model(
            fitted, matrix, self.windows, context["calibration_idx"],
            sequence_length=int(self.plan["deep_sequence_length"]), threads=claim.threads,
        )
        outer_rows, raw_outer = predict_deep_model(
            fitted, matrix, self.windows, context["validation_idx"],
            sequence_length=int(self.plan["deep_sequence_length"]), threads=claim.threads,
        )
        regime_cal = bundle.regime_matrix(cal_rows)
        policy = select_safe_policy(
            bundle.y[cal_rows], raw_cal, bundle.dates[cal_rows], regime=regime_cal,
            regime_names=bundle.regime_features, methods=tuple(self.plan["calibration_methods"]),
            max_roc_drop=float(self.plan["calibration_max_roc_drop"]),
            max_pr_drop=float(self.plan["calibration_max_pr_drop"]),
            min_rank_correlation=float(self.plan["calibration_min_rank_correlation"]),
        )
        calibrated_cal = apply_calibrator(policy.method, policy.params, raw_cal, regime_cal)
        calibrated_outer = apply_calibrator(policy.method, policy.params, raw_outer, bundle.regime_matrix(outer_rows))
        metadata = {
            "task_id": claim.task_id, "family": claim.family, "feature_profile": profile,
            "outer_fold": claim.outer_fold, "seed": claim.seed, "actual_backend": fitted.actual_backend,
            "config_name": f"{claim.family}_temporal", "calibration_method": policy.method,
        }
        outer_frame = _prediction_frame(bundle, outer_rows, raw_outer, calibrated_outer, metadata)
        cal_frame = _prediction_frame(bundle, cal_rows, raw_cal, calibrated_cal, metadata)
        outer_path = self.result_dir / "predictions" / "outer" / claim.family / profile / f"{claim.task_id}.parquet"
        cal_path = self.result_dir / "predictions" / "calibration" / claim.family / profile / f"{claim.task_id}.parquet"
        atomic_parquet(outer_frame, outer_path)
        atomic_parquet(cal_frame, cal_path)
        values = _metric_bundle(bundle.y[outer_rows], raw_outer, calibrated_outer, bundle.dates[outer_rows], policy.threshold)
        metrics_path = self.result_dir / "model_metrics" / claim.family / profile / f"{claim.task_id}.parquet"
        atomic_parquet(pd.DataFrame([{
            **metadata, "rows": len(outer_rows), "feature_count": len(fitted.selected_columns),
            "decision_threshold": policy.threshold, "calibration_fallback_reason": policy.fallback_reason,
            "elapsed_seconds": time.perf_counter() - started, "batch_size": fitted.batch_size,
            "epochs": fitted.epochs, **values,
        }]), metrics_path)
        atomic_json({
            **metadata, "policy": policy.payload(), "raw_roc_auc": values.get("raw_roc_auc"),
            "calibrated_roc_auc": values.get("roc_auc"), "raw_pr_auc": values.get("raw_pr_auc"),
            "calibrated_pr_auc": values.get("pr_auc"),
        }, self.result_dir / "calibration_audits" / claim.family / profile / f"{claim.task_id}.json")
        selected_names = [feature_names[index] for index in fitted.selected_columns]
        atomic_csv(pd.DataFrame({"feature": selected_names}), self.result_dir / "deep_selected_features" / claim.family / profile / f"{claim.task_id}.csv")
        atomic_json({
            **metadata, "status": "completed", "outer_prediction": str(outer_path),
            "calibration_prediction": str(cal_path), "metrics_path": str(metrics_path),
            "feature_count": len(selected_names), "batch_size": fitted.batch_size, "epochs": fitted.epochs,
        }, record_path)
        del matrix, fitted
        gc.collect()
        return str(record_path)

    def run_deep_task(self, claim: TaskClaim) -> tuple[str, str]:
        records = []
        for profile in ("full_reduced", "common_period"):
            records.append(self._run_deep_profile(claim, profile))
            self.registry.heartbeat(claim.task_id)
            if self.stop_requested():
                raise InterruptedError("safe stop after completed deep profile")
        task_record = self.result_dir / "task_records" / f"{claim.task_id}.json"
        atomic_json({"task_id": claim.task_id, "stage": "deep", "records": records, "status": "completed"}, task_record)
        return str(task_record), str(task_record)

    def loop(self, stages: tuple[str, ...], worker_name: str) -> None:
        self.load()
        while not self.stop_requested():
            claim = self.registry.claim_next(
                profile=self.profile, stages=stages, worker_name=worker_name,
                backend=self.requested_backend, threads=self.threads,
            )
            if claim is None:
                return
            try:
                if claim.stage == "tree":
                    result, record = self.run_tree_task(claim)
                elif claim.stage == "deep":
                    result, record = self.run_deep_task(claim)
                else:
                    raise ValueError(f"unsupported worker stage: {claim.stage}")
                self.registry.complete(claim.task_id, result_path=result, record_path=record)
            except InterruptedError as exc:
                self.registry.release(claim.task_id, str(exc))
                return
            except Exception:
                self.registry.fail(claim.task_id, traceback.format_exc())


def _worker_entry(payload: dict[str, Any]) -> None:
    worker = RefineWorker(payload)
    worker.loop(tuple(payload["stages"]), str(payload["worker_name"]))


def _task_plan(bundle: RefineDataBundle, plan: dict[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for family_index, family in enumerate(plan["tree_families"]):
        # Group pending tasks by seed before fold so parallel workers claim
        # different folds and do not wait on the same shared tuning policy.
        for seed_index, seed in enumerate(plan["outer_seeds"]):
            for fold in range(int(plan["outer_folds"])):
                task_id = stable_hash({
                    "schema": SCHEMA_VERSION, "dataset": bundle.dataset_signature,
                    "stage": "tree", "family": family, "fold": fold, "seed": seed,
                })
                tasks.append({
                    "task_id": task_id, "stage": "tree", "family": family,
                    "outer_fold": fold, "seed": int(seed),
                    "priority": 900000 - family_index * 10000 - seed_index * 100 - fold,
                    "required_profile": "any", "payload": {},
                })
    for family_index, family in enumerate(plan["deep_families"]):
        for fold in plan["deep_outer_folds"]:
            for seed in plan["deep_seeds"]:
                task_id = stable_hash({
                    "schema": SCHEMA_VERSION, "dataset": bundle.dataset_signature,
                    "stage": "deep", "family": family, "fold": fold, "seed": seed,
                })
                tasks.append({
                    "task_id": task_id, "stage": "deep", "family": family,
                    "outer_fold": int(fold), "seed": int(seed), "priority": 5000 - family_index * 100 - int(fold),
                    "required_profile": "full", "payload": {},
                })
    return tasks


def _read_nvidia_smi() -> dict[str, float]:
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ], text=True, timeout=5).strip().splitlines()[0]
        values = [float(item.strip()) for item in output.split(",")]
        return {"gpu_util_pct": values[0], "gpu_memory_used_mb": values[1], "gpu_memory_total_mb": values[2], "gpu_temp_c": values[3], "gpu_power_w": values[4]}
    except Exception:
        return {"gpu_util_pct": math.nan, "gpu_memory_used_mb": math.nan, "gpu_memory_total_mb": math.nan, "gpu_temp_c": math.nan, "gpu_power_w": math.nan}


def _resource_monitor_loop(
    result_dir: Path,
    stop_event: threading.Event,
    stop_flag: Path,
    plan: dict[str, Any],
    profile: str,
) -> None:
    import psutil

    rows = []
    path = result_dir / "resource_usage.csv"
    while not stop_event.wait(10.0):
        vm = psutil.virtual_memory()
        gpu = _read_nvidia_smi()
        row = {
            "timestamp": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "profile": profile,
            "cpu_percent": psutil.cpu_percent(interval=None), "ram_used_gb": vm.used / 1024**3,
            "ram_available_gb": vm.available / 1024**3, **gpu,
        }
        rows.append(row)
        if len(rows) >= 6:
            existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
            atomic_csv(pd.concat([existing, pd.DataFrame(rows)], ignore_index=True), path)
            rows.clear()
        # In PUBG mode CUDA is hidden from CrashWatch. The foreground game's GPU
        # temperature must not be mistaken for experiment GPU overheating.
        if (
            profile == "full"
            and np.isfinite(gpu["gpu_temp_c"])
            and gpu["gpu_temp_c"] >= float(plan["gpu_temperature_stop"])
        ):
            stop_flag.write_text("gpu temperature safety stop", encoding="utf-8")
        if vm.available / 1024**3 < float(plan["min_available_ram_gb"]):
            stop_flag.write_text("low RAM safety stop", encoding="utf-8")
    if rows:
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        atomic_csv(pd.concat([existing, pd.DataFrame(rows)], ignore_index=True), path)


def _collect_calibration_audits(result_dir: Path) -> None:
    rows = []
    for path in sorted((result_dir / "calibration_audits").rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            policy = payload.pop("policy", {})
            rows.append({
                **payload, "threshold": policy.get("threshold"), "rank_correlation": policy.get("rank_correlation"),
                "validation_raw_roc_auc": policy.get("validation_raw_roc_auc"),
                "validation_calibrated_roc_auc": policy.get("validation_calibrated_roc_auc"),
                "validation_raw_pr_auc": policy.get("validation_raw_pr_auc"),
                "validation_calibrated_pr_auc": policy.get("validation_calibrated_pr_auc"),
                "fallback_reason": policy.get("fallback_reason"),
            })
        except Exception:
            continue
    atomic_csv(pd.DataFrame(rows), result_dir / "model_calibration_audit.csv")


def _choose_candidate(result_dir: Path) -> dict[str, Any] | None:
    path = result_dir / "model_family_summary.csv"
    if not path.exists():
        return None
    summary = pd.read_csv(path)
    tree = summary.loc[summary["family"].isin(["xgboost", "lightgbm", "catboost"])].copy()
    if tree.empty:
        return None
    row = tree.sort_values("base_score", ascending=False).iloc[0]
    return row.to_dict()


def _train_base_candidate(project: Path, dataset: Path | None, result_dir: Path, plan: dict[str, Any], profile: str) -> dict[str, Any]:
    candidate = _choose_candidate(result_dir)
    if candidate is None:
        return {"status": "no_tree_candidate"}
    bundle = load_refine_data(project, dataset, result_dir, plan)
    family = str(candidate["family"])
    feature_profile = str(candidate["feature_profile"])
    matrix, feature_names = bundle.matrix(np.arange(len(bundle.df), dtype=np.int32), feature_profile)
    mask = bundle.profile_row_mask(feature_profile)
    dates = pd.DatetimeIndex(np.unique(bundle.dates[mask])).sort_values()
    calibration_dates = dates[-60:]
    fit_dates = dates[:-80]
    fit_idx = np.flatnonzero(mask & np.isin(bundle.dates, fit_dates.to_numpy()))
    calibration_idx = np.flatnonzero(mask & np.isin(bundle.dates, calibration_dates.to_numpy()))
    policy_files = sorted((result_dir / "tuning_policies" / family / feature_profile).glob("*.json"))
    names = []
    payloads = {}
    for path in policy_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            name = payload["selected_config"]["name"]
            names.append(name)
            payloads[name] = payload["selected_config"]
        except Exception:
            continue
    config_name = Counter(names).most_common(1)[0][0] if names else TREE_CONFIGS[family][0]["name"]
    config = payloads.get(config_name, next(item for item in TREE_CONFIGS[family] if item["name"] == config_name))
    requested_backend = "cuda" if profile == "full" else "cpu"
    threads = 16 if profile == "full" else 4
    calibration_model = fit_tree_model(
        family, config, matrix[fit_idx], bundle.y[fit_idx], seed=CALIBRATION_SEED,
        requested_backend=requested_backend, threads=threads,
    )
    actual_backend = calibration_model.actual_backend
    raw_cal = calibration_model.predict(matrix[calibration_idx])
    release_tree(calibration_model)
    safe_policy = select_safe_policy(
        bundle.y[calibration_idx], raw_cal, bundle.dates[calibration_idx],
        regime=bundle.regime_matrix(calibration_idx), regime_names=bundle.regime_features,
        methods=tuple(plan["calibration_methods"]), max_roc_drop=float(plan["calibration_max_roc_drop"]),
        max_pr_drop=float(plan["calibration_max_pr_drop"]), min_rank_correlation=float(plan["calibration_min_rank_correlation"]),
    )
    force = "cpu" if "cpu" in actual_backend else None
    model = fit_tree_model(
        family, config, matrix[fit_idx], bundle.y[fit_idx], seed=FINAL_SEED,
        requested_backend="cpu" if force else requested_backend, threads=threads, force_backend=force,
    )
    candidate_dir = result_dir / "base_candidate"
    model.save(candidate_dir / "base_candidate")
    importance = model.feature_importance()
    release_tree(model)
    feature_frame = pd.DataFrame({"feature": feature_names, "feature_index": range(len(feature_names))})
    if len(importance) == len(feature_names):
        feature_frame["importance"] = importance
    atomic_csv(feature_frame, candidate_dir / "feature_list.csv")
    atomic_json(safe_policy.payload(), candidate_dir / "calibration_policy.json")
    card = {
        "status": "development_base_not_final", "family": family, "feature_profile": feature_profile,
        "config": config, "backend": actual_backend, "threads": threads,
        "dataset_signature": bundle.dataset_signature, "train_rows": len(fit_idx),
        "calibration_rows": len(calibration_idx), "feature_count": len(feature_names),
        "common_start_date": str(bundle.common_start_date.date()),
        "redundant_features_removed": len(bundle.redundant_features),
        "high_missing_features_removed": len(bundle.high_missing_features) if feature_profile == "common_period" else 0,
        "selection_summary": candidate,
        "warning": "This is a development base candidate, not a production probability model.",
    }
    atomic_json(card, candidate_dir / "model_card.json")
    return card


def run_supervisor(
    project: Path,
    dataset: Path | None,
    *,
    profile: str,
    hours: float,
    tree_workers: int | None = None,
    result_dir: Path | None = None,
    base12_source: Path | None = None,
) -> dict[str, Any]:
    project = project.resolve()
    plan = json.loads((project / "configs" / "refine12h_plan.json").read_text(encoding="utf-8"))
    if profile == "pubg":
        _apply_affinity(profile, int(plan["pubg_threads"]), 0, 1)
    paths = get_paths(project)
    result_dir = (result_dir or paths.data_root / "refine12h").resolve()
    result_dir.mkdir(parents=True, exist_ok=True)
    stop_flag = result_dir / "REQUEST_SAFE_STOP.flag"
    stop_flag.unlink(missing_ok=True)
    deadline_epoch = time.time() + max(0.1, hours) * 3600
    bundle = load_refine_data(project, dataset, result_dir, plan)
    registry = RefineTaskRegistry(result_dir / "task_registry.sqlite")
    registry.set_meta("schema", SCHEMA_VERSION)
    registry.set_meta("dataset_signature", bundle.dataset_signature)
    registry.ensure_tasks(_task_plan(bundle, plan))
    registry.reset_stale_claims()

    base12_source = (base12_source or paths.data_root / "ablation_base12h").resolve()
    reaggregation_status_path = result_dir / "raw_reaggregation_status.json"
    if not reaggregation_status_path.exists() or json.loads(reaggregation_status_path.read_text(encoding="utf-8")).get("dataset_signature") not in {None, bundle.dataset_signature}:
        reaggregation = reaggregate_base12h_raw(base12_source, result_dir, bundle, plan)
        reaggregation["dataset_signature"] = bundle.dataset_signature
        atomic_json(reaggregation, reaggregation_status_path)
    else:
        reaggregation = json.loads(reaggregation_status_path.read_text(encoding="utf-8"))

    # On Windows the spawned worker loads its own copy of the 1,800+ column
    # panel. Keeping the supervisor copy alive wastes several GB for the whole
    # run, which is particularly harmful while a game is in the foreground.
    dataset_signature = bundle.dataset_signature
    common_start_date = str(bundle.common_start_date.date())
    del bundle
    _trim_process_memory()

    monitor_stop = threading.Event()
    monitor = threading.Thread(
        target=_resource_monitor_loop,
        args=(result_dir, monitor_stop, stop_flag, plan, profile),
        daemon=True,
    )
    monitor.start()
    started = time.time()
    processes: list[mp.Process] = []
    context = mp.get_context("spawn")
    if profile == "pubg":
        worker_count = 1
        threads = int(plan["pubg_threads"])
        backend = "cpu"
    else:
        worker_count = int(tree_workers or plan["full_tree_workers"])
        threads = int(plan["full_threads_per_tree_worker"])
        backend = "cuda"
    for index in range(worker_count):
        payload = {
            "project": str(project), "dataset": str(dataset) if dataset else None,
            "result_dir": str(result_dir), "profile": profile, "worker_index": index,
            "worker_count": worker_count, "threads": threads, "backend": backend,
            "deadline_epoch": deadline_epoch, "stop_flag": str(stop_flag),
            "registry_path": str(registry.path), "stages": ["tree"],
            "worker_name": f"{profile}_tree_{index}",
        }
        process = context.Process(target=_worker_entry, args=(payload,), name=payload["worker_name"])
        process.start()
        processes.append(process)
    for process in processes:
        remaining = max(1.0, deadline_epoch - time.time())
        process.join(timeout=remaining)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)

    if profile == "full" and not stop_flag.exists() and time.time() < deadline_epoch - 300:
        payload = {
            "project": str(project), "dataset": str(dataset) if dataset else None,
            "result_dir": str(result_dir), "profile": profile, "worker_index": 0,
            "worker_count": 1, "threads": int(plan["full_deep_threads"]), "backend": "cuda",
            "deadline_epoch": deadline_epoch, "stop_flag": str(stop_flag),
            "registry_path": str(registry.path), "stages": ["deep"], "worker_name": "full_deep_0",
        }
        deep_process = context.Process(target=_worker_entry, args=(payload,), name="full_deep_0")
        deep_process.start()
        deep_process.join(timeout=max(1.0, deadline_epoch - time.time()))
        if deep_process.is_alive():
            deep_process.terminate()
            deep_process.join(timeout=10)

    monitor_stop.set()
    monitor.join(timeout=15)
    registry.reset_stale_claims(stale_seconds=1)
    registry_status = registry.status()
    atomic_json(registry_status, result_dir / "registry_status.json")
    _collect_calibration_audits(result_dir)
    compile_result = compile_model_summary(result_dir)
    ensemble_result = compile_ensembles(result_dir, plan)
    # Include ensemble rows in a second summary pass.
    if int(ensemble_result.get("ensemble_rows", 0)) > 0:
        ensemble = pd.read_csv(result_dir / "ensemble_metrics.csv")
        model_metrics = pd.read_csv(result_dir / "model_family_metrics.csv") if (result_dir / "model_family_metrics.csv").exists() else pd.DataFrame()
        if not ensemble.empty:
            atomic_csv(pd.concat([model_metrics, ensemble], ignore_index=True, sort=False), result_dir / "model_family_metrics_with_ensemble.csv")
    bottleneck = write_bottleneck_report(result_dir)
    candidate = _train_base_candidate(project, dataset, result_dir, plan, profile) if registry_status["tasks_completed"] else {"status": "not_enough_tasks"}
    run_summary = {
        "schema": SCHEMA_VERSION, "run_id": pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6],
        "profile": profile, "hours": hours, "elapsed_hours": (time.time() - started) / 3600,
        "tree_workers": worker_count, "dataset_signature": dataset_signature,
        "common_start_date": common_start_date, "registry": registry_status,
        "raw_reaggregation": reaggregation, "compile": compile_result, "ensemble": ensemble_result,
        "bottleneck": bottleneck, "base_candidate": candidate, "safe_stop": stop_flag.exists(),
        "result_dir": str(result_dir),
    }
    atomic_json(run_summary, result_dir / "run_summary.json")
    write_result_brief(result_dir, run_summary)
    package = create_result_package(result_dir)
    run_summary["result_package"] = str(package)
    atomic_json(run_summary, result_dir / "run_summary.json")
    return run_summary
