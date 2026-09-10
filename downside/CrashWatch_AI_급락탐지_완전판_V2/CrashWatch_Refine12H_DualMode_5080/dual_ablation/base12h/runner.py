from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import shutil
import socket
import subprocess
import threading
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from ..config import get_paths, load_baskets
from ..experiment.splits import make_walk_forward_folds
from ..finance11h.model import CANDIDATES, ModelConfig
from ..finance11h.runner import Experiment, _feature_quality, _hash_file_metadata, _load_catalog, _merge_catalogs, _metrics
from ..finance_nested.calibration import CalibrationPolicy, apply_calibrator, select_train_only_policy
from ..finance_nested.runner import _exact_sign_p, _bh_qvalues, _metric_bundle
from ..io_utils import atomic_csv, atomic_json, atomic_parquet, normalize_date, normalize_ticker, stable_hash
from .registry import BlockClaim, TaskRegistry

SCHEMA_VERSION = "base12h_v1"
TUNE_POLICY_DEFAULTS = {
    0: "depth6_fast", 1: "depth8_slow", 2: "depth8_slow", 3: "lossguide64",
    4: "depth7_balanced", 5: "depth7_balanced", 6: "lossguide64", 7: "lossguide64",
}
CALIBRATION_SEED = 2718


def _strict_fit_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    config: ModelConfig,
    *,
    threads: int,
    backend: str,
):
    from xgboost import XGBClassifier

    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    scale_pos_weight = max(1.0, negatives / max(1, positives))
    params = dict(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        max_bin=max(config.max_bin, 384 if backend == "cuda" else config.max_bin),
        grow_policy=config.grow_policy,
        max_leaves=config.max_leaves,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=threads,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cuda" if backend == "cuda" else "cpu",
        verbosity=0,
        validate_parameters=True,
    )
    model = XGBClassifier(**params)
    model.fit(x_train, y_train)
    return model


def _apply_affinity(profile: str, threads: int, worker_index: int, worker_count: int) -> dict[str, Any]:
    import psutil

    process = psutil.Process()
    available = process.cpu_affinity() if hasattr(process, "cpu_affinity") else list(range(psutil.cpu_count() or threads))
    if profile == "pubg":
        physical_first = available[::2] or available
        selected = (physical_first[:threads] or available[:threads])
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
    return {"affinity": selected, "profile": profile, "threads": threads}


def _lock_file(path: Path, timeout: float = 600.0):
    class _Lock:
        def __enter__(self):
            deadline = time.time() + timeout
            path.parent.mkdir(parents=True, exist_ok=True)
            while True:
                try:
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.write(fd, f"{os.getpid()}\n".encode())
                    os.close(fd)
                    return self
                except FileExistsError:
                    if time.time() > deadline:
                        try:
                            if time.time() - path.stat().st_mtime > timeout:
                                path.unlink(missing_ok=True)
                                continue
                        except OSError:
                            pass
                        raise TimeoutError(f"lock timeout: {path}")
                    time.sleep(0.5)

        def __exit__(self, exc_type, exc, tb):
            path.unlink(missing_ok=True)
    return _Lock()


class Base12HWorker:
    def __init__(
        self,
        project: Path,
        dataset: Path | None,
        *,
        result_dir: Path,
        cache_namespace: str,
        deadline_epoch: float,
        stop_flag: Path,
        worker_name: str,
        worker_index: int,
        worker_count: int,
    ) -> None:
        self.project = project.resolve()
        self.paths = get_paths(self.project)
        self.dataset = dataset or self.paths.data_root / "development" / "training_dataset_finance11h.parquet"
        self.result_dir = result_dir
        self.cache_namespace = cache_namespace
        self.cache_dir = result_dir / "prediction_cache" / cache_namespace
        self.policy_dir = result_dir / "policies" / cache_namespace
        self.task_metrics_dir = result_dir / "task_metrics"
        self.task_records_dir = result_dir / "task_records"
        for p in [self.cache_dir, self.policy_dir, self.task_metrics_dir, self.task_records_dir]:
            p.mkdir(parents=True, exist_ok=True)
        self.deadline_epoch = deadline_epoch
        self.stop_flag = stop_flag
        self.worker_name = worker_name
        self.worker_index = worker_index
        self.worker_count = worker_count
        self.plan = json.loads((self.paths.configs / "base12h_plan.json").read_text(encoding="utf-8"))

    def stop_requested(self) -> bool:
        return self.stop_flag.exists() or time.time() >= self.deadline_epoch

    def load(self) -> None:
        if not self.dataset.exists():
            raise FileNotFoundError(f"Finance11H dataset missing: {self.dataset}")
        df = normalize_date(pd.read_parquet(self.dataset))
        df["ticker"] = normalize_ticker(df["ticker"])
        if "label_abs_crash_20" not in df.columns:
            raise KeyError("label_abs_crash_20 missing")
        baskets = load_baskets(self.paths)
        if "bucket" not in df.columns:
            df = df.merge(baskets[["ticker", "bucket"]], on="ticker", how="left")
        df["bucket"] = df["bucket"].fillna("other").astype(str)
        self.baskets_df = baskets.loc[baskets["ticker"].isin(set(df["ticker"]))].copy()

        required_summary_path = self.paths.raw_dual / "required_data_v3" / "required_data_summary.json"
        if required_summary_path.exists():
            required_summary = json.loads(required_summary_path.read_text(encoding="utf-8"))
            actual_short = required_summary.get("actual_short", {})
            lending = required_summary.get("stock_lending", {})
            if not actual_short.get("is_actual_short") or float(actual_short.get("coverage", 0.0)) < 0.80:
                raise RuntimeError("actual KRX short coverage below 80%")
            if float(lending.get("coverage", 0.0)) < 0.80:
                raise RuntimeError("stock lending coverage below 80%")

        base_u = _load_catalog(self.paths.feature_dual / "feature_catalog_universe.json")
        base_t = _load_catalog(self.paths.feature_dual / "feature_catalog_ticker.json")
        fin_u = _load_catalog(self.paths.feature_dual / "feature_catalog_finance11h_universe.json")
        fin_t = _load_catalog(self.paths.feature_dual / "feature_catalog_finance11h_ticker.json")
        catalog = _merge_catalogs(base_u, base_t, fin_u, fin_t)
        short_cols = catalog.get("t_financial_shorting", [])
        catalog["t_finshort_crowding_cluster"] = sorted({
            c for c in short_cols if any(k in c.lower() for k in ["crowding", "squeeze", "stress", "crash_pressure"])
        })

        market_cols = catalog.get("u_financial_market", [])
        def select(*words: str) -> list[str]:
            return sorted({c for c in market_cols if any(w in c.lower() for w in words)})
        catalog["u_market_fx"] = select("usd", "krw", "fx", "dxy", "exchange", "dollar")
        catalog["u_market_rates"] = select("rate", "yield", "treasury", "term", "base_rate", "gov", "korea_3y", "korea_10y")
        catalog["u_market_credit"] = select("credit", "spread", "corporate", "high_yield", "hy_", "aa", "baa", "aaa")
        catalog["u_market_volatility"] = select("vix", "vkospi", "volatility", "_vol", "vol_")
        catalog["u_market_conditions"] = select("nfci", "stlfsi", "financial_condition", "liquidity_condition", "stress_index")
        catalog["u_core_short_etf"] = sorted(set(catalog.get("u_financial_shorting", []) + catalog.get("u_etf_pressure", [])))
        catalog["u_core_short_market"] = sorted(set(catalog.get("u_financial_shorting", []) + market_cols))
        catalog["u_core_market_etf"] = sorted(set(market_cols + catalog.get("u_etf_pressure", [])))
        catalog["u_core_all"] = sorted(set(catalog.get("u_financial_shorting", []) + market_cols + catalog.get("u_etf_pressure", [])))

        candidates = sorted(set(sum(catalog.values(), [])))
        valid, quality = _feature_quality(df, candidates)
        valid_set = set(valid)
        quality_map = quality.set_index("feature")["missing_ratio"].to_dict() if not quality.empty else {}
        catalog["u_market_high_missing"] = sorted([c for c in market_cols if c in valid_set and float(quality_map.get(c, 0)) >= 0.60])

        df_numeric = df[valid].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")
        redundant = self._find_redundant(df_numeric, quality_map)
        catalog["redundant_feature_cluster"] = redundant
        self.catalog = {k: [c for c in v if c in valid_set] for k, v in catalog.items()}

        required_groups = [x["group"] for x in self.plan["global_groups"]] + [x["group"] for x in self.plan["bucket_combinations"]]
        missing = [g for g in required_groups if not self.catalog.get(g)]
        # 진단 하위군은 실제 열 이름에 따라 비어 있을 수 있다. 핵심 그룹만 strict다.
        strict = {"u_financial_shorting", "u_etf_pressure", "u_financial_market", "t_financial_shorting", "t_stock_lending"}
        strict_missing = sorted(strict.intersection(missing))
        if strict_missing:
            raise RuntimeError(f"required feature groups missing: {strict_missing}")

        self.features = valid
        self.df = df.reset_index(drop=True)
        self.X = np.ascontiguousarray(df_numeric.to_numpy(dtype=np.float32))
        self.y = pd.to_numeric(self.df["label_abs_crash_20"], errors="raise").to_numpy(dtype=np.int8)
        self.dates = self.df["date"].to_numpy()
        self.tickers = self.df["ticker"].astype(str).to_numpy()
        self.buckets = self.df["bucket"].astype(str).to_numpy()
        self.feature_index = {c: i for i, c in enumerate(self.features)}
        self.outer_folds = make_walk_forward_folds(self.df["date"], int(self.plan["outer_folds"]), 60, 20, 500)
        self.outer_indices: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for fold in self.outer_folds:
            fid = int(fold["fold_id"])
            train_idx = np.flatnonzero(np.isin(self.dates, pd.DatetimeIndex(fold["train_dates_index"]).to_numpy()))
            val_idx = np.flatnonzero(np.isin(self.dates, pd.DatetimeIndex(fold["validation_dates_index"]).to_numpy()))
            self.outer_indices[fid] = (train_idx, val_idx)
        self.dataset_signature = stable_hash({
            "metadata": _hash_file_metadata(self.dataset), "rows": len(self.df), "features": self.features,
            "date_min": str(self.df["date"].min()), "date_max": str(self.df["date"].max()), "schema": SCHEMA_VERSION,
        })
        quality = quality.copy()
        quality["group"] = quality["feature"].map(lambda c: next((g for g, cols in self.catalog.items() if c in cols), None))
        atomic_csv(quality, self.result_dir / "valid_feature_audit.csv")
        atomic_csv(pd.DataFrame({"feature": redundant}), self.result_dir / "redundant_features.csv")
        atomic_json({
            "dataset": str(self.dataset), "dataset_signature": self.dataset_signature, "rows": len(self.df),
            "tickers": int(self.df["ticker"].nunique()), "features": len(self.features),
            "date_min": str(self.df["date"].min()), "date_max": str(self.df["date"].max()),
        }, self.result_dir / "data_summary.json")

    @staticmethod
    def _find_redundant(df: pd.DataFrame, missing_map: dict[str, float]) -> list[str]:
        if df.empty or df.shape[1] < 2:
            return []
        sample = df.iloc[np.linspace(0, len(df) - 1, min(len(df), 30000), dtype=int)].copy()
        # median fill is only for correlation diagnostics.
        sample = sample.fillna(sample.median(numeric_only=True)).fillna(0.0)
        corr = sample.corr(method="pearson").abs()
        columns = list(corr.columns)
        drop: set[str] = set()
        for i, left in enumerate(columns):
            if left in drop:
                continue
            for right in columns[i + 1:]:
                if right in drop:
                    continue
                value = corr.at[left, right]
                if np.isfinite(value) and value >= 0.995:
                    # 결측이 많은 쪽을 제거한다.
                    left_missing = float(missing_map.get(left, 0.0))
                    right_missing = float(missing_map.get(right, 0.0))
                    drop.add(right if right_missing >= left_missing else left)
                    if left in drop:
                        break
        return sorted(drop)

    def experiments(self) -> list[Experiment]:
        rows: list[tuple[int, Experiment]] = [(10_000, Experiment("baseline", "baseline", "none"))]
        for item in self.plan["global_groups"]:
            if self.catalog.get(item["group"]):
                rows.append((int(item["priority"]), Experiment(f"global__{item['group']}", "base12h_global", "global_drop", item["group"])))
        for item in self.plan["bucket_combinations"]:
            if self.catalog.get(item["group"]):
                rows.append((int(item["priority"]), Experiment(
                    f"bucket__{item['bucket']}__{item['group']}", "base12h_bucket", "bucket_mask",
                    item["group"], target_bucket=item["bucket"],
                )))
        return [exp for _, exp in sorted(rows, key=lambda x: (-x[0], x[1].name))]

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

    def split_context(self, outer_fold: int) -> dict[str, Any]:
        outer_train_idx, _ = self.outer_indices[outer_fold]
        train_dates = pd.DatetimeIndex(np.unique(self.dates[outer_train_idx])).sort_values()
        cal_days = int(self.plan["calibration_days"])
        cal_purge = int(self.plan["calibration_purge_days"])
        required_tail = cal_days + cal_purge
        if len(train_dates) < required_tail + 420:
            raise RuntimeError(f"outer fold {outer_fold}: insufficient dates")
        calibration_dates = train_dates[-cal_days:]
        calibration_train_dates = train_dates[:-required_tail]
        return {
            "calibration_train_idx": np.flatnonzero(np.isin(self.dates, calibration_train_dates.to_numpy())),
            "calibration_idx": np.flatnonzero(np.isin(self.dates, calibration_dates.to_numpy())),
        }

    def model_config(self, outer_fold: int) -> ModelConfig:
        # 이전 FocusedNested 결과의 nested 정책을 재사용한다. 사용자의 data directory에 정책이 있으면 우선한다.
        old_policy_root = self.paths.data_root / "ablation_finance_nested_focus" / "nested_policies" / "finance_nested_focus_v1"
        candidates = list(old_policy_root.glob(f"**/outer_{outer_fold:02d}_model_policy.json")) if old_policy_root.exists() else []
        for path in candidates:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                cfg = payload.get("selected_config", {})
                allowed = set(ModelConfig.__dataclass_fields__)
                return ModelConfig(**{k: cfg[k] for k in allowed if k in cfg})
            except Exception:
                continue
        name = TUNE_POLICY_DEFAULTS.get(outer_fold, "depth7_balanced")
        return next(c for c in CANDIDATES if c.name == name)

    def _policy_path(self, exp: Experiment, outer_fold: int, backend: str, config: ModelConfig) -> Path:
        key = stable_hash({"schema": SCHEMA_VERSION, "dataset": self.dataset_signature, "exp": asdict(exp), "fold": outer_fold, "backend": backend, "config": config.payload()})
        return self.policy_dir / "calibration" / backend / f"{key}.json"

    def calibration_policy(
        self,
        exp: Experiment,
        outer_fold: int,
        backend: str,
        threads: int,
        config: ModelConfig,
        context: dict[str, Any],
        baseline_policy: CalibrationPolicy | None,
    ) -> CalibrationPolicy:
        path = self._policy_path(exp, outer_fold, backend, config)
        lock = path.with_suffix(".lock")
        with _lock_file(lock):
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                return CalibrationPolicy(**payload["policy"])
            train_idx, cal_idx = context["calibration_train_idx"], context["calibration_idx"]
            x_train, x_cal, names = self._matrix(exp, train_idx, cal_idx)
            raw_key = stable_hash({"schema": SCHEMA_VERSION, "dataset": self.dataset_signature, "exp": asdict(exp), "fold": outer_fold, "backend": backend, "config": config.payload(), "features": names})
            raw_path = self.cache_dir / "calibration" / backend / f"{raw_key}.parquet"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            if raw_path.exists():
                pred = pd.read_parquet(raw_path)
            else:
                model = _strict_fit_xgb(x_train, self.y[train_idx], CALIBRATION_SEED, config, threads=threads, backend=backend)
                raw = model.predict_proba(x_cal)[:, 1].astype(np.float32)
                pred = self.df.loc[cal_idx, ["date", "ticker", "bucket"]].copy()
                pred["target"] = self.y[cal_idx]
                pred["raw_prediction"] = raw
                atomic_parquet(pred, raw_path)
            forced_method = baseline_policy.method if baseline_policy else None
            methods = (forced_method,) if forced_method else ("none", "sigmoid", "beta", "isotonic")
            policy = select_train_only_policy(
                pred["target"].to_numpy(dtype=np.int8), pred["raw_prediction"].to_numpy(dtype=float),
                pred["date"].to_numpy(), methods=methods,
            )
            if baseline_policy is not None:
                policy = CalibrationPolicy(
                    method=policy.method, params=policy.params, threshold=baseline_policy.threshold,
                    selection_score=policy.selection_score, validation_brier=policy.validation_brier,
                    validation_logloss=policy.validation_logloss, validation_rows=policy.validation_rows,
                    validation_positives=policy.validation_positives,
                )
            atomic_json({
                "schema": SCHEMA_VERSION, "dataset_signature": self.dataset_signature, "backend": backend,
                "outer_fold": outer_fold, "experiment": asdict(exp), "config": config.payload(),
                "raw_prediction_path": str(raw_path), "policy": policy.payload(),
            }, path)
            return policy

    def execute_task(
        self,
        claim: BlockClaim,
        exp: Experiment,
        config: ModelConfig,
        policy: CalibrationPolicy,
        context: dict[str, Any],
    ) -> tuple[str, Path, Path]:
        train_idx = context["calibration_train_idx"]
        _, val_idx = self.outer_indices[claim.outer_fold]
        x_train, x_val, names = self._matrix(exp, train_idx, val_idx)
        task_id = stable_hash({
            "schema": SCHEMA_VERSION, "dataset": self.dataset_signature, "block": claim.block_id,
            "backend": claim.backend, "experiment": asdict(exp), "config": config.payload(),
            "calibration_method": policy.method, "threshold": policy.threshold, "features": names,
        })
        pred_path = self.cache_dir / "outer" / claim.backend / f"{task_id}.parquet"
        metrics_path = self.task_metrics_dir / f"{task_id}.parquet"
        record_path = self.task_records_dir / f"{task_id}.json"
        if metrics_path.exists() and record_path.exists():
            return task_id, metrics_path, record_path
        started = time.perf_counter()
        if pred_path.exists():
            pred = pd.read_parquet(pred_path)
            cache_status = "hit"
        else:
            model = _strict_fit_xgb(x_train, self.y[train_idx], claim.seed, config, threads=claim.threads, backend=claim.backend)
            raw = model.predict_proba(x_val)[:, 1].astype(np.float32)
            calibrated = apply_calibrator(policy.method, policy.params, raw)
            pred = self.df.loc[val_idx, ["date", "ticker", "bucket"]].copy()
            pred["target"] = self.y[val_idx]
            pred["raw_prediction"] = raw
            pred["prediction"] = calibrated
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_parquet(pred, pred_path)
            cache_status = "miss"
        scopes = [("all", "all_validation", pred)]
        if exp.name == "baseline":
            for bucket in sorted(set(self.baskets_df["bucket"].astype(str))):
                scopes.append(("bucket", bucket, pred.loc[pred["bucket"].eq(bucket)]))
        if exp.target_bucket:
            scopes.append(("target_bucket", exp.target_bucket, pred.loc[pred["bucket"].eq(exp.target_bucket)]))
        rows = []
        for scope_type, scope_value, block in scopes:
            if block.empty:
                continue
            rows.append({
                "task_id": task_id, "block_id": claim.block_id, "dataset_signature": self.dataset_signature,
                "backend": claim.backend, "threads": claim.threads, "experiment": exp.name,
                "stage": exp.stage, "mode": exp.mode, "group": exp.group,
                "target_bucket": exp.target_bucket, "outer_fold": claim.outer_fold, "seed": claim.seed,
                "scope_type": scope_type, "scope_value": scope_value, "config": config.name,
                "calibration_method": policy.method, "decision_threshold": policy.threshold,
                **_metric_bundle(
                    block["target"].to_numpy(dtype=np.int8), block["raw_prediction"].to_numpy(dtype=float),
                    block["prediction"].to_numpy(dtype=float), block["date"].to_numpy(), policy.threshold,
                ),
            })
        metrics = pd.DataFrame(rows)
        atomic_parquet(metrics, metrics_path)
        atomic_json({
            "task_id": task_id, "block_id": claim.block_id, "dataset_signature": self.dataset_signature,
            "backend": claim.backend, "threads": claim.threads,
            "experiment": exp.name, "outer_fold": claim.outer_fold, "seed": claim.seed,
            "config": config.name, "calibration_method": policy.method, "cache_status": cache_status,
            "feature_count": len(names), "elapsed_seconds": time.perf_counter() - started,
            "prediction_path": str(pred_path), "metrics_path": str(metrics_path), "status": "completed",
        }, record_path)
        return task_id, metrics_path, record_path

    def process_block(self, registry: TaskRegistry, claim: BlockClaim) -> None:
        context = self.split_context(claim.outer_fold)
        config = self.model_config(claim.outer_fold)
        baseline_policy: CalibrationPolicy | None = None
        for exp in self.experiments():
            registry.heartbeat(claim.block_id)
            policy = self.calibration_policy(exp, claim.outer_fold, claim.backend, claim.threads, config, context, baseline_policy)
            if exp.name == "baseline":
                baseline_policy = policy
            task_stub = stable_hash({"block": claim.block_id, "experiment": exp.name, "backend": claim.backend, "schema": SCHEMA_VERSION})
            # 실제 task id에는 config/calibration/features가 들어가지만 registry 선점은 stub로 한다.
            if registry.task_completed(task_stub):
                continue
            registry.start_task(task_id=task_stub, block_id=claim.block_id, experiment=exp.name, backend=claim.backend, threads=claim.threads)
            try:
                _, metrics_path, record_path = self.execute_task(claim, exp, config, policy, context)
                registry.complete_task(task_id=task_stub, result_path=str(metrics_path), record_path=str(record_path))
            except Exception as exc:
                registry.fail_task(task_stub, traceback.format_exc())
                raise RuntimeError(f"block={claim.block_id}, experiment={exp.name}: {exc}") from exc
            if self.stop_requested():
                registry.release_block(claim.block_id, "safe stop after completed experiment")
                return
        registry.finish_block(claim.block_id)


def _worker_main(payload: dict[str, Any]) -> None:
    project = Path(payload["project"])
    result_dir = Path(payload["result_dir"])
    registry = TaskRegistry(Path(payload["registry_path"]))
    worker_name = str(payload["worker_name"])
    profile = str(payload["profile"])
    default_backend = str(payload["backend"])
    default_threads = int(payload["threads"])
    worker_index = int(payload["worker_index"])
    worker_count = int(payload["worker_count"])
    _apply_affinity(profile, default_threads, worker_index, worker_count)
    if profile == "pubg":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    worker = Base12HWorker(
        project, Path(payload["dataset"]) if payload.get("dataset") else None,
        result_dir=result_dir, cache_namespace=str(payload["cache_namespace"]),
        deadline_epoch=float(payload["deadline_epoch"]), stop_flag=Path(payload["stop_flag"]),
        worker_name=worker_name, worker_index=worker_index, worker_count=worker_count,
    )
    worker.load()
    while not worker.stop_requested():
        claim = registry.claim_next_block(
            worker_name=worker_name, default_backend=default_backend,
            default_threads=default_threads, profile=profile,
        )
        if claim is None:
            return
        # 이전 배그 실행에서 시작된 block은 풀로드 실행에서도 CPU4로 끝까지 보존한다.
        _apply_affinity("pubg" if claim.backend == "cpu" else "full", claim.threads, worker_index, worker_count)
        try:
            worker.process_block(registry, claim)
        except Exception:
            registry.release_block(claim.block_id, traceback.format_exc())
            error_path = result_dir / "worker_errors" / f"{worker_name}_{int(time.time())}.txt"
            error_path.parent.mkdir(parents=True, exist_ok=True)
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            time.sleep(3)


def _collect_task_outputs(result_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_files = sorted((result_dir / "task_metrics").glob("*.parquet"))
    metrics = pd.concat([pd.read_parquet(p) for p in metric_files], ignore_index=True) if metric_files else pd.DataFrame()
    records = []
    for p in sorted((result_dir / "task_records").glob("*.json")):
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return metrics, pd.DataFrame(records)


def _compile_summary(result_dir: Path, plan: dict[str, Any], dataset_signature: str) -> dict[str, Any]:
    metrics, records = _collect_task_outputs(result_dir)
    if metrics.empty:
        return {"metrics_rows": 0, "paired_rows": 0}
    metrics = metrics.loc[metrics["dataset_signature"].eq(dataset_signature)].copy()
    metrics = metrics.drop_duplicates(["task_id", "scope_type", "scope_value"], keep="last")
    atomic_csv(metrics, result_dir / "all_metrics.csv")
    if not records.empty:
        if "dataset_signature" in records.columns:
            records = records.loc[records["dataset_signature"].eq(dataset_signature)].copy()
        atomic_csv(records, result_dir / "task_manifest.csv")

    baseline = metrics.loc[metrics["experiment"].eq("baseline")].copy()
    metric_cols = ["pr_auc", "raw_pr_auc", "roc_auc", "raw_roc_auc", "balanced_accuracy", "accuracy", "brier", "logloss", "top_3pct_precision"]
    base = baseline.copy()
    base["pair_scope_type"] = base["scope_type"].replace({"target_bucket": "bucket"})
    keys = ["block_id", "outer_fold", "seed", "backend", "pair_scope_type", "scope_value"]
    base = base[keys + metric_cols].drop_duplicates(keys).rename(columns={c: f"baseline_{c}" for c in metric_cols})
    compared = metrics.loc[~metrics["experiment"].eq("baseline")].copy()
    compared["pair_scope_type"] = compared["scope_type"].replace({"target_bucket": "bucket"})
    compared = compared.merge(base, on=keys, how="left", validate="many_to_one")
    compared["backend_pair_ok"] = compared["baseline_pr_auc"].notna()
    if not compared["backend_pair_ok"].all():
        raise RuntimeError("backend-paired baseline missing; task registry is inconsistent")
    compared["pr_auc_loss"] = compared["baseline_pr_auc"] - compared["pr_auc"]
    compared["raw_pr_auc_loss"] = compared["baseline_raw_pr_auc"] - compared["raw_pr_auc"]
    compared["roc_auc_loss"] = compared["baseline_roc_auc"] - compared["roc_auc"]
    compared["balanced_accuracy_loss"] = compared["baseline_balanced_accuracy"] - compared["balanced_accuracy"]
    compared["brier_increase"] = compared["brier"] - compared["baseline_brier"]
    compared["logloss_increase"] = compared["logloss"] - compared["baseline_logloss"]
    compared["top_3pct_precision_loss"] = compared["baseline_top_3pct_precision"] - compared["top_3pct_precision"]
    atomic_csv(compared, result_dir / "paired_ablation_deltas.csv")

    group_keys = ["experiment", "stage", "mode", "group", "target_bucket", "scope_type", "scope_value"]
    rows = []
    for values, block in compared.groupby(group_keys, dropna=False, sort=False):
        row = dict(zip(group_keys, values if isinstance(values, tuple) else (values,)))
        row["backend_pair_rate"] = float(block["backend_pair_ok"].mean())
        row["block_count"] = int(block["block_id"].nunique())
        row["outer_fold_count"] = int(block["outer_fold"].nunique())
        row["seed_count"] = int(block["seed"].nunique())
        row["cpu_blocks"] = int(block.loc[block["backend"].eq("cpu"), "block_id"].nunique())
        row["cuda_blocks"] = int(block.loc[block["backend"].eq("cuda"), "block_id"].nunique())
        for col in ["pr_auc_loss", "raw_pr_auc_loss", "roc_auc_loss", "balanced_accuracy_loss", "brier_increase", "logloss_increase", "top_3pct_precision_loss"]:
            fold_values = block.groupby("outer_fold")[col].mean().dropna().to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.mean(fold_values)) if len(fold_values) else np.nan
            row[f"{col}_median"] = float(np.median(fold_values)) if len(fold_values) else np.nan
            row[f"{col}_std"] = float(np.std(fold_values, ddof=1)) if len(fold_values) > 1 else np.nan
            row[f"{col}_positive_fold_ratio"] = float(np.mean(fold_values > 0)) if len(fold_values) else np.nan
            row[f"{col}_sign_p"] = _exact_sign_p(fold_values)
            for backend in ["cpu", "cuda"]:
                bvals = block.loc[block["backend"].eq(backend)].groupby("outer_fold")[col].mean().dropna().to_numpy(dtype=float)
                row[f"{backend}_{col}_mean"] = float(np.mean(bvals)) if len(bvals) else np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["test_family"] = np.where(summary["stage"].eq("base12h_global"), "global_all", "bucket_target")
        summary["pr_auc_loss_sign_q"] = summary.groupby("test_family", group_keys=False)["pr_auc_loss_sign_p"].apply(_bh_qvalues)
        summary["raw_pr_auc_loss_sign_q"] = summary.groupby("test_family", group_keys=False)["raw_pr_auc_loss_sign_p"].apply(_bh_qvalues)
    atomic_csv(summary, result_dir / "base12h_ablation_summary.csv")

    pairing = compared.groupby(["backend", "outer_fold"], as_index=False).agg(
        pair_count=("backend_pair_ok", "size"), pair_rate=("backend_pair_ok", "mean")
    )
    atomic_csv(pairing, result_dir / "backend_pairing_audit.csv")

    baseline_all = baseline.loc[baseline["scope_type"].eq("all")].copy()
    calibration = baseline_all.groupby(["outer_fold", "calibration_method"], as_index=False).agg(
        prevalence=("positive_rate", "mean"), raw_pr_auc=("raw_pr_auc", "mean"), calibrated_pr_auc=("pr_auc", "mean"),
        raw_roc_auc=("raw_roc_auc", "mean"), calibrated_roc_auc=("roc_auc", "mean"),
        brier=("brier", "mean"), logloss=("logloss", "mean"), mean_prediction=("mean_prediction", "mean"),
    )
    atomic_csv(calibration, result_dir / "calibration_by_fold.csv")

    # 병목 진단용 표
    diagnostics: list[dict[str, Any]] = []
    means = baseline_all.select_dtypes(include=[np.number]).mean().to_dict()
    diagnostics.append({
        "bottleneck": "ranking_capacity", "severity": "high" if float(means.get("roc_auc", 0)) < 0.70 else "medium",
        "evidence": f"ROC-AUC={means.get('roc_auc', float('nan')):.4f}, PR-AUC={means.get('pr_auc', float('nan')):.4f}",
        "interpretation": "피처 추가만으로 해결되지 않는 모델 순위 예측력 병목",
        "next_action": "LightGBM/CatBoost/temporal model과 동일 split 비교; label horizon 재검토",
    })
    diagnostics.append({
        "bottleneck": "calibration_regime_shift", "severity": "high" if float(means.get("brier_skill", -1)) < 0 else "medium",
        "evidence": f"Brier skill={means.get('brier_skill', float('nan')):.4f}, LogLoss skill={means.get('logloss_skill', float('nan')):.4f}",
        "interpretation": "시장 급락률 국면 변화에 확률 보정이 적응하지 못함",
        "next_action": "regime-conditioned calibration, rolling prior correction, raw ranking과 probability model 분리",
    })
    audit = pd.read_csv(result_dir / "valid_feature_audit.csv") if (result_dir / "valid_feature_audit.csv").exists() else pd.DataFrame()
    if not audit.empty:
        high_missing = int((pd.to_numeric(audit["missing_ratio"], errors="coerce") >= 0.60).sum())
        diagnostics.append({
            "bottleneck": "feature_missingness", "severity": "high" if high_missing >= 20 else "medium",
            "evidence": f"missing_ratio>=60% features={high_missing}",
            "interpretation": "결측 indicator가 데이터 출처·기간을 대리할 가능성",
            "next_action": "공통기간 모델과 full-period 모델 분리, availability indicator ablation",
        })
    redundant_path = result_dir / "redundant_features.csv"
    redundant_count = len(pd.read_csv(redundant_path)) if redundant_path.exists() else 0
    diagnostics.append({
        "bottleneck": "feature_redundancy", "severity": "high" if redundant_count >= 30 else "medium",
        "evidence": f"abs(corr)>=0.995 redundant features={redundant_count}",
        "interpretation": "중복 피처가 학습시간과 중요도 해석을 악화",
        "next_action": "상관 클러스터 대표 피처 선택 및 cluster ablation",
    })
    if not summary.empty:
        market = summary.loc[summary["group"].astype(str).str.startswith("u_market_")]
        if not market.empty:
            top = market.sort_values("pr_auc_loss_mean", ascending=False).iloc[0]
            diagnostics.append({
                "bottleneck": "financial_market_subgroup", "severity": "medium",
                "evidence": f"best subgroup={top['group']}, loss={top['pr_auc_loss_mean']:.5f}",
                "interpretation": "u_financial_market 효과가 특정 하위군에 집중되는지 확인 가능",
                "next_action": "양수 하위군만 유지하고 고결측·음수 하위군 제거",
            })
    diagnostics.append({
        "bottleneck": "backend_contamination", "severity": "resolved",
        "evidence": f"paired backend rate={compared['backend_pair_ok'].mean():.3f}",
        "interpretation": "각 fold-seed block에서 baseline과 ablation backend를 고정",
        "next_action": "CPU/CUDA 효과 차이는 backend-stratified summary로만 해석",
    })
    atomic_csv(pd.DataFrame(diagnostics), result_dir / "bottleneck_diagnostics.csv")

    goals = plan["success_targets"]
    goal_report = {
        "mean_accuracy": means.get("accuracy"), "mean_balanced_accuracy": means.get("balanced_accuracy"),
        "mean_roc_auc": means.get("roc_auc"), "mean_pr_auc": means.get("pr_auc"),
        "mean_pr_auc_lift": means.get("pr_auc_lift"), "mean_top_3pct_precision": means.get("top_3pct_precision"),
        "mean_brier_skill": means.get("brier_skill"), "mean_logloss_skill": means.get("logloss_skill"),
    }
    goal_report["base_readiness_score"] = float(np.mean([
        min(1.0, max(0.0, float(goal_report.get("mean_roc_auc") or 0) / goals["roc_auc"])),
        min(1.0, max(0.0, float(goal_report.get("mean_balanced_accuracy") or 0) / goals["balanced_accuracy"])),
        min(1.0, max(0.0, float(goal_report.get("mean_pr_auc_lift") or 0) / goals["pr_auc_lift"])),
        min(1.0, max(0.0, float(goal_report.get("mean_top_3pct_precision") or 0) / goals["top_3pct_precision"])),
    ]))
    atomic_json(goal_report, result_dir / "base_readiness_report.json")
    return {"metrics_rows": len(metrics), "paired_rows": len(compared), "summary_rows": len(summary), "goal": goal_report}


def _train_development_base_model(project: Path, result_dir: Path, profile: str) -> dict[str, Any]:
    """완성 모델이 아닌 개발용 base model을 저장한다.

    마지막 60 거래일은 calibration에만 쓰고 그 이전으로 XGBoost를 학습한다.
    """
    worker = Base12HWorker(
        project, None, result_dir=result_dir, cache_namespace="finance_base12h_shared_v1",
        deadline_epoch=time.time() + 7200, stop_flag=result_dir / "NO_STOP", worker_name="base_model",
        worker_index=0, worker_count=1,
    )
    worker.load()
    summary_path = result_dir / "base12h_ablation_summary.csv"
    drop_groups: list[str] = []
    gating_rules: list[dict[str, Any]] = []
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        global_rows = summary.loc[(summary["stage"] == "base12h_global") & (summary["scope_type"] == "all")]
        for _, row in global_rows.iterrows():
            if float(row.get("pr_auc_loss_mean", 0.0)) < -0.002 and float(row.get("pr_auc_loss_positive_fold_ratio", 0.0)) <= 0.375:
                drop_groups.append(str(row["group"]))
        bucket_rows = summary.loc[(summary["stage"] == "base12h_bucket") & (summary["scope_type"] == "target_bucket")]
        for _, row in bucket_rows.iterrows():
            gating_rules.append({
                "bucket": row.get("target_bucket"), "group": row.get("group"),
                "keep": bool(float(row.get("pr_auc_loss_mean", 0.0)) > 0.003 and float(row.get("pr_auc_loss_positive_fold_ratio", 0.0)) >= 0.625),
                "pr_auc_loss_mean": row.get("pr_auc_loss_mean"),
            })
    drop_features = set(worker.catalog.get("redundant_feature_cluster", []))
    for group in drop_groups:
        drop_features.update(worker.catalog.get(group, []))
    selected = [c for c in worker.features if c not in drop_features]
    indices = np.asarray([worker.feature_index[c] for c in selected], dtype=np.int32)
    unique_dates = pd.DatetimeIndex(np.unique(worker.dates)).sort_values()
    cal_dates = unique_dates[-60:]
    purge_dates = set(unique_dates[-80:-60])
    train_mask = np.asarray([(d not in set(cal_dates)) and (d not in purge_dates) for d in worker.dates], dtype=bool)
    cal_mask = np.isin(worker.dates, cal_dates.to_numpy())
    x_train = np.ascontiguousarray(worker.X[train_mask][:, indices])
    x_cal = np.ascontiguousarray(worker.X[cal_mask][:, indices])
    config = next(c for c in CANDIDATES if c.name == "lossguide64")
    backend = "cuda" if profile == "full" else "cpu"
    threads = 16 if profile == "full" else 4
    model = _strict_fit_xgb(x_train, worker.y[train_mask], 20260731, config, threads=threads, backend=backend)
    raw_cal = model.predict_proba(x_cal)[:, 1]
    policy = select_train_only_policy(worker.y[cal_mask], raw_cal, worker.dates[cal_mask])
    model_dir = result_dir / "development_base_model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "base_model_xgboost.json")
    atomic_csv(pd.DataFrame({"feature": selected, "feature_index": range(len(selected))}), model_dir / "feature_list.csv")
    atomic_json(policy.payload(), model_dir / "calibration_policy.json")
    atomic_json({
        "status": "development_base_not_final", "backend": backend, "threads": threads,
        "config": config.payload(), "dataset_signature": worker.dataset_signature,
        "train_rows": int(train_mask.sum()), "calibration_rows": int(cal_mask.sum()),
        "selected_feature_count": len(selected), "dropped_groups": drop_groups,
        "dropped_redundant_features": len(worker.catalog.get("redundant_feature_cluster", [])),
        "gating_rules": gating_rules,
    }, model_dir / "model_card.json")
    return {"model_dir": str(model_dir), "backend": backend, "selected_features": len(selected)}


def _desktop_path() -> Path:
    candidates = [
        Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Desktop",
        Path(os.environ.get("OneDrive", "")) / "Desktop" if os.environ.get("OneDrive") else Path("__missing__"),
        Path.home() / "Desktop",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return Path.home()


def create_result_package(result_dir: Path) -> Path:
    timestamp = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d_%H%M%S")
    destination = _desktop_path() / f"CrashWatch_Base12H_RESULTS_{timestamp}.zip"
    include_names = [
        "RESULT_BRIEF.md", "run_summary.json", "registry_status.json", "data_summary.json",
        "base_readiness_report.json", "base12h_ablation_summary.csv", "paired_ablation_deltas.csv",
        "backend_pairing_audit.csv", "calibration_by_fold.csv", "bottleneck_diagnostics.csv",
        "valid_feature_audit.csv", "redundant_features.csv", "task_manifest.csv",
        "resource_usage.csv", "development_base_model/model_card.json",
        "development_base_model/feature_list.csv", "development_base_model/calibration_policy.json",
    ]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in include_names:
            path = result_dir / name
            if path.exists() and path.is_file():
                zf.write(path, arcname=name)
    return destination


def _write_result_brief(result_dir: Path, run_summary: dict[str, Any]) -> None:
    goal = run_summary.get("compile", {}).get("goal", {})
    status = run_summary.get("registry", {})
    text = f"""# CrashWatch Base12H Result Brief

## 실행 상태
- 완료 block: {status.get('blocks_completed')} / {status.get('blocks_total')}
- 진행률: {status.get('progress_ratio')}
- 실행 프로필: {run_summary.get('profile')}
- 시간예산: {run_summary.get('hours')}시간
- RTX 5080 풀로드 workers: {run_summary.get('workers') if run_summary.get('profile') == 'full' else 0}
- backend-paired 설계: 각 fold×seed block의 baseline과 ablation이 동일 backend/threads 사용

## Baseline
- ROC-AUC: {goal.get('mean_roc_auc')}
- PR-AUC: {goal.get('mean_pr_auc')}
- PR-AUC lift: {goal.get('mean_pr_auc_lift')}
- Balanced accuracy: {goal.get('mean_balanced_accuracy')}
- Top 3% precision: {goal.get('mean_top_3pct_precision')}
- Brier skill: {goal.get('mean_brier_skill')}
- Base readiness score: {goal.get('base_readiness_score')}

## 우선 확인 파일
1. base12h_ablation_summary.csv
2. bottleneck_diagnostics.csv
3. calibration_by_fold.csv
4. backend_pairing_audit.csv
5. development_base_model/model_card.json

이 모델은 완성 모델이 아니라 다음 개발 단계를 결정하기 위한 base model이다.
"""
    (result_dir / "RESULT_BRIEF.md").write_text(text, encoding="utf-8")



def _read_nvidia_smi() -> dict[str, float]:
    query = [
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(query, text=True, stderr=subprocess.DEVNULL, timeout=5).strip().splitlines()
        if not output:
            return {}
        values = [float(x.strip()) for x in output[0].split(",")]
        return {
            "gpu_util_pct": values[0], "gpu_memory_used_mb": values[1], "gpu_memory_total_mb": values[2],
            "gpu_temp_c": values[3], "gpu_power_w": values[4],
        }
    except Exception:
        return {}


def _resource_monitor_loop(result_dir: Path, stop_event: threading.Event, stop_flag: Path) -> None:
    import psutil

    rows: list[dict[str, Any]] = []
    path = result_dir / "resource_usage.csv"
    while not stop_event.wait(10.0):
        vm = psutil.virtual_memory()
        row = {
            "timestamp": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
            "system_cpu_pct": psutil.cpu_percent(interval=None),
            "system_ram_used_gb": (vm.total - vm.available) / (1024 ** 3),
            "system_ram_available_gb": vm.available / (1024 ** 3),
            "process_count_python": sum(1 for p in psutil.process_iter(["name"]) if "python" in str(p.info.get("name", "")).lower()),
            **_read_nvidia_smi(),
        }
        rows.append(row)
        if len(rows) >= 6:
            existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
            atomic_csv(pd.concat([existing, pd.DataFrame(rows)], ignore_index=True), path)
            rows.clear()
        # 안전 한계. 풀로드라도 장시간 unattended 실행의 하드 리밋은 유지한다.
        if float(row.get("gpu_temp_c", 0.0)) >= 88.0 or float(row["system_ram_available_gb"]) < 2.0:
            stop_flag.write_text("resource hard limit\n", encoding="utf-8")
    if rows:
        existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
        atomic_csv(pd.concat([existing, pd.DataFrame(rows)], ignore_index=True), path)


def run_supervisor(
    project: Path,
    dataset: Path | None,
    *,
    profile: str,
    hours: float,
    workers: int,
    result_dir: Path | None = None,
) -> dict[str, Any]:
    project = project.resolve()
    paths = get_paths(project)
    result_dir = result_dir or paths.data_root / "ablation_base12h"
    result_dir.mkdir(parents=True, exist_ok=True)
    stop_flag = result_dir / "REQUEST_SAFE_STOP.flag"
    stop_flag.unlink(missing_ok=True)
    registry = TaskRegistry(result_dir / "task_registry.sqlite3")
    plan = json.loads((paths.configs / "base12h_plan.json").read_text(encoding="utf-8"))
    # Dataset signature를 block id에 포함해 데이터가 바뀌면 과거 완료 block을 잘못 재사용하지 않는다.
    preflight_worker = Base12HWorker(
        project, dataset, result_dir=result_dir, cache_namespace=plan["cache_namespace"],
        deadline_epoch=time.time() + 7200, stop_flag=result_dir / "PREFLIGHT_NO_STOP.flag",
        worker_name="preflight", worker_index=0, worker_count=1,
    )
    preflight_worker.load()
    dataset_signature = preflight_worker.dataset_signature
    registry.set_meta("schema_version", SCHEMA_VERSION)
    registry.set_meta("plan", plan)
    registry.set_meta("dataset_signature", dataset_signature)
    blocks = []
    # 각 seed의 모든 fold를 먼저 채워 시간예산이 짧아도 fold coverage를 확보한다.
    for seed_rank, seed in enumerate(plan["outer_seeds"]):
        for fold in range(int(plan["outer_folds"])):
            block_id = f"{dataset_signature[:10]}_fold{fold:02d}_seed{int(seed)}"
            priority = 10000 - seed_rank * 100 - fold
            blocks.append((block_id, fold, int(seed), priority))
    registry.ensure_blocks(blocks)
    registry.reset_stale_claims()

    deadline_epoch = time.time() + float(hours) * 3600
    if profile == "pubg":
        backend, default_threads, workers = "cpu", 4, 1
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        backend, default_threads = "cuda", max(1, 16 // max(1, workers))

    payloads = []
    for index in range(workers):
        payloads.append({
            "project": str(project), "dataset": str(dataset) if dataset else None,
            "result_dir": str(result_dir), "registry_path": str(result_dir / "task_registry.sqlite3"),
            "cache_namespace": plan["cache_namespace"], "deadline_epoch": deadline_epoch,
            "stop_flag": str(stop_flag), "worker_name": f"{profile}_{index}_{uuid.uuid4().hex[:5]}",
            "worker_index": index, "worker_count": workers, "profile": profile,
            "backend": backend, "threads": default_threads,
        })

    started = time.time()
    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(
        target=_resource_monitor_loop, args=(result_dir, monitor_stop, stop_flag), daemon=True
    )
    monitor_thread.start()
    processes: list[mp.Process] = []
    try:
        for payload in payloads:
            process = mp.Process(target=_worker_main, args=(payload,), daemon=False)
            process.start()
            processes.append(process)
        while any(p.is_alive() for p in processes):
            if time.time() >= deadline_epoch:
                stop_flag.write_text("deadline\n", encoding="utf-8")
            time.sleep(5)
        exit_codes = [p.exitcode for p in processes]
    except KeyboardInterrupt:
        stop_flag.write_text("keyboard interrupt\n", encoding="utf-8")
        for p in processes:
            p.join(timeout=300)
        exit_codes = [p.exitcode for p in processes]
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
        monitor_stop.set()
        monitor_thread.join(timeout=30)

    compile_result = _compile_summary(result_dir, plan, dataset_signature)
    model_result: dict[str, Any] = {}
    try:
        if registry.status()["blocks_completed"] >= 8:
            model_result = _train_development_base_model(project, result_dir, profile)
    except Exception:
        (result_dir / "base_model_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    status = registry.status()
    atomic_json(status, result_dir / "registry_status.json")
    run_summary = {
        "schema_version": SCHEMA_VERSION, "dataset_signature": dataset_signature,
        "profile": profile, "hours": hours, "workers": workers,
        "default_backend": backend, "default_threads_per_worker": default_threads,
        "elapsed_hours": (time.time() - started) / 3600, "exit_codes": exit_codes,
        "safe_stop": stop_flag.exists(), "registry": status, "compile": compile_result,
        "development_base_model": model_result, "result_dir": str(result_dir),
        "host": socket.gethostname(), "finished_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
    }
    atomic_json(run_summary, result_dir / "run_summary.json")
    _write_result_brief(result_dir, run_summary)
    try:
        package = create_result_package(result_dir)
        run_summary["result_package"] = str(package)
        atomic_json(run_summary, result_dir / "run_summary.json")
    except Exception:
        (result_dir / "result_package_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return run_summary
