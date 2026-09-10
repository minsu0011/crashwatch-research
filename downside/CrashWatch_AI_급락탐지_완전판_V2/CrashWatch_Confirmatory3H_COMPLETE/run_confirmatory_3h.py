from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import datetime as dt
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

TARGET_DEFAULT = "label_abs_crash_20"
SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SCRIPT_DIR / "reference"
_WORKER: dict[str, Any] = {}




def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{hashlib.sha256(path.name.encode('utf-8')).hexdigest()[:6]}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{hashlib.sha256(path.name.encode('utf-8')).hexdigest()[:6]}.{os.getpid()}.tmp"
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_csv_schema(frame: pd.DataFrame, path: Path, columns: Sequence[str] = ()) -> None:
    """Write a CSV atomically and keep empty outputs parseable."""
    output = frame
    if output.empty and len(output.columns) == 0:
        output = pd.DataFrame(columns=list(columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{hashlib.sha256(path.name.encode('utf-8')).hexdigest()[:6]}.{os.getpid()}.tmp"
    output.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def hash_text(text: str, length: int = 24) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def hash_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def normalize_ticker(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value)).zfill(6)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value)).zfill(6)
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def day_number(value: Any) -> int:
    return int(pd.Timestamp(value).normalize().value // 86_400_000_000_000)


def day_string(value: int) -> str:
    return str(pd.Timestamp(int(value), unit="D").date())


def clip_prob(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), 1e-7, 1 - 1e-7)


def nan_or_float(value: Any) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else float("nan")
    except Exception:
        return float("nan")


def safe_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else float("nan")


def ensure_dependencies() -> None:
    missing: list[str] = []
    for module, package in [
        ("lightgbm", "lightgbm>=4.3"),
        ("sklearn", "scikit-learn>=1.4"),
        ("psutil", "psutil>=5.9"),
        ("scipy", "scipy>=1.11"),
    ]:
        try:
            __import__(module)
        except Exception:
            missing.append(package)
    if missing:
        raise RuntimeError("필수 패키지가 없습니다: " + ", ".join(missing) + "\npython -m pip install -r requirements.txt")




def default_config() -> dict[str, Any]:
    return {
        "schema_version": "confirmatory3h_v2",
        "dataset_path": "",
        "results_dir": "",
        "target_column": TARGET_DEFAULT,
        "date_column": "",
        "ticker_column": "",
        "bucket_column": "",
        "max_hours": 3.0,
        "soft_stop_minutes_before_deadline": 10,
        "finalization_reserve_minutes": 8,
        "outer_folds": 8,
        "validation_days": 60,
        "purge_days": 20,
        "min_train_days": 500,
        "fold_strategy": "reference_or_generated",
        "inner_validation_days": 60,
        "inner_purge_days": 20,
        "seeds_core": [17, 43, 101, 211, 503],
        "seeds_optional": [17, 101, 503],
        "cpu_workers": 4,
        "cpu_workers_with_gpu": 3,
        "threads_per_cpu_worker": 4,
        "gpu_threads": 2,
        "enable_gpu_sensitivity": True,
        "gpu_min_remaining_minutes": 25,
        "canary_fold": 7,
        "canary_profiles": ["common_period", "full_reduced"],
        "strict_reference_features": True,
        "allow_generated_folds": True,
        "full_dataset_sha256": True,
        "save_predictions": True,
        "save_models": False,
        "max_task_attempts": 2,
        "low_ram_warning_gb": 6.0,
        "critical_ram_stop_gb": 3.5,
        "critical_ram_consecutive_samples": 3,
        "cpu_temp_stop_c": 93.0,
        "resource_sample_seconds": 5,
        "optional_priority": ["battery", "etf"],
        "lightgbm": {
            "learning_rate": 0.028,
            "num_leaves": 127,
            "max_depth": -1,
            "min_child_samples": 55,
            "subsample": 1.0,
            "colsample_bytree": 0.72,
            "reg_alpha": 0.25,
            "reg_lambda": 1.8,
            "max_bin": 255,
            "max_estimators": 1200,
            "early_stopping_rounds": 80,
            "fallback_estimators": 350,
        },
        "xgboost_gpu": {
            "n_estimators": 700,
            "learning_rate": 0.025,
            "max_depth": 0,
            "max_leaves": 96,
            "grow_policy": "lossguide",
            "subsample": 0.90,
            "colsample_bytree": 0.72,
            "min_child_weight": 7.0,
            "reg_alpha": 0.20,
            "reg_lambda": 1.8,
            "max_bin": 512,
            "folds": [4, 5, 6, 7],
            "seeds": [17, 503],
            "conditions": ["B0", "A1", "A2", "A3"],
        },
        "pass_thresholds": {
            "main_mean_loss": 0.005,
            "main_median_loss": 0.0,
            "main_positive_folds": 6,
            "main_recent_positive_folds": 3,
            "main_worst_fold": -0.005,
            "joint_incremental_mean": 0.003,
            "etf_incremental_mean": 0.001,
            "battery_mean_loss": 0.010,
        },
    }


def load_config(path: Path | None) -> dict[str, Any]:
    cfg = default_config()
    if path and path.exists():
        user_cfg = json.loads(path.read_text(encoding="utf-8"))

        def merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
            for key, value in incoming.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    merge(base[key], value)
                else:
                    base[key] = value
            return base

        merge(cfg, user_cfg)
    return cfg


def resolve_dataset(config: dict[str, Any], cli_path: str | None) -> Path:
    candidates: list[Path] = []
    for raw in [cli_path, os.environ.get("CRASHWATCH_DATASET"), config.get("dataset_path")]:
        if raw:
            candidates.append(Path(str(raw)).expanduser())

    expected = "training_dataset_finance11h.parquet"
    expected_csv = "training_dataset_finance11h.csv"
    roots = [SCRIPT_DIR, Path.cwd(), SCRIPT_DIR.parent]
    roots.extend(list(SCRIPT_DIR.parents)[:3])
    for root in roots:
        candidates.extend([
            root / expected,
            root / expected_csv,
            root / "development" / expected,
            root / "crashwatch_ai_data" / "development" / expected,
            root / "CrashWatch_AI_V4_FocusedNested_DualMode" / "crashwatch_ai_data" / "development" / expected,
        ])

    old_path = Path(r"C:\Users\minsu\Downloads\CrashWatch_AI_급락탐지_완전판_V2\CrashWatch_AI_급락탐지_완전판_V2\CrashWatch_AI_V4_FocusedNested_DualMode\crashwatch_ai_data\development\training_dataset_finance11h.parquet")
    candidates.append(old_path)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return candidate.resolve()

    for root in [SCRIPT_DIR, Path.cwd()]:
        try:
            for name in (expected, expected_csv):
                found = next(root.rglob(name), None)
                if found and found.is_file():
                    return found.resolve()
        except Exception:
            pass

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        selected = filedialog.askopenfilename(
            title="CrashWatch 학습 데이터 선택",
            filetypes=[("Parquet/CSV", "*.parquet *.csv"), ("All files", "*.*")],
        )
        root.destroy()
        if selected:
            return Path(selected).resolve()
    except Exception:
        pass

    raise FileNotFoundError(
        "training_dataset_finance11h.parquet를 찾지 못했습니다. config.json의 dataset_path를 지정하거나 "
        "CRASHWATCH_DATASET 환경변수를 설정하세요."
    )


def resolve_results_dir(config: dict[str, Any], dataset_path: Path, cli_path: str | None) -> Path:
    raw = cli_path or config.get("results_dir")
    if raw:
        result = Path(str(raw)).expanduser().resolve()
    else:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if dataset_path.parent.name.lower() == "development":
            result = dataset_path.parent.parent / f"confirmatory3h_{stamp}"
        else:
            result = SCRIPT_DIR / f"results_confirmatory3h_{stamp}"
    result.mkdir(parents=True, exist_ok=True)
    return result



class RunLock:
    def __init__(self, result_dir: Path):
        self.path = result_dir / ".run.lock"
        self.acquired = False

    def __enter__(self) -> "RunLock":
        if self.path.exists():
            try:
                old = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(old.get("pid", -1))
                import psutil
                if pid > 0 and psutil.pid_exists(pid):
                    raise RuntimeError(f"같은 결과 폴더에서 이미 실행 중입니다. PID={pid}")
            except RuntimeError:
                raise
            except Exception:
                pass
            self.path.unlink(missing_ok=True)
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        fd = os.open(self.path, flags)
        os.write(fd, json.dumps({"pid": os.getpid(), "started_at": now_iso()}).encode("utf-8"))
        os.close(fd)
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)



def parquet_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise RuntimeError("Parquet 입력에는 pyarrow가 필요합니다. python -m pip install pyarrow") from exc
    return list(pq.ParquetFile(path).schema.names)


def source_columns(path: Path) -> list[str]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return parquet_columns(path)
    return pd.read_csv(path, nrows=0).columns.tolist()


def read_source(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=list(columns))
    return pd.read_csv(path, usecols=list(columns), low_memory=False)


def detect_column(columns: Sequence[str], explicit: str, candidates: Sequence[str], role: str) -> str:
    if explicit:
        if explicit not in columns:
            raise KeyError(f"지정한 {role} 열이 없습니다: {explicit}")
        return explicit
    lowered = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise KeyError(f"{role} 열을 자동 탐지하지 못했습니다. 후보={list(candidates)}")


def load_reference_data() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, str]]:
    feature_quality = pd.read_csv(REFERENCE_DIR / "feature_quality_refine.csv")
    group_audit = pd.read_csv(REFERENCE_DIR / "valid_feature_audit.csv")
    folds = json.loads((REFERENCE_DIR / "outer_walk_forward_folds.json").read_text(encoding="utf-8"))
    ticker_map_df = pd.read_csv(REFERENCE_DIR / "ticker_bucket_map.csv", dtype={"ticker": str})
    ticker_map = {normalize_ticker(row.ticker): str(row.bucket) for row in ticker_map_df.itertuples(index=False)}
    return feature_quality, group_audit, folds, ticker_map


def dataset_signature(path: Path, config: dict[str, Any]) -> str:
    if config.get("full_dataset_sha256", True):
        return hash_file(path)[:24]
    stat = path.stat()
    return hash_text(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}")


def prepare_cache(dataset_path: Path, result_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    feature_quality, group_audit, _, ticker_bucket_map = load_reference_data()
    columns = source_columns(dataset_path)
    target_col = detect_column(columns, config.get("target_column", ""), [TARGET_DEFAULT, "target", "label"], "target")
    date_col = detect_column(columns, config.get("date_column", ""), ["date", "trade_date", "trading_date", "datetime", "일자"], "date")
    ticker_col = detect_column(columns, config.get("ticker_column", ""), ["ticker", "code", "stock_code", "symbol", "종목코드"], "ticker")

    bucket_col = ""
    explicit_bucket = config.get("bucket_column", "")
    if explicit_bucket:
        if explicit_bucket not in columns:
            raise KeyError(f"지정한 bucket 열이 없습니다: {explicit_bucket}")
        bucket_col = explicit_bucket
    else:
        for candidate in ["bucket", "sector_bucket", "industry_bucket", "sector", "industry_group"]:
            if candidate in columns:
                bucket_col = candidate
                break

    full_mask = as_bool_series(feature_quality["selected_full_reduced"])
    common_mask = as_bool_series(feature_quality["selected_common_period"])
    full_reference = feature_quality.loc[full_mask, "feature"].astype(str).tolist()
    common_reference = feature_quality.loc[common_mask, "feature"].astype(str).tolist()
    missing_full = [f for f in full_reference if f not in columns]
    if missing_full and config.get("strict_reference_features", True):
        raise RuntimeError(
            f"reference full_reduced 피처 {len(missing_full)}개가 데이터에 없습니다. 예: {missing_full[:10]}\n"
            "다른 데이터 버전이면 strict_reference_features=false로 명시적으로 변경하세요."
        )
    full_features = [f for f in full_reference if f in columns]
    common_features = [f for f in common_reference if f in full_features]
    if not full_features:
        raise RuntimeError("사용 가능한 reference 피처가 0개입니다.")

    sig = dataset_signature(dataset_path, config)
    feature_hash = hash_text("\n".join(full_features), 24)
    cache_dir = result_dir / "matrix_cache" / f"{sig}_{feature_hash}"
    meta_path = cache_dir / "cache_meta.json"
    expected_meta = {
        "dataset_signature": sig,
        "feature_hash": feature_hash,
        "dataset_path": str(dataset_path),
        "target_column": target_col,
        "date_column": date_col,
        "ticker_column": ticker_col,
        "bucket_column": bucket_col,
        "full_features": full_features,
        "common_features": common_features,
    }
    required_files = ["X_full.npy", "y.npy", "date_days.npy", "ticker_codes.npy", "bucket_codes.npy", "cache_meta.json"]
    if meta_path.exists() and all((cache_dir / name).exists() for name in required_files):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if all(meta.get(k) == expected_meta.get(k) for k in ["dataset_signature", "feature_hash", "target_column", "date_column", "ticker_column"]):
            print(f"[CACHE] 기존 matrix cache 재사용: {cache_dir}")
            return meta

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[DATA] 입력 로드: {dataset_path}")
    read_cols = [date_col, ticker_col, target_col] + ([bucket_col] if bucket_col else []) + full_features
    read_cols = list(dict.fromkeys(read_cols))
    df = read_source(dataset_path, read_cols)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df[df[date_col].notna() & df[target_col].notna()].copy()
    df[ticker_col] = df[ticker_col].map(normalize_ticker)
    df = df[df[ticker_col].ne("")]
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df[df[target_col].isin([0, 1])]
    df.sort_values([date_col, ticker_col], kind="mergesort", inplace=True)
    df.reset_index(drop=True, inplace=True)

    tickers = sorted(df[ticker_col].dropna().unique().tolist())
    ticker_to_code = {ticker: idx for idx, ticker in enumerate(tickers)}
    if bucket_col:
        buckets_raw = df[bucket_col].astype(str).str.strip().str.lower().replace({"nan": "unknown", "": "unknown"})
    else:
        buckets_raw = df[ticker_col].map(ticker_bucket_map).fillna("unknown")
    buckets = sorted(buckets_raw.unique().tolist())
    bucket_to_code = {bucket: idx for idx, bucket in enumerate(buckets)}

    rows = len(df)
    matrix_path = cache_dir / "X_full.npy"
    X = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(rows, len(full_features)))
    chunk_size = 32
    for start in range(0, len(full_features), chunk_size):
        names = full_features[start:start + chunk_size]
        chunk = df[names].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, copy=True)
        chunk[~np.isfinite(chunk)] = np.nan
        X[:, start:start + len(names)] = chunk
        del chunk
    X.flush()
    del X

    np.save(cache_dir / "y.npy", df[target_col].to_numpy(dtype=np.int8))
    np.save(cache_dir / "date_days.npy", (df[date_col].astype("int64") // 86_400_000_000_000).to_numpy(dtype=np.int32))
    np.save(cache_dir / "ticker_codes.npy", df[ticker_col].map(ticker_to_code).to_numpy(dtype=np.int16))
    np.save(cache_dir / "bucket_codes.npy", buckets_raw.map(bucket_to_code).to_numpy(dtype=np.int16))

    group_map: dict[str, list[str]] = {}
    for group, part in group_audit[group_audit["status"].eq("valid")].groupby("group", dropna=True):
        group_map[str(group)] = [f for f in part["feature"].astype(str).tolist() if f in full_features]

    meta = {
        **expected_meta,
        "created_at": now_iso(),
        "rows": rows,
        "tickers": tickers,
        "ticker_to_code": ticker_to_code,
        "buckets": buckets,
        "bucket_to_code": bucket_to_code,
        "group_map": group_map,
        "date_min": str(df[date_col].min().date()),
        "date_max": str(df[date_col].max().date()),
        "positive_rate": float(df[target_col].mean()),
        "missing_reference_features": missing_full,
        "full_feature_count": len(full_features),
        "common_feature_count": len(common_features),
        "paths": {
            "X": str(matrix_path),
            "y": str(cache_dir / "y.npy"),
            "date_days": str(cache_dir / "date_days.npy"),
            "ticker_codes": str(cache_dir / "ticker_codes.npy"),
            "bucket_codes": str(cache_dir / "bucket_codes.npy"),
        },
    }
    atomic_json(meta_path, meta)
    del df
    gc.collect()
    return meta



def build_folds(cache_meta: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    _, _, reference_folds, _ = load_reference_data()
    dates = np.load(cache_meta["paths"]["date_days"], mmap_mode="r")
    unique_dates = np.unique(dates)
    date_set = set(int(x) for x in unique_dates)
    min_train = int(config["min_train_days"])
    expected_val = int(config["validation_days"])

    def validate_reference() -> bool:
        if len(reference_folds) != int(config["outer_folds"]):
            return False
        for fold in reference_folds:
            v_start, v_end = day_number(fold["validation_start"]), day_number(fold["validation_end"])
            t_end = day_number(fold["train_end"])
            val_count = int(np.sum((unique_dates >= v_start) & (unique_dates <= v_end)))
            train_count = int(np.sum(unique_dates <= t_end))
            if v_start not in date_set or v_end not in date_set or val_count != expected_val or train_count < min_train:
                return False
        return True

    strategy = str(config.get("fold_strategy", "reference_or_generated"))
    if strategy in {"reference", "reference_strict", "reference_or_generated"} and validate_reference():
        output = []
        for f in reference_folds:
            output.append({**f, "source": "reference", "train_start_day": day_number(f["train_start"]), "train_end_day": day_number(f["train_end"]), "validation_start_day": day_number(f["validation_start"]), "validation_end_day": day_number(f["validation_end"])})
        return output
    if strategy in {"reference", "reference_strict"} or not config.get("allow_generated_folds", True):
        raise RuntimeError("reference outer fold 날짜가 현재 데이터와 일치하지 않습니다.")

    folds = int(config["outer_folds"])
    val_days = int(config["validation_days"])
    purge_days = int(config["purge_days"])
    earliest_start = min_train + purge_days
    latest_start = len(unique_dates) - val_days
    if latest_start < earliest_start:
        raise RuntimeError("데이터 기간이 outer walk-forward 설정보다 짧습니다.")
    starts = np.linspace(earliest_start, latest_start, folds).round().astype(int)
    starts = np.maximum.accumulate(starts)
    output = []
    for fold_id, v_idx in enumerate(starts):
        train_end_idx = int(v_idx - purge_days - 1)
        val_end_idx = int(v_idx + val_days - 1)
        if train_end_idx + 1 < min_train or val_end_idx >= len(unique_dates):
            raise RuntimeError(f"생성 fold {fold_id}가 유효하지 않습니다.")
        output.append({
            "fold_id": fold_id,
            "source": "generated",
            "train_start": day_string(unique_dates[0]),
            "train_end": day_string(unique_dates[train_end_idx]),
            "purge_start": day_string(unique_dates[train_end_idx + 1]),
            "purge_end": day_string(unique_dates[v_idx - 1]),
            "validation_start": day_string(unique_dates[v_idx]),
            "validation_end": day_string(unique_dates[val_end_idx]),
            "train_dates": train_end_idx + 1,
            "validation_dates": val_days,
            "train_start_day": int(unique_dates[0]),
            "train_end_day": int(unique_dates[train_end_idx]),
            "validation_start_day": int(unique_dates[v_idx]),
            "validation_end_day": int(unique_dates[val_end_idx]),
        })
    return output


def condition_specs(cache_meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    groups = cache_meta["group_map"]

    def union(*names: str) -> list[str]:
        values: list[str] = []
        for name in names:
            values.extend(groups.get(name, []))
        return list(dict.fromkeys(values))

    return {
        "B0": {"label": "baseline", "mode": "baseline", "features": []},
        "A1": {"label": "drop_shorting", "mode": "global_drop", "features": union("u_financial_shorting")},
        "A2": {"label": "drop_market", "mode": "global_drop", "features": union("u_financial_market")},
        "A3": {"label": "drop_shorting_market", "mode": "global_drop", "features": union("u_financial_shorting", "u_financial_market")},
        "A4": {"label": "drop_etf", "mode": "global_drop", "features": union("u_etf_pressure")},
        "A5": {"label": "drop_market_etf", "mode": "global_drop", "features": union("u_financial_market", "u_etf_pressure")},
        "A6": {"label": "drop_shorting_market_etf", "mode": "global_drop", "features": union("u_financial_shorting", "u_financial_market", "u_etf_pressure")},
        "S1": {"label": "battery_mask_shorting", "mode": "sector_mask", "target_bucket": "battery_materials", "features": union("t_financial_shorting")},
        "S2": {"label": "battery_mask_shorting_lending", "mode": "sector_mask", "target_bucket": "battery_materials", "features": union("t_financial_shorting", "t_stock_lending")},
        "S3": {"label": "battery_mask_shorting_interaction", "mode": "sector_mask", "target_bucket": "battery_materials", "features": union("t_financial_shorting", "t_financial_interaction")},
    }



def registry_connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, timeout=60, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            phase TEXT,
            family TEXT,
            profile TEXT,
            fold_id INTEGER,
            seed INTEGER,
            condition_id TEXT,
            status TEXT,
            attempts INTEGER DEFAULT 0,
            started_at TEXT,
            ended_at TEXT,
            elapsed_seconds REAL,
            error TEXT,
            metrics_path TEXT,
            prediction_path TEXT,
            config_hash TEXT,
            dataset_signature TEXT
        )
        """
    )
    return conn


def reset_stale_tasks(registry_path: Path) -> None:
    conn = registry_connect(registry_path)
    conn.execute("UPDATE tasks SET status='pending', error='recovered_from_previous_process' WHERE status='running'")
    conn.close()


def claim_task(registry_path: Path, task: dict[str, Any], max_attempts: int) -> str:
    conn = registry_connect(registry_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT status, attempts, config_hash, metrics_path FROM tasks WHERE task_id=?", (task["task_id"],)).fetchone()
        if row:
            status, attempts, config_hash, metrics_path = row
            if status == "completed" and config_hash == task["config_hash"] and metrics_path and Path(metrics_path).exists():
                conn.execute("COMMIT")
                return "completed"
            if status == "running":
                conn.execute("COMMIT")
                return "running"
            if status == "failed" and int(attempts or 0) >= max_attempts:
                conn.execute("COMMIT")
                return "max_attempts"
            new_attempts = int(attempts or 0) + 1
            conn.execute(
                "UPDATE tasks SET status='running', attempts=?, started_at=?, ended_at=NULL, error=NULL, config_hash=? WHERE task_id=?",
                (new_attempts, now_iso(), task["config_hash"], task["task_id"]),
            )
        else:
            conn.execute(
                "INSERT INTO tasks(task_id, phase, family, profile, fold_id, seed, condition_id, status, attempts, started_at, config_hash, dataset_signature) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (task["task_id"], task["phase"], task["family"], task["profile"], task["fold_id"], task["seed"], task["condition_id"], "running", 1, now_iso(), task["config_hash"], task["dataset_signature"]),
            )
        conn.execute("COMMIT")
        return "claimed"
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def finish_task(registry_path: Path, task_id: str, status: str, elapsed: float, *, error: str = "", metrics_path: str = "", prediction_path: str = "") -> None:
    conn = registry_connect(registry_path)
    conn.execute(
        "UPDATE tasks SET status=?, ended_at=?, elapsed_seconds=?, error=?, metrics_path=?, prediction_path=? WHERE task_id=?",
        (status, now_iso(), elapsed, error[:10000], metrics_path, prediction_path, task_id),
    )
    conn.close()



def scale_pos_weight(y: np.ndarray) -> float:
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    return max(1.0, negatives / max(positives, 1))


def compute_metrics(y: np.ndarray, pred: np.ndarray, date_days: np.ndarray, threshold: float = 0.5) -> dict[str, float | int]:
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

    y = np.asarray(y, dtype=np.int8)
    pred = clip_prob(pred)
    rows = len(y)
    positives = int(y.sum())
    positive_rate = positives / rows if rows else float("nan")
    binary = (pred >= threshold).astype(np.int8)
    two_classes = len(np.unique(y)) == 2
    pr_auc = float(average_precision_score(y, pred)) if positives > 0 else float("nan")
    roc_auc = float(roc_auc_score(y, pred)) if two_classes else float("nan")
    brier = float(brier_score_loss(y, pred)) if rows else float("nan")
    logloss = float(log_loss(y, pred, labels=[0, 1])) if rows else float("nan")
    accuracy = float(accuracy_score(y, binary)) if rows else float("nan")
    balanced = float(balanced_accuracy_score(y, binary)) if two_classes else float("nan")
    precision = float(precision_score(y, binary, zero_division=0)) if rows else float("nan")
    recall = float(recall_score(y, binary, zero_division=0)) if rows else float("nan")
    f1 = float(f1_score(y, binary, zero_division=0)) if rows else float("nan")

    def daily_top(fraction: float) -> tuple[float, float]:
        selected: list[int] = []
        for date in np.unique(date_days):
            idx = np.flatnonzero(date_days == date)
            if len(idx) == 0:
                continue
            k = max(1, int(math.ceil(len(idx) * fraction)))
            local = idx[np.argpartition(pred[idx], -k)[-k:]]
            selected.extend(local.tolist())
        if not selected:
            return float("nan"), float("nan")
        selected_arr = np.asarray(selected, dtype=int)
        tp = int(y[selected_arr].sum())
        return tp / len(selected_arr), tp / positives if positives else float("nan")

    top3_p, top3_r = daily_top(0.03)
    baseline_brier = positive_rate * (1 - positive_rate) if 0 < positive_rate < 1 else float("nan")
    base_prob = np.full(rows, positive_rate if math.isfinite(positive_rate) else 0.5)
    baseline_logloss = float(log_loss(y, clip_prob(base_prob), labels=[0, 1])) if rows else float("nan")
    return {
        "rows": rows,
        "positives": positives,
        "positive_rate": positive_rate,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "raw_pr_auc": pr_auc,
        "raw_roc_auc": roc_auc,
        "raw_brier": brier,
        "raw_logloss": logloss,
        "raw_mean_prediction": float(np.mean(pred)) if rows else float("nan"),
        "raw_top_3pct_precision": top3_p,
        "raw_top_3pct_recall": top3_r,
        "raw_pr_auc_lift": pr_auc / positive_rate if positive_rate and math.isfinite(pr_auc) else float("nan"),
        "brier_skill": 1 - brier / baseline_brier if baseline_brier and math.isfinite(brier) else float("nan"),
        "logloss_skill": 1 - logloss / baseline_logloss if baseline_logloss and math.isfinite(logloss) else float("nan"),
    }


def fit_lightgbm(x_train: np.ndarray, y_train: np.ndarray, seed: int, threads: int, config: dict[str, Any], n_estimators: int, eval_set: tuple[np.ndarray, np.ndarray] | None = None):
    from lightgbm import LGBMClassifier, early_stopping, log_evaluation

    params = dict(
        objective="binary",
        learning_rate=float(config["learning_rate"]),
        num_leaves=int(config["num_leaves"]),
        max_depth=int(config["max_depth"]),
        min_child_samples=int(config["min_child_samples"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_alpha=float(config["reg_alpha"]),
        reg_lambda=float(config["reg_lambda"]),
        max_bin=int(config["max_bin"]),
        n_estimators=int(n_estimators),
        random_state=int(seed),
        bagging_seed=int(seed),
        feature_fraction_seed=int(seed),
        data_random_seed=int(seed),
        n_jobs=int(threads),
        scale_pos_weight=scale_pos_weight(y_train),
        device_type="cpu",
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model = LGBMClassifier(**params)
    if eval_set is None:
        model.fit(x_train, y_train)
    else:
        x_eval, y_eval = eval_set
        callbacks = [log_evaluation(0), early_stopping(int(config["early_stopping_rounds"]), first_metric_only=True, verbose=False)]
        try:
            model.fit(x_train, y_train, eval_set=[(x_eval, y_eval)], eval_metric="average_precision", callbacks=callbacks)
        except Exception:
            model.fit(x_train, y_train, eval_set=[(x_eval, y_eval)], eval_metric="auc", callbacks=callbacks)
    return model


def fit_xgboost_gpu(x_train: np.ndarray, y_train: np.ndarray, seed: int, threads: int, config: dict[str, Any]):
    import xgboost as xgb
    from xgboost import XGBClassifier

    version_major = int(str(xgb.__version__).split(".")[0])
    params = dict(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=int(seed),
        n_jobs=int(threads),
        scale_pos_weight=scale_pos_weight(y_train),
        verbosity=0,
        validate_parameters=True,
        **{k: v for k, v in config.items() if k not in {"folds", "seeds", "conditions"}},
    )
    if version_major >= 2:
        params.update(tree_method="hist", device="cuda")
    else:
        params.update(tree_method="gpu_hist", predictor="gpu_predictor")
    model = XGBClassifier(**params)
    model.fit(x_train, y_train, verbose=False)
    return model



def worker_init(context_path: str) -> None:
    global _WORKER
    context = json.loads(Path(context_path).read_text(encoding="utf-8"))
    os.environ["OMP_NUM_THREADS"] = str(context["config"]["threads_per_cpu_worker"])
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    meta = context["cache_meta"]
    _WORKER = {
        "context": context,
        "X": np.load(meta["paths"]["X"], mmap_mode="r"),
        "y": np.load(meta["paths"]["y"], mmap_mode="r"),
        "date_days": np.load(meta["paths"]["date_days"], mmap_mode="r"),
        "ticker_codes": np.load(meta["paths"]["ticker_codes"], mmap_mode="r"),
        "bucket_codes": np.load(meta["paths"]["bucket_codes"], mmap_mode="r"),
        "feature_to_idx": {name: idx for idx, name in enumerate(meta["full_features"])},
        "ticker_values": meta["tickers"],
        "bucket_values": meta["buckets"],
    }


def clear_worker() -> None:
    global _WORKER
    _WORKER = {}
    gc.collect()


def should_stop() -> tuple[bool, str]:
    context = _WORKER["context"]
    if time.time() >= float(context["deadline_epoch"]):
        return True, "deadline"
    stop_file = Path(context["stop_file"])
    if stop_file.exists():
        return True, "stop_file"
    return False, ""


def fold_indices(fold_id: int) -> tuple[np.ndarray, np.ndarray]:
    context = _WORKER["context"]
    fold = next(f for f in context["folds"] if int(f["fold_id"]) == int(fold_id))
    dates = _WORKER["date_days"]
    train_idx = np.flatnonzero(dates <= int(fold["train_end_day"]))
    val_idx = np.flatnonzero((dates >= int(fold["validation_start_day"])) & (dates <= int(fold["validation_end_day"])))
    return train_idx.astype(np.int64), val_idx.astype(np.int64)


def profile_indices(profile: str) -> list[int]:
    meta = _WORKER["context"]["cache_meta"]
    names = meta["full_features"] if profile == "full_reduced" else meta["common_features"]
    mapping = _WORKER["feature_to_idx"]
    return [mapping[name] for name in names]


def tuning_path(profile: str, fold_id: int) -> Path:
    return Path(_WORKER["context"]["result_dir"]) / "tuning" / f"{profile}_fold{fold_id}.json"


def tune_best_iteration(profile: str, fold_id: int) -> int:
    path = tuning_path(profile, fold_id)
    if path.exists():
        try:
            return int(json.loads(path.read_text(encoding="utf-8"))["best_iteration"])
        except Exception:
            pass
    config = _WORKER["context"]["config"]
    train_idx, _ = fold_indices(fold_id)
    dates = _WORKER["date_days"]
    train_dates = np.unique(dates[train_idx])
    inner_val_days = int(config["inner_validation_days"])
    inner_purge = int(config["inner_purge_days"])
    if len(train_dates) <= inner_val_days + inner_purge + 50:
        best = int(config["lightgbm"]["fallback_estimators"])
        atomic_json(path, {"best_iteration": best, "reason": "insufficient_inner_dates", "created_at": now_iso()})
        return best
    inner_val_start = int(train_dates[-inner_val_days])
    inner_train_end = int(train_dates[-inner_val_days - inner_purge - 1])
    inner_train_idx = train_idx[dates[train_idx] <= inner_train_end]
    inner_val_idx = train_idx[dates[train_idx] >= inner_val_start]
    selected = profile_indices(profile)
    X = _WORKER["X"]
    y = _WORKER["y"]
    x_train = np.asarray(X[np.ix_(inner_train_idx, selected)], dtype=np.float32)
    x_val = np.asarray(X[np.ix_(inner_val_idx, selected)], dtype=np.float32)
    y_train = np.asarray(y[inner_train_idx], dtype=np.int8)
    y_val = np.asarray(y[inner_val_idx], dtype=np.int8)
    model = fit_lightgbm(
        x_train,
        y_train,
        seed=17,
        threads=int(config["threads_per_cpu_worker"]),
        config=config["lightgbm"],
        n_estimators=int(config["lightgbm"]["max_estimators"]),
        eval_set=(x_val, y_val),
    )
    best = int(getattr(model, "best_iteration_", 0) or config["lightgbm"]["fallback_estimators"])
    best = max(25, min(best, int(config["lightgbm"]["max_estimators"])))
    atomic_json(path, {
        "profile": profile,
        "fold_id": fold_id,
        "best_iteration": best,
        "inner_train_rows": len(inner_train_idx),
        "inner_validation_rows": len(inner_val_idx),
        "inner_train_end": day_string(inner_train_end),
        "inner_validation_start": day_string(inner_val_start),
        "created_at": now_iso(),
    })
    del model, x_train, x_val, y_train, y_val
    gc.collect()
    return best


def make_task(family: str, phase: str, profile: str, fold_id: int, seed: int, condition_id: str, model_cfg: dict[str, Any], best_iteration: int | None) -> dict[str, Any]:
    context = _WORKER["context"]
    specs = context["conditions"]
    payload = {
        "dataset_signature": context["cache_meta"]["dataset_signature"],
        "feature_hash": context["cache_meta"]["feature_hash"],
        "family": family,
        "phase": phase,
        "profile": profile,
        "fold_id": fold_id,
        "seed": seed,
        "condition_id": condition_id,
        "condition": specs[condition_id],
        "model_config": model_cfg,
        "best_iteration": best_iteration,
    }
    config_hash = hash_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), 32)
    return {
        **payload,
        "task_id": hash_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), 24),
        "config_hash": config_hash,
    }


def build_task_matrices(profile: str, fold_id: int, condition_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    context = _WORKER["context"]
    spec = context["conditions"][condition_id]
    train_idx, val_idx = fold_indices(fold_id)
    selected = profile_indices(profile)
    feature_to_idx = _WORKER["feature_to_idx"]
    affected = {feature_to_idx[f] for f in spec.get("features", []) if f in feature_to_idx}
    if spec["mode"] == "global_drop":
        selected = [idx for idx in selected if idx not in affected]
    X = _WORKER["X"]
    y = _WORKER["y"]
    x_train = np.asarray(X[np.ix_(train_idx, selected)], dtype=np.float32)
    x_val = np.asarray(X[np.ix_(val_idx, selected)], dtype=np.float32)
    if spec["mode"] == "sector_mask":
        target_bucket = spec["target_bucket"]
        bucket_code = context["cache_meta"]["bucket_to_code"].get(target_bucket)
        if bucket_code is None:
            raise RuntimeError(f"target bucket이 데이터에 없습니다: {target_bucket}")
        positions = [pos for pos, full_idx in enumerate(selected) if full_idx in affected]
        if not positions:
            raise RuntimeError(f"{condition_id}에 마스킹할 피처가 현재 profile에 없습니다.")
        train_rows = np.flatnonzero(_WORKER["bucket_codes"][train_idx] == bucket_code)
        val_rows = np.flatnonzero(_WORKER["bucket_codes"][val_idx] == bucket_code)
        if len(train_rows):
            x_train[np.ix_(train_rows, positions)] = np.nan
        if len(val_rows):
            x_val[np.ix_(val_rows, positions)] = np.nan
    return (
        x_train,
        np.asarray(y[train_idx], dtype=np.int8),
        x_val,
        np.asarray(y[val_idx], dtype=np.int8),
        train_idx,
        val_idx,
        selected,
    )


def save_predictions(task: dict[str, Any], val_idx: np.ndarray, pred: np.ndarray) -> str:
    context = _WORKER["context"]
    if not context["config"].get("save_predictions", True):
        return ""
    result_dir = Path(context["result_dir"])
    out = result_dir / "predictions" / task["family"] / task["profile"] / f"fold{task['fold_id']}" / f"{task['condition_id']}_seed{task['seed']}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    date_values = _WORKER["date_days"][val_idx]
    ticker_values = [_WORKER["ticker_values"][int(code)] for code in _WORKER["ticker_codes"][val_idx]]
    bucket_values = [_WORKER["bucket_values"][int(code)] for code in _WORKER["bucket_codes"][val_idx]]
    frame = pd.DataFrame({
        "date": pd.to_datetime(date_values.astype(np.int64), unit="D"),
        "ticker": ticker_values,
        "bucket": bucket_values,
        "y_true": np.asarray(_WORKER["y"][val_idx], dtype=np.int8),
        "prediction": np.asarray(pred, dtype=np.float32),
    })
    try:
        frame.to_parquet(out, index=False)
        return str(out)
    except Exception:
        fallback = out.with_suffix(".csv.gz")
        frame.to_csv(fallback, index=False, compression="gzip")
        return str(fallback)


def evaluate_scopes(val_idx: np.ndarray, pred: np.ndarray) -> list[dict[str, Any]]:
    y = np.asarray(_WORKER["y"][val_idx], dtype=np.int8)
    dates = np.asarray(_WORKER["date_days"][val_idx], dtype=np.int32)
    metrics = [{"scope_type": "all", "scope_value": "all_validation", **compute_metrics(y, pred, dates)}]
    battery_code = _WORKER["context"]["cache_meta"]["bucket_to_code"].get("battery_materials")
    if battery_code is not None:
        mask = np.asarray(_WORKER["bucket_codes"][val_idx] == battery_code)
        if int(mask.sum()) > 0:
            metrics.append({"scope_type": "bucket", "scope_value": "battery_materials", **compute_metrics(y[mask], pred[mask], dates[mask])})
    return metrics


def execute_model_task(family: str, phase: str, profile: str, fold_id: int, seed: int, condition_id: str, best_iteration: int | None = None) -> dict[str, Any]:
    context = _WORKER["context"]
    stopped, reason = should_stop()
    if stopped:
        return {"status": "skipped", "reason": reason, "family": family, "profile": profile, "fold_id": fold_id, "seed": seed, "condition_id": condition_id}
    model_cfg = context["config"]["lightgbm"] if family == "lightgbm" else context["config"]["xgboost_gpu"]
    task = make_task(family, phase, profile, fold_id, seed, condition_id, model_cfg, best_iteration)
    registry_path = Path(context["registry_path"])
    claim = claim_task(registry_path, task, int(context["config"]["max_task_attempts"]))
    if claim == "completed":
        return {"status": "cached", **{k: task[k] for k in ["task_id", "family", "profile", "fold_id", "seed", "condition_id"]}}
    if claim != "claimed":
        return {"status": "skipped", "reason": claim, **{k: task[k] for k in ["task_id", "family", "profile", "fold_id", "seed", "condition_id"]}}

    started = time.perf_counter()
    try:
        x_train, y_train, x_val, y_val, train_idx, val_idx, selected = build_task_matrices(profile, fold_id, condition_id)
        if len(np.unique(y_train)) < 2:
            raise RuntimeError("outer training target이 단일 클래스입니다.")
        if family == "lightgbm":
            model = fit_lightgbm(
                x_train,
                y_train,
                seed=seed,
                threads=int(context["config"]["threads_per_cpu_worker"]),
                config=context["config"]["lightgbm"],
                n_estimators=int(best_iteration or context["config"]["lightgbm"]["fallback_estimators"]),
            )
            actual_backend = "cpu"
        elif family == "xgboost_gpu":
            model = fit_xgboost_gpu(x_train, y_train, seed, int(context["config"]["gpu_threads"]), context["config"]["xgboost_gpu"])
            actual_backend = "cuda"
        else:
            raise ValueError(family)
        if family == "lightgbm":
            pred = np.asarray(model.booster_.predict(x_val, num_iteration=int(best_iteration or 0) or None), dtype=np.float32)
        else:
            pred = np.asarray(model.predict_proba(x_val)[:, 1], dtype=np.float32)
        scope_metrics = evaluate_scopes(val_idx, pred)
        elapsed = time.perf_counter() - started
        result_dir = Path(context["result_dir"])
        metrics_path = result_dir / "task_metrics" / family / profile / f"fold{fold_id}" / f"{condition_id}_seed{seed}.json"
        prediction_path = save_predictions(task, val_idx, pred)
        fold = next(f for f in context["folds"] if int(f["fold_id"]) == fold_id)
        payload = {
            **task,
            "status": "completed",
            "actual_backend": actual_backend,
            "elapsed_seconds": elapsed,
            "best_iteration": best_iteration,
            "feature_count": len(selected),
            "affected_feature_count": len(set(context["conditions"][condition_id].get("features", [])) & set(context["cache_meta"]["full_features"] if profile == "full_reduced" else context["cache_meta"]["common_features"])),
            "train_rows": len(train_idx),
            "validation_rows": len(val_idx),
            "validation_start": fold["validation_start"],
            "validation_end": fold["validation_end"],
            "validation_row_hash": hash_text(np.asarray(val_idx, dtype=np.int64).tobytes().hex(), 24),
            "metrics": scope_metrics,
            "prediction_path": prediction_path,
            "completed_at": now_iso(),
        }
        atomic_json(metrics_path, payload)
        if context["config"].get("save_models", False):
            model_path = result_dir / "models" / family / profile / f"fold{fold_id}" / f"{condition_id}_seed{seed}.txt"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if family == "lightgbm":
                model.booster_.save_model(str(model_path))
            else:
                model.save_model(str(model_path.with_suffix(".json")))
        finish_task(registry_path, task["task_id"], "completed", elapsed, metrics_path=str(metrics_path), prediction_path=prediction_path)
        del model, x_train, x_val, y_train, y_val, pred
        gc.collect()
        return {"status": "completed", "elapsed_seconds": elapsed, **{k: task[k] for k in ["task_id", "family", "profile", "fold_id", "seed", "condition_id"]}}
    except Exception as exc:
        elapsed = time.perf_counter() - started
        error = traceback.format_exc()
        error_path = Path(context["result_dir"]) / "errors" / f"{task['task_id']}.log"
        atomic_text(error_path, error)
        finish_task(registry_path, task["task_id"], "failed", elapsed, error=f"{type(exc).__name__}: {exc}")
        gc.collect()
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", **{k: task[k] for k in ["task_id", "family", "profile", "fold_id", "seed", "condition_id"]}}


def run_cpu_block(block: dict[str, Any]) -> dict[str, Any]:
    profile = block["profile"]
    fold_id = int(block["fold_id"])
    best_iteration = tune_best_iteration(profile, fold_id)
    results: list[dict[str, Any]] = []
    for condition_id in block["conditions"]:
        for seed in block["seeds"]:
            stopped, reason = should_stop()
            if stopped:
                return {"block": block, "best_iteration": best_iteration, "results": results, "stopped": reason}
            results.append(execute_model_task("lightgbm", block["phase"], profile, fold_id, int(seed), condition_id, best_iteration))
    return {"block": block, "best_iteration": best_iteration, "results": results}


def run_gpu_block(block: dict[str, Any]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for condition_id in block["conditions"]:
        for seed in block["seeds"]:
            stopped, reason = should_stop()
            if stopped:
                return {"block": block, "results": results, "stopped": reason}
            results.append(execute_model_task("xgboost_gpu", block["phase"], "common_period", int(block["fold_id"]), int(seed), condition_id, None))
    return {"block": block, "results": results}



class ResourceMonitor:
    def __init__(self, result_dir: Path, config: dict[str, Any], stop_file: Path):
        self.result_dir = result_dir
        self.config = config
        self.stop_file = stop_file
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.path = result_dir / "resource_usage.csv"

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10)

    @staticmethod
    def gpu_stats() -> tuple[float, float, float]:
        try:
            command = ["nvidia-smi", "--query-gpu=memory.used,memory.free,temperature.gpu", "--format=csv,noheader,nounits"]
            raw = subprocess.check_output(command, text=True, timeout=3).strip().splitlines()[0]
            used, free, temp = [float(x.strip()) for x in raw.split(",")[:3]]
            return used, free, temp
        except Exception:
            return float("nan"), float("nan"), float("nan")

    @staticmethod
    def cpu_temp() -> float:
        try:
            import psutil
            temps = psutil.sensors_temperatures()
            values = [float(item.current) for items in temps.values() for item in items if item.current is not None]
            return max(values) if values else float("nan")
        except Exception:
            return float("nan")

    def _run(self) -> None:
        import psutil

        header = ["timestamp", "cpu_percent", "available_ram_gb", "used_ram_percent", "process_rss_gb", "cpu_temp_c", "gpu_used_mb", "gpu_free_mb", "gpu_temp_c"]
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(header)
        process = psutil.Process(os.getpid())
        critical_count = 0
        temp_count = 0
        while not self.stop_event.wait(float(self.config["resource_sample_seconds"])):
            vm = psutil.virtual_memory()
            cpu_temp = self.cpu_temp()
            gpu_used, gpu_free, gpu_temp = self.gpu_stats()
            row = [now_iso(), psutil.cpu_percent(interval=None), vm.available / 2**30, vm.percent, process.memory_info().rss / 2**30, cpu_temp, gpu_used, gpu_free, gpu_temp]
            with self.path.open("a", newline="", encoding="utf-8-sig") as f:
                csv.writer(f).writerow(row)
            critical_count = critical_count + 1 if vm.available / 2**30 < float(self.config["critical_ram_stop_gb"]) else 0
            temp_count = temp_count + 1 if math.isfinite(cpu_temp) and cpu_temp >= float(self.config["cpu_temp_stop_c"]) else 0
            if critical_count >= int(self.config["critical_ram_consecutive_samples"]):
                atomic_json(self.stop_file, {"reason": "critical_low_ram", "available_ram_gb": vm.available / 2**30, "time": now_iso()})
            if temp_count >= 3:
                atomic_json(self.stop_file, {"reason": "critical_cpu_temperature", "cpu_temp_c": cpu_temp, "time": now_iso()})



def flatten_block_results(block_result: dict[str, Any]) -> list[dict[str, Any]]:
    return list(block_result.get("results", []))


def run_blocks(blocks: list[dict[str, Any]], workers: int, context_path: Path, gpu: bool = False) -> list[dict[str, Any]]:
    if not blocks:
        return []
    outputs: list[dict[str, Any]] = []
    fn = run_gpu_block if gpu else run_cpu_block
    ctx = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=worker_init, initargs=(str(context_path),)) as executor:
        futures = {executor.submit(fn, block): block for block in blocks}
        completed_blocks = 0
        for future in cf.as_completed(futures):
            block = futures[future]
            completed_blocks += 1
            try:
                result = future.result()
                outputs.extend(flatten_block_results(result))
                statuses = pd.Series([x.get("status") for x in result.get("results", [])]).value_counts().to_dict()
                print(f"[BLOCK {completed_blocks}/{len(blocks)}] {block['phase']} {block.get('profile','common_period')} fold={block['fold_id']} {statuses}")
            except Exception as exc:
                outputs.append({"status": "block_failed", "error": f"{type(exc).__name__}: {exc}", "block": block})
                print(f"[BLOCK FAILED] {block}: {exc}")
    return outputs


def gpu_available() -> bool:
    try:
        subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=5)
        import xgboost  # noqa: F401
        return True
    except Exception:
        return False


def registry_dataframe(registry_path: Path) -> pd.DataFrame:
    conn = registry_connect(registry_path)
    df = pd.read_sql_query("SELECT * FROM tasks", conn)
    conn.close()
    return df


def average_completed_seconds(registry_path: Path, family: str = "lightgbm") -> float:
    df = registry_dataframe(registry_path)
    values = pd.to_numeric(df.loc[(df["family"] == family) & (df["status"] == "completed"), "elapsed_seconds"], errors="coerce")
    return float(values.median()) if values.notna().any() else 30.0


def estimated_minutes(task_count: int, worker_count: int, registry_path: Path, family: str = "lightgbm") -> float:
    return average_completed_seconds(registry_path, family) * task_count / max(worker_count, 1) * 1.20 / 60.0



def read_task_metrics(result_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in result_dir.glob("task_metrics/**/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            base = {k: payload.get(k) for k in ["task_id", "family", "phase", "profile", "fold_id", "seed", "condition_id", "status", "actual_backend", "elapsed_seconds", "best_iteration", "feature_count", "affected_feature_count", "train_rows", "validation_rows", "validation_start", "validation_end", "validation_row_hash", "prediction_path"]}
            for metric in payload.get("metrics", []):
                rows.append({**base, **metric, "source_file": str(path)})
        except Exception:
            continue
    return pd.DataFrame(rows)


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if not len(valid):
        return out
    order = valid[np.argsort(p[valid])]
    adjusted = p[order] * np.arange(len(order), 0, -1)
    adjusted = np.maximum.accumulate(adjusted)
    out[order] = np.minimum(adjusted, 1.0)
    return out


def fold_bootstrap_ci(values: Sequence[float], seed: int = 20260804, draws: int = 10000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(arr, size=(draws, len(arr)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def one_sided_wilcoxon(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3 or np.allclose(arr, 0):
        return 1.0
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(arr, alternative="greater", zero_method="wilcox").pvalue)
    except Exception:
        return float("nan")


def summarize_fold_values(frame: pd.DataFrame, value_col: str, keys: list[str]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for group_keys, part in frame.groupby(keys, dropna=False):
        if not isinstance(group_keys, tuple):
            group_keys = (group_keys,)
        fold_means = part.groupby("fold_id")[value_col].mean().sort_index()
        values = fold_means.to_numpy(dtype=float)
        low, high = fold_bootstrap_ci(values)
        row = {key: value for key, value in zip(keys, group_keys)}
        row.update({
            "fold_count": int(np.isfinite(values).sum()),
            "mean": safe_mean(values),
            "median": float(np.nanmedian(values)) if np.any(np.isfinite(values)) else float("nan"),
            "std": float(np.nanstd(values, ddof=1)) if np.isfinite(values).sum() > 1 else float("nan"),
            "minimum": float(np.nanmin(values)) if np.any(np.isfinite(values)) else float("nan"),
            "maximum": float(np.nanmax(values)) if np.any(np.isfinite(values)) else float("nan"),
            "positive_folds": int(np.sum(values > 0)),
            "positive_fold_ratio": float(np.mean(values > 0)) if len(values) else float("nan"),
            "recent_mean": safe_mean(fold_means[fold_means.index >= 4].tolist()),
            "recent_positive_folds": int(np.sum(fold_means[fold_means.index >= 4].to_numpy() > 0)),
            "early_mean": safe_mean(fold_means[fold_means.index <= 3].tolist()),
            "ci95_low": low,
            "ci95_high": high,
            "p_value_one_sided": one_sided_wilcoxon(values),
        })
        output.append(row)
    return pd.DataFrame(output)


def aggregate_results(result_dir: Path, context: dict[str, Any], started_at: float) -> None:
    metrics = read_task_metrics(result_dir)
    if metrics.empty:
        metrics = pd.DataFrame(columns=["task_id", "family", "profile", "fold_id", "seed", "condition_id", "scope_type", "scope_value", "raw_pr_auc", "raw_roc_auc", "raw_brier", "raw_logloss"])
    write_csv_schema(metrics, result_dir / "all_metrics.csv")
    write_csv_schema(metrics, result_dir / "outer_metrics.csv")

    main = metrics[(metrics["family"] == "lightgbm")].copy() if "family" in metrics else metrics.copy()
    global_all = main[(main["scope_type"] == "all") & (main["condition_id"].isin(["B0", "A1", "A2", "A3", "A4", "A5", "A6"]))].copy()
    baseline = global_all[global_all["condition_id"] == "B0"].copy()
    paired_rows: list[pd.DataFrame] = []
    for condition in ["A1", "A2", "A3", "A4", "A5", "A6"]:
        ablated = global_all[global_all["condition_id"] == condition].copy()
        keys = ["profile", "fold_id", "seed", "scope_type", "scope_value"]
        merged = baseline.merge(ablated, on=keys, suffixes=("_baseline", "_ablated"))
        if merged.empty:
            continue
        merged["condition_id"] = condition
        for metric in ["raw_pr_auc", "raw_roc_auc", "balanced_accuracy", "raw_top_3pct_precision"]:
            merged[f"{metric}_loss"] = merged[f"{metric}_baseline"] - merged[f"{metric}_ablated"]
        for metric in ["raw_brier", "raw_logloss"]:
            merged[f"{metric}_increase"] = merged[f"{metric}_ablated"] - merged[f"{metric}_baseline"]
        paired_rows.append(merged)
    paired = pd.concat(paired_rows, ignore_index=True) if paired_rows else pd.DataFrame()
    write_csv_schema(
        paired,
        result_dir / "paired_global_deltas.csv",
        ["profile", "fold_id", "seed", "condition_id", "raw_pr_auc_loss", "raw_roc_auc_loss", "raw_brier_increase", "raw_logloss_increase"],
    )

    if not paired.empty:
        summary = summarize_fold_values(paired, "raw_pr_auc_loss", ["profile", "condition_id"])
    else:
        summary = pd.DataFrame(columns=["profile", "condition_id", "mean", "median"])
    write_csv_schema(summary, result_dir / "global_feature_utility_summary.csv")

    profile_rows: list[dict[str, Any]] = []
    for condition, part in summary.groupby("condition_id") if not summary.empty else []:
        values = {row.profile: row for row in part.itertuples(index=False)}
        full = values.get("full_reduced")
        common = values.get("common_period")
        profile_rows.append({
            "condition_id": condition,
            "full_mean": getattr(full, "mean", np.nan),
            "common_mean": getattr(common, "mean", np.nan),
            "direction_agreement": bool(full and common and np.sign(full.mean) == np.sign(common.mean)),
            "both_positive": bool(full and common and full.mean > 0 and common.mean > 0),
        })
    profile_consistency = pd.DataFrame(profile_rows)
    write_csv_schema(profile_consistency, result_dir / "profile_consistency.csv", ["condition_id", "full_mean", "common_mean", "direction_agreement", "both_positive"])

    drift = summary[[c for c in ["profile", "condition_id", "early_mean", "recent_mean", "minimum", "positive_folds", "recent_positive_folds"] if c in summary.columns]].copy()
    if not drift.empty:
        drift["recent_minus_early"] = drift["recent_mean"] - drift["early_mean"]
    write_csv_schema(drift, result_dir / "recent_drift_summary.csv")

    battery_metrics = main[(main["scope_type"] == "bucket") & (main["scope_value"] == "battery_materials") & (main["profile"] == "common_period")].copy()
    battery_base = battery_metrics[battery_metrics["condition_id"] == "B0"]
    battery_pairs: list[pd.DataFrame] = []
    for condition in ["S1", "S2", "S3"]:
        ablated = battery_metrics[battery_metrics["condition_id"] == condition]
        merged = battery_base.merge(ablated, on=["profile", "fold_id", "seed", "scope_type", "scope_value"], suffixes=("_baseline", "_ablated"))
        if not merged.empty:
            merged["condition_id"] = condition
            merged["raw_pr_auc_loss"] = merged["raw_pr_auc_baseline"] - merged["raw_pr_auc_ablated"]
            merged["raw_roc_auc_loss"] = merged["raw_roc_auc_baseline"] - merged["raw_roc_auc_ablated"]
            battery_pairs.append(merged)
    battery_paired = pd.concat(battery_pairs, ignore_index=True) if battery_pairs else pd.DataFrame()
    battery_summary = summarize_fold_values(battery_paired, "raw_pr_auc_loss", ["condition_id"]) if not battery_paired.empty else pd.DataFrame()
    battery_columns = ["condition_id", "fold_count", "mean", "median", "std", "minimum", "maximum", "positive_folds", "recent_positive_folds", "p_value_one_sided"]
    write_csv_schema(battery_summary, result_dir / "battery_sector_confirmatory.csv", battery_columns)
    write_csv_schema(battery_summary, result_dir / "battery_confirmatory.csv", battery_columns)

    # Joint incremental evidence: A3 - max(A1, A2), computed within identical profile/fold/seed.
    incremental = pd.DataFrame()
    etf_summaries: list[pd.DataFrame] = []
    if not paired.empty:
        wide = paired.pivot_table(index=["profile", "fold_id", "seed"], columns="condition_id", values="raw_pr_auc_loss", aggfunc="first").reset_index()
        if all(c in wide.columns for c in ["A1", "A2", "A3"]):
            wide["joint_incremental"] = wide["A3"] - wide[["A1", "A2"]].max(axis=1)
            incremental = summarize_fold_values(wide, "joint_incremental", ["profile"])
            incremental["hypothesis"] = "A3_incremental"
        if all(c in wide.columns for c in ["A2", "A5"]):
            wide["etf_market_incremental"] = wide["A5"] - wide["A2"]
            part = summarize_fold_values(wide, "etf_market_incremental", ["profile"])
            part["hypothesis"] = "ETF_incremental_over_market"
            etf_summaries.append(part)
        if all(c in wide.columns for c in ["A3", "A6"]):
            wide["etf_joint_incremental"] = wide["A6"] - wide["A3"]
            part = summarize_fold_values(wide, "etf_joint_incremental", ["profile"])
            part["hypothesis"] = "ETF_incremental_over_shorting_market"
            etf_summaries.append(part)
        write_csv_schema(wide, result_dir / "interaction_incremental_by_seed.csv")
    etf_summary = pd.concat(etf_summaries, ignore_index=True) if etf_summaries else pd.DataFrame()
    summary_columns = ["profile", "fold_count", "mean", "median", "std", "minimum", "maximum", "positive_folds", "recent_positive_folds", "p_value_one_sided", "hypothesis"]
    write_csv_schema(incremental, result_dir / "joint_incremental_summary.csv", summary_columns)
    write_csv_schema(etf_summary, result_dir / "etf_conditional_summary.csv", summary_columns)

    hypotheses: list[dict[str, Any]] = []
    for condition in ["A1", "A2"]:
        part = summary[summary["condition_id"] == condition]
        if not part.empty:
            hypotheses.append({"hypothesis": condition, "p_value": float(part["p_value_one_sided"].max()), "effect_mean": float(part["mean"].min())})
    if not incremental.empty:
        hypotheses.append({"hypothesis": "A3_incremental", "p_value": float(incremental["p_value_one_sided"].max()), "effect_mean": float(incremental["mean"].min())})
    if not etf_summary.empty:
        # H4 permits either ETF increment; adjust the selected candidate inside H4 conservatively.
        h4_p = min(1.0, float(etf_summary["p_value_one_sided"].min()) * len(etf_summary))
        hypotheses.append({"hypothesis": "H4_ETF_conditional", "p_value": h4_p, "effect_mean": float(etf_summary["mean"].max())})
    if not battery_summary.empty and (battery_summary["condition_id"] == "S1").any():
        row = battery_summary[battery_summary["condition_id"] == "S1"].iloc[0]
        hypotheses.append({"hypothesis": "S1_battery", "p_value": row["p_value_one_sided"], "effect_mean": row["mean"]})
    testing = pd.DataFrame(hypotheses)
    if not testing.empty:
        testing["q_value_holm"] = holm_adjust(testing["p_value"].to_numpy())
    write_csv_schema(testing, result_dir / "multiple_testing_summary.csv", ["hypothesis", "p_value", "effect_mean", "q_value_holm"])

    thresholds = context["config"]["pass_thresholds"]
    decisions: list[dict[str, Any]] = []
    for condition in ["A1", "A2"]:
        part = summary[summary["condition_id"] == condition]
        profile_passes = []
        for row in part.itertuples(index=False):
            passed = (
                row.mean >= thresholds["main_mean_loss"]
                and row.median > thresholds["main_median_loss"]
                and row.positive_folds >= thresholds["main_positive_folds"]
                and row.recent_positive_folds >= thresholds["main_recent_positive_folds"]
                and row.minimum >= thresholds["main_worst_fold"]
            )
            profile_passes.append(bool(passed))
        decision = "KEEP" if len(profile_passes) == 2 and all(profile_passes) else "CONDITIONAL_KEEP" if any(profile_passes) else "HOLD"
        decisions.append({"hypothesis": condition, "decision": decision, "profiles_passed": sum(profile_passes), "profiles_evaluated": len(profile_passes)})
    if not incremental.empty:
        passed = bool(len(incremental) == 2 and np.all(incremental["mean"] >= thresholds["joint_incremental_mean"]) and np.all(incremental["positive_folds"] >= thresholds["main_positive_folds"]))
        decisions.append({"hypothesis": "A3_incremental", "decision": "KEEP" if passed else "HOLD", "profiles_passed": int(np.sum((incremental["mean"] >= thresholds["joint_incremental_mean"]) & (incremental["positive_folds"] >= thresholds["main_positive_folds"]))), "profiles_evaluated": len(incremental)})
    if not etf_summary.empty:
        etf_pass = (
            (etf_summary["mean"] >= thresholds["etf_incremental_mean"])
            & (etf_summary["minimum"] >= -0.003)
            & (etf_summary["recent_positive_folds"] >= thresholds["main_recent_positive_folds"])
        )
        decisions.append({
            "hypothesis": "H4_ETF_conditional",
            "decision": "CONDITIONAL_KEEP" if bool(etf_pass.any()) else "HOLD",
            "profiles_passed": int(etf_pass.sum()),
            "profiles_evaluated": len(etf_summary),
        })
    if not battery_summary.empty:
        row = battery_summary[battery_summary["condition_id"] == "S1"]
        if not row.empty:
            r = row.iloc[0]
            passed = r["mean"] >= thresholds["battery_mean_loss"] and r["median"] > 0 and r["positive_folds"] >= thresholds["main_positive_folds"] and r["recent_positive_folds"] >= thresholds["main_recent_positive_folds"]
            decisions.append({"hypothesis": "S1_battery", "decision": "CONDITIONAL_KEEP" if passed else "HOLD", "profiles_passed": int(passed), "profiles_evaluated": 1})
    write_csv_schema(pd.DataFrame(decisions), result_dir / "decision_table.csv", ["hypothesis", "decision", "profiles_passed", "profiles_evaluated"])

    worst_rows: list[dict[str, Any]] = []
    if not paired.empty:
        folded = paired.groupby(["profile", "condition_id", "fold_id"], as_index=False)["raw_pr_auc_loss"].mean()
        for (profile, condition), part in folded.groupby(["profile", "condition_id"]):
            row = part.loc[part["raw_pr_auc_loss"].idxmin()]
            worst_rows.append({"profile": profile, "condition_id": condition, "worst_fold": int(row["fold_id"]), "worst_raw_pr_auc_loss": float(row["raw_pr_auc_loss"])})
    write_csv_schema(pd.DataFrame(worst_rows), result_dir / "worst_fold_diagnostics.csv", ["profile", "condition_id", "worst_fold", "worst_raw_pr_auc_loss"])

    gpu_metrics = metrics[metrics["family"] == "xgboost_gpu"].copy() if not metrics.empty else pd.DataFrame()
    write_csv_schema(gpu_metrics, result_dir / "xgboost_sensitivity_metrics.csv", list(metrics.columns))

    registry_path = Path(context["registry_path"])
    registry = registry_dataframe(registry_path)
    failed = registry[registry["status"].isin(["failed", "skipped", "pending", "running"])] if not registry.empty else registry
    write_csv_schema(failed, result_dir / "failed_tasks.csv", list(registry.columns))

    tuning_rows: list[dict[str, Any]] = []
    for path in sorted((result_dir / "tuning").glob("*.json")):
        try:
            tuning_rows.append(json.loads(path.read_text(encoding="utf-8")))
        except Exception:
            continue
    write_csv_schema(
        pd.DataFrame(tuning_rows),
        result_dir / "best_iteration_by_fold_profile.csv",
        ["profile", "fold_id", "best_iteration", "inner_train_rows", "inner_validation_rows", "inner_train_end", "inner_validation_start", "created_at"],
    )
    status_counts = registry["status"].value_counts().to_dict() if not registry.empty else {}
    atomic_json(result_dir / "registry_status.json", {"total": len(registry), "status_counts": status_counts, "generated_at": now_iso()})

    completed_main = registry[(registry["family"] == "lightgbm") & (registry["status"] == "completed")] if not registry.empty else registry
    summary_payload = {
        "schema_version": context["config"]["schema_version"],
        "started_at": dt.datetime.fromtimestamp(started_at).astimezone().isoformat(),
        "finished_at": now_iso(),
        "elapsed_hours": (time.time() - started_at) / 3600,
        "dataset_signature": context["cache_meta"]["dataset_signature"],
        "result_dir": str(result_dir),
        "registry_status": status_counts,
        "completed_lightgbm_tasks": len(completed_main),
        "completed_gpu_tasks": int(((registry["family"] == "xgboost_gpu") & (registry["status"] == "completed")).sum()) if not registry.empty else 0,
        "stop_requested": Path(context["stop_file"]).exists(),
        "stop_detail": json.loads(Path(context["stop_file"]).read_text(encoding="utf-8")) if Path(context["stop_file"]).exists() else None,
    }
    atomic_json(result_dir / "run_summary.json", summary_payload)

    pairing_issues = []
    if not paired.empty:
        for _, row in paired.iterrows():
            if row.get("validation_row_hash_baseline") != row.get("validation_row_hash_ablated"):
                pairing_issues.append({"profile": row["profile"], "fold_id": row["fold_id"], "seed": row["seed"], "condition_id": row["condition_id"], "issue": "validation_row_hash_mismatch"})
    atomic_json(result_dir / "execution_audit.json", {
        "generated_at": now_iso(),
        "pairing_issue_count": len(pairing_issues),
        "pairing_issues": pairing_issues[:100],
        "fold_sources": sorted({f["source"] for f in context["folds"]}),
        "backend_policy": {"main": "lightgbm_cpu", "sensitivity": "xgboost_cuda_optional"},
        "sealed_data_used": False,
    })


def write_context(result_dir: Path, cache_meta: dict[str, Any], folds: list[dict[str, Any]], config: dict[str, Any], deadline_epoch: float) -> Path:
    registry_path = result_dir / "task_registry.sqlite"
    stop_file = result_dir / "STOP_REQUESTED.json"
    context = {
        "result_dir": str(result_dir),
        "registry_path": str(registry_path),
        "stop_file": str(stop_file),
        "deadline_epoch": deadline_epoch,
        "cache_meta": cache_meta,
        "folds": folds,
        "conditions": condition_specs(cache_meta),
        "config": config,
    }
    path = result_dir / "execution_context.json"
    atomic_json(path, context)
    return path



def create_self_test_dataset(path: Path) -> None:
    feature_quality = pd.read_csv(REFERENCE_DIR / "feature_quality_refine.csv")
    features = feature_quality.loc[as_bool_series(feature_quality["selected_full_reduced"]), "feature"].astype(str).tolist()
    ticker_map = pd.read_csv(REFERENCE_DIR / "ticker_bucket_map.csv", dtype={"ticker": str})
    tickers = ticker_map["ticker"].head(8).str.zfill(6).tolist()
    bucket_map = dict(zip(ticker_map["ticker"].str.zfill(6), ticker_map["bucket"]))
    dates = pd.bdate_range("2023-01-02", periods=180)
    rng = np.random.default_rng(123)
    rows = []
    for date in dates:
        for ticker in tickers:
            base = rng.normal()
            row = {"date": date, "ticker": ticker, "bucket": bucket_map[ticker]}
            for feature in features:
                row[feature] = rng.normal()
            signal_names = [f for f in features if f.startswith("u_finshort_") or f.startswith("u_finmarket_")][:8]
            signal = sum(row[f] for f in signal_names) / max(len(signal_names), 1)
            probability = 1 / (1 + math.exp(-(-1.4 + 0.8 * signal + 0.2 * base)))
            row[TARGET_DEFAULT] = int(rng.random() < probability)
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def run_self_test() -> int:
    temp = Path(tempfile.mkdtemp(prefix="crashwatch_selftest_"))
    dataset = temp / "training_dataset_finance11h.csv"
    results = temp / "results"
    create_self_test_dataset(dataset)
    cfg = default_config()
    cfg.update({
        "dataset_path": str(dataset),
        "results_dir": str(results),
        "max_hours": 0.20,
        "outer_folds": 2,
        "validation_days": 20,
        "purge_days": 5,
        "min_train_days": 80,
        "inner_validation_days": 20,
        "inner_purge_days": 5,
        "fold_strategy": "generated",
        "seeds_core": [17],
        "seeds_optional": [17],
        "cpu_workers": 2,
        "cpu_workers_with_gpu": 2,
        "threads_per_cpu_worker": 1,
        "enable_gpu_sensitivity": False,
        "strict_reference_features": True,
        "full_dataset_sha256": False,
        "optional_priority": [],
    })
    cfg["lightgbm"].update({"max_estimators": 60, "early_stopping_rounds": 8, "fallback_estimators": 30, "num_leaves": 15, "min_child_samples": 15})
    code = run_experiment(cfg, dataset_override=str(dataset), results_override=str(results), self_test=True)
    print(f"[SELF-TEST] 결과: {results}")
    return code



def run_experiment(config: dict[str, Any], dataset_override: str | None = None, results_override: str | None = None, self_test: bool = False) -> int:
    ensure_dependencies()
    started_at = time.time()
    dataset_path = resolve_dataset(config, dataset_override)
    result_dir = resolve_results_dir(config, dataset_path, results_override)
    deadline_epoch = started_at + float(config["max_hours"]) * 3600

    fatal_failure = False
    with RunLock(result_dir):
        atomic_json(result_dir / "effective_config.json", config)
        print(f"[RUN] dataset={dataset_path}")
        print(f"[RUN] results={result_dir}")
        print(f"[RUN] max_hours={config['max_hours']}")
        cache_meta = prepare_cache(dataset_path, result_dir, config)
        folds = build_folds(cache_meta, config)
        atomic_json(result_dir / "outer_fold_manifest.json", folds)
        atomic_json(result_dir / "fold_manifest.json", folds)
        atomic_json(result_dir / "runtime_plan.json", {
            "generated_at": now_iso(),
            "cpu_workers": int(config["cpu_workers"]),
            "cpu_workers_with_gpu": int(config["cpu_workers_with_gpu"]),
            "threads_per_cpu_worker": int(config["threads_per_cpu_worker"]),
            "allocated_cpu_threads": int(config["cpu_workers"]) * int(config["threads_per_cpu_worker"]),
            "core_lightgbm_tasks": int(config["outer_folds"]) * 2 * 4 * len(config["seeds_core"]),
            "battery_lightgbm_tasks_max": int(config["outer_folds"]) * 3 * len(config["seeds_optional"]),
            "etf_lightgbm_tasks_max": int(config["outer_folds"]) * 2 * 3 * len(config["seeds_optional"]),
            "xgboost_cuda_tasks_max": len(config["xgboost_gpu"]["folds"]) * len(config["xgboost_gpu"]["conditions"]) * len(config["xgboost_gpu"]["seeds"]),
            "canary_reused_by_core": True,
            "max_hours": float(config["max_hours"]),
        })
        atomic_json(result_dir / "dataset_and_feature_manifest.json", {
            **{k: cache_meta[k] for k in ["dataset_signature", "feature_hash", "dataset_path", "rows", "tickers", "buckets", "date_min", "date_max", "positive_rate", "full_feature_count", "common_feature_count", "missing_reference_features"]},
            "condition_feature_counts": {cid: {"full_reduced": len(set(spec["features"]) & set(cache_meta["full_features"])), "common_period": len(set(spec["features"]) & set(cache_meta["common_features"]))} for cid, spec in condition_specs(cache_meta).items()},
            "sealed_data_used": False,
        })
        registry_path = result_dir / "task_registry.sqlite"
        reset_stale_tasks(registry_path)
        context_path = write_context(result_dir, cache_meta, folds, config, deadline_epoch)
        context = json.loads(context_path.read_text(encoding="utf-8"))
        stop_file = Path(context["stop_file"])
        stop_file.unlink(missing_ok=True)
        monitor = ResourceMonitor(result_dir, config, stop_file)
        monitor.start()

        try:
            # Full canary: last fold, both profiles, B0-A3, seed 17, including save/registry/report schema paths.
            if not self_test:
                worker_init(str(context_path))
                canary_results: list[dict[str, Any]] = []
                canary_fold = min(int(config["canary_fold"]), max(int(f["fold_id"]) for f in folds))
                for profile in config["canary_profiles"]:
                    # Use the core phase identity so these exact fits are cache hits in Phase 1.
                    canary_results.extend(run_cpu_block({"phase": "core", "profile": profile, "fold_id": canary_fold, "conditions": ["B0", "A1", "A2", "A3"], "seeds": [17]})["results"])
                clear_worker()
                failures = [r for r in canary_results if r.get("status") == "failed"]
                atomic_json(result_dir / "canary_summary.json", {"results": canary_results, "failed": len(failures), "completed_at": now_iso()})
                if failures:
                    atomic_json(stop_file, {"reason": "canary_failure", "failures": failures, "time": now_iso()})
                    raise RuntimeError(f"Canary 실패 {len(failures)}건. errors 폴더를 확인하세요.")

            fold_order = sorted([int(f["fold_id"]) for f in folds], reverse=True)
            phase1 = [
                {"phase": "core", "profile": profile, "fold_id": fold_id, "conditions": ["B0", "A1", "A2", "A3"], "seeds": config["seeds_core"]}
                for fold_id in fold_order
                for profile in ["common_period", "full_reduced"]
            ]
            core_workers = int(config["cpu_workers"])
            try:
                import psutil
                if psutil.virtual_memory().available / 2**30 < float(config["low_ram_warning_gb"]):
                    core_workers = max(1, core_workers - 1)
                    print(f"[RAM] 가용 RAM이 낮아 core workers를 {core_workers}로 축소합니다.")
            except Exception:
                pass
            core_results = run_blocks(phase1, core_workers, context_path)
            failed_core = [r for r in core_results if r.get("status") == "failed"]
            if failed_core and time.time() < deadline_epoch - float(config["finalization_reserve_minutes"]) * 60:
                retry_blocks = [
                    {
                        "phase": "core",
                        "profile": r["profile"],
                        "fold_id": int(r["fold_id"]),
                        "conditions": [r["condition_id"]],
                        "seeds": [int(r["seed"])],
                    }
                    for r in failed_core
                ]
                print(f"[RETRY] core 실패 {len(retry_blocks)}건을 1회 재시도합니다.")
                retry_results = run_blocks(retry_blocks, min(core_workers, max(1, len(retry_blocks))), context_path)
                core_results.extend(retry_results)
            remaining_core_failures = len(registry_dataframe(registry_path).query("family == 'lightgbm' and phase == 'core' and status == 'failed'"))
            if remaining_core_failures:
                print(f"[WARN] 재시도 후 core 실패={remaining_core_failures}. 실패 task는 failed_tasks.csv에 기록됩니다.")

            remaining_minutes = (deadline_epoch - time.time()) / 60
            reserve = float(config["finalization_reserve_minutes"])
            gpu_executor = None
            gpu_futures: list[Any] = []
            gpu_active = False
            if config.get("enable_gpu_sensitivity", True) and remaining_minutes >= float(config["gpu_min_remaining_minutes"]) + reserve and gpu_available():
                gpu_cfg = config["xgboost_gpu"]
                gpu_blocks = [{"phase": "gpu_sensitivity", "fold_id": fold, "conditions": gpu_cfg["conditions"], "seeds": gpu_cfg["seeds"]} for fold in sorted(gpu_cfg["folds"], reverse=True) if fold in fold_order]
                ctx = mp.get_context("spawn")
                gpu_executor = cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx, initializer=worker_init, initargs=(str(context_path),))
                gpu_futures = [gpu_executor.submit(run_gpu_block, block) for block in gpu_blocks]
                gpu_active = True
                print(f"[GPU] XGBoost CUDA 민감도 검증 시작: {sum(len(b['conditions']) * len(b['seeds']) for b in gpu_blocks)} tasks")

            optional_workers = int(config["cpu_workers_with_gpu"] if gpu_active else config["cpu_workers"])
            for optional_name in config.get("optional_priority", ["battery", "etf"]):
                remaining_minutes = (deadline_epoch - time.time()) / 60
                if remaining_minutes <= reserve:
                    break
                if optional_name == "battery":
                    blocks = [
                        {"phase": "battery", "profile": "common_period", "fold_id": fold_id, "conditions": ["S1", "S2", "S3"], "seeds": config["seeds_optional"]}
                        for fold_id in fold_order
                    ]
                    task_count = len(blocks) * 3 * len(config["seeds_optional"])
                elif optional_name == "etf":
                    blocks = [
                        {"phase": "etf", "profile": profile, "fold_id": fold_id, "conditions": ["A4", "A5", "A6"], "seeds": config["seeds_optional"]}
                        for fold_id in fold_order
                        for profile in ["common_period", "full_reduced"]
                    ]
                    task_count = len(blocks) * 3 * len(config["seeds_optional"])
                else:
                    continue
                estimate = estimated_minutes(task_count, optional_workers, registry_path)
                if estimate + reserve > remaining_minutes:
                    recent = [f for f in fold_order if f >= 4]
                    blocks = [b for b in blocks if b["fold_id"] in recent]
                    task_count = sum(len(b["conditions"]) * len(b["seeds"]) for b in blocks)
                    estimate = estimated_minutes(task_count, optional_workers, registry_path)
                    print(f"[TIME] {optional_name}을 최근 fold 4~7로 축소. ETA≈{estimate:.1f}분")
                if estimate + reserve > remaining_minutes:
                    print(f"[TIME] {optional_name} 생략: remaining={remaining_minutes:.1f}분, ETA={estimate:.1f}분")
                    continue
                run_blocks(blocks, optional_workers, context_path)

            if gpu_executor is not None:
                for future in cf.as_completed(gpu_futures):
                    try:
                        result = future.result()
                        statuses = pd.Series([x.get("status") for x in result.get("results", [])]).value_counts().to_dict()
                        print(f"[GPU BLOCK] fold={result['block']['fold_id']} {statuses}")
                    except Exception as exc:
                        print(f"[GPU BLOCK FAILED] {exc}")
                gpu_executor.shutdown(wait=True, cancel_futures=False)

        except KeyboardInterrupt:
            atomic_json(stop_file, {"reason": "keyboard_interrupt", "time": now_iso()})
            print("[STOP] 사용자가 중단했습니다. 완료 task는 보존됩니다.")
        except Exception as exc:
            fatal_failure = True
            atomic_json(result_dir / "fatal_error.json", {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(), "time": now_iso()})
            print(f"[FATAL] {type(exc).__name__}: {exc}")
        finally:
            monitor.stop()
            aggregate_results(result_dir, context, started_at)
            print(f"[DONE] 결과 폴더: {result_dir}")
            print(f"[DONE] 요약: {result_dir / 'run_summary.json'}")

    return 1 if fatal_failure else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CrashWatch 3-hour confirmatory paired ablation")
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.json"), help="config JSON")
    parser.add_argument("--dataset", default=None, help="training_dataset_finance11h.parquet path")
    parser.add_argument("--results", default=None, help="result directory")
    parser.add_argument("--max-hours", type=float, default=None, help="override max_hours")
    parser.add_argument("--skip-gpu", action="store_true", help="disable XGBoost CUDA sensitivity")
    parser.add_argument("--self-test", action="store_true", help="run a small synthetic end-to-end test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        return run_self_test()
    config_path = Path(args.config) if args.config else None
    config = load_config(config_path)
    if args.max_hours is not None:
        config["max_hours"] = args.max_hours
    if args.skip_gpu:
        config["enable_gpu_sensitivity"] = False
    return run_experiment(config, dataset_override=args.dataset, results_override=args.results)


if __name__ == "__main__":
    mp.freeze_support()
    raise SystemExit(main())
