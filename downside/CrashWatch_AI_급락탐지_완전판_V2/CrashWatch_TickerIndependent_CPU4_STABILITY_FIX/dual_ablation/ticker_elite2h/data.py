from __future__ import annotations

import gc
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..io_utils import atomic_csv, atomic_json
from ..refine12h.data import load_refine_data


@dataclass(frozen=True)
class AdaptiveFold:
    fold_id: int
    fit_idx: np.ndarray
    calibration_idx: np.ndarray
    validation_idx: np.ndarray
    inner_fit_idx: np.ndarray
    inner_validation_idx: np.ndarray
    validation_days: int
    fit_date_min: str
    fit_date_max: str
    calibration_date_min: str
    calibration_date_max: str
    validation_date_min: str
    validation_date_max: str


@dataclass
class CachedTickerData:
    ticker: str
    bucket: str
    X: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    features: list[str]
    folds: list[AdaptiveFold]
    cache_dir: Path


def _date_text(values: np.ndarray) -> tuple[str, str]:
    if len(values) == 0:
        return "", ""
    index = pd.DatetimeIndex(values)
    return str(index.min().date()), str(index.max().date())


def _build_folds_for_window(dates: np.ndarray, y: np.ndarray, validation_days: int, plan: dict[str, Any]) -> list[AdaptiveFold]:
    date_index = pd.DatetimeIndex(pd.to_datetime(dates))
    unique_dates = date_index.unique().sort_values()
    target_folds = int(plan["target_outer_folds"])
    purge_days = int(plan["purge_days"])
    calibration_days = int(plan["calibration_days"])
    calibration_purge = int(plan["calibration_purge_days"])
    inner_validation_days = int(plan["inner_validation_days"])
    inner_purge = int(plan["inner_purge_days"])
    min_fit_days = int(plan["min_fit_days"])
    min_pos = int(plan["min_fold_positives"])
    min_neg = int(plan["min_fold_negatives"])
    folds: list[AdaptiveFold] = []

    for fold_id in range(target_folds):
        validation_end = len(unique_dates) - (target_folds - 1 - fold_id) * validation_days
        validation_start = validation_end - validation_days
        calibration_end = validation_start - purge_days
        calibration_start = calibration_end - calibration_days
        fit_end = calibration_start - calibration_purge
        inner_validation_end = fit_end
        inner_validation_start = inner_validation_end - inner_validation_days
        inner_fit_end = inner_validation_start - inner_purge
        if min(validation_start, calibration_start, fit_end, inner_validation_start, inner_fit_end) <= 0:
            continue
        if inner_fit_end < min_fit_days:
            continue

        val_dates = unique_dates[validation_start:validation_end]
        cal_dates = unique_dates[calibration_start:calibration_end]
        fit_dates = unique_dates[:fit_end]
        inner_fit_dates = unique_dates[:inner_fit_end]
        inner_val_dates = unique_dates[inner_validation_start:inner_validation_end]

        raw_dates = date_index.to_numpy()
        fit_idx = np.flatnonzero(np.isin(raw_dates, fit_dates.to_numpy()))
        cal_idx = np.flatnonzero(np.isin(raw_dates, cal_dates.to_numpy()))
        val_idx = np.flatnonzero(np.isin(raw_dates, val_dates.to_numpy()))
        inner_fit_idx = np.flatnonzero(np.isin(raw_dates, inner_fit_dates.to_numpy()))
        inner_val_idx = np.flatnonzero(np.isin(raw_dates, inner_val_dates.to_numpy()))
        if any(len(idx) == 0 for idx in (fit_idx, cal_idx, val_idx, inner_fit_idx, inner_val_idx)):
            continue
        y_val = y[val_idx]
        if int(y_val.sum()) < min_pos or int((1 - y_val).sum()) < min_neg:
            continue
        if len(np.unique(y[cal_idx])) < 2 or len(np.unique(y[inner_val_idx])) < 2 or len(np.unique(y[inner_fit_idx])) < 2:
            continue

        fit_min, fit_max = _date_text(raw_dates[fit_idx])
        cal_min, cal_max = _date_text(raw_dates[cal_idx])
        val_min, val_max = _date_text(raw_dates[val_idx])
        folds.append(AdaptiveFold(
            fold_id=fold_id,
            fit_idx=fit_idx.astype(np.int32),
            calibration_idx=cal_idx.astype(np.int32),
            validation_idx=val_idx.astype(np.int32),
            inner_fit_idx=inner_fit_idx.astype(np.int32),
            inner_validation_idx=inner_val_idx.astype(np.int32),
            validation_days=int(validation_days),
            fit_date_min=fit_min,
            fit_date_max=fit_max,
            calibration_date_min=cal_min,
            calibration_date_max=cal_max,
            validation_date_min=val_min,
            validation_date_max=val_max,
        ))
    return folds


def choose_adaptive_folds(dates: np.ndarray, y: np.ndarray, plan: dict[str, Any]) -> tuple[list[AdaptiveFold], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    minimum_folds = int(plan["minimum_outer_folds"])
    for window in plan["candidate_validation_days"]:
        folds = _build_folds_for_window(dates, y, int(window), plan)
        if folds:
            positives = [int(y[fold.validation_idx].sum()) for fold in folds]
            negatives = [int((1 - y[fold.validation_idx]).sum()) for fold in folds]
            score = (
                1000.0 * len(folds)
                + 8.0 * min(positives)
                + 2.0 * float(np.median(positives))
                + 0.25 * min(negatives)
                - 0.02 * abs(int(window) - 80)
            )
        else:
            positives, negatives, score = [], [], -1e9
        candidates.append({
            "validation_days": int(window),
            "fold_count": len(folds),
            "min_validation_positives": min(positives) if positives else 0,
            "median_validation_positives": float(np.median(positives)) if positives else 0.0,
            "min_validation_negatives": min(negatives) if negatives else 0,
            "selection_score": float(score),
            "folds": folds,
        })
    feasible = [row for row in candidates if row["fold_count"] >= minimum_folds]
    selected = max(feasible or candidates, key=lambda row: row["selection_score"])
    audit = {key: value for key, value in selected.items() if key != "folds"}
    audit["candidate_windows"] = [
        {key: value for key, value in row.items() if key != "folds"} for row in candidates
    ]
    return list(selected["folds"]), audit


def prepare_ticker_cache(project: Path, dataset: Path | None, result_dir: Path, plan: dict[str, Any]) -> pd.DataFrame:
    cache_root = result_dir / "ticker_cache"
    manifest_path = cache_root / "cache_manifest.json"
    bundle = load_refine_data(project, dataset, result_dir, plan)
    profile_name = str(plan.get("feature_profile", "full_reduced"))
    profile = bundle.profiles[profile_name]
    signature = f"{bundle.dataset_signature}::{profile_name}::{len(profile.features)}"

    if manifest_path.exists():
        try:
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            readiness_path = result_dir / "ticker_data_readiness.csv"
            if old.get("signature") == signature and readiness_path.exists():
                del bundle
                gc.collect()
                return pd.read_csv(readiness_path, dtype={"ticker": str})
        except Exception:
            pass

    if cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    feature_indices = np.asarray([bundle.feature_index[name] for name in profile.features], dtype=np.int32)
    normalized_tickers = pd.Series(bundle.tickers).astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6).to_numpy()
    tickers = sorted(pd.Series(normalized_tickers).drop_duplicates().tolist())
    rows_out: list[dict[str, Any]] = []

    for ticker in tickers:
        rows = np.flatnonzero(normalized_tickers == ticker)
        order = np.argsort(bundle.dates[rows], kind="mergesort")
        rows = rows[order]
        X = np.ascontiguousarray(bundle.X_all[rows][:, feature_indices], dtype=np.float32)
        y = np.asarray(bundle.y[rows], dtype=np.int8)
        dates = np.asarray(bundle.dates[rows], dtype="datetime64[ns]")
        bucket_values = pd.Series(bundle.buckets[rows]).dropna().astype(str)
        bucket = bucket_values.mode().iloc[0] if not bucket_values.empty else "other"
        folds, fold_audit = choose_adaptive_folds(dates, y, plan)
        ticker_dir = cache_root / ticker
        ticker_dir.mkdir(parents=True, exist_ok=True)
        np.save(ticker_dir / "X.npy", X, allow_pickle=False)
        np.save(ticker_dir / "y.npy", y, allow_pickle=False)
        np.save(ticker_dir / "dates.npy", dates, allow_pickle=False)
        metadata = {
            "ticker": ticker,
            "bucket": str(bucket),
            "features": list(profile.features),
            "rows": int(len(y)),
            "positives": int(y.sum()),
            "negative_count": int((1 - y).sum()),
            "positive_rate": float(y.mean()) if len(y) else None,
            "date_min": str(pd.Timestamp(dates.min()).date()) if len(dates) else "",
            "date_max": str(pd.Timestamp(dates.max()).date()) if len(dates) else "",
            "fold_audit": fold_audit,
            "signature": signature,
        }
        atomic_json(metadata, ticker_dir / "metadata.json")
        ready = (
            len(y) >= int(plan["min_ticker_rows"])
            and int(y.sum()) >= int(plan["min_total_positives"])
            and len(folds) >= int(plan["minimum_outer_folds"])
        )
        reasons: list[str] = []
        if len(y) < int(plan["min_ticker_rows"]):
            reasons.append("insufficient_rows")
        if int(y.sum()) < int(plan["min_total_positives"]):
            reasons.append("insufficient_positive_events")
        if len(folds) < int(plan["minimum_outer_folds"]):
            reasons.append("insufficient_feasible_folds")
        rows_out.append({
            "ticker": ticker,
            "bucket": str(bucket),
            "rows": int(len(y)),
            "positives": int(y.sum()),
            "negatives": int((1 - y).sum()),
            "positive_rate": float(y.mean()) if len(y) else np.nan,
            "date_min": metadata["date_min"],
            "date_max": metadata["date_max"],
            "fold_count": int(len(folds)),
            "validation_days": int(fold_audit.get("validation_days", 0)),
            "min_validation_positives": int(fold_audit.get("min_validation_positives", 0)),
            "ready": bool(ready),
            "reason": "|".join(reasons),
        })
        del X, y, dates
        gc.collect()

    readiness = pd.DataFrame(rows_out).sort_values(["ready", "positives", "rows"], ascending=[False, False, False])
    atomic_csv(readiness, result_dir / "ticker_data_readiness.csv")
    atomic_json({
        "signature": signature,
        "dataset_signature": bundle.dataset_signature,
        "profile": profile_name,
        "feature_count": len(profile.features),
        "ticker_count": len(tickers),
    }, manifest_path)
    del bundle
    gc.collect()
    return readiness


def list_cached_tickers(result_dir: Path) -> list[str]:
    root = result_dir / "ticker_cache"
    return sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "metadata.json").exists())


def load_cached_ticker(result_dir: Path, ticker: str, plan: dict[str, Any], mmap: bool = True) -> CachedTickerData:
    ticker = str(ticker).zfill(6)
    ticker_dir = result_dir / "ticker_cache" / ticker
    metadata = json.loads((ticker_dir / "metadata.json").read_text(encoding="utf-8"))
    mode = "r" if mmap else None
    X = np.load(ticker_dir / "X.npy", mmap_mode=mode, allow_pickle=False)
    y = np.load(ticker_dir / "y.npy", mmap_mode=mode, allow_pickle=False)
    dates = np.load(ticker_dir / "dates.npy", mmap_mode=mode, allow_pickle=False)
    folds, _ = choose_adaptive_folds(np.asarray(dates), np.asarray(y), plan)
    return CachedTickerData(
        ticker=ticker,
        bucket=str(metadata.get("bucket", "other")),
        X=X,
        y=y,
        dates=dates,
        features=list(metadata["features"]),
        folds=folds,
        cache_dir=ticker_dir,
    )


def apply_training_window(indices: np.ndarray, dates: np.ndarray, window_days: int) -> np.ndarray:
    if int(window_days) <= 0 or len(indices) == 0:
        return np.asarray(indices, dtype=np.int32)
    block_dates = pd.DatetimeIndex(np.asarray(dates)[indices]).unique().sort_values()
    if len(block_dates) <= int(window_days):
        return np.asarray(indices, dtype=np.int32)
    keep_dates = block_dates[-int(window_days):]
    mask = np.isin(np.asarray(dates)[indices], keep_dates.to_numpy())
    return np.asarray(indices, dtype=np.int32)[mask]
