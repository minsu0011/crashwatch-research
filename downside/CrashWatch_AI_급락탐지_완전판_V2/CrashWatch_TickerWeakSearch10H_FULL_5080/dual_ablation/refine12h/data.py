from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..config import get_paths, load_baskets
from ..experiment.splits import make_walk_forward_folds
from ..io_utils import atomic_csv, atomic_json, normalize_date, normalize_ticker, stable_hash


def _hash_file_metadata(path: Path) -> str:
    payload = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
    }
    return stable_hash(payload)


def _load_catalog(path: Path) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _merge_catalogs(*catalogs: dict[str, list[str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for catalog in catalogs:
        for group, columns in catalog.items():
            out[group] = sorted(set(out.get(group, [])) | set(columns))
    return out


def _feature_quality(
    df: pd.DataFrame,
    candidates: Iterable[str],
    min_ticker_coverage: float = 0.80,
) -> tuple[list[str], pd.DataFrame]:
    tickers = df["ticker"].astype(str)
    ticker_count = tickers.nunique()
    rows: list[dict[str, Any]] = []
    valid: list[str] = []
    for column in sorted(set(candidates)):
        if column not in df.columns:
            rows.append({
                "feature": column, "status": "missing_column",
                "missing_ratio": 1.0, "ticker_coverage": 0.0,
            })
            continue
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        nonnull_by_ticker = values.notna().groupby(tickers).sum()
        coverage = float((nonnull_by_ticker >= 100).sum() / max(1, ticker_count))
        missing = float(values.isna().mean())
        unique = int(values.nunique(dropna=True))
        nonnull = int(values.notna().sum())
        status = (
            "valid"
            if missing <= 0.98 and unique >= 2 and nonnull >= 500 and coverage >= min_ticker_coverage
            else "invalid"
        )
        rows.append({
            "feature": column, "status": status, "missing_ratio": missing,
            "unique_count": unique, "non_null_count": nonnull,
            "ticker_coverage": coverage, "required_ticker_coverage": min_ticker_coverage,
        })
        if status == "valid":
            valid.append(column)
    return valid, pd.DataFrame(rows)


@dataclass(frozen=True)
class FeatureProfile:
    name: str
    features: list[str]
    start_date: pd.Timestamp | None
    description: str


@dataclass
class RefineDataBundle:
    project: Path
    dataset_path: Path
    df: pd.DataFrame
    X_all: np.ndarray
    y: np.ndarray
    dates: np.ndarray
    tickers: np.ndarray
    buckets: np.ndarray
    all_features: list[str]
    feature_index: dict[str, int]
    feature_quality: pd.DataFrame
    redundant_features: list[str]
    high_missing_features: list[str]
    profiles: dict[str, FeatureProfile]
    common_start_date: pd.Timestamp
    folds: list[dict[str, Any]]
    dataset_signature: str
    regime_features: list[str]

    def matrix(self, rows: np.ndarray, profile: str) -> tuple[np.ndarray, list[str]]:
        selected = self.profiles[profile].features
        indices = np.asarray([self.feature_index[name] for name in selected], dtype=np.int32)
        return np.ascontiguousarray(self.X_all[rows][:, indices]), selected

    def profile_row_mask(self, profile: str) -> np.ndarray:
        start = self.profiles[profile].start_date
        if start is None:
            return np.ones(len(self.df), dtype=bool)
        return self.dates >= np.datetime64(start)

    def fold_indices(self, fold_id: int, profile: str) -> dict[str, np.ndarray]:
        fold = next(item for item in self.folds if int(item["fold_id"]) == int(fold_id))
        train_dates = pd.DatetimeIndex(fold["train_dates_index"])
        validation_dates = pd.DatetimeIndex(fold["validation_dates_index"])
        mask = self.profile_row_mask(profile)
        train_idx_all = np.flatnonzero(mask & np.isin(self.dates, train_dates.to_numpy()))
        validation_idx = np.flatnonzero(mask & np.isin(self.dates, validation_dates.to_numpy()))
        unique_train = pd.DatetimeIndex(np.unique(self.dates[train_idx_all])).sort_values()
        cal_days = 60
        purge_days = 20
        if len(unique_train) < cal_days + purge_days + 300:
            raise RuntimeError(f"fold={fold_id}, profile={profile}: insufficient training dates ({len(unique_train)})")
        calibration_dates = unique_train[-cal_days:]
        fit_dates = unique_train[: -(cal_days + purge_days)]
        calibration_idx = np.flatnonzero(mask & np.isin(self.dates, calibration_dates.to_numpy()))
        fit_idx = np.flatnonzero(mask & np.isin(self.dates, fit_dates.to_numpy()))
        return {
            "fit_idx": fit_idx,
            "calibration_idx": calibration_idx,
            "validation_idx": validation_idx,
            "fit_dates": fit_dates.to_numpy(),
            "calibration_dates": calibration_dates.to_numpy(),
            "validation_dates": validation_dates.to_numpy(),
        }

    def regime_matrix(self, rows: np.ndarray) -> np.ndarray | None:
        if not self.regime_features:
            return None
        indices = np.asarray([self.feature_index[name] for name in self.regime_features], dtype=np.int32)
        return np.asarray(self.X_all[rows][:, indices], dtype=np.float64)


def _catalog(paths) -> dict[str, list[str]]:
    return _merge_catalogs(
        _load_catalog(paths.feature_dual / "feature_catalog_universe.json"),
        _load_catalog(paths.feature_dual / "feature_catalog_ticker.json"),
        _load_catalog(paths.feature_dual / "feature_catalog_finance11h_universe.json"),
        _load_catalog(paths.feature_dual / "feature_catalog_finance11h_ticker.json"),
    )


def _compute_redundant(matrix: pd.DataFrame, quality_map: dict[str, float], threshold: float) -> list[str]:
    if matrix.shape[1] < 2:
        return []
    sample_rows = np.linspace(0, len(matrix) - 1, min(len(matrix), 30000), dtype=int)
    sample = matrix.iloc[sample_rows].copy()
    sample = sample.fillna(sample.median(numeric_only=True)).fillna(0.0)
    corr = sample.corr(method="pearson").abs()
    columns = list(corr.columns)
    dropped: set[str] = set()
    for i, left in enumerate(columns):
        if left in dropped:
            continue
        for right in columns[i + 1 :]:
            if right in dropped:
                continue
            value = corr.at[left, right]
            if np.isfinite(value) and value >= threshold:
                left_missing = float(quality_map.get(left, 0.0))
                right_missing = float(quality_map.get(right, 0.0))
                dropped.add(right if right_missing >= left_missing else left)
                if left in dropped:
                    break
    return sorted(dropped)


def _load_existing_redundant(paths, result_dir: Path) -> list[str]:
    candidates = [
        paths.data_root / "ablation_base12h" / "redundant_features.csv",
        result_dir.parent / "ablation_base12h" / "redundant_features.csv",
        result_dir / "imported_base12h" / "redundant_features.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "feature" in frame.columns:
            values = frame["feature"].dropna().astype(str).drop_duplicates().tolist()
            if values:
                return values
    return []


def _common_start(
    df: pd.DataFrame,
    features: list[str],
    *,
    threshold: float,
    sustain_days: int,
    min_days: int,
) -> tuple[pd.Timestamp, pd.DataFrame]:
    finite = df[features].replace([np.inf, -np.inf], np.nan).notna()
    daily = finite.groupby(df["date"], sort=True).mean().mean(axis=1).rename("daily_feature_coverage")
    rolling = daily.rolling(60, min_periods=60).median().rename("rolling_60d_median")
    dates = pd.DatetimeIndex(daily.index)
    selected: pd.Timestamp | None = None
    for index in range(59, max(60, len(dates) - sustain_days + 1)):
        future = rolling.iloc[index : index + sustain_days]
        remaining_days = len(dates) - index
        if remaining_days < min_days:
            break
        if len(future) >= sustain_days and float((future >= threshold).mean()) >= 0.90:
            selected = pd.Timestamp(dates[index])
            break
    if selected is None:
        fallback_index = max(0, len(dates) - min_days)
        selected = pd.Timestamp(dates[fallback_index])
    report = pd.concat([daily, rolling], axis=1).reset_index().rename(columns={"index": "date"})
    report["selected_common_start"] = report["date"].eq(selected)
    return selected, report


def _select_regime_features(catalog: dict[str, list[str]], valid: set[str], quality_map: dict[str, float]) -> list[str]:
    candidates: list[str] = []
    group_priority = ["u_financial_shorting", "u_etf_pressure", "u_financial_market", "u_market_volatility"]
    keywords = ["vix", "vkospi", "vol", "short", "credit", "spread", "stress", "breadth", "market_ret", "kospi"]
    for group in group_priority:
        candidates.extend(catalog.get(group, []))
    scored = []
    for name in sorted(set(candidates)):
        if name not in valid or quality_map.get(name, 1.0) > 0.35:
            continue
        keyword_score = sum(word in name.lower() for word in keywords)
        scored.append((keyword_score, -quality_map.get(name, 1.0), name))
    scored.sort(reverse=True)
    return [name for _, _, name in scored[:8]]


def load_refine_data(
    project: Path,
    dataset: Path | None,
    result_dir: Path,
    plan: dict[str, Any],
) -> RefineDataBundle:
    project = project.resolve()
    paths = get_paths(project)
    dataset_path = (dataset or paths.data_root / "development" / "training_dataset_finance11h.parquet").resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Finance11H dataset missing: {dataset_path}")
    df = normalize_date(pd.read_parquet(dataset_path))
    df["ticker"] = normalize_ticker(df["ticker"])
    if "label_abs_crash_20" not in df.columns:
        raise KeyError("label_abs_crash_20 missing")
    baskets = load_baskets(paths)
    if "bucket" not in df.columns:
        df = df.merge(baskets[["ticker", "bucket"]], on="ticker", how="left")
    df["bucket"] = df["bucket"].fillna("other").astype(str)
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    catalog = _catalog(paths)
    candidates = sorted(set(sum(catalog.values(), [])))
    valid, quality = _feature_quality(df, candidates)
    valid_set = set(valid)
    quality_map = quality.set_index("feature")["missing_ratio"].to_dict() if not quality.empty else {}
    numeric = df[valid].replace([np.inf, -np.inf], np.nan).apply(pd.to_numeric, errors="coerce")

    redundant = _load_existing_redundant(paths, result_dir)
    redundant = [name for name in redundant if name in valid_set]
    if not redundant:
        redundant = _compute_redundant(numeric, quality_map, float(plan["redundant_correlation_threshold"]))
    high_missing = sorted([name for name in valid if float(quality_map.get(name, 0.0)) >= float(plan["high_missing_threshold"])])
    full_features = [name for name in valid if name not in set(redundant)]
    common_features = [name for name in full_features if name not in set(high_missing)]
    if len(common_features) < 50:
        raise RuntimeError(f"too few common-period features: {len(common_features)}")

    common_start, common_report = _common_start(
        df,
        common_features,
        threshold=float(plan["common_daily_coverage_threshold"]),
        sustain_days=int(plan["common_sustain_days"]),
        min_days=int(plan["common_min_days"]),
    )
    common_dates = pd.DatetimeIndex(df.loc[df["date"] >= common_start, "date"].unique()).sort_values()
    folds = make_walk_forward_folds(
        common_dates,
        n_folds=int(plan["outer_folds"]),
        validation_days=int(plan["validation_days"]),
        purge_days=int(plan["purge_days"]),
        min_train_days=int(plan["min_train_days"]),
        min_train_fraction=0.45,
    )
    X_all = np.ascontiguousarray(numeric.to_numpy(dtype=np.float32))
    y = pd.to_numeric(df["label_abs_crash_20"], errors="raise").to_numpy(dtype=np.int8)
    regime_features = _select_regime_features(catalog, valid_set, quality_map)
    signature = stable_hash({
        "dataset": _hash_file_metadata(dataset_path),
        "rows": len(df),
        "valid_features": valid,
        "redundant": redundant,
        "high_missing": high_missing,
        "common_start": str(common_start),
        "schema": plan["schema_version"],
    })

    profiles = {
        "full_reduced": FeatureProfile(
            "full_reduced", full_features, None,
            "75개 고상관 중복 피처를 제거하고 전체 기간을 사용하는 모델",
        ),
        "common_period": FeatureProfile(
            "common_period", common_features, common_start,
            "고결측 피처를 제거하고 공통 데이터 가용 기간만 사용하는 모델",
        ),
    }
    quality = quality.copy()
    quality["redundant_drop"] = quality["feature"].isin(redundant)
    quality["high_missing_drop"] = quality["feature"].isin(high_missing)
    quality["selected_full_reduced"] = quality["feature"].isin(full_features)
    quality["selected_common_period"] = quality["feature"].isin(common_features)
    atomic_csv(quality, result_dir / "feature_quality_refine.csv")
    atomic_csv(pd.DataFrame({"feature": redundant}), result_dir / "redundant_features_maintained.csv")
    atomic_csv(pd.DataFrame({"feature": high_missing}), result_dir / "high_missing_features_removed.csv")
    atomic_csv(common_report, result_dir / "common_period_coverage.csv")
    atomic_json({
        "dataset_path": str(dataset_path),
        "dataset_signature": signature,
        "rows": len(df),
        "tickers": int(df["ticker"].nunique()),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "valid_feature_count": len(valid),
        "redundant_feature_count": len(redundant),
        "full_reduced_feature_count": len(full_features),
        "high_missing_feature_count": len(high_missing),
        "common_feature_count": len(common_features),
        "common_start_date": str(common_start.date()),
        "regime_features": regime_features,
    }, result_dir / "refine_data_summary.json")

    return RefineDataBundle(
        project=project,
        dataset_path=dataset_path,
        df=df,
        X_all=X_all,
        y=y,
        dates=df["date"].to_numpy(),
        tickers=df["ticker"].astype(str).to_numpy(),
        buckets=df["bucket"].astype(str).to_numpy(),
        all_features=valid,
        feature_index={name: index for index, name in enumerate(valid)},
        feature_quality=quality,
        redundant_features=redundant,
        high_missing_features=high_missing,
        profiles=profiles,
        common_start_date=common_start,
        folds=folds,
        dataset_signature=signature,
        regime_features=regime_features,
    )
