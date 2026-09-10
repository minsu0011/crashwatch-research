from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


TARGET_NAME = "label_abs_surge_3d_5pct"
TARGET_VALID_COLUMN = "target_valid"
SURGE_HORIZON = 3
SURGE_THRESHOLD = 0.05
SURGE_COMPARISON_ATOL = 1e-12
DEFAULT_THRESHOLDS = (0.80, 0.90, 0.92, 0.95, 0.98)
DEFAULT_PRIMARY_THRESHOLD = 0.92
DEFAULT_FOLD_ROLES = {
    "selection": [0, 1, 2, 3, 4],
    "confirmation": [5, 6],
    "recent_audit": [7],
}
EXPLICIT_LEAKAGE_COLUMNS = {
    TARGET_NAME,
    TARGET_VALID_COLUMN,
    "first_hit_day",
    "best_forward_return_3d",
    "source_row_id",
}
LEAKAGE_NAME_FRAGMENTS = (
    "forward_return",
    "future_return",
    "future_ret",
    "fwd_return",
    "fwd_ret",
    "next_return",
    "next_ret",
    "first_hit",
    "best_forward",
    "days_to_hit",
    "target_valid",
)


@dataclass(frozen=True)
class FoldDefinition:
    fold_id: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    purge_start: str = ""
    purge_end: str = ""


@dataclass
class MatrixBundle:
    pearson: np.ndarray
    spearman: np.ndarray
    within_date: np.ndarray
    within_ticker: np.ndarray
    missingness: np.ndarray
    combined_abs: np.ndarray
    cluster_basis_abs: np.ndarray
    fold_ids: np.ndarray
    fold_pearson: np.ndarray
    source: str
    source_manifest: dict[str, Any]


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        parent = self.parent
        root = item
        while parent[root] != root:
            root = int(parent[root])
        while parent[item] != item:
            next_item = int(parent[item])
            parent[item] = root
            item = next_item
        return root

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def stable_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_feature_list(features: Sequence[str]) -> str:
    return sha256_bytes("\n".join(features).encode("utf-8"))


def hash_arrays(features: Sequence[str], arrays: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    digest.update("\n".join(features).encode("utf-8"))
    for array in arrays:
        contiguous = np.ascontiguousarray(array, dtype=np.float32)
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(frame: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    is_gzip = path.name.lower().endswith(".gz")
    suffix = ".tmp.gz" if is_gzip else ".tmp"
    temporary = path.with_name(f".{path.stem}.{os.getpid()}{suffix}")
    frame.to_csv(temporary, index=index, compression="gzip" if is_gzip else None)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


class RunLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(
                f"실행 lock이 이미 존재합니다: {self.path}. 다른 실행이 없으면 lock 파일을 삭제하십시오."
            ) from error
        os.write(self.fd, f"pid={os.getpid()} started={utc_now()}\n".encode("utf-8"))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        self.path.unlink(missing_ok=True)


class RunTracker:
    def __init__(self, output_dir: Path, config: dict[str, Any]) -> None:
        self.path = output_dir / "RUN_STATUS.json"
        self.started_perf_counter = time.perf_counter()
        self.state: dict[str, Any] = {
            "status": "RUNNING",
            "started_at": utc_now(),
            "updated_at": utc_now(),
            "config_hash": sha256_bytes(stable_json_bytes(config)),
            "config": config,
            "stages": {},
        }
        atomic_json(self.state, self.path)

    def stage(self, name: str, status: str, **details: object) -> None:
        self.state["stages"][name] = {
            "status": status,
            "updated_at": utc_now(),
            **details,
        }
        self.state["updated_at"] = utc_now()
        atomic_json(self.state, self.path)

    def complete(self, **details: object) -> None:
        self.state.update(
            {
                "status": "SUCCESS",
                "completed_at": utc_now(),
                "elapsed_seconds": time.perf_counter() - self.started_perf_counter,
                **details,
            }
        )
        self.state["updated_at"] = utc_now()
        atomic_json(self.state, self.path)

    def fail(self, error: BaseException) -> None:
        self.state.update(
            {
                "status": "FAILED",
                "failed_at": utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "error_message": str(error),
                "elapsed_seconds": time.perf_counter() - self.started_perf_counter,
                "traceback": traceback.format_exc(),
            }
        )
        self.state["updated_at"] = utc_now()
        atomic_json(self.state, self.path)


def normalize_ticker(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        if series.isna().any():
            raise ValueError(f"boolean 열에 결측값이 있습니다: {int(series.isna().sum())}행")
        return series.astype(bool)
    numeric = pd.to_numeric(series, errors="coerce")
    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    numeric_mask = numeric.notna()
    invalid_numeric = numeric_mask & ~numeric.isin([0, 1])
    if invalid_numeric.any():
        samples = series[invalid_numeric].astype(str).head(10).tolist()
        raise ValueError(f"boolean 열은 0/1만 허용합니다: samples={samples}")
    result.loc[numeric_mask] = numeric.loc[numeric_mask].astype(np.int8).astype(bool)
    text_mask = ~numeric_mask & series.notna()
    normalized = series.loc[text_mask].astype(str).str.strip().str.lower()
    true_values = {"true", "t", "yes", "y"}
    false_values = {"false", "f", "no", "n"}
    recognized = normalized.isin(true_values | false_values)
    if (~recognized).any():
        samples = series.loc[normalized.index[~recognized]].astype(str).head(10).tolist()
        raise ValueError(f"boolean 열에 해석할 수 없는 값이 있습니다: samples={samples}")
    result.loc[normalized.index] = normalized.isin(true_values).to_numpy()
    if result.isna().any():
        raise ValueError(f"boolean 열에 결측값이 있습니다: {int(result.isna().sum())}행")
    return result.astype(bool)


def require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Parquet 입력에는 pyarrow가 필요합니다. "
            "`python -m pip install -r starter_code/requirements_surge_correlation.txt`를 실행하십시오."
        ) from error
    return pa, pq


def table_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"} or path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path, nrows=0).columns.astype(str).tolist()
    if suffix == ".parquet":
        _, pq = require_pyarrow()
        return pq.ParquetFile(path).schema_arrow.names
    raise ValueError(f"지원하지 않는 입력 형식입니다: {path}")


def table_row_count(path: Path) -> int:
    if path.suffix.lower() == ".parquet":
        _, pq = require_pyarrow()
        return int(pq.ParquetFile(path).metadata.num_rows)
    if path.suffix.lower() == ".csv" or path.name.lower().endswith(".csv.gz"):
        return int(sum(1 for _ in path.open("rb")) - 1) if path.suffix.lower() == ".csv" else int(len(pd.read_csv(path, usecols=[0])))
    raise ValueError(f"지원하지 않는 입력 형식입니다: {path}")


def read_table_columns(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    unique_columns = list(dict.fromkeys(columns))
    available = set(table_columns(path))
    missing = [column for column in unique_columns if column not in available]
    if missing:
        raise KeyError(f"{path.name}에 필요한 열이 없습니다: {missing[:20]}")
    if path.suffix.lower() == ".parquet":
        _, pq = require_pyarrow()
        table = pq.read_table(path, columns=unique_columns, memory_map=True, use_threads=True)
        return table.to_pandas(split_blocks=True)
    return pd.read_csv(path, usecols=unique_columns, low_memory=False)


def leakage_reason(column: str) -> str | None:
    lowered = column.lower()
    if column in EXPLICIT_LEAKAGE_COLUMNS:
        return "explicit_target_or_sidecar_column"
    if lowered.startswith("label_abs_crash_") or lowered.startswith("label_idio_crash_"):
        return "legacy_crash_label"
    if lowered.startswith("label_"):
        return "label_prefix"
    if lowered == "target" or lowered.startswith("target_") or lowered.endswith("_target"):
        return "target_name"
    for fragment in LEAKAGE_NAME_FRAGMENTS:
        if fragment in lowered:
            return f"future_or_target_fragment:{fragment}"
    return None


def load_profile_features(profile_manifest: Path | None, profile_name: str) -> list[str]:
    if profile_manifest is None or not profile_manifest.exists():
        return []
    payload = json.loads(profile_manifest.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {})
    if profile_name not in profiles:
        available = ", ".join(sorted(profiles))
        raise KeyError(f"profile `{profile_name}`가 없습니다. 사용 가능: {available}")
    features = profiles[profile_name].get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError(f"profile `{profile_name}`의 feature 목록이 비어 있습니다")
    return [str(feature) for feature in features]


def read_feature_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            payload = payload.get("features", [])
        if not isinstance(payload, list):
            raise ValueError("feature-list JSON은 배열 또는 features 배열을 가진 객체여야 합니다")
        return [str(item) for item in payload]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def infer_candidate_features(columns: Sequence[str]) -> list[str]:
    candidates: list[str] = []
    for column in columns:
        if leakage_reason(column) is not None:
            continue
        if column in {"date", "ticker", "close", "sealed_do_not_train_or_tune"}:
            continue
        if column.startswith(("t_", "u_")):
            candidates.append(column)
    return candidates


def parse_int_list(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("threshold 목록이 비어 있습니다")
    return sorted(set(values))


def load_fold_definitions(path: Path) -> list[FoldDefinition]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("fold JSON은 배열이어야 합니다")
    result: list[FoldDefinition] = []
    for item in payload:
        result.append(
            FoldDefinition(
                fold_id=int(item["fold_id"]),
                train_start=str(item["train_start"]),
                train_end=str(item["train_end"]),
                validation_start=str(item["validation_start"]),
                validation_end=str(item["validation_end"]),
                purge_start=str(item.get("purge_start", "")),
                purge_end=str(item.get("purge_end", "")),
            )
        )
    return result


def role_for_fold(fold_id: int, roles: dict[str, list[int]]) -> str:
    for role, fold_ids in roles.items():
        if fold_id in fold_ids:
            return role
    return "unassigned"


def validate_fold_definitions(
    dates: pd.Series,
    target_valid: np.ndarray,
    target: np.ndarray,
    definitions: Sequence[FoldDefinition],
    roles: dict[str, list[int]],
    minimum_purge_days: int,
) -> list[dict[str, Any]]:
    date_values = pd.to_datetime(dates, errors="coerce")
    if date_values.isna().any():
        raise ValueError(f"date 파싱 실패 행={int(date_values.isna().sum())}")
    unique_dates = np.array(sorted(date_values.unique()))
    rows: list[dict[str, Any]] = []
    for fold in definitions:
        train_start = pd.Timestamp(fold.train_start)
        train_end = pd.Timestamp(fold.train_end)
        validation_start = pd.Timestamp(fold.validation_start)
        validation_end = pd.Timestamp(fold.validation_end)
        train_mask = target_valid & date_values.between(train_start, train_end).to_numpy()
        validation_mask = target_valid & date_values.between(validation_start, validation_end).to_numpy()
        train_dates = unique_dates[(unique_dates >= np.datetime64(train_start)) & (unique_dates <= np.datetime64(train_end))]
        validation_dates = unique_dates[
            (unique_dates >= np.datetime64(validation_start)) & (unique_dates <= np.datetime64(validation_end))
        ]
        purge_dates = unique_dates[(unique_dates > np.datetime64(train_end)) & (unique_dates < np.datetime64(validation_start))]
        reasons: list[str] = []
        if train_end >= validation_start:
            reasons.append("train_validation_overlap")
        if len(purge_dates) < minimum_purge_days:
            reasons.append(f"purge_dates={len(purge_dates)}<{minimum_purge_days}")
        if not train_mask.any():
            reasons.append("empty_train")
        if not validation_mask.any():
            reasons.append("empty_validation")
        rows.append(
            {
                "fold_id": fold.fold_id,
                "fold_role": role_for_fold(fold.fold_id, roles),
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "purge_start": fold.purge_start,
                "purge_end": fold.purge_end,
                "validation_start": validation_start.strftime("%Y-%m-%d"),
                "validation_end": validation_end.strftime("%Y-%m-%d"),
                "train_dates": int(len(train_dates)),
                "purge_dates": int(len(purge_dates)),
                "validation_dates": int(len(validation_dates)),
                "train_rows_target_valid": int(train_mask.sum()),
                "train_positives": int(target[train_mask].sum()),
                "validation_rows_target_valid": int(validation_mask.sum()),
                "validation_positives": int(target[validation_mask].sum()),
                "eligible": not reasons,
                "reason": ";".join(reasons),
            }
        )
    invalid = [row for row in rows if not row["eligible"]]
    if invalid:
        raise ValueError(f"부적격 fold가 있습니다: {invalid}")
    return rows


def deterministic_complete_date_sample(
    dates: np.ndarray,
    eligible_mask: np.ndarray,
    maximum_rows: int,
    seed: int,
) -> np.ndarray:
    eligible_indices = np.flatnonzero(eligible_mask)
    if maximum_rows <= 0 or len(eligible_indices) <= maximum_rows:
        return eligible_indices
    eligible_dates = dates[eligible_indices]
    unique_dates, counts = np.unique(eligible_dates, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique_dates))
    chosen_dates: list[np.datetime64] = []
    total = 0
    for position in order:
        chosen_dates.append(unique_dates[position])
        total += int(counts[position])
        if total >= maximum_rows:
            break
    chosen = np.isin(dates, np.asarray(chosen_dates)) & eligible_mask
    return np.flatnonzero(chosen)


def effective_min_periods(requested: int, rows: int) -> int:
    if rows <= 2:
        return 2
    if requested <= rows:
        return max(2, requested)
    # Small synthetic fixtures should remain testable without weakening real-data thresholds.
    return max(2, rows // 5)


def pairwise_complete_corr(values: np.ndarray, min_periods: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values는 2차원이어야 합니다")
    rows, columns = values.shape
    if columns == 0:
        return np.empty((0, 0), dtype=np.float32)
    minimum = effective_min_periods(min_periods, rows)
    finite = np.isfinite(values)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        means = np.nanmean(values, axis=0)
        scales = np.nanstd(values, axis=0)
    scales[~np.isfinite(scales) | (scales <= 0)] = 1.0
    standardized = (values - means) / scales
    standardized[~finite] = 0.0
    mask = finite.astype(np.float64)
    counts = mask.T @ mask
    sums_left = standardized.T @ mask
    sums_right = sums_left.T
    products = standardized.T @ standardized
    squares_left = (standardized * standardized).T @ mask
    squares_right = squares_left.T
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance_numerator = products - (sums_left * sums_right / counts)
        variance_left = squares_left - (sums_left * sums_left / counts)
        variance_right = squares_right - (sums_right * sums_right / counts)
        denominator = np.sqrt(np.maximum(variance_left, 0.0) * np.maximum(variance_right, 0.0))
        correlation = covariance_numerator / denominator
    correlation[(counts < minimum) | (denominator <= 0)] = np.nan
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    finite_counts = finite.sum(axis=0)
    valid_diagonal = finite_counts >= minimum
    diagonal = np.arange(columns)
    correlation[diagonal[valid_diagonal], diagonal[valid_diagonal]] = 1.0
    correlation[diagonal[~valid_diagonal], diagonal[~valid_diagonal]] = np.nan
    return correlation.astype(np.float32)


def residualize_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame(values)
    group_series = pd.Series(groups, index=frame.index)
    means = frame.groupby(group_series, sort=False, observed=True).transform("mean")
    return (frame - means).to_numpy(dtype=np.float64, copy=False)


def combined_absolute_correlation(*matrices: np.ndarray) -> np.ndarray:
    if not matrices:
        raise ValueError("matrix가 하나 이상 필요합니다")
    stack = np.stack([np.abs(np.asarray(matrix, dtype=np.float32)) for matrix in matrices], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = np.nanmax(stack, axis=0)
    result[~np.isfinite(result)] = 0.0
    np.fill_diagonal(result, 1.0)
    return result.astype(np.float32)


def compute_feature_structure_on_rows(
    feature_frame: pd.DataFrame,
    dates: np.ndarray,
    tickers: np.ndarray,
    row_indices: np.ndarray,
    spearman_row_indices: np.ndarray,
    min_periods: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sampled = feature_frame.iloc[row_indices].to_numpy(dtype=np.float64, copy=True)
    sampled_dates = dates[row_indices]
    sampled_tickers = tickers[row_indices]
    pearson = pairwise_complete_corr(sampled, min_periods)
    rank_frame = pd.DataFrame(feature_frame.iloc[spearman_row_indices].to_numpy(dtype=np.float64, copy=True)).rank(
        method="average", na_option="keep"
    )
    spearman = pairwise_complete_corr(rank_frame.to_numpy(dtype=np.float64, copy=False), min_periods)
    within_date = pairwise_complete_corr(residualize_by_group(sampled, sampled_dates), min_periods)
    within_ticker = pairwise_complete_corr(residualize_by_group(sampled, sampled_tickers), min_periods)
    missingness = pairwise_complete_corr(np.isnan(sampled).astype(np.float64), min_periods)
    missingness[~np.isfinite(missingness)] = 0.0
    np.fill_diagonal(missingness, 1.0)
    combined = combined_absolute_correlation(pearson, spearman, within_date, within_ticker)
    return pearson, spearman, within_date, within_ticker, missingness, combined


def vector_target_corr(values: np.ndarray, target: np.ndarray, min_periods: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values는 2차원이어야 합니다")
    finite_target = np.isfinite(target)
    finite = np.isfinite(values) & finite_target[:, None]
    clean_values = np.where(finite, values, 0.0)
    clean_target = np.where(finite_target, target, 0.0)
    counts = finite.sum(axis=0).astype(np.float64)
    sums_x = clean_values.sum(axis=0)
    sums_x2 = (clean_values * clean_values).sum(axis=0)
    sums_y = finite.T @ clean_target
    sums_y2 = finite.T @ (clean_target * clean_target)
    sums_xy = clean_values.T @ clean_target
    with np.errstate(divide="ignore", invalid="ignore"):
        covariance = sums_xy - (sums_x * sums_y / counts)
        variance_x = sums_x2 - (sums_x * sums_x / counts)
        variance_y = sums_y2 - (sums_y * sums_y / counts)
        denominator = np.sqrt(np.maximum(variance_x, 0.0) * np.maximum(variance_y, 0.0))
        correlation = covariance / denominator
    minimum = effective_min_periods(min_periods, len(target))
    correlation[(counts < minimum) | (denominator <= 0)] = np.nan
    return correlation.astype(np.float64), counts.astype(np.int64), sums_y.astype(np.int64)


def residualize_vector_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    group_series = pd.Series(groups, index=series.index)
    means = series.groupby(group_series, sort=False, observed=True).transform("mean")
    return (series - means).to_numpy(dtype=np.float64)


def compute_target_correlations(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    dates: np.ndarray,
    tickers: np.ndarray,
    feature_names: Sequence[str],
    min_periods: int,
    chunk_size: int,
    include_grouped: bool,
) -> pd.DataFrame:
    target = np.asarray(target, dtype=np.float64)
    if len(feature_frame) != len(target):
        raise ValueError("feature와 target 행 수가 다릅니다")
    date_target_residual = residualize_vector_by_group(target, dates) if include_grouped else None
    ticker_target_residual = residualize_vector_by_group(target, tickers) if include_grouped else None
    records: list[dict[str, Any]] = []
    for start in range(0, len(feature_names), chunk_size):
        names = list(feature_names[start : start + chunk_size])
        values = feature_frame[names].to_numpy(dtype=np.float64, copy=True)
        pearson, counts, positives = vector_target_corr(values, target, min_periods)
        ranks = pd.DataFrame(values).rank(method="average", na_option="keep").to_numpy(dtype=np.float64, copy=False)
        spearman, _, _ = vector_target_corr(ranks, target, min_periods)
        if include_grouped:
            within_date_values = residualize_by_group(values, dates)
            within_ticker_values = residualize_by_group(values, tickers)
            within_date, _, _ = vector_target_corr(within_date_values, date_target_residual, min_periods)
            within_ticker, _, _ = vector_target_corr(within_ticker_values, ticker_target_residual, min_periods)
        else:
            within_date = np.full(len(names), np.nan, dtype=np.float64)
            within_ticker = np.full(len(names), np.nan, dtype=np.float64)
        for offset, name in enumerate(names):
            records.append(
                {
                    "feature": name,
                    "target_pearson": float(pearson[offset]) if np.isfinite(pearson[offset]) else np.nan,
                    "target_spearman": float(spearman[offset]) if np.isfinite(spearman[offset]) else np.nan,
                    "target_within_date_pearson": (
                        float(within_date[offset]) if np.isfinite(within_date[offset]) else np.nan
                    ),
                    "target_within_ticker_pearson": (
                        float(within_ticker[offset]) if np.isfinite(within_ticker[offset]) else np.nan
                    ),
                    "finite_rows": int(counts[offset]),
                    "positive_rows": int(positives[offset]),
                    "positive_rate": float(positives[offset] / counts[offset]) if counts[offset] else np.nan,
                }
            )
    return pd.DataFrame.from_records(records)


def compute_fold_target_correlations(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    target_valid: np.ndarray,
    dates: np.ndarray,
    tickers: np.ndarray,
    features: Sequence[str],
    definitions: Sequence[FoldDefinition],
    roles: dict[str, list[int]],
    train_sample_rows: int,
    min_periods: int,
    chunk_size: int,
    seed: int,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    date_series = pd.Series(pd.to_datetime(dates))
    for fold in definitions:
        train_mask = target_valid & date_series.between(fold.train_start, fold.train_end).to_numpy()
        validation_mask = target_valid & date_series.between(fold.validation_start, fold.validation_end).to_numpy()
        train_indices = deterministic_complete_date_sample(
            dates,
            train_mask,
            train_sample_rows,
            seed + fold.fold_id * 1009 + 17,
        )
        validation_indices = np.flatnonzero(validation_mask)
        if not len(train_indices) or not len(validation_indices):
            raise ValueError(f"fold {fold.fold_id}의 train 또는 validation이 비었습니다")
        train_corr = compute_target_correlations(
            feature_frame.iloc[train_indices],
            target[train_indices],
            dates[train_indices],
            tickers[train_indices],
            features,
            min_periods,
            chunk_size,
            include_grouped=False,
        ).rename(
            columns={
                "target_pearson": "train_target_pearson",
                "target_spearman": "train_target_spearman",
                "finite_rows": "train_finite_rows",
                "positive_rows": "train_positive_rows",
                "positive_rate": "train_positive_rate",
            }
        )
        train_corr = train_corr[
            [
                "feature",
                "train_target_pearson",
                "train_target_spearman",
                "train_finite_rows",
                "train_positive_rows",
                "train_positive_rate",
            ]
        ]
        validation_corr = compute_target_correlations(
            feature_frame.iloc[validation_indices],
            target[validation_indices],
            dates[validation_indices],
            tickers[validation_indices],
            features,
            min_periods,
            chunk_size,
            include_grouped=False,
        ).rename(
            columns={
                "target_pearson": "validation_target_pearson",
                "target_spearman": "validation_target_spearman",
                "finite_rows": "validation_finite_rows",
                "positive_rows": "validation_positive_rows",
                "positive_rate": "validation_positive_rate",
            }
        )
        validation_corr = validation_corr[
            [
                "feature",
                "validation_target_pearson",
                "validation_target_spearman",
                "validation_finite_rows",
                "validation_positive_rows",
                "validation_positive_rate",
            ]
        ]
        merged = train_corr.merge(validation_corr, on="feature", how="inner", validate="one_to_one")
        merged.insert(0, "outer_fold", fold.fold_id)
        merged.insert(1, "fold_role", role_for_fold(fold.fold_id, roles))
        merged.insert(2, "train_start", fold.train_start)
        merged.insert(3, "train_end", fold.train_end)
        merged.insert(4, "validation_start", fold.validation_start)
        merged.insert(5, "validation_end", fold.validation_end)
        merged.insert(6, "train_sample_rows_total", int(len(train_indices)))
        merged.insert(7, "validation_rows_total", int(len(validation_indices)))
        records.append(merged)
        log(
            f"fold {fold.fold_id}: train sample {len(train_indices):,}, validation {len(validation_indices):,}, "
            f"role={role_for_fold(fold.fold_id, roles)}"
        )
    return pd.concat(records, ignore_index=True)


def sign_consistency(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array) & (np.abs(array) > 1e-12)]
    if not len(array):
        return np.nan
    positive = float(np.mean(array > 0))
    negative = float(np.mean(array < 0))
    return max(positive, negative)


def aggregate_metric(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "mean": np.nan,
            "std": np.nan,
            "abs_mean": np.nan,
            "abs_min": np.nan,
            "abs_max": np.nan,
            "sign_consistency": np.nan,
            "folds": 0,
        }
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "abs_mean": float(np.mean(np.abs(values))),
        "abs_min": float(np.min(np.abs(values))),
        "abs_max": float(np.max(np.abs(values))),
        "sign_consistency": float(sign_consistency(values)),
        "folds": int(len(values)),
    }


def summarize_fold_target_correlations(
    fold_correlations: pd.DataFrame,
    features: Sequence[str],
    roles: dict[str, list[int]],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouped = fold_correlations.groupby("feature", sort=False)
    for feature in features:
        part = grouped.get_group(feature)
        record: dict[str, Any] = {"feature": feature}
        for split in ("train", "validation"):
            for metric in ("pearson", "spearman"):
                column = f"{split}_target_{metric}"
                all_stats = aggregate_metric(part[column].to_numpy())
                for key, value in all_stats.items():
                    record[f"all_{split}_{metric}_{key}"] = value
                for role in ("selection", "confirmation", "recent_audit"):
                    role_part = part[part["outer_fold"].isin(roles.get(role, []))]
                    role_stats = aggregate_metric(role_part[column].to_numpy())
                    for key, value in role_stats.items():
                        record[f"{role}_{split}_{metric}_{key}"] = value
        selection = part[part["outer_fold"].isin(roles.get("selection", []))]
        if len(selection):
            stacked = np.column_stack(
                [
                    np.abs(selection["validation_target_pearson"].to_numpy(dtype=np.float64)),
                    np.abs(selection["validation_target_spearman"].to_numpy(dtype=np.float64)),
                ]
            )
            combined = np.full(len(stacked), np.nan, dtype=np.float64)
            finite_rows = np.isfinite(stacked).any(axis=1)
            if finite_rows.any():
                combined[finite_rows] = np.nanmax(stacked[finite_rows], axis=1)
            finite_combined = combined[np.isfinite(combined)]
            if len(finite_combined):
                record["selection_validation_combined_abs_mean"] = float(np.mean(finite_combined))
                record["selection_validation_combined_abs_min"] = float(np.min(finite_combined))
            else:
                record["selection_validation_combined_abs_mean"] = np.nan
                record["selection_validation_combined_abs_min"] = np.nan
        else:
            record["selection_validation_combined_abs_mean"] = np.nan
            record["selection_validation_combined_abs_min"] = np.nan
        selection_sign = np.sign(record.get("selection_validation_pearson_mean", np.nan))
        confirmation_sign = np.sign(record.get("confirmation_validation_pearson_mean", np.nan))
        recent_sign = np.sign(record.get("recent_audit_validation_pearson_mean", np.nan))
        record["confirmation_sign_matches_selection"] = bool(
            np.isfinite(selection_sign)
            and np.isfinite(confirmation_sign)
            and selection_sign != 0
            and selection_sign == confirmation_sign
        )
        record["recent_sign_matches_selection"] = bool(
            np.isfinite(selection_sign)
            and np.isfinite(recent_sign)
            and selection_sign != 0
            and selection_sign == recent_sign
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def binned_mutual_information(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    features: Sequence[str],
    bins: int,
    min_rows: int,
) -> pd.DataFrame:
    target = np.asarray(target, dtype=np.int8)
    records: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(feature_frame[feature], errors="coerce").to_numpy(dtype=np.float64)
        mask = np.isfinite(values) & np.isfinite(target)
        if int(mask.sum()) < min_rows or len(np.unique(target[mask])) < 2:
            records.append({"feature": feature, "mutual_information": np.nan, "normalized_mutual_information": np.nan})
            continue
        x = values[mask]
        y = target[mask].astype(np.int8)
        ranks = pd.Series(x).rank(method="average", pct=True).to_numpy(dtype=np.float64)
        codes = np.minimum((ranks * bins).astype(np.int32), bins - 1)
        contingency = np.bincount(codes * 2 + y, minlength=bins * 2).reshape(bins, 2).astype(np.float64)
        total = contingency.sum()
        pxy = contingency / total
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        expected = px @ py
        valid = (pxy > 0) & (expected > 0)
        mi = float(np.sum(pxy[valid] * np.log(pxy[valid] / expected[valid])))
        py_flat = py.ravel()
        entropy_y = float(-np.sum(py_flat[py_flat > 0] * np.log(py_flat[py_flat > 0])))
        normalized = mi / entropy_y if entropy_y > 0 else np.nan
        records.append({"feature": feature, "mutual_information": mi, "normalized_mutual_information": normalized})
    return pd.DataFrame.from_records(records)


def percentile_rank(values: pd.Series, ascending: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.zeros(len(values), dtype=np.float64), index=values.index)
    ranked = numeric.rank(method="average", pct=True, ascending=ascending)
    return ranked.fillna(0.0).astype(float)


def compute_feature_quality(
    feature_frame: pd.DataFrame,
    tickers: np.ndarray,
    features: Sequence[str],
    reference_audit: Path | None,
) -> pd.DataFrame:
    observed = feature_frame.notna()
    non_null_count = observed.sum(axis=0)
    missing_ratio = 1.0 - observed.mean(axis=0)
    unique_count = feature_frame.nunique(dropna=True)
    coverage = observed.groupby(pd.Series(tickers, index=feature_frame.index), sort=False).any().mean(axis=0)
    group_map: dict[str, str] = {}
    reference_status: dict[str, str] = {}
    if reference_audit is not None and reference_audit.exists():
        reference = pd.read_csv(reference_audit)
        if "feature" in reference.columns:
            if "group" in reference.columns:
                group_map = reference.set_index("feature")["group"].astype(str).to_dict()
            if "status" in reference.columns:
                reference_status = reference.set_index("feature")["status"].astype(str).to_dict()
    records: list[dict[str, Any]] = []
    for feature in features:
        missing = float(missing_ratio[feature])
        unique = int(unique_count[feature])
        ticker_coverage = float(coverage[feature])
        status = "valid" if missing <= 0.995 and unique >= 2 else "invalid"
        completeness = 1.0 - missing
        uniqueness_score = min(1.0, math.log1p(unique) / math.log1p(1000.0))
        data_quality_score = 0.55 * completeness + 0.30 * ticker_coverage + 0.15 * uniqueness_score
        records.append(
            {
                "feature": feature,
                "status": status,
                "missing_ratio": missing,
                "non_null_count": int(non_null_count[feature]),
                "unique_count": unique,
                "ticker_coverage": ticker_coverage,
                "group": group_map.get(feature, "unknown"),
                "reference_status": reference_status.get(feature, "unknown"),
                "data_quality_score": float(data_quality_score),
            }
        )
    return pd.DataFrame.from_records(records)


def recompute_surge_target_from_returns(
    source: pd.DataFrame,
    horizon: int = SURGE_HORIZON,
    threshold: float = SURGE_THRESHOLD,
    comparison_atol: float = SURGE_COMPARISON_ATOL,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Independently rebuild the point-in-time target from t_price_ret_1.

    For row t, only t+1..t+horizon returns of the same ticker are used. The
    source_row_id mapping makes the result comparable to the sidecar even when
    the source panel is not globally sorted by date.
    """
    required = {"source_row_id", "date", "ticker", "t_price_ret_1"}
    missing = sorted(required - set(source.columns))
    if missing:
        raise KeyError(f"target 재계산 필수 열 누락: {missing}")
    if horizon < 1:
        raise ValueError("horizon은 1 이상이어야 합니다")
    if threshold <= 0:
        raise ValueError("threshold는 양수여야 합니다")

    rows = len(source)
    labels = np.zeros(rows, dtype=np.uint8)
    valid = np.zeros(rows, dtype=bool)
    first_hit = np.zeros(rows, dtype=np.int8)
    best_forward_return = np.full(rows, np.nan, dtype=np.float64)
    numeric_returns = pd.to_numeric(source["t_price_ret_1"], errors="coerce").to_numpy(dtype=np.float64)

    for _, part in source.groupby("ticker", sort=False, observed=True):
        ordered = part.sort_values(["date", "source_row_id"], kind="mergesort")
        source_rows = ordered["source_row_id"].to_numpy(dtype=np.int64)
        ticker_returns = numeric_returns[source_rows]
        candidate_count = len(source_rows) - horizon
        if candidate_count <= 0:
            continue
        future_returns = np.column_stack(
            [ticker_returns[offset : offset + candidate_count] for offset in range(1, horizon + 1)]
        )
        finite_horizon = np.isfinite(future_returns).all(axis=1)
        if not finite_horizon.any():
            continue
        cumulative = np.cumprod(1.0 + future_returns[finite_horizon], axis=1) - 1.0
        destination_rows = source_rows[:candidate_count][finite_horizon]
        valid[destination_rows] = True
        best_forward_return[destination_rows] = np.max(cumulative, axis=1)
        hits = cumulative >= (threshold - comparison_atol)
        positive = hits.any(axis=1)
        labels[destination_rows[positive]] = 1
        if positive.any():
            first_hit[destination_rows[positive]] = np.argmax(hits[positive], axis=1).astype(np.int8) + 1

    return labels, valid, first_hit, best_forward_return


def audit_recomputed_surge_target(
    source: pd.DataFrame,
    target_sidecar: pd.DataFrame,
    target_values: np.ndarray,
    target_valid: np.ndarray,
) -> dict[str, Any]:
    if "t_price_ret_1" not in source.columns:
        return {
            "status": "skipped",
            "reason": "source does not contain t_price_ret_1",
            "horizon_trading_days": SURGE_HORIZON,
            "rise_threshold": SURGE_THRESHOLD,
        }

    rebuilt_label, rebuilt_valid, rebuilt_first_hit, rebuilt_best = recompute_surge_target_from_returns(source)
    validity_mismatch = int(np.sum(rebuilt_valid != target_valid))
    common_valid = rebuilt_valid & target_valid
    label_mismatch = int(np.sum(rebuilt_label[common_valid] != target_values[common_valid]))

    first_hit_mismatch: int | None = None
    if "first_hit_day" in target_sidecar.columns:
        sidecar_first_hit = pd.to_numeric(target_sidecar["first_hit_day"], errors="coerce").fillna(0).to_numpy()
        first_hit_mismatch = int(np.sum(sidecar_first_hit[common_valid] != rebuilt_first_hit[common_valid]))

    best_return_mismatch: int | None = None
    best_return_max_abs_error: float | None = None
    if "best_forward_return_3d" in target_sidecar.columns:
        sidecar_best = pd.to_numeric(target_sidecar["best_forward_return_3d"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite_common = common_valid & np.isfinite(sidecar_best) & np.isfinite(rebuilt_best)
        missing_best = common_valid & ~(np.isfinite(sidecar_best) & np.isfinite(rebuilt_best))
        close = np.zeros(len(source), dtype=bool)
        close[finite_common] = np.isclose(
            sidecar_best[finite_common], rebuilt_best[finite_common], rtol=1e-5, atol=5e-7
        )
        best_return_mismatch = int(np.sum(missing_best) + np.sum(finite_common & ~close))
        if finite_common.any():
            best_return_max_abs_error = float(
                np.max(np.abs(sidecar_best[finite_common] - rebuilt_best[finite_common]))
            )

    audit = {
        "status": "passed",
        "return_input": "t_price_ret_1[t] = close_t / close_(t-1) - 1",
        "horizon_trading_days": SURGE_HORIZON,
        "rise_threshold": SURGE_THRESHOLD,
        "inclusive_comparison_absolute_tolerance": SURGE_COMPARISON_ATOL,
        "recomputed_valid_rows": int(rebuilt_valid.sum()),
        "recomputed_positives": int(rebuilt_label[rebuilt_valid].sum()),
        "sidecar_validity_mismatch_rows": validity_mismatch,
        "sidecar_label_mismatch_rows_on_common_valid": label_mismatch,
        "sidecar_first_hit_mismatch_rows": first_hit_mismatch,
        "sidecar_best_forward_return_mismatch_rows": best_return_mismatch,
        "sidecar_best_forward_return_max_abs_error": best_return_max_abs_error,
    }
    hard_failures = validity_mismatch + label_mismatch
    if first_hit_mismatch is not None:
        hard_failures += first_hit_mismatch
    if best_return_mismatch is not None:
        hard_failures += best_return_mismatch
    if hard_failures:
        audit["status"] = "failed"
        raise ValueError(f"급등 target 독립 재계산 결과가 sidecar와 불일치합니다: {audit}")
    return audit


def load_and_validate_data(
    source_path: Path,
    target_path: Path,
    requested_features: Sequence[str],
    valid_feature_audit: Path | None,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    dict[str, Any],
]:
    source_columns = table_columns(source_path)
    required_source = ["date", "ticker"]
    missing_required = [column for column in required_source if column not in source_columns]
    if missing_required:
        raise KeyError(f"원본 데이터 필수 열 누락: {missing_required}")
    missing_features = [feature for feature in requested_features if feature not in source_columns]
    if missing_features:
        raise KeyError(f"profile 피처가 원본에 없습니다: {missing_features[:30]}")
    leakage = [{"feature": feature, "reason": leakage_reason(feature)} for feature in requested_features if leakage_reason(feature)]
    if leakage:
        raise ValueError(f"요청 피처에 누수 의심 열이 있습니다: {leakage[:20]}")
    read_columns = ["date", "ticker"] + list(requested_features)
    if "t_price_ret_1" in source_columns:
        read_columns.append("t_price_ret_1")
    if "sealed_do_not_train_or_tune" in source_columns:
        read_columns.append("sealed_do_not_train_or_tune")
    # Arrow can return a highly fragmented pandas frame. Consolidate once before
    # adding metadata so the full 439-feature run stays warning-free.
    source = read_table_columns(source_path, read_columns).copy()
    source.insert(0, "source_row_id", np.arange(len(source), dtype=np.int64))
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    if source["date"].isna().any():
        raise ValueError(f"원본 date 파싱 실패 행={int(source['date'].isna().sum())}")
    source["ticker"] = source["ticker"].map(normalize_ticker)
    duplicate_date_ticker = int(source.duplicated(["date", "ticker"]).sum())
    if duplicate_date_ticker:
        raise ValueError(f"원본에 date/ticker 중복 행이 있습니다: {duplicate_date_ticker}")
    if "sealed_do_not_train_or_tune" in source.columns:
        safety = pd.to_numeric(source["sealed_do_not_train_or_tune"], errors="coerce").fillna(0)
        unsafe_rows = int(safety.ne(0).sum())
        if unsafe_rows:
            raise ValueError(f"sealed_do_not_train_or_tune 위반 행={unsafe_rows}")
    target_columns = table_columns(target_path)
    required_target = [
        "source_row_id",
        "date",
        "ticker",
        TARGET_NAME,
        TARGET_VALID_COLUMN,
    ]
    missing_target = [column for column in required_target if column not in target_columns]
    if missing_target:
        raise KeyError(f"target sidecar 필수 열 누락: {missing_target}")
    optional_target_columns = [
        column for column in ("first_hit_day", "best_forward_return_3d") if column in target_columns
    ]
    target_sidecar = read_table_columns(target_path, required_target + optional_target_columns)
    if len(target_sidecar) != len(source):
        raise ValueError(f"원본/target 행 수 불일치: {len(source):,} vs {len(target_sidecar):,}")
    source_row_id = pd.to_numeric(target_sidecar["source_row_id"], errors="coerce")
    if source_row_id.isna().any() or source_row_id.duplicated().any():
        raise ValueError("target source_row_id가 비유한 값 또는 중복을 포함합니다")
    target_sidecar = target_sidecar.assign(source_row_id=source_row_id.astype(np.int64)).sort_values("source_row_id")
    expected = np.arange(len(source), dtype=np.int64)
    if not np.array_equal(target_sidecar["source_row_id"].to_numpy(dtype=np.int64), expected):
        raise ValueError("target source_row_id가 0..N-1의 완전한 일대일 키가 아닙니다")
    target_sidecar = target_sidecar.reset_index(drop=True)
    target_dates = pd.to_datetime(target_sidecar["date"], errors="coerce")
    target_tickers = target_sidecar["ticker"].map(normalize_ticker)
    if not np.array_equal(source["date"].to_numpy(dtype="datetime64[ns]"), target_dates.to_numpy(dtype="datetime64[ns]")):
        mismatch = int(
            np.sum(
                source["date"].to_numpy(dtype="datetime64[ns]")
                != target_dates.to_numpy(dtype="datetime64[ns]")
            )
        )
        raise ValueError(f"source_row_id 기준 date 불일치 행={mismatch}")
    if not np.array_equal(source["ticker"].to_numpy(dtype=str), target_tickers.to_numpy(dtype=str)):
        mismatch = int(np.sum(source["ticker"].to_numpy(dtype=str) != target_tickers.to_numpy(dtype=str)))
        raise ValueError(f"source_row_id 기준 ticker 불일치 행={mismatch}")
    target_valid = parse_bool_series(target_sidecar[TARGET_VALID_COLUMN]).to_numpy(dtype=bool)
    target_numeric = pd.to_numeric(target_sidecar[TARGET_NAME], errors="coerce")
    if target_numeric[target_valid].isna().any():
        raise ValueError("target_valid=true인 행에 target NaN이 있습니다")
    invalid_labels = ~target_numeric[target_valid].isin([0, 1])
    if invalid_labels.any():
        samples = target_numeric[target_valid][invalid_labels].head(10).tolist()
        raise ValueError(f"target은 정확한 0/1 이진값이어야 합니다: samples={samples}")
    target_values = target_numeric.fillna(0).astype(np.int8).to_numpy()
    target_revalidation = audit_recomputed_surge_target(source, target_sidecar, target_values, target_valid)
    numeric_features: dict[str, pd.Series] = {}
    for feature in requested_features:
        values = pd.to_numeric(source[feature], errors="coerce")
        numeric_features[feature] = values.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    feature_frame = pd.DataFrame(numeric_features, index=source.index)
    quality = compute_feature_quality(feature_frame, source["ticker"].to_numpy(), requested_features, valid_feature_audit)
    valid_features = quality.loc[quality["status"].eq("valid"), "feature"].astype(str).tolist()
    if not valid_features:
        raise ValueError("결측률/유일값 감사 후 유효 피처가 없습니다")
    quality["selected_for_correlation"] = quality["feature"].isin(valid_features)
    feature_frame = feature_frame[valid_features]
    date_non_decreasing = bool(source["date"].is_monotonic_increasing)
    audit = {
        "source_rows": int(len(source)),
        "target_rows": int(len(target_sidecar)),
        "requested_feature_count": int(len(requested_features)),
        "valid_feature_count": int(len(valid_features)),
        "rejected_feature_count": int(len(requested_features) - len(valid_features)),
        "date_min": source["date"].min().strftime("%Y-%m-%d"),
        "date_max": source["date"].max().strftime("%Y-%m-%d"),
        "tickers": int(source["ticker"].nunique()),
        "target_valid_rows": int(target_valid.sum()),
        "target_invalid_rows": int((~target_valid).sum()),
        "target_positives": int(target_values[target_valid].sum()),
        "target_positive_rate": float(target_values[target_valid].mean()) if target_valid.any() else None,
        "duplicate_date_ticker_rows": duplicate_date_ticker,
        "date_non_decreasing": date_non_decreasing,
        "sealed_nonzero_rows": 0,
        "source_row_id_date_ticker_match": True,
        "target_revalidation": target_revalidation,
    }
    source_meta = source[["source_row_id", "date", "ticker"]].copy()
    return source_meta, feature_frame, target_values, target_valid, source_meta["date"].to_numpy(), quality, audit


def build_target_audit(
    source_meta: pd.DataFrame,
    target: np.ndarray,
    target_valid: np.ndarray,
    source_sha256: str,
    target_sha256: str,
    independent_revalidation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    valid_target = target[target_valid]
    return {
        "target_name": TARGET_NAME,
        "definition": "max cumulative close-to-close return over D+1..D+3 >= +0.05",
        "horizon_trading_days": 3,
        "rise_threshold": 0.05,
        "comparison": "inclusive; sidecar builder uses absolute tolerance 1e-12",
        "rows": int(len(source_meta)),
        "valid_rows": int(target_valid.sum()),
        "invalid_incomplete_or_nonfinite_horizon_rows": int((~target_valid).sum()),
        "positives": int(valid_target.sum()),
        "positive_rate": float(valid_target.mean()) if len(valid_target) else None,
        "tickers": int(source_meta["ticker"].nunique()),
        "date_min": source_meta["date"].min().strftime("%Y-%m-%d"),
        "date_max": source_meta["date"].max().strftime("%Y-%m-%d"),
        "source_sha256": source_sha256,
        "target_sha256": target_sha256,
        "source_row_id_date_ticker_match": True,
        "leakage_rule": "target, target_valid, first_hit_day, best_forward_return_3d and future derivatives are excluded",
        "independent_revalidation": independent_revalidation or {"status": "not_available"},
    }


def build_ticker_target_quality(
    source_meta: pd.DataFrame,
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    target_valid: np.ndarray,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    ticker_values = source_meta["ticker"].to_numpy(dtype=str)
    for ticker in sorted(source_meta["ticker"].unique()):
        mask = ticker_values == ticker
        valid_mask = mask & target_valid
        missing_by_feature = feature_frame.loc[mask].isna().mean(axis=0)
        records.append(
            {
                "ticker": ticker,
                "rows": int(mask.sum()),
                "date_min": source_meta.loc[mask, "date"].min().strftime("%Y-%m-%d"),
                "date_max": source_meta.loc[mask, "date"].max().strftime("%Y-%m-%d"),
                "target_valid_rows": int(valid_mask.sum()),
                "target_invalid_rows": int((mask & ~target_valid).sum()),
                "positives": int(target[valid_mask].sum()),
                "positive_rate": float(target[valid_mask].mean()) if valid_mask.any() else np.nan,
                "mean_feature_missing_ratio": float(missing_by_feature.mean()),
                "median_feature_missing_ratio": float(missing_by_feature.median()),
                "max_feature_missing_ratio": float(missing_by_feature.max()),
                "features_missing_ge_99_5pct": int((missing_by_feature >= 0.995).sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def build_yearly_target_quality(
    source_meta: pd.DataFrame,
    target: np.ndarray,
    target_valid: np.ndarray,
) -> pd.DataFrame:
    years = source_meta["date"].dt.year.to_numpy(dtype=np.int32)
    records: list[dict[str, Any]] = []
    for year in sorted(np.unique(years)):
        mask = years == year
        valid_mask = mask & target_valid
        records.append(
            {
                "year": int(year),
                "rows": int(mask.sum()),
                "target_valid_rows": int(valid_mask.sum()),
                "target_invalid_rows": int((mask & ~target_valid).sum()),
                "positives": int(target[valid_mask].sum()),
                "positive_rate": float(target[valid_mask].mean()) if valid_mask.any() else np.nan,
                "tickers": int(source_meta.loc[mask, "ticker"].nunique()),
            }
        )
    return pd.DataFrame.from_records(records)


def source_column_leakage_audit(columns: Sequence[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for column in columns:
        reason = leakage_reason(column)
        if reason is not None:
            records.append({"column": column, "excluded": True, "reason": reason})
    return pd.DataFrame.from_records(records, columns=["column", "excluded", "reason"])


def legacy_reuse_compatibility(
    source_path: Path,
    fold_path: Path,
    features: Sequence[str],
    package_root: Path,
    legacy_dir: Path,
) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {
        "legacy_dir_exists": legacy_dir.exists(),
        "dataset_hash_match": False,
        "features_subset_of_legacy": False,
        "fold_definition_match": False,
        "required_files_present": False,
    }
    required = [
        legacy_dir / "correlation_manifest.json",
        legacy_dir / "correlation_matrices.npz",
        legacy_dir / "fold_pearson_matrices.npz",
        legacy_dir / "pearson.csv.gz",
    ]
    details["required_files_present"] = all(path.exists() for path in required)
    if not details["legacy_dir_exists"] or not details["required_files_present"]:
        return False, details
    dataset_audit = package_root / "data" / "DATASET_AUDIT.json"
    source_hash = sha256_file(source_path)
    details["source_sha256"] = source_hash
    if dataset_audit.exists():
        audit = json.loads(dataset_audit.read_text(encoding="utf-8"))
        expected = str(audit.get("dataset_sha256", ""))
        details["reference_dataset_sha256"] = expected
        details["dataset_hash_match"] = bool(expected and source_hash == expected)
    profile_manifest = package_root / "references" / "feature_metadata" / "profile_manifest.json"
    if profile_manifest.exists():
        payload = json.loads(profile_manifest.read_text(encoding="utf-8"))
        legacy_features = payload.get("profiles", {}).get("P0_FULL_439", {}).get("features", [])
        details["legacy_feature_count"] = len(legacy_features)
        details["features_subset_of_legacy"] = set(features).issubset(set(legacy_features))
    reference_folds = package_root / "references" / "feature_metadata" / "outer_walk_forward_folds.json"
    if reference_folds.exists() and fold_path.exists():
        current_fold_payload = json.loads(fold_path.read_text(encoding="utf-8"))
        reference_fold_payload = json.loads(reference_folds.read_text(encoding="utf-8"))
        details["current_fold_hash"] = sha256_bytes(stable_json_bytes(current_fold_payload))
        details["reference_fold_hash"] = sha256_bytes(stable_json_bytes(reference_fold_payload))
        details["fold_definition_match"] = current_fold_payload == reference_fold_payload
    return bool(
        details["dataset_hash_match"]
        and details["features_subset_of_legacy"]
        and details["fold_definition_match"]
        and details["required_files_present"]
    ), details


def load_legacy_matrix_bundle(
    legacy_dir: Path,
    package_root: Path,
    features: Sequence[str],
) -> MatrixBundle:
    profile_payload = json.loads(
        (package_root / "references" / "feature_metadata" / "profile_manifest.json").read_text(encoding="utf-8")
    )
    legacy_features = [str(item) for item in profile_payload["profiles"]["P0_FULL_439"]["features"]]
    indices = np.array([legacy_features.index(feature) for feature in features], dtype=np.int64)
    matrix_payload = np.load(legacy_dir / "correlation_matrices.npz", allow_pickle=False)
    fold_payload = np.load(legacy_dir / "fold_pearson_matrices.npz", allow_pickle=False)

    def subset(name: str) -> np.ndarray:
        matrix = np.asarray(matrix_payload[name], dtype=np.float32)
        return matrix[np.ix_(indices, indices)]

    fold_matrices = np.asarray(fold_payload["matrices"], dtype=np.float32)
    fold_matrices = fold_matrices[:, indices][:, :, indices]
    manifest = json.loads((legacy_dir / "correlation_manifest.json").read_text(encoding="utf-8"))
    return MatrixBundle(
        pearson=subset("pearson"),
        spearman=subset("spearman"),
        within_date=subset("within_date"),
        within_ticker=subset("within_ticker"),
        missingness=subset("missingness"),
        combined_abs=subset("combined_abs"),
        cluster_basis_abs=subset("cluster_basis_abs"),
        fold_ids=np.asarray(fold_payload["fold_ids"], dtype=np.int16),
        fold_pearson=fold_matrices,
        source="legacy_feature_structure_verified",
        source_manifest=manifest,
    )


def recompute_matrix_bundle(
    feature_frame: pd.DataFrame,
    dates: np.ndarray,
    tickers: np.ndarray,
    target_valid: np.ndarray,
    definitions: Sequence[FoldDefinition],
    feature_structure_rows: int,
    spearman_rows: int,
    cluster_basis_rows: int,
    fold_rows: int,
    min_periods: int,
    seed: int,
) -> MatrixBundle:
    all_mask = np.ones(len(feature_frame), dtype=bool)
    global_indices = deterministic_complete_date_sample(dates, all_mask, feature_structure_rows, seed + 1)
    spearman_indices = deterministic_complete_date_sample(dates, all_mask, spearman_rows, seed + 2)
    log(
        f"피처 구조 재계산: Pearson/within sample={len(global_indices):,}, Spearman sample={len(spearman_indices):,}"
    )
    pearson, spearman, within_date, within_ticker, missingness, combined = compute_feature_structure_on_rows(
        feature_frame,
        dates,
        tickers,
        global_indices,
        spearman_indices,
        min_periods,
    )
    fold_zero = definitions[0]
    date_series = pd.Series(pd.to_datetime(dates))
    cluster_mask = target_valid & date_series.between(fold_zero.train_start, fold_zero.train_end).to_numpy()
    cluster_indices = deterministic_complete_date_sample(dates, cluster_mask, cluster_basis_rows, seed + 3)
    cluster_spearman_indices = deterministic_complete_date_sample(
        dates, cluster_mask, min(cluster_basis_rows, spearman_rows), seed + 4
    )
    _, _, cluster_within_date, cluster_within_ticker, _, cluster_global_combined = compute_feature_structure_on_rows(
        feature_frame,
        dates,
        tickers,
        cluster_indices,
        cluster_spearman_indices,
        min_periods,
    )
    cluster_pearson = pairwise_complete_corr(
        feature_frame.iloc[cluster_indices].to_numpy(dtype=np.float64, copy=True), min_periods
    )
    cluster_ranks = pd.DataFrame(
        feature_frame.iloc[cluster_spearman_indices].to_numpy(dtype=np.float64, copy=True)
    ).rank(method="average", na_option="keep")
    cluster_spearman = pairwise_complete_corr(cluster_ranks.to_numpy(dtype=np.float64, copy=False), min_periods)
    cluster_basis = combined_absolute_correlation(
        cluster_pearson, cluster_spearman, cluster_within_date, cluster_within_ticker, cluster_global_combined
    )
    fold_ids: list[int] = []
    fold_matrices: list[np.ndarray] = []
    fold_sample_rows: dict[str, int] = {}
    for fold in definitions:
        mask = target_valid & date_series.between(fold.train_start, fold.train_end).to_numpy()
        indices = deterministic_complete_date_sample(dates, mask, fold_rows, seed + 100 + fold.fold_id)
        fold_matrix = pairwise_complete_corr(
            feature_frame.iloc[indices].to_numpy(dtype=np.float64, copy=True), min_periods
        )
        fold_ids.append(fold.fold_id)
        fold_matrices.append(fold_matrix)
        fold_sample_rows[str(fold.fold_id)] = int(len(indices))
        log(f"피처-피처 fold {fold.fold_id} Pearson 완료: sample={len(indices):,}")
    return MatrixBundle(
        pearson=pearson,
        spearman=spearman,
        within_date=within_date,
        within_ticker=within_ticker,
        missingness=missingness,
        combined_abs=combined,
        cluster_basis_abs=cluster_basis,
        fold_ids=np.asarray(fold_ids, dtype=np.int16),
        fold_pearson=np.stack(fold_matrices).astype(np.float32),
        source="recomputed_from_current_data",
        source_manifest={
            "feature_structure_sample_rows": int(len(global_indices)),
            "spearman_sample_rows": int(len(spearman_indices)),
            "cluster_basis_rows": int(len(cluster_indices)),
            "fold_correlation_sample_rows_limit": int(fold_rows),
            "fold_correlation_sample_rows_by_fold": fold_sample_rows,
        },
    )


def matrix_to_frame(matrix: np.ndarray, features: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=features, columns=features)


def fold_pair_stats(fold_values: np.ndarray, primary_threshold: float) -> dict[str, float]:
    values = np.asarray(fold_values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "fold_pearson_mean": np.nan,
            "fold_abs_pearson_mean": np.nan,
            "fold_abs_pearson_min": np.nan,
            "fold_abs_pearson_max": np.nan,
            "fold_sign_consistency": np.nan,
            "fold_ratio_abs_ge_080": np.nan,
            "fold_ratio_abs_ge_primary": np.nan,
        }
    return {
        "fold_pearson_mean": float(np.mean(values)),
        "fold_abs_pearson_mean": float(np.mean(np.abs(values))),
        "fold_abs_pearson_min": float(np.min(np.abs(values))),
        "fold_abs_pearson_max": float(np.max(np.abs(values))),
        "fold_sign_consistency": float(sign_consistency(values)),
        "fold_ratio_abs_ge_080": float(np.mean(np.abs(values) >= 0.80)),
        "fold_ratio_abs_ge_primary": float(np.mean(np.abs(values) >= primary_threshold)),
    }


def build_high_correlation_pairs(
    bundle: MatrixBundle,
    features: Sequence[str],
    primary_threshold: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    rows, cols = np.triu_indices(len(features), k=1)
    for left, right in zip(rows, cols):
        combined = float(bundle.combined_abs[left, right])
        if not np.isfinite(combined) or combined < primary_threshold:
            continue
        stats = fold_pair_stats(bundle.fold_pearson[:, left, right], primary_threshold)
        records.append(
            {
                "feature_a": features[left],
                "feature_b": features[right],
                "pearson": float(bundle.pearson[left, right]),
                "spearman": float(bundle.spearman[left, right]),
                "within_date_pearson": float(bundle.within_date[left, right]),
                "within_ticker_pearson": float(bundle.within_ticker[left, right]),
                "missingness_corr": float(bundle.missingness[left, right]),
                "combined_abs_corr": combined,
                **stats,
                "is_near_exact_duplicate": bool(combined >= 0.999999),
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "feature_a",
                "feature_b",
                "pearson",
                "spearman",
                "within_date_pearson",
                "within_ticker_pearson",
                "missingness_corr",
                "combined_abs_corr",
                "fold_pearson_mean",
                "fold_abs_pearson_mean",
                "fold_abs_pearson_min",
                "fold_abs_pearson_max",
                "fold_sign_consistency",
                "fold_ratio_abs_ge_080",
                "fold_ratio_abs_ge_primary",
                "is_near_exact_duplicate",
            ]
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["combined_abs_corr", "fold_abs_pearson_min"], ascending=[False, False]
    ).reset_index(drop=True)


def connected_components(matrix: np.ndarray, threshold: float) -> list[list[int]]:
    """Graph components retained for diagnostics; production clusters use average linkage."""
    size = matrix.shape[0]
    union_find = UnionFind(size)
    rows, cols = np.triu_indices(size, k=1)
    strong = np.isfinite(matrix[rows, cols]) & (matrix[rows, cols] >= threshold)
    for left, right in zip(rows[strong], cols[strong]):
        union_find.union(int(left), int(right))
    groups: dict[int, list[int]] = {}
    for item in range(size):
        root = union_find.find(item)
        groups.setdefault(root, []).append(item)
    components = list(groups.values())
    components.sort(key=lambda members: min(members))
    return components


def average_linkage_tree(matrix: np.ndarray) -> np.ndarray:
    """Build the same average-linkage hierarchy used by the legacy correlation map."""
    try:
        from scipy.cluster.hierarchy import linkage
        from scipy.spatial.distance import squareform
    except ImportError as error:
        raise RuntimeError(
            "상관 cluster 계산에는 scipy가 필요합니다. requirements_surge_correlation.txt를 설치하십시오."
        ) from error
    similarity = np.asarray(matrix, dtype=np.float64)
    similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=0.0)
    similarity = np.clip((similarity + similarity.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(similarity, 1.0)
    distance = 1.0 - similarity
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    return np.asarray(linkage(condensed, method="average", optimal_ordering=False), dtype=np.float64)


def average_linkage_components(
    matrix: np.ndarray,
    threshold: float,
    linkage_tree: np.ndarray | None = None,
) -> list[list[int]]:
    try:
        from scipy.cluster.hierarchy import fcluster
    except ImportError as error:
        raise RuntimeError(
            "상관 cluster 계산에는 scipy가 필요합니다. requirements_surge_correlation.txt를 설치하십시오."
        ) from error
    tree = average_linkage_tree(matrix) if linkage_tree is None else linkage_tree
    labels = fcluster(tree, t=max(0.0, 1.0 - float(threshold)), criterion="distance")
    groups: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(int(label), []).append(index)
    components = list(groups.values())
    components.sort(key=lambda members: min(members))
    return components


def within_cluster_centrality(matrix: np.ndarray, members: Sequence[int]) -> dict[int, float]:
    if len(members) == 1:
        return {int(members[0]): 1.0}
    result: dict[int, float] = {}
    for member in members:
        peers = [peer for peer in members if peer != member]
        values = matrix[int(member), peers]
        values = values[np.isfinite(values)]
        result[int(member)] = float(np.mean(values)) if len(values) else 0.0
    return result


def representative_scores(
    features: Sequence[str],
    quality: pd.DataFrame,
    fold_correlations: pd.DataFrame,
    fold_id: int,
) -> pd.Series:
    fold = fold_correlations[fold_correlations["outer_fold"].eq(fold_id)].set_index("feature")
    quality_indexed = quality.set_index("feature")
    signal = np.maximum(
        np.abs(fold.reindex(features)["train_target_pearson"].to_numpy(dtype=np.float64)),
        np.abs(fold.reindex(features)["train_target_spearman"].to_numpy(dtype=np.float64)),
    )
    signal_rank = pd.Series(signal, index=features).rank(method="average", pct=True).fillna(0.0)
    data_quality = quality_indexed.reindex(features)["data_quality_score"].fillna(0.0)
    missing_penalty = 1.0 - quality_indexed.reindex(features)["missing_ratio"].fillna(1.0)
    score = 0.65 * signal_rank + 0.25 * data_quality + 0.10 * missing_penalty
    return score.astype(float)


def choose_representative(
    members: Sequence[int],
    features: Sequence[str],
    scores: pd.Series,
    centrality: dict[int, float],
    quality: pd.DataFrame,
) -> tuple[int, float]:
    quality_indexed = quality.set_index("feature")
    candidates: list[tuple[float, float, float, str, int]] = []
    for member in members:
        feature = features[int(member)]
        base = float(scores.get(feature, 0.0))
        central = float(centrality.get(int(member), 0.0))
        missing = float(quality_indexed.at[feature, "missing_ratio"])
        final = 0.75 * base + 0.25 * central
        candidates.append((final, central, -missing, feature, int(member)))
    candidates.sort(reverse=True)
    best = candidates[0]
    return best[-1], float(best[0])


def build_cluster_assignments(
    matrix: np.ndarray,
    features: Sequence[str],
    thresholds: Sequence[float],
    quality: pd.DataFrame,
    fold_correlations: pd.DataFrame,
    representative_fold_id: int,
) -> tuple[pd.DataFrame, dict[float, list[list[int]]]]:
    scores = representative_scores(features, quality, fold_correlations, representative_fold_id)
    records: list[dict[str, Any]] = []
    components_by_threshold: dict[float, list[list[int]]] = {}
    linkage_tree = average_linkage_tree(matrix)
    for threshold in thresholds:
        components = average_linkage_components(matrix, threshold, linkage_tree)
        components_by_threshold[float(threshold)] = components
        for cluster_id, members in enumerate(components, start=1):
            centrality = within_cluster_centrality(matrix, members)
            representative_index, representative_quality = choose_representative(
                members, features, scores, centrality, quality
            )
            representative = features[representative_index]
            for member in members:
                feature = features[int(member)]
                records.append(
                    {
                        "threshold": float(threshold),
                        "cluster_id": int(cluster_id),
                        "cluster_size": int(len(members)),
                        "feature": feature,
                        "representative": representative,
                        "is_representative": bool(member == representative_index),
                        "predictive_quality_score": float(scores.get(feature, 0.0)),
                        "representative_quality_score": representative_quality,
                        "within_cluster_centrality": float(centrality[int(member)]),
                        "representative_selection_fold": int(representative_fold_id),
                    }
                )
    return pd.DataFrame.from_records(records), components_by_threshold


def build_fold_specific_representatives(
    primary_components: Sequence[Sequence[int]],
    matrix: np.ndarray,
    features: Sequence[str],
    quality: pd.DataFrame,
    fold_correlations: pd.DataFrame,
    definitions: Sequence[FoldDefinition],
    roles: dict[str, list[int]],
    primary_threshold: float,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for fold in definitions:
        scores = representative_scores(features, quality, fold_correlations, fold.fold_id)
        for cluster_id, members in enumerate(primary_components, start=1):
            centrality = within_cluster_centrality(matrix, members)
            representative_index, final_score = choose_representative(members, features, scores, centrality, quality)
            records.append(
                {
                    "outer_fold": fold.fold_id,
                    "fold_role": role_for_fold(fold.fold_id, roles),
                    "threshold": primary_threshold,
                    "cluster_id": cluster_id,
                    "cluster_size": len(members),
                    "representative": features[representative_index],
                    "representative_score": final_score,
                }
            )
    return pd.DataFrame.from_records(records)


def attach_legacy_representatives(
    primary_clusters: pd.DataFrame,
    legacy_primary_path: Path,
) -> pd.DataFrame:
    result = primary_clusters.copy()
    result["legacy_crash_representative"] = ""
    result["surge_representative_changed"] = False
    if not legacy_primary_path.exists():
        return result
    legacy = pd.read_csv(legacy_primary_path)
    if not {"feature", "representative"}.issubset(legacy.columns):
        return result
    legacy_map = legacy.drop_duplicates("feature").set_index("feature")["representative"].astype(str)
    result["legacy_crash_representative"] = result["feature"].map(legacy_map).fillna("")
    result["surge_representative_changed"] = (
        result["legacy_crash_representative"].ne("")
        & result["representative"].ne(result["legacy_crash_representative"])
    )
    return result


def feature_structure_stats(bundle: MatrixBundle, features: Sequence[str], primary_threshold: float) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    size = len(features)
    for index, feature in enumerate(features):
        combined_row = bundle.combined_abs[index].astype(np.float64).copy()
        combined_row[index] = np.nan
        if np.isfinite(combined_row).any():
            peer_index = int(np.nanargmax(combined_row))
            max_corr = float(combined_row[peer_index])
            peer = features[peer_index]
            fold_stats = fold_pair_stats(bundle.fold_pearson[:, index, peer_index], primary_threshold)
        else:
            peer_index = -1
            max_corr = np.nan
            peer = ""
            fold_stats = fold_pair_stats(np.array([], dtype=float), primary_threshold)
        missing_row = np.abs(bundle.missingness[index].astype(np.float64)).copy()
        missing_row[index] = np.nan
        max_missing = float(np.nanmax(missing_row)) if np.isfinite(missing_row).any() else np.nan
        records.append(
            {
                "feature": feature,
                "max_abs_correlation": max_corr,
                "max_corr_peer": peer,
                "max_peer_fold_abs_corr_mean": fold_stats["fold_abs_pearson_mean"],
                "max_peer_fold_abs_corr_min": fold_stats["fold_abs_pearson_min"],
                "max_peer_fold_primary_threshold_ratio": fold_stats["fold_ratio_abs_ge_primary"],
                "max_missingness_correlation": max_missing,
                "correlated_peers_ge_primary": int(
                    np.sum(np.isfinite(combined_row) & (combined_row >= primary_threshold))
                ),
                "feature_index": index,
                "feature_count": size,
            }
        )
    return pd.DataFrame.from_records(records)


def build_feature_summary(
    quality: pd.DataFrame,
    global_target: pd.DataFrame,
    fold_summary: pd.DataFrame,
    mutual_information: pd.DataFrame,
    structure_stats: pd.DataFrame,
    primary_clusters: pd.DataFrame,
) -> pd.DataFrame:
    summary = quality.merge(global_target, on="feature", how="left", validate="one_to_one")
    summary = summary.merge(fold_summary, on="feature", how="left", validate="one_to_one")
    summary = summary.merge(mutual_information, on="feature", how="left", validate="one_to_one")
    summary = summary.merge(structure_stats, on="feature", how="left", validate="one_to_one")
    cluster_columns = [
        "feature",
        "cluster_id",
        "cluster_size",
        "representative",
        "is_representative",
        "predictive_quality_score",
        "representative_quality_score",
        "within_cluster_centrality",
        "legacy_crash_representative",
        "surge_representative_changed",
    ]
    cluster_info = primary_clusters[cluster_columns].rename(
        columns={
            "cluster_id": "primary_cluster_id",
            "cluster_size": "primary_cluster_size",
            "representative": "primary_representative",
            "is_representative": "is_primary_representative",
        }
    )
    summary = summary.merge(cluster_info, on="feature", how="left", validate="one_to_one")
    signal = summary["selection_validation_combined_abs_mean"].fillna(0.0)
    stability = pd.concat(
        [
            summary["selection_validation_pearson_sign_consistency"],
            summary["selection_validation_spearman_sign_consistency"],
        ],
        axis=1,
    ).mean(axis=1, skipna=True).fillna(0.0)
    mi_rank = percentile_rank(summary["normalized_mutual_information"].fillna(0.0))
    signal_rank = percentile_rank(signal)
    # Global within-date/within-ticker correlations are descriptive only because
    # they include confirmation and recent periods. They must not affect the
    # selection-fold priority score.
    summary["selection_priority_score"] = (
        0.65 * signal_rank
        + 0.20 * stability.clip(0.0, 1.0)
        + 0.10 * mi_rank
        + 0.05 * summary["data_quality_score"].fillna(0.0)
    )
    summary["confirmation_supported"] = (
        summary["confirmation_sign_matches_selection"].fillna(False)
        & summary["confirmation_validation_pearson_abs_mean"].fillna(0.0).gt(0)
    )
    summary["recent_supported"] = summary["recent_sign_matches_selection"].fillna(False)
    summary = summary.sort_values(
        ["selection_priority_score", "selection_validation_combined_abs_mean"],
        ascending=[False, False],
    ).reset_index(drop=True)
    summary.insert(0, "surge_priority_rank", np.arange(1, len(summary) + 1, dtype=np.int32))
    return summary


def build_feature_priority(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "surge_priority_rank",
        "feature",
        "group",
        "selection_priority_score",
        "target_pearson",
        "target_spearman",
        "target_within_date_pearson",
        "target_within_ticker_pearson",
        "selection_validation_pearson_mean",
        "selection_validation_spearman_mean",
        "selection_validation_combined_abs_mean",
        "selection_validation_pearson_sign_consistency",
        "selection_validation_spearman_sign_consistency",
        "confirmation_validation_pearson_mean",
        "recent_audit_validation_pearson_mean",
        "confirmation_supported",
        "recent_supported",
        "normalized_mutual_information",
        "max_abs_correlation",
        "max_corr_peer",
        "primary_cluster_id",
        "primary_representative",
        "is_primary_representative",
        "legacy_crash_representative",
        "surge_representative_changed",
    ]
    return summary[[column for column in columns if column in summary.columns]].copy()


def build_surge_crash_feature_comparison(
    summary: pd.DataFrame,
    legacy_summary_path: Path,
) -> pd.DataFrame:
    """Compare independently computed surge correlations with legacy crash correlations.

    Ranks are deterministic descending ranks of absolute Pearson correlation. A positive
    ``rank_change`` means that a feature ranks higher for the surge target than it did
    for the crash target.
    """

    surge = summary[["feature", "target_pearson", "is_primary_representative"]].copy()
    surge = surge.rename(
        columns={
            "target_pearson": "surge_corr",
            "is_primary_representative": "surge_cluster_rep",
        }
    )

    def deterministic_absolute_rank(frame: pd.DataFrame, value_column: str) -> pd.Series:
        valid = frame[["feature", value_column]].dropna().copy()
        valid["_abs_value"] = valid[value_column].abs()
        valid = valid.sort_values(
            ["_abs_value", "feature"],
            ascending=[False, True],
            kind="mergesort",
        )
        rank_map = pd.Series(
            np.arange(1, len(valid) + 1, dtype=np.int32),
            index=valid["feature"],
        )
        return frame["feature"].map(rank_map).astype("Int64")

    surge["surge_rank"] = deterministic_absolute_rank(surge, "surge_corr")
    if legacy_summary_path.exists():
        legacy = pd.read_csv(legacy_summary_path)
        required = {"feature", "target_pearson"}
        if not required.issubset(legacy.columns):
            raise ValueError(
                f"레거시 급락 상관 요약에 필수 열이 없습니다: {sorted(required - set(legacy.columns))}"
            )
        crash_columns = ["feature", "target_pearson"]
        if "is_primary_representative" in legacy.columns:
            crash_columns.append("is_primary_representative")
        crash = legacy[crash_columns].drop_duplicates("feature").copy()
        crash = crash.rename(
            columns={
                "target_pearson": "crash_corr",
                "is_primary_representative": "crash_cluster_rep",
            }
        )
        if "crash_cluster_rep" not in crash.columns:
            crash["crash_cluster_rep"] = False
        crash["crash_rank"] = deterministic_absolute_rank(crash, "crash_corr")
        comparison = surge.merge(crash, on="feature", how="left", validate="one_to_one")
    else:
        comparison = surge.copy()
        comparison["crash_corr"] = np.nan
        comparison["crash_rank"] = pd.Series(pd.NA, index=comparison.index, dtype="Int64")
        comparison["crash_cluster_rep"] = False

    both_valid = comparison["surge_corr"].notna() & comparison["crash_corr"].notna()
    same_direction = pd.Series(pd.NA, index=comparison.index, dtype="boolean")
    same_direction.loc[both_valid] = (
        np.sign(comparison.loc[both_valid, "surge_corr"])
        == np.sign(comparison.loc[both_valid, "crash_corr"])
    )
    comparison["same_direction"] = same_direction
    comparison["abs_difference"] = (comparison["surge_corr"] - comparison["crash_corr"]).abs()
    comparison["rank_change"] = comparison["crash_rank"] - comparison["surge_rank"]
    comparison["surge_cluster_rep"] = comparison["surge_cluster_rep"].fillna(False).astype(bool)
    comparison["crash_cluster_rep"] = comparison["crash_cluster_rep"].fillna(False).astype(bool)
    ordered_columns = [
        "feature",
        "surge_corr",
        "crash_corr",
        "same_direction",
        "abs_difference",
        "surge_rank",
        "crash_rank",
        "rank_change",
        "surge_cluster_rep",
        "crash_cluster_rep",
    ]
    return comparison[ordered_columns].sort_values(
        ["surge_rank", "feature"],
        ascending=[True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)


def leakage_correlation_audit(global_target: pd.DataFrame, feature_frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    indexed = global_target.set_index("feature")
    for feature in feature_frame.columns:
        pearson = float(indexed.at[feature, "target_pearson"])
        values = feature_frame[feature]
        unique = values.dropna().unique()
        binary_like = len(unique) <= 2 and set(np.asarray(unique).astype(float).tolist()).issubset({0.0, 1.0})
        reasons: list[str] = []
        name_reason = leakage_reason(feature)
        if name_reason:
            reasons.append(name_reason)
        if np.isfinite(pearson) and abs(pearson) >= 0.999999:
            reasons.append("near_perfect_target_correlation")
        if binary_like and np.isfinite(pearson) and abs(pearson) >= 0.999:
            reasons.append("binary_target_clone_suspect")
        records.append(
            {
                "feature": feature,
                "target_pearson": pearson,
                "binary_like": bool(binary_like),
                "leakage_suspect": bool(reasons),
                "reasons": ";".join(reasons),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["leakage_suspect", "target_pearson"], ascending=[False, False]
    )


def save_matrix_outputs(output_dir: Path, bundle: MatrixBundle, features: Sequence[str]) -> None:
    matrices = {
        "pearson.csv.gz": bundle.pearson,
        "spearman.csv.gz": bundle.spearman,
        "within_date_pearson.csv.gz": bundle.within_date,
        "within_ticker_pearson.csv.gz": bundle.within_ticker,
        "missingness_correlation.csv.gz": bundle.missingness,
        "combined_abs_correlation.csv.gz": bundle.combined_abs,
        "cluster_basis_combined_abs.csv.gz": bundle.cluster_basis_abs,
    }
    for filename, matrix in matrices.items():
        atomic_csv(matrix_to_frame(matrix, features), output_dir / filename, index=True)
    atomic_npz(
        output_dir / "correlation_matrices.npz",
        pearson=bundle.pearson,
        spearman=bundle.spearman,
        within_date=bundle.within_date,
        within_ticker=bundle.within_ticker,
        missingness=bundle.missingness,
        combined_abs=bundle.combined_abs,
        cluster_basis_abs=bundle.cluster_basis_abs,
        features=np.asarray(features, dtype="U"),
    )
    atomic_npz(
        output_dir / "fold_pearson_matrices.npz",
        fold_ids=bundle.fold_ids,
        matrices=bundle.fold_pearson,
        features=np.asarray(features, dtype="U"),
    )


def maybe_plot_outputs(summary: pd.DataFrame, bundle: MatrixBundle, features: Sequence[str], output_dir: Path, top_n: int) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib이 없어 PNG 시각화를 건너뜁니다")
        return []
    created: list[str] = []
    top = summary.nlargest(top_n, "selection_priority_score").copy()
    top = top.sort_values("selection_validation_pearson_mean")
    fig, ax = plt.subplots(figsize=(12, max(8, len(top) * 0.24)))
    ax.barh(top["feature"], top["selection_validation_pearson_mean"])
    ax.axvline(0.0, linewidth=0.8)
    ax.set_title("Surge target: selection-fold validation Pearson mean")
    ax.set_xlabel("Correlation with 3D/+5% target")
    fig.tight_layout()
    path = output_dir / "surge_top_target_correlations.png"
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    fig.savefig(temporary, dpi=160, bbox_inches="tight")
    plt.close(fig)
    os.replace(temporary, path)
    created.append(path.name)

    heatmap_features = summary.nlargest(min(top_n, len(summary)), "selection_priority_score")["feature"].tolist()
    indices = [features.index(feature) for feature in heatmap_features]
    matrix = bundle.combined_abs[np.ix_(indices, indices)]
    fig, ax = plt.subplots(figsize=(13, 11))
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(heatmap_features)))
    ax.set_yticks(np.arange(len(heatmap_features)))
    ax.set_xticklabels(heatmap_features, rotation=90, fontsize=6)
    ax.set_yticklabels(heatmap_features, fontsize=6)
    ax.set_title("Top surge features: combined absolute feature correlation")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    path = output_dir / "surge_top_feature_heatmap.png"
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    fig.savefig(temporary, dpi=160, bbox_inches="tight")
    plt.close(fig)
    os.replace(temporary, path)
    created.append(path.name)
    return created


def format_top_rows(frame: pd.DataFrame, correlation_column: str, count: int, ascending: bool) -> list[str]:
    selected = frame.sort_values(correlation_column, ascending=ascending).head(count)
    lines: list[str] = []
    for _, row in selected.iterrows():
        lines.append(
            f"- `{row['feature']}`: {correlation_column}={row[correlation_column]:.6f}, "
            f"selection abs mean={row.get('selection_validation_combined_abs_mean', np.nan):.6f}, "
            f"confirmation sign={bool(row.get('confirmation_sign_matches_selection', False))}, "
            f"recent sign={bool(row.get('recent_sign_matches_selection', False))}"
        )
    return lines


def write_korean_report(
    output_dir: Path,
    audit: dict[str, Any],
    summary: pd.DataFrame,
    primary_clusters: pd.DataFrame,
    high_pairs: pd.DataFrame,
    bundle: MatrixBundle,
) -> None:
    changed = primary_clusters[
        primary_clusters["is_representative"].astype(bool)
        & primary_clusters["surge_representative_changed"].astype(bool)
    ]
    positive = summary[summary["selection_validation_pearson_mean"].fillna(0.0) > 0]
    negative = summary[summary["selection_validation_pearson_mean"].fillna(0.0) < 0]
    lines = [
        "# CrashWatch Surge 3D/+5% 상관관계 지도 실행 요약",
        "",
        "## 데이터",
        "",
        f"- 유효 target 행: {audit['target_valid_rows']:,}",
        f"- 양성: {audit['target_positives']:,}",
        f"- 양성률: {audit['target_positive_rate']:.6%}",
        f"- 피처 수: {audit['valid_feature_count']:,}",
        f"- 피처-피처 구조 출처: `{bundle.source}`",
        "",
        "## 해석 규칙",
        "",
        "- 아래 값은 급등 target과의 연관성이지 인과관계나 모델 중요도가 아니다.",
        "- selection fold에서 후보를 찾고 confirmation/recent는 지지 여부만 확인한다.",
        "- 기존 급락 상관의 부호를 뒤집지 않았으며 급등 target으로 새로 계산했다.",
        "- cluster 구조는 피처 간 중복 구조이고, 대표 피처는 각 outer fold의 train 구간에서 별도 선정한다.",
        "",
        "## selection validation 기준 양의 상관 상위",
        "",
        *format_top_rows(positive, "selection_validation_pearson_mean", 15, ascending=False),
        "",
        "## selection validation 기준 음의 상관 상위",
        "",
        *format_top_rows(negative, "selection_validation_pearson_mean", 15, ascending=True),
        "",
        "## 중복 구조",
        "",
        f"- primary threshold: {float(primary_clusters['threshold'].iloc[0]) if len(primary_clusters) else np.nan}",
        f"- primary cluster 수: {int(primary_clusters['cluster_id'].nunique()) if len(primary_clusters) else 0}",
        f"- primary threshold 이상 edge 수: {len(high_pairs):,}",
        f"- 급락용 대표와 달라진 급등용 대표 cluster 수: {len(changed):,}",
        "",
        "상세값은 `surge_feature_correlation_summary.csv`, `surge_feature_target_correlation_by_fold.csv`, "
        "`primary_clusters.csv`, `surge_representatives_by_fold.csv`를 기준으로 확인한다.",
        "",
    ]
    atomic_text("\n".join(lines), output_dir / "SURGE_CORRELATION_MAP_GUIDE_KO.md")


def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    package_root = Path(args.package_root).resolve() if args.package_root else script_path.parents[1]
    args.package_root = package_root
    args.input = Path(args.input).resolve() if args.input else package_root / "data" / "training_dataset_finance11h.parquet"
    args.target = Path(args.target).resolve() if args.target else package_root / "data" / "surge_target_3d5.parquet"
    args.output = Path(args.output).resolve() if args.output else package_root / "outputs" / "surge_correlation_map"
    args.profile_manifest = (
        Path(args.profile_manifest).resolve()
        if args.profile_manifest
        else package_root / "references" / "feature_metadata" / "profile_manifest.json"
    )
    args.valid_feature_audit = (
        Path(args.valid_feature_audit).resolve()
        if args.valid_feature_audit
        else package_root / "references" / "feature_metadata" / "valid_feature_audit.csv"
    )
    args.folds = (
        Path(args.folds).resolve()
        if args.folds
        else package_root / "references" / "feature_metadata" / "outer_walk_forward_folds.json"
    )
    args.legacy_correlation_dir = (
        Path(args.legacy_correlation_dir).resolve()
        if args.legacy_correlation_dir
        else package_root / "references" / "correlation_map_legacy_crash"
    )
    args.feature_list = Path(args.feature_list).resolve() if args.feature_list else None
    return args


def build_config(
    args: argparse.Namespace,
    features: Sequence[str],
    roles: dict[str, list[int]],
    input_hashes: dict[str, str | None],
) -> dict[str, Any]:
    return {
        "schema": "crashwatch_surge_correlation_map_v1",
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "target": TARGET_NAME,
        "input": str(args.input),
        "target_sidecar": str(args.target),
        "output": str(args.output),
        "profile": args.profile,
        "feature_count_requested": len(features),
        "feature_hash_requested": hash_feature_list(features),
        "fold_roles": roles,
        "thresholds": args.thresholds,
        "primary_threshold": args.primary_threshold,
        "reuse_legacy": args.reuse_legacy,
        "feature_structure_sample_rows": args.feature_structure_sample_rows,
        "spearman_sample_rows": args.spearman_sample_rows,
        "cluster_basis_rows": args.cluster_basis_rows,
        "fold_correlation_sample_rows": args.fold_correlation_sample_rows,
        "target_train_sample_rows": args.target_train_sample_rows,
        "mi_sample_rows": args.mi_sample_rows,
        "mi_bins": args.mi_bins,
        "min_periods": args.min_periods,
        "target_min_periods": args.target_min_periods,
        "chunk_size": args.chunk_size,
        "minimum_purge_days": args.minimum_purge_days,
        "seed": args.seed,
        "skip_mi": args.skip_mi,
        "skip_plots": args.skip_plots,
        "plot_top_n": args.plot_top_n,
        "input_hashes": input_hashes,
    }


def build_output_inventory(output_dir: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    excluded = {"RUN_STATUS.json", "surge_correlation_manifest.json"}
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if (
            not path.is_file()
            or path.name.startswith(".")
            or path.name in excluded
            or path.suffix.lower() == ".log"
        ):
            continue
        inventory.append(
            {
                "name": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return inventory


def completed_manifest_is_reusable(prior: dict[str, Any], output_dir: Path, config_hash: str) -> bool:
    if prior.get("status") != "complete" or prior.get("config_hash") != config_hash:
        return False
    inventory = prior.get("output_inventory")
    if isinstance(inventory, list) and inventory:
        for item in inventory:
            if not isinstance(item, dict) or not item.get("name"):
                return False
            path = output_dir / str(item["name"])
            if not path.is_file() or path.stat().st_size <= 0:
                return False
            if int(item.get("bytes", -1)) != int(path.stat().st_size):
                return False
            expected_hash = str(item.get("sha256", ""))
            if not expected_hash or sha256_file(path) != expected_hash:
                return False
        return True
    required = {
        "DATASET_COMPATIBILITY.json",
        "SURGE_TARGET_AUDIT.json",
        "feature_quality_surge.csv",
        "walk_forward_folds.json",
        "surge_feature_target_correlation_by_fold.csv",
        "surge_feature_correlation_summary.csv",
        "correlation_matrices.npz",
        "primary_clusters.csv",
    }
    return all((output_dir / name).is_file() and (output_dir / name).stat().st_size > 0 for name in required)


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    pipeline_started_at = utc_now()
    pipeline_started_perf_counter = time.perf_counter()
    args = resolve_default_paths(args)
    args.thresholds = parse_float_list(args.thresholds) if isinstance(args.thresholds, str) else list(args.thresholds)
    if args.primary_threshold not in args.thresholds:
        args.thresholds = sorted(set(args.thresholds + [args.primary_threshold]))
    roles = {
        "selection": parse_int_list(args.selection_folds),
        "confirmation": parse_int_list(args.confirmation_folds),
        "recent_audit": parse_int_list(args.recent_folds),
    }
    if not any(roles.values()):
        roles = DEFAULT_FOLD_ROLES.copy()
    for path in (args.input, args.target, args.folds):
        if not path.exists():
            raise FileNotFoundError(path)
    source_columns = table_columns(args.input)
    if args.feature_list:
        requested_features = read_feature_list(args.feature_list)
    else:
        requested_features = load_profile_features(args.profile_manifest, args.profile)
        if not requested_features:
            requested_features = infer_candidate_features(source_columns)
    if not requested_features:
        raise ValueError("사용할 피처를 찾지 못했습니다")
    log("resume/cache 안전성을 위해 원본·target·fold 입력 해시를 계산합니다")
    source_sha = sha256_file(args.input)
    target_sha = sha256_file(args.target)
    input_hashes: dict[str, str | None] = {
        "dataset_sha256": source_sha,
        "target_sha256": target_sha,
        "folds_sha256": sha256_file(args.folds),
        "feature_list_sha256": sha256_file(args.feature_list) if args.feature_list and args.feature_list.exists() else None,
        "profile_manifest_sha256": (
            sha256_file(args.profile_manifest) if args.profile_manifest and args.profile_manifest.exists() else None
        ),
        "valid_feature_audit_sha256": (
            sha256_file(args.valid_feature_audit)
            if args.valid_feature_audit and args.valid_feature_audit.exists()
            else None
        ),
    }
    for filename, key in (
        ("feature_correlation_summary.csv", "legacy_feature_summary_sha256"),
        ("primary_clusters.csv", "legacy_primary_clusters_sha256"),
    ):
        path = args.legacy_correlation_dir / filename
        input_hashes[key] = sha256_file(path) if path.exists() else None
    if args.reuse_legacy != "never":
        for filename, key in (
            ("correlation_manifest.json", "legacy_manifest_sha256"),
            ("correlation_matrices.npz", "legacy_matrices_sha256"),
            ("fold_pearson_matrices.npz", "legacy_fold_matrices_sha256"),
        ):
            path = args.legacy_correlation_dir / filename
            input_hashes[key] = sha256_file(path) if path.exists() else None
    config = build_config(args, requested_features, roles, input_hashes)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "surge_correlation_manifest.json"
    config_hash = sha256_bytes(stable_json_bytes(config))
    if args.resume and manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if completed_manifest_is_reusable(prior, args.output, config_hash):
            log("동일 config의 완료 manifest를 확인하여 재사용합니다")
            return prior
        log("기존 manifest의 config 또는 산출물 무결성이 맞지 않아 다시 계산합니다")
    with RunLock(args.output / ".run.lock"):
        tracker = RunTracker(args.output, config)
        try:
            tracker.stage("data_audit", "running")
            log("원본/급등 target sidecar를 source_row_id로 검증하고 필요한 피처만 읽습니다")
            source, feature_frame, target, target_valid, dates, quality_audit, data_audit = load_and_validate_data(
                args.input,
                args.target,
                requested_features,
                args.valid_feature_audit,
            )
            features = feature_frame.columns.astype(str).tolist()
            quality = quality_audit[quality_audit["feature"].isin(features)].reset_index(drop=True)
            tickers = source["ticker"].to_numpy(dtype=str)
            date_values = pd.to_datetime(source["date"]).to_numpy(dtype="datetime64[ns]")
            data_audit.update(
                {
                    "source_file": args.input.name,
                    "source_sha256": source_sha,
                    "target_file": args.target.name,
                    "target_sha256": target_sha,
                    "feature_hash": hash_feature_list(features),
                    "target_name": TARGET_NAME,
                    "target_definition": "max cumulative close-to-close return over D+1..D+3 >= +5%",
                    "leakage_exclusions": sorted(EXPLICIT_LEAKAGE_COLUMNS),
                }
            )
            target_audit = build_target_audit(
                source,
                target,
                target_valid,
                source_sha,
                target_sha,
                data_audit.get("target_revalidation"),
            )
            ticker_quality = build_ticker_target_quality(source, feature_frame, target, target_valid)
            yearly_quality = build_yearly_target_quality(source, target, target_valid)
            schema_leakage = source_column_leakage_audit(source_columns)
            data_audit["source_columns_excluded_by_name_count"] = int(len(schema_leakage))
            data_audit["source_columns_excluded_by_name"] = schema_leakage["column"].tolist()
            atomic_json(data_audit, args.output / "DATASET_COMPATIBILITY.json")
            atomic_json(target_audit, args.output / "SURGE_TARGET_AUDIT.json")
            atomic_csv(quality_audit, args.output / "feature_quality_surge.csv")
            atomic_csv(ticker_quality, args.output / "ticker_target_quality.csv")
            atomic_csv(yearly_quality, args.output / "yearly_target_quality.csv")
            atomic_csv(schema_leakage, args.output / "leakage_source_column_audit.csv")
            tracker.stage("data_audit", "complete", **data_audit)

            tracker.stage("fold_audit", "running")
            definitions = load_fold_definitions(args.folds)
            fold_manifest = validate_fold_definitions(
                source["date"], target_valid, target, definitions, roles, args.minimum_purge_days
            )
            atomic_json(fold_manifest, args.output / "walk_forward_folds.json")
            tracker.stage("fold_audit", "complete", folds=len(definitions))

            tracker.stage("target_correlations", "running")
            valid_indices = np.flatnonzero(target_valid)
            log(f"급등 target 전용 전체/within-date/within-ticker 상관 계산: {len(valid_indices):,}행")
            global_target = compute_target_correlations(
                feature_frame.iloc[valid_indices],
                target[valid_indices],
                date_values[valid_indices],
                tickers[valid_indices],
                features,
                args.target_min_periods,
                args.chunk_size,
                include_grouped=True,
            )
            fold_correlations = compute_fold_target_correlations(
                feature_frame,
                target,
                target_valid,
                date_values,
                tickers,
                features,
                definitions,
                roles,
                args.target_train_sample_rows,
                args.target_min_periods,
                args.chunk_size,
                args.seed,
            )
            fold_summary = summarize_fold_target_correlations(fold_correlations, features, roles)
            atomic_csv(fold_correlations, args.output / "surge_feature_target_correlation_by_fold.csv")
            tracker.stage(
                "target_correlations",
                "complete",
                global_rows=len(valid_indices),
                fold_rows=len(fold_correlations),
            )

            tracker.stage("mutual_information", "running")
            fold_zero = definitions[0]
            fold_zero_mask = target_valid & pd.Series(source["date"]).between(
                fold_zero.train_start, fold_zero.train_end
            ).to_numpy()
            mi_indices = deterministic_complete_date_sample(
                date_values, fold_zero_mask, args.mi_sample_rows, args.seed + 500
            )
            if args.skip_mi:
                mutual_information = pd.DataFrame(
                    {
                        "feature": features,
                        "mutual_information": np.nan,
                        "normalized_mutual_information": np.nan,
                    }
                )
            else:
                log(f"fold 0 train 전용 binned mutual information 계산: {len(mi_indices):,}행")
                mutual_information = binned_mutual_information(
                    feature_frame.iloc[mi_indices],
                    target[mi_indices],
                    features,
                    bins=args.mi_bins,
                    min_rows=args.target_min_periods,
                )
            tracker.stage("mutual_information", "complete", sample_rows=len(mi_indices), skipped=args.skip_mi)

            tracker.stage("feature_structure", "running")
            compatible, reuse_details = legacy_reuse_compatibility(
                args.input, args.folds, features, args.package_root, args.legacy_correlation_dir
            )
            if args.reuse_legacy == "always" and not compatible:
                raise ValueError(f"legacy 피처 구조 강제 재사용 조건 불충족: {reuse_details}")
            should_reuse = args.reuse_legacy == "always" or (args.reuse_legacy == "auto" and compatible)
            if should_reuse:
                log("dataset hash와 feature 목록이 일치하여 기존 급락 자료의 피처-피처 구조만 재사용합니다")
                bundle = load_legacy_matrix_bundle(args.legacy_correlation_dir, args.package_root, features)
            else:
                log("피처-피처 구조를 현재 데이터에서 재계산합니다")
                bundle = recompute_matrix_bundle(
                    feature_frame,
                    date_values,
                    tickers,
                    target_valid,
                    definitions,
                    args.feature_structure_sample_rows,
                    args.spearman_sample_rows,
                    args.cluster_basis_rows,
                    args.fold_correlation_sample_rows,
                    args.min_periods,
                    args.seed,
                )
            save_matrix_outputs(args.output, bundle, features)
            tracker.stage(
                "feature_structure",
                "complete",
                source=bundle.source,
                reuse_compatibility=reuse_details,
            )

            tracker.stage("clusters", "running")
            high_pairs = build_high_correlation_pairs(bundle, features, args.primary_threshold)
            all_clusters, components_by_threshold = build_cluster_assignments(
                bundle.cluster_basis_abs,
                features,
                args.thresholds,
                quality,
                fold_correlations,
                representative_fold_id=definitions[0].fold_id,
            )
            primary_clusters = all_clusters[all_clusters["threshold"].eq(args.primary_threshold)].copy()
            primary_clusters = attach_legacy_representatives(
                primary_clusters,
                args.legacy_correlation_dir / "primary_clusters.csv",
            )
            primary_components = components_by_threshold[float(args.primary_threshold)]
            fold_representatives = build_fold_specific_representatives(
                primary_components,
                bundle.cluster_basis_abs,
                features,
                quality,
                fold_correlations,
                definitions,
                roles,
                args.primary_threshold,
            )
            structure_stats = feature_structure_stats(bundle, features, args.primary_threshold)
            summary = build_feature_summary(
                quality,
                global_target,
                fold_summary,
                mutual_information,
                structure_stats,
                primary_clusters,
            )
            feature_priority = build_feature_priority(summary)
            surge_vs_crash = build_surge_crash_feature_comparison(
                summary,
                args.legacy_correlation_dir / "feature_correlation_summary.csv",
            )
            leakage_audit = leakage_correlation_audit(global_target, feature_frame.iloc[valid_indices])
            suspicious = leakage_audit[leakage_audit["leakage_suspect"]]
            if len(suspicious):
                log(f"경고: 상관 기반 누수 의심 피처 {len(suspicious)}개를 leakage_audit.csv에 기록했습니다")
            atomic_csv(high_pairs, args.output / "high_correlation_pairs.csv")
            atomic_csv(all_clusters, args.output / "all_cluster_assignments.csv")
            atomic_csv(primary_clusters, args.output / "primary_clusters.csv")
            atomic_csv(fold_representatives, args.output / "surge_representatives_by_fold.csv")
            atomic_csv(summary, args.output / "surge_feature_correlation_summary.csv")
            atomic_csv(feature_priority, args.output / "surge_feature_priority.csv")
            atomic_csv(surge_vs_crash, args.output / "surge_vs_crash_feature_comparison.csv")
            atomic_csv(leakage_audit, args.output / "leakage_audit.csv")
            representatives_payload = {
                str(int(row.cluster_id)): row.representative
                for row in primary_clusters[primary_clusters["is_representative"]].itertuples()
            }
            atomic_json(representatives_payload, args.output / "surge_primary_representatives.json")
            tracker.stage(
                "clusters",
                "complete",
                primary_clusters=int(primary_clusters["cluster_id"].nunique()),
                high_correlation_edges=len(high_pairs),
                leakage_suspects=len(suspicious),
            )

            tracker.stage("reporting", "running")
            plot_files = [] if args.skip_plots else maybe_plot_outputs(
                summary, bundle, features, args.output, args.plot_top_n
            )
            write_korean_report(args.output, data_audit, summary, primary_clusters, high_pairs, bundle)
            matrix_hash = hash_arrays(
                features,
                [
                    bundle.pearson,
                    bundle.spearman,
                    bundle.within_date,
                    bundle.within_ticker,
                    bundle.combined_abs,
                    bundle.cluster_basis_abs,
                ],
            )
            output_inventory = build_output_inventory(args.output)
            manifest = {
                "status": "complete",
                "schema": "crashwatch_surge_correlation_map_v1",
                "run_time": pipeline_started_at,
                "created_at": utc_now(),
                "elapsed_seconds": time.perf_counter() - pipeline_started_perf_counter,
                "python_version": sys.version,
                "platform": platform.platform(),
                "config_hash": config_hash,
                "source_dataset_path": str(args.input),
                "source_dataset_sha256": source_sha,
                "target_path": str(args.target),
                "dataset_sha256": source_sha,
                "target_sha256": target_sha,
                "target_name": TARGET_NAME,
                "target_valid_rows": data_audit["target_valid_rows"],
                "target_positives": data_audit["target_positives"],
                "target_positive_rate": data_audit["target_positive_rate"],
                "feature_count": len(features),
                "row_count": data_audit["source_rows"],
                "date_min": data_audit["date_min"],
                "date_max": data_audit["date_max"],
                "feature_hash": hash_feature_list(features),
                "feature_structure_source": bundle.source,
                "feature_structure_source_manifest": bundle.source_manifest,
                "legacy_reuse_compatibility": reuse_details,
                "fold_roles": roles,
                "fold_definition": [asdict(definition) for definition in definitions],
                "thresholds": args.thresholds,
                "primary_threshold": args.primary_threshold,
                "cluster_threshold": args.primary_threshold,
                "reuse_legacy_mode": args.reuse_legacy,
                "legacy_reused": should_reuse,
                "git_commit": None,
                "primary_cluster_count": int(primary_clusters["cluster_id"].nunique()),
                "primary_representative_count": int(primary_clusters["is_representative"].sum()),
                "high_correlation_edge_count": len(high_pairs),
                "matrix_hash": matrix_hash,
                "leakage_suspect_count": len(suspicious),
                "plot_files": plot_files,
                "outputs": [item["name"] for item in output_inventory] + ["RUN_STATUS.json"],
                "output_inventory": output_inventory,
                "interpretation_rule": (
                    "feature-feature structure may be reused after hash verification; all target associations "
                    "and representatives are recomputed for the 3D/+5% surge target"
                ),
            }
            atomic_json(manifest, manifest_path)
            tracker.stage("reporting", "complete", plots=plot_files)
            tracker.complete(manifest=str(manifest_path), outputs=len(manifest["outputs"]))
            log(f"완료: {args.output}")
            return manifest
        except BaseException as error:
            tracker.fail(error)
            raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CrashWatch 3거래일 내 +5% 급등 target 전용 상관관계 지도를 생성합니다. "
            "인자 없이 실행하면 패키지의 data/references 경로를 자동 사용합니다."
        )
    )
    parser.add_argument("--package-root", default=None, help="압축 해제한 패키지 루트")
    parser.add_argument("--input", default=None, help="training_dataset_finance11h.parquet 또는 CSV")
    parser.add_argument("--target", default=None, help="surge_target_3d5.parquet 또는 CSV")
    parser.add_argument("--output", default=None, help="출력 디렉터리")
    parser.add_argument("--profile-manifest", default=None)
    parser.add_argument("--profile", default="P0_FULL_439")
    parser.add_argument("--feature-list", default=None, help="한 줄당 피처 하나 또는 JSON 배열")
    parser.add_argument("--valid-feature-audit", default=None)
    parser.add_argument("--folds", default=None)
    parser.add_argument("--legacy-correlation-dir", default=None)
    parser.add_argument(
        "--reuse-legacy",
        choices=("auto", "always", "never"),
        default="auto",
        help="기존 급락 자료에서 target과 무관한 피처-피처 구조만 재사용할지 결정",
    )
    parser.add_argument("--thresholds", default="0.80,0.90,0.92,0.95,0.98")
    parser.add_argument("--primary-threshold", type=float, default=DEFAULT_PRIMARY_THRESHOLD)
    parser.add_argument("--selection-folds", default="0,1,2,3,4")
    parser.add_argument("--confirmation-folds", default="5,6")
    parser.add_argument("--recent-folds", default="7")
    parser.add_argument("--minimum-purge-days", type=int, default=3)
    parser.add_argument("--feature-structure-sample-rows", type=int, default=40000)
    parser.add_argument("--spearman-sample-rows", type=int, default=15000)
    parser.add_argument("--cluster-basis-rows", type=int, default=20000)
    parser.add_argument("--fold-correlation-sample-rows", type=int, default=12000)
    parser.add_argument("--target-train-sample-rows", type=int, default=12000)
    parser.add_argument("--mi-sample-rows", type=int, default=20000)
    parser.add_argument("--mi-bins", type=int, default=16)
    parser.add_argument("--min-periods", type=int, default=4000)
    parser.add_argument("--target-min-periods", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--skip-mi", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plot-top-n", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        manifest = run_pipeline(args)
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr)
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
