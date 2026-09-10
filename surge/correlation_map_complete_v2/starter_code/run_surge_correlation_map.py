"""CrashWatch Surge 3D5 complete correlation-map pipeline.

The program audits and independently revalidates a 3-trading-day +5 percent
surge target, blocks future-derived features, computes global and grouped
feature-target associations, evaluates expanding walk-forward stability, builds
target-independent feature-feature redundancy matrices, selects target-dependent
cluster representatives using selection-fold train-only consensus, and separates
surge-directional signals from common large-move signals using the legacy crash
reference.

Confirmation and recent-audit folds are diagnostic gates only.  They never enter
the base selection priority score or cluster representative vote.  Missing and
infinite values are audited explicitly; no forward fill, backward fill, or target
imputation is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shutil
import sys
import time
import traceback
import warnings
from collections import Counter, defaultdict
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
SCHEMA_VERSION = "crashwatch_surge_correlation_map_v2_complete"
DEFAULT_MIN_SELECTION_ABS_CORR = 0.02
DEFAULT_MIN_SELECTION_SIGN_CONSISTENCY = 0.80
DEFAULT_MIN_CONFIRMATION_ABS_CORR = 0.02
DEFAULT_MIN_CONFIRMATION_SIGN_CONSISTENCY = 1.00
DEFAULT_MIN_CONFIRMATION_RETENTION = 0.40
DEFAULT_MIN_RECENT_ABS_CORR = 0.02
DEFAULT_MIN_RECENT_SIGN_CONSISTENCY = 1.00
DEFAULT_MIN_RECENT_RETENTION = 0.40
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
    """Load fold definitions, reject ambiguous IDs, and return deterministic order."""

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
    fold_ids = [fold.fold_id for fold in result]
    duplicates = sorted(fold_id for fold_id, count in Counter(fold_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"fold_id 중복: {duplicates}")
    return sorted(result, key=lambda fold: fold.fold_id)


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
        if train_start > train_end:
            reasons.append("train_start_after_train_end")
        if validation_start > validation_end:
            reasons.append("validation_start_after_validation_end")
        if train_end >= validation_start:
            reasons.append("train_validation_overlap")
        if len(purge_dates) < minimum_purge_days:
            reasons.append(f"purge_dates={len(purge_dates)}<{minimum_purge_days}")
        if bool(fold.purge_start) != bool(fold.purge_end):
            reasons.append("partial_declared_purge_bounds")
        if fold.purge_start and fold.purge_end:
            purge_start = pd.Timestamp(fold.purge_start)
            purge_end = pd.Timestamp(fold.purge_end)
            if purge_start > purge_end:
                reasons.append("purge_start_after_purge_end")
            if purge_start <= train_end:
                reasons.append("declared_purge_overlaps_train")
            if purge_end >= validation_start:
                reasons.append("declared_purge_overlaps_validation")
            declared_purge_dates = unique_dates[
                (unique_dates >= np.datetime64(purge_start)) & (unique_dates <= np.datetime64(purge_end))
            ]
            if len(declared_purge_dates) < minimum_purge_days:
                reasons.append(f"declared_purge_dates={len(declared_purge_dates)}<{minimum_purge_days}")
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
    """Compute train and validation target associations for every walk-forward fold."""

    records: list[pd.DataFrame] = []
    date_series = pd.Series(pd.to_datetime(dates))
    base_columns = [
        "feature",
        "target_pearson",
        "target_spearman",
        "target_within_date_pearson",
        "target_within_ticker_pearson",
        "finite_rows",
        "positive_rows",
        "positive_rate",
    ]
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
            include_grouped=True,
        )[base_columns].rename(
            columns={column: f"train_{column}" for column in base_columns if column != "feature"}
        )
        validation_corr = compute_target_correlations(
            feature_frame.iloc[validation_indices],
            target[validation_indices],
            dates[validation_indices],
            tickers[validation_indices],
            features,
            min_periods,
            chunk_size,
            include_grouped=True,
        )[base_columns].rename(
            columns={column: f"validation_{column}" for column in base_columns if column != "feature"}
        )
        merged = train_corr.merge(validation_corr, on="feature", how="inner", validate="one_to_one")
        metadata = [
            ("outer_fold", fold.fold_id),
            ("fold_role", role_for_fold(fold.fold_id, roles)),
            ("train_start", fold.train_start),
            ("train_end", fold.train_end),
            ("validation_start", fold.validation_start),
            ("validation_end", fold.validation_end),
            ("train_sample_rows_total", int(len(train_indices))),
            ("validation_rows_total", int(len(validation_indices))),
        ]
        for position, (column, value) in enumerate(metadata):
            merged.insert(position, column, value)
        records.append(merged)
        log(
            f"fold {fold.fold_id}: train {len(train_indices):,}, validation {len(validation_indices):,}, "
            f"role={role_for_fold(fold.fold_id, roles)}"
        )
    return pd.concat(records, ignore_index=True)




def binary_roc_auc(scores: np.ndarray, target: np.ndarray) -> float:
    """Tie-aware ROC-AUC without importing a model library."""

    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    mask = np.isfinite(scores) & np.isin(target, [0, 1])
    scores = scores[mask]
    target = target[mask]
    positives = int(np.sum(target == 1))
    negatives = int(np.sum(target == 0))
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    positive_rank_sum = float(ranks[target == 1].sum())
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(np.clip(auc, 0.0, 1.0))


def binary_average_precision(scores: np.ndarray, target: np.ndarray) -> float:
    """Threshold/tie-aware average precision for binary labels."""

    scores = np.asarray(scores, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    mask = np.isfinite(scores) & np.isin(target, [0, 1])
    scores = scores[mask]
    target = target[mask]
    positive_total = int(np.sum(target == 1))
    if positive_total == 0 or positive_total == len(target):
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_target = target[order]
    if not len(sorted_scores):
        return np.nan
    boundaries = np.r_[np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1, len(sorted_scores)]
    starts = np.r_[0, boundaries[:-1]]
    group_positives = np.add.reduceat(sorted_target.astype(np.float64), starts)
    group_rows = boundaries - starts
    cumulative_positives = np.cumsum(group_positives)
    cumulative_rows = np.cumsum(group_rows)
    precision = cumulative_positives / cumulative_rows
    recall_increment = group_positives / positive_total
    return float(np.sum(precision * recall_increment))


def directional_tail_metrics(
    values: np.ndarray,
    target: np.ndarray,
    direction: int,
    top_fraction: float,
) -> dict[str, float | int]:
    """Evaluate a validation feature after fixing its direction on train only."""

    if not 0 < top_fraction <= 0.5:
        raise ValueError(f"top_fraction은 (0, 0.5] 범위여야 합니다: {top_fraction}")
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.int8)
    mask = np.isfinite(values) & np.isin(target, [0, 1])
    x = values[mask]
    y = target[mask]
    rows = int(len(x))
    positives = int(np.sum(y == 1))
    negatives = rows - positives
    base_rate = float(positives / rows) if rows else np.nan
    empty = {
        "validation_finite_rows": rows,
        "validation_positive_rows": positives,
        "validation_positive_rate": base_rate,
        "validation_tail_rows": 0,
        "validation_top_event_rate": np.nan,
        "validation_top_lift": np.nan,
        "validation_bottom_event_rate": np.nan,
        "validation_rate_spread": np.nan,
        "validation_oriented_roc_auc": np.nan,
        "validation_oriented_average_precision": np.nan,
        "validation_oriented_pr_lift": np.nan,
    }
    if rows < 4 or positives == 0 or negatives == 0 or direction not in (-1, 1):
        return empty
    oriented = x * float(direction)
    order = np.argsort(oriented, kind="mergesort")
    tail_rows = max(1, int(math.ceil(rows * top_fraction)))
    bottom = order[:tail_rows]
    top = order[-tail_rows:]
    top_rate = float(np.mean(y[top]))
    bottom_rate = float(np.mean(y[bottom]))
    auc = binary_roc_auc(oriented, y)
    average_precision = binary_average_precision(oriented, y)
    empty.update(
        {
            "validation_tail_rows": int(tail_rows),
            "validation_top_event_rate": top_rate,
            "validation_top_lift": float(top_rate / base_rate) if base_rate > 0 else np.nan,
            "validation_bottom_event_rate": bottom_rate,
            "validation_rate_spread": float(top_rate - bottom_rate),
            "validation_oriented_roc_auc": auc,
            "validation_oriented_average_precision": average_precision,
            "validation_oriented_pr_lift": (
                float(average_precision / base_rate)
                if np.isfinite(average_precision) and base_rate > 0
                else np.nan
            ),
        }
    )
    return empty


def compute_fold_univariate_maps(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    target_valid: np.ndarray,
    dates: np.ndarray,
    features: Sequence[str],
    definitions: Sequence[FoldDefinition],
    roles: dict[str, list[int]],
    fold_correlations: pd.DataFrame,
    top_fraction: float,
) -> pd.DataFrame:
    """Build fold-safe univariate ranking diagnostics.

    Direction is selected from each fold's train Pearson (Spearman fallback),
    then frozen before validation AUC/AP/top-tail lift are measured.
    """

    date_series = pd.Series(pd.to_datetime(dates))
    indexed = fold_correlations.set_index(["outer_fold", "feature"])
    records: list[dict[str, Any]] = []
    for fold in definitions:
        validation_mask = target_valid & date_series.between(
            fold.validation_start, fold.validation_end
        ).to_numpy()
        validation_indices = np.flatnonzero(validation_mask)
        values = feature_frame.iloc[validation_indices][list(features)].to_numpy(dtype=np.float64, copy=True)
        y = target[validation_indices]
        for feature_index, feature in enumerate(features):
            row = indexed.loc[(fold.fold_id, feature)]
            train_pearson = float(row["train_target_pearson"])
            train_spearman = float(row["train_target_spearman"])
            if np.isfinite(train_pearson) and abs(train_pearson) > 1e-12:
                direction = int(np.sign(train_pearson))
                direction_source = "train_pearson"
            elif np.isfinite(train_spearman) and abs(train_spearman) > 1e-12:
                direction = int(np.sign(train_spearman))
                direction_source = "train_spearman"
            else:
                direction = 0
                direction_source = "unavailable"
            metrics = directional_tail_metrics(values[:, feature_index], y, direction, top_fraction)
            records.append(
                {
                    "outer_fold": int(fold.fold_id),
                    "fold_role": role_for_fold(fold.fold_id, roles),
                    "feature": feature,
                    "train_direction": int(direction),
                    "train_direction_source": direction_source,
                    "train_target_pearson": train_pearson,
                    "train_target_spearman": train_spearman,
                    "top_fraction": float(top_fraction),
                    **metrics,
                }
            )
        log(f"fold {fold.fold_id}: train 방향 고정 validation tail/AUC/AP 완료")
    return pd.DataFrame.from_records(records)


def summarize_fold_univariate_maps(
    fold_map: pd.DataFrame,
    features: Sequence[str],
    roles: dict[str, list[int]],
) -> pd.DataFrame:
    metrics = [
        "validation_top_event_rate",
        "validation_top_lift",
        "validation_bottom_event_rate",
        "validation_rate_spread",
        "validation_oriented_roc_auc",
        "validation_oriented_average_precision",
        "validation_oriented_pr_lift",
    ]
    records: list[dict[str, Any]] = []
    grouped = fold_map.groupby("feature", sort=False)
    for feature in features:
        part = grouped.get_group(feature)
        record: dict[str, Any] = {"feature": feature}
        directions = part[part["outer_fold"].isin(roles.get("selection", []))]["train_direction"]
        record["selection_train_direction_consistency"] = sign_consistency(directions)
        for role in ("selection", "confirmation", "recent_audit"):
            role_part = part[part["outer_fold"].isin(roles.get(role, []))]
            for metric in metrics:
                stats = aggregate_metric(role_part[metric].to_numpy(dtype=np.float64))
                for key in ("mean", "std", "min", "max", "folds"):
                    record[f"{role}_{metric}_{key}"] = stats[key]
        records.append(record)
    return pd.DataFrame.from_records(records)


def compute_groupwise_correlation_matrix(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    features: Sequence[str],
    min_rows: int,
    min_positive: int = 2,
    min_negative: int = 2,
) -> dict[str, np.ndarray]:
    """Compute one feature-target Pearson vector per group."""

    group_series = pd.Series(groups)
    codes, uniques = pd.factorize(group_series, sort=True)
    group_count = len(uniques)
    feature_count = len(features)
    correlations = np.full((group_count, feature_count), np.nan, dtype=np.float32)
    finite_rows = np.zeros((group_count, feature_count), dtype=np.int32)
    positive_rows = np.zeros((group_count, feature_count), dtype=np.int32)
    total_rows = np.zeros(group_count, dtype=np.int32)
    total_positives = np.zeros(group_count, dtype=np.int32)
    matrix = feature_frame[list(features)]
    for group_index in range(group_count):
        indices = np.flatnonzero(codes == group_index)
        y = np.asarray(target[indices], dtype=np.int8)
        total_rows[group_index] = len(indices)
        total_positives[group_index] = int(np.sum(y == 1))
        if len(indices) < min_rows or total_positives[group_index] < min_positive:
            continue
        if len(indices) - total_positives[group_index] < min_negative:
            continue
        values = matrix.iloc[indices].to_numpy(dtype=np.float64, copy=True)
        corr, counts, positives = vector_target_corr(values, y, min_rows)
        negatives = counts - positives
        invalid = (counts < min_rows) | (positives < min_positive) | (negatives < min_negative)
        corr[invalid] = np.nan
        correlations[group_index] = corr.astype(np.float32)
        finite_rows[group_index] = counts.astype(np.int32)
        positive_rows[group_index] = positives.astype(np.int32)
    return {
        "groups": np.asarray(uniques.astype(str), dtype=str),
        "features": np.asarray(list(features), dtype=str),
        "correlations": correlations,
        "finite_rows": finite_rows,
        "positive_rows": positive_rows,
        "total_rows": total_rows,
        "total_positives": total_positives,
    }


def groupwise_long_frame(payload: dict[str, np.ndarray], group_column: str) -> pd.DataFrame:
    groups = np.asarray(payload["groups"], dtype=str)
    features = np.asarray(payload["features"], dtype=str)
    correlations = np.asarray(payload["correlations"], dtype=np.float32)
    finite_rows = np.asarray(payload["finite_rows"], dtype=np.int32)
    positive_rows = np.asarray(payload["positive_rows"], dtype=np.int32)
    total_rows = np.asarray(payload["total_rows"], dtype=np.int32)
    total_positives = np.asarray(payload["total_positives"], dtype=np.int32)
    with np.errstate(divide="ignore", invalid="ignore"):
        feature_positive_rate = positive_rows / finite_rows
        group_positive_rate = total_positives / total_rows
    return pd.DataFrame(
        {
            group_column: np.repeat(groups, len(features)),
            "feature": np.tile(features, len(groups)),
            "target_pearson": correlations.reshape(-1),
            "finite_rows": finite_rows.reshape(-1),
            "positive_rows": positive_rows.reshape(-1),
            "positive_rate": feature_positive_rate.reshape(-1),
            "group_rows": np.repeat(total_rows, len(features)),
            "group_positives": np.repeat(total_positives, len(features)),
            "group_positive_rate": np.repeat(group_positive_rate, len(features)),
        }
    )


def summarize_groupwise_correlations(
    payload: dict[str, np.ndarray],
    prefix: str,
) -> pd.DataFrame:
    features = np.asarray(payload["features"], dtype=str)
    correlations = np.asarray(payload["correlations"], dtype=np.float64)
    records: list[dict[str, Any]] = []
    for feature_index, feature in enumerate(features):
        values = correlations[:, feature_index]
        finite = values[np.isfinite(values)]
        if len(finite):
            record = {
                "feature": feature,
                f"{prefix}_corr_mean": float(np.mean(finite)),
                f"{prefix}_corr_median": float(np.median(finite)),
                f"{prefix}_corr_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
                f"{prefix}_corr_abs_mean": float(np.mean(np.abs(finite))),
                f"{prefix}_corr_min": float(np.min(finite)),
                f"{prefix}_corr_max": float(np.max(finite)),
                f"{prefix}_corr_sign_consistency": float(sign_consistency(finite)),
                f"{prefix}_positive_group_count": int(np.sum(finite > 0)),
                f"{prefix}_negative_group_count": int(np.sum(finite < 0)),
                f"{prefix}_valid_group_count": int(len(finite)),
            }
        else:
            record = {
                "feature": feature,
                f"{prefix}_corr_mean": np.nan,
                f"{prefix}_corr_median": np.nan,
                f"{prefix}_corr_std": np.nan,
                f"{prefix}_corr_abs_mean": np.nan,
                f"{prefix}_corr_min": np.nan,
                f"{prefix}_corr_max": np.nan,
                f"{prefix}_corr_sign_consistency": np.nan,
                f"{prefix}_positive_group_count": 0,
                f"{prefix}_negative_group_count": 0,
                f"{prefix}_valid_group_count": 0,
            }
        records.append(record)
    return pd.DataFrame.from_records(records)


def compute_selection_fold_mutual_information(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    target_valid: np.ndarray,
    dates: np.ndarray,
    features: Sequence[str],
    definitions: Sequence[FoldDefinition],
    roles: dict[str, list[int]],
    sample_rows: int,
    bins: int,
    min_rows: int,
    seed: int,
    skip: bool,
) -> pd.DataFrame:
    date_series = pd.Series(pd.to_datetime(dates))
    records: list[pd.DataFrame] = []
    selection_ids = set(roles.get("selection", []))
    for fold in definitions:
        if fold.fold_id not in selection_ids:
            continue
        train_mask = target_valid & date_series.between(fold.train_start, fold.train_end).to_numpy()
        indices = deterministic_complete_date_sample(
            dates, train_mask, sample_rows, seed + 500 + fold.fold_id * 101
        )
        if skip:
            part = pd.DataFrame(
                {
                    "feature": list(features),
                    "mutual_information": np.nan,
                    "normalized_mutual_information": np.nan,
                }
            )
        else:
            part = binned_mutual_information(
                feature_frame.iloc[indices], target[indices], features, bins=bins, min_rows=min_rows
            )
        part.insert(0, "outer_fold", int(fold.fold_id))
        part.insert(1, "fold_role", "selection")
        part.insert(2, "train_sample_rows", int(len(indices)))
        records.append(part)
        log(f"fold {fold.fold_id}: train-only mutual information 완료 sample={len(indices):,}")
    if not records:
        raise ValueError("selection fold가 없어 mutual information을 계산할 수 없습니다")
    return pd.concat(records, ignore_index=True)


def summarize_selection_fold_mutual_information(
    fold_mi: pd.DataFrame,
    features: Sequence[str],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouped = fold_mi.groupby("feature", sort=False)
    for feature in features:
        part = grouped.get_group(feature)
        mi = aggregate_metric(part["mutual_information"].to_numpy(dtype=np.float64))
        nmi = aggregate_metric(part["normalized_mutual_information"].to_numpy(dtype=np.float64))
        records.append(
            {
                "feature": feature,
                "mutual_information": mi["mean"],
                "normalized_mutual_information": nmi["mean"],
                "selection_train_mi_mean": mi["mean"],
                "selection_train_mi_std": mi["std"],
                "selection_train_mi_min": mi["abs_min"],
                "selection_train_mi_max": mi["abs_max"],
                "selection_train_mi_folds": mi["folds"],
                "selection_train_normalized_mi_mean": nmi["mean"],
                "selection_train_normalized_mi_std": nmi["std"],
                "selection_train_normalized_mi_min": nmi["abs_min"],
                "selection_train_normalized_mi_max": nmi["abs_max"],
            }
        )
    return pd.DataFrame.from_records(records)

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
            "median": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "abs_mean": np.nan,
            "abs_min": np.nan,
            "abs_max": np.nan,
            "sign_consistency": np.nan,
            "positive_count": 0,
            "negative_count": 0,
            "zero_count": 0,
            "folds": 0,
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "abs_mean": float(np.mean(np.abs(values))),
        "abs_min": float(np.min(np.abs(values))),
        "abs_max": float(np.max(np.abs(values))),
        "sign_consistency": float(sign_consistency(values)),
        "positive_count": int(np.sum(values > 1e-12)),
        "negative_count": int(np.sum(values < -1e-12)),
        "zero_count": int(np.sum(np.abs(values) <= 1e-12)),
        "folds": int(len(values)),
    }


def summarize_fold_target_correlations(
    fold_correlations: pd.DataFrame,
    features: Sequence[str],
    roles: dict[str, list[int]],
) -> pd.DataFrame:
    """Summarize fold associations without mixing confirmation into selection scores."""

    metric_names = ("pearson", "spearman", "within_date_pearson", "within_ticker_pearson")
    records: list[dict[str, Any]] = []
    grouped = fold_correlations.groupby("feature", sort=False)
    role_names = ("selection", "confirmation", "recent_audit")
    weights = np.asarray([0.35, 0.25, 0.25, 0.15], dtype=np.float64)

    for feature in features:
        part = grouped.get_group(feature).sort_values("outer_fold", kind="mergesort")
        record: dict[str, Any] = {"feature": feature}
        for split in ("train", "validation"):
            for metric in metric_names:
                column = f"{split}_target_{metric}"
                all_stats = aggregate_metric(part[column].to_numpy(dtype=np.float64))
                for key, value in all_stats.items():
                    record[f"all_{split}_{metric}_{key}"] = value
                for role in role_names:
                    role_part = part[part["outer_fold"].isin(roles.get(role, []))]
                    role_stats = aggregate_metric(role_part[column].to_numpy(dtype=np.float64))
                    for key, value in role_stats.items():
                        record[f"{role}_{split}_{metric}_{key}"] = value

        for role in role_names:
            role_part = part[part["outer_fold"].isin(roles.get(role, []))]
            if role_part.empty:
                record[f"{role}_validation_combined_abs_mean"] = np.nan
                record[f"{role}_validation_combined_abs_min"] = np.nan
                record[f"{role}_validation_combined_abs_std"] = np.nan
                continue
            matrix = np.column_stack(
                [
                    np.abs(role_part[f"validation_target_{metric}"].to_numpy(dtype=np.float64))
                    for metric in metric_names
                ]
            )
            finite = np.isfinite(matrix)
            weighted_values = np.where(finite, matrix * weights[None, :], 0.0)
            denominators = np.where(finite, weights[None, :], 0.0).sum(axis=1)
            combined = np.divide(
                weighted_values.sum(axis=1),
                denominators,
                out=np.full(len(matrix), np.nan, dtype=np.float64),
                where=denominators > 0,
            )
            finite_combined = combined[np.isfinite(combined)]
            record[f"{role}_validation_combined_abs_mean"] = (
                float(np.mean(finite_combined)) if len(finite_combined) else np.nan
            )
            record[f"{role}_validation_combined_abs_min"] = (
                float(np.min(finite_combined)) if len(finite_combined) else np.nan
            )
            record[f"{role}_validation_combined_abs_std"] = (
                float(np.std(finite_combined, ddof=1)) if len(finite_combined) > 1 else 0.0
            ) if len(finite_combined) else np.nan

        selection_mean = float(record.get("selection_validation_pearson_mean", np.nan))
        confirmation_mean = float(record.get("confirmation_validation_pearson_mean", np.nan))
        recent_mean = float(record.get("recent_audit_validation_pearson_mean", np.nan))
        selection_sign = np.sign(selection_mean)
        confirmation_sign = np.sign(confirmation_mean)
        recent_sign = np.sign(recent_mean)
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
        selection_abs = abs(selection_mean) if np.isfinite(selection_mean) else np.nan
        record["confirmation_to_selection_abs_ratio"] = (
            abs(confirmation_mean) / selection_abs
            if np.isfinite(confirmation_mean) and np.isfinite(selection_abs) and selection_abs > 1e-12
            else np.nan
        )
        record["recent_to_selection_abs_ratio"] = (
            abs(recent_mean) / selection_abs
            if np.isfinite(recent_mean) and np.isfinite(selection_abs) and selection_abs > 1e-12
            else np.nan
        )
        records.append(record)
    return pd.DataFrame.from_records(records)




def compute_groupwise_target_correlation_distribution(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    groups: np.ndarray,
    features: Sequence[str],
    prefix: str,
    min_group_rows: int,
    chunk_size: int,
) -> pd.DataFrame:
    """Aggregate literal per-date or per-ticker correlations for every feature.

    This is distinct from pooled group demeaning. Groups with too few pairwise
    finite rows or a single target class remain NaN and are not coerced to zero.
    """

    target = np.asarray(target, dtype=np.float64)
    groups = np.asarray(groups)
    if len(feature_frame) != len(target) or len(groups) != len(target):
        raise ValueError("groupwise correlation input lengths differ")
    codes, unique_groups = pd.factorize(groups, sort=False)
    if np.any(codes < 0):
        valid_group_rows = codes >= 0
        codes = codes[valid_group_rows]
        target = target[valid_group_rows]
        feature_frame = feature_frame.iloc[np.flatnonzero(valid_group_rows)].reset_index(drop=True)
    order = np.argsort(codes, kind="stable")
    counts_by_group = np.bincount(codes, minlength=len(unique_groups))
    split_offsets = np.cumsum(counts_by_group, dtype=np.int64)[:-1]
    group_indices = np.split(order, split_offsets) if len(unique_groups) else []
    records: list[dict[str, Any]] = []

    for start in range(0, len(features), chunk_size):
        names = list(features[start : start + chunk_size])
        values = feature_frame[names].to_numpy(dtype=np.float64, copy=True)
        correlation_rows: list[np.ndarray] = []
        count_rows: list[np.ndarray] = []
        for indices in group_indices:
            if len(indices) < min_group_rows:
                continue
            group_target = target[indices]
            finite_target = group_target[np.isfinite(group_target)]
            if len(np.unique(finite_target)) < 2:
                continue
            correlations, counts, _ = vector_target_corr(values[indices], group_target, min_group_rows)
            correlation_rows.append(correlations)
            count_rows.append(counts.astype(np.float64))
        if correlation_rows:
            correlation_matrix = np.vstack(correlation_rows)
            count_matrix = np.vstack(count_rows)
        else:
            correlation_matrix = np.empty((0, len(names)), dtype=np.float64)
            count_matrix = np.empty((0, len(names)), dtype=np.float64)

        for offset, feature in enumerate(names):
            correlations = correlation_matrix[:, offset] if len(correlation_matrix) else np.asarray([], dtype=float)
            counts = count_matrix[:, offset] if len(count_matrix) else np.asarray([], dtype=float)
            valid = np.isfinite(correlations)
            finite = correlations[valid]
            finite_counts = counts[valid]
            if len(finite):
                weighted_mean = float(np.average(finite, weights=np.maximum(finite_counts, 1.0)))
                positive_ratio = float(np.mean(finite > 1e-12))
                negative_ratio = float(np.mean(finite < -1e-12))
                q25, q75 = np.quantile(finite, [0.25, 0.75])
            else:
                weighted_mean = positive_ratio = negative_ratio = q25 = q75 = np.nan
            stats = aggregate_metric(finite)
            records.append(
                {
                    "feature": feature,
                    f"{prefix}_group_count_total": int(len(unique_groups)),
                    f"{prefix}_group_count_eligible": int(len(correlation_rows)),
                    f"{prefix}_group_count_valid": int(stats["folds"]),
                    f"{prefix}_corr_mean": stats["mean"],
                    f"{prefix}_corr_weighted_mean": weighted_mean,
                    f"{prefix}_corr_median": stats["median"],
                    f"{prefix}_corr_std": stats["std"],
                    f"{prefix}_corr_q25": float(q25),
                    f"{prefix}_corr_q75": float(q75),
                    f"{prefix}_corr_abs_mean": stats["abs_mean"],
                    f"{prefix}_corr_sign_consistency": stats["sign_consistency"],
                    f"{prefix}_corr_positive_ratio": positive_ratio,
                    f"{prefix}_corr_negative_ratio": negative_ratio,
                }
            )
    return pd.DataFrame.from_records(records)


def compute_mutual_information_by_selection_fold(
    feature_frame: pd.DataFrame,
    target: np.ndarray,
    target_valid: np.ndarray,
    dates: np.ndarray,
    features: Sequence[str],
    definitions: Sequence[FoldDefinition],
    selection_fold_ids: Sequence[int],
    sample_rows: int,
    bins: int,
    min_rows: int,
    seed: int,
    skip: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate binned MI in each selection fold's train segment only."""

    selection_ids = set(int(value) for value in selection_fold_ids)
    long_frames: list[pd.DataFrame] = []
    date_series = pd.Series(pd.to_datetime(dates))
    if not skip:
        for fold in definitions:
            if fold.fold_id not in selection_ids:
                continue
            mask = target_valid & date_series.between(fold.train_start, fold.train_end).to_numpy()
            indices = deterministic_complete_date_sample(
                dates,
                mask,
                sample_rows,
                seed + 500 + fold.fold_id * 37,
            )
            if not len(indices):
                raise ValueError(f"selection fold {fold.fold_id} MI train rows are empty")
            log(f"selection fold {fold.fold_id} train MI: {len(indices):,} rows")
            part = binned_mutual_information(
                feature_frame.iloc[indices],
                target[indices],
                features,
                bins=bins,
                min_rows=min_rows,
            )
            part.insert(0, "outer_fold", fold.fold_id)
            part.insert(1, "sample_rows", int(len(indices)))
            long_frames.append(part)

    if long_frames:
        long_table = pd.concat(long_frames, ignore_index=True)
        records: list[dict[str, Any]] = []
        for feature, part in long_table.groupby("feature", sort=False):
            mi_stats = aggregate_metric(part["mutual_information"].to_numpy(dtype=np.float64))
            nmi_stats = aggregate_metric(part["normalized_mutual_information"].to_numpy(dtype=np.float64))
            records.append(
                {
                    "feature": feature,
                    "mutual_information": mi_stats["mean"],
                    "normalized_mutual_information": nmi_stats["mean"],
                    "selection_mi_mean": mi_stats["mean"],
                    "selection_mi_median": mi_stats["median"],
                    "selection_mi_std": mi_stats["std"],
                    "selection_mi_min": mi_stats["min"],
                    "selection_mi_max": mi_stats["max"],
                    "selection_normalized_mi_mean": nmi_stats["mean"],
                    "selection_normalized_mi_median": nmi_stats["median"],
                    "selection_normalized_mi_std": nmi_stats["std"],
                    "selection_normalized_mi_min": nmi_stats["min"],
                    "selection_normalized_mi_max": nmi_stats["max"],
                    "selection_mi_folds": nmi_stats["folds"],
                }
            )
        summary = pd.DataFrame.from_records(records)
    else:
        long_table = pd.DataFrame(
            columns=["outer_fold", "sample_rows", "feature", "mutual_information", "normalized_mutual_information"]
        )
        summary = pd.DataFrame(
            {
                "feature": list(features),
                "mutual_information": np.nan,
                "normalized_mutual_information": np.nan,
                "selection_mi_mean": np.nan,
                "selection_mi_median": np.nan,
                "selection_mi_std": np.nan,
                "selection_mi_min": np.nan,
                "selection_mi_max": np.nan,
                "selection_normalized_mi_mean": np.nan,
                "selection_normalized_mi_median": np.nan,
                "selection_normalized_mi_std": np.nan,
                "selection_normalized_mi_min": np.nan,
                "selection_normalized_mi_max": np.nan,
                "selection_mi_folds": 0,
            }
        )
    return long_table, summary

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
    inf_counts: dict[str, int] | None = None,
    source_dtypes: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build a full feature-quality table from cleaned point-in-time values.

    Infinite values are counted before cleaning by ``load_and_validate_data`` and
    are represented as missing values in ``feature_frame``. No interpolation or
    forward/backward filling is performed.
    """

    inf_counts = inf_counts or {}
    source_dtypes = source_dtypes or {}
    n_rows = int(len(feature_frame))
    observed = feature_frame.notna()
    non_null_count = observed.sum(axis=0)
    missing_count = n_rows - non_null_count
    missing_ratio = 1.0 - observed.mean(axis=0)
    unique_count = feature_frame.nunique(dropna=True)
    ticker_series = pd.Series(tickers, index=feature_frame.index, name="ticker")
    coverage = observed.groupby(ticker_series, sort=False, observed=True).any().mean(axis=0)

    group_map: dict[str, str] = {}
    reference_status: dict[str, str] = {}
    if reference_audit is not None and reference_audit.exists():
        reference = pd.read_csv(reference_audit)
        if "feature" in reference.columns:
            reference = reference.drop_duplicates("feature")
            if "group" in reference.columns:
                group_map = reference.set_index("feature")["group"].astype(str).to_dict()
            if "status" in reference.columns:
                reference_status = reference.set_index("feature")["status"].astype(str).to_dict()

    records: list[dict[str, Any]] = []
    for feature in features:
        series = pd.to_numeric(feature_frame[feature], errors="coerce")
        finite_values = series.dropna().to_numpy(dtype=np.float64)
        valid_count = int(len(finite_values))
        missing = float(missing_ratio[feature])
        unique = int(unique_count[feature])
        ticker_coverage = float(coverage[feature])
        inf_count = int(inf_counts.get(feature, 0))
        std = float(np.std(finite_values, ddof=1)) if valid_count > 1 else np.nan
        near_zero_variance = bool(np.isfinite(std) and std <= 1e-12)
        rejection_reasons: list[str] = []
        if missing > 0.995:
            rejection_reasons.append("missing_ratio_gt_0.995")
        if valid_count < 2:
            rejection_reasons.append("fewer_than_2_finite_rows")
        if unique < 2:
            rejection_reasons.append("fewer_than_2_unique_values")
        if near_zero_variance:
            rejection_reasons.append("near_zero_variance")
        status = "valid" if not rejection_reasons else "invalid"

        if valid_count:
            quantiles = np.quantile(finite_values, [0.01, 0.05, 0.50, 0.95, 0.99])
            mean = float(np.mean(finite_values))
            value_min = float(np.min(finite_values))
            value_max = float(np.max(finite_values))
        else:
            quantiles = np.full(5, np.nan, dtype=np.float64)
            mean = value_min = value_max = np.nan

        completeness = 1.0 - missing
        uniqueness_score = min(1.0, math.log1p(unique) / math.log1p(1000.0))
        finite_quality = 1.0 - min(1.0, inf_count / max(1, n_rows))
        data_quality_score = (
            0.45 * completeness
            + 0.25 * ticker_coverage
            + 0.15 * uniqueness_score
            + 0.15 * finite_quality
        )
        records.append(
            {
                "feature": feature,
                "source_dtype": source_dtypes.get(feature, "unknown"),
                "status": status,
                "rejection_reason": ";".join(rejection_reasons),
                "n_rows": n_rows,
                "n_valid": valid_count,
                "non_null_count": int(non_null_count[feature]),
                "missing_count": int(missing_count[feature]),
                "missing_ratio": missing,
                "inf_count": inf_count,
                "inf_ratio": float(inf_count / n_rows) if n_rows else np.nan,
                "unique_count": unique,
                "near_zero_variance": near_zero_variance,
                "mean": mean,
                "std": std,
                "min": value_min,
                "max": value_max,
                "p01": float(quantiles[0]),
                "p05": float(quantiles[1]),
                "p50": float(quantiles[2]),
                "p95": float(quantiles[3]),
                "p99": float(quantiles[4]),
                "ticker_coverage": ticker_coverage,
                "group": group_map.get(feature, "unknown"),
                "reference_status": reference_status.get(feature, "unknown"),
                "data_quality_score": float(np.clip(data_quality_score, 0.0, 1.0)),
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
    requested_features = list(dict.fromkeys(str(feature) for feature in requested_features))
    missing_features = [feature for feature in requested_features if feature not in source_columns]
    if missing_features:
        raise KeyError(f"profile 피처가 원본에 없습니다: {missing_features[:30]}")
    leakage = [
        {"feature": feature, "reason": leakage_reason(feature)}
        for feature in requested_features
        if leakage_reason(feature)
    ]
    if leakage:
        raise ValueError(f"요청 피처에 누수 의심 열이 있습니다: {leakage[:20]}")

    read_columns = ["date", "ticker"] + requested_features
    if "t_price_ret_1" in source_columns and "t_price_ret_1" not in read_columns:
        read_columns.append("t_price_ret_1")
    if "sealed_do_not_train_or_tune" in source_columns:
        read_columns.append("sealed_do_not_train_or_tune")
    source = read_table_columns(source_path, read_columns).copy()
    source.insert(0, "source_row_id", np.arange(len(source), dtype=np.int64))
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    if source["date"].isna().any():
        raise ValueError(f"원본 date 파싱 실패 행={int(source['date'].isna().sum())}")
    source["ticker"] = source["ticker"].map(normalize_ticker)
    if source["ticker"].eq("").any():
        raise ValueError(f"원본 ticker가 비어 있는 행={int(source['ticker'].eq('').sum())}")
    duplicate_date_ticker = int(source.duplicated(["date", "ticker"]).sum())
    if duplicate_date_ticker:
        raise ValueError(f"원본에 date/ticker 중복 행이 있습니다: {duplicate_date_ticker}")
    if "sealed_do_not_train_or_tune" in source.columns:
        safety = pd.to_numeric(source["sealed_do_not_train_or_tune"], errors="coerce").fillna(0)
        unsafe_rows = int(safety.ne(0).sum())
        if unsafe_rows:
            raise ValueError(f"sealed_do_not_train_or_tune 위반 행={unsafe_rows}")

    target_columns = table_columns(target_path)
    required_target = ["source_row_id", "date", "ticker", TARGET_NAME, TARGET_VALID_COLUMN]
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
    target_sidecar = target_sidecar.assign(source_row_id=source_row_id.astype(np.int64)).sort_values(
        "source_row_id", kind="mergesort"
    )
    expected = np.arange(len(source), dtype=np.int64)
    if not np.array_equal(target_sidecar["source_row_id"].to_numpy(dtype=np.int64), expected):
        raise ValueError("target source_row_id가 0..N-1의 완전한 일대일 키가 아닙니다")
    target_sidecar = target_sidecar.reset_index(drop=True)
    target_dates = pd.to_datetime(target_sidecar["date"], errors="coerce")
    target_tickers = target_sidecar["ticker"].map(normalize_ticker)
    if not np.array_equal(
        source["date"].to_numpy(dtype="datetime64[ns]"),
        target_dates.to_numpy(dtype="datetime64[ns]"),
    ):
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
    inf_counts: dict[str, int] = {}
    source_dtypes: dict[str, str] = {}
    for feature in requested_features:
        source_dtypes[feature] = str(source[feature].dtype)
        values = pd.to_numeric(source[feature], errors="coerce")
        raw = values.to_numpy(dtype=np.float64, copy=False)
        inf_counts[feature] = int(np.isinf(raw).sum())
        numeric_features[feature] = values.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    feature_frame = pd.DataFrame(numeric_features, index=source.index)
    quality = compute_feature_quality(
        feature_frame,
        source["ticker"].to_numpy(),
        requested_features,
        valid_feature_audit,
        inf_counts=inf_counts,
        source_dtypes=source_dtypes,
    )
    valid_features = quality.loc[quality["status"].eq("valid"), "feature"].astype(str).tolist()
    if not valid_features:
        raise ValueError("결측률/유일값/분산 감사 후 유효 피처가 없습니다")
    quality["selected_for_correlation"] = quality["feature"].isin(valid_features)
    feature_frame = feature_frame[valid_features].copy()

    ticker_date_order_ok = bool(
        source.groupby("ticker", sort=False, observed=True)["date"].apply(lambda values: values.is_monotonic_increasing).all()
    )
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
        "date_non_decreasing_global": bool(source["date"].is_monotonic_increasing),
        "date_non_decreasing_within_ticker": ticker_date_order_ok,
        "sealed_nonzero_rows": 0,
        "source_row_id_date_ticker_match": True,
        "total_infinite_feature_values_replaced_with_nan": int(sum(inf_counts.values())),
        "missing_value_imputation_used": False,
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
    """Verify every prerequisite before reusing target-independent matrices."""

    details: dict[str, Any] = {
        "legacy_dir_exists": legacy_dir.exists(),
        "dataset_hash_match": False,
        "features_subset_of_legacy": False,
        "fold_definition_match": False,
        "required_files_present": False,
        "required_file_hashes_match": False,
    }
    required_names = [
        "correlation_manifest.json",
        "correlation_matrices.npz",
        "fold_pearson_matrices.npz",
        "pearson.csv.gz",
        "spearman.csv.gz",
        "within_date_pearson.csv.gz",
        "within_ticker_pearson.csv.gz",
        "cluster_basis_combined_abs.csv.gz",
    ]
    required = [legacy_dir / name for name in required_names]
    details["required_files_present"] = all(path.is_file() and path.stat().st_size > 0 for path in required)
    if not details["legacy_dir_exists"] or not details["required_files_present"]:
        return False, details

    source_hash = sha256_file(source_path)
    details["source_sha256"] = source_hash
    dataset_audit = package_root / "data" / "DATASET_AUDIT.json"
    if dataset_audit.exists():
        audit = json.loads(dataset_audit.read_text(encoding="utf-8"))
        expected = str(audit.get("dataset_sha256", ""))
        details["reference_dataset_sha256"] = expected
        details["dataset_hash_match"] = bool(expected and source_hash == expected)

    profile_manifest = package_root / "references" / "feature_metadata" / "profile_manifest.json"
    if profile_manifest.exists():
        payload = json.loads(profile_manifest.read_text(encoding="utf-8"))
        legacy_features = [
            str(value)
            for value in payload.get("profiles", {}).get("P0_FULL_439", {}).get("features", [])
        ]
        details["legacy_feature_count"] = len(legacy_features)
        details["features_subset_of_legacy"] = set(features).issubset(set(legacy_features))
        details["requested_feature_order_hash"] = hash_feature_list(features)

    reference_folds = package_root / "references" / "feature_metadata" / "outer_walk_forward_folds.json"
    if reference_folds.exists() and fold_path.exists():
        current_fold_payload = json.loads(fold_path.read_text(encoding="utf-8"))
        reference_fold_payload = json.loads(reference_folds.read_text(encoding="utf-8"))
        details["current_fold_hash"] = sha256_bytes(stable_json_bytes(current_fold_payload))
        details["reference_fold_hash"] = sha256_bytes(stable_json_bytes(reference_fold_payload))
        details["fold_definition_match"] = current_fold_payload == reference_fold_payload

    checksums_path = package_root / "SHA256SUMS.json"
    hash_records: list[dict[str, Any]] = []
    if checksums_path.exists():
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
        for path in required:
            try:
                relative = path.resolve().relative_to(package_root.resolve()).as_posix()
            except ValueError:
                relative = ""
            expected_hash = str(checksums.get(relative, "")) if relative else ""
            actual_hash = sha256_file(path)
            hash_records.append(
                {
                    "file": path.name,
                    "relative_path": relative,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual_hash,
                    "match": bool(expected_hash and expected_hash == actual_hash),
                }
            )
        details["required_file_hashes_match"] = bool(hash_records) and all(
            record["match"] for record in hash_records
        )
    details["legacy_file_hash_audit"] = hash_records

    compatible = bool(
        details["dataset_hash_match"]
        and details["features_subset_of_legacy"]
        and details["fold_definition_match"]
        and details["required_files_present"]
        and details["required_file_hashes_match"]
    )
    details["compatible"] = compatible
    return compatible, details




def load_legacy_matrix_bundle(
    legacy_dir: Path,
    package_root: Path,
    features: Sequence[str],
) -> MatrixBundle:
    profile_payload = json.loads(
        (package_root / "references" / "feature_metadata" / "profile_manifest.json").read_text(encoding="utf-8")
    )
    legacy_features = [str(item) for item in profile_payload["profiles"]["P0_FULL_439"]["features"]]
    missing = [feature for feature in features if feature not in legacy_features]
    if missing:
        raise ValueError(f"legacy matrix에 없는 피처: {missing[:20]}")
    indices = np.array([legacy_features.index(feature) for feature in features], dtype=np.int64)
    with np.load(legacy_dir / "correlation_matrices.npz", allow_pickle=False) as matrix_payload:
        required_keys = {
            "pearson",
            "spearman",
            "within_date",
            "within_ticker",
            "missingness",
            "combined_abs",
            "cluster_basis_abs",
        }
        missing_keys = sorted(required_keys - set(matrix_payload.files))
        if missing_keys:
            raise ValueError(f"legacy correlation_matrices.npz key 누락: {missing_keys}")

        def subset(name: str) -> np.ndarray:
            matrix = np.asarray(matrix_payload[name], dtype=np.float32)
            if matrix.shape != (len(legacy_features), len(legacy_features)):
                raise ValueError(f"legacy {name} shape 불일치: {matrix.shape}")
            result = matrix[np.ix_(indices, indices)]
            if not np.allclose(result, result.T, atol=1e-5, equal_nan=True):
                raise ValueError(f"legacy {name} matrix가 대칭이 아닙니다")
            return result

        matrices = {name: subset(name) for name in required_keys}

    with np.load(legacy_dir / "fold_pearson_matrices.npz", allow_pickle=False) as fold_payload:
        if not {"fold_ids", "matrices"}.issubset(fold_payload.files):
            raise ValueError("legacy fold matrix key 누락")
        fold_ids = np.asarray(fold_payload["fold_ids"], dtype=np.int16)
        fold_matrices = np.asarray(fold_payload["matrices"], dtype=np.float32)
        if fold_matrices.ndim != 3 or fold_matrices.shape[1:] != (
            len(legacy_features),
            len(legacy_features),
        ):
            raise ValueError(f"legacy fold matrix shape 불일치: {fold_matrices.shape}")
        fold_matrices = fold_matrices[:, indices][:, :, indices]

    manifest = json.loads((legacy_dir / "correlation_manifest.json").read_text(encoding="utf-8"))
    return MatrixBundle(
        pearson=matrices["pearson"],
        spearman=matrices["spearman"],
        within_date=matrices["within_date"],
        within_ticker=matrices["within_ticker"],
        missingness=matrices["missingness"],
        combined_abs=matrices["combined_abs"],
        cluster_basis_abs=matrices["cluster_basis_abs"],
        fold_ids=fold_ids,
        fold_pearson=fold_matrices,
        source="legacy_feature_structure_hash_verified",
        source_manifest=manifest,
    )



def intersect_row_limits(first: int, second: int) -> int:
    """Intersect two row limits where zero means unlimited."""

    first = int(first)
    second = int(second)
    if first <= 0 and second <= 0:
        return 0
    if first <= 0:
        return second
    if second <= 0:
        return first
    return min(first, second)


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
        dates, cluster_mask, intersect_row_limits(cluster_basis_rows, spearman_rows), seed + 4
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
    """Train-only candidate score used inside a correlation cluster."""

    fold = fold_correlations[fold_correlations["outer_fold"].eq(fold_id)].set_index("feature")
    if fold.empty:
        raise ValueError(f"representative score fold {fold_id} is missing")
    quality_indexed = quality.set_index("feature")
    metric_weights = {
        "train_target_pearson": 0.35,
        "train_target_spearman": 0.25,
        "train_target_within_date_pearson": 0.25,
        "train_target_within_ticker_pearson": 0.15,
    }
    metric_arrays: list[np.ndarray] = []
    weights: list[float] = []
    for column, weight in metric_weights.items():
        if column in fold.columns:
            metric_arrays.append(np.abs(fold.reindex(features)[column].to_numpy(dtype=np.float64)))
            weights.append(weight)
    if not metric_arrays:
        raise ValueError("representative score에 사용할 train correlation 열이 없습니다")
    matrix = np.column_stack(metric_arrays)
    weight_array = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(matrix)
    weighted = np.where(finite, matrix * weight_array[None, :], 0.0)
    denominators = np.where(finite, weight_array[None, :], 0.0).sum(axis=1)
    signal = np.divide(
        weighted.sum(axis=1),
        denominators,
        out=np.zeros(len(features), dtype=np.float64),
        where=denominators > 0,
    )
    signal_rank = pd.Series(signal, index=features).rank(method="average", pct=True).fillna(0.0)
    data_quality = quality_indexed.reindex(features)["data_quality_score"].fillna(0.0)
    missing_penalty = 1.0 - quality_indexed.reindex(features)["missing_ratio"].fillna(1.0)
    score = 0.70 * signal_rank + 0.20 * data_quality + 0.10 * missing_penalty
    return score.clip(0.0, 1.0).astype(float)




def choose_representative(
    members: Sequence[int],
    features: Sequence[str],
    scores: pd.Series,
    centrality: dict[int, float],
    quality: pd.DataFrame,
) -> tuple[int, float]:
    """Choose a deterministic fold-specific representative."""

    quality_indexed = quality.set_index("feature")
    candidates: list[tuple[float, float, float, str, int]] = []
    for member in members:
        member_index = int(member)
        feature = features[member_index]
        base = float(scores.get(feature, 0.0))
        central = float(centrality.get(member_index, 0.0))
        missing = float(quality_indexed.at[feature, "missing_ratio"])
        final = 0.75 * base + 0.25 * central
        candidates.append((final, central, -missing, feature, member_index))
    candidates.sort(key=lambda item: (-item[0], -item[1], -item[2], item[3]))
    best = candidates[0]
    return best[-1], float(best[0])




def build_consensus_cluster_details(
    components: Sequence[Sequence[int]],
    matrix: np.ndarray,
    features: Sequence[str],
    quality: pd.DataFrame,
    fold_correlations: pd.DataFrame,
    selection_fold_ids: Sequence[int],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build consensus representatives from all selection-fold train segments.

    Confirmation and recent folds are deliberately excluded. Each selection
    fold casts one winner vote, while the final representative also considers
    mean/minimum train-only score, score dispersion, centrality and missingness.
    """

    selection_ids = sorted(set(int(value) for value in selection_fold_ids))
    if not selection_ids:
        raise ValueError("consensus representative에는 selection fold가 하나 이상 필요합니다")
    score_by_fold = {
        fold_id: representative_scores(features, quality, fold_correlations, fold_id)
        for fold_id in selection_ids
    }
    quality_indexed = quality.set_index("feature")
    assignment_records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    winner_records: list[dict[str, Any]] = []

    for cluster_id, members_value in enumerate(components, start=1):
        members = [int(value) for value in members_value]
        centrality = within_cluster_centrality(matrix, members)
        fold_winners: dict[int, int] = {}
        fold_candidate_scores: dict[int, dict[int, float]] = {}
        for fold_id in selection_ids:
            scores = score_by_fold[fold_id]
            candidate_scores = {
                member: 0.75 * float(scores.get(features[member], 0.0))
                + 0.25 * float(centrality.get(member, 0.0))
                for member in members
            }
            fold_candidate_scores[fold_id] = candidate_scores
            winner = sorted(
                members,
                key=lambda member: (
                    -candidate_scores[member],
                    -float(centrality.get(member, 0.0)),
                    float(quality_indexed.at[features[member], "missing_ratio"]),
                    features[member],
                ),
            )[0]
            fold_winners[fold_id] = winner
            winner_records.append(
                {
                    "outer_fold": fold_id,
                    "fold_role": "selection",
                    "threshold": float(threshold),
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(members)),
                    "representative": features[winner],
                    "representative_score": float(candidate_scores[winner]),
                }
            )

        vote_counter = Counter(fold_winners.values())
        candidate_stats: dict[int, dict[str, float | int]] = {}
        for member in members:
            values = np.asarray(
                [fold_candidate_scores[fold_id][member] for fold_id in selection_ids],
                dtype=np.float64,
            )
            mean_score = float(np.mean(values))
            min_score = float(np.min(values))
            std_score = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            vote_count = int(vote_counter.get(member, 0))
            vote_ratio = float(vote_count / len(selection_ids))
            central = float(centrality.get(member, 0.0))
            dispersion_score = float(np.clip(1.0 - std_score, 0.0, 1.0))
            consensus_score = (
                0.50 * mean_score
                + 0.15 * min_score
                + 0.10 * dispersion_score
                + 0.15 * central
                + 0.10 * vote_ratio
            )
            candidate_stats[member] = {
                "mean": mean_score,
                "min": min_score,
                "std": std_score,
                "vote_count": vote_count,
                "vote_ratio": vote_ratio,
                "centrality": central,
                "consensus": float(consensus_score),
            }

        representative_index = sorted(
            members,
            key=lambda member: (
                -float(candidate_stats[member]["consensus"]),
                -float(candidate_stats[member]["vote_ratio"]),
                -float(candidate_stats[member]["mean"]),
                -float(candidate_stats[member]["min"]),
                -float(candidate_stats[member]["centrality"]),
                float(quality_indexed.at[features[member], "missing_ratio"]),
                features[member],
            ),
        )[0]
        representative = features[representative_index]
        representative_stats = candidate_stats[representative_index]

        for member in members:
            feature = features[member]
            stats = candidate_stats[member]
            candidate_records.append(
                {
                    "threshold": float(threshold),
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(members)),
                    "feature": feature,
                    "selection_fold_score_mean": stats["mean"],
                    "selection_fold_score_min": stats["min"],
                    "selection_fold_score_std": stats["std"],
                    "winner_vote_count": stats["vote_count"],
                    "winner_vote_ratio": stats["vote_ratio"],
                    "within_cluster_centrality": stats["centrality"],
                    "consensus_representative_score": stats["consensus"],
                    "is_consensus_representative": bool(member == representative_index),
                }
            )
            assignment_records.append(
                {
                    "threshold": float(threshold),
                    "cluster_id": int(cluster_id),
                    "cluster_size": int(len(members)),
                    "feature": feature,
                    "representative": representative,
                    "is_representative": bool(member == representative_index),
                    "predictive_quality_score": float(stats["mean"]),
                    "representative_quality_score": float(representative_stats["consensus"]),
                    "within_cluster_centrality": float(stats["centrality"]),
                    "representative_selection_folds": ",".join(map(str, selection_ids)),
                    "representative_vote_count": int(representative_stats["vote_count"]),
                    "representative_vote_ratio": float(representative_stats["vote_ratio"]),
                    "representative_selection_score_mean": float(representative_stats["mean"]),
                    "representative_selection_score_min": float(representative_stats["min"]),
                    "representative_selection_score_std": float(representative_stats["std"]),
                }
            )

    return (
        pd.DataFrame.from_records(assignment_records),
        pd.DataFrame.from_records(candidate_records),
        pd.DataFrame.from_records(winner_records),
    )



def build_cluster_assignments(
    matrix: np.ndarray,
    features: Sequence[str],
    thresholds: Sequence[float],
    quality: pd.DataFrame,
    fold_correlations: pd.DataFrame,
    selection_fold_ids: Sequence[int],
) -> tuple[
    pd.DataFrame,
    dict[float, list[list[int]]],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Cluster features and select representatives from all selection folds.

    The correlation structure is target-independent, but the representative is
    target-dependent.  For each threshold the average-linkage partition is built
    once.  Every selection fold then casts a train-only winner vote inside each
    cluster.  Confirmation and recent-audit folds never participate in the
    representative choice.
    """

    selection_ids = sorted(set(int(item) for item in selection_fold_ids))
    if not selection_ids:
        raise ValueError("cluster 대표 선정을 위한 selection fold가 없습니다")

    linkage_tree = average_linkage_tree(matrix)
    components_by_threshold: dict[float, list[list[int]]] = {}
    assignment_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    winner_frames: list[pd.DataFrame] = []

    for threshold_value in thresholds:
        threshold = float(threshold_value)
        components = average_linkage_components(matrix, threshold, linkage_tree)
        components_by_threshold[threshold] = components
        assignments, candidates, winners = build_consensus_cluster_details(
            components=components,
            matrix=matrix,
            features=features,
            quality=quality,
            fold_correlations=fold_correlations,
            selection_fold_ids=selection_ids,
            threshold=threshold,
        )
        assignment_frames.append(assignments)
        candidate_frames.append(candidates)
        winner_frames.append(winners)

    assignments = pd.concat(assignment_frames, ignore_index=True)
    candidates = pd.concat(candidate_frames, ignore_index=True)
    winners = pd.concat(winner_frames, ignore_index=True)
    assignments = assignments.sort_values(
        ["threshold", "cluster_id", "feature"],
        kind="mergesort",
    ).reset_index(drop=True)
    candidates = candidates.sort_values(
        ["threshold", "cluster_id", "consensus_representative_score", "feature"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    winners = winners.sort_values(
        ["threshold", "outer_fold", "cluster_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    return assignments, components_by_threshold, candidates, winners


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
            representative_index, final_score = choose_representative(
                members, features, scores, centrality, quality
            )
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
    return pd.DataFrame.from_records(records).sort_values(
        ["outer_fold", "cluster_id"], kind="mergesort"
    ).reset_index(drop=True)




def build_representative_consensus_table(primary_clusters: pd.DataFrame) -> pd.DataFrame:
    representatives = primary_clusters[primary_clusters["is_representative"]].copy()
    columns = [
        "threshold",
        "cluster_id",
        "cluster_size",
        "representative",
        "representative_vote_count",
        "representative_vote_share",
        "representative_quality_score",
        "representative_runner_up",
        "representative_selection_method",
        "representative_selection_folds",
        "consensus_fold_representatives",
        "within_cluster_centrality",
    ]
    return representatives[[column for column in columns if column in representatives.columns]].sort_values(
        ["cluster_id", "representative"], kind="mergesort"
    ).reset_index(drop=True)

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




def derive_selection_direction(row: pd.Series) -> tuple[int, float, str]:
    """Choose a selection-only direction and the metric that established it.

    The strongest mean validation association among Pearson, Spearman, pooled
    within-date Pearson and pooled within-ticker Pearson is used.  The same
    metric is then used for all strict fold-support checks, preventing a
    Spearman-derived direction from being validated with unrelated raw Pearson
    signs.
    """

    candidates = [
        ("pearson", float(row.get("selection_validation_pearson_mean", np.nan))),
        ("spearman", float(row.get("selection_validation_spearman_mean", np.nan))),
        ("within_date_pearson", float(row.get("selection_validation_within_date_pearson_mean", np.nan))),
        ("within_ticker_pearson", float(row.get("selection_validation_within_ticker_pearson_mean", np.nan))),
    ]
    finite = [(name, value) for name, value in candidates if np.isfinite(value) and abs(value) > 1e-12]
    if not finite:
        return 0, np.nan, "none"
    name, value = sorted(finite, key=lambda item: (-abs(item[1]), item[0]))[0]
    return int(np.sign(value)), float(abs(value)), name


def evaluate_support_evidence(
    summary: pd.DataFrame,
    fold_correlations: pd.DataFrame,
    roles: dict[str, list[int]],
    selection_min_abs_corr: float,
    selection_min_sign_consistency: float,
    confirmation_min_abs_corr: float,
    confirmation_min_sign_consistency: float,
    confirmation_min_retention: float,
    recent_min_abs_corr: float,
    recent_min_sign_consistency: float,
    recent_min_retention: float,
) -> pd.DataFrame:
    """Apply strict selection, confirmation and recent-audit support rules.

    Selection ranking has already been calculated before this function runs.
    Therefore confirmation and recent evidence can only gate diagnostic profiles;
    it cannot improve the selection priority score.

    Confirmation/recent support requires all configured folds to be present, all
    fold signs to agree with the selection direction, the sign-consistency gate,
    a minimum absolute correlation in *every* fold, and a minimum mean-signal
    retention ratio relative to selection.  Missing values are failures rather
    than zeros or implicit passes.
    """

    metric_to_column = {
        "pearson": "validation_target_pearson",
        "spearman": "validation_target_spearman",
        "within_date_pearson": "validation_target_within_date_pearson",
        "within_ticker_pearson": "validation_target_within_ticker_pearson",
    }
    fold_index = {
        str(feature): part.set_index("outer_fold", drop=False)
        for feature, part in fold_correlations.groupby("feature", sort=False)
    }
    expected_selection = sorted(set(int(value) for value in roles.get("selection", [])))
    expected_confirmation = sorted(set(int(value) for value in roles.get("confirmation", [])))
    expected_recent = sorted(set(int(value) for value in roles.get("recent_audit", [])))
    records: list[dict[str, Any]] = []

    def role_values(
        part: pd.DataFrame,
        fold_ids: Sequence[int],
        metric_column: str,
    ) -> np.ndarray:
        if not fold_ids or part.empty or metric_column not in part.columns:
            return np.asarray([], dtype=np.float64)
        return part.reindex(fold_ids)[metric_column].to_numpy(dtype=np.float64)

    def period_statistics(values: np.ndarray) -> dict[str, float | int]:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return {
                "finite_count": 0,
                "mean": np.nan,
                "abs_mean_of_signed_mean": np.nan,
                "mean_abs": np.nan,
                "min_abs": np.nan,
                "sign_consistency": np.nan,
            }
        mean = float(np.mean(finite))
        return {
            "finite_count": int(len(finite)),
            "mean": mean,
            "abs_mean_of_signed_mean": abs(mean),
            "mean_abs": float(np.mean(np.abs(finite))),
            "min_abs": float(np.min(np.abs(finite))),
            "sign_consistency": float(sign_consistency(finite)),
        }

    for row in summary.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        feature = str(row_series["feature"])
        direction, selection_abs_strength, direction_source = derive_selection_direction(row_series)
        metric_column = metric_to_column.get(direction_source, "validation_target_pearson")
        part = fold_index.get(feature, pd.DataFrame())

        selection_values = role_values(part, expected_selection, metric_column)
        confirmation_values = role_values(part, expected_confirmation, metric_column)
        recent_values = role_values(part, expected_recent, metric_column)
        selection_stats = period_statistics(selection_values)
        confirmation_stats = period_statistics(confirmation_values)
        recent_stats = period_statistics(recent_values)

        selection_finite = selection_values[np.isfinite(selection_values)]
        selection_reasons: list[str] = []
        if not expected_selection:
            selection_reasons.append("no_selection_fold_configured")
        if direction == 0:
            selection_reasons.append("no_selection_direction")
        if int(selection_stats["finite_count"]) != len(expected_selection):
            selection_reasons.append("missing_selection_fold")
        if not np.isfinite(selection_abs_strength) or selection_abs_strength < selection_min_abs_corr:
            selection_reasons.append("selection_abs_corr_below_min")
        selection_consistency = float(selection_stats["sign_consistency"])
        if not np.isfinite(selection_consistency) or selection_consistency < selection_min_sign_consistency:
            selection_reasons.append("selection_sign_consistency_below_min")
        selection_all_sign_match = bool(
            direction != 0
            and len(selection_finite) == len(expected_selection)
            and len(selection_finite) > 0
            and np.all(np.abs(selection_finite) > 1e-12)
            and np.all(np.sign(selection_finite) == direction)
        )
        selection_supported = not selection_reasons

        confirmation_finite = confirmation_values[np.isfinite(confirmation_values)]
        confirmation_abs = float(confirmation_stats["abs_mean_of_signed_mean"])
        confirmation_mean_abs = float(confirmation_stats["mean_abs"])
        confirmation_min_fold_abs = float(confirmation_stats["min_abs"])
        confirmation_consistency = float(confirmation_stats["sign_consistency"])
        confirmation_retention = (
            confirmation_abs / selection_abs_strength
            if np.isfinite(confirmation_abs)
            and np.isfinite(selection_abs_strength)
            and selection_abs_strength > 1e-12
            else np.nan
        )
        confirmation_all_sign_match = bool(
            direction != 0
            and len(confirmation_finite) == len(expected_confirmation)
            and len(confirmation_finite) > 0
            and np.all(np.abs(confirmation_finite) > 1e-12)
            and np.all(np.sign(confirmation_finite) == direction)
        )
        confirmation_reasons: list[str] = []
        if not expected_confirmation:
            confirmation_reasons.append("no_confirmation_fold_configured")
        if int(confirmation_stats["finite_count"]) != len(expected_confirmation):
            confirmation_reasons.append("missing_confirmation_fold")
        if not confirmation_all_sign_match:
            confirmation_reasons.append("confirmation_fold_direction_mismatch")
        if not np.isfinite(confirmation_consistency) or confirmation_consistency < confirmation_min_sign_consistency:
            confirmation_reasons.append("confirmation_sign_consistency_below_min")
        if not np.isfinite(confirmation_abs) or confirmation_abs < confirmation_min_abs_corr:
            confirmation_reasons.append("confirmation_mean_abs_corr_below_min")
        if not np.isfinite(confirmation_min_fold_abs) or confirmation_min_fold_abs < confirmation_min_abs_corr:
            confirmation_reasons.append("confirmation_one_or_more_folds_below_min")
        if not np.isfinite(confirmation_retention) or confirmation_retention < confirmation_min_retention:
            confirmation_reasons.append("confirmation_retention_below_min")
        confirmation_supported = bool(selection_supported and not confirmation_reasons)

        recent_finite = recent_values[np.isfinite(recent_values)]
        recent_abs = float(recent_stats["abs_mean_of_signed_mean"])
        recent_mean_abs = float(recent_stats["mean_abs"])
        recent_min_fold_abs = float(recent_stats["min_abs"])
        recent_consistency = float(recent_stats["sign_consistency"])
        recent_retention = (
            recent_abs / selection_abs_strength
            if np.isfinite(recent_abs)
            and np.isfinite(selection_abs_strength)
            and selection_abs_strength > 1e-12
            else np.nan
        )
        recent_all_sign_match = bool(
            direction != 0
            and len(recent_finite) == len(expected_recent)
            and len(recent_finite) > 0
            and np.all(np.abs(recent_finite) > 1e-12)
            and np.all(np.sign(recent_finite) == direction)
        )
        recent_reasons: list[str] = []
        if not expected_recent:
            recent_reasons.append("no_recent_fold_configured")
        if int(recent_stats["finite_count"]) != len(expected_recent):
            recent_reasons.append("missing_recent_fold")
        if not recent_all_sign_match:
            recent_reasons.append("recent_fold_direction_mismatch")
        if not np.isfinite(recent_consistency) or recent_consistency < recent_min_sign_consistency:
            recent_reasons.append("recent_sign_consistency_below_min")
        if not np.isfinite(recent_abs) or recent_abs < recent_min_abs_corr:
            recent_reasons.append("recent_mean_abs_corr_below_min")
        if not np.isfinite(recent_min_fold_abs) or recent_min_fold_abs < recent_min_abs_corr:
            recent_reasons.append("recent_one_or_more_folds_below_min")
        if not np.isfinite(recent_retention) or recent_retention < recent_min_retention:
            recent_reasons.append("recent_retention_below_min")
        recent_supported = bool(selection_supported and not recent_reasons)

        strict_stable = bool(selection_supported and confirmation_supported and recent_supported)
        if strict_stable:
            tier = "STRICT_STABLE"
        elif selection_supported and confirmation_supported:
            tier = "CONFIRMED_NOT_RECENT"
        elif selection_supported:
            tier = "SELECTION_ONLY"
        else:
            tier = "UNSTABLE_OR_WEAK"

        records.append(
            {
                "feature": feature,
                "selection_direction": direction,
                "selection_direction_source": direction_source,
                "selection_support_metric": metric_column,
                "selection_abs_strength": selection_abs_strength,
                "selection_min_fold_abs_corr": float(selection_stats["min_abs"]),
                "selection_strict_sign_consistency": selection_consistency,
                "selection_all_fold_sign_match": selection_all_sign_match,
                "selection_supported": bool(selection_supported),
                "selection_support_reason": ";".join(selection_reasons),
                "confirmation_all_fold_sign_match": confirmation_all_sign_match,
                "confirmation_strict_sign_consistency": confirmation_consistency,
                "confirmation_abs_strength": confirmation_abs,
                "confirmation_mean_fold_abs_corr": confirmation_mean_abs,
                "confirmation_min_fold_abs_corr": confirmation_min_fold_abs,
                "confirmation_retention_ratio": confirmation_retention,
                "confirmation_supported": confirmation_supported,
                "confirmation_support_reason": ";".join(confirmation_reasons),
                "recent_all_fold_sign_match": recent_all_sign_match,
                "recent_strict_sign_consistency": recent_consistency,
                "recent_abs_strength": recent_abs,
                "recent_mean_fold_abs_corr": recent_mean_abs,
                "recent_min_fold_abs_corr": recent_min_fold_abs,
                "recent_retention_ratio": recent_retention,
                "recent_supported": recent_supported,
                "recent_support_reason": ";".join(recent_reasons),
                "strict_stable_supported": strict_stable,
                "support_tier": tier,
            }
        )
    return pd.DataFrame.from_records(records)


def deterministic_absolute_rank(frame: pd.DataFrame, value_column: str) -> pd.Series:
    valid = frame[["feature", value_column]].dropna().copy()
    valid["_abs_value"] = valid[value_column].abs()
    valid = valid.sort_values(["_abs_value", "feature"], ascending=[False, True], kind="mergesort")
    rank_map = pd.Series(np.arange(1, len(valid) + 1, dtype=np.int32), index=valid["feature"])
    return frame["feature"].map(rank_map).astype("Int64")


def build_feature_profiles(
    summary: pd.DataFrame,
    directional_map: pd.DataFrame,
    selection_top_n: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Create explicit correlation-map feature sets without claiming model optimality."""

    merged = summary.merge(
        directional_map[
            [
                "feature",
                "target_relation_class",
                "directional_selection_score",
                "directional_rank",
            ]
        ],
        on="feature",
        how="left",
        validate="one_to_one",
    )
    selection_top_n = max(1, min(int(selection_top_n), len(merged)))
    membership = pd.DataFrame({"feature": merged["feature"]})
    membership["P0_ALL_VALID"] = True
    top_features = set(merged.nsmallest(selection_top_n, "surge_priority_rank")["feature"])
    membership["P1_SELECTION_TOP"] = membership["feature"].isin(top_features)
    strict_features = set(merged.loc[merged["strict_stable_supported"], "feature"])
    membership["P2_STRICT_STABLE"] = membership["feature"].isin(strict_features)
    strict_representatives = set(
        merged.loc[
            merged["strict_stable_supported"] & merged["is_primary_representative"].fillna(False),
            "feature",
        ]
    )
    membership["P3_STRICT_CLUSTER_REP"] = membership["feature"].isin(strict_representatives)
    directional_features = set(
        merged.loc[
            merged["strict_stable_supported"]
            & merged["target_relation_class"].isin(["SURGE_SPECIFIC", "OPPOSITE_DIRECTION"]),
            "feature",
        ]
    )
    membership["P4_SURGE_DIRECTIONAL"] = membership["feature"].isin(directional_features)
    common_move_features = set(
        merged.loc[
            merged["strict_stable_supported"]
            & merged["target_relation_class"].eq("COMMON_LARGE_MOVE"),
            "feature",
        ]
    )
    membership["P5_COMMON_LARGE_MOVE"] = membership["feature"].isin(common_move_features)
    balanced = strict_representatives | directional_features
    membership["P6_BALANCED_MAP"] = membership["feature"].isin(balanced)

    profile_order = [column for column in membership.columns if column != "feature"]
    profiles: dict[str, Any] = {
        "schema": "crashwatch_surge_correlation_profiles_v2",
        "warning": "These are correlation-map candidates, not model-frozen feature sets.",
        "profiles": {},
    }
    ranking = merged.set_index("feature")["surge_priority_rank"].to_dict()
    directional_ranking = merged.set_index("feature")["directional_rank"].to_dict()
    for profile in profile_order:
        features = membership.loc[membership[profile], "feature"].tolist()
        if profile == "P4_SURGE_DIRECTIONAL":
            features.sort(key=lambda feature: (directional_ranking.get(feature, 10**9), feature))
        else:
            features.sort(key=lambda feature: (ranking.get(feature, 10**9), feature))
        profiles["profiles"][profile] = {
            "feature_count": len(features),
            "features": features,
        }
    return membership, profiles


def build_feature_summary(
    quality: pd.DataFrame,
    global_target: pd.DataFrame,
    groupwise_target: pd.DataFrame,
    fold_summary: pd.DataFrame,
    mutual_information: pd.DataFrame,
    structure_stats: pd.DataFrame,
    primary_clusters: pd.DataFrame,
    fold_correlations: pd.DataFrame,
    roles: dict[str, list[int]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    summary = quality.merge(global_target, on="feature", how="left", validate="one_to_one")
    summary = summary.merge(groupwise_target, on="feature", how="left", validate="one_to_one")
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
        "representative_selection_folds",
        "representative_vote_count",
        "representative_vote_ratio",
        "representative_selection_score_mean",
        "representative_selection_score_min",
        "representative_selection_score_std",
        "legacy_crash_representative",
        "surge_representative_changed",
    ]
    cluster_info = primary_clusters[[column for column in cluster_columns if column in primary_clusters.columns]].rename(
        columns={
            "cluster_id": "primary_cluster_id",
            "cluster_size": "primary_cluster_size",
            "representative": "primary_representative",
            "is_representative": "is_primary_representative",
        }
    )
    summary = summary.merge(cluster_info, on="feature", how="left", validate="one_to_one")

    signal = summary["selection_validation_combined_abs_mean"].fillna(0.0)
    signal_min = summary["selection_validation_combined_abs_min"].fillna(0.0)
    stability_columns = [
        column
        for column in (
            "selection_validation_pearson_sign_consistency",
            "selection_validation_spearman_sign_consistency",
            "selection_validation_within_date_pearson_sign_consistency",
            "selection_validation_within_ticker_pearson_sign_consistency",
        )
        if column in summary.columns
    ]
    stability = summary[stability_columns].mean(axis=1, skipna=True).fillna(0.0)
    cross_sectional = summary.get(
        "selection_validation_within_date_pearson_abs_mean",
        pd.Series(0.0, index=summary.index),
    ).fillna(0.0)
    mi_rank = percentile_rank(summary["selection_normalized_mi_mean"].fillna(0.0))
    signal_rank = percentile_rank(signal)
    signal_min_rank = percentile_rank(signal_min)
    cross_rank = percentile_rank(cross_sectional)
    summary["selection_signal_rank"] = signal_rank
    summary["selection_min_signal_rank"] = signal_min_rank
    summary["selection_stability_score"] = stability.clip(0.0, 1.0)
    summary["selection_cross_sectional_rank"] = cross_rank
    summary["selection_mi_rank"] = mi_rank
    summary["selection_priority_score"] = (
        0.45 * signal_rank
        + 0.15 * signal_min_rank
        + 0.15 * stability.clip(0.0, 1.0)
        + 0.10 * cross_rank
        + 0.10 * mi_rank
        + 0.05 * summary["data_quality_score"].fillna(0.0)
    ).clip(0.0, 1.0)

    support = evaluate_support_evidence(
        summary,
        fold_correlations,
        roles,
        selection_min_abs_corr=args.support_selection_min_abs_corr,
        selection_min_sign_consistency=args.support_selection_min_sign_consistency,
        confirmation_min_abs_corr=args.support_confirmation_min_abs_corr,
        confirmation_min_sign_consistency=args.support_confirmation_min_sign_consistency,
        confirmation_min_retention=args.support_confirmation_min_retention,
        recent_min_abs_corr=args.support_recent_min_abs_corr,
        recent_min_sign_consistency=args.support_recent_min_sign_consistency,
        recent_min_retention=args.support_recent_min_retention,
    )
    summary = summary.merge(support, on="feature", how="left", validate="one_to_one")
    summary["confirmation_supported_loose"] = summary["confirmation_sign_matches_selection"].fillna(False)
    summary["recent_supported_loose"] = summary["recent_sign_matches_selection"].fillna(False)
    summary = summary.sort_values(
        ["selection_priority_score", "selection_validation_combined_abs_mean", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary.insert(0, "surge_priority_rank", np.arange(1, len(summary) + 1, dtype=np.int32))
    strict_rank = pd.Series(pd.NA, index=summary.index, dtype="Int64")
    strict_indices = summary.index[summary["strict_stable_supported"].fillna(False)]
    strict_rank.loc[strict_indices] = np.arange(1, len(strict_indices) + 1, dtype=np.int32)
    summary.insert(1, "strict_stable_rank", strict_rank)
    return summary





def build_feature_priority(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "surge_priority_rank",
        "strict_stable_rank",
        "feature",
        "group",
        "selection_priority_score",
        "selection_signal_rank",
        "selection_stability_score",
        "selection_validation_pearson_mean",
        "selection_validation_spearman_mean",
        "selection_validation_within_date_pearson_mean",
        "selection_validation_within_ticker_pearson_mean",
        "selection_validation_combined_abs_mean",
        "selection_validation_combined_abs_min",
        "selection_validation_pearson_sign_consistency",
        "selection_validation_spearman_sign_consistency",
        "selection_direction",
        "selection_direction_source",
        "selection_supported",
        "confirmation_validation_pearson_mean",
        "confirmation_strict_sign_consistency",
        "confirmation_retention_ratio",
        "confirmation_supported",
        "recent_audit_validation_pearson_mean",
        "recent_strict_sign_consistency",
        "recent_retention_ratio",
        "recent_supported",
        "strict_stable_supported",
        "support_tier",
        "target_relation_class",
        "directional_selection_score",
        "directional_rank",
        "surge_specific_strength",
        "common_large_move_strength",
        "crash_selection_corr",
        "selection_normalized_mi_mean",
        "target_pearson",
        "target_spearman",
        "target_within_date_pearson",
        "target_within_ticker_pearson",
        "date_corr_mean",
        "date_corr_sign_consistency",
        "ticker_corr_mean",
        "ticker_corr_sign_consistency",
        "max_abs_correlation",
        "max_corr_peer",
        "primary_cluster_id",
        "primary_cluster_size",
        "primary_representative",
        "is_primary_representative",
        "representative_vote_count",
        "representative_vote_ratio",
        "legacy_crash_representative",
        "surge_representative_changed",
    ]
    return summary[[column for column in columns if column in summary.columns]].copy()





def build_surge_crash_feature_comparison(
    summary: pd.DataFrame,
    legacy_summary_path: Path,
    legacy_fold_path: Path,
    selection_fold_ids: Sequence[int],
    directional_min_abs_corr: float,
    directional_crash_weak_abs_corr: float,
) -> pd.DataFrame:
    """Separate common large-move signals from surge-directional signals.

    Directional scores use selection-fold associations only. Global correlations
    are retained solely as descriptive columns.
    """

    surge_columns = [
        "surge_priority_rank",
        "strict_stable_rank",
        "feature",
        "target_pearson",
        "selection_validation_pearson_mean",
        "selection_validation_pearson_sign_consistency",
        "selection_priority_score",
        "strict_stable_supported",
        "confirmation_supported",
        "recent_supported",
        "is_primary_representative",
    ]
    surge = summary[surge_columns].copy().rename(
        columns={
            "target_pearson": "surge_global_corr",
            "selection_validation_pearson_mean": "surge_selection_corr",
            "selection_validation_pearson_sign_consistency": "surge_selection_sign_consistency",
            "is_primary_representative": "surge_cluster_rep",
        }
    )
    surge["surge_global_rank"] = deterministic_absolute_rank(surge.rename(columns={"surge_global_corr": "value"}), "value")
    surge["surge_selection_rank"] = deterministic_absolute_rank(
        surge.rename(columns={"surge_selection_corr": "value"}), "value"
    )

    if legacy_summary_path.exists():
        legacy = pd.read_csv(legacy_summary_path)
        required = {"feature", "target_pearson"}
        if not required.issubset(legacy.columns):
            raise ValueError(f"레거시 급락 상관 요약 필수 열 누락: {sorted(required - set(legacy.columns))}")
        crash_global_columns = ["feature", "target_pearson"]
        if "is_primary_representative" in legacy.columns:
            crash_global_columns.append("is_primary_representative")
        crash = legacy[crash_global_columns].drop_duplicates("feature").rename(
            columns={
                "target_pearson": "crash_global_corr",
                "is_primary_representative": "crash_cluster_rep",
            }
        )
        if "crash_cluster_rep" not in crash.columns:
            crash["crash_cluster_rep"] = False
    else:
        crash = pd.DataFrame({"feature": surge["feature"], "crash_global_corr": np.nan, "crash_cluster_rep": False})

    if legacy_fold_path.exists():
        legacy_fold = pd.read_csv(legacy_fold_path)
        required_fold = {"outer_fold", "feature", "target_pearson"}
        if not required_fold.issubset(legacy_fold.columns):
            raise ValueError(f"레거시 급락 fold 상관 필수 열 누락: {sorted(required_fold - set(legacy_fold.columns))}")
        selection = legacy_fold[legacy_fold["outer_fold"].isin(selection_fold_ids)].copy()
        records: list[dict[str, Any]] = []
        for feature, part in selection.groupby("feature", sort=False):
            stats = aggregate_metric(part["target_pearson"].to_numpy(dtype=np.float64))
            records.append(
                {
                    "feature": feature,
                    "crash_selection_corr": stats["mean"],
                    "crash_selection_sign_consistency": stats["sign_consistency"],
                    "crash_selection_folds": stats["folds"],
                }
            )
        crash_selection = pd.DataFrame.from_records(records)
        crash = crash.merge(crash_selection, on="feature", how="left", validate="one_to_one")
    else:
        crash["crash_selection_corr"] = np.nan
        crash["crash_selection_sign_consistency"] = np.nan
        crash["crash_selection_folds"] = 0

    comparison = surge.merge(crash, on="feature", how="left", validate="one_to_one")
    comparison["crash_global_rank"] = deterministic_absolute_rank(
        comparison.rename(columns={"crash_global_corr": "value"}), "value"
    )
    comparison["crash_selection_rank"] = deterministic_absolute_rank(
        comparison.rename(columns={"crash_selection_corr": "value"}), "value"
    )
    surge_corr = comparison["surge_selection_corr"].to_numpy(dtype=np.float64)
    crash_corr = comparison["crash_selection_corr"].to_numpy(dtype=np.float64)
    both = np.isfinite(surge_corr) & np.isfinite(crash_corr)
    same_direction = np.zeros(len(comparison), dtype=bool)
    same_direction[both] = np.sign(surge_corr[both]) == np.sign(crash_corr[both])
    comparison["same_direction_selection"] = pd.Series(pd.NA, index=comparison.index, dtype="boolean")
    comparison.loc[both, "same_direction_selection"] = same_direction[both]
    comparison["selection_abs_difference"] = np.abs(surge_corr - crash_corr)

    crash_overlap = np.zeros(len(comparison), dtype=np.float64)
    crash_overlap[both] = np.maximum(0.0, np.sign(surge_corr[both]) * crash_corr[both])
    surge_specific = np.full(len(comparison), np.nan, dtype=np.float64)
    surge_specific[both] = np.maximum(0.0, np.abs(surge_corr[both]) - crash_overlap[both])
    opposite = both & ~same_direction
    surge_specific[opposite] = np.abs(surge_corr[opposite])
    common_strength = np.zeros(len(comparison), dtype=np.float64)
    common_strength[both & same_direction] = np.minimum(
        np.abs(surge_corr[both & same_direction]),
        np.abs(crash_corr[both & same_direction]),
    )
    comparison["surge_specific_strength"] = surge_specific
    comparison["directional_contrast"] = np.abs(surge_corr - crash_corr)
    comparison["common_large_move_strength"] = common_strength
    comparison["surge_directionality_ratio"] = np.divide(
        surge_specific,
        np.abs(surge_corr),
        out=np.full(len(comparison), np.nan, dtype=np.float64),
        where=np.abs(surge_corr) > 1e-12,
    )

    classes: list[str] = []
    for surge_value, crash_value in zip(surge_corr, crash_corr):
        if not np.isfinite(surge_value) or abs(surge_value) < directional_min_abs_corr:
            classes.append("WEAK_OR_UNCLEAR")
        elif not np.isfinite(crash_value):
            classes.append("CRASH_REFERENCE_UNAVAILABLE")
        elif abs(crash_value) < directional_crash_weak_abs_corr:
            classes.append("SURGE_SPECIFIC")
        elif np.sign(surge_value) != np.sign(crash_value):
            classes.append("OPPOSITE_DIRECTION")
        elif min(abs(surge_value), abs(crash_value)) >= directional_min_abs_corr:
            classes.append("COMMON_LARGE_MOVE")
        else:
            classes.append("MIXED_OVERLAP")
    comparison["target_relation_class"] = classes

    specific_rank = percentile_rank(comparison["surge_specific_strength"].fillna(0.0))
    contrast_rank = percentile_rank(comparison["directional_contrast"].fillna(0.0))
    stability = comparison["surge_selection_sign_consistency"].fillna(0.0).clip(0.0, 1.0)
    comparison["directional_selection_score"] = (
        0.45 * specific_rank
        + 0.25 * contrast_rank
        + 0.20 * comparison["selection_priority_score"].fillna(0.0)
        + 0.10 * stability
    ).clip(0.0, 1.0)
    ordered = comparison.sort_values(
        ["directional_selection_score", "feature"],
        ascending=[False, True],
        kind="mergesort",
    ).index
    directional_rank = pd.Series(pd.NA, index=comparison.index, dtype="Int64")
    directional_rank.loc[ordered] = np.arange(1, len(comparison) + 1, dtype=np.int32)
    comparison["directional_rank"] = directional_rank
    comparison["global_same_direction"] = pd.Series(pd.NA, index=comparison.index, dtype="boolean")
    global_both = comparison["surge_global_corr"].notna() & comparison["crash_global_corr"].notna()
    comparison.loc[global_both, "global_same_direction"] = (
        np.sign(comparison.loc[global_both, "surge_global_corr"])
        == np.sign(comparison.loc[global_both, "crash_global_corr"])
    )
    comparison["global_abs_difference"] = (
        comparison["surge_global_corr"] - comparison["crash_global_corr"]
    ).abs()
    comparison["surge_rank"] = comparison["surge_global_rank"]
    comparison["crash_rank"] = comparison["crash_global_rank"]
    comparison["rank_change"] = comparison["crash_global_rank"] - comparison["surge_global_rank"]
    comparison["surge_corr"] = comparison["surge_global_corr"]
    comparison["crash_corr"] = comparison["crash_global_corr"]
    comparison["same_direction"] = comparison["global_same_direction"]
    comparison["abs_difference"] = comparison["global_abs_difference"]
    comparison["surge_cluster_rep"] = comparison["surge_cluster_rep"].fillna(False).astype(bool)
    comparison["crash_cluster_rep"] = comparison["crash_cluster_rep"].fillna(False).astype(bool)
    return comparison.sort_values(
        ["surge_priority_rank" if "surge_priority_rank" in comparison.columns else "selection_priority_score", "feature"],
        ascending=[True if "surge_priority_rank" in comparison.columns else False, True],
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
    directional_map: pd.DataFrame,
    profile_payload: dict[str, Any],
) -> None:
    changed = primary_clusters[
        primary_clusters["is_representative"].astype(bool)
        & primary_clusters["surge_representative_changed"].astype(bool)
    ]
    positive = summary[summary["selection_validation_pearson_mean"].fillna(0.0) > 0]
    negative = summary[summary["selection_validation_pearson_mean"].fillna(0.0) < 0]
    strict_count = int(summary["strict_stable_supported"].fillna(False).sum())
    confirmed_count = int(summary["confirmation_supported"].fillna(False).sum())
    recent_count = int(summary["recent_supported"].fillna(False).sum())
    class_counts = directional_map["target_relation_class"].value_counts().to_dict()
    profile_counts = {
        name: int(payload.get("feature_count", 0))
        for name, payload in profile_payload.get("profiles", {}).items()
    }
    lines = [
        "# CrashWatch Surge 3D/+5% 완성형 상관관계 지도 실행 요약",
        "",
        "## 데이터와 감사",
        "",
        f"- 유효 target 행: {audit['target_valid_rows']:,}",
        f"- 양성: {audit['target_positives']:,}",
        f"- 양성률: {audit['target_positive_rate']:.6%}",
        f"- 유효 피처 수: {audit['valid_feature_count']:,}",
        f"- 피처-피처 구조 출처: `{bundle.source}`",
        "- 미래값 보간: 사용하지 않음",
        "",
        "## selection / confirmation / recent 판정",
        "",
        f"- 엄격 confirmation 통과: {confirmed_count:,}",
        f"- 엄격 recent 통과: {recent_count:,}",
        f"- selection+confirmation+recent 모두 통과: {strict_count:,}",
        "- confirmation과 recent 결과는 selection_priority_score에 포함하지 않았다.",
        "- 각 지지 판정은 방향, 최소 절대상관, fold 완전성, 부호 일치율, selection 대비 유지비율을 모두 검사한다.",
        "",
        "## selection validation 기준 양의 상관 상위",
        "",
        *format_top_rows(positive, "selection_validation_pearson_mean", 15, ascending=False),
        "",
        "## selection validation 기준 음의 상관 상위",
        "",
        *format_top_rows(negative, "selection_validation_pearson_mean", 15, ascending=True),
        "",
        "## 방향성 분해",
        "",
        f"- SURGE_SPECIFIC: {int(class_counts.get('SURGE_SPECIFIC', 0)):,}",
        f"- OPPOSITE_DIRECTION: {int(class_counts.get('OPPOSITE_DIRECTION', 0)):,}",
        f"- COMMON_LARGE_MOVE: {int(class_counts.get('COMMON_LARGE_MOVE', 0)):,}",
        f"- MIXED_OVERLAP: {int(class_counts.get('MIXED_OVERLAP', 0)):,}",
        f"- WEAK_OR_UNCLEAR: {int(class_counts.get('WEAK_OR_UNCLEAR', 0)):,}",
        f"- CRASH_REFERENCE_UNAVAILABLE: {int(class_counts.get('CRASH_REFERENCE_UNAVAILABLE', 0)):,}",
        "- COMMON_LARGE_MOVE는 급등 방향 전용 신호가 아니라 급등·급락 모두와 같은 방향으로 연관된 상태 신호다.",
        "- CRASH_REFERENCE_UNAVAILABLE은 급락 참조가 없어 방향 특화 여부를 판정하지 않았다는 뜻이며 SURGE_SPECIFIC으로 간주하지 않는다.",
        "",
        "## 중복 구조와 대표 피처",
        "",
        f"- primary threshold: {float(primary_clusters['threshold'].iloc[0]) if len(primary_clusters) else np.nan}",
        f"- primary cluster 수: {int(primary_clusters['cluster_id'].nunique()) if len(primary_clusters) else 0}",
        f"- primary threshold 이상 edge 수: {len(high_pairs):,}",
        f"- 급락용 대표와 달라진 급등용 대표 cluster 수: {len(changed):,}",
        "- primary 대표는 fold 0 하나가 아니라 모든 selection fold의 train-only 점수와 투표를 합친 consensus로 선정했다.",
        "",
        "## 생성된 피처 후보 프로필",
        "",
        *[f"- {name}: {count:,}개" for name, count in profile_counts.items()],
        "",
        "상관관계 지도는 인과관계나 최종 모델 효용을 보장하지 않는다. 최종 KEEP/DROP은 동일 fold의 LightGBM/XGBoost 이탈 실험으로 검증한다.",
        "",
    ]
    atomic_text("\n".join(lines), output_dir / "SURGE_CORRELATION_MAP_GUIDE_KO.md")




def resolve_default_paths(args: argparse.Namespace) -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    package_root = Path(args.package_root).resolve() if args.package_root else script_path.parents[1]
    args.package_root = package_root
    args.input = Path(args.input).resolve() if args.input else package_root / "data" / "training_dataset_finance11h.parquet"
    args.target = Path(args.target).resolve() if args.target else package_root / "data" / "surge_target_3d5.parquet"
    args.output = (
        Path(args.output).resolve()
        if args.output
        else package_root / "outputs" / "surge_correlation_map_complete"
    )
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
        "schema": SCHEMA_VERSION,
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
        "groupwise_min_date_rows": args.groupwise_min_date_rows,
        "groupwise_min_ticker_rows": args.groupwise_min_ticker_rows,
        "skip_groupwise_distribution": args.skip_groupwise_distribution,
        "support_criteria": {
            "selection_min_abs_corr": args.support_selection_min_abs_corr,
            "selection_min_sign_consistency": args.support_selection_min_sign_consistency,
            "confirmation_min_abs_corr": args.support_confirmation_min_abs_corr,
            "confirmation_min_sign_consistency": args.support_confirmation_min_sign_consistency,
            "confirmation_min_retention": args.support_confirmation_min_retention,
            "recent_min_abs_corr": args.support_recent_min_abs_corr,
            "recent_min_sign_consistency": args.support_recent_min_sign_consistency,
            "recent_min_retention": args.support_recent_min_retention,
        },
        "directional_min_abs_corr": args.directional_min_abs_corr,
        "directional_crash_weak_abs_corr": args.directional_crash_weak_abs_corr,
        "selection_profile_top_n": args.selection_profile_top_n,
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
        "surge_feature_groupwise_correlation_summary.csv",
        "surge_feature_correlation_summary.csv",
        "surge_support_audit.csv",
        "surge_directional_map.csv",
        "surge_feature_profiles.json",
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
    if not 0.0 < args.primary_threshold <= 1.0:
        raise ValueError("primary-threshold는 0보다 크고 1 이하여야 합니다")
    if any(not 0.0 < threshold <= 1.0 for threshold in args.thresholds):
        raise ValueError("모든 cluster threshold는 0보다 크고 1 이하여야 합니다")

    roles = {
        "selection": parse_int_list(args.selection_folds),
        "confirmation": parse_int_list(args.confirmation_folds),
        "recent_audit": parse_int_list(args.recent_folds),
    }
    if not any(roles.values()):
        roles = {key: list(value) for key, value in DEFAULT_FOLD_ROLES.items()}
    role_members = [fold_id for values in roles.values() for fold_id in values]
    if len(role_members) != len(set(role_members)):
        raise ValueError("하나의 fold가 둘 이상의 역할에 중복 배정되었습니다")
    if not roles["selection"]:
        raise ValueError("selection fold가 하나 이상 필요합니다")

    unit_interval_arguments = {
        "support_selection_min_sign_consistency": args.support_selection_min_sign_consistency,
        "support_confirmation_min_sign_consistency": args.support_confirmation_min_sign_consistency,
        "support_confirmation_min_retention": args.support_confirmation_min_retention,
        "support_recent_min_sign_consistency": args.support_recent_min_sign_consistency,
        "support_recent_min_retention": args.support_recent_min_retention,
    }
    for name, value in unit_interval_arguments.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}은 0~1 범위여야 합니다: {value}")
    nonnegative_arguments = {
        "support_selection_min_abs_corr": args.support_selection_min_abs_corr,
        "support_confirmation_min_abs_corr": args.support_confirmation_min_abs_corr,
        "support_recent_min_abs_corr": args.support_recent_min_abs_corr,
        "directional_min_abs_corr": args.directional_min_abs_corr,
        "directional_crash_weak_abs_corr": args.directional_crash_weak_abs_corr,
    }
    for name, value in nonnegative_arguments.items():
        if value < 0:
            raise ValueError(f"{name}은 음수일 수 없습니다: {value}")
    if args.selection_profile_top_n < 1:
        raise ValueError("selection-profile-top-n은 1 이상이어야 합니다")

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
    requested_features = list(dict.fromkeys(requested_features))
    if not requested_features:
        raise ValueError("사용할 피처를 찾지 못했습니다")

    log("resume/cache 검증을 위해 원본·target·fold·reference 해시를 계산합니다")
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
        ("feature_target_correlation_by_fold.csv", "legacy_target_fold_sha256"),
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
            log("동일 config와 SHA-256 산출물의 완료 manifest를 재사용합니다")
            return prior
        log("기존 manifest 또는 산출물 무결성이 맞지 않아 전체 파이프라인을 다시 계산합니다")

    with RunLock(args.output / ".run.lock"):
        tracker = RunTracker(args.output, config)
        try:
            tracker.stage("data_audit", "running")
            log("원본과 급등 target sidecar를 source_row_id로 검증하고 필요한 피처만 읽습니다")
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
            defined_ids = {definition.fold_id for definition in definitions}
            unknown_role_ids = sorted(set(role_members) - defined_ids)
            if unknown_role_ids:
                raise ValueError(f"역할에 배정됐지만 fold JSON에 없는 ID: {unknown_role_ids}")
            fold_manifest = validate_fold_definitions(
                source["date"], target_valid, target, definitions, roles, args.minimum_purge_days
            )
            atomic_json(fold_manifest, args.output / "walk_forward_folds.json")
            tracker.stage("fold_audit", "complete", folds=len(definitions), roles=roles)

            tracker.stage("target_correlations", "running")
            valid_indices = np.flatnonzero(target_valid)
            valid_features_frame = feature_frame.iloc[valid_indices]
            valid_target = target[valid_indices]
            valid_dates = date_values[valid_indices]
            valid_tickers = tickers[valid_indices]
            log(f"전체 Pearson/Spearman 및 pooled within 상관: {len(valid_indices):,}행")
            global_target = compute_target_correlations(
                valid_features_frame,
                valid_target,
                valid_dates,
                valid_tickers,
                features,
                args.target_min_periods,
                args.chunk_size,
                include_grouped=True,
            )
            if args.skip_groupwise_distribution:
                groupwise_target = pd.DataFrame({"feature": features})
            else:
                log("날짜별 literal correlation 분포를 계산합니다")
                date_groupwise = compute_groupwise_target_correlation_distribution(
                    valid_features_frame,
                    valid_target,
                    valid_dates,
                    features,
                    prefix="date",
                    min_group_rows=args.groupwise_min_date_rows,
                    chunk_size=args.chunk_size,
                )
                log("종목별 literal correlation 분포를 계산합니다")
                ticker_groupwise = compute_groupwise_target_correlation_distribution(
                    valid_features_frame,
                    valid_target,
                    valid_tickers,
                    features,
                    prefix="ticker",
                    min_group_rows=args.groupwise_min_ticker_rows,
                    chunk_size=args.chunk_size,
                )
                groupwise_target = date_groupwise.merge(
                    ticker_groupwise, on="feature", how="outer", validate="one_to_one"
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
            atomic_csv(groupwise_target, args.output / "surge_feature_groupwise_correlation_summary.csv")
            tracker.stage(
                "target_correlations",
                "complete",
                global_rows=len(valid_indices),
                fold_rows=len(fold_correlations),
                groupwise_skipped=args.skip_groupwise_distribution,
            )

            tracker.stage("mutual_information", "running")
            mi_by_fold, mutual_information = compute_mutual_information_by_selection_fold(
                feature_frame,
                target,
                target_valid,
                date_values,
                features,
                definitions,
                roles["selection"],
                sample_rows=args.mi_sample_rows,
                bins=args.mi_bins,
                min_rows=args.target_min_periods,
                seed=args.seed,
                skip=args.skip_mi,
            )
            atomic_csv(mi_by_fold, args.output / "surge_feature_mi_by_selection_fold.csv")
            tracker.stage(
                "mutual_information",
                "complete",
                selection_folds=roles["selection"],
                rows=len(mi_by_fold),
                skipped=args.skip_mi,
            )

            tracker.stage("feature_structure", "running")
            compatible, reuse_details = legacy_reuse_compatibility(
                args.input,
                args.folds,
                features,
                args.package_root,
                args.legacy_correlation_dir,
            )
            if args.reuse_legacy == "always" and not compatible:
                raise ValueError(f"legacy 피처 구조 강제 재사용 조건 불충족: {reuse_details}")
            should_reuse = args.reuse_legacy == "always" or (args.reuse_legacy == "auto" and compatible)
            if should_reuse:
                log("hash와 fold가 일치하여 target 독립 피처-피처 구조만 재사용합니다")
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

            tracker.stage("clusters_and_ranking", "running")
            high_pairs = build_high_correlation_pairs(bundle, features, args.primary_threshold)
            (
                all_clusters,
                components_by_threshold,
                consensus_candidates,
                selection_representative_votes,
            ) = build_cluster_assignments(
                bundle.cluster_basis_abs,
                features,
                args.thresholds,
                quality,
                fold_correlations,
                selection_fold_ids=roles["selection"],
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
            primary_consensus = consensus_candidates[
                consensus_candidates["threshold"].eq(args.primary_threshold)
            ].copy()
            primary_selection_votes = selection_representative_votes[
                selection_representative_votes["threshold"].eq(args.primary_threshold)
            ].copy()
            structure_stats = feature_structure_stats(bundle, features, args.primary_threshold)
            summary_core = build_feature_summary(
                quality,
                global_target,
                groupwise_target,
                fold_summary,
                mutual_information,
                structure_stats,
                primary_clusters,
                fold_correlations,
                roles,
                args,
            )
            directional_map = build_surge_crash_feature_comparison(
                summary_core,
                args.legacy_correlation_dir / "feature_correlation_summary.csv",
                args.legacy_correlation_dir / "feature_target_correlation_by_fold.csv",
                roles["selection"],
                directional_min_abs_corr=args.directional_min_abs_corr,
                directional_crash_weak_abs_corr=args.directional_crash_weak_abs_corr,
            )
            profile_membership, profile_payload = build_feature_profiles(
                summary_core,
                directional_map,
                args.selection_profile_top_n,
            )
            directional_columns = [
                "feature",
                "surge_selection_corr",
                "crash_selection_corr",
                "same_direction_selection",
                "surge_specific_strength",
                "directional_contrast",
                "common_large_move_strength",
                "surge_directionality_ratio",
                "target_relation_class",
                "directional_selection_score",
                "directional_rank",
                "crash_global_corr",
                "crash_global_rank",
                "crash_selection_rank",
            ]
            summary = summary_core.merge(
                directional_map[[column for column in directional_columns if column in directional_map.columns]],
                on="feature",
                how="left",
                validate="one_to_one",
            )
            feature_priority = build_feature_priority(summary)
            support_columns = [
                "feature",
                "surge_priority_rank",
                "strict_stable_rank",
                "selection_direction",
                "selection_direction_source",
                "selection_support_metric",
                "selection_abs_strength",
                "selection_min_fold_abs_corr",
                "selection_strict_sign_consistency",
                "selection_all_fold_sign_match",
                "selection_supported",
                "selection_support_reason",
                "confirmation_all_fold_sign_match",
                "confirmation_strict_sign_consistency",
                "confirmation_abs_strength",
                "confirmation_mean_fold_abs_corr",
                "confirmation_min_fold_abs_corr",
                "confirmation_retention_ratio",
                "confirmation_supported",
                "confirmation_support_reason",
                "recent_all_fold_sign_match",
                "recent_strict_sign_consistency",
                "recent_abs_strength",
                "recent_mean_fold_abs_corr",
                "recent_min_fold_abs_corr",
                "recent_retention_ratio",
                "recent_supported",
                "recent_support_reason",
                "strict_stable_supported",
                "support_tier",
            ]
            support_audit = summary[[column for column in support_columns if column in summary.columns]].copy()
            leakage_audit = leakage_correlation_audit(global_target, valid_features_frame)
            suspicious = leakage_audit[leakage_audit["leakage_suspect"]]
            if len(suspicious):
                log(f"경고: 상관 기반 누수 의심 피처 {len(suspicious)}개를 기록했습니다")

            atomic_csv(high_pairs, args.output / "high_correlation_pairs.csv")
            atomic_csv(all_clusters, args.output / "all_cluster_assignments.csv")
            atomic_csv(primary_clusters, args.output / "primary_clusters.csv")
            atomic_csv(consensus_candidates, args.output / "surge_cluster_consensus_candidates.csv")
            atomic_csv(primary_consensus, args.output / "surge_primary_cluster_consensus.csv")
            atomic_csv(selection_representative_votes, args.output / "surge_selection_representative_votes.csv")
            atomic_csv(primary_selection_votes, args.output / "surge_primary_selection_representative_votes.csv")
            atomic_csv(fold_representatives, args.output / "surge_representatives_by_fold.csv")
            atomic_csv(summary, args.output / "surge_feature_correlation_summary.csv")
            atomic_csv(feature_priority, args.output / "surge_feature_priority.csv")
            atomic_csv(support_audit, args.output / "surge_support_audit.csv")
            atomic_csv(directional_map, args.output / "surge_directional_map.csv")
            atomic_csv(directional_map, args.output / "surge_vs_crash_feature_comparison.csv")
            atomic_csv(profile_membership, args.output / "surge_feature_profile_membership.csv")
            atomic_json(profile_payload, args.output / "surge_feature_profiles.json")
            atomic_csv(leakage_audit, args.output / "leakage_audit.csv")
            representatives_payload = {
                str(int(row.cluster_id)): {
                    "representative": row.representative,
                    "cluster_size": int(row.cluster_size),
                    "selection_folds": row.representative_selection_folds,
                    "vote_count": int(row.representative_vote_count),
                    "vote_ratio": float(row.representative_vote_ratio),
                    "consensus_score": float(row.representative_quality_score),
                }
                for row in primary_clusters[primary_clusters["is_representative"]].itertuples()
            }
            atomic_json(representatives_payload, args.output / "surge_primary_representatives.json")

            strict_count = int(summary["strict_stable_supported"].fillna(False).sum())
            confirmed_count = int(summary["confirmation_supported"].fillna(False).sum())
            recent_count = int(summary["recent_supported"].fillna(False).sum())
            relation_counts = {
                str(key): int(value)
                for key, value in directional_map["target_relation_class"].value_counts().items()
            }
            summary_metrics = {
                "feature_count": len(features),
                "strict_stable_feature_count": strict_count,
                "confirmation_supported_feature_count": confirmed_count,
                "recent_supported_feature_count": recent_count,
                "primary_cluster_count": int(primary_clusters["cluster_id"].nunique()),
                "primary_representative_count": int(primary_clusters["is_representative"].sum()),
                "target_relation_class_counts": relation_counts,
                "profile_feature_counts": {
                    name: int(payload["feature_count"])
                    for name, payload in profile_payload["profiles"].items()
                },
            }
            atomic_json(summary_metrics, args.output / "SURGE_CORRELATION_MAP_SUMMARY.json")
            tracker.stage(
                "clusters_and_ranking",
                "complete",
                primary_clusters=int(primary_clusters["cluster_id"].nunique()),
                strict_stable_features=strict_count,
                confirmation_supported_features=confirmed_count,
                recent_supported_features=recent_count,
                high_correlation_edges=len(high_pairs),
                leakage_suspects=len(suspicious),
            )

            tracker.stage("reporting", "running")
            plot_files = [] if args.skip_plots else maybe_plot_outputs(
                summary, bundle, features, args.output, args.plot_top_n
            )
            write_korean_report(
                args.output,
                data_audit,
                summary,
                primary_clusters,
                high_pairs,
                bundle,
                directional_map,
                profile_payload,
            )
            matrix_hash = hash_arrays(
                features,
                [
                    bundle.pearson,
                    bundle.spearman,
                    bundle.within_date,
                    bundle.within_ticker,
                    bundle.missingness,
                    bundle.combined_abs,
                    bundle.cluster_basis_abs,
                ],
            )
            output_inventory = build_output_inventory(args.output)
            manifest = {
                "status": "complete",
                "schema": SCHEMA_VERSION,
                "run_time": pipeline_started_at,
                "created_at": utc_now(),
                "elapsed_seconds": time.perf_counter() - pipeline_started_perf_counter,
                "python_version": sys.version,
                "platform": platform.platform(),
                "config_hash": config_hash,
                "source_dataset_path": str(args.input),
                "source_dataset_sha256": source_sha,
                "target_path": str(args.target),
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
                "reuse_legacy_mode": args.reuse_legacy,
                "legacy_reused": should_reuse,
                "support_criteria": config["support_criteria"],
                "directional_criteria": {
                    "minimum_surge_abs_corr": args.directional_min_abs_corr,
                    "crash_weak_abs_corr": args.directional_crash_weak_abs_corr,
                },
                "strict_stable_feature_count": strict_count,
                "confirmation_supported_feature_count": confirmed_count,
                "recent_supported_feature_count": recent_count,
                "target_relation_class_counts": relation_counts,
                "profile_feature_counts": summary_metrics["profile_feature_counts"],
                "primary_cluster_count": int(primary_clusters["cluster_id"].nunique()),
                "primary_representative_count": int(primary_clusters["is_representative"].sum()),
                "high_correlation_edge_count": len(high_pairs),
                "matrix_hash": matrix_hash,
                "leakage_suspect_count": len(suspicious),
                "plot_files": plot_files,
                "git_commit": None,
                "outputs": [item["name"] for item in output_inventory] + ["RUN_STATUS.json"],
                "output_inventory": output_inventory,
                "selection_isolation_rule": (
                    "selection_priority_score and cluster consensus use selection folds/train-only evidence; "
                    "confirmation and recent folds only produce validation support flags"
                ),
                "directionality_rule": (
                    "legacy crash fold correlations are used only to label common-large-move versus directional "
                    "surge associations; they do not alter the base surge selection score"
                ),
            }
            atomic_json(manifest, manifest_path)
            tracker.stage("reporting", "complete", plots=plot_files)
            tracker.complete(manifest=str(manifest_path), outputs=len(manifest["outputs"]))
            log(
                f"완료: features={len(features)}, strict={strict_count}, "
                f"clusters={manifest['primary_cluster_count']}, output={args.output}"
            )
            return manifest
        except BaseException as error:
            tracker.fail(error)
            raise




def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CrashWatch 3거래일 내 +5% 급등 target의 완성형 상관관계 지도를 생성합니다. "
            "기본값은 모든 eligible 행을 사용하는 full-load 실행입니다."
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
        help="hash 검증 후 target 독립 피처-피처 구조만 재사용",
    )
    parser.add_argument("--thresholds", default="0.80,0.90,0.92,0.95,0.98")
    parser.add_argument("--primary-threshold", type=float, default=DEFAULT_PRIMARY_THRESHOLD)
    parser.add_argument("--selection-folds", default="0,1,2,3,4")
    parser.add_argument("--confirmation-folds", default="5,6")
    parser.add_argument("--recent-folds", default="7")
    parser.add_argument("--minimum-purge-days", type=int, default=3)

    parser.add_argument(
        "--feature-structure-sample-rows",
        type=int,
        default=0,
        help="0이면 전체 행; 양수이면 완전한 거래일 단위 결정론 표본",
    )
    parser.add_argument("--spearman-sample-rows", type=int, default=0, help="0이면 전체 행")
    parser.add_argument("--cluster-basis-rows", type=int, default=0, help="0이면 fold 0 train 전체")
    parser.add_argument("--fold-correlation-sample-rows", type=int, default=0, help="0이면 각 train 전체")
    parser.add_argument("--target-train-sample-rows", type=int, default=0, help="0이면 각 train 전체")
    parser.add_argument("--mi-sample-rows", type=int, default=0, help="0이면 각 selection train 전체")
    parser.add_argument("--mi-bins", type=int, default=16)
    parser.add_argument("--min-periods", type=int, default=4000)
    parser.add_argument("--target-min-periods", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--groupwise-min-date-rows", type=int, default=8)
    parser.add_argument("--groupwise-min-ticker-rows", type=int, default=200)
    parser.add_argument("--skip-groupwise-distribution", action="store_true")

    parser.add_argument("--support-selection-min-abs-corr", type=float, default=DEFAULT_MIN_SELECTION_ABS_CORR)
    parser.add_argument("--support-selection-min-sign-consistency", type=float, default=DEFAULT_MIN_SELECTION_SIGN_CONSISTENCY)
    parser.add_argument("--support-confirmation-min-abs-corr", type=float, default=DEFAULT_MIN_CONFIRMATION_ABS_CORR)
    parser.add_argument("--support-confirmation-min-sign-consistency", type=float, default=DEFAULT_MIN_CONFIRMATION_SIGN_CONSISTENCY)
    parser.add_argument("--support-confirmation-min-retention", type=float, default=DEFAULT_MIN_CONFIRMATION_RETENTION)
    parser.add_argument("--support-recent-min-abs-corr", type=float, default=DEFAULT_MIN_RECENT_ABS_CORR)
    parser.add_argument("--support-recent-min-sign-consistency", type=float, default=DEFAULT_MIN_RECENT_SIGN_CONSISTENCY)
    parser.add_argument("--support-recent-min-retention", type=float, default=DEFAULT_MIN_RECENT_RETENTION)

    parser.add_argument("--directional-min-abs-corr", type=float, default=0.03)
    parser.add_argument("--directional-crash-weak-abs-corr", type=float, default=0.015)
    parser.add_argument("--selection-profile-top-n", type=int, default=100)
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
