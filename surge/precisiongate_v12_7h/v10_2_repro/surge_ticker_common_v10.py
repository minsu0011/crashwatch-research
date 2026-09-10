from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import tempfile
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


SCHEMA_VERSION = "crashwatch_surge_tickerwise_correlation_map_v10"
TARGET_COLUMN_DEFAULT = "label_abs_surge_3d_5pct"
TARGET_VALID_COLUMN_DEFAULT = "target_valid"
DATE_COLUMN_DEFAULT = "date"
TICKER_COLUMN_DEFAULT = "ticker"

ERROR_OTHER = 0
ERROR_A_TOP_TP = 1
ERROR_B_TOP_FP = 2
ERROR_C_LOW_TP = 3
ERROR_D_LOW_TN = 4
ERROR_GROUP_NAMES = {
    ERROR_OTHER: "OTHER",
    ERROR_A_TOP_TP: "A_TOP_TRUE_POSITIVE",
    ERROR_B_TOP_FP: "B_TOP_FALSE_POSITIVE",
    ERROR_C_LOW_TP: "C_LOW_MISSED_POSITIVE",
    ERROR_D_LOW_TN: "D_LOW_TRUE_NEGATIVE",
}

LEAKAGE_TOKENS = (
    "future",
    "forward",
    "fwd",
    "target",
    "label",
    "first_hit",
    "hit_day",
    "best_forward",
    "next_return",
    "next_price",
    "t+1",
    "t+2",
    "t+3",
)
LEAKAGE_ALLOWLIST = frozenset(
    {
        "t_network_market_lead_beta_60",
        "t_network_peer_lead_beta_60",
        "t_peer_lead_lag_1",
        "t_peer_lead_lag_3",
    }
)


@dataclass(frozen=True)
class FoldSpec:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    purge_start: pd.Timestamp | None = None
    purge_end: pd.Timestamp | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_id": int(self.fold_id),
            "train_start": self.train_start.strftime("%Y-%m-%d"),
            "train_end": self.train_end.strftime("%Y-%m-%d"),
            "purge_start": self.purge_start.strftime("%Y-%m-%d") if self.purge_start is not None else None,
            "purge_end": self.purge_end.strftime("%Y-%m-%d") if self.purge_end is not None else None,
            "validation_start": self.validation_start.strftime("%Y-%m-%d"),
            "validation_end": self.validation_end.strftime("%Y-%m-%d"),
        }


@dataclass(frozen=True)
class MatchPair:
    fold_id: int
    axis: str
    case_index: int
    control_index: int
    distance: float
    same_date: bool
    same_market: bool
    same_bucket: bool
    same_industry: bool


class RunStatus:
    def __init__(self, output_dir: Path, run_name: str):
        self.output_dir = output_dir
        self.path = output_dir / "RUN_STATUS.json"
        self.started = datetime.now(timezone.utc)
        self.payload: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "run_name": run_name,
            "status": "RUNNING",
            "started_at": utc_now(),
            "stages": {},
        }
        atomic_write_json(self.path, self.payload)

    def stage(self, name: str, status: str, **details: Any) -> None:
        self.payload["stages"][name] = {"status": status, "updated_at": utc_now(), **details}
        self.payload["elapsed_seconds"] = (datetime.now(timezone.utc) - self.started).total_seconds()
        atomic_write_json(self.path, self.payload)

    def success(self, **details: Any) -> None:
        self.payload.update(
            {
                "status": "SUCCESS",
                "completed_at": utc_now(),
                "elapsed_seconds": (datetime.now(timezone.utc) - self.started).total_seconds(),
                **details,
            }
        )
        atomic_write_json(self.path, self.payload)

    def failure(self, exc: BaseException) -> None:
        self.payload.update(
            {
                "status": "FAILED",
                "failed_at": utc_now(),
                "elapsed_seconds": (datetime.now(timezone.utc) - self.started).total_seconds(),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        atomic_write_json(self.path, self.payload)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"JSON conversion unsupported: {type(value)!r}")


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def hash_strings(values: Sequence[str]) -> str:
    return sha256_bytes(stable_json_bytes(list(values)))


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default).encode("utf-8"),
    )


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".csv", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_table(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet 입력에는 pyarrow가 필요합니다. requirements_surge_separation_map_v9.txt를 설치하세요."
            ) from exc
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, usecols=list(columns) if columns is not None else None)
    if suffixes.endswith(".feather"):
        return pd.read_feather(path, columns=list(columns) if columns is not None else None)
    if suffixes.endswith(".pkl") or suffixes.endswith(".pickle"):
        frame = pd.read_pickle(path)
        return frame[list(columns)] if columns is not None else frame
    raise ValueError(f"Unsupported table format: {path}")


def table_columns(path: Path) -> list[str]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet schema 확인에는 pyarrow가 필요합니다.") from exc
        return list(pq.ParquetFile(path).schema_arrow.names)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return list(pd.read_csv(path, nrows=0).columns)
    return list(read_table(path).columns)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(float).ne(0)
    normalized = series.astype("string").str.strip().str.lower()
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n", "", "<na>", "nan", "none"}
    unknown = sorted(set(normalized.dropna().unique()) - truthy - falsy)
    if unknown:
        raise ValueError(f"Boolean conversion failed: {unknown[:10]}")
    return normalized.isin(truthy)


def leakage_reason(column: str) -> str | None:
    normalized = str(column).strip().lower()
    if normalized in LEAKAGE_ALLOWLIST:
        return None
    for token in LEAKAGE_TOKENS:
        if token in normalized:
            return f"name_contains:{token}"
    if normalized in {
        "date",
        "datetime",
        "timestamp",
        "ticker",
        "symbol",
        "code",
        "name",
        "source_row_id",
        "target_valid",
        "sealed_do_not_train_or_tune",
    }:
        return "metadata_or_identifier"
    return None


def verify_feature_names(features: Sequence[str], dataset_columns: Sequence[str]) -> None:
    missing = [feature for feature in features if feature not in dataset_columns]
    if missing:
        raise ValueError(f"Dataset missing features ({len(missing)}): {missing[:20]}")
    duplicates = pd.Series(list(features)).duplicated(keep=False)
    if duplicates.any():
        repeated = pd.Series(list(features))[duplicates].tolist()
        raise ValueError(f"Duplicate features: {repeated[:20]}")
    blocked = [(feature, leakage_reason(feature)) for feature in features if leakage_reason(feature) is not None]
    if blocked:
        raise ValueError(f"Leakage/metadata features detected: {blocked[:20]}")


def join_source_and_target(
    source: pd.DataFrame,
    target_sidecar: pd.DataFrame,
    target_column: str = TARGET_COLUMN_DEFAULT,
    target_valid_column: str = TARGET_VALID_COLUMN_DEFAULT,
    date_column: str = DATE_COLUMN_DEFAULT,
    ticker_column: str = TICKER_COLUMN_DEFAULT,
) -> pd.DataFrame:
    source = source.copy()
    target_sidecar = target_sidecar.copy()
    if "source_row_id" not in source.columns:
        source.insert(0, "source_row_id", np.arange(len(source), dtype=np.int64))
    if "source_row_id" not in target_sidecar.columns:
        if len(source) != len(target_sidecar):
            raise ValueError("Target sidecar has no source_row_id and row count differs")
        target_sidecar.insert(0, "source_row_id", np.arange(len(target_sidecar), dtype=np.int64))
    if source["source_row_id"].duplicated().any() or target_sidecar["source_row_id"].duplicated().any():
        raise ValueError("source_row_id is not a one-to-one key")
    required = {"source_row_id", target_column, target_valid_column}
    missing = required - set(target_sidecar.columns)
    if missing:
        raise ValueError(f"Target sidecar missing columns: {sorted(missing)}")
    carry = ["source_row_id", target_column, target_valid_column]
    for column in (date_column, ticker_column):
        if column in target_sidecar.columns:
            carry.append(column)
    side = target_sidecar[carry].rename(
        columns={date_column: f"{date_column}__target", ticker_column: f"{ticker_column}__target"}
    )
    joined = source.merge(side, on="source_row_id", how="left", validate="one_to_one", sort=False)
    if len(joined) != len(source):
        raise ValueError("Target join changed row count")
    side_date = f"{date_column}__target"
    if date_column in joined.columns and side_date in joined.columns:
        left = pd.to_datetime(joined[date_column], errors="coerce")
        right = pd.to_datetime(joined[side_date], errors="coerce")
        mismatch = left.notna() & right.notna() & left.ne(right)
        if mismatch.any():
            raise ValueError(f"Date mismatch after target join: {int(mismatch.sum())}")
        joined.drop(columns=[side_date], inplace=True)
    side_ticker = f"{ticker_column}__target"
    if ticker_column in joined.columns and side_ticker in joined.columns:
        left = joined[ticker_column].astype("string").str.strip()
        right = joined[side_ticker].astype("string").str.strip()
        mismatch = left.notna() & right.notna() & left.ne(right)
        if mismatch.any():
            raise ValueError(f"Ticker mismatch after target join: {int(mismatch.sum())}")
        joined.drop(columns=[side_ticker], inplace=True)
    joined[target_valid_column] = parse_bool_series(joined[target_valid_column])
    joined[target_column] = pd.to_numeric(joined[target_column], errors="coerce")
    invalid = joined[target_valid_column] & ~joined[target_column].isin([0, 1])
    if invalid.any():
        raise ValueError(f"Non-binary labels in target_valid rows: {int(invalid.sum())}")
    return joined


def load_folds(path: Path) -> list[FoldSpec]:
    raw = load_json(path)
    if isinstance(raw, Mapping) and "folds" in raw:
        raw = raw["folds"]
    if not isinstance(raw, list):
        raise ValueError("Fold JSON must be a list")
    folds: list[FoldSpec] = []
    for item in raw:
        folds.append(
            FoldSpec(
                fold_id=int(item["fold_id"]),
                train_start=pd.Timestamp(item["train_start"]),
                train_end=pd.Timestamp(item.get("train_end", item.get("train_date_max"))),
                purge_start=pd.Timestamp(item["purge_start"]) if item.get("purge_start") else None,
                purge_end=pd.Timestamp(item["purge_end"]) if item.get("purge_end") else None,
                validation_start=pd.Timestamp(item.get("validation_start", item.get("validation_date_min"))),
                validation_end=pd.Timestamp(item.get("validation_end", item.get("validation_date_max"))),
            )
        )
    validate_folds(folds)
    return sorted(folds, key=lambda fold: fold.fold_id)


def validate_folds(folds: Sequence[FoldSpec]) -> None:
    ids = [fold.fold_id for fold in folds]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate fold IDs: {ids}")
    for fold in folds:
        if fold.train_start > fold.train_end:
            raise ValueError(f"fold {fold.fold_id}: train_start > train_end")
        if fold.validation_start > fold.validation_end:
            raise ValueError(f"fold {fold.fold_id}: validation_start > validation_end")
        if fold.train_end >= fold.validation_start:
            raise ValueError(f"fold {fold.fold_id}: train/validation overlap")
        if (fold.purge_start is None) != (fold.purge_end is None):
            raise ValueError(f"fold {fold.fold_id}: incomplete purge interval")
        if fold.purge_start is not None and fold.purge_end is not None:
            if fold.purge_start <= fold.train_end:
                raise ValueError(f"fold {fold.fold_id}: purge overlaps train")
            if fold.purge_end >= fold.validation_start:
                raise ValueError(f"fold {fold.fold_id}: purge overlaps validation")


def role_for_fold(fold_id: int, roles: Mapping[str, Sequence[int]]) -> str:
    for role, fold_ids in roles.items():
        if int(fold_id) in {int(value) for value in fold_ids}:
            return str(role)
    return "unassigned"


def load_feature_profile(path: Path, profile: str = "P0_FULL_439") -> list[str]:
    payload = load_json(path)
    profiles = payload.get("profiles", payload)
    if profile not in profiles:
        raise KeyError(f"Feature profile missing: {profile}")
    raw = profiles[profile]
    features = raw.get("features", raw) if isinstance(raw, Mapping) else raw
    if not isinstance(features, list) or not features:
        raise ValueError(f"Feature profile is empty: {profile}")
    result = [str(feature) for feature in features]
    if len(result) != len(set(result)):
        raise ValueError(f"Feature profile contains duplicates: {profile}")
    return result


def load_correlation_matrix(path: Path, features: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col=0)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    missing_rows = [feature for feature in features if feature not in frame.index]
    missing_cols = [feature for feature in features if feature not in frame.columns]
    if missing_rows or missing_cols:
        raise ValueError(f"Correlation feature mismatch rows={missing_rows[:10]} cols={missing_cols[:10]}")
    matrix = frame.loc[list(features), list(features)].apply(pd.to_numeric, errors="coerce")
    values = np.abs(matrix.to_numpy(dtype=np.float64))
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.maximum(values, values.T)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(np.clip(values, 0.0, 1.0), index=list(features), columns=list(features))


def datewise_percentile_rank(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    dates_series = pd.Series(pd.to_datetime(dates, errors="coerce"))
    series = pd.Series(values)
    ranked = series.groupby(dates_series, sort=False).rank(method="average", pct=True, na_option="keep")
    return ranked.to_numpy(dtype=np.float64)


def groupwise_percentile_rank(values: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    keys = [pd.Series(group).astype("string").fillna("<NA>") for group in groups]
    if len(keys) == 1:
        group_key: Any = keys[0]
    else:
        group_key = pd.MultiIndex.from_arrays([key.to_numpy() for key in keys])
    ranked = pd.Series(values).groupby(group_key, sort=False).rank(method="average", pct=True, na_option="keep")
    return ranked.to_numpy(dtype=np.float64)


def groupwise_robust_z(values: np.ndarray, groups: Sequence[np.ndarray], clip: float = 8.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    keys = [pd.Series(group).astype("string").fillna("<NA>") for group in groups]
    if len(keys) == 1:
        group_key: Any = keys[0]
    else:
        group_key = pd.MultiIndex.from_arrays([key.to_numpy() for key in keys])
    series = pd.Series(values)
    grouped = series.groupby(group_key, sort=False)
    median = grouped.transform("median")
    deviation = (series - median).abs()
    mad = deviation.groupby(group_key, sort=False).transform("median")
    scale = 1.4826 * mad
    fallback = grouped.transform("std")
    scale = scale.where(scale > 1e-12, fallback)
    z = (series - median) / scale.replace(0.0, np.nan)
    return np.clip(z.to_numpy(dtype=np.float64), -clip, clip)


def groupwise_median_delta(values: np.ndarray, groups: Sequence[np.ndarray]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    keys = [pd.Series(group).astype("string").fillna("<NA>") for group in groups]
    if len(keys) == 1:
        group_key: Any = keys[0]
    else:
        group_key = pd.MultiIndex.from_arrays([key.to_numpy() for key in keys])
    series = pd.Series(values)
    median = series.groupby(group_key, sort=False).transform("median")
    return (series - median).to_numpy(dtype=np.float64)


def build_error_group_codes(
    target: np.ndarray,
    base_score: np.ndarray,
    dates: np.ndarray,
    top_quantile: float = 0.80,
    low_quantile: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.5 < top_quantile < 1.0:
        raise ValueError("top_quantile must be in (0.5, 1.0)")
    if not 0.0 < low_quantile < top_quantile:
        raise ValueError("low_quantile must be in (0, top_quantile)")
    y = np.asarray(target, dtype=np.int8)
    rank = datewise_percentile_rank(base_score, dates)
    codes = np.zeros(len(y), dtype=np.int8)
    finite = np.isfinite(rank) & np.isin(y, [0, 1])
    high = finite & (rank >= float(top_quantile))
    low = finite & (rank <= float(low_quantile))
    codes[high & (y == 1)] = ERROR_A_TOP_TP
    codes[high & (y == 0)] = ERROR_B_TOP_FP
    codes[low & (y == 1)] = ERROR_C_LOW_TP
    codes[low & (y == 0)] = ERROR_D_LOW_TN
    return codes, rank


def _matching_penalty(
    case_row: pd.Series,
    control_rows: pd.DataFrame,
    score_column: str,
    market_column: str,
    bucket_column: str,
    industry_column: str,
) -> np.ndarray:
    score_distance = np.abs(pd.to_numeric(control_rows[score_column], errors="coerce").to_numpy(dtype=np.float64) - float(case_row[score_column]))
    penalty = score_distance.copy()
    if market_column in control_rows.columns:
        penalty += np.where(control_rows[market_column].astype(str).to_numpy() == str(case_row.get(market_column, "")), 0.0, 0.08)
    if bucket_column in control_rows.columns:
        penalty += np.where(control_rows[bucket_column].astype(str).to_numpy() == str(case_row.get(bucket_column, "")), 0.0, 0.04)
    if industry_column in control_rows.columns:
        penalty += np.where(control_rows[industry_column].astype(str).to_numpy() == str(case_row.get(industry_column, "")), 0.0, 0.02)
    return penalty


def build_matched_pairs(
    frame: pd.DataFrame,
    fold_id: int,
    axis: str,
    case_code: int,
    control_code: int,
    controls_per_case: int = 3,
    date_column: str = "date",
    score_column: str = "base_rank",
    market_column: str = "market",
    bucket_column: str = "bucket",
    industry_column: str = "industry_name",
) -> pd.DataFrame:
    subset = frame[frame["error_code"].isin([case_code, control_code])].copy()
    if subset.empty:
        return pd.DataFrame(columns=[field.name for field in dataclasses.fields(MatchPair)])
    subset[date_column] = pd.to_datetime(subset[date_column], errors="coerce")
    records: list[dict[str, Any]] = []
    controls_per_case = max(1, int(controls_per_case))
    for current_date, date_part in subset.groupby(date_column, sort=True):
        cases = date_part[date_part["error_code"].eq(case_code)]
        controls = date_part[date_part["error_code"].eq(control_code)]
        if cases.empty or controls.empty:
            continue
        for case_index, case_row in cases.iterrows():
            penalties = _matching_penalty(
                case_row,
                controls,
                score_column,
                market_column,
                bucket_column,
                industry_column,
            )
            order = np.argsort(penalties, kind="mergesort")[: min(controls_per_case, len(controls))]
            selected = controls.iloc[order]
            for local_position, (control_index, control_row) in enumerate(selected.iterrows()):
                records.append(
                    dataclasses.asdict(
                        MatchPair(
                            fold_id=int(fold_id),
                            axis=str(axis),
                            case_index=int(case_index),
                            control_index=int(control_index),
                            distance=float(penalties[order[local_position]]),
                            same_date=True,
                            same_market=str(case_row.get(market_column, "")) == str(control_row.get(market_column, "")),
                            same_bucket=str(case_row.get(bucket_column, "")) == str(control_row.get(bucket_column, "")),
                            same_industry=str(case_row.get(industry_column, "")) == str(control_row.get(industry_column, "")),
                        )
                    )
                )
    return pd.DataFrame.from_records(records)


def safe_roc_auc(target: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(s) & np.isin(y, [0, 1])
    if int(mask.sum()) < 4 or len(np.unique(y[mask])) < 2:
        return float("nan")
    if np.nanstd(s[mask]) <= 1e-15:
        return 0.5
    return float(roc_auc_score(y[mask], s[mask]))


def oriented_auc(target: np.ndarray, scores: np.ndarray) -> tuple[float, int]:
    auc = safe_roc_auc(target, scores)
    if not np.isfinite(auc):
        return float("nan"), 0
    if auc >= 0.5:
        return auc, 1
    return 1.0 - auc, -1


def standardized_mean_difference(positive: np.ndarray, negative: np.ndarray) -> float:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    denominator = math.sqrt((float(np.var(pos, ddof=1)) + float(np.var(neg, ddof=1))) / 2.0)
    if denominator <= 1e-15:
        return 0.0
    return float((np.mean(pos) - np.mean(neg)) / denominator)


def jensen_shannon_divergence(positive: np.ndarray, negative: np.ndarray, bins: int = 10) -> float:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 4 or len(neg) < 4:
        return float("nan")
    combined = np.concatenate([pos, neg])
    edges = np.unique(np.quantile(combined, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 3:
        return 0.0
    p, _ = np.histogram(pos, bins=edges)
    q, _ = np.histogram(neg, bins=edges)
    p = (p.astype(np.float64) + 0.5) / (float(np.sum(p)) + 0.5 * len(p))
    q = (q.astype(np.float64) + 0.5) / (float(np.sum(q)) + 0.5 * len(q))
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(np.clip(p / m, 1e-15, None))))
    kl_qm = float(np.sum(q * np.log(np.clip(q / m, 1e-15, None))))
    return float(0.5 * (kl_pm + kl_qm))


def normalized_wasserstein(positive: np.ndarray, negative: np.ndarray) -> float:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    combined = np.concatenate([pos, neg])
    q25, q75 = np.quantile(combined, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= 1e-12:
        scale = float(np.std(combined))
    if scale <= 1e-12:
        return 0.0
    return float(wasserstein_distance(pos, neg) / scale)


def matched_pair_concordance(
    values: np.ndarray,
    pairs: pd.DataFrame,
    row_indices: np.ndarray | None = None,
) -> tuple[float, int, int]:
    """Return oriented case-control concordance.

    ``pairs`` stores stable row identifiers. ``values`` may be a local fold array, so
    ``row_indices`` maps those identifiers to local positions. This prevents global
    row IDs from being interpreted as array offsets.
    """
    if pairs.empty:
        return float("nan"), 0, 0
    array = np.asarray(values, dtype=np.float64)
    if row_indices is None:
        case_indices = pairs["case_index"].to_numpy(dtype=np.int64)
        control_indices = pairs["control_index"].to_numpy(dtype=np.int64)
        in_bounds = (
            (case_indices >= 0)
            & (control_indices >= 0)
            & (case_indices < len(array))
            & (control_indices < len(array))
        )
        case_positions = case_indices[in_bounds]
        control_positions = control_indices[in_bounds]
    else:
        identifiers = np.asarray(row_indices, dtype=np.int64)
        lookup = {int(identifier): int(position) for position, identifier in enumerate(identifiers)}
        case_positions_list: list[int] = []
        control_positions_list: list[int] = []
        for case_identifier, control_identifier in pairs[["case_index", "control_index"]].itertuples(index=False):
            case_position = lookup.get(int(case_identifier))
            control_position = lookup.get(int(control_identifier))
            if case_position is None or control_position is None:
                continue
            case_positions_list.append(case_position)
            control_positions_list.append(control_position)
        case_positions = np.asarray(case_positions_list, dtype=np.int64)
        control_positions = np.asarray(control_positions_list, dtype=np.int64)
    if len(case_positions) == 0:
        return float("nan"), 0, 0
    valid = np.isfinite(array[case_positions]) & np.isfinite(array[control_positions])
    valid_count = int(valid.sum())
    if valid_count < 4:
        return float("nan"), 0, valid_count
    differences = array[case_positions[valid]] - array[control_positions[valid]]
    positive = float(np.mean(differences > 0) + 0.5 * np.mean(differences == 0))
    if positive >= 0.5:
        return positive, 1, valid_count
    return 1.0 - positive, -1, valid_count


def separation_metrics(
    values: np.ndarray,
    error_codes: np.ndarray,
    positive_code: int,
    negative_code: int,
    matched_pairs: pd.DataFrame | None = None,
    row_indices: np.ndarray | None = None,
) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    codes = np.asarray(error_codes, dtype=np.int8)
    pos_mask = codes == int(positive_code)
    neg_mask = codes == int(negative_code)
    pos = x[pos_mask]
    neg = x[neg_mask]
    pos_valid = pos[np.isfinite(pos)]
    neg_valid = neg[np.isfinite(neg)]
    target = np.concatenate([np.ones(len(pos_valid), dtype=np.int8), np.zeros(len(neg_valid), dtype=np.int8)])
    scores = np.concatenate([pos_valid, neg_valid])
    auc, orientation = oriented_auc(target, scores)
    ks_value = float("nan")
    if len(pos_valid) >= 2 and len(neg_valid) >= 2:
        ks_value = float(ks_2samp(pos_valid, neg_valid, alternative="two-sided", method="auto").statistic)
    pair_auc, pair_orientation, pair_count = matched_pair_concordance(
        x,
        matched_pairs if matched_pairs is not None else pd.DataFrame(),
        row_indices=row_indices,
    )
    missing_pos = float(np.mean(~np.isfinite(pos))) if len(pos) else float("nan")
    missing_neg = float(np.mean(~np.isfinite(neg))) if len(neg) else float("nan")
    return {
        "positive_rows": int(len(pos)),
        "negative_rows": int(len(neg)),
        "positive_valid_n": int(len(pos_valid)),
        "negative_valid_n": int(len(neg_valid)),
        "positive_coverage": float(len(pos_valid) / len(pos)) if len(pos) else float("nan"),
        "negative_coverage": float(len(neg_valid) / len(neg)) if len(neg) else float("nan"),
        "raw_auc": safe_roc_auc(target, scores),
        "oriented_auc": auc,
        "orientation": int(orientation),
        "smd": standardized_mean_difference(pos_valid, neg_valid),
        "ks": ks_value,
        "wasserstein_iqr": normalized_wasserstein(pos_valid, neg_valid),
        "js_divergence": jensen_shannon_divergence(pos_valid, neg_valid),
        "cliff_delta_abs": abs(2.0 * auc - 1.0) if np.isfinite(auc) else float("nan"),
        "positive_median": float(np.median(pos_valid)) if len(pos_valid) else float("nan"),
        "negative_median": float(np.median(neg_valid)) if len(neg_valid) else float("nan"),
        "missing_rate_positive": missing_pos,
        "missing_rate_negative": missing_neg,
        "missing_rate_difference": missing_pos - missing_neg if np.isfinite(missing_pos) and np.isfinite(missing_neg) else float("nan"),
        "matched_concordance": pair_auc,
        "matched_orientation": int(pair_orientation),
        "matched_pair_count": int(pair_count),
    }


def safe_median_fill(values: np.ndarray, fallback: float = 0.0) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    median = float(np.median(finite)) if len(finite) else float(fallback)
    return np.where(np.isfinite(array), array, median), median


def cluster_innovation_values(
    feature_values: np.ndarray,
    peer_values: Sequence[np.ndarray],
    dates: np.ndarray,
) -> np.ndarray:
    base_rank = datewise_percentile_rank(feature_values, dates)
    if not peer_values:
        return np.full(len(base_rank), np.nan, dtype=np.float64)
    peer_ranks = [datewise_percentile_rank(values, dates) for values in peer_values]
    peer_center = np.nanmedian(np.vstack(peer_ranks), axis=0)
    return base_rank - peer_center


def peer_weighted_innovation_values(
    feature_values: np.ndarray,
    peers: Sequence[tuple[np.ndarray, float]],
    dates: np.ndarray,
) -> np.ndarray:
    base_rank = datewise_percentile_rank(feature_values, dates)
    if not peers:
        return np.full(len(base_rank), np.nan, dtype=np.float64)
    weighted_sum = np.zeros(len(base_rank), dtype=np.float64)
    weight_sum = np.zeros(len(base_rank), dtype=np.float64)
    for values, weight in peers:
        peer_rank = datewise_percentile_rank(values, dates)
        valid = np.isfinite(peer_rank)
        weighted_sum[valid] += float(weight) * peer_rank[valid]
        weight_sum[valid] += float(weight)
    peer_center = np.divide(
        weighted_sum,
        weight_sum,
        out=np.full(len(base_rank), np.nan, dtype=np.float64),
        where=weight_sum > 0,
    )
    return base_rank - peer_center


def parse_horizon_feature(feature: str) -> tuple[str, int] | None:
    tokens = str(feature).rsplit("_", 1)
    if len(tokens) != 2 or not tokens[1].isdigit():
        return None
    horizon = int(tokens[1])
    if horizon not in {1, 2, 3, 5, 10, 20, 60, 120, 250}:
        return None
    return tokens[0], horizon


def build_horizon_pairs(features: Sequence[str]) -> pd.DataFrame:
    grouped: dict[str, list[tuple[int, str]]] = {}
    for feature in features:
        parsed = parse_horizon_feature(feature)
        if parsed is None:
            continue
        stem, horizon = parsed
        grouped.setdefault(stem, []).append((horizon, feature))
    records: list[dict[str, Any]] = []
    for stem, members in sorted(grouped.items()):
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                short_horizon, short_feature = members[i]
                long_horizon, long_feature = members[j]
                records.append(
                    {
                        "stem": stem,
                        "short_horizon": int(short_horizon),
                        "long_horizon": int(long_horizon),
                        "short_feature": short_feature,
                        "long_feature": long_feature,
                        "transform_name": f"HORIZON_RANK_SPREAD::{short_feature}::{long_feature}",
                    }
                )
    return pd.DataFrame.from_records(records)


def horizon_rank_spread(short_values: np.ndarray, long_values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    return datewise_percentile_rank(short_values, dates) - datewise_percentile_rank(long_values, dates)


def wilson_lower_bound(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return float((center - margin) / denominator)


def precision_curve_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    minimum_alerts: int = 30,
    target_precision: float = 0.70,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s) & np.isin(y, [0, 1])
    y = y[valid]
    s = s[valid]
    rows = int(len(y))
    positives = int(np.sum(y == 1))
    if rows == 0:
        return {
            "rows": 0,
            "positives": 0,
            "best_precision_min_alerts": float("nan"),
            "best_precision_alerts": 0,
            "best_precision_tp": 0,
            "best_precision_recall": float("nan"),
            "best_precision_wilson_lcb": float("nan"),
            "max_recall_at_precision_target": 0.0,
            "precision_target_alerts": 0,
            "precision_target_precision": float("nan"),
            "precision_target_wilson_lcb": float("nan"),
        }
    order = np.argsort(-s, kind="mergesort")
    sorted_y = y[order]
    cumulative_tp = np.cumsum(sorted_y == 1)
    alert_counts = np.arange(1, rows + 1)
    precision = cumulative_tp / alert_counts
    recall = cumulative_tp / positives if positives else np.full(rows, np.nan)
    minimum_alerts = max(1, int(minimum_alerts))
    eligible = alert_counts >= minimum_alerts
    if np.any(eligible):
        eligible_indices = np.flatnonzero(eligible)
        best_local = int(np.argmax(precision[eligible]))
        best_index = int(eligible_indices[best_local])
        best_alerts = int(alert_counts[best_index])
        best_tp = int(cumulative_tp[best_index])
        best_precision = float(precision[best_index])
        best_recall = float(recall[best_index]) if positives else float("nan")
        best_lcb = wilson_lower_bound(best_tp, best_alerts)
    else:
        best_alerts = 0
        best_tp = 0
        best_precision = float("nan")
        best_recall = float("nan")
        best_lcb = float("nan")
    safe = eligible & (precision >= float(target_precision))
    if np.any(safe):
        safe_indices = np.flatnonzero(safe)
        best_safe_local = int(np.argmax(recall[safe]))
        safe_index = int(safe_indices[best_safe_local])
        safe_alerts = int(alert_counts[safe_index])
        safe_tp = int(cumulative_tp[safe_index])
        safe_recall = float(recall[safe_index])
        safe_precision = float(precision[safe_index])
        safe_lcb = wilson_lower_bound(safe_tp, safe_alerts)
    else:
        safe_alerts = 0
        safe_recall = 0.0
        safe_precision = float("nan")
        safe_lcb = float("nan")
    return {
        "rows": rows,
        "positives": positives,
        "best_precision_min_alerts": best_precision,
        "best_precision_alerts": best_alerts,
        "best_precision_tp": best_tp,
        "best_precision_recall": best_recall,
        "best_precision_wilson_lcb": best_lcb,
        "max_recall_at_precision_target": safe_recall,
        "precision_target_alerts": safe_alerts,
        "precision_target_precision": safe_precision,
        "precision_target_wilson_lcb": safe_lcb,
    }


def safe_binary_metrics(target: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(s) & np.isin(y, [0, 1])
    y = y[valid]
    s = s[valid]
    rows = int(len(y))
    positives = int(np.sum(y == 1))
    negatives = rows - positives
    base_rate = float(positives / rows) if rows else float("nan")
    result: dict[str, Any] = {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "base_rate": base_rate,
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "pr_auc_lift": float("nan"),
    }
    if rows and positives and negatives:
        result["pr_auc"] = float(average_precision_score(y, s))
        result["roc_auc"] = float(roc_auc_score(y, s))
        result["pr_auc_lift"] = float(result["pr_auc"] / base_rate) if base_rate > 0 else float("nan")
    return result


def aggregate_fold_metrics(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    numeric_columns = [
        column
        for column in frame.columns
        if column not in set(group_columns) | {"fold_id", "fold_role", "orientation", "matched_orientation"}
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    records: list[dict[str, Any]] = []
    for keys, part in frame.groupby(list(group_columns), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_columns, keys))
        record["fold_count"] = int(part["fold_id"].nunique()) if "fold_id" in part.columns else int(len(part))
        for column in numeric_columns:
            values = pd.to_numeric(part[column], errors="coerce").to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            record[f"mean_{column}"] = float(np.mean(finite)) if len(finite) else float("nan")
            record[f"median_{column}"] = float(np.median(finite)) if len(finite) else float("nan")
            record[f"min_{column}"] = float(np.min(finite)) if len(finite) else float("nan")
            record[f"max_{column}"] = float(np.max(finite)) if len(finite) else float("nan")
            record[f"std_{column}"] = float(np.std(finite, ddof=0)) if len(finite) else float("nan")
            record[f"positive_ratio_{column}"] = float(np.mean(finite > 0)) if len(finite) else float("nan")
        if "orientation" in part.columns:
            orientation = pd.to_numeric(part["orientation"], errors="coerce").to_numpy(dtype=np.float64)
            orientation = orientation[np.isfinite(orientation) & (orientation != 0)]
            record["orientation_consistency"] = (
                float(max(np.mean(orientation > 0), np.mean(orientation < 0))) if len(orientation) else float("nan")
            )
        if "matched_orientation" in part.columns:
            orientation = pd.to_numeric(part["matched_orientation"], errors="coerce").to_numpy(dtype=np.float64)
            orientation = orientation[np.isfinite(orientation) & (orientation != 0)]
            record["matched_orientation_consistency"] = (
                float(max(np.mean(orientation > 0), np.mean(orientation < 0))) if len(orientation) else float("nan")
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def compute_interaction_moment_screen(
    matrix: np.ndarray,
    feature_names: Sequence[str],
    error_codes: np.ndarray,
    positive_code: int,
    negative_code: int,
    minimum_valid_rows: int = 30,
) -> pd.DataFrame:
    x = np.asarray(matrix, dtype=np.float64)
    codes = np.asarray(error_codes, dtype=np.int8)
    pos_mask = codes == int(positive_code)
    neg_mask = codes == int(negative_code)
    if int(pos_mask.sum()) < minimum_valid_rows or int(neg_mask.sum()) < minimum_valid_rows:
        return pd.DataFrame()
    combined = x[pos_mask | neg_mask]
    medians = np.nanmedian(combined, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    filled = np.where(np.isfinite(x), x, medians)
    scale = np.nanstd(filled[pos_mask | neg_mask], axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    center = np.nanmean(filled[pos_mask | neg_mask], axis=0)
    standardized = (filled - center) / scale
    pos = standardized[pos_mask]
    neg = standardized[neg_mask]
    pos_centered = pos - np.mean(pos, axis=0, keepdims=True)
    neg_centered = neg - np.mean(neg, axis=0, keepdims=True)
    cov_pos = (pos_centered.T @ pos_centered) / max(1, len(pos) - 1)
    cov_neg = (neg_centered.T @ neg_centered) / max(1, len(neg) - 1)
    delta_cov = cov_pos - cov_neg
    mean_pos = np.mean(pos, axis=0)
    mean_neg = np.mean(neg, axis=0)
    main_effect = np.abs(mean_pos - mean_neg)
    interaction_score = np.abs(delta_cov) - 0.25 * (main_effect[:, None] + main_effect[None, :])
    records: list[dict[str, Any]] = []
    p = len(feature_names)
    for i in range(p):
        for j in range(i + 1, p):
            records.append(
                {
                    "feature_a": str(feature_names[i]),
                    "feature_b": str(feature_names[j]),
                    "delta_covariance": float(delta_cov[i, j]),
                    "abs_delta_covariance": float(abs(delta_cov[i, j])),
                    "main_effect_a": float(main_effect[i]),
                    "main_effect_b": float(main_effect[j]),
                    "interaction_excess_score": float(interaction_score[i, j]),
                }
            )
    return pd.DataFrame.from_records(records)


def pair_transform_values(
    values_a: np.ndarray,
    values_b: np.ndarray,
    dates: np.ndarray,
) -> dict[str, np.ndarray]:
    rank_a = datewise_percentile_rank(values_a, dates)
    rank_b = datewise_percentile_rank(values_b, dates)
    return {
        "rank_difference": rank_a - rank_b,
        "rank_product": rank_a * rank_b,
        "rank_min": np.minimum(rank_a, rank_b),
        "rank_max": np.maximum(rank_a, rank_b),
        "rank_disagreement": np.abs(rank_a - rank_b),
    }


def fit_forward_logistic_probe(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    negative_weight: float = 2.0,
    regularization_c: float = 0.1,
    seed: int = 17,
) -> np.ndarray:
    columns = list(feature_columns)
    x_train = train_frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    x_validation = validation_frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    medians = np.nanmedian(x_train, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    x_train = np.where(np.isfinite(x_train), x_train, medians)
    x_validation = np.where(np.isfinite(x_validation), x_validation, medians)
    means = np.mean(x_train, axis=0)
    scales = np.std(x_train, axis=0)
    scales = np.where(scales > 1e-9, scales, 1.0)
    x_train = (x_train - means) / scales
    x_validation = (x_validation - means) / scales
    y_train = pd.to_numeric(train_frame[target_column], errors="coerce").to_numpy(dtype=np.int8)
    if len(np.unique(y_train)) < 2:
        return np.full(len(validation_frame), float(np.mean(y_train)) if len(y_train) else 0.0)
    weights = np.where(y_train == 0, float(negative_weight), 1.0)
    model = LogisticRegression(
        C=float(regularization_c),
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        random_state=int(seed),
    )
    model.fit(x_train, y_train, sample_weight=weights)
    return model.predict_proba(x_validation)[:, 1].astype(np.float64)


def compute_output_inventory(output_dir: Path, exclude: Sequence[str] | None = None) -> list[dict[str, Any]]:
    excluded = {Path(item).as_posix() for item in (exclude or [])}
    records: list[dict[str, Any]] = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        if relative in excluded:
            continue
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return records


def verify_output_inventory(output_dir: Path, inventory: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for item in inventory:
        relative = str(item.get("path", ""))
        path = output_dir / relative
        if not path.exists():
            reasons.append(f"missing:{relative}")
            continue
        if int(item.get("bytes", -1)) != path.stat().st_size:
            reasons.append(f"size:{relative}")
        expected = str(item.get("sha256", ""))
        if expected and sha256_file(path) != expected:
            reasons.append(f"sha256:{relative}")
    return not reasons, reasons


def graphml_escape(value: Any) -> str:
    text = str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def write_graphml(path: Path, nodes: pd.DataFrame, edges: pd.DataFrame) -> None:
    node_attributes = [column for column in nodes.columns if column != "node_id"]
    edge_attributes = [column for column in edges.columns if column not in {"source", "target"}]
    key_lines: list[str] = []
    for column in node_attributes:
        key_lines.append(f'<key id="n_{graphml_escape(column)}" for="node" attr.name="{graphml_escape(column)}" attr.type="string"/>')
    for column in edge_attributes:
        key_lines.append(f'<key id="e_{graphml_escape(column)}" for="edge" attr.name="{graphml_escape(column)}" attr.type="string"/>')
    node_lines: list[str] = []
    for row in nodes.itertuples(index=False):
        record = row._asdict()
        data = "".join(
            f'<data key="n_{graphml_escape(column)}">{graphml_escape(record.get(column, ""))}</data>'
            for column in node_attributes
        )
        node_lines.append(f'<node id="{graphml_escape(record["node_id"])}">{data}</node>')
    edge_lines: list[str] = []
    for edge_id, row in enumerate(edges.itertuples(index=False)):
        record = row._asdict()
        data = "".join(
            f'<data key="e_{graphml_escape(column)}">{graphml_escape(record.get(column, ""))}</data>'
            for column in edge_attributes
        )
        edge_lines.append(
            f'<edge id="e{edge_id}" source="{graphml_escape(record["source"])}" target="{graphml_escape(record["target"])}">{data}</edge>'
        )
    document = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
            *key_lines,
            '<graph id="G" edgedefault="undirected">',
            *node_lines,
            *edge_lines,
            "</graph>",
            "</graphml>",
        ]
    )
    atomic_write_text(path, document)


# V10 ticker-specific extensions


from scipy.cluster.hierarchy import linkage as scipy_linkage, fcluster as scipy_fcluster
from scipy.spatial.distance import squareform as scipy_squareform
from scipy.stats import pearsonr as scipy_pearsonr, spearmanr as scipy_spearmanr
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss


@dataclass(frozen=True)
class TickerFoldEligibility:
    ticker: str
    fold_id: int
    role: str
    train_rows: int
    train_positive: int
    train_negative: int
    validation_rows: int
    validation_positive: int
    validation_negative: int
    eligible_map: bool
    eligible_model: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def safe_pearson(values: np.ndarray, target: np.ndarray, minimum_rows: int = 8) -> tuple[float, int]:
    x = np.asarray(values, dtype=float)
    y = np.asarray(target, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < int(minimum_rows):
        return float("nan"), n
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan"), n
    return float(np.corrcoef(x, y)[0, 1]), n


def safe_spearman(values: np.ndarray, target: np.ndarray, minimum_rows: int = 8) -> tuple[float, int]:
    x = np.asarray(values, dtype=float)
    y = np.asarray(target, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < int(minimum_rows):
        return float("nan"), n
    x = x[mask]
    y = y[mask]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), n
    result = scipy_spearmanr(x, y, nan_policy="omit")
    return float(result.statistic), n


def auc_standard_error(auc: float, positive_n: int, negative_n: int) -> float:
    if not math.isfinite(float(auc)) or positive_n <= 0 or negative_n <= 0:
        return float("nan")
    auc = float(auc)
    q1 = auc / max(2.0 - auc, 1e-12)
    q2 = 2.0 * auc * auc / max(1.0 + auc, 1e-12)
    numerator = (
        auc * (1.0 - auc)
        + (positive_n - 1) * (q1 - auc * auc)
        + (negative_n - 1) * (q2 - auc * auc)
    )
    denominator = positive_n * negative_n
    return float(math.sqrt(max(numerator / max(denominator, 1), 0.0)))


def shrink_auc_to_half(
    auc: float,
    positive_n: int,
    negative_n: int,
    prior_strength: float = 80.0,
) -> tuple[float, float]:
    if not math.isfinite(float(auc)) or positive_n <= 0 or negative_n <= 0:
        return float("nan"), 0.0
    harmonic = 2.0 / (1.0 / positive_n + 1.0 / negative_n)
    reliability = float(harmonic / (harmonic + max(float(prior_strength), 1e-12)))
    shrunk = 0.5 + reliability * (float(auc) - 0.5)
    return float(shrunk), reliability


def shrink_correlation(
    correlation: float,
    valid_n: int,
    prior_strength: float = 60.0,
) -> tuple[float, float]:
    if not math.isfinite(float(correlation)) or valid_n <= 3:
        return float("nan"), 0.0
    reliability = float((valid_n - 3) / max((valid_n - 3) + prior_strength, 1e-12))
    return float(float(correlation) * reliability), reliability


def fixed_direction_auc(
    target: np.ndarray,
    values: np.ndarray,
    direction: int,
) -> tuple[float, int, int, int]:
    y = np.asarray(target, dtype=int)
    x = np.asarray(values, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    y = y[mask]
    x = x[mask]
    positive_n = int(np.sum(y == 1))
    negative_n = int(np.sum(y == 0))
    if positive_n <= 0 or negative_n <= 0:
        return float("nan"), int(mask.sum()), positive_n, negative_n
    score = x if int(direction) >= 0 else -x
    return float(roc_auc_score(y, score)), int(mask.sum()), positive_n, negative_n


def infer_direction_from_auc(auc: float, tolerance: float = 1e-12) -> int:
    if not math.isfinite(float(auc)):
        return 0
    if float(auc) > 0.5 + tolerance:
        return 1
    if float(auc) < 0.5 - tolerance:
        return -1
    return 0


def summarize_fixed_direction(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str],
    selection_fold_ids: Sequence[int],
    confirmation_fold_ids: Sequence[int],
    recent_fold_ids: Sequence[int],
    raw_auc_column: str = "raw_auc",
    valid_rows_column: str = "valid_rows",
    positive_n_column: str = "positive_n",
    negative_n_column: str = "negative_n",
    prior_strength: float = 80.0,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    selection_set = {int(value) for value in selection_fold_ids}
    confirmation_set = {int(value) for value in confirmation_fold_ids}
    recent_set = {int(value) for value in recent_fold_ids}
    for keys, part in frame.groupby(list(group_columns), dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {column: value for column, value in zip(group_columns, keys)}
        selection = part.loc[part["fold_id"].astype(int).isin(selection_set)].copy()
        weighted_edges: list[tuple[float, float]] = []
        for row in selection.itertuples(index=False):
            auc = float(getattr(row, raw_auc_column))
            pos = int(getattr(row, positive_n_column))
            neg = int(getattr(row, negative_n_column))
            if not math.isfinite(auc) or pos <= 0 or neg <= 0:
                continue
            weight = 2.0 / (1.0 / pos + 1.0 / neg)
            weighted_edges.append((auc - 0.5, weight))
        if weighted_edges:
            signed_edge = float(
                sum(edge * weight for edge, weight in weighted_edges)
                / max(sum(weight for _, weight in weighted_edges), 1e-12)
            )
            direction = 1 if signed_edge > 0 else -1 if signed_edge < 0 else 0
        else:
            signed_edge = float("nan")
            direction = 0
        record["selection_direction"] = int(direction)
        record["selection_signed_auc_edge"] = signed_edge
        for role_name, fold_set in (
            ("selection", selection_set),
            ("confirmation", confirmation_set),
            ("recent", recent_set),
        ):
            subset = part.loc[part["fold_id"].astype(int).isin(fold_set)].copy()
            fixed_values: list[float] = []
            shrunk_values: list[float] = []
            weights: list[float] = []
            signs: list[int] = []
            for row in subset.itertuples(index=False):
                raw_auc = float(getattr(row, raw_auc_column))
                pos = int(getattr(row, positive_n_column))
                neg = int(getattr(row, negative_n_column))
                if not math.isfinite(raw_auc) or pos <= 0 or neg <= 0 or direction == 0:
                    continue
                fixed = raw_auc if direction > 0 else 1.0 - raw_auc
                shrunk, reliability = shrink_auc_to_half(fixed, pos, neg, prior_strength=prior_strength)
                fixed_values.append(float(fixed))
                shrunk_values.append(float(shrunk))
                weights.append(max(reliability, 1e-12))
                raw_direction = infer_direction_from_auc(raw_auc)
                signs.append(1 if raw_direction == direction else 0)
            prefix = role_name
            record[f"{prefix}_fold_count"] = int(len(fixed_values))
            record[f"{prefix}_mean_fixed_auc"] = (
                float(np.mean(fixed_values)) if fixed_values else float("nan")
            )
            record[f"{prefix}_min_fixed_auc"] = (
                float(np.min(fixed_values)) if fixed_values else float("nan")
            )
            record[f"{prefix}_mean_shrunk_auc"] = (
                float(np.average(shrunk_values, weights=weights)) if shrunk_values else float("nan")
            )
            record[f"{prefix}_direction_consistency"] = (
                float(np.mean(signs)) if signs else float("nan")
            )
        selection_auc = record.get("selection_mean_shrunk_auc", float("nan"))
        selection_min = record.get("selection_min_fixed_auc", float("nan"))
        selection_consistency = record.get("selection_direction_consistency", float("nan"))
        evidence = 0.0
        if math.isfinite(selection_auc):
            evidence += max(selection_auc - 0.5, 0.0) * 5.0
        if math.isfinite(selection_min):
            evidence += max(selection_min - 0.5, 0.0) * 2.0
        if math.isfinite(selection_consistency):
            evidence += max(selection_consistency - 0.5, 0.0) * 0.5
        record["selection_evidence_score"] = float(evidence)
        rows.append(record)
    return pd.DataFrame(rows)


def rolling_z_score(
    values: np.ndarray,
    window: int,
    minimum_periods: int | None = None,
    clip: float = 8.0,
) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    minimum_periods = int(minimum_periods or max(5, window // 3))
    mean = series.rolling(window, min_periods=minimum_periods).mean().shift(1)
    std = series.rolling(window, min_periods=minimum_periods).std(ddof=0).shift(1)
    result = (series - mean) / std.replace(0.0, np.nan)
    return result.clip(-clip, clip).to_numpy(dtype=float)


def rolling_percentile_rank(
    values: np.ndarray,
    window: int,
    minimum_periods: int | None = None,
) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    minimum_periods = int(minimum_periods or max(5, window // 3))
    result = np.full(len(x), np.nan, dtype=float)
    for index in range(len(x)):
        start = max(0, index - window)
        history = x[start:index]
        history = history[np.isfinite(history)]
        if history.size < minimum_periods or not math.isfinite(float(x[index])):
            continue
        less = float(np.sum(history < x[index]))
        equal = float(np.sum(history == x[index]))
        result[index] = (less + 0.5 * equal) / history.size
    return result


def lag_change(values: np.ndarray, periods: int) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return (series - series.shift(int(periods))).to_numpy(dtype=float)


def lag_ratio_change(values: np.ndarray, periods: int, epsilon: float = 1e-9) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    lagged = series.shift(int(periods))
    denominator = lagged.abs().clip(lower=epsilon)
    return ((series - lagged) / denominator).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)


def make_ticker_transform_name(feature: str, transform: str) -> str:
    return f"{feature}__ticker_{transform}"


def apply_ticker_time_transform(
    values: np.ndarray,
    transform: str,
) -> np.ndarray:
    if transform == "z20":
        return rolling_z_score(values, 20)
    if transform == "z60":
        return rolling_z_score(values, 60)
    if transform == "rank60":
        return rolling_percentile_rank(values, 60)
    if transform == "change5":
        return lag_change(values, 5)
    if transform == "change20":
        return lag_change(values, 20)
    if transform == "ratio_change5":
        return lag_ratio_change(values, 5)
    raise KeyError(f"Unknown ticker transform: {transform}")


def pairwise_valid_correlation(
    left: np.ndarray,
    right: np.ndarray,
    method: str = "spearman",
    minimum_rows: int = 30,
) -> tuple[float, int]:
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    if n < int(minimum_rows):
        return float("nan"), n
    x = x[mask]
    y = y[mask]
    if np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan"), n
    if method == "pearson":
        return float(np.corrcoef(x, y)[0, 1]), n
    if method == "spearman":
        result = scipy_spearmanr(x, y, nan_policy="omit")
        return float(result.statistic), n
    raise ValueError(f"Unsupported correlation method: {method}")


def compute_ticker_correlation_matrix(
    frame: pd.DataFrame,
    features: Sequence[str],
    *,
    method: str = "spearman",
    minimum_rows: int = 30,
    shrinkage_strength: float = 60.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_list = list(features)
    numeric = frame[feature_list].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if method == "spearman":
        numeric_for_corr = numeric.rank(method="average", pct=False, na_option="keep")
    elif method == "pearson":
        numeric_for_corr = numeric
    else:
        raise ValueError(f"Unsupported correlation method: {method}")
    correlation = numeric_for_corr.corr(method="pearson", min_periods=int(minimum_rows))
    finite = numeric.notna().to_numpy(dtype=np.int32)
    counts = finite.T @ finite
    values = correlation.to_numpy(dtype=float)
    reliability = np.maximum(counts - 3, 0) / np.maximum(counts - 3 + float(shrinkage_strength), 1e-12)
    values = values * reliability
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(values, 1.0)
    return (
        pd.DataFrame(values.astype(np.float32), index=feature_list, columns=feature_list),
        pd.DataFrame(counts.astype(np.int32), index=feature_list, columns=feature_list),
    )


def cluster_from_correlation(
    correlation: pd.DataFrame,
    threshold: float = 0.92,
) -> pd.DataFrame:
    features = correlation.index.astype(str).tolist()
    if not features:
        return pd.DataFrame(columns=["feature", "cluster_id", "cluster_size"])
    if len(features) == 1:
        return pd.DataFrame({"feature": features, "cluster_id": [1], "cluster_size": [1]})
    values = correlation.to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(np.abs(values), 0.0, 1.0)
    np.fill_diagonal(values, 1.0)
    distance = np.clip(1.0 - values, 0.0, 1.0)
    np.fill_diagonal(distance, 0.0)
    condensed = scipy_squareform(distance, checks=False)
    linkage_matrix = scipy_linkage(condensed, method="average")
    labels = scipy_fcluster(linkage_matrix, t=max(1.0 - float(threshold), 0.0), criterion="distance")
    assignments = pd.DataFrame({"feature": features, "cluster_id": labels.astype(int)})
    sizes = assignments.groupby("cluster_id")["feature"].size().rename("cluster_size")
    assignments = assignments.merge(sizes, on="cluster_id", how="left")
    return assignments.sort_values(["cluster_id", "feature"], kind="mergesort").reset_index(drop=True)


def stable_correlation_edges(
    full_correlation: pd.DataFrame,
    fold_correlations: Mapping[int, pd.DataFrame],
    fold_roles: Mapping[int, str],
    *,
    threshold: float = 0.92,
    minimum_fold_abs_correlation: float = 0.80,
    minimum_sign_consistency: float = 0.80,
    maximum_edges: int = 5000,
) -> pd.DataFrame:
    features = full_correlation.index.astype(str).tolist()
    rows: list[dict[str, Any]] = []
    matrix = full_correlation.to_numpy(dtype=float)
    for left_index, left in enumerate(features):
        for right_index in range(left_index + 1, len(features)):
            full_value = float(matrix[left_index, right_index])
            if not math.isfinite(full_value) or abs(full_value) < float(threshold):
                continue
            fold_values: list[float] = []
            selection_values: list[float] = []
            for fold_id, fold_matrix in fold_correlations.items():
                value = float(fold_matrix.loc[left, features[right_index]])
                if not math.isfinite(value):
                    continue
                fold_values.append(value)
                if fold_roles.get(int(fold_id)) == "selection":
                    selection_values.append(value)
            if selection_values:
                dominant_sign = 1 if np.nanmean(selection_values) >= 0 else -1
                sign_consistency = float(np.mean([1 if np.sign(v) == dominant_sign else 0 for v in selection_values]))
                minimum_selection_abs = float(np.nanmin(np.abs(selection_values)))
            else:
                dominant_sign = 0
                sign_consistency = float("nan")
                minimum_selection_abs = float("nan")
            stable = bool(
                selection_values
                and sign_consistency >= float(minimum_sign_consistency)
                and minimum_selection_abs >= float(minimum_fold_abs_correlation)
            )
            rows.append(
                {
                    "feature_a": left,
                    "feature_b": features[right_index],
                    "full_correlation": full_value,
                    "full_abs_correlation": abs(full_value),
                    "selection_min_abs_correlation": minimum_selection_abs,
                    "selection_sign_consistency": sign_consistency,
                    "stable_edge": stable,
                    "fold_count": len(fold_values),
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(
        ["stable_edge", "full_abs_correlation", "selection_min_abs_correlation"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    if int(maximum_edges) > 0:
        result = result.head(int(maximum_edges))
    return result.reset_index(drop=True)


def expected_calibration_error(
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> float:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = np.clip(p[mask], 0.0, 1.0)
    if y.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    bucket = np.clip(np.digitize(p, edges[1:-1], right=False), 0, int(bins) - 1)
    total = float(len(y))
    ece = 0.0
    for index in range(int(bins)):
        current = bucket == index
        if not np.any(current):
            continue
        ece += (float(np.sum(current)) / total) * abs(float(np.mean(p[current])) - float(np.mean(y[current])))
    return float(ece)


def probability_metrics(target: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=int)
    p = np.asarray(probability, dtype=float)
    mask = np.isfinite(p) & np.isfinite(y)
    y = y[mask]
    p = np.clip(p[mask], 1e-8, 1.0 - 1e-8)
    if y.size == 0:
        return {"brier": float("nan"), "logloss": float("nan"), "ece": float("nan")}
    return {
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "ece": expected_calibration_error(y, p),
    }


def select_threshold_for_precision(
    target: np.ndarray,
    score: np.ndarray,
    *,
    target_precision: float = 0.70,
    minimum_alerts: int = 3,
    minimum_recall: float = 0.0,
    minimum_wilson_lcb: float = 0.0,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    y = y[mask]
    s = s[mask]
    if y.size == 0:
        return {"threshold": float("inf"), "gate_pass": False, "alerts": 0, "precision": float("nan"), "recall": 0.0}
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    s_sorted = s[order]
    tp = np.cumsum(y_sorted == 1)
    alerts = np.arange(1, len(y_sorted) + 1)
    fp = alerts - tp
    precision = tp / alerts
    positives = int(np.sum(y_sorted == 1))
    recall = tp / max(positives, 1)
    best_index: int | None = None
    for index in range(len(alerts)):
        if alerts[index] < int(minimum_alerts):
            continue
        lcb = wilson_lower_bound(int(tp[index]), int(alerts[index]))
        if (
            precision[index] >= float(target_precision)
            and recall[index] >= float(minimum_recall)
            and lcb >= float(minimum_wilson_lcb)
        ):
            if best_index is None:
                best_index = index
            else:
                current_key = (recall[index], alerts[index], precision[index], lcb)
                best_key = (recall[best_index], alerts[best_index], precision[best_index], wilson_lower_bound(int(tp[best_index]), int(alerts[best_index])))
                if current_key > best_key:
                    best_index = index
    practical_index = None
    for index in range(len(alerts)):
        if alerts[index] < int(minimum_alerts):
            continue
        if practical_index is None or (precision[index], recall[index], alerts[index]) > (
            precision[practical_index], recall[practical_index], alerts[practical_index]
        ):
            practical_index = index
    if best_index is None:
        payload = {
            "threshold": float("inf"),
            "gate_pass": False,
            "alerts": 0,
            "tp": 0,
            "fp": 0,
            "precision": float("nan"),
            "recall": 0.0,
            "wilson_lcb": float("nan"),
        }
    else:
        threshold = float(s_sorted[best_index])
        payload = {
            "threshold": threshold,
            "gate_pass": True,
            "alerts": int(alerts[best_index]),
            "tp": int(tp[best_index]),
            "fp": int(fp[best_index]),
            "precision": float(precision[best_index]),
            "recall": float(recall[best_index]),
            "wilson_lcb": float(wilson_lower_bound(int(tp[best_index]), int(alerts[best_index]))),
        }
    if practical_index is not None:
        payload.update(
            {
                "best_practical_threshold": float(s_sorted[practical_index]),
                "best_practical_alerts": int(alerts[practical_index]),
                "best_practical_tp": int(tp[practical_index]),
                "best_practical_fp": int(fp[practical_index]),
                "best_practical_precision": float(precision[practical_index]),
                "best_practical_recall": float(recall[practical_index]),
                "best_practical_wilson_lcb": float(wilson_lower_bound(int(tp[practical_index]), int(alerts[practical_index]))),
            }
        )
    return payload


def percentile_rank_1d(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average", pct=True).to_numpy(dtype=float)


def build_ticker_error_group_codes(
    target: np.ndarray,
    base_score: np.ndarray,
    *,
    top_quantile: float = 0.80,
    low_quantile: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.5 < float(top_quantile) < 1.0:
        raise ValueError("top_quantile must be in (0.5, 1.0)")
    if not 0.0 < float(low_quantile) < float(top_quantile):
        raise ValueError("low_quantile must be in (0, top_quantile)")
    y = np.asarray(target, dtype=np.int8)
    score = np.asarray(base_score, dtype=float)
    rank = percentile_rank_1d(score)
    codes = np.zeros(len(y), dtype=np.int8)
    finite = np.isfinite(score) & np.isfinite(rank) & np.isin(y, [0, 1])
    high = finite & (rank >= float(top_quantile))
    low = finite & (rank <= float(low_quantile))
    codes[high & (y == 1)] = ERROR_A_TOP_TP
    codes[high & (y == 0)] = ERROR_B_TOP_FP
    codes[low & (y == 1)] = ERROR_C_LOW_TP
    codes[low & (y == 0)] = ERROR_D_LOW_TN
    return codes, rank


def build_ticker_temporal_matched_pairs(
    frame: pd.DataFrame,
    *,
    fold_id: int,
    ticker: str,
    axis: str,
    case_code: int,
    control_code: int,
    controls_per_case: int = 3,
    date_column: str = "date",
    score_column: str = "base_rank",
    regime_columns: Sequence[str] | None = None,
    score_weight: float = 1.0,
    time_weight: float = 0.10,
    regime_weight: float = 0.25,
    maximum_day_distance: int = 756,
) -> pd.DataFrame:
    subset = frame.loc[frame["error_code"].isin([int(case_code), int(control_code)])].copy()
    columns = [
        "fold_id",
        "ticker",
        "axis",
        "case_index",
        "control_index",
        "distance",
        "day_distance",
        "score_distance",
        "regime_distance",
    ]
    if subset.empty:
        return pd.DataFrame(columns=columns)
    subset[date_column] = pd.to_datetime(subset[date_column], errors="coerce")
    cases = subset.loc[subset["error_code"].eq(int(case_code))]
    controls = subset.loc[subset["error_code"].eq(int(control_code))]
    if cases.empty or controls.empty:
        return pd.DataFrame(columns=columns)
    regime_columns = [column for column in (regime_columns or []) if column in subset.columns]
    control_dates = controls[date_column].to_numpy(dtype="datetime64[D]")
    control_scores = pd.to_numeric(controls[score_column], errors="coerce").to_numpy(dtype=float)
    if regime_columns:
        control_regime = controls[regime_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        medians = np.nanmedian(control_regime, axis=0)
        scales = np.nanmedian(np.abs(control_regime - medians), axis=0)
        scales = np.where(np.isfinite(scales) & (scales > 1e-9), scales, 1.0)
    else:
        control_regime = np.empty((len(controls), 0), dtype=float)
        medians = np.empty(0, dtype=float)
        scales = np.empty(0, dtype=float)
    records: list[dict[str, Any]] = []
    for case_index, case_row in cases.iterrows():
        case_date = np.datetime64(pd.Timestamp(case_row[date_column]).date(), "D")
        day_distance = np.abs((control_dates - case_date).astype("timedelta64[D]").astype(int)).astype(float)
        score_distance = np.abs(control_scores - float(case_row[score_column]))
        if regime_columns:
            case_regime = pd.to_numeric(case_row[regime_columns], errors="coerce").to_numpy(dtype=float)
            diff = np.abs((control_regime - case_regime) / scales)
            regime_distance = np.nanmean(diff, axis=1)
            regime_distance = np.nan_to_num(regime_distance, nan=1.0, posinf=2.0, neginf=2.0)
        else:
            regime_distance = np.zeros(len(controls), dtype=float)
        normalized_time = np.minimum(day_distance, float(maximum_day_distance)) / max(float(maximum_day_distance), 1.0)
        distance = (
            float(score_weight) * np.nan_to_num(score_distance, nan=1.0)
            + float(time_weight) * normalized_time
            + float(regime_weight) * regime_distance
        )
        valid = np.isfinite(distance)
        if maximum_day_distance > 0:
            valid &= day_distance <= float(maximum_day_distance)
        candidate_positions = np.flatnonzero(valid)
        if candidate_positions.size == 0:
            candidate_positions = np.flatnonzero(np.isfinite(distance))
        if candidate_positions.size == 0:
            continue
        order_local = candidate_positions[np.argsort(distance[candidate_positions], kind="mergesort")]
        selected_positions = order_local[: min(int(controls_per_case), len(order_local))]
        for position in selected_positions:
            control_index = int(controls.index[position])
            records.append(
                {
                    "fold_id": int(fold_id),
                    "ticker": str(ticker),
                    "axis": str(axis),
                    "case_index": int(case_index),
                    "control_index": control_index,
                    "distance": float(distance[position]),
                    "day_distance": int(day_distance[position]),
                    "score_distance": float(score_distance[position]),
                    "regime_distance": float(regime_distance[position]),
                }
            )
    return pd.DataFrame(records, columns=columns)


def ticker_precision_policy_metrics(
    frame: pd.DataFrame,
    *,
    ticker_column: str,
    target_column: str,
    score_column: str,
    target_precision: float,
    minimum_alerts_per_ticker: int,
    minimum_recall_per_ticker: float = 0.0,
    minimum_wilson_lcb: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    policy_rows: list[dict[str, Any]] = []
    alert_rows: list[pd.DataFrame] = []
    for ticker, part in frame.groupby(ticker_column, sort=True):
        metrics = select_threshold_for_precision(
            part[target_column].to_numpy(dtype=int),
            part[score_column].to_numpy(dtype=float),
            target_precision=target_precision,
            minimum_alerts=minimum_alerts_per_ticker,
            minimum_recall=minimum_recall_per_ticker,
            minimum_wilson_lcb=minimum_wilson_lcb,
        )
        record = {"ticker": str(ticker), **metrics}
        policy_rows.append(record)
        scored = part.copy()
        threshold = float(metrics.get("threshold", float("inf")))
        scored["ticker_threshold"] = threshold
        scored["ticker_alert"] = np.isfinite(scored[score_column]) & (scored[score_column] >= threshold)
        alert_rows.append(scored)
    policies = pd.DataFrame(policy_rows)
    alerts = pd.concat(alert_rows, ignore_index=True) if alert_rows else pd.DataFrame()
    return policies, alerts
