from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
except Exception as exc:  # pragma: no cover - dependency error is reported at runtime
    raise RuntimeError("scikit-learn이 필요합니다. requirements_surge_ablation.txt를 설치하세요.") from exc


SCHEMA_VERSION = "crashwatch_surge_ablation_v3"
TARGET_COLUMN_DEFAULT = "label_abs_surge_3d_5pct"
TARGET_VALID_COLUMN_DEFAULT = "target_valid"
DATE_COLUMN_DEFAULT = "date"
TICKER_COLUMN_DEFAULT = "ticker"

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
    "lead_",
    "t+1",
    "t+2",
    "t+3",
)

# These are historical peer/network descriptors whose names contain "lead_",
# not forward-looking target columns.  They are part of the sealed 439-feature
# universe and passed the V2 leakage audit, so the broad name guard must not
# reject them solely because of that token.
LEAKAGE_NAME_ALLOWLIST = frozenset(
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
class TaskSpec:
    backend: str
    stage: str
    test_type: str
    condition_id: str
    fold_id: int
    seed: int
    enabled_feature_indices: tuple[int, ...]
    dropped_features: tuple[str, ...]
    representative_feature: str | None = None
    cluster_id: int | None = None
    feature_group: str | None = None
    profile_name: str | None = None


class FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.acquired = False

    def __enter__(self) -> "FileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            owner = ""
            with contextlib.suppress(Exception):
                owner = self.path.read_text(encoding="utf-8")
            raise RuntimeError(f"다른 실행이 lock을 보유하고 있습니다: {self.path}\n{owner}") from exc
        payload = {
            "pid": os.getpid(),
            "created_at": utc_now(),
            "host": platform.node(),
            "argv": sys.argv,
        }
        os.write(fd, stable_json_bytes(payload))
        os.close(fd)
        self.acquired = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.acquired:
            with contextlib.suppress(FileNotFoundError):
                self.path.unlink()
            self.acquired = False


class RunStatus:
    def __init__(self, output_dir: Path, run_name: str):
        self.output_dir = output_dir
        self.run_name = run_name
        self.path = output_dir / "RUN_STATUS.json"
        self.started = time.monotonic()
        self.payload: dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "run_name": run_name,
            "status": "RUNNING",
            "started_at": utc_now(),
            "stages": {},
        }
        atomic_write_json(self.path, self.payload)

    def stage(self, name: str, status: str, **details: Any) -> None:
        self.payload["stages"][name] = {
            "status": status,
            "updated_at": utc_now(),
            **details,
        }
        self.payload["elapsed_seconds"] = time.monotonic() - self.started
        atomic_write_json(self.path, self.payload)

    def success(self, **details: Any) -> None:
        self.payload.update(
            {
                "status": "SUCCESS",
                "completed_at": utc_now(),
                "elapsed_seconds": time.monotonic() - self.started,
                **details,
            }
        )
        atomic_write_json(self.path, self.payload)

    def failure(self, exc: BaseException) -> None:
        self.payload.update(
            {
                "status": "FAILED",
                "failed_at": utc_now(),
                "elapsed_seconds": time.monotonic() - self.started,
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


def stable_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default).encode("utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"JSON 변환 불가 타입: {type(value)!r}")


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
    atomic_write_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default).encode("utf-8"))


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
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
        raise ValueError(f"boolean으로 변환할 수 없는 값: {unknown[:10]}")
    return normalized.isin(truthy)


def read_table(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet 입력을 읽으려면 pyarrow가 필요합니다. "
                "python -m pip install -r starter_code/requirements_surge_ablation.txt 를 실행하세요."
            ) from exc
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, usecols=list(columns) if columns is not None else None)
    if suffixes.endswith(".feather"):
        return pd.read_feather(path, columns=list(columns) if columns is not None else None)
    if suffixes.endswith(".pkl") or suffixes.endswith(".pickle"):
        frame = pd.read_pickle(path)
        if columns is not None:
            frame = frame[list(columns)]
        return frame
    raise ValueError(f"지원하지 않는 입력 형식: {path}")


def table_columns(path: Path) -> list[str]:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet schema 확인에 pyarrow가 필요합니다.") from exc
        return list(pq.ParquetFile(path).schema_arrow.names)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return list(pd.read_csv(path, nrows=0).columns)
    return list(read_table(path).columns)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def parse_int_list(text: str | Sequence[int]) -> list[int]:
    if isinstance(text, str):
        if not text.strip():
            return []
        return [int(token.strip()) for token in text.split(",") if token.strip()]
    return [int(value) for value in text]


def parse_float_list(text: str | Sequence[float]) -> list[float]:
    if isinstance(text, str):
        if not text.strip():
            return []
        return [float(token.strip()) for token in text.split(",") if token.strip()]
    return [float(value) for value in text]


def load_folds(path: Path) -> list[FoldSpec]:
    raw = load_json(path)
    if isinstance(raw, Mapping) and "folds" in raw:
        raw = raw["folds"]
    if not isinstance(raw, list):
        raise ValueError(f"fold JSON은 list여야 합니다: {path}")
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


def validate_folds(folds: Sequence[FoldSpec], minimum_purge_trading_days: int = 3, all_dates: Sequence[pd.Timestamp] | None = None) -> None:
    ids = [fold.fold_id for fold in folds]
    if len(ids) != len(set(ids)):
        raise ValueError(f"중복 fold ID: {ids}")
    date_index: pd.DatetimeIndex | None = None
    if all_dates is not None:
        date_index = pd.DatetimeIndex(pd.to_datetime(pd.Series(all_dates).dropna().unique())).sort_values()
    for fold in folds:
        if fold.train_start > fold.train_end:
            raise ValueError(f"fold {fold.fold_id}: train_start > train_end")
        if fold.validation_start > fold.validation_end:
            raise ValueError(f"fold {fold.fold_id}: validation_start > validation_end")
        if fold.train_end >= fold.validation_start:
            raise ValueError(f"fold {fold.fold_id}: train/validation overlap")
        if (fold.purge_start is None) != (fold.purge_end is None):
            raise ValueError(f"fold {fold.fold_id}: purge_start/purge_end가 한쪽만 존재")
        if fold.purge_start is not None and fold.purge_end is not None:
            if fold.purge_start <= fold.train_end:
                raise ValueError(f"fold {fold.fold_id}: purge가 train을 침범")
            if fold.purge_end >= fold.validation_start:
                raise ValueError(f"fold {fold.fold_id}: purge가 validation을 침범")
        if date_index is not None:
            between = date_index[(date_index > fold.train_end) & (date_index < fold.validation_start)]
            if len(between) < minimum_purge_trading_days:
                raise ValueError(
                    f"fold {fold.fold_id}: 실제 purge 거래일 {len(between)} < {minimum_purge_trading_days}"
                )




def validate_role_assignments(
    roles: Mapping[str, Sequence[int]],
    folds: Sequence[FoldSpec],
    require_selection: bool = True,
) -> None:
    """Validate that role fold IDs exist and that no fold leaks across roles."""
    known_fold_ids = {int(fold.fold_id) for fold in folds}
    ownership: dict[int, str] = {}
    for role, raw_ids in roles.items():
        for raw_id in raw_ids:
            fold_id = int(raw_id)
            if fold_id not in known_fold_ids:
                raise ValueError(f"role {role!r}에 존재하지 않는 fold ID가 있습니다: {fold_id}")
            previous = ownership.get(fold_id)
            if previous is not None and previous != str(role):
                raise ValueError(
                    f"fold {fold_id}가 여러 역할에 중복 배정됐습니다: {previous!r}, {role!r}"
                )
            ownership[fold_id] = str(role)
    if require_selection and not [int(value) for value in roles.get("selection", [])]:
        raise ValueError("selection fold가 비어 있습니다")


def role_for_fold(fold_id: int, roles: Mapping[str, Sequence[int]]) -> str:
    for role, fold_ids in roles.items():
        if int(fold_id) in {int(value) for value in fold_ids}:
            return str(role)
    return "unassigned"


def load_roles_from_manifest(manifest_path: Path | None, defaults: Mapping[str, Sequence[int]] | None = None) -> dict[str, list[int]]:
    if manifest_path is not None and manifest_path.exists():
        manifest = load_json(manifest_path)
        roles = manifest.get("fold_roles")
        if isinstance(roles, Mapping):
            return {str(key): [int(value) for value in values] for key, values in roles.items()}
    defaults = defaults or {"selection": [0, 1, 2, 3, 4], "confirmation": [5, 6], "recent_audit": [7]}
    return {str(key): [int(value) for value in values] for key, values in defaults.items()}


def leakage_reason(column: str) -> str | None:
    normalized = str(column).strip().lower()
    if normalized in LEAKAGE_NAME_ALLOWLIST:
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
        raise ValueError(f"데이터에 없는 피처 {len(missing)}개: {missing[:20]}")
    duplicates = pd.Series(list(features)).duplicated(keep=False)
    if duplicates.any():
        repeated = pd.Series(list(features))[duplicates].tolist()
        raise ValueError(f"중복 피처: {repeated[:20]}")
    blocked = [(feature, leakage_reason(feature)) for feature in features if leakage_reason(feature) is not None]
    if blocked:
        raise ValueError(f"누수/메타 피처가 feature list에 포함됨: {blocked[:20]}")


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
            raise ValueError("target sidecar에 source_row_id가 없고 행 수도 원본과 다릅니다")
        target_sidecar.insert(0, "source_row_id", np.arange(len(target_sidecar), dtype=np.int64))
    if source["source_row_id"].duplicated().any() or target_sidecar["source_row_id"].duplicated().any():
        raise ValueError("source_row_id가 일대일 키가 아닙니다")
    required = {"source_row_id", target_column, target_valid_column}
    missing = required - set(target_sidecar.columns)
    if missing:
        raise ValueError(f"target sidecar 필수 열 누락: {sorted(missing)}")
    carry = ["source_row_id", target_column, target_valid_column]
    for column in (date_column, ticker_column):
        if column in target_sidecar.columns:
            carry.append(column)
    side = target_sidecar[carry].rename(
        columns={
            date_column: f"{date_column}__target_sidecar",
            ticker_column: f"{ticker_column}__target_sidecar",
        }
    )
    joined = source.merge(side, on="source_row_id", how="left", validate="one_to_one", sort=False)
    if len(joined) != len(source):
        raise ValueError("target join 후 행 수가 변했습니다")
    if joined[target_column].isna().all():
        raise ValueError("target join 결과가 전부 결측입니다")
    side_date = f"{date_column}__target_sidecar"
    if date_column in joined.columns and side_date in joined.columns:
        left = pd.to_datetime(joined[date_column], errors="coerce")
        right = pd.to_datetime(joined[side_date], errors="coerce")
        mismatch = left.notna() & right.notna() & left.ne(right)
        if mismatch.any():
            raise ValueError(f"date join mismatch: {int(mismatch.sum())}행")
        joined.drop(columns=[side_date], inplace=True)
    side_ticker = f"{ticker_column}__target_sidecar"
    if ticker_column in joined.columns and side_ticker in joined.columns:
        left = joined[ticker_column].astype("string").str.strip()
        right = joined[side_ticker].astype("string").str.strip()
        mismatch = left.notna() & right.notna() & left.ne(right)
        if mismatch.any():
            raise ValueError(f"ticker join mismatch: {int(mismatch.sum())}행")
        joined.drop(columns=[side_ticker], inplace=True)
    joined[target_valid_column] = parse_bool_series(joined[target_valid_column])
    joined[target_column] = pd.to_numeric(joined[target_column], errors="coerce")
    invalid_label = joined[target_valid_column] & ~joined[target_column].isin([0, 1])
    if invalid_label.any():
        raise ValueError(f"target_valid 행 중 binary label이 아닌 행: {int(invalid_label.sum())}")
    return joined


def finite_pair_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xv = x[mask]
    yv = y[mask]
    if np.nanstd(xv) <= 1e-15 or np.nanstd(yv) <= 1e-15:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def residualize_by_group(values: np.ndarray, groups: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=np.float64))
    grouped = pd.Series(groups)
    return (series - series.groupby(grouped, sort=False).transform("mean")).to_numpy(dtype=np.float64)


def target_metric_for_feature(
    values: np.ndarray,
    target: np.ndarray,
    metric: str,
    dates: np.ndarray | None = None,
    tickers: np.ndarray | None = None,
) -> float:
    if metric == "pearson":
        return finite_pair_corr(values, target)
    if metric == "spearman":
        x_rank = pd.Series(values).rank(method="average", na_option="keep").to_numpy(dtype=np.float64)
        y_rank = pd.Series(target).rank(method="average", na_option="keep").to_numpy(dtype=np.float64)
        return finite_pair_corr(x_rank, y_rank)
    if metric == "within_date_pearson":
        if dates is None:
            return float("nan")
        return finite_pair_corr(residualize_by_group(values, dates), residualize_by_group(target, dates))
    if metric == "within_ticker_pearson":
        if tickers is None:
            return float("nan")
        return finite_pair_corr(residualize_by_group(values, tickers), residualize_by_group(target, tickers))
    raise ValueError(f"지원하지 않는 metric: {metric}")


def safe_binary_metrics(target: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(target, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(p) & np.isin(y, [0, 1])
    y = y[mask]
    p = np.clip(p[mask], 1e-7, 1.0 - 1e-7)
    rows = int(len(y))
    positives = int(np.sum(y == 1))
    negatives = rows - positives
    positive_rate = float(positives / rows) if rows else float("nan")
    result: dict[str, float | int] = {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": positive_rate,
        "mean_prediction": float(np.mean(p)) if rows else float("nan"),
        "pr_auc": float("nan"),
        "roc_auc": float("nan"),
        "brier": float("nan"),
        "logloss": float("nan"),
        "pr_auc_lift": float("nan"),
        "ece_10": float("nan"),
    }
    if rows == 0:
        return result
    result["brier"] = float(brier_score_loss(y, p))
    result["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    result["ece_10"] = expected_calibration_error(y, p, bins=10)
    if positives > 0 and negatives > 0:
        result["pr_auc"] = float(average_precision_score(y, p))
        result["roc_auc"] = float(roc_auc_score(y, p))
        if positive_rate > 0:
            result["pr_auc_lift"] = float(result["pr_auc"] / positive_rate)
    return result


def expected_calibration_error(target: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    if len(y) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = float(len(y))
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        count = int(mask.sum())
        if count:
            value += (count / total) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(value)


def daily_top_fraction_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    dates: np.ndarray,
    fraction: float,
) -> dict[str, float | int]:
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction 범위 오류: {fraction}")
    y = np.asarray(target, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    d = pd.to_datetime(pd.Series(dates), errors="coerce").to_numpy(dtype="datetime64[ns]")
    valid = np.isfinite(p) & np.isin(y, [0, 1]) & ~pd.isna(d)
    y = y[valid]
    p = p[valid]
    d = d[valid]
    rows = len(y)
    positives = int(np.sum(y == 1))
    base_rate = float(positives / rows) if rows else float("nan")
    selected = np.zeros(rows, dtype=bool)
    if rows:
        date_codes, unique_dates = pd.factorize(d, sort=True)
        for code in range(len(unique_dates)):
            indices = np.flatnonzero(date_codes == code)
            count = max(1, int(math.ceil(len(indices) * fraction)))
            local_order = np.argsort(-p[indices], kind="mergesort")[:count]
            selected[indices[local_order]] = True
    alerts = int(selected.sum())
    alert_positives = int(np.sum(y[selected] == 1)) if alerts else 0
    precision = float(alert_positives / alerts) if alerts else float("nan")
    recall = float(alert_positives / positives) if positives else float("nan")
    unique_date_count = int(pd.Series(d).nunique()) if rows else 0
    annualized_alerts = float(alerts / unique_date_count * 250.0) if unique_date_count else float("nan")
    return {
        "fraction": float(fraction),
        "alert_rows": alerts,
        "alert_positives": alert_positives,
        "precision": precision,
        "recall": recall,
        "lift": float(precision / base_rate) if base_rate > 0 and np.isfinite(precision) else float("nan"),
        "base_rate": base_rate,
        "date_count": unique_date_count,
        "annualized_alerts": annualized_alerts,
    }


def pooled_top_fraction_metrics(target: np.ndarray, scores: np.ndarray, fraction: float) -> dict[str, float | int]:
    y = np.asarray(target, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(p) & np.isin(y, [0, 1])
    y = y[valid]
    p = p[valid]
    rows = len(y)
    positives = int(np.sum(y == 1))
    base_rate = float(positives / rows) if rows else float("nan")
    count = max(1, int(math.ceil(rows * fraction))) if rows else 0
    if count:
        order = np.argsort(-p, kind="mergesort")[:count]
        alert_positives = int(np.sum(y[order] == 1))
    else:
        alert_positives = 0
    precision = float(alert_positives / count) if count else float("nan")
    recall = float(alert_positives / positives) if positives else float("nan")
    return {
        "fraction": float(fraction),
        "alert_rows": int(count),
        "alert_positives": alert_positives,
        "precision": precision,
        "recall": recall,
        "lift": float(precision / base_rate) if base_rate > 0 and np.isfinite(precision) else float("nan"),
        "base_rate": base_rate,
    }


def evaluate_prediction_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    dates: np.ndarray,
    top_fractions: Sequence[float],
) -> dict[str, Any]:
    result: dict[str, Any] = safe_binary_metrics(target, scores)
    for fraction in top_fractions:
        label = fraction_label(fraction)
        daily = daily_top_fraction_metrics(target, scores, dates, fraction)
        pooled = pooled_top_fraction_metrics(target, scores, fraction)
        for key, value in daily.items():
            if key != "fraction":
                result[f"daily_top_{label}_{key}"] = value
        for key, value in pooled.items():
            if key != "fraction":
                result[f"pooled_top_{label}_{key}"] = value
    return result


def fraction_label(fraction: float) -> str:
    percent = fraction * 100.0
    if abs(percent - round(percent)) < 1e-9:
        return f"{int(round(percent))}pct"
    return f"{percent:g}pct".replace(".", "p")


def sign_consistency(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array) & (np.abs(array) > 1e-12)]
    if len(array) == 0:
        return float("nan")
    positive = float(np.mean(array > 0))
    negative = float(np.mean(array < 0))
    return max(positive, negative)


def exact_sign_flip_p(values: Sequence[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    n = len(array)
    if n == 0:
        return float("nan")
    observed = abs(float(np.mean(array)))
    if n <= 20:
        count = 0
        total = 1 << n
        magnitudes = np.abs(array)
        for mask in range(total):
            signs = np.fromiter((1.0 if (mask >> bit) & 1 else -1.0 for bit in range(n)), dtype=np.float64, count=n)
            statistic = abs(float(np.mean(magnitudes * signs)))
            if statistic >= observed - 1e-15:
                count += 1
        return float(count / total)
    rng = np.random.default_rng(20260809)
    permutations = 20000
    count = 0
    magnitudes = np.abs(array)
    for _ in range(permutations):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        if abs(float(np.mean(magnitudes * signs))) >= observed - 1e-15:
            count += 1
    return float((count + 1) / (permutations + 1))


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    seed: int,
    repetitions: int = 5000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return float("nan"), float("nan")
    if len(array) == 1:
        return float(array[0]), float(array[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repetitions, len(array)))
    means = np.mean(array[indices], axis=1)
    return float(np.quantile(means, alpha / 2.0)), float(np.quantile(means, 1.0 - alpha / 2.0))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not len(finite_indices):
        return pd.Series(result, index=p_values.index)
    finite_values = values[finite_indices]
    order = np.argsort(finite_values, kind="mergesort")
    ranked = finite_values[order]
    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    restored = np.empty(m, dtype=np.float64)
    restored[order] = adjusted
    result[finite_indices] = restored
    return pd.Series(result, index=p_values.index)


def aggregate_numeric(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if not len(finite):
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "positive_ratio": float("nan"),
            "negative_ratio": float("nan"),
            "sign_consistency": float("nan"),
        }
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=0)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "positive_ratio": float(np.mean(finite > 0)),
        "negative_ratio": float(np.mean(finite < 0)),
        "sign_consistency": sign_consistency(finite),
    }


def compute_output_inventory(
    output_dir: Path,
    names: Sequence[str] | None = None,
    exclude_relative_paths: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    excluded = {Path(value).as_posix() for value in (exclude_relative_paths or [])}
    if names is None:
        paths = sorted(path for path in output_dir.rglob("*") if path.is_file())
    else:
        paths = [output_dir / name for name in names if (output_dir / name).is_file()]
    records: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(output_dir).as_posix()
        if relative in excluded:
            continue
        records.append(
            {
                "relative_path": relative,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return records


def verify_output_inventory(output_dir: Path, inventory: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """Verify every listed artifact by existence, size, and SHA-256."""
    reasons: list[str] = []
    for record in inventory:
        relative = str(record.get("relative_path", ""))
        if not relative:
            reasons.append("inventory_record_without_relative_path")
            continue
        path = output_dir / relative
        if not path.exists():
            reasons.append(f"missing:{relative}")
            continue
        expected_bytes = record.get("bytes")
        if expected_bytes is not None and int(path.stat().st_size) != int(expected_bytes):
            reasons.append(f"size_mismatch:{relative}")
            continue
        expected_hash = record.get("sha256")
        if expected_hash and sha256_file(path) != str(expected_hash):
            reasons.append(f"sha256_mismatch:{relative}")
    return not reasons, reasons


def ensure_free_disk(path: Path, minimum_gb: float) -> None:
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    free_gb = usage.free / (1024**3)
    if free_gb < minimum_gb:
        raise RuntimeError(f"디스크 여유 공간 부족: {free_gb:.2f} GiB < {minimum_gb:.2f} GiB")


def model_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": sys.version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for name in ("sklearn", "lightgbm", "xgboost", "pyarrow"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = None
    return versions


def task_identity_hash(payload: Mapping[str, Any]) -> str:
    return sha256_bytes(stable_json_bytes(payload))[:24]


def with_payload_checksum(payload: Mapping[str, Any]) -> dict[str, Any]:
    resolved = dict(payload)
    resolved.pop("payload_sha256", None)
    resolved["payload_sha256"] = sha256_bytes(stable_json_bytes(resolved))
    return resolved


def payload_checksum_is_valid(payload: Mapping[str, Any]) -> bool:
    expected = payload.get("payload_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    resolved = dict(payload)
    resolved.pop("payload_sha256", None)
    return sha256_bytes(stable_json_bytes(resolved)) == expected


def result_file_is_valid(path: Path, expected_identity_hash: str) -> bool:
    if not path.exists() or path.stat().st_size <= 2:
        return False
    try:
        payload = load_json(path)
    except Exception:
        return False
    return (
        payload.get("status") == "completed"
        and payload.get("identity_hash") == expected_identity_hash
        and payload_checksum_is_valid(payload)
    )


def nan_to_none(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): nan_to_none(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [nan_to_none(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def deterministic_seed(*values: Any, base: int = 17) -> int:
    digest = hashlib.sha256(stable_json_bytes([base, *values])).digest()
    return int.from_bytes(digest[:4], "little", signed=False) & 0x7FFFFFFF
