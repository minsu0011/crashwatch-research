from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import os
import platform
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
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        log_loss,
        precision_recall_curve,
        roc_auc_score,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn이 필요합니다.") from exc


SCHEMA = "crashwatch_surge_model_zoo_v4"
TARGET_COLUMN = "label_abs_surge_3d_5pct"
TARGET_VALID_COLUMN = "target_valid"
DATE_COLUMN = "date"
TICKER_COLUMN = "ticker"


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
class ThresholdPolicy:
    kind: str
    threshold: float | None = None
    daily_fraction: float | None = None
    target_recall: float = 0.70
    achieved_recall: float | None = None
    achieved_precision: float | None = None
    achieved_alert_rate: float | None = None
    source: str = "selection_oof"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


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
            raise RuntimeError(f"실행 lock이 이미 존재합니다: {self.path}\n{owner}") from exc
        payload = {
            "pid": os.getpid(),
            "host": platform.node(),
            "created_at": utc_now(),
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
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.path = output_dir / "RUN_STATUS.json"
        self.started = time.monotonic()
        self.payload: dict[str, Any] = {
            "schema": SCHEMA,
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
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    raise TypeError(f"JSON 변환 불가 타입: {type(value)!r}")


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
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    text = series.astype("string").str.strip().str.lower()
    return text.isin(["1", "true", "t", "yes", "y"])


def read_table(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path, columns=list(columns) if columns is not None else None)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet 입력에는 pyarrow가 필요합니다. "
                "python -m pip install pyarrow 를 실행하세요."
            ) from exc
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path, usecols=list(columns) if columns is not None else None)
    if suffix in {".pkl", ".pickle"}:
        frame = pd.read_pickle(path)
        return frame.loc[:, list(columns)] if columns is not None else frame
    raise ValueError(f"지원하지 않는 표 형식: {path}")


def table_columns(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError("Parquet schema 확인에는 pyarrow가 필요합니다.") from exc
        return list(pq.ParquetFile(path).schema_arrow.names)
    if suffix in {".csv", ".txt"}:
        return list(pd.read_csv(path, nrows=0).columns)
    if suffix in {".pkl", ".pickle"}:
        return list(pd.read_pickle(path).columns)
    raise ValueError(f"지원하지 않는 표 형식: {path}")


def join_source_and_target(
    source: pd.DataFrame,
    sidecar: pd.DataFrame,
    target_column: str,
    target_valid_column: str,
    date_column: str,
    ticker_column: str,
) -> pd.DataFrame:
    source = source.copy()
    sidecar = sidecar.copy()
    if "source_row_id" not in source.columns:
        source.insert(0, "source_row_id", np.arange(len(source), dtype=np.int64))
    if "source_row_id" not in sidecar.columns:
        raise ValueError("target sidecar에 source_row_id가 없습니다")
    if source["source_row_id"].duplicated().any() or sidecar["source_row_id"].duplicated().any():
        raise ValueError("source_row_id가 일대일이 아닙니다")
    rename: dict[str, str] = {}
    for column in [date_column, ticker_column]:
        if column in sidecar.columns and column in source.columns:
            rename[column] = f"{column}__sidecar"
    sidecar.rename(columns=rename, inplace=True)
    joined = source.merge(sidecar, on="source_row_id", how="left", validate="one_to_one", sort=False)
    if len(joined) != len(source):
        raise ValueError("target join 후 행 수가 변했습니다")
    side_date = f"{date_column}__sidecar"
    if side_date in joined.columns:
        left = pd.to_datetime(joined[date_column], errors="coerce")
        right = pd.to_datetime(joined[side_date], errors="coerce")
        mismatch = left.notna() & right.notna() & left.ne(right)
        if mismatch.any():
            raise ValueError(f"sidecar date mismatch: {int(mismatch.sum())}행")
        joined.drop(columns=[side_date], inplace=True)
    side_ticker = f"{ticker_column}__sidecar"
    if side_ticker in joined.columns:
        left = joined[ticker_column].astype("string").str.strip()
        right = joined[side_ticker].astype("string").str.strip()
        mismatch = left.notna() & right.notna() & left.ne(right)
        if mismatch.any():
            raise ValueError(f"sidecar ticker mismatch: {int(mismatch.sum())}행")
        joined.drop(columns=[side_ticker], inplace=True)
    if target_column not in joined.columns or target_valid_column not in joined.columns:
        raise ValueError("target 또는 target_valid이 join되지 않았습니다")
    joined[target_valid_column] = parse_bool_series(joined[target_valid_column])
    joined[target_column] = pd.to_numeric(joined[target_column], errors="coerce")
    invalid = joined[target_valid_column] & ~joined[target_column].isin([0, 1])
    if invalid.any():
        raise ValueError(f"target_valid 행 중 binary가 아닌 label: {int(invalid.sum())}행")
    return joined


def load_folds(path: Path) -> list[FoldSpec]:
    payload = load_json(path)
    if isinstance(payload, dict):
        records = payload.get("folds", payload.get("outer_folds", payload.get("data", [])))
    else:
        records = payload
    if not isinstance(records, list) or not records:
        raise ValueError(f"fold JSON에서 fold 목록을 찾지 못했습니다: {path}")
    folds: list[FoldSpec] = []
    for record in records:
        folds.append(
            FoldSpec(
                fold_id=int(record["fold_id"]),
                train_start=pd.Timestamp(record["train_start"]),
                train_end=pd.Timestamp(record["train_end"]),
                purge_start=pd.Timestamp(record["purge_start"]) if record.get("purge_start") else None,
                purge_end=pd.Timestamp(record["purge_end"]) if record.get("purge_end") else None,
                validation_start=pd.Timestamp(record["validation_start"]),
                validation_end=pd.Timestamp(record["validation_end"]),
            )
        )
    ids = [fold.fold_id for fold in folds]
    if len(ids) != len(set(ids)):
        raise ValueError("fold_id가 중복되었습니다")
    return sorted(folds, key=lambda item: item.fold_id)


def validate_folds(folds: Sequence[FoldSpec], dates: Sequence[Any], minimum_purge_days: int = 3) -> None:
    unique_dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(dates), errors="coerce").dropna().unique()).sort_values()
    if len(unique_dates) == 0:
        raise ValueError("유효 거래일이 없습니다")
    for fold in folds:
        if fold.train_start > fold.train_end:
            raise ValueError(f"fold {fold.fold_id}: train 날짜 역전")
        if fold.validation_start > fold.validation_end:
            raise ValueError(f"fold {fold.fold_id}: validation 날짜 역전")
        if fold.train_end >= fold.validation_start:
            raise ValueError(f"fold {fold.fold_id}: train/validation 중첩")
        train_end_pos = int(unique_dates.searchsorted(fold.train_end, side="right") - 1)
        valid_start_pos = int(unique_dates.searchsorted(fold.validation_start, side="left"))
        gap = valid_start_pos - train_end_pos - 1
        if gap < minimum_purge_days:
            raise ValueError(
                f"fold {fold.fold_id}: purge 거래일 {gap} < 최소 {minimum_purge_days}"
            )


def role_for_fold(fold_id: int, roles: Mapping[str, Sequence[int]]) -> str:
    for role, ids in roles.items():
        if int(fold_id) in {int(value) for value in ids}:
            return role
    return "unassigned"


def deterministic_seed(*parts: Any) -> int:
    digest = sha256_bytes(stable_json_bytes(list(parts)))
    return int(digest[:8], 16) & 0x7FFFFFFF


def with_payload_checksum(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result


def payload_checksum_is_valid(payload: Mapping[str, Any]) -> bool:
    expected = payload.get("payload_sha256")
    if not isinstance(expected, str):
        return False
    clean = dict(payload)
    clean.pop("payload_sha256", None)
    return expected == sha256_bytes(stable_json_bytes(clean))


def safe_binary_metrics(target: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(target, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    mask = np.isfinite(p) & np.isin(y, [0, 1])
    y = y[mask]
    p = np.clip(p[mask], 1e-7, 1.0 - 1e-7)
    rows = int(len(y))
    positives = int(np.sum(y == 1))
    negatives = rows - positives
    rate = float(positives / rows) if rows else float("nan")
    result: dict[str, float | int] = {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "positive_rate": rate,
        "mean_score": float(np.mean(p)) if rows else float("nan"),
        "pr_auc": float("nan"),
        "pr_auc_lift": float("nan"),
        "roc_auc": float("nan"),
        "brier": float("nan"),
        "logloss": float("nan"),
        "ece_10": float("nan"),
    }
    if not rows:
        return result
    result["brier"] = float(brier_score_loss(y, p))
    result["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    result["ece_10"] = expected_calibration_error(y, p, bins=10)
    if positives > 0 and negatives > 0:
        result["pr_auc"] = float(average_precision_score(y, p))
        result["roc_auc"] = float(roc_auc_score(y, p))
        if rate > 0:
            result["pr_auc_lift"] = float(result["pr_auc"] / rate)
    return result


def expected_calibration_error(target: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(target, dtype=np.float64)
    p = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(p) & np.isin(y, [0, 1])
    y = y[valid]
    p = np.clip(p[valid], 0.0, 1.0)
    if not len(y):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        if index == bins - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        count = int(mask.sum())
        if count:
            value += (count / len(y)) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(value)


def _selected_metrics(target: np.ndarray, selected: np.ndarray) -> dict[str, float | int]:
    y = np.asarray(target, dtype=np.int8)
    selected = np.asarray(selected, dtype=bool)
    valid = np.isin(y, [0, 1])
    y = y[valid]
    selected = selected[valid]
    rows = int(len(y))
    positives = int(np.sum(y == 1))
    negatives = rows - positives
    alerts = int(selected.sum())
    tp = int(np.sum((y == 1) & selected))
    fp = int(np.sum((y == 0) & selected))
    fn = positives - tp
    tn = negatives - fp
    base_rate = float(positives / rows) if rows else float("nan")
    precision = float(tp / alerts) if alerts else float("nan")
    recall = float(tp / positives) if positives else float("nan")
    fpr = float(fp / negatives) if negatives else float("nan")
    specificity = float(tn / negatives) if negatives else float("nan")
    alert_rate = float(alerts / rows) if rows else float("nan")
    return {
        "rows": rows,
        "positives": positives,
        "negatives": negatives,
        "alerts": alerts,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "positive_rate": base_rate,
        "precision": precision,
        "recall": recall,
        "lift": float(precision / base_rate) if base_rate > 0 and np.isfinite(precision) else float("nan"),
        "alert_rate": alert_rate,
        "false_positive_rate": fpr,
        "specificity": specificity,
    }


def threshold_for_target_recall(
    target: np.ndarray,
    scores: np.ndarray,
    target_recall: float,
) -> tuple[float, dict[str, float | int]]:
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall은 (0, 1] 범위여야 합니다")
    y = np.asarray(target, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(p) & np.isin(y, [0, 1])
    y = y[valid]
    p = p[valid]
    positives = int(np.sum(y == 1))
    if not len(y) or positives == 0:
        return float("nan"), _selected_metrics(y, np.zeros(len(y), dtype=bool))
    order = np.argsort(-p, kind="mergesort")
    cumulative = np.cumsum(y[order] == 1)
    required = int(math.ceil(positives * target_recall - 1e-12))
    position = int(np.searchsorted(cumulative, required, side="left"))
    position = min(position, len(order) - 1)
    threshold = float(p[order[position]])
    selected = p >= threshold
    return threshold, _selected_metrics(y, selected)


def oracle_metrics_at_recall(target: np.ndarray, scores: np.ndarray, target_recall: float) -> dict[str, float | int]:
    threshold, metrics = threshold_for_target_recall(target, scores, target_recall)
    return {"threshold": threshold, **metrics}


def daily_fraction_selection(scores: np.ndarray, dates: np.ndarray, fraction: float) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("daily fraction은 (0,1] 범위여야 합니다")
    p = np.asarray(scores, dtype=np.float64)
    d = pd.to_datetime(pd.Series(dates), errors="coerce").to_numpy(dtype="datetime64[ns]")
    selected = np.zeros(len(p), dtype=bool)
    valid = np.isfinite(p) & ~pd.isna(d)
    if not valid.any():
        return selected
    valid_indices = np.flatnonzero(valid)
    date_codes, unique_dates = pd.factorize(d[valid], sort=True)
    for code in range(len(unique_dates)):
        local = valid_indices[date_codes == code]
        count = max(1, int(math.ceil(len(local) * fraction)))
        order = np.argsort(-p[local], kind="mergesort")[:count]
        selected[local[order]] = True
    return selected


def daily_fraction_for_target_recall(
    target: np.ndarray,
    scores: np.ndarray,
    dates: np.ndarray,
    target_recall: float,
    grid: Sequence[float],
) -> tuple[float, dict[str, float | int]]:
    candidates = sorted({float(value) for value in grid if 0 < float(value) <= 1.0})
    if not candidates:
        raise ValueError("daily fraction grid가 비어 있습니다")
    chosen = candidates[-1]
    chosen_metrics = _selected_metrics(target, daily_fraction_selection(scores, dates, chosen))
    for fraction in candidates:
        selected = daily_fraction_selection(scores, dates, fraction)
        metrics = _selected_metrics(target, selected)
        if np.isfinite(metrics["recall"]) and float(metrics["recall"]) >= target_recall:
            chosen = fraction
            chosen_metrics = metrics
            break
    return chosen, chosen_metrics


def choose_threshold_policy(
    target: np.ndarray,
    scores: np.ndarray,
    dates: np.ndarray,
    target_recall: float,
    daily_fraction_grid: Sequence[float],
    minimum_precision_lift: float = 1.0,
) -> ThresholdPolicy:
    threshold, global_metrics = threshold_for_target_recall(target, scores, target_recall)
    fraction, daily_metrics = daily_fraction_for_target_recall(
        target,
        scores,
        dates,
        target_recall,
        daily_fraction_grid,
    )
    candidates = [
        ("global_threshold", global_metrics, threshold, None),
        ("daily_fraction", daily_metrics, None, fraction),
    ]
    feasible = [
        item
        for item in candidates
        if np.isfinite(item[1]["recall"])
        and float(item[1]["recall"]) >= target_recall
        and np.isfinite(item[1]["lift"])
        and float(item[1]["lift"]) >= minimum_precision_lift
    ]
    pool = feasible if feasible else candidates
    chosen = min(
        pool,
        key=lambda item: (
            float(item[1]["alert_rate"]) if np.isfinite(item[1]["alert_rate"]) else 2.0,
            -float(item[1]["precision"]) if np.isfinite(item[1]["precision"]) else 0.0,
        ),
    )
    kind, metrics, threshold_value, fraction_value = chosen
    return ThresholdPolicy(
        kind=kind,
        threshold=float(threshold_value) if threshold_value is not None else None,
        daily_fraction=float(fraction_value) if fraction_value is not None else None,
        target_recall=float(target_recall),
        achieved_recall=float(metrics["recall"]) if np.isfinite(metrics["recall"]) else None,
        achieved_precision=float(metrics["precision"]) if np.isfinite(metrics["precision"]) else None,
        achieved_alert_rate=float(metrics["alert_rate"]) if np.isfinite(metrics["alert_rate"]) else None,
    )


def apply_threshold_policy(policy: ThresholdPolicy, scores: np.ndarray, dates: np.ndarray) -> np.ndarray:
    p = np.asarray(scores, dtype=np.float64)
    if policy.kind == "global_threshold":
        if policy.threshold is None:
            raise ValueError("global threshold 값이 없습니다")
        return np.isfinite(p) & (p >= float(policy.threshold))
    if policy.kind == "daily_fraction":
        if policy.daily_fraction is None:
            raise ValueError("daily fraction 값이 없습니다")
        return daily_fraction_selection(p, dates, float(policy.daily_fraction))
    if policy.kind == "hybrid_or":
        if policy.threshold is None or policy.daily_fraction is None:
            raise ValueError("hybrid policy 값이 불완전합니다")
        return (np.isfinite(p) & (p >= float(policy.threshold))) | daily_fraction_selection(
            p,
            dates,
            float(policy.daily_fraction),
        )
    raise ValueError(f"지원하지 않는 threshold policy: {policy.kind}")


def evaluate_threshold_policy(
    policy: ThresholdPolicy,
    target: np.ndarray,
    scores: np.ndarray,
    dates: np.ndarray,
) -> dict[str, float | int | str]:
    selected = apply_threshold_policy(policy, scores, dates)
    metrics = _selected_metrics(target, selected)
    valid_dates = pd.to_datetime(pd.Series(dates), errors="coerce")
    date_count = int(valid_dates.nunique())
    metrics["alerts_per_day"] = float(metrics["alerts"] / date_count) if date_count else float("nan")
    metrics["annualized_alerts"] = float(metrics["alerts"] / date_count * 250.0) if date_count else float("nan")
    metrics["policy_kind"] = policy.kind
    metrics["policy_threshold"] = policy.threshold if policy.threshold is not None else float("nan")
    metrics["policy_daily_fraction"] = policy.daily_fraction if policy.daily_fraction is not None else float("nan")
    return metrics


def daily_top_fraction_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    dates: np.ndarray,
    fraction: float,
) -> dict[str, float | int]:
    selected = daily_fraction_selection(scores, dates, fraction)
    metrics = _selected_metrics(target, selected)
    valid_dates = pd.to_datetime(pd.Series(dates), errors="coerce")
    date_count = int(valid_dates.nunique())
    metrics["fraction"] = float(fraction)
    metrics["date_count"] = date_count
    metrics["alerts_per_day"] = float(metrics["alerts"] / date_count) if date_count else float("nan")
    return metrics


def evaluate_prediction_metrics(
    target: np.ndarray,
    scores: np.ndarray,
    dates: np.ndarray,
    target_recall: float,
    top_fractions: Sequence[float] = (0.03, 0.05, 0.10, 0.20, 0.30, 0.50),
) -> dict[str, Any]:
    result: dict[str, Any] = safe_binary_metrics(target, scores)
    oracle = oracle_metrics_at_recall(target, scores, target_recall)
    for key, value in oracle.items():
        result[f"oracle_r{int(round(target_recall * 100))}_{key}"] = value
    for fraction in top_fractions:
        label = f"{int(round(fraction * 100))}pct"
        values = daily_top_fraction_metrics(target, scores, dates, fraction)
        for key, value in values.items():
            result[f"daily_top_{label}_{key}"] = value
    return result


def theoretical_minimum_alert_rate(positive_rate: float, target_recall: float) -> float:
    if not np.isfinite(positive_rate) or positive_rate < 0:
        return float("nan")
    return float(min(1.0, positive_rate * target_recall))


def maximum_recall_at_alert_rate(positive_rate: float, alert_rate: float) -> float:
    if not np.isfinite(positive_rate) or positive_rate <= 0:
        return float("nan")
    return float(min(1.0, alert_rate / positive_rate))


def fit_calibrator(
    target: np.ndarray,
    raw_scores: np.ndarray,
    dates: np.ndarray,
    minimum_rows: int = 400,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.int8)
    s = np.asarray(raw_scores, dtype=np.float64)
    d = pd.to_datetime(pd.Series(dates), errors="coerce").to_numpy(dtype="datetime64[ns]")
    valid = np.isfinite(s) & np.isin(y, [0, 1]) & ~pd.isna(d)
    y = y[valid]
    s = np.clip(s[valid], 1e-7, 1.0 - 1e-7)
    d = d[valid]
    if len(y) < minimum_rows or len(np.unique(y)) < 2:
        return {"kind": "identity"}
    unique_dates = pd.DatetimeIndex(pd.Series(d).unique()).sort_values()
    split_position = max(1, int(math.floor(len(unique_dates) * 0.70)))
    split_position = min(split_position, len(unique_dates) - 1)
    split_date = unique_dates[split_position]
    train_mask = d < np.datetime64(split_date)
    validation_mask = ~train_mask
    if int(train_mask.sum()) < minimum_rows // 2 or int(validation_mask.sum()) < 100:
        order = np.argsort(d, kind="mergesort")
        cut = max(1, int(len(order) * 0.70))
        train_mask = np.zeros(len(y), dtype=bool)
        train_mask[order[:cut]] = True
        validation_mask = ~train_mask

    candidates: list[tuple[str, dict[str, Any], float]] = []
    identity_brier = float(brier_score_loss(y[validation_mask], s[validation_mask]))
    candidates.append(("identity", {"kind": "identity"}, identity_brier))

    logits = np.log(s / (1.0 - s)).reshape(-1, 1)
    try:
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(logits[train_mask], y[train_mask])
        predicted = model.predict_proba(logits[validation_mask])[:, 1]
        spec = {
            "kind": "platt",
            "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
        }
        candidates.append(("platt", spec, float(brier_score_loss(y[validation_mask], predicted))))
    except Exception:
        pass

    try:
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
        isotonic.fit(s[train_mask], y[train_mask])
        predicted = isotonic.predict(s[validation_mask])
        spec = {
            "kind": "isotonic",
            "x_thresholds": [float(value) for value in isotonic.X_thresholds_],
            "y_thresholds": [float(value) for value in isotonic.y_thresholds_],
        }
        candidates.append(("isotonic", spec, float(brier_score_loss(y[validation_mask], predicted))))
    except Exception:
        pass

    _, chosen, _ = min(candidates, key=lambda item: item[2])
    kind = chosen["kind"]
    if kind == "platt":
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        model.fit(logits, y)
        return {
            "kind": "platt",
            "coefficient": float(model.coef_[0, 0]),
            "intercept": float(model.intercept_[0]),
        }
    if kind == "isotonic":
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1.0 - 1e-6)
        isotonic.fit(s, y)
        return {
            "kind": "isotonic",
            "x_thresholds": [float(value) for value in isotonic.X_thresholds_],
            "y_thresholds": [float(value) for value in isotonic.y_thresholds_],
        }
    return {"kind": "identity"}


def apply_calibrator(spec: Mapping[str, Any], raw_scores: np.ndarray) -> np.ndarray:
    s = np.asarray(raw_scores, dtype=np.float64)
    clipped = np.clip(s, 1e-7, 1.0 - 1e-7)
    kind = str(spec.get("kind", "identity"))
    if kind == "identity":
        return clipped
    if kind == "platt":
        logits = np.log(clipped / (1.0 - clipped))
        values = float(spec["coefficient"]) * logits + float(spec["intercept"])
        return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
    if kind == "isotonic":
        x = np.asarray(spec["x_thresholds"], dtype=np.float64)
        y = np.asarray(spec["y_thresholds"], dtype=np.float64)
        return np.clip(np.interp(clipped, x, y, left=y[0], right=y[-1]), 1e-7, 1.0 - 1e-7)
    raise ValueError(f"지원하지 않는 calibrator: {kind}")


def rank_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    result = np.full(len(array), np.nan, dtype=np.float64)
    valid = np.isfinite(array)
    if not valid.any():
        return result
    ranks = pd.Series(array[valid]).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    result[valid] = np.clip(ranks, 1e-6, 1.0 - 1e-6)
    return result


def datewise_rank_normalize(values: np.ndarray, dates: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    d = pd.to_datetime(pd.Series(dates), errors="coerce")
    frame = pd.DataFrame({"score": array, "date": d})
    return frame.groupby("date", sort=False)["score"].rank(method="average", pct=True).to_numpy(dtype=np.float64)


def build_sample_weights(
    target: np.ndarray,
    dates: np.ndarray,
    train_indices: np.ndarray,
    positive_weight_mode: str,
    time_decay_half_life_days: float | None,
) -> np.ndarray:
    y = np.asarray(target, dtype=np.int8)[train_indices]
    d = pd.to_datetime(pd.Series(np.asarray(dates)[train_indices]), errors="coerce")
    weights = np.ones(len(train_indices), dtype=np.float64)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positive_weight_mode == "sqrt_balance" and positives > 0:
        multiplier = math.sqrt(max(1.0, negatives / positives))
        weights[y == 1] *= multiplier
    elif positive_weight_mode == "balanced" and positives > 0:
        multiplier = max(1.0, negatives / positives)
        weights[y == 1] *= multiplier
    elif positive_weight_mode.startswith("fixed:"):
        multiplier = float(positive_weight_mode.split(":", 1)[1])
        weights[y == 1] *= multiplier
    elif positive_weight_mode != "none":
        raise ValueError(f"지원하지 않는 positive_weight_mode: {positive_weight_mode}")
    if time_decay_half_life_days is not None and float(time_decay_half_life_days) > 0:
        latest = d.max()
        age_days = (latest - d).dt.total_seconds().to_numpy(dtype=np.float64) / 86400.0
        decay = np.power(0.5, np.maximum(0.0, age_days) / float(time_decay_half_life_days))
        weights *= decay
    mean_weight = float(np.mean(weights)) if len(weights) else 1.0
    if mean_weight > 0:
        weights /= mean_weight
    return weights.astype(np.float32)


def build_inner_windows(
    dates: np.ndarray,
    outer_train_indices: np.ndarray,
    validation_days: int,
    purge_days: int,
    windows: int,
    step_days: int,
    minimum_train_days: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    train_dates = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(np.asarray(dates)[outer_train_indices]), errors="coerce").dropna().unique()
    ).sort_values()
    if len(train_dates) < minimum_train_days + purge_days + validation_days:
        return []
    date_values = pd.to_datetime(pd.Series(dates), errors="coerce").to_numpy(dtype="datetime64[ns]")
    results: list[tuple[np.ndarray, np.ndarray]] = []
    last_valid_end = len(train_dates) - 1
    for offset in range(windows):
        valid_end_pos = last_valid_end - offset * step_days
        valid_start_pos = valid_end_pos - validation_days + 1
        train_end_pos = valid_start_pos - purge_days - 1
        if valid_start_pos < 0 or train_end_pos < minimum_train_days - 1:
            continue
        valid_start = np.datetime64(train_dates[valid_start_pos])
        valid_end = np.datetime64(train_dates[valid_end_pos])
        train_end = np.datetime64(train_dates[train_end_pos])
        outer_mask = np.zeros(len(date_values), dtype=bool)
        outer_mask[outer_train_indices] = True
        train_mask = outer_mask & (date_values <= train_end)
        validation_mask = outer_mask & (date_values >= valid_start) & (date_values <= valid_end)
        train_idx = np.flatnonzero(train_mask)
        valid_idx = np.flatnonzero(validation_mask)
        if len(train_idx) and len(valid_idx):
            results.append((train_idx, valid_idx))
    return list(reversed(results))


def apply_training_window(
    indices: np.ndarray,
    dates: np.ndarray,
    policy: str,
    rolling_days: int | None,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if policy in {"expanding", "decay"}:
        return indices
    if policy == "rolling":
        if rolling_days is None or rolling_days <= 0:
            raise ValueError("rolling 정책에는 rolling_days가 필요합니다")
        local_dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(np.asarray(dates)[indices]), errors="coerce").dropna().unique()).sort_values()
        if len(local_dates) <= rolling_days:
            return indices
        start = np.datetime64(local_dates[-rolling_days])
        values = pd.to_datetime(pd.Series(np.asarray(dates)[indices]), errors="coerce").to_numpy(dtype="datetime64[ns]")
        return indices[values >= start]
    raise ValueError(f"지원하지 않는 train policy: {policy}")


def derive_forward_targets(
    frame: pd.DataFrame,
    return_column: str,
    ticker_column: str,
    date_column: str,
    threshold: float = 0.05,
) -> pd.DataFrame:
    required = {return_column, ticker_column, date_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"forward target 계산 열 누락: {sorted(missing)}")
    working = frame[[ticker_column, date_column, return_column]].copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working[return_column] = pd.to_numeric(working[return_column], errors="coerce")
    working["__row"] = np.arange(len(working), dtype=np.int64)
    working.sort_values([ticker_column, date_column, "__row"], kind="mergesort", inplace=True)
    columns = [
        "surge_d1",
        "surge_d2",
        "surge_d3",
        "crash_d3",
        "forward_valid",
        "first_hit_day",
        "best_forward_return_3d",
    ]
    results = pd.DataFrame(index=np.arange(len(working)), columns=columns, dtype=float)
    tolerance = 1e-12
    for _, group in working.groupby(ticker_column, sort=False):
        ret = group[return_column].to_numpy(dtype=np.float64)
        rows = group["__row"].to_numpy(dtype=np.int64)
        n = len(ret)
        for local in range(n):
            if local + 3 >= n:
                results.loc[rows[local], :] = [np.nan, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan]
                continue
            future = ret[local + 1 : local + 4]
            if not np.isfinite(future).all():
                results.loc[rows[local], :] = [np.nan, np.nan, np.nan, np.nan, 0.0, np.nan, np.nan]
                continue
            cumulative = np.cumprod(1.0 + future) - 1.0
            hits = np.flatnonzero(cumulative >= threshold - tolerance)
            surge1 = float(cumulative[0] >= threshold - tolerance)
            surge2 = float(np.max(cumulative[:2]) >= threshold - tolerance)
            surge3 = float(np.max(cumulative) >= threshold - tolerance)
            crash3 = float(np.min(cumulative) <= -threshold + tolerance)
            first_hit = float(hits[0] + 1) if len(hits) else np.nan
            best_forward = float(np.max(cumulative))
            results.loc[rows[local], :] = [
                surge1,
                surge2,
                surge3,
                crash3,
                1.0,
                first_hit,
                best_forward,
            ]
    return results.sort_index()


def aggregate_metric_records(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.DataFrame:
    numeric = [
        column
        for column in frame.columns
        if column not in set(group_columns)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    records: list[dict[str, Any]] = []
    for keys, part in frame.groupby(list(group_columns), dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = {column: value for column, value in zip(group_columns, keys)}
        record["row_count"] = int(len(part))
        for column in numeric:
            values = pd.to_numeric(part[column], errors="coerce").to_numpy(dtype=np.float64)
            values = values[np.isfinite(values)]
            record[f"mean_{column}"] = float(np.mean(values)) if len(values) else float("nan")
            record[f"std_{column}"] = float(np.std(values, ddof=0)) if len(values) else float("nan")
            record[f"min_{column}"] = float(np.min(values)) if len(values) else float("nan")
            record[f"max_{column}"] = float(np.max(values)) if len(values) else float("nan")
        records.append(record)
    return pd.DataFrame(records)


def compute_output_inventory(root: Path, paths: Iterable[Path]) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted({Path(value) for value in paths}):
        if not path.exists() or not path.is_file():
            continue
        inventory.append(
            {
                "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
        )
    return inventory
