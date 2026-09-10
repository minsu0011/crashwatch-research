from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score


SCHEMA_VERSION = "crashwatch_surge_hardfp_common_v7"


@dataclass(frozen=True)
class PrecisionRule:
    kind: str
    score_threshold: float = math.inf
    verifier_threshold: float = -math.inf
    pairwise_threshold: float = -math.inf
    recent_threshold: float = -math.inf
    target_precision: float = 0.70
    minimum_lcb: float = 0.60
    minimum_alerts: int = 30
    minimum_alert_days: int = 10
    minimum_recall: float = 0.03
    gate_pass: bool = False
    diagnostic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "score_threshold": finite_or_none(self.score_threshold),
            "verifier_threshold": finite_or_none(self.verifier_threshold),
            "pairwise_threshold": finite_or_none(self.pairwise_threshold),
            "recent_threshold": finite_or_none(self.recent_threshold),
            "target_precision": self.target_precision,
            "minimum_lcb": self.minimum_lcb,
            "minimum_alerts": self.minimum_alerts,
            "minimum_alert_days": self.minimum_alert_days,
            "minimum_recall": self.minimum_recall,
            "gate_pass": self.gate_pass,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class ErrorGroupConfig:
    top_quantile: float = 0.80
    missed_quantile: float = 0.50
    minimum_group_rows: int = 20


@dataclass(frozen=True)
class RelativeFeatureSpec:
    base_features: tuple[str, ...]
    include_raw: bool = True
    include_date_rank: bool = True
    include_date_z: bool = True
    include_market_rank: bool = True
    include_bucket_rank: bool = True
    include_industry_rank: bool = True
    include_market_delta: bool = True
    include_bucket_delta: bool = True
    include_industry_delta: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_features": list(self.base_features),
            "include_raw": self.include_raw,
            "include_date_rank": self.include_date_rank,
            "include_date_z": self.include_date_z,
            "include_market_rank": self.include_market_rank,
            "include_bucket_rank": self.include_bucket_rank,
            "include_industry_rank": self.include_industry_rank,
            "include_market_delta": self.include_market_delta,
            "include_bucket_delta": self.include_bucket_delta,
            "include_industry_delta": self.include_industry_delta,
        }


def finite_or_none(value: Any) -> Any:
    if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
        return None
    return value


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return finite_or_none(float(value))
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return finite_or_none(value)
    return value


def stable_json_bytes(payload: Any) -> bytes:
    return json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_bytes(data)
        os.replace(temporary, path)
    except Exception:
        with contextlib_suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


class contextlib_suppress:
    def __init__(self, *exceptions: type[BaseException]):
        self.exceptions = exceptions
    def __enter__(self) -> None:
        return None
    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return bool(exc_type is not None and issubclass(exc_type, self.exceptions))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, json.dumps(json_safe(payload), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"))


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".csv", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig")
        os.replace(temporary, path)
    except Exception:
        with contextlib_suppress(FileNotFoundError):
            os.unlink(temporary)
        raise


def with_checksum(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("payload_sha256", None)
    result["payload_sha256"] = sha256_bytes(stable_json_bytes(result))
    return result


def checksum_valid(payload: Mapping[str, Any]) -> bool:
    expected = payload.get("payload_sha256")
    if not isinstance(expected, str):
        return False
    clean = dict(payload)
    clean.pop("payload_sha256", None)
    return expected == sha256_bytes(stable_json_bytes(clean))


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else math.nan


def wilson_lower_bound(successes: int, trials: int, confidence: float = 0.95) -> float:
    if trials <= 0:
        return math.nan
    p = successes / trials
    z = float(norm.ppf(0.5 + confidence / 2.0))
    denom = 1.0 + z * z / trials
    center = p + z * z / (2.0 * trials)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials)
    return float((center - margin) / denom)


def datewise_rank(values: Sequence[float], dates: Sequence[Any]) -> np.ndarray:
    frame = pd.DataFrame({"date": pd.to_datetime(pd.Series(dates), errors="coerce"), "value": np.asarray(values, dtype=np.float64)})
    result = frame.groupby("date", sort=False, dropna=False)["value"].rank(method="average", pct=True).to_numpy(dtype=np.float64)
    return result


def event_structure(target: Sequence[int], tickers: Sequence[Any], dates: Sequence[Any]) -> pd.DataFrame:
    y = np.asarray(target, dtype=np.uint8)
    ticker_arr = np.asarray(pd.Series(tickers).astype("string").fillna("__NA__"), dtype=object)
    date_arr = pd.to_datetime(pd.Series(dates), errors="coerce").to_numpy(dtype="datetime64[ns]")
    n = len(y)
    event_id = np.full(n, -1, dtype=np.int64)
    event_start = np.zeros(n, dtype=np.uint8)
    event_end = np.zeros(n, dtype=np.uint8)
    event_weight = np.ones(n, dtype=np.float64)
    event_counter = 0
    order = np.lexsort((date_arr.astype(np.int64), ticker_arr.astype(str)))
    start = 0
    while start < n:
        ticker = ticker_arr[order[start]]
        end = start + 1
        while end < n and ticker_arr[order[end]] == ticker:
            end += 1
        local = order[start:end]
        values = y[local]
        pos = 0
        while pos < len(local):
            if values[pos] != 1:
                pos += 1
                continue
            run_end = pos + 1
            while run_end < len(local) and values[run_end] == 1:
                run_end += 1
            rows = local[pos:run_end]
            event_id[rows] = event_counter
            event_start[rows[0]] = 1
            event_end[rows[-1]] = 1
            event_weight[rows] = 1.0 / len(rows)
            event_counter += 1
            pos = run_end
        start = end
    return pd.DataFrame(
        {
            "event_id": event_id,
            "event_start": event_start,
            "event_end": event_end,
            "event_weight": event_weight,
        }
    )


def assign_error_groups(
    target: Sequence[int],
    base_score: Sequence[float],
    dates: Sequence[Any],
    config: ErrorGroupConfig,
) -> pd.DataFrame:
    y = np.asarray(target, dtype=np.uint8)
    score = np.asarray(base_score, dtype=np.float64)
    rank = datewise_rank(score, dates)
    group = np.full(len(y), "OTHER", dtype=object)
    high = np.isfinite(rank) & (rank >= float(config.top_quantile))
    low = np.isfinite(rank) & (rank <= float(config.missed_quantile))
    group[(y == 1) & high] = "A_TOP_TRUE_POSITIVE"
    group[(y == 0) & high] = "B_TOP_FALSE_POSITIVE"
    group[(y == 1) & low] = "C_LOW_MISSED_POSITIVE"
    group[(y == 1) & ~(high | low)] = "C_MID_MISSED_POSITIVE"
    return pd.DataFrame({"base_score": score, "base_date_rank": rank, "error_group": group})


def _binary_auc(values: np.ndarray, label: np.ndarray) -> float:
    mask = np.isfinite(values) & np.isfinite(label)
    if int(mask.sum()) < 4 or len(np.unique(label[mask])) < 2:
        return math.nan
    return float(roc_auc_score(label[mask], values[mask]))


def _standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return math.nan
    va = float(np.var(a, ddof=1))
    vb = float(np.var(b, ddof=1))
    pooled = math.sqrt(max(0.0, ((len(a) - 1) * va + (len(b) - 1) * vb) / max(1, len(a) + len(b) - 2)))
    if pooled <= 1e-15:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / pooled)


def _robust_location_effect(a: np.ndarray, b: np.ndarray) -> float:
    both = np.concatenate([a[np.isfinite(a)], b[np.isfinite(b)]])
    if len(both) < 4:
        return math.nan
    q25, q75 = np.quantile(both, [0.25, 0.75])
    scale = float(q75 - q25)
    if scale <= 1e-15:
        return 0.0
    return float((np.nanmedian(a) - np.nanmedian(b)) / scale)


def contrast_feature_stats(
    feature: str,
    raw: np.ndarray,
    rank: np.ndarray,
    groups: Sequence[str],
    positive_group: str,
    negative_group: str,
    fold_id: int,
    contrast: str,
) -> dict[str, Any]:
    group_arr = np.asarray(groups, dtype=object)
    pos = group_arr == positive_group
    neg = group_arr == negative_group
    selected = pos | neg
    labels = pos[selected].astype(np.uint8)
    raw_sel = raw[selected].astype(np.float64, copy=False)
    rank_sel = rank[selected].astype(np.float64, copy=False)
    auc_raw = _binary_auc(raw_sel, labels)
    auc_rank = _binary_auc(rank_sel, labels)
    oriented_raw = max(auc_raw, 1.0 - auc_raw) if np.isfinite(auc_raw) else math.nan
    oriented_rank = max(auc_rank, 1.0 - auc_rank) if np.isfinite(auc_rank) else math.nan
    direction_raw = int(np.sign(auc_raw - 0.5)) if np.isfinite(auc_raw) and abs(auc_raw - 0.5) > 1e-12 else 0
    direction_rank = int(np.sign(auc_rank - 0.5)) if np.isfinite(auc_rank) and abs(auc_rank - 0.5) > 1e-12 else 0
    return {
        "fold_id": int(fold_id),
        "contrast": contrast,
        "feature": feature,
        "positive_group": positive_group,
        "negative_group": negative_group,
        "positive_rows": int(pos.sum()),
        "negative_rows": int(neg.sum()),
        "raw_auc": auc_raw,
        "raw_oriented_auc": oriented_raw,
        "raw_direction": direction_raw,
        "rank_auc": auc_rank,
        "rank_oriented_auc": oriented_rank,
        "rank_direction": direction_rank,
        "raw_smd": _standardized_mean_difference(raw[pos], raw[neg]),
        "rank_smd": _standardized_mean_difference(rank[pos], rank[neg]),
        "raw_robust_location_effect": _robust_location_effect(raw[pos], raw[neg]),
        "rank_robust_location_effect": _robust_location_effect(rank[pos], rank[neg]),
        "positive_raw_median": float(np.nanmedian(raw[pos])) if np.isfinite(raw[pos]).any() else math.nan,
        "negative_raw_median": float(np.nanmedian(raw[neg])) if np.isfinite(raw[neg]).any() else math.nan,
    }


def summarize_error_map(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    records: list[dict[str, Any]] = []
    for (feature, contrast), part in rows.groupby(["feature", "contrast"], sort=False):
        raw_auc = pd.to_numeric(part["raw_oriented_auc"], errors="coerce").to_numpy(dtype=np.float64)
        rank_auc = pd.to_numeric(part["rank_oriented_auc"], errors="coerce").to_numpy(dtype=np.float64)
        raw_dir = pd.to_numeric(part["raw_direction"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        rank_dir = pd.to_numeric(part["rank_direction"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        raw_smd = np.abs(pd.to_numeric(part["raw_smd"], errors="coerce").to_numpy(dtype=np.float64))
        rank_smd = np.abs(pd.to_numeric(part["rank_smd"], errors="coerce").to_numpy(dtype=np.float64))
        nonzero_raw = raw_dir[raw_dir != 0]
        nonzero_rank = rank_dir[rank_dir != 0]
        raw_consistency = max(np.mean(nonzero_raw > 0), np.mean(nonzero_raw < 0)) if len(nonzero_raw) else 0.0
        rank_consistency = max(np.mean(nonzero_rank > 0), np.mean(nonzero_rank < 0)) if len(nonzero_rank) else 0.0
        mean_raw_edge = float(np.nanmean(np.maximum(0.0, raw_auc - 0.5))) if np.isfinite(raw_auc).any() else 0.0
        mean_rank_edge = float(np.nanmean(np.maximum(0.0, rank_auc - 0.5))) if np.isfinite(rank_auc).any() else 0.0
        min_rank_edge = float(np.nanmin(np.maximum(0.0, rank_auc - 0.5))) if np.isfinite(rank_auc).any() else 0.0
        mean_smd = float(np.nanmean(np.concatenate([raw_smd[np.isfinite(raw_smd)], rank_smd[np.isfinite(rank_smd)]]))) if (np.isfinite(raw_smd).any() or np.isfinite(rank_smd).any()) else 0.0
        score = (
            0.34 * min(1.0, mean_raw_edge / 0.20)
            + 0.34 * min(1.0, mean_rank_edge / 0.20)
            + 0.12 * min(1.0, min_rank_edge / 0.10)
            + 0.10 * raw_consistency
            + 0.10 * rank_consistency
        )
        records.append(
            {
                "feature": str(feature),
                "contrast": str(contrast),
                "fold_count": int(part["fold_id"].nunique()),
                "mean_raw_oriented_auc": float(np.nanmean(raw_auc)) if np.isfinite(raw_auc).any() else math.nan,
                "mean_rank_oriented_auc": float(np.nanmean(rank_auc)) if np.isfinite(rank_auc).any() else math.nan,
                "min_rank_oriented_auc": float(np.nanmin(rank_auc)) if np.isfinite(rank_auc).any() else math.nan,
                "raw_direction_consistency": float(raw_consistency),
                "rank_direction_consistency": float(rank_consistency),
                "mean_abs_smd": mean_smd,
                "contrast_score": float(score),
            }
        )
    result = pd.DataFrame(records)
    pivot = result.pivot_table(index="feature", columns="contrast", values="contrast_score", aggfunc="max")
    for contrast in ["A_VS_B", "C_VS_B", "A_VS_C"]:
        if contrast not in pivot.columns:
            pivot[contrast] = 0.0
    bottleneck = (
        0.50 * pivot["A_VS_B"].fillna(0.0)
        + 0.40 * pivot["C_VS_B"].fillna(0.0)
        + 0.10 * pivot["A_VS_C"].fillna(0.0)
    )
    result = result.merge(bottleneck.rename("bottleneck_score"), left_on="feature", right_index=True, how="left")
    return result.sort_values(["bottleneck_score", "contrast_score"], ascending=[False, False], kind="mergesort").reset_index(drop=True)


def select_error_features(
    summary: pd.DataFrame,
    max_features: int,
    minimum_contrast_score: float = 0.12,
    extra_features: Sequence[str] = (),
) -> list[str]:
    if summary.empty:
        return list(dict.fromkeys(str(x) for x in extra_features))[:max_features]
    per_feature = summary.groupby("feature", as_index=False).agg(
        bottleneck_score=("bottleneck_score", "max"),
        best_contrast=("contrast_score", "max"),
        folds=("fold_count", "max"),
    )
    eligible = per_feature[
        (per_feature["best_contrast"] >= float(minimum_contrast_score))
        | (per_feature["bottleneck_score"] >= float(minimum_contrast_score))
    ].sort_values(["bottleneck_score", "best_contrast", "feature"], ascending=[False, False, True], kind="mergesort")
    features = eligible["feature"].astype(str).tolist()[: int(max_features)]
    for feature in extra_features:
        feature = str(feature)
        if feature not in features:
            features.append(feature)
    return features[: int(max_features)]


def _zscore_group(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_cols, sort=False, dropna=False)[value_cols]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0.0, np.nan)
    return (frame[value_cols] - mean) / std


def _median_delta(frame: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    median = frame.groupby(group_cols, sort=False, dropna=False)[value_cols].transform("median")
    return frame[value_cols] - median


def build_relative_feature_frame(
    raw_matrix: np.ndarray,
    feature_names: Sequence[str],
    metadata: pd.DataFrame,
    spec: RelativeFeatureSpec,
    date_column: str = "date",
    market_column: str = "market",
    bucket_column: str = "bucket",
    industry_column: str = "industry_name",
) -> pd.DataFrame:
    feature_names = list(feature_names)
    lookup = {name: idx for idx, name in enumerate(feature_names)}
    missing = [name for name in spec.base_features if name not in lookup]
    if missing:
        raise KeyError(f"relative feature source 누락: {missing[:10]}")
    base = pd.DataFrame(
        raw_matrix[:, [lookup[name] for name in spec.base_features]],
        columns=list(spec.base_features),
    )
    meta = metadata.reset_index(drop=True).copy()
    if len(base) != len(meta):
        raise ValueError("relative feature metadata row mismatch")
    meta[date_column] = pd.to_datetime(meta[date_column], errors="coerce")
    work = pd.concat([meta[[column for column in [date_column, market_column, bucket_column, industry_column] if column in meta.columns]], base], axis=1)
    outputs: list[pd.DataFrame] = []
    names = list(spec.base_features)
    if spec.include_raw:
        raw = base.copy()
        raw.columns = [f"raw__{name}" for name in names]
        outputs.append(raw)
    if spec.include_date_rank:
        rank = work.groupby(date_column, sort=False, dropna=False)[names].rank(method="average", pct=True)
        rank.columns = [f"date_rank__{name}" for name in names]
        outputs.append(rank)
    if spec.include_date_z:
        z = _zscore_group(work, [date_column], names)
        z.columns = [f"date_z__{name}" for name in names]
        outputs.append(z)
    if market_column in work.columns:
        if spec.include_market_rank:
            rank = work.groupby([date_column, market_column], sort=False, dropna=False)[names].rank(method="average", pct=True)
            rank.columns = [f"market_rank__{name}" for name in names]
            outputs.append(rank)
        if spec.include_market_delta:
            delta = _median_delta(work, [date_column, market_column], names)
            delta.columns = [f"market_delta__{name}" for name in names]
            outputs.append(delta)
    if bucket_column in work.columns:
        if spec.include_bucket_rank:
            rank = work.groupby([date_column, bucket_column], sort=False, dropna=False)[names].rank(method="average", pct=True)
            rank.columns = [f"bucket_rank__{name}" for name in names]
            outputs.append(rank)
        if spec.include_bucket_delta:
            delta = _median_delta(work, [date_column, bucket_column], names)
            delta.columns = [f"bucket_delta__{name}" for name in names]
            outputs.append(delta)
    if industry_column in work.columns:
        if spec.include_industry_rank:
            rank = work.groupby([date_column, industry_column], sort=False, dropna=False)[names].rank(method="average", pct=True)
            rank.columns = [f"industry_rank__{name}" for name in names]
            outputs.append(rank)
        if spec.include_industry_delta:
            delta = _median_delta(work, [date_column, industry_column], names)
            delta.columns = [f"industry_delta__{name}" for name in names]
            outputs.append(delta)
    result = pd.concat(outputs, axis=1) if outputs else pd.DataFrame(index=np.arange(len(base)))
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result.astype(np.float32)


def add_scope_one_hot(frame: pd.DataFrame, metadata: pd.DataFrame, columns: Sequence[str] = ("market", "bucket"), max_levels: int = 20) -> pd.DataFrame:
    outputs = [frame.reset_index(drop=True)]
    for column in columns:
        if column not in metadata.columns:
            continue
        series = metadata[column].astype("string").fillna("__NA__")
        top = series.value_counts(dropna=False).head(max_levels).index
        clipped = series.where(series.isin(top), "__OTHER__")
        dummies = pd.get_dummies(clipped, prefix=f"scope__{column}", dtype=np.float32)
        outputs.append(dummies.reset_index(drop=True))
    return pd.concat(outputs, axis=1)


def build_pairwise_training_frame(
    features: pd.DataFrame,
    groups: Sequence[str],
    dates: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    group_arr = np.asarray(groups, dtype=object)
    keep = np.isin(group_arr, ["A_TOP_TRUE_POSITIVE", "B_TOP_FALSE_POSITIVE", "C_LOW_MISSED_POSITIVE", "C_MID_MISSED_POSITIVE"])
    labels = np.zeros(len(group_arr), dtype=np.float32)
    labels[group_arr == "A_TOP_TRUE_POSITIVE"] = 1.0
    labels[group_arr == "C_MID_MISSED_POSITIVE"] = 1.0
    labels[group_arr == "C_LOW_MISSED_POSITIVE"] = 2.0
    work = features.loc[keep].reset_index(drop=True)
    label = labels[keep]
    date_values = pd.to_datetime(pd.Series(np.asarray(dates)[keep]), errors="coerce")
    audit = pd.DataFrame({"date": date_values, "label": label})
    date_stats = audit.groupby("date", sort=True)["label"].agg(["count", "min", "max"]).reset_index()
    valid_dates = set(date_stats.loc[(date_stats["count"] >= 2) & (date_stats["min"] == 0) & (date_stats["max"] > 0), "date"])
    valid_mask = date_values.isin(valid_dates).to_numpy(dtype=bool)
    work = work.loc[valid_mask].reset_index(drop=True)
    label = label[valid_mask]
    dates_kept = date_values.loc[valid_mask].reset_index(drop=True)
    order = np.argsort(dates_kept.to_numpy(dtype="datetime64[ns]"), kind="mergesort")
    work = work.iloc[order].reset_index(drop=True)
    label = label[order]
    dates_sorted = dates_kept.iloc[order].reset_index(drop=True)
    qid, unique = pd.factorize(dates_sorted, sort=True)
    audit = pd.DataFrame({"date": dates_sorted, "label": label})
    return work.to_numpy(dtype=np.float32), label.astype(np.float32), qid.astype(np.int64), audit


def fit_lgb_ranker(x: np.ndarray, y: np.ndarray, qid: np.ndarray, params: Mapping[str, Any], rounds: int, seed: int, threads: int) -> Any:
    import lightgbm as lgb
    if len(x) == 0:
        raise ValueError("empty pairwise train")
    order = np.argsort(qid, kind="mergesort")
    x = x[order]
    y = y[order]
    qid = qid[order]
    counts = pd.Series(qid).value_counts(sort=False).sort_index().to_numpy(dtype=np.int32)
    resolved = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10, 20],
        "learning_rate": 0.035,
        "num_leaves": 31,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.82,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 4.0,
        "verbosity": -1,
        "num_threads": int(threads),
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "label_gain": [0, 1, 3],
    }
    resolved.update(dict(params))
    dataset = lgb.Dataset(x, label=y, group=counts, free_raw_data=False)
    return lgb.train(resolved, dataset, num_boost_round=int(rounds), callbacks=[lgb.log_evaluation(period=0)])


def fit_xgb_ranker(x: np.ndarray, y: np.ndarray, qid: np.ndarray, params: Mapping[str, Any], rounds: int, seed: int, threads: int, device: str) -> Any:
    import xgboost as xgb
    if len(x) == 0:
        raise ValueError("empty pairwise train")
    order = np.argsort(qid, kind="mergesort")
    x = x[order]
    y = y[order]
    qid = qid[order]
    dtrain = xgb.DMatrix(x, label=y, qid=qid)
    resolved = {
        "objective": "rank:pairwise",
        "eval_metric": "ndcg@20",
        "tree_method": "hist",
        "max_depth": 5,
        "min_child_weight": 8.0,
        "eta": 0.035,
        "subsample": 0.82,
        "colsample_bytree": 0.82,
        "lambda": 4.0,
        "alpha": 0.2,
        "seed": int(seed),
        "nthread": int(threads),
        "device": str(device),
    }
    resolved.update(dict(params))
    return xgb.train(resolved, dtrain, num_boost_round=int(rounds), verbose_eval=False)


def predict_ranker(model: Any, family: str, x: np.ndarray) -> np.ndarray:
    if family == "lightgbm":
        return np.asarray(model.predict(x), dtype=np.float64)
    if family == "xgboost":
        import xgboost as xgb
        return np.asarray(model.predict(xgb.DMatrix(x)), dtype=np.float64)
    raise ValueError(f"unknown ranker family: {family}")


def binary_metrics(target: Sequence[int], score: Sequence[float]) -> dict[str, float]:
    y = np.asarray(target, dtype=np.uint8)
    s = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(y) == 0 or len(np.unique(y)) < 2:
        return {"rows": float(len(y)), "positive_rate": float(np.mean(y)) if len(y) else math.nan, "pr_auc": math.nan, "roc_auc": math.nan}
    return {
        "rows": float(len(y)),
        "positive_rate": float(np.mean(y)),
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
    }


def apply_rule(frame: pd.DataFrame, rule: PrecisionRule, score_column: str) -> np.ndarray:
    if rule.kind == "no_alert" or not np.isfinite(rule.score_threshold):
        return np.zeros(len(frame), dtype=bool)
    alert = pd.to_numeric(frame[score_column], errors="coerce").to_numpy(dtype=np.float64) >= float(rule.score_threshold)
    if rule.kind == "cascade":
        if np.isfinite(rule.verifier_threshold) and "verifier_mean" in frame:
            alert &= pd.to_numeric(frame["verifier_mean"], errors="coerce").to_numpy(dtype=np.float64) >= rule.verifier_threshold
        if np.isfinite(rule.pairwise_threshold) and "pairwise_mean" in frame:
            alert &= pd.to_numeric(frame["pairwise_mean"], errors="coerce").to_numpy(dtype=np.float64) >= rule.pairwise_threshold
        if np.isfinite(rule.recent_threshold) and "recent_mean" in frame:
            alert &= pd.to_numeric(frame["recent_mean"], errors="coerce").to_numpy(dtype=np.float64) >= rule.recent_threshold
    return alert


def evaluate_alert_mask(
    target: Sequence[int],
    alert: Sequence[bool],
    dates: Sequence[Any],
    event_id: Sequence[int] | None = None,
    confidence: float = 0.95,
) -> dict[str, Any]:
    y = np.asarray(target, dtype=np.uint8)
    a = np.asarray(alert, dtype=bool)
    date_values = pd.to_datetime(pd.Series(dates), errors="coerce")
    tp = int(np.sum(a & (y == 1)))
    fp = int(np.sum(a & (y == 0)))
    positives = int(np.sum(y == 1))
    alerts = int(np.sum(a))
    alert_days = int(date_values[a].nunique()) if alerts else 0
    precision = safe_div(tp, alerts)
    recall = safe_div(tp, positives)
    result: dict[str, Any] = {
        "rows": int(len(y)),
        "positives": positives,
        "alerts": alerts,
        "true_positives": tp,
        "false_positives": fp,
        "alert_days": alert_days,
        "precision": precision,
        "recall": recall,
        "alert_rate": safe_div(alerts, len(y)),
        "precision_lcb": wilson_lower_bound(tp, alerts, confidence=confidence) if alerts else math.nan,
        "base_rate": float(np.mean(y)) if len(y) else math.nan,
        "precision_lift": safe_div(precision, float(np.mean(y))) if alerts and np.mean(y) > 0 else math.nan,
    }
    if event_id is not None:
        ids = np.asarray(event_id, dtype=np.int64)
        positive_events = set(ids[(y == 1) & (ids >= 0)].tolist())
        captured_events = set(ids[a & (y == 1) & (ids >= 0)].tolist())
        result["positive_events"] = len(positive_events)
        result["captured_events"] = len(captured_events)
        result["event_recall"] = safe_div(len(captured_events), len(positive_events))
        result["alerts_per_captured_event"] = safe_div(alerts, len(captured_events))
    return result


def rule_gate(metrics: Mapping[str, Any], rule: PrecisionRule) -> bool:
    precision = float(metrics.get("precision", math.nan))
    lcb = float(metrics.get("precision_lcb", math.nan))
    recall = float(metrics.get("recall", math.nan))
    return bool(
        np.isfinite(precision)
        and precision >= rule.target_precision
        and np.isfinite(lcb)
        and lcb >= rule.minimum_lcb
        and int(metrics.get("alerts", 0)) >= rule.minimum_alerts
        and int(metrics.get("alert_days", 0)) >= rule.minimum_alert_days
        and np.isfinite(recall)
        and recall >= rule.minimum_recall
    )


def _threshold_candidates(values: np.ndarray, maximum: int = 300) -> np.ndarray:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.asarray([], dtype=np.float64)
    unique = np.unique(values)
    if len(unique) <= maximum:
        return unique[::-1]
    quantiles = np.linspace(0.0, 1.0, maximum)
    return np.unique(np.quantile(values, quantiles))[::-1]


def select_precision_threshold(
    frame: pd.DataFrame,
    target_column: str,
    score_column: str,
    date_column: str,
    event_id_column: str | None,
    target_precision: float,
    minimum_lcb: float,
    minimum_alerts: int,
    minimum_alert_days: int,
    minimum_recall: float,
    confidence: float = 0.95,
    maximum_candidates: int = 300,
) -> tuple[PrecisionRule, pd.DataFrame, dict[str, Any]]:
    scores = pd.to_numeric(frame[score_column], errors="coerce").to_numpy(dtype=np.float64)
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=np.uint8)
    dates = frame[date_column]
    event_ids = frame[event_id_column].to_numpy(dtype=np.int64) if event_id_column and event_id_column in frame else None
    rows: list[dict[str, Any]] = []
    safe: list[tuple[PrecisionRule, dict[str, Any]]] = []
    practical: list[tuple[float, float, int, PrecisionRule, dict[str, Any]]] = []
    for threshold in _threshold_candidates(scores, maximum_candidates):
        base = PrecisionRule(
            kind="threshold",
            score_threshold=float(threshold),
            target_precision=float(target_precision),
            minimum_lcb=float(minimum_lcb),
            minimum_alerts=int(minimum_alerts),
            minimum_alert_days=int(minimum_alert_days),
            minimum_recall=float(minimum_recall),
        )
        alert = scores >= threshold
        metrics = evaluate_alert_mask(target, alert, dates, event_ids, confidence)
        passed = rule_gate(metrics, base)
        record = {"kind": "threshold", "threshold": float(threshold), "gate_pass": passed, **metrics}
        rows.append(record)
        candidate = PrecisionRule(**{**base.__dict__, "gate_pass": passed})
        if passed:
            safe.append((candidate, metrics))
        if metrics["alerts"] >= minimum_alerts and metrics["alert_days"] >= minimum_alert_days and metrics["recall"] >= minimum_recall:
            practical.append((float(metrics["precision"]), float(metrics["recall"]), int(metrics["alerts"]), candidate, metrics))
    if safe:
        safe.sort(key=lambda item: (float(item[1]["recall"]), int(item[1]["alerts"]), float(item[1]["precision"])), reverse=True)
        chosen, metrics = safe[0]
    else:
        chosen = PrecisionRule(
            kind="no_alert",
            score_threshold=math.inf,
            target_precision=float(target_precision),
            minimum_lcb=float(minimum_lcb),
            minimum_alerts=int(minimum_alerts),
            minimum_alert_days=int(minimum_alert_days),
            minimum_recall=float(minimum_recall),
            gate_pass=False,
        )
        metrics = evaluate_alert_mask(target, np.zeros(len(target), dtype=bool), dates, event_ids, confidence)
    diagnostic: dict[str, Any] = {}
    if practical:
        practical.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        _, _, _, diag_rule, diag_metrics = practical[0]
        diagnostic = {"rule": diag_rule.to_dict(), "metrics": diag_metrics}
    return chosen, pd.DataFrame(rows), diagnostic


def search_precision_cascade(
    frame: pd.DataFrame,
    target_column: str,
    score_column: str,
    date_column: str,
    event_id_column: str | None,
    target_precision: float,
    minimum_lcb: float,
    minimum_alerts: int,
    minimum_alert_days: int,
    minimum_recall: float,
    confidence: float = 0.95,
    quantiles: Sequence[float] = (0.55, 0.70, 0.80, 0.88, 0.93),
) -> tuple[PrecisionRule, pd.DataFrame, dict[str, Any]]:
    required = [score_column]
    optional = [column for column in ["verifier_mean", "pairwise_mean", "recent_mean"] if column in frame.columns]
    target = pd.to_numeric(frame[target_column], errors="coerce").to_numpy(dtype=np.uint8)
    dates = frame[date_column]
    event_ids = frame[event_id_column].to_numpy(dtype=np.int64) if event_id_column and event_id_column in frame else None
    grids: dict[str, list[float]] = {}
    for column in required + optional:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
        valid = values[np.isfinite(values)]
        if len(valid) == 0:
            grids[column] = [-math.inf]
        else:
            grids[column] = sorted(set(float(np.quantile(valid, q)) for q in quantiles))
    score_grid = grids[score_column]
    verifier_grid = grids.get("verifier_mean", [-math.inf])
    pair_grid = grids.get("pairwise_mean", [-math.inf])
    recent_grid = grids.get("recent_mean", [-math.inf])
    rows: list[dict[str, Any]] = []
    safe: list[tuple[PrecisionRule, dict[str, Any]]] = []
    practical: list[tuple[float, float, int, PrecisionRule, dict[str, Any]]] = []
    for st in score_grid:
        for vt in verifier_grid:
            for pt in pair_grid:
                for rt in recent_grid:
                    rule = PrecisionRule(
                        kind="cascade",
                        score_threshold=st,
                        verifier_threshold=vt,
                        pairwise_threshold=pt,
                        recent_threshold=rt,
                        target_precision=target_precision,
                        minimum_lcb=minimum_lcb,
                        minimum_alerts=minimum_alerts,
                        minimum_alert_days=minimum_alert_days,
                        minimum_recall=minimum_recall,
                    )
                    alert = apply_rule(frame, rule, score_column)
                    metrics = evaluate_alert_mask(target, alert, dates, event_ids, confidence)
                    passed = rule_gate(metrics, rule)
                    rule = PrecisionRule(**{**rule.__dict__, "gate_pass": passed})
                    rows.append({
                        "kind": "cascade",
                        "score_threshold": st,
                        "verifier_threshold": vt,
                        "pairwise_threshold": pt,
                        "recent_threshold": rt,
                        "gate_pass": passed,
                        **metrics,
                    })
                    if passed:
                        safe.append((rule, metrics))
                    if metrics["alerts"] >= minimum_alerts and metrics["alert_days"] >= minimum_alert_days and metrics["recall"] >= minimum_recall:
                        practical.append((float(metrics["precision"]), float(metrics["recall"]), int(metrics["alerts"]), rule, metrics))
    if safe:
        safe.sort(key=lambda item: (float(item[1]["recall"]), int(item[1]["alerts"]), float(item[1]["precision"])), reverse=True)
        chosen, metrics = safe[0]
    else:
        chosen = PrecisionRule(
            kind="no_alert",
            score_threshold=math.inf,
            target_precision=target_precision,
            minimum_lcb=minimum_lcb,
            minimum_alerts=minimum_alerts,
            minimum_alert_days=minimum_alert_days,
            minimum_recall=minimum_recall,
            gate_pass=False,
        )
        metrics = evaluate_alert_mask(target, np.zeros(len(target), dtype=bool), dates, event_ids, confidence)
    diagnostic: dict[str, Any] = {}
    if practical:
        practical.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        _, _, _, diag_rule, diag_metrics = practical[0]
        diagnostic = {"rule": diag_rule.to_dict(), "metrics": diag_metrics}
    return chosen, pd.DataFrame(rows), diagnostic


def method_score(frame: pd.DataFrame, method: str) -> np.ndarray:
    def col(name: str, default: float = 0.5) -> np.ndarray:
        if name not in frame:
            return np.full(len(frame), default, dtype=np.float64)
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        if np.isfinite(values).any():
            fill = float(np.nanmedian(values[np.isfinite(values)]))
        else:
            fill = default
        values[~np.isfinite(values)] = fill
        return values
    base = col("base_score")
    verifier = col("verifier_mean")
    pairwise = col("pairwise_mean")
    recent = col("recent_mean")
    event = col("event_start_score")
    scope = col("scope_conditional_score")
    if method == "base":
        score = base
    elif method == "pairwise":
        score = pairwise
    elif method == "verifier":
        score = verifier
    elif method == "recent":
        score = recent
    elif method == "balanced_blend":
        score = 0.28 * base + 0.24 * verifier + 0.24 * pairwise + 0.14 * recent + 0.06 * event + 0.04 * scope
    elif method == "hardfp_blend":
        score = 0.18 * base + 0.34 * verifier + 0.30 * pairwise + 0.12 * recent + 0.06 * event
    elif method == "regime_blend":
        score = 0.18 * base + 0.16 * verifier + 0.18 * pairwise + 0.34 * recent + 0.08 * event + 0.06 * scope
    else:
        raise KeyError(method)
    return np.clip(score, 0.0, 1.0)


def build_output_inventory(
    root: Path,
    exclude_names: Sequence[str] = (
        "OUTPUT_INVENTORY_V7.json",
        "RUN_STATUS.json",
        "VERIFICATION_REPORT_V7.json",
    ),
) -> dict[str, Any]:
    excluded = set(exclude_names)
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        records.append({
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "size": int(path.stat().st_size),
            "sha256": sha256_file(path),
        })
    return with_checksum({"schema": "crashwatch_surge_hardfp_inventory_v7", "files": records})
