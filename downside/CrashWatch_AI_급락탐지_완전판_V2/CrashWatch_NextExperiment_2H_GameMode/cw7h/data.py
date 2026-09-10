from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import atomic_json, canonical_hash, free_disk_gb, hash_file, hash_strings, normalize_ticker, read_json

LOGGER = logging.getLogger(__name__)
CACHE_VERSION = "cw7h_all_valid_v1"
SAFETY_FLAG_COLUMNS = ["sealed_do_not_train_or_tune"]
DATE_CANDIDATES = ["date", "trade_date", "trading_date", "datetime", "dt", "일자", "날짜"]
TICKER_CANDIDATES = ["ticker", "stock_code", "code", "symbol", "종목코드", "단축코드"]
BUCKET_CANDIDATES = ["bucket", "sector_bucket", "industry_bucket", "sector", "industry", "업종"]


@dataclass(frozen=True)
class References:
    all_valid_features: list[str]
    groups: dict[str, list[str]]
    audit: pd.DataFrame
    ticker_to_bucket: dict[str, str]
    folds: list[dict[str, Any]]
    hashes: dict[str, str]


@dataclass(frozen=True)
class PreparedData:
    root: Path
    dataset_path: Path
    signature: str
    manifest: dict[str, Any]
    feature_names: list[str]
    feature_groups: dict[str, str]

    @property
    def x_path(self) -> Path:
        return self.root / "X_all_valid.npy"

    @property
    def y_path(self) -> Path:
        return self.root / "target.npy"

    @property
    def dates_path(self) -> Path:
        return self.root / "dates_ns.npy"

    @property
    def tickers_path(self) -> Path:
        return self.root / "tickers.npy"

    @property
    def buckets_path(self) -> Path:
        return self.root / "buckets.npy"

    @property
    def row_ids_path(self) -> Path:
        return self.root / "original_row_id.npy"


def load_references(reference_dir: Path, strict_counts: bool = True) -> References:
    reference_dir = Path(reference_dir)
    audit_path = reference_dir / "valid_feature_audit.csv"
    quality_path = reference_dir / "feature_quality_refine.csv"
    ticker_path = reference_dir / "ticker_bucket_map.csv"
    folds_path = reference_dir / "outer_walk_forward_folds.json"
    if not audit_path.exists():
        raise FileNotFoundError(audit_path)
    audit = pd.read_csv(audit_path)
    valid = audit.copy()
    if "status" in valid.columns:
        valid = valid[valid["status"].astype(str).str.lower().eq("valid")]
    valid["feature"] = valid["feature"].astype(str)
    valid = valid.drop_duplicates("feature", keep="first").reset_index(drop=True)
    features = valid["feature"].tolist()
    groups: dict[str, list[str]] = {}
    if "group" in valid.columns:
        for group, part in valid.dropna(subset=["group"]).groupby("group", sort=True):
            groups[str(group)] = sorted(part["feature"].astype(str).tolist())
    ticker_map = pd.read_csv(ticker_path, dtype=str)
    ticker_to_bucket = {normalize_ticker(row.ticker): str(row.bucket) for row in ticker_map.itertuples(index=False)}
    folds = read_json(folds_path)
    if strict_counts:
        actual = {"valid_features": len(features), "tickers": len(ticker_to_bucket), "folds": len(folds)}
        expected = {"valid_features": 439, "tickers": 48, "folds": 8}
        if actual != expected:
            raise ValueError(f"reference count mismatch: expected={expected}, actual={actual}")
    hashes = {
        "valid_feature_audit": hash_file(audit_path),
        "feature_quality_refine": hash_file(quality_path) if quality_path.exists() else "",
        "ticker_bucket_map": hash_file(ticker_path),
        "folds": hash_file(folds_path),
        "all_valid_feature_hash": hash_strings(features),
    }
    return References(features, groups, valid, ticker_to_bucket, folds, hashes)


def discover_dataset(project_root: Path, configured: str, cli_dataset: str | None, sealed_tokens: list[str]) -> Path:
    candidates: list[Path] = []
    if cli_dataset:
        candidates.append(Path(cli_dataset).expanduser())
    if configured and configured.upper() != "AUTO":
        p = Path(configured).expanduser()
        candidates.append(p if p.is_absolute() else project_root / p)
    for p in candidates:
        if p.exists() and p.is_file():
            return _validate_dataset(p.resolve(), sealed_tokens)
    names = ["training_dataset_finance11h.parquet", "training_dataset.parquet"]
    roots = [project_root, project_root.parent, Path.home() / "Downloads"]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for name in names:
            direct = root / "crashwatch_ai_data" / "development" / name
            if direct.exists():
                found.append(direct)
        try:
            for name in names:
                found.extend(root.rglob(name))
        except (OSError, PermissionError):
            pass
    if not found:
        raise FileNotFoundError("training_dataset_finance11h.parquet를 찾지 못했습니다. --dataset 경로를 지정하세요.")
    unique = sorted({p.resolve() for p in found}, key=lambda p: (
        "development" not in str(p).lower(),
        "crashwatch_ai_data" not in str(p).lower(),
        len(str(p)),
    ))
    return _validate_dataset(unique[0], sealed_tokens)


def _validate_dataset(path: Path, sealed_tokens: list[str]) -> Path:
    lower = str(path).lower()
    if any(str(token).lower() in lower for token in sealed_tokens if str(token)):
        raise ValueError(f"sealed/holdout 경로는 사용할 수 없습니다: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Parquet 데이터만 지원합니다: {path}")
    return path


def _detect_column(names: list[str], requested: str, candidates: list[str], kind: str, required: bool = True) -> str | None:
    lower = {name.lower(): name for name in names}
    if requested and requested.upper() != "AUTO":
        if requested not in names:
            raise KeyError(f"configured {kind} column not found: {requested}")
        return requested
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    if required:
        raise KeyError(f"unable to auto-detect {kind} column")
    return None


def inspect_parquet(dataset_path: Path, config: dict[str, Any]) -> tuple[list[str], str, str, str | None]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow가 필요합니다. INSTALL_REQUIREMENTS.bat를 실행하세요.") from exc
    names = pq.ParquetFile(dataset_path).schema_arrow.names
    date_col = _detect_column(names, str(config.get("date_column", "AUTO")), DATE_CANDIDATES, "date")
    ticker_col = _detect_column(names, str(config.get("ticker_column", "AUTO")), TICKER_CANDIDATES, "ticker")
    bucket_col = _detect_column(names, str(config.get("bucket_column", "AUTO")), BUCKET_CANDIDATES, "bucket", required=False)
    return names, str(date_col), str(ticker_col), bucket_col


def prepare_data(
    dataset_path: Path,
    cache_dir: Path,
    config: dict[str, Any],
    refs: References,
    *,
    force: bool = False,
) -> PreparedData:
    parquet_columns, date_col, ticker_col, bucket_col = inspect_parquet(dataset_path, config)
    target_col = str(config["target_column"])
    if target_col not in parquet_columns:
        raise KeyError(f"target column missing: {target_col}")
    missing = [feature for feature in refs.all_valid_features if feature not in parquet_columns]
    if missing:
        raise KeyError(f"valid features missing from dataset ({len(missing)}): {missing[:30]}")
    safety_flag_columns = [name for name in SAFETY_FLAG_COLUMNS if name in parquet_columns]
    signature_payload = {
        "cache_version": CACHE_VERSION,
        "dataset_hash": hash_file(dataset_path, strong=bool(config.get("strong_dataset_hash", False)), length=32),
        "feature_hash": refs.hashes["all_valid_feature_hash"],
        "target": target_col,
        "date": date_col,
        "ticker": ticker_col,
        "safety_flag_columns": safety_flag_columns,
    }
    signature = canonical_hash(signature_payload, length=24)
    root = Path(cache_dir) / signature
    manifest_path = root / "cache_manifest.json"
    required_files = [
        root / "X_all_valid.npy", root / "target.npy", root / "dates_ns.npy", root / "tickers.npy",
        root / "buckets.npy", root / "original_row_id.npy", root / "feature_names.json",
    ]
    if not force and manifest_path.exists() and all(p.exists() and p.stat().st_size > 0 for p in required_files):
        manifest = read_json(manifest_path)
        if (
            manifest.get("status") == "complete"
            and manifest.get("signature") == signature
            and manifest.get("safety_flags_verified") is True
        ):
            feature_names = read_json(root / "feature_names.json")
            feature_groups = read_json(root / "feature_groups.json", {})
            LOGGER.info("matrix cache hit: %s", root)
            return PreparedData(root, dataset_path, signature, manifest, feature_names, feature_groups)
    root.mkdir(parents=True, exist_ok=True)
    if free_disk_gb(root) < float(config.get("minimum_free_disk_gb", 8.0)):
        raise RuntimeError(f"캐시 디스크 공간 부족: {free_disk_gb(root):.2f}GB")
    atomic_json({"status": "building", "signature": signature, "dataset_path": str(dataset_path)}, manifest_path)
    read_columns = list(dict.fromkeys(
        [date_col, ticker_col, target_col]
        + ([bucket_col] if bucket_col else [])
        + safety_flag_columns
        + refs.all_valid_features
    ))
    LOGGER.info("reading parquet once: rows/columns will be cached, columns=%s", len(read_columns))
    frame = pd.read_parquet(dataset_path, columns=read_columns, engine="pyarrow")
    for safety_column in safety_flag_columns:
        safety_values = pd.to_numeric(frame[safety_column], errors="coerce").fillna(0)
        unsafe_rows = int(safety_values.ne(0).sum())
        if unsafe_rows:
            raise ValueError(
                f"development safety flag violation: {safety_column} has {unsafe_rows} non-zero rows"
            )
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    if frame[date_col].isna().any():
        raise ValueError(f"date parse failed rows={int(frame[date_col].isna().sum())}")
    frame["__ticker_norm__"] = frame[ticker_col].map(normalize_ticker)
    frame["__row_id__"] = np.arange(len(frame), dtype=np.int64)
    frame.sort_values([date_col, "__ticker_norm__", "__row_id__"], kind="mergesort", inplace=True)
    frame.reset_index(drop=True, inplace=True)
    target = pd.to_numeric(frame[target_col], errors="coerce")
    valid_target = target.isin([0, 1])
    if not valid_target.all():
        LOGGER.warning("dropping invalid target rows=%s", int((~valid_target).sum()))
        frame = frame.loc[valid_target].reset_index(drop=True)
        target = target.loc[valid_target].reset_index(drop=True)
    mapped = frame["__ticker_norm__"].map(refs.ticker_to_bucket).fillna("")
    if bucket_col:
        bucket_values = frame[bucket_col].astype("string").fillna("").astype(str)
        bucket_values = bucket_values.where(bucket_values.ne(""), mapped)
    else:
        bucket_values = mapped
    unknown = sorted(frame.loc[bucket_values.eq(""), "__ticker_norm__"].unique().tolist())
    if unknown:
        raise ValueError(f"bucket mapping missing: {unknown}")
    # Numeric conversion is explicit; non-numeric residues become NaN and remain valid LightGBM missing values.
    feature_frame = frame[refs.all_valid_features].apply(pd.to_numeric, errors="coerce")
    X = feature_frame.to_numpy(dtype=np.float32, copy=True)
    non_finite_cells = int(np.sum(~np.isfinite(X) & ~np.isnan(X)))
    if non_finite_cells:
        LOGGER.warning("replacing +/-inf feature cells with NaN: %s", non_finite_cells)
        X[~np.isfinite(X)] = np.nan
    y = target.to_numpy(dtype=np.uint8, copy=True)
    dates_ns = frame[date_col].astype("datetime64[ns]").astype("int64").to_numpy(copy=True)
    tickers = frame["__ticker_norm__"].astype(str).to_numpy(dtype="U16")
    buckets = bucket_values.astype(str).to_numpy(dtype="U32")
    row_ids = frame["__row_id__"].to_numpy(dtype=np.int64, copy=True)
    np.save(root / "X_all_valid.npy", X, allow_pickle=False)
    np.save(root / "target.npy", y, allow_pickle=False)
    np.save(root / "dates_ns.npy", dates_ns, allow_pickle=False)
    np.save(root / "tickers.npy", tickers, allow_pickle=False)
    np.save(root / "buckets.npy", buckets, allow_pickle=False)
    np.save(root / "original_row_id.npy", row_ids, allow_pickle=False)
    atomic_json(refs.all_valid_features, root / "feature_names.json")
    group_lookup: dict[str, str] = {}
    for group, features in refs.groups.items():
        for feature in features:
            group_lookup.setdefault(feature, group)
    atomic_json(group_lookup, root / "feature_groups.json")
    manifest = {
        "status": "complete",
        "signature": signature,
        "dataset_path": str(dataset_path),
        "date_column": date_col,
        "ticker_column": ticker_col,
        "bucket_column": bucket_col,
        "target_column": target_col,
        "rows": int(len(y)),
        "features": int(X.shape[1]),
        "feature_hash": hash_strings(refs.all_valid_features),
        "date_min": pd.Timestamp(dates_ns.min()).isoformat(),
        "date_max": pd.Timestamp(dates_ns.max()).isoformat(),
        "tickers": int(len(np.unique(tickers))),
        "positive_rate": float(y.mean()),
        "missing_cells": int(np.isnan(X).sum()),
        "replaced_non_finite_cells": non_finite_cells,
        "safety_flag_columns": safety_flag_columns,
        "safety_flags_verified": True,
        "matrix_bytes": int(X.nbytes),
        "reference_hashes": refs.hashes,
    }
    atomic_json(manifest, manifest_path)
    del frame, feature_frame, X, y, dates_ns, tickers, buckets, row_ids
    return PreparedData(root, dataset_path, signature, manifest, refs.all_valid_features, group_lookup)
