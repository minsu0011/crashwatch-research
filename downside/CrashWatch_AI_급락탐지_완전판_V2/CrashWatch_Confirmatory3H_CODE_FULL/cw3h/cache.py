from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import RuntimePaths
from .references import ReferenceBundle, battery_mask_groups, condition_feature_lists
from .utils import atomic_json, canonical_json_hash, free_disk_gb, hash_file, hash_strings, normalize_ticker, read_json

LOGGER = logging.getLogger(__name__)
CACHE_FORMAT_VERSION = "cw3h_matrix_v3"
DATE_CANDIDATES = ["date", "trade_date", "trading_date", "datetime", "dt", "일자", "날짜"]
TICKER_CANDIDATES = ["ticker", "stock_code", "code", "symbol", "종목코드", "단축코드"]
BUCKET_CANDIDATES = ["bucket", "sector_bucket", "industry_bucket", "sector", "industry", "업종"]

@dataclass(frozen=True)
class PreparedCache:
    root: Path
    dataset_path: Path
    dataset_signature: str
    date_column: str
    ticker_column: str
    bucket_column: str | None
    manifest: dict[str, Any]

    def profile_dir(self, profile: str) -> Path:
        return self.root / "profiles" / profile

    def matrix_path(self, profile: str, condition: str) -> Path:
        return self.profile_dir(profile) / "matrices" / f"{condition}.npy"

    def profile_manifest(self, profile: str) -> dict[str, Any]:
        return read_json(self.profile_dir(profile) / "profile_manifest.json")


def discover_dataset(config: dict[str, Any], paths: RuntimePaths, cli_dataset: str | None) -> Path:
    candidates: list[Path] = []
    if cli_dataset:
        candidates.append(Path(cli_dataset).expanduser())
    configured = str(config.get("dataset_path", "AUTO"))
    if configured and configured.upper() != "AUTO":
        configured_path = Path(configured).expanduser()
        candidates.append(configured_path if configured_path.is_absolute() else paths.project_root / configured_path)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return _validate_development_path(candidate.resolve(), config)

    relative = Path("crashwatch_ai_data") / "development" / "training_dataset_finance11h.parquet"
    roots = [paths.project_root, paths.package_root.parent, *list(paths.project_root.parents)[:3]]
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        for candidate in [root / relative, root / "development" / "training_dataset_finance11h.parquet"]:
            if candidate.exists():
                return _validate_development_path(candidate.resolve(), config)

    found: list[Path] = []
    for root in [paths.project_root, Path.home() / "Downloads"]:
        if not root.exists():
            continue
        try:
            found.extend(root.rglob("training_dataset_finance11h.parquet"))
        except (OSError, PermissionError):
            pass
    if not found:
        raise FileNotFoundError("training_dataset_finance11h.parquet를 찾지 못했습니다. --dataset으로 직접 지정하세요.")
    found = [p.resolve() for p in found]
    found.sort(key=lambda p: ("development" not in str(p).lower(), "crashwatch_ai_data" not in str(p).lower(), len(str(p))))
    return _validate_development_path(found[0], config)


def _validate_development_path(path: Path, config: dict[str, Any]) -> Path:
    lower = str(path).lower()
    if any(str(token).lower() in lower for token in config.get("sealed_path_tokens", []) if str(token)):
        raise ValueError(f"sealed/holdout 경로 사용 금지: {path}")
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"dataset must be parquet: {path}")
    return path


def _detect_column(names: list[str], requested: str, candidates: list[str], kind: str, required: bool = True) -> str | None:
    lower_map = {name.lower(): name for name in names}
    if requested and requested.upper() != "AUTO":
        if requested not in names:
            raise KeyError(f"configured {kind} column not found: {requested}")
        return requested
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise KeyError(f"unable to auto-detect {kind} column")
    return None


def inspect_columns(dataset_path: Path, config: dict[str, Any]) -> tuple[list[str], str, str, str | None]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow가 필요합니다. INSTALL_REQUIREMENTS.bat 또는 pip install -r requirements.txt를 실행하세요.") from exc
    names = pq.ParquetFile(dataset_path).schema_arrow.names
    date_col = _detect_column(names, str(config.get("date_column", "AUTO")), DATE_CANDIDATES, "date")
    ticker_col = _detect_column(names, str(config.get("ticker_column", "AUTO")), TICKER_CANDIDATES, "ticker")
    bucket_col = _detect_column(names, str(config.get("bucket_column", "AUTO")), BUCKET_CANDIDATES, "bucket", required=False)
    return names, str(date_col), str(ticker_col), bucket_col


def compute_dataset_signature(dataset_path: Path, parquet_columns: list[str], refs: ReferenceBundle, config: dict[str, Any]) -> str:
    payload = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "file_hash": hash_file(dataset_path, strong=bool(config.get("strong_dataset_hash", False)), length=32),
        "columns_hash": hash_strings(parquet_columns, length=32),
        "target": config["target_column"],
        "reference_hashes": refs.hashes,
        "common_start_date": config["common_start_date"],
        "common_period_mode": config["common_period_mode"]
    }
    return canonical_json_hash(payload, length=24)


def prepare_cache(dataset_path: Path, paths: RuntimePaths, config: dict[str, Any], refs: ReferenceBundle, force: bool = False) -> PreparedCache:
    parquet_columns, date_col, ticker_col, bucket_col = inspect_columns(dataset_path, config)
    target_col = str(config["target_column"])
    if target_col not in parquet_columns:
        raise KeyError(f"target column not found: {target_col}")
    missing_features = [f for f in refs.valid_features if f not in parquet_columns]
    if missing_features:
        raise KeyError(f"reference features missing ({len(missing_features)}): {missing_features[:20]}")

    dataset_signature = compute_dataset_signature(dataset_path, parquet_columns, refs, config)
    root = paths.cache_dir / dataset_signature
    manifest_path = root / "cache_manifest.json"
    if not force and manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("status") == "complete" and _cache_files_complete(root, manifest):
            LOGGER.info("prepared matrix cache hit: %s", root)
            return PreparedCache(root, dataset_path, dataset_signature, date_col, ticker_col, bucket_col, manifest)
        LOGGER.warning("incomplete cache found; rebuilding: %s", root)

    root.mkdir(parents=True, exist_ok=True)
    if free_disk_gb(root) < float(config.get("minimum_free_disk_gb", 8.0)):
        raise RuntimeError(f"캐시 디스크 공간 부족: {free_disk_gb(root):.2f}GB")
    (root / "BUILDING").write_text("building", encoding="utf-8")
    manifest: dict[str, Any] = {
        "status": "building",
        "dataset_path": str(dataset_path),
        "dataset_signature": dataset_signature,
        "date_column": date_col,
        "ticker_column": ticker_col,
        "bucket_column": bucket_col,
        "target_column": target_col,
        "reference_hashes": refs.hashes,
        "config_hash": config.get("config_hash")
    }
    atomic_json(manifest, manifest_path)

    read_columns = list(dict.fromkeys([date_col, ticker_col, target_col] + ([bucket_col] if bucket_col else []) + refs.valid_features))
    LOGGER.info("parquet single read: %s columns", len(read_columns))
    frame = pd.read_parquet(dataset_path, columns=read_columns, engine="pyarrow")
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    if frame[date_col].isna().any():
        raise ValueError(f"date parse failure rows={int(frame[date_col].isna().sum())}")
    frame["__ticker_norm__"] = frame[ticker_col].map(normalize_ticker)
    frame["__row_id__"] = np.arange(len(frame), dtype=np.int64)
    frame.sort_values([date_col, "__ticker_norm__", "__row_id__"], kind="mergesort", inplace=True)
    frame.reset_index(drop=True, inplace=True)

    mapped_bucket = frame["__ticker_norm__"].map(refs.ticker_to_bucket).fillna("")
    if bucket_col:
        buckets = frame[bucket_col].astype("string").fillna("").astype(str)
        buckets = buckets.where(buckets.ne(""), mapped_bucket)
    else:
        buckets = mapped_bucket
    if (buckets == "").any():
        unknown = sorted(frame.loc[buckets == "", "__ticker_norm__"].unique().tolist())
        raise ValueError(f"bucket mapping missing: {unknown}")

    y_numeric = pd.to_numeric(frame[target_col], errors="coerce")
    valid_label = y_numeric.isin([0, 1])
    if not valid_label.all():
        LOGGER.warning("dropping invalid target rows=%s", int((~valid_label).sum()))
        frame = frame.loc[valid_label].reset_index(drop=True)
        buckets = buckets.loc[valid_label].reset_index(drop=True)
        y_numeric = y_numeric.loc[valid_label].reset_index(drop=True)

    canonical_dir = root / "canonical"
    canonical_dir.mkdir(parents=True, exist_ok=True)
    dates_ns = frame[date_col].astype("datetime64[ns]").astype("int64").to_numpy(copy=True)
    tickers = frame["__ticker_norm__"].astype(str).to_numpy(dtype="U16")
    bucket_values = buckets.astype(str).to_numpy(dtype="U32")
    y = y_numeric.to_numpy(dtype=np.uint8, copy=True)
    row_ids = frame["__row_id__"].to_numpy(dtype=np.int64, copy=True)
    np.save(canonical_dir / "dates_ns.npy", dates_ns, allow_pickle=False)
    np.save(canonical_dir / "tickers.npy", tickers, allow_pickle=False)
    np.save(canonical_dir / "buckets.npy", bucket_values, allow_pickle=False)
    np.save(canonical_dir / "target.npy", y, allow_pickle=False)
    np.save(canonical_dir / "original_row_id.npy", row_ids, allow_pickle=False)

    canonical_matrix = canonical_dir / "X_valid.npy"
    _write_frame_matrix(frame, refs.valid_features, canonical_matrix, int(config.get("matrix_chunk_rows", 8192)))
    atomic_json({"features": refs.valid_features, "feature_hash": hash_strings(refs.valid_features)}, canonical_dir / "feature_manifest.json")

    profiles = {"full_reduced": refs.full_reduced_features, "common_period": refs.common_features}
    global_conditions = condition_feature_lists(refs)
    battery_groups = battery_mask_groups(refs)
    valid_index = {f: i for i, f in enumerate(refs.valid_features)}
    profile_manifests: dict[str, Any] = {}

    for profile, profile_features in profiles.items():
        LOGGER.info("building profile cache: %s", profile)
        profile_dir = root / "profiles" / profile
        matrix_dir = profile_dir / "matrices"
        matrix_dir.mkdir(parents=True, exist_ok=True)
        if profile == "common_period" and config.get("common_period_mode") == "strict_start_date":
            start_ns = pd.Timestamp(config["common_start_date"]).value
            row_indices = np.flatnonzero(dates_ns >= start_ns).astype(np.int64)
        else:
            row_indices = np.arange(len(frame), dtype=np.int64)
        np.save(profile_dir / "canonical_row_indices.npy", row_indices, allow_pickle=False)
        np.save(profile_dir / "dates_ns.npy", dates_ns[row_indices], allow_pickle=False)
        np.save(profile_dir / "tickers.npy", tickers[row_indices], allow_pickle=False)
        np.save(profile_dir / "buckets.npy", bucket_values[row_indices], allow_pickle=False)
        np.save(profile_dir / "target.npy", y[row_indices], allow_pickle=False)
        np.save(profile_dir / "original_row_id.npy", row_ids[row_indices], allow_pickle=False)

        base_indices = np.asarray([valid_index[f] for f in profile_features], dtype=np.int64)
        _write_subset_matrix(canonical_matrix, row_indices, base_indices, matrix_dir / "B0.npy", int(config.get("matrix_chunk_rows", 8192)))
        condition_manifests: dict[str, Any] = {
            "B0": {"features": profile_features, "feature_hash": hash_strings(profile_features), "mode": "baseline"}
        }
        profile_index = {f: i for i, f in enumerate(profile_features)}
        for condition, removed_features in global_conditions.items():
            if condition == "B0":
                continue
            kept = [f for f in profile_features if f not in set(removed_features)]
            keep_indices = np.asarray([profile_index[f] for f in kept], dtype=np.int64)
            _write_subset_matrix(matrix_dir / "B0.npy", None, keep_indices, matrix_dir / f"{condition}.npy", int(config.get("matrix_chunk_rows", 8192)))
            condition_manifests[condition] = {
                "features": kept,
                "feature_hash": hash_strings(kept),
                "removed_features": sorted(set(profile_features) - set(kept)),
                "mode": "global_drop"
            }

        if profile == "common_period":
            profile_buckets = bucket_values[row_indices]
            battery_rows = profile_buckets == "battery_materials"
            for condition, mask_features in battery_groups.items():
                available = [f for f in mask_features if f in profile_index]
                mask_indices = np.asarray([profile_index[f] for f in available], dtype=np.int64)
                _write_masked_copy(matrix_dir / "B0.npy", matrix_dir / f"{condition}.npy", battery_rows, mask_indices, int(config.get("matrix_chunk_rows", 8192)))
                condition_manifests[condition] = {
                    "features": profile_features,
                    "feature_hash": hash_strings(profile_features),
                    "masked_features": available,
                    "masked_bucket": "battery_materials",
                    "mode": "bucket_mask"
                }

        profile_manifest = {
            "profile": profile,
            "rows": int(len(row_indices)),
            "date_min": pd.Timestamp(dates_ns[row_indices].min()).isoformat() if len(row_indices) else None,
            "date_max": pd.Timestamp(dates_ns[row_indices].max()).isoformat() if len(row_indices) else None,
            "base_feature_count": len(profile_features),
            "base_feature_hash": hash_strings(profile_features),
            "common_period_mode": config.get("common_period_mode"),
            "conditions": condition_manifests
        }
        atomic_json(profile_manifest, profile_dir / "profile_manifest.json")
        profile_manifests[profile] = profile_manifest

    manifest.update({
        "status": "complete",
        "rows": int(len(frame)),
        "tickers": int(pd.Series(tickers).nunique()),
        "date_min": pd.Timestamp(dates_ns.min()).isoformat(),
        "date_max": pd.Timestamp(dates_ns.max()).isoformat(),
        "positive_rate": float(y.mean()),
        "valid_feature_count": len(refs.valid_features),
        "profiles": profile_manifests,
        "cache_bytes": _directory_size(root)
    })
    atomic_json(manifest, manifest_path)
    (root / "BUILDING").unlink(missing_ok=True)
    LOGGER.info("prepared cache complete: %.2fGB", manifest["cache_bytes"] / 1024**3)
    return PreparedCache(root, dataset_path, dataset_signature, date_col, ticker_col, bucket_col, manifest)


def _write_frame_matrix(frame: pd.DataFrame, features: list[str], target_path: Path, chunk_rows: int) -> None:
    target_path.unlink(missing_ok=True)
    temp_path = target_path.with_suffix(".tmp.npy")
    temp_path.unlink(missing_ok=True)
    mmap = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float32, shape=(len(frame), len(features)))
    for start in range(0, len(frame), chunk_rows):
        end = min(len(frame), start + chunk_rows)
        values = frame.iloc[start:end][features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
        mmap[start:end] = values
    mmap.flush()
    del mmap
    os.replace(temp_path, target_path)


def _write_subset_matrix(source_path: Path, row_indices: np.ndarray | None, column_indices: np.ndarray, target_path: Path, chunk_rows: int) -> None:
    target_path.unlink(missing_ok=True)
    source = np.load(source_path, mmap_mode="r")
    n_rows = len(row_indices) if row_indices is not None else source.shape[0]
    temp_path = target_path.with_suffix(".tmp.npy")
    temp_path.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float32, shape=(n_rows, len(column_indices)))
    for start in range(0, n_rows, chunk_rows):
        end = min(n_rows, start + chunk_rows)
        source_rows = source[start:end] if row_indices is None else source[row_indices[start:end]]
        target[start:end] = source_rows[:, column_indices]
    target.flush()
    del target, source
    os.replace(temp_path, target_path)


def _write_masked_copy(source_path: Path, target_path: Path, row_mask: np.ndarray, column_indices: np.ndarray, chunk_rows: int) -> None:
    source = np.load(source_path, mmap_mode="r")
    temp_path = target_path.with_suffix(".tmp.npy")
    temp_path.unlink(missing_ok=True)
    target = np.lib.format.open_memmap(temp_path, mode="w+", dtype=np.float32, shape=source.shape)
    for start in range(0, source.shape[0], chunk_rows):
        end = min(source.shape[0], start + chunk_rows)
        block = np.array(source[start:end], dtype=np.float32, copy=True, order="C")
        local_rows = np.flatnonzero(row_mask[start:end])
        if len(local_rows) and len(column_indices):
            block[np.ix_(local_rows, column_indices)] = np.nan
        target[start:end] = block
    target.flush()
    del target, source
    os.replace(temp_path, target_path)


def _cache_files_complete(root: Path, manifest: dict[str, Any]) -> bool:
    required = [root / "canonical" / "X_valid.npy", root / "canonical" / "dates_ns.npy", root / "canonical" / "target.npy"]
    for profile, profile_manifest in manifest.get("profiles", {}).items():
        profile_dir = root / "profiles" / profile
        required += [profile_dir / "profile_manifest.json", profile_dir / "dates_ns.npy", profile_dir / "target.npy"]
        for condition in profile_manifest.get("conditions", {}):
            required.append(profile_dir / "matrices" / f"{condition}.npy")
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def _directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
