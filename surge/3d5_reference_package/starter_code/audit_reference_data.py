from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_ticker(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.zfill(6) if text.isdigit() else text


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def close_target(frame: pd.DataFrame, horizon: int = 3, threshold: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.zeros(len(frame), dtype=np.uint8)
    valid = np.zeros(len(frame), dtype=bool)
    best = np.full(len(frame), np.nan, dtype=np.float64)
    indexed = frame.assign(source_row_id=np.arange(len(frame), dtype=np.int64))
    for _, part in indexed.groupby("ticker", sort=False):
        part = part.sort_values(["date", "source_row_id"], kind="mergesort")
        rows = part["source_row_id"].to_numpy(dtype=np.int64)
        close = pd.to_numeric(part["close"], errors="coerce").to_numpy(dtype=np.float64)
        for local_index in range(max(0, len(part) - horizon)):
            current = close[local_index]
            future = close[local_index + 1 : local_index + 1 + horizon]
            if not np.isfinite(current) or current <= 0 or len(future) != horizon or not np.isfinite(future).all():
                continue
            source_index = rows[local_index]
            valid[source_index] = True
            best[source_index] = float(np.max(future / current - 1.0))
            labels[source_index] = np.uint8(best[source_index] >= threshold)
    return labels, valid, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    dataset = Path(args.dataset).resolve()
    target_path = Path(args.target).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(dataset)
    columns = parquet.schema_arrow.names
    selected = ["date", "ticker", "close", "t_price_ret_1"]
    if "sealed_do_not_train_or_tune" in columns:
        selected.append("sealed_do_not_train_or_tune")
    source = pd.read_parquet(dataset, columns=selected)
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["ticker"] = source["ticker"].map(normalize_ticker)
    target = pd.read_parquet(target_path)
    target["date"] = pd.to_datetime(target["date"], errors="coerce")
    target["ticker"] = target["ticker"].map(normalize_ticker)

    row_ids_ok = bool(np.array_equal(target["source_row_id"].to_numpy(dtype=np.int64), np.arange(len(source), dtype=np.int64)))
    date_match = bool(np.array_equal(source["date"].to_numpy(), target["date"].to_numpy()))
    ticker_match = bool(np.array_equal(source["ticker"].to_numpy(), target["ticker"].to_numpy()))
    safety_nonzero = 0
    if "sealed_do_not_train_or_tune" in source.columns:
        safety = pd.to_numeric(source["sealed_do_not_train_or_tune"], errors="coerce").fillna(0)
        safety_nonzero = int(safety.ne(0).sum())

    direct_label, direct_valid, direct_best = close_target(source)
    side_valid = target["target_valid"].to_numpy(dtype=bool)
    side_label = target["label_abs_surge_3d_5pct"].to_numpy(dtype=np.uint8)
    common_valid = direct_valid & side_valid
    close_label_mismatches = int(np.sum(direct_label[common_valid] != side_label[common_valid]))
    validity_mismatches = int(np.sum(direct_valid != side_valid))
    mismatch_index = np.flatnonzero(common_valid & (direct_label != side_label))
    mismatch_rows = pd.DataFrame(
        {
            "source_row_id": mismatch_index,
            "date": source.loc[mismatch_index, "date"].to_numpy(),
            "ticker": source.loc[mismatch_index, "ticker"].to_numpy(),
            "close_direct_best_forward_return": direct_best[mismatch_index],
            "ret1_compounded_best_forward_return": target.loc[mismatch_index, "best_forward_return_3d"].to_numpy(),
            "close_direct_label": direct_label[mismatch_index],
            "ret1_compounded_label": side_label[mismatch_index],
        }
    )
    mismatch_rows.to_csv(output / "close_vs_ret1_target_mismatches.csv", index=False, encoding="utf-8-sig")

    quality = target.loc[target["target_valid"]].copy()
    ticker_quality = (
        quality.groupby("ticker", sort=True)
        .agg(
            valid_rows=("target_valid", "size"),
            positives=("label_abs_surge_3d_5pct", "sum"),
            positive_rate=("label_abs_surge_3d_5pct", "mean"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            mean_best_forward_return=("best_forward_return_3d", "mean"),
        )
        .reset_index()
    )
    ticker_quality.to_csv(output / "ticker_target_quality.csv", index=False, encoding="utf-8-sig")
    yearly = quality.assign(year=quality["date"].dt.year).groupby("year", sort=True).agg(valid_rows=("target_valid", "size"), positives=("label_abs_surge_3d_5pct", "sum"), positive_rate=("label_abs_surge_3d_5pct", "mean")).reset_index()
    yearly.to_csv(output / "yearly_target_quality.csv", index=False, encoding="utf-8-sig")
    atomic_text("\n".join(columns) + "\n", output / "PARQUET_COLUMNS.txt")

    prefix_counts: dict[str, int] = {}
    for name in columns:
        prefix = name.split("_", 1)[0] if "_" in name else "other"
        prefix_counts[prefix] = prefix_counts.get(prefix, 0) + 1
    audit = {
        "dataset_file": dataset.name,
        "dataset_bytes": dataset.stat().st_size,
        "dataset_sha256": sha256_file(dataset),
        "rows": int(parquet.metadata.num_rows),
        "columns": len(columns),
        "row_groups": int(parquet.metadata.num_row_groups),
        "date_min": source["date"].min().isoformat(),
        "date_max": source["date"].max().isoformat(),
        "tickers": int(source["ticker"].nunique()),
        "safety_flag_present": "sealed_do_not_train_or_tune" in columns,
        "safety_flag_nonzero_rows": safety_nonzero,
        "existing_label_columns": [name for name in columns if name.startswith("label_")],
        "column_prefix_counts": dict(sorted(prefix_counts.items())),
        "target_sidecar": target_path.name,
        "target_sha256": sha256_file(target_path),
        "sidecar_source_row_id_exact": row_ids_ok,
        "sidecar_date_exact": date_match,
        "sidecar_ticker_exact": ticker_match,
        "close_vs_return_target_common_valid_rows": int(common_valid.sum()),
        "close_vs_return_label_mismatch_rows": close_label_mismatches,
        "close_vs_return_validity_mismatch_rows": validity_mismatches,
    }
    if not row_ids_ok or not date_match or not ticker_match or safety_nonzero or close_label_mismatches:
        raise RuntimeError(f"dataset audit failed: {audit}")
    atomic_text(json.dumps(audit, ensure_ascii=False, indent=2), output / "DATASET_AUDIT.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
