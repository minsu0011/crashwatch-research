from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TARGET_NAME = "label_abs_surge_3d_5pct"
COMPARISON_ATOL = 1e-12


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


def atomic_json(data: object, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def build_target(frame: pd.DataFrame, horizon: int, threshold: float) -> tuple[pd.DataFrame, dict[str, object]]:
    if horizon < 1:
        raise ValueError("horizon must be at least one trading day")
    if threshold <= 0:
        raise ValueError("surge threshold must be positive")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if work["date"].isna().any():
        raise ValueError(f"date parse failed rows={int(work['date'].isna().sum())}")
    work["ticker"] = work["ticker"].map(normalize_ticker)
    work["t_price_ret_1"] = pd.to_numeric(work["t_price_ret_1"], errors="coerce")
    work["source_row_id"] = np.arange(len(work), dtype=np.int64)

    rows = len(work)
    label = np.zeros(rows, dtype=np.uint8)
    valid = np.zeros(rows, dtype=bool)
    first_hit = np.zeros(rows, dtype=np.int8)
    best_forward_return = np.full(rows, np.nan, dtype=np.float32)

    for _, part in work.groupby("ticker", sort=False):
        part = part.sort_values(["date", "source_row_id"], kind="mergesort")
        source_rows = part["source_row_id"].to_numpy(dtype=np.int64)
        returns = part["t_price_ret_1"].to_numpy(dtype=np.float64)
        for local_index in range(max(0, len(part) - horizon)):
            future = returns[local_index + 1 : local_index + 1 + horizon]
            if len(future) != horizon or not np.isfinite(future).all():
                continue
            cumulative = np.cumprod(1.0 + future) - 1.0
            source_index = source_rows[local_index]
            valid[source_index] = True
            best_forward_return[source_index] = float(np.max(cumulative))
            # Exact +5% moves can land a few ulps below 0.05 after multiplying
            # daily returns. Treat the mathematical boundary as inclusive.
            hits = np.flatnonzero(cumulative >= threshold - COMPARISON_ATOL)
            if hits.size:
                label[source_index] = 1
                first_hit[source_index] = np.int8(int(hits[0]) + 1)

    output = pd.DataFrame(
        {
            "source_row_id": work["source_row_id"].to_numpy(dtype=np.int64),
            "date": work["date"].to_numpy(),
            "ticker": work["ticker"].to_numpy(),
            TARGET_NAME: label,
            "target_valid": valid,
            "first_hit_day": first_hit,
            "best_forward_return_3d": best_forward_return,
        }
    )
    positives = int(label[valid].sum())
    valid_rows = int(valid.sum())
    audit = {
        "target_name": TARGET_NAME,
        "definition": f"max cumulative close-to-close return over t+1..t+{horizon} >= +{threshold:.4f}",
        "return_input": "t_price_ret_1[t] = close_t / close_(t-1) - 1",
        "horizon_trading_days": horizon,
        "rise_threshold": threshold,
        "inclusive_comparison_absolute_tolerance": COMPARISON_ATOL,
        "rows": rows,
        "valid_rows": valid_rows,
        "invalid_incomplete_or_nonfinite_horizon_rows": rows - valid_rows,
        "positives": positives,
        "positive_rate": positives / valid_rows if valid_rows else None,
        "first_hit_day_counts": {str(day): int(np.sum(first_hit[valid] == day)) for day in range(1, horizon + 1)},
        "tickers": int(work["ticker"].nunique()),
        "date_min": work["date"].min().isoformat(),
        "date_max": work["date"].max().isoformat(),
        "leakage_rule": "target columns and all forward-return derivatives must never be model features",
    }
    return output, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a point-in-time sidecar target for a 3-day +5% surge")
    parser.add_argument("--input", required=True, help="training_dataset_finance11h.parquet")
    parser.add_argument("--output", required=True, help="output target sidecar parquet")
    parser.add_argument("--audit", default=None, help="optional audit JSON path")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    source = Path(args.input).resolve()
    destination = Path(args.output).resolve()
    audit_path = Path(args.audit).resolve() if args.audit else destination.with_name("SURGE_TARGET_AUDIT.json")
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f"output already exists; use --overwrite: {destination}")
    schema_names = pq.ParquetFile(source).schema_arrow.names
    required = ["date", "ticker", "t_price_ret_1"]
    missing = [column for column in required if column not in schema_names]
    if missing:
        raise KeyError(f"missing required columns: {missing}")
    columns = required + (["sealed_do_not_train_or_tune"] if "sealed_do_not_train_or_tune" in schema_names else [])
    frame = pd.read_parquet(source, columns=columns)
    if "sealed_do_not_train_or_tune" in frame.columns:
        safety = pd.to_numeric(frame["sealed_do_not_train_or_tune"], errors="coerce").fillna(0)
        unsafe = int(safety.ne(0).sum())
        if unsafe:
            raise ValueError(f"development safety flag violation: {unsafe} rows")
    target, audit = build_target(frame, int(args.horizon), float(args.threshold))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.parquet")
    target.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, destination)
    audit.update(
        {
            "source_file": source.name,
            "source_sha256": sha256_file(source),
            "target_file": destination.name,
            "target_sha256": sha256_file(destination),
        }
    )
    atomic_json(audit, audit_path)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
