#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


def _resolve_cache(record: dict, cache_root: Path) -> Path:
    original = Path(str(record.get("cache", "")))
    if original.exists():
        return original
    return cache_root / str(record.get("namespace", "baseline")) / original.name


def _prepare(path: Path) -> pd.DataFrame:
    pred = pd.read_parquet(path, columns=["date", "ticker", "bucket", "target", "prediction"])
    pred["date"] = pd.to_datetime(pred["date"], errors="coerce").dt.tz_localize(None)
    pred["ticker"] = pred["ticker"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    pred["target"] = pd.to_numeric(pred["target"], errors="coerce").astype(int)
    pred["prediction"] = pd.to_numeric(pred["prediction"], errors="coerce").clip(1e-7, 1 - 1e-7)
    pred["_rank"] = pred.groupby("date", sort=False)["prediction"].rank(method="first", ascending=False)
    pred["_daily_rows"] = pred.groupby("date", sort=False)["prediction"].transform("size")
    return pred


def _scope(pred: pd.DataFrame, record: dict) -> tuple[str, str, pd.DataFrame]:
    ticker = record.get("target_ticker")
    bucket = record.get("target_bucket")
    if ticker not in {None, "", "nan"}:
        value = str(ticker).zfill(6)
        return "ticker", value, pred.loc[pred["ticker"].eq(value)]
    if bucket not in {None, "", "nan"}:
        value = str(bucket)
        return "bucket", value, pred.loc[pred["bucket"].astype(str).eq(value)]
    return "all", "all_validation", pred


def _metrics(block: pd.DataFrame, fraction: float) -> dict[str, float]:
    if block.empty:
        return {k: np.nan for k in ["rows", "positives", "pr_auc", "roc_auc", "brier", "logloss", "alert_count", "alert_precision", "alert_recall"]}
    y = block["target"].to_numpy(dtype=int)
    p = block["prediction"].to_numpy(dtype=float)
    cutoff = np.maximum(1, np.ceil(block["_daily_rows"].to_numpy(dtype=float) * fraction))
    alert = block["_rank"].to_numpy(dtype=float) <= cutoff
    true_alert = int(np.sum(alert & (y == 1)))
    alert_count = int(np.sum(alert))
    positives = int(np.sum(y == 1))
    two_classes = np.unique(y).size == 2
    return {
        "rows": int(len(block)),
        "positives": positives,
        "pr_auc": float(average_precision_score(y, p)) if two_classes else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if two_classes else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "alert_count": alert_count,
        "alert_precision": true_alert / alert_count if alert_count else np.nan,
        "alert_recall": true_alert / positives if positives else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="완료된 V3 prediction cache로 경보예산 민감도 재분석")
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fractions", default="0.01,0.02,0.03,0.05,0.10")
    args = parser.parse_args()

    source = args.source_data_root.resolve()
    output = args.output_root.resolve() / "legacy_cache_reanalysis"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = source / "ablation_dual" / "run_manifest.json"
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    unique: dict[tuple, dict] = {}
    for record in records:
        key = (record.get("experiment"), record.get("fold"), record.get("seed"))
        unique[key] = record
    records = list(unique.values())
    baseline_records = {
        (int(r["fold"]), int(r["seed"])): r for r in records if r.get("experiment") == "baseline"
    }
    cache_root = source / "ablation_dual" / "prediction_cache"
    fractions = [float(x) for x in args.fractions.split(",") if x.strip()]
    baseline_frames: dict[tuple[int, int], pd.DataFrame] = {}
    baseline_metric_cache: dict[tuple, dict[str, float]] = {}
    rows: list[dict] = []
    missing = invalid = 0

    for index, record in enumerate(records, start=1):
        if record.get("experiment") == "baseline":
            continue
        fold, seed = int(record["fold"]), int(record["seed"])
        path = _resolve_cache(record, cache_root)
        if not path.exists():
            missing += 1
            continue
        try:
            pred = _prepare(path)
            scope_type, scope_value, block = _scope(pred, record)
            base_key = (fold, seed)
            if base_key not in baseline_frames:
                base_record = baseline_records[base_key]
                base_path = _resolve_cache(base_record, cache_root)
                baseline_frames[base_key] = _prepare(base_path)
            _, _, baseline_block = _scope(baseline_frames[base_key], record)
            for fraction in fractions:
                metric_key = (fold, seed, scope_type, scope_value, fraction)
                if metric_key not in baseline_metric_cache:
                    baseline_metric_cache[metric_key] = _metrics(baseline_block, fraction)
                base = baseline_metric_cache[metric_key]
                ablated = _metrics(block, fraction)
                rows.append({
                    "experiment": record.get("experiment"),
                    "namespace": record.get("namespace"),
                    "ablation_mode": record.get("ablation_mode"),
                    "target_group": record.get("target_group"),
                    "target_bucket": record.get("target_bucket"),
                    "target_ticker": record.get("target_ticker"),
                    "fold": fold,
                    "seed": seed,
                    "scope_type": scope_type,
                    "scope_value": scope_value,
                    "top_fraction": fraction,
                    "rows": ablated["rows"],
                    "positives": ablated["positives"],
                    "pr_auc_loss_when_removed": base["pr_auc"] - ablated["pr_auc"],
                    "roc_auc_loss_when_removed": base["roc_auc"] - ablated["roc_auc"],
                    "brier_increase_when_removed": ablated["brier"] - base["brier"],
                    "logloss_increase_when_removed": ablated["logloss"] - base["logloss"],
                    "alert_precision_loss_when_removed": base["alert_precision"] - ablated["alert_precision"],
                    "alert_recall_loss_when_removed": base["alert_recall"] - ablated["alert_recall"],
                })
        except Exception:
            invalid += 1
        if index % 200 == 0 and rows:
            frame = pd.DataFrame(rows)
            temp = output / "alert_budget_deltas.csv.tmp"
            frame.to_csv(temp, index=False, encoding="utf-8-sig")
            os.replace(temp, output / "alert_budget_deltas.csv")

    detail = pd.DataFrame(rows)
    detail_path = output / "alert_budget_deltas.csv"
    temp = detail_path.with_suffix(".csv.tmp")
    detail.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, detail_path)
    value_cols = [c for c in detail.columns if c.endswith(("_when_removed",))]
    group_cols = ["namespace", "ablation_mode", "target_group", "scope_type", "top_fraction"]
    summary = detail.groupby(group_cols, dropna=False)[value_cols].agg(["count", "mean", "median", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary_path = output / "alert_budget_summary.csv"
    temp = summary_path.with_suffix(".csv.tmp")
    summary.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, summary_path)
    report = {
        "created_at": pd.Timestamp.now(tz="Asia/Seoul").isoformat(),
        "source_manifest": str(manifest_path),
        "cache_records_reused": int(detail[["experiment", "fold", "seed"]].drop_duplicates().shape[0]),
        "detail_rows": int(len(detail)),
        "fractions": fractions,
        "missing_cache_records": missing,
        "invalid_cache_records": invalid,
        "output": str(output),
    }
    report_path = output / "run_summary.json"
    temp = report_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp, report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
