from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE_REVISION = "v13_20260816_r04"
V10_CODE = HERE / "v10_2_repro"
for path in (HERE, V10_CODE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_surge_tickerwise_correlation_map_v10 as v10  # type: ignore
import run_surge_tickerwise_validation_v10_2 as v102  # type: ignore
from surge_ticker_common_v10 import load_folds  # type: ignore

from surge_v13_data import (
    DIRECTION_COLUMN,
    MOVE_COLUMN,
    TARGET_COLUMN,
    V11_LAG0_PRIMARY_DIRECTION_COLUMNS,
    build_preexposed_directed_features,
    build_self_state_features,
    build_v11_lag0_features,
    derive_future_path_labels_from_history,
    feature_family_audit,
    fit_ab_state,
    load_json,
    normalize_ticker,
    read_table,
    serialize_ab_specs,
    table_columns,
    transform_ab_evidence,
)
from surge_v13_models import (
    SCHEMA,
    V13Config,
    apply_policy,
    build_policy_score,
    combine_stage_probabilities,
    discover_ab_specs_v13,
    fit_binary_model,
    fold_score_metrics,
    gpu_preflight,
    make_sample_weights,
    match_direction_training_rows,
    oracle_top_k,
    past_oof_base_rank,
    predict_binary_model,
    predefined_configs,
    recalibrate_policy_from_prior_oof,
    rank_direction_features,
    rank_move_features,
    safe_pr_auc,
    safe_roc_auc,
    select_frozen_policy,
    select_ranked_features,
)


@dataclass
class RuntimeData:
    frame: pd.DataFrame
    raw_features: list[str]
    candidate_features: list[str]
    v10args: argparse.Namespace
    folds: list[Any]
    fold_index: dict[int, dict[str, np.ndarray]]
    validation_fold_id: np.ndarray
    base_oof: pd.DataFrame
    ab_specs: dict[str, list[Any]]
    lag0_features: pd.DataFrame
    self_features: pd.DataFrame
    preexposed_features: pd.DataFrame
    cache_fingerprint: str


@dataclass
class DerivedFoldFeatures:
    ab_train: pd.DataFrame
    ab_validation: pd.DataFrame


def log(message: str) -> None:
    print(f"[V13] {message}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False, compression=compression)
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def runtime_data_fingerprint(frame: pd.DataFrame, candidate_features: Sequence[str]) -> str:
    """Compact identity for resume caches; excludes feature values but binds row/label scope."""
    columns = [c for c in ["source_row_id", "date", "ticker", TARGET_COLUMN, MOVE_COLUMN, DIRECTION_COLUMN] if c in frame.columns]
    identity = frame[columns].copy()
    if "date" in identity:
        identity["date"] = pd.to_datetime(identity["date"], errors="raise").astype("int64")
    if "ticker" in identity:
        identity["ticker"] = identity["ticker"].astype(str).map(normalize_ticker)
    hashed = pd.util.hash_pandas_object(identity, index=False).to_numpy(np.uint64)
    digest = hashlib.sha1()
    digest.update(CACHE_REVISION.encode("utf-8"))
    digest.update(hashed.tobytes())
    digest.update("\n".join(map(str, candidate_features)).encode("utf-8"))
    return digest.hexdigest()[:24]


def ab_spec_fingerprint(specs: Mapping[str, Sequence[Any]]) -> str:
    rows: list[tuple[Any, ...]] = []
    for ticker in sorted(specs):
        for spec in specs[ticker]:
            rows.append((str(ticker), str(spec.node_id), int(spec.direction), float(spec.weight), int(spec.rank)))
    return hashlib.sha1(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def system_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "platform": platform.platform(), "python": sys.version,
        "cpu_count": os.cpu_count(), "timestamp_utc": utc_now(),
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        payload["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        payload["nvidia_smi"] = f"unavailable: {exc}"
    return payload


def build_v10_runtime_args(args: argparse.Namespace, scratch_output: Path) -> argparse.Namespace:
    runtime = v10.build_parser().parse_args([])
    runtime.package_root = Path(args.package_root).resolve()
    runtime.output = scratch_output
    runtime.dataset = Path(args.dataset).expanduser().resolve()
    runtime.target_sidecar = Path(args.target_sidecar).expanduser().resolve()
    runtime.folds = Path(args.folds).expanduser().resolve()
    runtime.feature_profile_manifest = Path(args.feature_profile_manifest).expanduser().resolve()
    runtime.feature_profile = str(args.feature_profile)
    runtime.require_full_439 = bool(args.require_full_439)
    runtime.device = "cpu"
    runtime.resume = True
    return runtime


def build_model_frame(args: argparse.Namespace, output: Path) -> tuple[pd.DataFrame, argparse.Namespace, list[str], list[str]]:
    v10args = build_v10_runtime_args(args, output / "_v10_runtime")
    columns = v10.table_columns(Path(v10args.dataset))
    features = v10.load_feature_universe(v10args, columns)
    frame = v10.load_dataset_frame(v10args, features)
    v10_output = Path(args.v10_output).expanduser().resolve()
    ticker_dtype = {v10args.ticker_column: str, "ticker": str}
    selected_sources = pd.read_csv(require_file(v10_output / "ticker_selected_source_features.csv", "V10 selected sources"), dtype=ticker_dtype)
    clusters = pd.read_csv(require_file(v10_output / "ticker_cluster_assignments.csv", "V10 clusters"), dtype=ticker_dtype)
    tickers = sorted(frame[v10args.ticker_column].astype(str).map(normalize_ticker).unique())
    matrix_payload = v102.load_matrix_payload(v10_output, tickers)
    transformed, _ = v10.build_selected_transforms(frame, selected_sources, clusters, matrix_payload, v10args)
    model_frame = v10.build_model_frame(frame, transformed, v10args).copy().reset_index(drop=True)
    model_frame["row_index"] = np.arange(len(model_frame), dtype=np.int64)
    model_frame[v10args.ticker_column] = model_frame[v10args.ticker_column].map(normalize_ticker)
    if v10args.ticker_column != "ticker":
        model_frame["ticker"] = model_frame[v10args.ticker_column]
    if v10args.date_column != "date":
        model_frame["date"] = model_frame[v10args.date_column]
    if v10args.market_column != "market":
        model_frame["market"] = model_frame[v10args.market_column]
    if v10args.bucket_column != "bucket":
        model_frame["bucket"] = model_frame[v10args.bucket_column]
    if v10args.target_column != TARGET_COLUMN:
        model_frame[TARGET_COLUMN] = model_frame[v10args.target_column]
    # V10.2's ticker-specific transforms are inherited rather than discarded.
    # Industry metadata was audited as invalid in V10.2, so date-industry ranks
    # are explicitly excluded.  Magnitude-family transforms remain available to
    # Stage 1, while Stage 2/A-B filters remove them by family.
    transformed_features = [
        str(column) for column in model_frame.columns
        if "__" in str(column)
        and not str(column).endswith("__date_industry_rank")
        and not str(column).endswith("__ticker_cluster_innovation")
    ]
    candidate_features = list(dict.fromkeys([*map(str, features), *transformed_features]))
    return model_frame, v10args, list(map(str, features)), candidate_features


def _resolve_reference_file(explicit_root: str | None, candidates: Sequence[str], fallback: Path) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        for name in candidates:
            candidate = root / name
            if candidate.exists():
                return candidate
    return require_file(fallback, "bundled reference")


def load_base_oof(v10_2_output: Path) -> pd.DataFrame:
    path = require_file(v10_2_output / "ticker_base_oof_predictions_v10_2.csv", "V10.2 full base OOF")
    frame = pd.read_csv(path, dtype={"ticker": str})
    frame["ticker"] = frame["ticker"].map(normalize_ticker)
    frame["fold_id"] = pd.to_numeric(frame["fold_id"], errors="coerce").astype("Int64")
    frame["row_index"] = pd.to_numeric(frame["row_index"], errors="coerce").astype("Int64")
    frame["source_row_id"] = pd.to_numeric(frame["source_row_id"], errors="coerce").astype("Int64")
    frame["base_score_raw"] = pd.to_numeric(frame["base_score_raw"], errors="coerce")
    frame = frame.loc[frame["fold_id"].notna() & frame["row_index"].notna()].copy()
    # Preserve LOW_EVIDENCE_NO_MODEL / INSUFFICIENT_FEATURES rows.  Dropping
    # them would recreate V12's silent scope contraction.  Their missing base
    # scores are audited and receive a neutral 0.5 past-rank fallback.
    duplicate = frame.duplicated(["fold_id", "row_index"], keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, ["fold_id", "row_index", "ticker"]].head(10).to_dict("records")
        raise RuntimeError(f"V10.2 base OOF has duplicate fold/row keys: {sample}")
    return frame.sort_values(["fold_id", "ticker", "row_index"], kind="mergesort").reset_index(drop=True)


def add_base_score(validation: pd.DataFrame, base_oof: pd.DataFrame, fold_id: int) -> pd.DataFrame:
    part = base_oof.loc[pd.to_numeric(base_oof["fold_id"], errors="coerce").eq(int(fold_id)), [
        "row_index", "source_row_id", "ticker", "base_score_raw"
    ]].copy()
    left = validation.copy()
    left["ticker"] = left["ticker"].map(normalize_ticker)
    merged = left.merge(part, on=["row_index", "source_row_id", "ticker"], how="left", validate="one_to_one")
    return merged


def build_fold_index(frame: pd.DataFrame, folds: Sequence[Any]) -> tuple[dict[int, dict[str, np.ndarray]], np.ndarray]:
    index = v10.build_global_fold_index(frame, folds, "date")
    validation_fold = np.full(len(frame), -1, dtype=np.int16)
    for fold in folds:
        positions = np.asarray(index[int(fold.fold_id)]["validation"], dtype=np.int64)
        overlap = validation_fold[positions] >= 0
        if overlap.any():
            raise RuntimeError(f"Validation windows overlap at fold {fold.fold_id}")
        validation_fold[positions] = int(fold.fold_id)
    return index, validation_fold


def audit_base_oof_scope(
    model_frame: pd.DataFrame,
    fold_index: Mapping[int, Mapping[str, np.ndarray]],
    folds: Sequence[Any],
    base_oof: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Verify that V10.2 OOF covers every expected validation row exactly once.

    V12 silently collapsed to six tickers after loading OOF.  V13 fails before
    model search if an expected ticker/fold/row is absent or mapped to a different
    source row. Rows with deliberately missing V10.2 scores remain in scope and
    receive a neutral base-rank fallback; they are audited rather than discarded.
    """
    expected_parts: list[pd.DataFrame] = []
    for fold in folds:
        fold_id = int(fold.fold_id)
        indices = np.asarray(fold_index[fold_id]["validation"], dtype=np.int64)
        part = model_frame.iloc[indices][["row_index", "source_row_id", "ticker"]].copy()
        part["fold_id"] = fold_id
        expected_parts.append(part)
    expected = pd.concat(expected_parts, ignore_index=True)
    expected["ticker"] = expected["ticker"].map(normalize_ticker)
    expected["fold_id"] = pd.to_numeric(expected["fold_id"], errors="raise").astype(np.int64)
    expected["row_index"] = pd.to_numeric(expected["row_index"], errors="raise").astype(np.int64)
    actual = base_oof[["fold_id", "row_index", "source_row_id", "ticker", "base_score_raw"]].copy()
    actual["ticker"] = actual["ticker"].map(normalize_ticker)
    actual["fold_id"] = pd.to_numeric(actual["fold_id"], errors="raise").astype(np.int64)
    actual["row_index"] = pd.to_numeric(actual["row_index"], errors="raise").astype(np.int64)

    keys = ["fold_id", "row_index", "ticker"]
    if expected.duplicated(keys).any():
        raise RuntimeError("Expected validation scope has duplicate fold/row/ticker keys")
    if actual.duplicated(keys).any():
        raise RuntimeError("V10.2 base OOF has duplicate fold/row/ticker keys")

    joined = expected.merge(
        actual, on=keys, how="left", validate="one_to_one", indicator=True,
        suffixes=("_expected", "_base"),
    )
    missing = joined["_merge"].ne("both")
    matched = joined["_merge"].eq("both")
    source_match = (
        joined["source_row_id_expected"].astype("string")
        .eq(joined["source_row_id_base"].astype("string"))
    )
    source_mismatch = matched & ~source_match
    finite_score = np.isfinite(pd.to_numeric(joined["base_score_raw"], errors="coerce").to_numpy(float))
    nonfinite = matched.to_numpy() & ~finite_score

    actual_scope = actual.merge(expected[keys], on=keys, how="left", indicator=True)
    extra = actual_scope["_merge"].ne("both")

    expected_group = expected.groupby(["fold_id", "ticker"], sort=True).size().rename("expected_rows").reset_index()
    actual_group = actual.groupby(["fold_id", "ticker"], sort=True).agg(
        base_rows=("row_index", "size"),
        finite_base_score=("base_score_raw", lambda x: int(np.isfinite(pd.to_numeric(x, errors="coerce")).sum())),
    ).reset_index()
    coverage = expected_group.merge(actual_group, on=["fold_id", "ticker"], how="outer")
    for column in ["expected_rows", "base_rows", "finite_base_score"]:
        coverage[column] = pd.to_numeric(coverage[column], errors="coerce").fillna(0).astype(int)
    coverage["missing_rows"] = coverage["expected_rows"] - coverage["base_rows"]
    coverage["nonfinite_base_score"] = coverage["base_rows"] - coverage["finite_base_score"]

    audit = {
        "expected_rows": int(len(expected)),
        "actual_rows": int(len(actual)),
        "missing_expected_rows": int(missing.sum()),
        "extra_rows": int(extra.sum()),
        "source_row_id_mismatches": int(source_mismatch.sum()),
        "nonfinite_base_score_rows": int(np.sum(nonfinite)),
        "expected_tickers": int(expected["ticker"].nunique()),
        "actual_tickers": int(actual["ticker"].nunique()),
        "expected_folds": sorted(expected["fold_id"].unique().astype(int).tolist()),
        "actual_folds": sorted(actual["fold_id"].unique().astype(int).tolist()),
        "complete": bool(not missing.any() and not extra.any() and not source_mismatch.any()),
        "finite_score_complete": bool(not np.any(nonfinite)),
        "finite_score_rate": float(1.0 - np.sum(nonfinite) / max(len(expected), 1)),
    }
    return audit, coverage.sort_values(["fold_id", "ticker"], kind="mergesort").reset_index(drop=True)


def prepare_runtime(args: argparse.Namespace, output: Path) -> tuple[RuntimeData, dict[str, Any]]:
    model_frame, v10args, raw_features, candidate_features = build_model_frame(args, output)
    if bool(args.require_full_439) and len(raw_features) != 439:
        raise RuntimeError(f"V13 expected inherited 439 features, found {len(raw_features)}")
    expected_rows = int(args.expected_target_valid_rows)
    expected_tickers = int(args.expected_ticker_count)
    actual_rows = int(len(model_frame))
    actual_tickers = int(model_frame["ticker"].astype(str).map(normalize_ticker).nunique())
    if expected_rows > 0 and actual_rows != expected_rows:
        raise RuntimeError(
            f"V13 target-valid scope mismatch: expected_rows={expected_rows}, actual_rows={actual_rows}. "
            "Use --expected-target-valid-rows 0 only for an intentionally updated dataset."
        )
    if expected_tickers > 0 and actual_tickers != expected_tickers:
        raise RuntimeError(
            f"V13 ticker scope mismatch: expected_tickers={expected_tickers}, actual_tickers={actual_tickers}. "
            "Use --expected-ticker-count 0 only for an intentionally changed universe."
        )
    # Reconstruct the symmetric path on the unfiltered source history.  The V10
    # target-valid frame omits each ticker's final three source dates, which are
    # still needed as D+1..D+3 returns for the last valid forecast dates.
    dataset_path = Path(args.dataset).expanduser().resolve()
    available_history = set(table_columns(dataset_path))
    history_columns = [
        c for c in ["source_row_id", v10args.date_column, v10args.ticker_column, "t_price_ret_1", "return_pct"]
        if c in available_history
    ]
    if "t_price_ret_1" not in history_columns and "return_pct" not in history_columns:
        raise RuntimeError("Source dataset has neither t_price_ret_1 nor return_pct for symmetric path reconstruction")
    return_history = read_table(dataset_path, columns=history_columns).rename(
        columns={v10args.date_column: "date", v10args.ticker_column: "ticker"}
    )
    model_frame, target_audit = derive_future_path_labels_from_history(
        model_frame, return_history, threshold=float(args.surge_threshold), target_column=TARGET_COLUMN,
        max_mismatch_rate=float(args.max_target_mismatch_rate), allow_mismatch=bool(args.allow_target_mismatch),
    )
    model_frame = model_frame.reset_index(drop=True)
    model_frame["ticker"] = model_frame["ticker"].map(normalize_ticker)
    model_frame["date"] = pd.to_datetime(model_frame["date"], errors="raise")
    model_frame["market"] = model_frame.get("market", pd.Series("UNKNOWN", index=model_frame.index)).astype("string").fillna("UNKNOWN")
    model_frame["bucket"] = model_frame.get("bucket", pd.Series("UNKNOWN", index=model_frame.index)).astype("string").fillna("UNKNOWN")

    folds = load_folds(Path(args.folds).expanduser().resolve())
    fold_index, validation_fold = build_fold_index(model_frame, folds)
    base_oof = load_base_oof(Path(args.v10_2_output).expanduser().resolve())
    model_tickers = set(model_frame["ticker"].astype(str).unique())
    base_tickers = set(base_oof["ticker"].astype(str).unique())
    base_scope_audit, base_coverage = audit_base_oof_scope(model_frame, fold_index, folds, base_oof)
    atomic_csv(output / "V10_2_BASE_OOF_COVERAGE_V13.csv", base_coverage)
    atomic_json(output / "V10_2_BASE_OOF_SCOPE_AUDIT_V13.json", base_scope_audit)
    if bool(args.require_full_base_oof) and (base_tickers != model_tickers or not bool(base_scope_audit["complete"])):
        missing = sorted(model_tickers - base_tickers)
        extra = sorted(base_tickers - model_tickers)
        raise RuntimeError(
            "V13 requires exact full-scope V10.2 base OOF coverage. "
            f"model_tickers={len(model_tickers)}, base_tickers={len(base_tickers)}, "
            f"missing_tickers={missing[:12]}, extra_tickers={extra[:12]}, "
            f"scope_audit={base_scope_audit}"
        )

    v10_2_output = Path(args.v10_2_output).expanduser().resolve()
    # Preserve the V10.2 100-node map as a reference-only artifact.  Its selection
    # process already inspected folds 3-4, so it is not eligible as a primary V13
    # input.  V13 rebuilds the A/B map from folds 0-2 after runtime preparation.
    ab_reference_candidates = [
        v10_2_output / "ticker_precision_separator_map_v10_2.csv",
        v10_2_output / "precision_selection_candidates_100.csv",
        HERE / "reference_v10_2" / "precision_selection_candidates_100.csv",
    ]
    ab_reference_path = next((p for p in ab_reference_candidates if p.exists()), None)
    if ab_reference_path is not None:
        ab_reference = pd.read_csv(ab_reference_path, dtype={"ticker": str})
        if "precision_separator_selection_candidate" in ab_reference.columns:
            flag = ab_reference["precision_separator_selection_candidate"].astype(str).str.lower().isin(["true", "1", "yes"])
            ab_reference = ab_reference.loc[flag].copy()
        atomic_csv(output / "V10_2_AB_REFERENCE_PREEXPOSED.csv", ab_reference)

    lag0_path = _resolve_reference_file(
        args.v11_1_output,
        ["maxstat_q_le_0p10.csv", "corrected_pair_maxstat_results_v11_1.csv", "maxstat_pair_results_v11_1.csv"],
        HERE / "reference_v11" / "maxstat_q_le_0p10.csv",
    )
    directed_path = _resolve_reference_file(
        args.v11_output,
        ["strong_directed_edges_7.csv", "strong_directed_edges.csv"],
        HERE / "reference_v11" / "strong_directed_edges_7.csv",
    )
    lag0_manifest = pd.read_csv(lag0_path, dtype={"ticker_a": str, "ticker_b": str})
    directed_manifest = pd.read_csv(directed_path, dtype={"leader": str, "follower": str})
    lag0_features, lag0_used = build_v11_lag0_features(model_frame, lag0_manifest)
    self_features = build_self_state_features(model_frame)
    preexposed_features, directed_used = build_preexposed_directed_features(model_frame, directed_manifest)

    atomic_json(output / "TARGET_RECONSTRUCTION_AUDIT_V13.json", target_audit)
    atomic_csv(output / "V13_V11_LAG0_EDGE_MANIFEST.csv", lag0_used)
    atomic_csv(output / "V13_V11_DIRECTED_PREEXPOSED_MANIFEST.csv", directed_used)
    family_audit = feature_family_audit(candidate_features)
    family_audit["origin"] = np.where(family_audit["feature"].astype(str).str.contains("__", regex=False), "V10_TICKER_TRANSFORM", "RAW_439")
    family_audit["eligible_stage1"] = ~family_audit["family"].isin(["metadata", "direction"])
    family_audit["eligible_stage2_raw"] = ~family_audit["family"].isin(["metadata", "magnitude"])
    atomic_csv(output / "V13_FEATURE_FAMILY_AUDIT.csv", family_audit)
    runtime = RuntimeData(
        frame=model_frame, raw_features=list(raw_features), candidate_features=list(candidate_features), v10args=v10args, folds=list(folds),
        fold_index=fold_index, validation_fold_id=validation_fold, base_oof=base_oof,
        ab_specs={}, lag0_features=lag0_features, self_features=self_features,
        preexposed_features=preexposed_features,
        cache_fingerprint=runtime_data_fingerprint(model_frame, candidate_features),
    )
    audit = {
        "schema": SCHEMA, "rows_full_target_valid": int(len(model_frame)),
        "expected_target_valid_rows": int(args.expected_target_valid_rows),
        "target_valid_row_scope_exact": bool(int(args.expected_target_valid_rows) <= 0 or len(model_frame) == int(args.expected_target_valid_rows)),
        "tickers_full": int(model_frame["ticker"].nunique()),
        "expected_ticker_count": int(args.expected_ticker_count),
        "ticker_scope_exact": bool(int(args.expected_ticker_count) <= 0 or model_frame["ticker"].nunique() == int(args.expected_ticker_count)),
        "raw_features": int(len(raw_features)),
        "v10_ticker_transformed_features": int(len(candidate_features) - len(raw_features)),
        "candidate_features_total": int(len(candidate_features)),
        "date_industry_rank_transforms_excluded": True,
        "full_history_cluster_innovation_transforms_excluded": True,
        "folds": sorted(int(x.fold_id) for x in folds),
        "base_oof_rows": int(len(base_oof)), "base_oof_tickers": int(base_oof["ticker"].nunique()),
        "require_full_base_oof": bool(args.require_full_base_oof),
        "base_oof_scope_complete": bool(base_scope_audit["complete"]),
        "base_oof_expected_rows": int(base_scope_audit["expected_rows"]),
        "base_oof_missing_expected_rows": int(base_scope_audit["missing_expected_rows"]),
        "base_oof_extra_rows": int(base_scope_audit["extra_rows"]),
        "base_oof_source_row_id_mismatches": int(base_scope_audit["source_row_id_mismatches"]),
        "base_oof_nonfinite_score_rows": int(base_scope_audit["nonfinite_base_score_rows"]),
        "base_oof_finite_score_rate": float(base_scope_audit["finite_score_rate"]),
        "base_oof_nonfinite_policy": "retain row; neutral base_past_rank=0.5; never filter model scope",
        "ab_spec_tickers": 0, "ab_spec_nodes": 0,
        "v11_lag0_edges_used": int(len(lag0_used)),
        "v11_lag0_primary_direction_columns": list(V11_LAG0_PRIMARY_DIRECTION_COLUMNS),
        "v11_directed_preexposed_edges": int(len(directed_used)),
        "primary_uses_directed_v11_edges": False,
        "six_ticker_probe_filter_applied": False,
    }
    atomic_json(output / "DATA_AUDIT_V13.json", audit)
    return runtime, target_audit


def discovery_validation_frame(runtime: RuntimeData, discovery_folds: Sequence[int]) -> pd.DataFrame:
    mask = np.isin(runtime.validation_fold_id, np.asarray(discovery_folds, dtype=int))
    part = runtime.frame.loc[mask].copy()
    part["fold_id"] = runtime.validation_fold_id[mask].astype(int)
    return part.reset_index(drop=True)


def discovery_validation_with_base(runtime: RuntimeData, discovery_folds: Sequence[int]) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fold_id in discovery_folds:
        indices = np.asarray(runtime.fold_index[int(fold_id)]["validation"], dtype=np.int64)
        part = runtime.frame.iloc[indices].copy().reset_index(drop=True)
        part = add_base_score(part, runtime.base_oof, int(fold_id))
        part["fold_id"] = int(fold_id)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _derived_fold_features(runtime: RuntimeData, fold_id: int, cache: dict[int, DerivedFoldFeatures]) -> DerivedFoldFeatures:
    if fold_id in cache:
        return cache[fold_id]
    train_idx = runtime.fold_index[fold_id]["train"]
    valid_idx = runtime.fold_index[fold_id]["validation"]
    train_full = runtime.frame.iloc[train_idx]
    valid_full = runtime.frame.iloc[valid_idx]
    state = fit_ab_state(train_full, runtime.ab_specs, max_slots=max(len(x) for x in runtime.ab_specs.values()))
    item = DerivedFoldFeatures(
        ab_train=transform_ab_evidence(train_full, state),
        ab_validation=transform_ab_evidence(valid_full, state),
    )
    cache[fold_id] = item
    return item


def _assemble_stage2_frame(
    runtime: RuntimeData,
    indices: np.ndarray,
    ab_features: pd.DataFrame,
    config: V13Config,
    raw_direction_features: Sequence[str],
    *,
    include_preexposed: bool,
) -> tuple[pd.DataFrame, list[str]]:
    meta = [
        "row_index", "source_row_id", "date", "ticker", "market", "bucket",
        TARGET_COLUMN, MOVE_COLUMN, DIRECTION_COLUMN, "future_abs_excursion_3d",
    ]
    keep = list(dict.fromkeys([*meta, *raw_direction_features]))
    frame = runtime.frame.iloc[indices][keep].copy().reset_index(drop=True)
    derived_names: list[str] = []
    if config.include_ab:
        ab = ab_features.loc[indices].reset_index(drop=True)
        frame = pd.concat([frame, ab], axis=1)
        derived_names.extend(ab.columns.astype(str).tolist())
    if config.include_v11_lag0:
        net_columns = [c for c in V11_LAG0_PRIMARY_DIRECTION_COLUMNS if c in runtime.lag0_features.columns]
        net = runtime.lag0_features.iloc[indices][net_columns].reset_index(drop=True)
        frame = pd.concat([frame, net], axis=1)
        derived_names.extend(net.columns.astype(str).tolist())
    if config.include_v11_self:
        self_part = runtime.self_features.iloc[indices].reset_index(drop=True)
        frame = pd.concat([frame, self_part], axis=1)
        derived_names.extend(self_part.columns.astype(str).tolist())
    if include_preexposed:
        pre = runtime.preexposed_features.iloc[indices].reset_index(drop=True)
        frame = pd.concat([frame, pre], axis=1)
        derived_names.extend(pre.columns.astype(str).tolist())
    feature_columns = list(dict.fromkeys([*raw_direction_features, *derived_names]))
    return frame, feature_columns


def _stage1_cache_key(
    config: V13Config, selected_move: Sequence[str], fold_id: int, seed: int, *, use_gpu: bool,
    runtime_fingerprint: str,
) -> str:
    payload = {
        "cache_revision": CACHE_REVISION, "runtime_fingerprint": str(runtime_fingerprint),
        "stage": dataclasses.asdict(config.move), "features": list(selected_move),
        "fold": int(fold_id), "seed": int(seed), "half_life": config.recency_half_life_days,
        "ticker_onehot": config.ticker_onehot, "market_bucket_onehot": config.market_bucket_onehot,
        "device": "gpu" if use_gpu else "cpu",
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


def fit_predict_fold(
    runtime: RuntimeData,
    config: V13Config,
    selected_move: Sequence[str],
    selected_direction: Sequence[str],
    fold_id: int,
    seed: int,
    *,
    cpu_threads: int,
    use_gpu: bool,
    output: Path,
    derived_cache: dict[int, DerivedFoldFeatures],
    matching_index_cache: dict[tuple[int, int, int], tuple[np.ndarray, dict[str, Any]]],
    include_preexposed: bool = False,
    resume: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = "_PREEXP" if include_preexposed else ""
    device_tag = "gpu" if use_gpu else "cpu"
    prediction_cache_payload = {
        "cache_revision": CACHE_REVISION,
        "runtime_fingerprint": runtime.cache_fingerprint,
        "config_key": config.key(),
        "selected_move": list(map(str, selected_move)),
        "selected_direction": list(map(str, selected_direction)),
        "ab_specs": ab_spec_fingerprint(runtime.ab_specs),
        "device": device_tag,
        "preexposed": bool(include_preexposed),
    }
    prediction_cache_key = hashlib.sha1(
        json.dumps(prediction_cache_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:24]
    cache_dir = output / "cache" / "predictions" / f"{config.config_id}__{prediction_cache_key}{suffix}" / f"seed_{seed:05d}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pred_path = cache_dir / f"fold_{fold_id}.pkl"
    diag_path = cache_dir / f"fold_{fold_id}.json"
    if resume and pred_path.exists() and diag_path.exists():
        return pd.read_pickle(pred_path), load_json(diag_path)

    t0 = time.monotonic()
    train_idx_all = np.asarray(runtime.fold_index[fold_id]["train"], dtype=np.int64)
    valid_idx = np.asarray(runtime.fold_index[fold_id]["validation"], dtype=np.int64)
    train_valid_mask = pd.to_numeric(runtime.frame.iloc[train_idx_all][MOVE_COLUMN], errors="coerce").isin([0, 1]).to_numpy()
    train_idx = train_idx_all[train_valid_mask]
    valid_move_values = pd.to_numeric(runtime.frame.iloc[valid_idx][MOVE_COLUMN], errors="coerce")
    if not valid_move_values.isin([0, 1]).all():
        raise RuntimeError(f"Fold {fold_id} contains target-valid rows without a valid 3-day move path")
    meta = ["row_index", "source_row_id", "date", "ticker", "market", "bucket", TARGET_COLUMN, MOVE_COLUMN, DIRECTION_COLUMN, "future_abs_excursion_3d"]
    move_keep = list(dict.fromkeys([*meta, *selected_move]))
    train_move = runtime.frame.iloc[train_idx][move_keep].copy().reset_index(drop=True)
    valid_move = runtime.frame.iloc[valid_idx][move_keep].copy().reset_index(drop=True)

    stage1_key = _stage1_cache_key(
        config, selected_move, fold_id, seed, use_gpu=use_gpu,
        runtime_fingerprint=runtime.cache_fingerprint,
    )
    stage1_path = output / "cache" / "stage1" / f"{stage1_key}.npz"
    stage1_path.parent.mkdir(parents=True, exist_ok=True)
    stage1_gpu = False
    if resume and stage1_path.exists():
        cached = np.load(stage1_path)
        p_move_train = cached["p_train"]
        p_move_valid = cached["p_valid"]
    else:
        move_weights = make_sample_weights(
            train_move, pd.to_numeric(train_move[MOVE_COLUMN], errors="raise").to_numpy(np.int8),
            half_life_days=config.recency_half_life_days,
        )
        move_bundle = fit_binary_model(
            train_move, selected_move, MOVE_COLUMN, config.move, sample_weight=move_weights,
            seed=int(seed) + int(fold_id) * 1009 + 11, cpu_threads=cpu_threads, use_gpu=use_gpu,
            ticker_onehot=config.ticker_onehot, market_bucket_onehot=config.market_bucket_onehot,
        )
        p_move_train = predict_binary_model(move_bundle, train_move)
        p_move_valid = predict_binary_model(move_bundle, valid_move)
        stage1_gpu = bool(move_bundle.gpu_used)
        np.savez_compressed(stage1_path, p_train=p_move_train, p_valid=p_move_valid)

    derived = _derived_fold_features(runtime, fold_id, derived_cache)
    train_direction, direction_columns = _assemble_stage2_frame(
        runtime, train_idx, derived.ab_train, config, selected_direction, include_preexposed=include_preexposed,
    )
    valid_direction, _ = _assemble_stage2_frame(
        runtime, valid_idx, derived.ab_validation, config, selected_direction, include_preexposed=include_preexposed,
    )
    match_key = (int(fold_id), int(config.match_negative_reuse), int(seed))
    if match_key not in matching_index_cache:
        matched, manifest = match_direction_training_rows(
            train_direction, max_negative_reuse=int(config.match_negative_reuse), seed=int(seed) + int(fold_id) * 3571,
        )
        matching_index_cache[match_key] = (matched.index.to_numpy(), {
            "pairs": int(len(manifest)), "matched_rows": int(len(matched)),
            "same_ticker_fraction": float(manifest["match_level"].eq("ticker").mean()) if not manifest.empty else 0.0,
            "median_abs_excursion_difference": float(pd.to_numeric(manifest["abs_excursion_difference"], errors="coerce").median()) if not manifest.empty else float("nan"),
            "max_negative_reuse": int(pd.to_numeric(manifest["negative_reuse_after_match"], errors="coerce").max()) if not manifest.empty else 0,
        })
    matched_index, match_summary = matching_index_cache[match_key]
    matched_direction = train_direction.loc[matched_index].copy()
    direction_weights = make_sample_weights(
        matched_direction, pd.to_numeric(matched_direction[DIRECTION_COLUMN], errors="raise").to_numpy(np.int8),
        half_life_days=config.recency_half_life_days,
    )
    direction_bundle = fit_binary_model(
        matched_direction, direction_columns, DIRECTION_COLUMN, config.direction,
        sample_weight=direction_weights, seed=int(seed) + int(fold_id) * 1009 + 29,
        cpu_threads=cpu_threads, use_gpu=use_gpu, ticker_onehot=config.ticker_onehot,
        market_bucket_onehot=config.market_bucket_onehot,
    )
    p_up_train = predict_binary_model(direction_bundle, train_direction)
    p_up_valid = predict_binary_model(direction_bundle, valid_direction)
    p_stage_train = combine_stage_probabilities(p_move_train, p_up_train)
    p_stage_valid = combine_stage_probabilities(p_move_valid, p_up_valid)

    valid_with_base = add_base_score(valid_move, runtime.base_oof, fold_id)
    base_rank = past_oof_base_rank(runtime.base_oof, valid_with_base, fold_id=fold_id)
    policy_score, stage_hist_rank, stage_date_rank = build_policy_score(
        train_move, valid_with_base, train_stage_score=p_stage_train,
        validation_stage_score=p_stage_valid, base_past_rank=base_rank,
        base_rank_blend=config.base_rank_blend,
    )
    pred = valid_with_base[["row_index", "source_row_id", "date", "ticker", "market", "bucket", TARGET_COLUMN, MOVE_COLUMN, DIRECTION_COLUMN, "base_score_raw"]].copy()
    pred["fold_id"] = int(fold_id)
    pred["config_id"] = config.config_id
    pred["seed"] = int(seed)
    pred["p_move"] = p_move_valid
    pred["p_up_given_move"] = p_up_valid
    pred["stage_probability"] = p_stage_valid
    pred["stage_historical_rank"] = stage_hist_rank
    pred["stage_date_rank"] = stage_date_rank
    pred["base_past_rank"] = base_rank
    pred["policy_score"] = policy_score
    pred["v11_directed_preexposed_included"] = bool(include_preexposed)

    diagnostic = {
        "config_id": config.config_id, "config_key": config.key(), "fold_id": int(fold_id), "seed": int(seed),
        "train_rows": int(len(train_move)), "validation_rows": int(len(valid_move)),
        "move_features": int(len(selected_move)), "direction_raw_features": int(len(selected_direction)),
        "direction_total_features": int(len(direction_columns)), "stage1_gpu": stage1_gpu,
        "stage2_gpu": bool(direction_bundle.gpu_used), "preexposed": bool(include_preexposed),
        "seconds": float(time.monotonic() - t0), **match_summary,
    }
    pred.to_pickle(pred_path)
    atomic_json(diag_path, diagnostic)
    return pred, diagnostic


def ensemble_predictions(predictions: Sequence[pd.DataFrame]) -> pd.DataFrame:
    if not predictions:
        raise ValueError("No predictions for ensemble")
    key = ["row_index", "source_row_id", "date", "ticker", "market", "bucket", TARGET_COLUMN, MOVE_COLUMN, DIRECTION_COLUMN, "fold_id"]
    ordered: list[pd.DataFrame] = []
    reference_index: pd.MultiIndex | None = None
    for index, prediction in enumerate(predictions):
        missing = set(key) - set(prediction.columns)
        if missing:
            raise ValueError(f"Ensemble prediction {index} missing keys: {sorted(missing)}")
        part = prediction.sort_values(["fold_id", "date", "ticker", "row_index"], kind="mergesort").reset_index(drop=True)
        if part.duplicated(key).any():
            raise ValueError(f"Ensemble prediction {index} has duplicate row keys")
        current_index = pd.MultiIndex.from_frame(part[key])
        if reference_index is None:
            reference_index = current_index
        elif not current_index.equals(reference_index):
            raise ValueError(f"Ensemble prediction {index} row keys/order differ from prediction 0")
        ordered.append(part)
    metadata = ["base_score_raw"]
    for optional in ["policy_calibration_source", "stage_calibration_prior_rows", "v11_directed_preexposed_included"]:
        if optional in ordered[0].columns:
            metadata.append(optional)
    base = ordered[0][key + metadata].copy()
    probability_columns = ["p_move", "p_up_given_move", "stage_historical_rank", "stage_date_rank", "base_past_rank", "policy_score"]
    for column in probability_columns:
        stack = np.vstack([pd.to_numeric(p[column], errors="coerce").to_numpy(float) for p in ordered])
        base[column] = np.nanmean(stack, axis=0)
    # Preserve the probabilistic decomposition after seed averaging. Averaging
    # each seed's product would not equal mean(P(move))*mean(P(up|move)).
    base["stage_probability"] = combine_stage_probabilities(base["p_move"], base["p_up_given_move"])
    base["ensemble_seeds"] = int(len(predictions))
    return base


def recalibrate_ensemble_predictions(predictions: pd.DataFrame, *, base_rank_blend: float) -> pd.DataFrame:
    """Rebuild policy ranks sequentially from the averaged stage probability."""
    parts: list[pd.DataFrame] = []
    prior: list[pd.DataFrame] = []
    for fold_id in sorted(pd.to_numeric(predictions["fold_id"], errors="raise").astype(int).unique()):
        part = predictions.loc[pd.to_numeric(predictions["fold_id"], errors="coerce").eq(int(fold_id))].copy()
        part = part.sort_values(["date", "ticker", "row_index"], kind="mergesort").reset_index(drop=True)
        prior_frame = pd.concat(prior, ignore_index=True) if prior else pd.DataFrame()
        fallback = pd.to_numeric(part.get("stage_historical_rank"), errors="coerce").to_numpy(float)
        policy_score, hist, date_rank, source = recalibrate_policy_from_prior_oof(
            prior_frame, part, base_rank_blend=float(base_rank_blend),
            fallback_historical_rank=fallback,
        )
        part["policy_score"] = policy_score
        part["stage_historical_rank"] = hist
        part["stage_date_rank"] = date_rank
        part["policy_calibration_source"] = source
        part["stage_calibration_prior_rows"] = int(len(prior_frame))
        parts.append(part)
        prior.append(part[["date", "ticker", "stage_probability"]].copy())
    return pd.concat(parts, ignore_index=True).sort_values(
        ["fold_id", "date", "ticker", "row_index"], kind="mergesort"
    ).reset_index(drop=True)


def run_config_seeds(
    runtime: RuntimeData,
    config: V13Config,
    move_ranking: pd.DataFrame,
    direction_ranking: pd.DataFrame,
    folds: Sequence[int],
    seeds: Sequence[int],
    *,
    args: argparse.Namespace,
    output: Path,
    derived_cache: dict[int, DerivedFoldFeatures],
    matching_cache: dict[tuple[int, int, int], tuple[np.ndarray, dict[str, Any]]],
    include_preexposed: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[str], list[str]]:
    selected_move = select_ranked_features(move_ranking, "move_score", config.move.feature_k)
    selected_direction = select_ranked_features(direction_ranking, "direction_purity_score", config.direction.feature_k)
    if len(selected_move) < 4 or len(selected_direction) < 4:
        raise RuntimeError(f"Insufficient selected features for {config.config_id}: move={len(selected_move)}, direction={len(selected_direction)}")
    seed_ensembles: list[pd.DataFrame] = []
    diagnostics: list[dict[str, Any]] = []
    for seed in seeds:
        fold_predictions: list[pd.DataFrame] = []
        prior_oof: list[pd.DataFrame] = []
        for fold_id in sorted(int(x) for x in folds):
            pred, diag = fit_predict_fold(
                runtime, config, selected_move, selected_direction, int(fold_id), int(seed),
                cpu_threads=int(args.cpu_threads), use_gpu=bool(args.gpu_available), output=output,
                derived_cache=derived_cache, matching_index_cache=matching_cache,
                include_preexposed=include_preexposed, resume=bool(args.resume),
            )
            prior_frame = pd.concat(prior_oof, ignore_index=True) if prior_oof else pd.DataFrame()
            policy_score, stage_hist_rank, stage_date_rank, calibration_source = recalibrate_policy_from_prior_oof(
                prior_frame,
                pred,
                base_rank_blend=config.base_rank_blend,
                fallback_historical_rank=pd.to_numeric(pred["stage_historical_rank"], errors="coerce").to_numpy(float),
            )
            pred = pred.copy()
            pred["policy_score"] = policy_score
            pred["stage_historical_rank"] = stage_hist_rank
            pred["stage_date_rank"] = stage_date_rank
            pred["policy_calibration_source"] = calibration_source
            pred["stage_calibration_prior_rows"] = int(len(prior_frame))
            diag = dict(diag)
            diag["policy_calibration_source"] = calibration_source
            diag["stage_calibration_prior_rows"] = int(len(prior_frame))
            fold_predictions.append(pred)
            diagnostics.append(diag)
            prior_oof.append(pred[["date", "ticker", "stage_probability"]].copy())
        seed_ensembles.append(pd.concat(fold_predictions, ignore_index=True))
    # Average seeds within each row/fold, then calibrate the averaged decomposition
    # sequentially. This keeps threshold search independent of seed-order artifacts.
    ensemble = ensemble_predictions(seed_ensembles)
    ensemble = recalibrate_ensemble_predictions(ensemble, base_rank_blend=config.base_rank_blend)
    return ensemble, diagnostics, selected_move, selected_direction


def config_result_row(
    config: V13Config,
    predictions: pd.DataFrame,
    discovery_folds: Sequence[int],
    minimum_alerts: int,
    target_precision: float,
    seeds: int,
) -> tuple[dict[str, Any], Any]:
    policy = select_frozen_policy(
        predictions, discovery_folds=discovery_folds, minimum_alerts=minimum_alerts,
        target_precision=target_precision,
    )
    metrics = fold_score_metrics(predictions)
    subset = metrics.loc[metrics["fold_id"].isin(discovery_folds)]
    row = {
        "config_id": config.config_id, "config_key": config.key(), "seeds": int(seeds),
        "safe_on_discovery": bool(policy.safe_on_discovery), "policy_threshold": float(policy.threshold),
        "policy_reason": policy.reason, "minimum_discovery_precision": policy.minimum_precision,
        "mean_discovery_precision": policy.mean_precision, "minimum_discovery_wilson_lower": policy.minimum_wilson_lower,
        "minimum_discovery_alerts": policy.minimum_alerts_observed,
        "mean_discovery_pr_auc": float(pd.to_numeric(subset["pr_auc"], errors="coerce").mean()),
        "minimum_discovery_pr_auc": float(pd.to_numeric(subset["pr_auc"], errors="coerce").min()),
        "mean_move_pr_auc": float(pd.to_numeric(subset["move_pr_auc"], errors="coerce").mean()),
        "mean_direction_pr_auc_on_large_move": float(pd.to_numeric(subset["direction_pr_auc_on_large_move"], errors="coerce").mean()),
        **{f"config_{k}": v for k, v in dataclasses.asdict(config).items() if k not in {"move", "direction"}},
        "move_backend": config.move.backend, "move_feature_k": config.move.feature_k,
        "direction_backend": config.direction.backend, "direction_feature_k": config.direction.feature_k,
    }
    return row, policy


def result_sort_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    def val(name: str, fallback: float = -999.0) -> float:
        try:
            x = float(row.get(name, fallback))
            return x if math.isfinite(x) else fallback
        except Exception:
            return fallback
    return (
        1.0 if bool(row.get("safe_on_discovery", False)) else 0.0,
        val("minimum_discovery_precision", 0.0), val("minimum_discovery_wilson_lower", 0.0),
        val("minimum_discovery_pr_auc", 0.0), val("mean_discovery_pr_auc", 0.0),
        val("mean_direction_pr_auc_on_large_move", 0.0),
    )


def development_config_row(
    config: V13Config,
    predictions: pd.DataFrame,
    *,
    discovery_policy: Any,
    development_folds: Sequence[int],
    minimum_alerts: int,
    target_precision: float,
) -> dict[str, Any]:
    policy_metrics = apply_policy(predictions, float(discovery_policy.threshold))
    dev_policy = policy_metrics.loc[policy_metrics["fold_id"].isin([int(x) for x in development_folds])].copy()
    score_metrics = fold_score_metrics(predictions)
    dev_scores = score_metrics.loc[score_metrics["fold_id"].isin([int(x) for x in development_folds])].copy()
    complete = len(dev_policy) == len(set(int(x) for x in development_folds))
    precision = pd.to_numeric(dev_policy.get("precision"), errors="coerce")
    alerts = pd.to_numeric(dev_policy.get("alerts"), errors="coerce")
    wilson = pd.to_numeric(dev_policy.get("wilson_lower_95"), errors="coerce")
    dev_safe = bool(
        complete
        and precision.notna().all()
        and (alerts >= int(minimum_alerts)).all()
        and (precision >= float(target_precision)).all()
    )
    return {
        "config_id": config.config_id,
        "config_key": config.key(),
        "discovery_frozen_threshold": float(discovery_policy.threshold),
        "discovery_safe": bool(discovery_policy.safe_on_discovery),
        "development_complete": bool(complete),
        "development_safe_at_discovery_threshold": dev_safe,
        "minimum_development_precision": float(precision.min()) if len(precision) and precision.notna().all() else 0.0,
        "mean_development_precision": float(precision.mean()) if precision.notna().any() else 0.0,
        "minimum_development_alerts": int(alerts.min()) if len(alerts) and alerts.notna().all() else 0,
        "minimum_development_wilson_lower": float(wilson.min()) if len(wilson) and wilson.notna().all() else 0.0,
        "minimum_development_pr_auc": float(pd.to_numeric(dev_scores.get("pr_auc"), errors="coerce").min()) if not dev_scores.empty else float("nan"),
        "mean_development_pr_auc": float(pd.to_numeric(dev_scores.get("pr_auc"), errors="coerce").mean()) if not dev_scores.empty else float("nan"),
        "mean_development_move_pr_auc": float(pd.to_numeric(dev_scores.get("move_pr_auc"), errors="coerce").mean()) if not dev_scores.empty else float("nan"),
        "mean_development_direction_pr_auc_on_large_move": float(pd.to_numeric(dev_scores.get("direction_pr_auc_on_large_move"), errors="coerce").mean()) if not dev_scores.empty else float("nan"),
    }


def development_sort_key(row: Mapping[str, Any]) -> tuple[float, ...]:
    def val(name: str, fallback: float = -999.0) -> float:
        try:
            value = float(row.get(name, fallback))
            return value if math.isfinite(value) else fallback
        except Exception:
            return fallback
    return (
        1.0 if bool(row.get("development_safe_at_discovery_threshold", False)) else 0.0,
        val("minimum_development_precision", 0.0),
        val("minimum_development_wilson_lower", 0.0),
        val("minimum_development_pr_auc", 0.0),
        val("mean_development_pr_auc", 0.0),
        val("mean_development_direction_pr_auc_on_large_move", 0.0),
    )


def base_matched_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, grp in predictions.groupby("fold_id", sort=True):
        part = grp.loc[pd.to_numeric(grp["base_score_raw"], errors="coerce").notna()].copy()
        y = pd.to_numeric(part[TARGET_COLUMN], errors="coerce").to_numpy(float)
        score = pd.to_numeric(part["base_score_raw"], errors="coerce").to_numpy(float)
        rows.append({
            "fold_id": int(fold), "matched_rows": int(len(part)), "positive": int(np.sum(y == 1)),
            "base_rate": float(np.mean(y == 1)) if len(y) else float("nan"),
            "base_pr_auc": safe_pr_auc(y, score), "base_roc_auc": safe_roc_auc(y, score),
        })
    return pd.DataFrame(rows)


def ticker_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, ticker), grp in predictions.groupby(["fold_id", "ticker"], sort=True):
        y = pd.to_numeric(grp[TARGET_COLUMN], errors="coerce").to_numpy(float)
        score = pd.to_numeric(grp["policy_score"], errors="coerce").to_numpy(float)
        base = pd.to_numeric(grp["base_score_raw"], errors="coerce").to_numpy(float)
        rows.append({
            "fold_id": int(fold), "ticker": str(ticker), "rows": int(len(grp)),
            "positive": int(np.sum(y == 1)), "base_rate": float(np.mean(y == 1)),
            "v13_pr_auc": safe_pr_auc(y, score), "v13_roc_auc": safe_roc_auc(y, score),
            "v10_2_base_pr_auc": safe_pr_auc(y, base), "v10_2_base_roc_auc": safe_roc_auc(y, base),
        })
    return pd.DataFrame(rows)


def bootstrap_policy_stability(
    predictions: pd.DataFrame,
    threshold: float,
    discovery_folds: Sequence[int],
    *,
    iterations: int,
    seed: int,
    minimum_alerts: int,
    target_precision: float,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    work = predictions.loc[predictions["fold_id"].isin(discovery_folds)].copy()
    rows: list[dict[str, Any]] = []
    groups = {(int(f), str(t)): g.index.to_numpy() for (f, t), g in work.groupby(["fold_id", "ticker"], sort=False)}
    for iteration in range(int(iterations)):
        sampled: list[int] = []
        for indices in groups.values():
            sampled.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
        boot = work.loc[sampled].copy()
        metrics = apply_policy(boot, threshold)
        complete = metrics.loc[metrics["fold_id"].isin(discovery_folds)]
        min_precision = float(pd.to_numeric(complete["precision"], errors="coerce").min())
        min_alert = int(complete["alerts"].min())
        rows.append({
            "iteration": iteration, "minimum_precision": min_precision,
            "mean_precision": float(pd.to_numeric(complete["precision"], errors="coerce").mean()),
            "minimum_alerts": min_alert,
            "gate_success": bool(min_alert >= minimum_alerts and min_precision >= target_precision),
        })
    return pd.DataFrame(rows)


def build_handoff(output: Path, package_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destination = Path.home() / "Desktop" / f"magnitude-direction-results-{stamp}.zip"
    include_names = [
        "RUN_STATUS.json", "DATA_AUDIT_V13.json", "TARGET_RECONSTRUCTION_AUDIT_V13.json",
        "FINAL_RECOMMENDATION_V13.json", "V13_CHAMPION_CONFIG.json", "V13_FROZEN_POLICY.json",
        "V13_CONFIG_SCREENING.csv", "V13_ROBUST_CONFIGS.csv", "V13_DEVELOPMENT_CONFIG_VALIDATION.csv",
        "v13_stage_metrics_by_fold.csv",
        "v13_frozen_policy_by_fold.csv", "v13_oracle_topk_by_fold.csv", "v10_2_base_matched_by_fold.csv",
        "v13_ticker_metrics_by_fold.csv", "v13_component_ablation.csv", "v13_seed_stability.csv",
        "v13_policy_bootstrap_stability.csv", "V13_DISCOVERY_AB_RANKING.csv",
        "V13_MOVE_FEATURE_RANKING_DISCOVERY.csv", "V13_DIRECTION_PURITY_RANKING_DISCOVERY.csv",
        "V13_AB_SELECTION_MANIFEST.csv", "V13_FEATURE_FAMILY_AUDIT.csv", "V13_V11_LAG0_EDGE_MANIFEST.csv",
        "V13_V11_DIRECTED_PREEXPOSED_MANIFEST.csv", "LEAKAGE_CONTRACT_V13.json",
        "V10_2_AB_REFERENCE_PREEXPOSED.csv", "V10_2_BASE_OOF_COVERAGE_V13.csv",
        "V10_2_BASE_OOF_SCOPE_AUDIT_V13.json",
        "VERIFIER_RESULTS_V13.json", "VERIFIER_STDOUT.txt", "v13_oof_predictions.csv.gz",
    ]
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for name in include_names:
            path = output / name
            if path.exists():
                zf.write(path, f"V13_RESULTS/{name}")
        for name in [
            "README_FIRST_KO.md", "run_surge_magnitude_direction_v13.py", "surge_v13_data.py",
            "surge_v13_models.py", "verify_surge_magnitude_direction_v13.py",
            "requirements_v13.txt", "LINEAGE_V13.json",
        ]:
            path = package_root / name
            if path.exists():
                zf.write(path, f"V13_RESULTS/repro_code/{name}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch Surge V13: full-48-ticker magnitude -> direction decomposition with V10.2 A/B and V11 lag-0 network signals.")
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--v10-output", required=True)
    parser.add_argument("--v10-2-output", required=True)
    parser.add_argument("--v11-output", default=None)
    parser.add_argument("--v11-1-output", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--target-sidecar", required=True)
    parser.add_argument("--folds", required=True)
    parser.add_argument("--feature-profile-manifest", required=True)
    parser.add_argument("--feature-profile", default="P0_ALL_VALID")
    parser.add_argument("--output", default="outputs/surge_magnitude_direction_v13")
    parser.add_argument("--require-full-439", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expected-target-valid-rows", type=int, default=91775,
                        help="Expected V10.2 target-valid row count; set 0 only for an intentional universe/data update.")
    parser.add_argument("--expected-ticker-count", type=int, default=48,
                        help="Expected ticker universe size; set 0 only for an intentional universe update.")
    parser.add_argument("--require-full-base-oof", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument("--target-hours", type=float, default=7.0)
    parser.add_argument("--fast-mode", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--discovery-folds", default="0,1,2")
    parser.add_argument("--development-folds", default="3,4")
    parser.add_argument("--confirmation-folds", default="5,6")
    parser.add_argument("--recent-folds", default="7")
    parser.add_argument("--minimum-alerts", type=int, default=30)
    parser.add_argument("--target-precision", type=float, default=0.70)
    parser.add_argument("--surge-threshold", type=float, default=0.05)
    parser.add_argument("--max-target-mismatch-rate", type=float, default=0.01)
    parser.add_argument("--allow-target-mismatch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-ab-features-per-ticker", type=int, default=12)
    parser.add_argument("--top-configs", type=int, default=3)
    parser.add_argument("--robust-seeds", type=int, default=5)
    parser.add_argument(
        "--final-seeds", type=int, default=9,
        help="Minimum number of single-seed post-freeze stability diagnostics; does not alter the primary ensemble.",
    )
    parser.add_argument(
        "--max-final-seeds", type=int, default=21,
        help="Maximum post-freeze diagnostic seeds when wall-clock remains; never changes primary predictions or threshold.",
    )
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13013)
    args = parser.parse_args()

    start = time.monotonic()
    budget = max(float(args.target_hours) * 3600.0, 180.0)
    if bool(args.fast_mode):
        budget = max(budget, 180.0)
    final_deadline = start + budget * 0.89
    ablation_deadline = start + budget * 0.96

    args.package_root = Path(args.package_root).expanduser().resolve()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = (args.package_root / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "RUN_STATUS.json", {"schema": SCHEMA, "status": "RUNNING", "started_utc": utc_now()})
    atomic_json(output / "SYSTEM_SNAPSHOT_V13.json", system_snapshot())

    try:
        gpu = gpu_preflight(int(args.seed))
        args.gpu_available = bool(gpu.get("available", False)) and not bool(args.fast_mode)
        atomic_json(output / "GPU_PREFLIGHT_V13.json", gpu)
        if bool(args.require_gpu) and not bool(args.gpu_available):
            raise RuntimeError(f"CUDA XGBoost preflight failed: {gpu}")
        runtime, _ = prepare_runtime(args, output)
        discovery_folds = parse_ints(args.discovery_folds)
        development_folds = parse_ints(args.development_folds)
        confirmation_folds = parse_ints(args.confirmation_folds)
        recent_folds = parse_ints(args.recent_folds)
        all_eval_folds = discovery_folds + development_folds + confirmation_folds + recent_folds
        if len(set(all_eval_folds)) != len(all_eval_folds):
            raise ValueError("Fold role overlap")

        discovery = discovery_validation_with_base(runtime, discovery_folds)
        move_ranking = rank_move_features(discovery, runtime.candidate_features)
        direction_ranking = rank_direction_features(discovery, runtime.candidate_features)
        if move_ranking.empty or direction_ranking.empty:
            raise RuntimeError("Discovery feature ranking returned no candidates")
        runtime.ab_specs, ab_ranking = discover_ab_specs_v13(
            discovery, runtime.candidate_features, candidate_quantile=0.65,
            max_features_per_ticker=int(args.max_ab_features_per_ticker),
        )
        if not runtime.ab_specs:
            raise RuntimeError("V13 discovery-only A/B reconstruction returned no ticker specifications")
        atomic_csv(output / "V13_MOVE_FEATURE_RANKING_DISCOVERY.csv", move_ranking)
        atomic_csv(output / "V13_DIRECTION_PURITY_RANKING_DISCOVERY.csv", direction_ranking)
        atomic_csv(output / "V13_DISCOVERY_AB_RANKING.csv", ab_ranking)
        atomic_csv(output / "V13_AB_SELECTION_MANIFEST.csv", serialize_ab_specs(runtime.ab_specs))
        data_audit = load_json(output / "DATA_AUDIT_V13.json")
        data_audit["ab_spec_tickers"] = int(len(runtime.ab_specs))
        data_audit["ab_spec_nodes"] = int(sum(len(x) for x in runtime.ab_specs.values()))
        data_audit["ab_primary_source"] = "V13 folds 0-2 reconstruction; V10.2 100-node map reference-only"
        atomic_json(output / "DATA_AUDIT_V13.json", data_audit)

        leakage_contract = {
            "schema": SCHEMA,
            "primary_parent": "V10.2 with V11/V11.1 reference signals",
            "feature_identity_and_shortlist_folds": discovery_folds,
            "feature_and_config_selection_folds": discovery_folds,
            "development_folds_for_champion_selection": development_folds,
            "development_rule": "feature identities and each candidate's threshold are frozen on folds 0-2; folds 3-4 select among the predeclared shortlist only",
            "confirmation_folds": confirmation_folds, "recent_diagnostic_folds": recent_folds,
            "v11_primary_fields": ["ticker_a", "ticker_b", "discovery_best_lag", "discovery_best_lag_correlation", "maxstat_q_value"],
            "v11_primary_rule": "maxstat_q<=0.10 and lag==0 only",
            "v11_directed_edges": "preexposed diagnostic only; never eligible for primary champion",
            "stage1_target": MOVE_COLUMN, "stage2_target": DIRECTION_COLUMN,
            "stage2_matching_outcomes_not_model_inputs": ["future_abs_excursion_3d", "date distance", "hierarchy"],
            "base_score_calibration": "earlier OOF folds only; current-fold base_rank is not used",
            "stage_score_calibration": "outer-train reference only for the first requested fold; earlier config-specific OOF folds thereafter",
            "inherited_v10_transforms": "safe ticker/time and current-date cross-sectional transforms only; invalid date-industry and full-history cluster-innovation transforms excluded",
            "primary_seed_ensemble": "same robust seed set on discovery, folds 3-4 champion selection, and final folds 5-7; extra seeds diagnostics only",
            "production_claim": "forbidden until genuinely new future data",
        }
        atomic_json(output / "LEAKAGE_CONTRACT_V13.json", leakage_contract)

        configs = predefined_configs(fast=bool(args.fast_mode))
        if bool(args.fast_mode):
            configs = configs[:3]
            args.robust_seeds = min(int(args.robust_seeds), 2)
            args.final_seeds = min(int(args.final_seeds), 2)
            args.max_final_seeds = min(int(args.max_final_seeds), 3)
            args.bootstrap_iterations = min(int(args.bootstrap_iterations), 100)

        derived_cache: dict[int, DerivedFoldFeatures] = {}
        matching_cache: dict[tuple[int, int, int], tuple[np.ndarray, dict[str, Any]]] = {}
        screening_rows: list[dict[str, Any]] = []
        screen_seed = int(args.seed)
        log(f"Screening {len(configs)} predeclared configs on discovery folds {discovery_folds}")
        for index, config in enumerate(configs):
            # Complete the entire predeclared set. Optional extra final seeds and
            # post-freeze ablations are the only time-budgeted stages.
            pred, diagnostics, selected_move, selected_direction = run_config_seeds(
                runtime, config, move_ranking, direction_ranking, discovery_folds, [screen_seed],
                args=args, output=output, derived_cache=derived_cache, matching_cache=matching_cache,
            )
            row, _ = config_result_row(config, pred, discovery_folds, int(args.minimum_alerts), float(args.target_precision), 1)
            row["selected_move_features"] = len(selected_move)
            row["selected_direction_raw_features"] = len(selected_direction)
            row["mean_seconds_per_fold"] = float(np.mean([x["seconds"] for x in diagnostics]))
            screening_rows.append(row)
            atomic_csv(output / "V13_CONFIG_SCREENING.csv", pd.DataFrame(screening_rows).sort_values("config_id"))
            log(f"screen {config.config_id}: minP={row['minimum_discovery_precision']:.3f} AP={row['mean_discovery_pr_auc']:.3f}")
        screening = pd.DataFrame(screening_rows)
        if screening.empty:
            raise RuntimeError("No V13 config completed screening")
        ranked_screen = sorted(screening_rows, key=result_sort_key, reverse=True)
        top_ids = [str(x["config_id"]) for x in ranked_screen[: max(1, int(args.top_configs))]]
        config_by_id = {c.config_id: c for c in configs}

        robust_rows: list[dict[str, Any]] = []
        robust_predictions: dict[str, pd.DataFrame] = {}
        robust_seed_values = [int(args.seed) + i * 101 for i in range(max(1, int(args.robust_seeds)))]
        log(f"Robust discovery ensemble for {top_ids} with seeds={robust_seed_values}")
        for config_id in top_ids:
            # Every discovery-shortlisted family receives the same robust seed set.
            config = config_by_id[config_id]
            pred, diagnostics, selected_move, selected_direction = run_config_seeds(
                runtime, config, move_ranking, direction_ranking, discovery_folds, robust_seed_values,
                args=args, output=output, derived_cache=derived_cache, matching_cache=matching_cache,
            )
            robust_predictions[config_id] = pred
            row, _ = config_result_row(config, pred, discovery_folds, int(args.minimum_alerts), float(args.target_precision), len(robust_seed_values))
            row["selected_move_features"] = len(selected_move)
            row["selected_direction_raw_features"] = len(selected_direction)
            robust_rows.append(row)
            atomic_csv(output / "V13_ROBUST_CONFIGS.csv", pd.DataFrame(robust_rows))
        if not robust_rows:
            # Fall back to the best single-seed screening config.
            best_id = top_ids[0]
            config = config_by_id[best_id]
            pred, _, _, _ = run_config_seeds(
                runtime, config, move_ranking, direction_ranking, discovery_folds, [screen_seed],
                args=args, output=output, derived_cache=derived_cache, matching_cache=matching_cache,
            )
            robust_predictions[best_id] = pred
            row, _ = config_result_row(config, pred, discovery_folds, int(args.minimum_alerts), float(args.target_precision), 1)
            robust_rows = [row]
        # Forward development validation: no new feature identities, lags, configs,
        # or threshold search are introduced here.  Each candidate carries the
        # threshold selected on folds 0-2 into folds 3-4 unchanged.
        development_rows: list[dict[str, Any]] = []
        development_predictions: dict[str, pd.DataFrame] = {}
        robust_policy_by_id: dict[str, Any] = {}
        ordered_robust_ids = [str(x["config_id"]) for x in sorted(robust_rows, key=result_sort_key, reverse=True)]
        log(f"Forward development validation on folds {development_folds}: {ordered_robust_ids}")
        for config_id in ordered_robust_ids:
            # Development comparison must be complete across the entire frozen
            # shortlist; otherwise the champion would depend on wall-clock order.
            config = config_by_id[config_id]
            discovery_policy = select_frozen_policy(
                robust_predictions[config_id], discovery_folds=discovery_folds,
                minimum_alerts=int(args.minimum_alerts), target_precision=float(args.target_precision),
            )
            robust_policy_by_id[config_id] = discovery_policy
            combined_pred, _, _, _ = run_config_seeds(
                runtime, config, move_ranking, direction_ranking, discovery_folds + development_folds,
                robust_seed_values, args=args, output=output, derived_cache=derived_cache,
                matching_cache=matching_cache,
            )
            development_predictions[config_id] = combined_pred
            development_rows.append(development_config_row(
                config, combined_pred, discovery_policy=discovery_policy,
                development_folds=development_folds, minimum_alerts=int(args.minimum_alerts),
                target_precision=float(args.target_precision),
            ))
            atomic_csv(output / "V13_DEVELOPMENT_CONFIG_VALIDATION.csv", pd.DataFrame(development_rows))
        if development_rows and len(development_rows) != len(ordered_robust_ids):
            raise RuntimeError(
                f"Incomplete development shortlist: completed={len(development_rows)}, "
                f"expected={len(ordered_robust_ids)}"
            )
        if development_rows:
            champion_row = max(development_rows, key=development_sort_key)
            champion = config_by_id[str(champion_row["config_id"])]
            champion_discovery = development_predictions[champion.config_id]
            frozen_policy = robust_policy_by_id[champion.config_id]
            champion_selection_role = "DEVELOPMENT_3_4_WITH_DISCOVERY_FROZEN_THRESHOLD"
        else:
            # Defensive fallback for a custom run with no development folds.
            # Normal V13 execution completes every shortlisted candidate.
            champion_row = max(robust_rows, key=result_sort_key)
            champion = config_by_id[str(champion_row["config_id"])]
            champion_discovery = robust_predictions[champion.config_id]
            frozen_policy = select_frozen_policy(
                champion_discovery, discovery_folds=discovery_folds,
                minimum_alerts=int(args.minimum_alerts), target_precision=float(args.target_precision),
            )
            champion_selection_role = "DISCOVERY_FALLBACK_NO_DEVELOPMENT_FOLDS"
        selected_move = select_ranked_features(move_ranking, "move_score", champion.move.feature_k)
        selected_direction = select_ranked_features(direction_ranking, "direction_purity_score", champion.direction.feature_k)
        atomic_json(output / "V13_CHAMPION_CONFIG.json", {
            "schema": SCHEMA, "champion": dataclasses.asdict(champion), "config_key": champion.key(),
            "selected_move_features": selected_move, "selected_direction_raw_features": selected_direction,
            "selection_basis": champion_row,
            "feature_set_frozen_before_development": True,
            "champion_selection_role": champion_selection_role,
            "primary_ensemble_seeds": list(robust_seed_values),
            "primary_ensemble_seed_count": len(robust_seed_values),
            "champion_frozen_before_confirmation": True,
        })
        atomic_json(output / "V13_FROZEN_POLICY.json", {
            "threshold": frozen_policy.threshold, "safe_on_discovery": frozen_policy.safe_on_discovery,
            "reason": frozen_policy.reason, "minimum_alerts": int(args.minimum_alerts),
            "target_precision": float(args.target_precision), "selected_on_folds": discovery_folds,
            "applied_unchanged_during_champion_selection_folds": development_folds,
            "primary_ensemble_seeds": list(robust_seed_values),
            "frozen_before_folds": development_folds + confirmation_folds + recent_folds,
        })
        log(f"Champion frozen: {champion.config_id}, threshold={frozen_policy.threshold:.6f}")

        # Champion identity, seed ensemble, and discovery threshold are frozen
        # together. The primary final model uses exactly the same robust seed set
        # that was evaluated on folds 3-4. Extra seeds are stability diagnostics
        # only; they never alter primary predictions or the frozen threshold.
        primary_seed_values = list(robust_seed_values)
        final_pred, final_diagnostics, _, _ = run_config_seeds(
            runtime, champion, move_ranking, direction_ranking, all_eval_folds,
            primary_seed_values, args=args, output=output,
            derived_cache=derived_cache, matching_cache=matching_cache,
        )

        minimum_diagnostic_seeds = max(len(primary_seed_values), int(args.final_seeds))
        maximum_diagnostic_seeds = max(minimum_diagnostic_seeds, int(args.max_final_seeds))
        diagnostic_seed_values: list[int] = []
        diagnostic_seed_predictions: list[pd.DataFrame] = []
        seed_index = 0
        while seed_index < maximum_diagnostic_seeds:
            if (
                seed_index >= minimum_diagnostic_seeds
                and not bool(args.fast_mode)
                and time.monotonic() >= final_deadline
            ):
                break
            seed = int(args.seed) + seed_index * 101
            seed_pred, _, _, _ = run_config_seeds(
                runtime, champion, move_ranking, direction_ranking, all_eval_folds,
                [seed], args=args, output=output, derived_cache=derived_cache,
                matching_cache=matching_cache,
            )
            diagnostic_seed_values.append(seed)
            diagnostic_seed_predictions.append(seed_pred)
            seed_index += 1
        final_pred = final_pred.sort_values(["fold_id", "date", "ticker", "row_index"], kind="mergesort").reset_index(drop=True)
        atomic_csv(output / "v13_oof_predictions.csv.gz", final_pred, compression="gzip")
        stage_metrics = fold_score_metrics(final_pred)
        policy_metrics = apply_policy(final_pred, frozen_policy.threshold)
        oracle = oracle_top_k(final_pred, minimum_alerts=int(args.minimum_alerts))
        base_metrics = base_matched_metrics(final_pred)
        tick_metrics = ticker_metrics(final_pred)
        atomic_csv(output / "v13_stage_metrics_by_fold.csv", stage_metrics)
        atomic_csv(output / "v13_frozen_policy_by_fold.csv", policy_metrics)
        atomic_csv(output / "v13_oracle_topk_by_fold.csv", oracle)
        atomic_csv(output / "v10_2_base_matched_by_fold.csv", base_metrics)
        atomic_csv(output / "v13_ticker_metrics_by_fold.csv", tick_metrics)
        atomic_csv(output / "V13_FINAL_FOLD_DIAGNOSTICS.csv", pd.DataFrame(final_diagnostics))

        # Seed stability on the already-frozen architecture and policy.
        stability_rows: list[dict[str, Any]] = []
        for seed_pred, seed in zip(diagnostic_seed_predictions, diagnostic_seed_values):
            metrics = apply_policy(seed_pred, frozen_policy.threshold)
            for _, row in metrics.iterrows():
                stability_rows.append({"seed": int(seed), **row.to_dict()})
        stability = pd.DataFrame(stability_rows)
        atomic_csv(output / "v13_seed_stability.csv", stability)

        # Optional diagnostics after champion freeze. They are never allowed to replace the champion.
        ablation_rows: list[dict[str, Any]] = []
        variants: list[tuple[str, V13Config, bool]] = [
            ("PRIMARY_FULL", champion, False),
            ("NO_AB", dataclasses.replace(champion, config_id=champion.config_id + "_NO_AB", include_ab=False), False),
            ("NO_V11_LAG0", dataclasses.replace(champion, config_id=champion.config_id + "_NO_LAG0", include_v11_lag0=False), False),
            ("NO_V11_SELF", dataclasses.replace(champion, config_id=champion.config_id + "_NO_SELF", include_v11_self=False), False),
            ("V11_DIRECTED_PREEXPOSED_DIAGNOSTIC", dataclasses.replace(champion, config_id=champion.config_id + "_PREEXP"), True),
        ]
        for name, variant, preexp in variants:
            if name == "PRIMARY_FULL":
                pred = final_pred
            else:
                if not bool(args.fast_mode) and time.monotonic() > ablation_deadline:
                    ablation_rows.append({"variant": name, "status": "SKIPPED_TIME_BUDGET"})
                    continue
                pred, _, _, _ = run_config_seeds(
                    runtime, variant, move_ranking, direction_ranking, all_eval_folds, [int(args.seed)],
                    args=args, output=output, derived_cache=derived_cache, matching_cache=matching_cache,
                    include_preexposed=preexp,
                )
            metrics = fold_score_metrics(pred)
            for _, row in metrics.iterrows():
                ablation_rows.append({
                    "variant": name, "status": "OK", "preexposed": preexp,
                    "eligible_for_champion": name == "PRIMARY_FULL", **row.to_dict(),
                })
        atomic_csv(output / "v13_component_ablation.csv", pd.DataFrame(ablation_rows))

        bootstrap = bootstrap_policy_stability(
            final_pred, frozen_policy.threshold, discovery_folds,
            iterations=int(args.bootstrap_iterations), seed=int(args.seed) + 99991,
            minimum_alerts=int(args.minimum_alerts), target_precision=float(args.target_precision),
        )
        atomic_csv(output / "v13_policy_bootstrap_stability.csv", bootstrap)

        dev_policy = policy_metrics.loc[policy_metrics["fold_id"].isin(development_folds)]
        conf_policy = policy_metrics.loc[policy_metrics["fold_id"].isin(confirmation_folds)]
        recent_policy = policy_metrics.loc[policy_metrics["fold_id"].isin(recent_folds)]
        dev_safe = bool(
            len(dev_policy) == len(development_folds)
            and (dev_policy["alerts"] >= int(args.minimum_alerts)).all()
            and (pd.to_numeric(dev_policy["precision"], errors="coerce") >= float(args.target_precision)).all()
        )
        confirmation_safe = bool(
            len(conf_policy) == len(confirmation_folds)
            and (conf_policy["alerts"] >= int(args.minimum_alerts)).all()
            and (pd.to_numeric(conf_policy["precision"], errors="coerce") >= float(args.target_precision)).all()
        )
        recent_safe = bool(
            len(recent_policy) == len(recent_folds)
            and (recent_policy["alerts"] >= int(args.minimum_alerts)).all()
            and (pd.to_numeric(recent_policy["precision"], errors="coerce") >= float(args.target_precision)).all()
        )
        recommendation = {
            "schema": SCHEMA, "status": "SUCCESS_VERIFIED_PENDING_EXTERNAL_VERIFIER",
            "champion_config_id": champion.config_id,
            "primary_ensemble_seeds": primary_seed_values,
            "primary_ensemble_seed_count": len(primary_seed_values),
            "diagnostic_seeds_completed": len(diagnostic_seed_predictions),
            "frozen_threshold": frozen_policy.threshold,
            "discovery_safe": frozen_policy.safe_on_discovery, "development_safe": dev_safe,
            "confirmation_safe": confirmation_safe, "recent_diagnostic_safe": recent_safe,
            "research_gate_all_observed_roles": bool(frozen_policy.safe_on_discovery and dev_safe and confirmation_safe and recent_safe),
            "production_decision": "NO_ALERT",
            "production_reason": "Folds 3-7 and V11 relationships are already research-exposed; only genuinely new future data can authorize production, even if diagnostics pass.",
            "primary_findings_to_review": {
                "full_48_tickers_restored": int(runtime.frame["ticker"].nunique()) == 48,
                "magnitude_direction_separated": True,
                "v11_lag0_network_primary": True,
                "v11_directed_edges_primary": False,
                "six_ticker_filter_removed": True,
            },
            "runtime_hours": float((time.monotonic() - start) / 3600.0),
            "generated_utc": utc_now(),
        }
        atomic_json(output / "FINAL_RECOMMENDATION_V13.json", recommendation)

        # Run external verifier before calling the result verified.
        verifier = HERE / "verify_surge_magnitude_direction_v13.py"
        verify_result = subprocess.run([sys.executable, str(verifier), "--output", str(output)], capture_output=True, text=True, check=False)
        (output / "VERIFIER_STDOUT.txt").write_text(verify_result.stdout + "\n" + verify_result.stderr, encoding="utf-8")
        if verify_result.returncode != 0:
            raise RuntimeError(f"V13 verifier failed; see {output / 'VERIFIER_STDOUT.txt'}")
        recommendation["status"] = "SUCCESS_VERIFIED"
        atomic_json(output / "FINAL_RECOMMENDATION_V13.json", recommendation)
        run_status = {
            "schema": SCHEMA, "status": "SUCCESS_VERIFIED", "started_utc": None,
            "completed_utc": utc_now(), "runtime_seconds": float(time.monotonic() - start),
            "champion": champion.config_id, "production_decision": "NO_ALERT",
        }
        atomic_json(output / "RUN_STATUS.json", run_status)
        handoff = build_handoff(output, HERE)
        log(f"SUCCESS_VERIFIED. compact result archive: {handoff}")
    except Exception as exc:
        atomic_json(output / "RUN_STATUS.json", {
            "schema": SCHEMA, "status": "FAILED", "failed_utc": utc_now(),
            "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
