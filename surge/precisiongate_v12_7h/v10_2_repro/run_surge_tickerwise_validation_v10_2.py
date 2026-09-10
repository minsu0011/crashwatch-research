from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

try:
    import run_surge_tickerwise_correlation_map_v10 as v10
    from surge_ticker_common_v10 import (
        ERROR_A_TOP_TP, ERROR_B_TOP_FP, ERROR_C_LOW_TP, ERROR_D_LOW_TN,
        percentile_rank_1d, safe_roc_auc, safe_pearson, safe_spearman,
        separation_metrics, summarize_fixed_direction, select_threshold_for_precision,
        safe_binary_metrics, probability_metrics,
    )
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "V10 starter_code files are required next to this script: "
        "run_surge_tickerwise_correlation_map_v10.py and surge_ticker_common_v10.py"
    ) from exc

from surge_ticker_hierarchy_v10_2 import (
    SCHEMA_VERSION,
    ConservativeHierarchyConfig,
    prepare_base_effect_frame,
    build_disjoint_peer_rings,
    build_hierarchy_for_strength,
    aggregate_hierarchy_sensitivity,
    build_precision_separator_map,
    deduplicate_precision_sources,
    build_residual_similarity,
)


def log(message: str) -> None:
    print(f"[V10.2] {message}", flush=True)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_ints(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        return [int(x.strip()) for x in value.split(",") if x.strip()]
    return [int(x) for x in value]


def parse_floats(value: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        return tuple(float(x.strip()) for x in value.split(",") if x.strip())
    return tuple(float(x) for x in value)


def find_output(root: Path, explicit: str | None, relative: str) -> Path:
    path = Path(explicit).expanduser() if explicit else root / relative
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_v10_args(args: argparse.Namespace, output: Path) -> argparse.Namespace:
    defaults = v10.build_parser().parse_args([])
    defaults.package_root = str(args.package_root)
    defaults.output = output
    if args.dataset is not None:
        defaults.dataset = Path(args.dataset)
    if args.target_sidecar is not None:
        defaults.target_sidecar = Path(args.target_sidecar)
    if args.folds is not None:
        defaults.folds = Path(args.folds)
    if args.feature_profile_manifest is not None:
        defaults.feature_profile_manifest = Path(args.feature_profile_manifest)
    defaults.feature_profile = str(args.feature_profile)
    defaults.device = args.device
    defaults.allow_cpu_fallback = bool(args.allow_cpu_fallback)
    defaults.strict_backend = bool(args.strict_backend)
    defaults.base_backends = args.base_backends
    defaults.probe_backends = args.probe_backends
    defaults.threads = int(args.threads)
    defaults.xgboost_threads = int(args.xgboost_threads)
    defaults.parallel_backends = bool(args.parallel_backends)
    defaults.seed = int(args.seed)
    defaults.base_feature_count = int(args.base_feature_count)
    defaults.minimum_base_feature_count = int(args.minimum_base_feature_count)
    defaults.base_dedup_threshold = float(args.base_dedup_threshold)
    defaults.resume = bool(args.resume)
    defaults.minimum_validation_rows_model = int(args.configured_min_validation_rows)
    defaults.minimum_validation_positive_model = int(args.minimum_validation_positive)
    defaults.minimum_validation_negative_model = int(args.minimum_validation_negative)
    defaults.require_full_439 = bool(args.require_full_439)
    v10.resolve_default_paths(Path(args.package_root), defaults)
    return defaults


def adaptive_eligibility(
    eligibility: pd.DataFrame,
    recent_folds: Sequence[int],
    configured_rows: int,
    row_fraction: float,
    row_floor: int,
    min_pos: int,
    min_neg: int,
    min_train_rows: int = 400,
    min_train_pos: int = 35,
    min_train_neg: int = 100,
) -> pd.DataFrame:
    out = eligibility.copy()
    out["eligible_model_v10_2"] = out["eligible_model"].astype(bool)
    out["adaptive_min_validation_rows"] = int(configured_rows)
    for fold_id in recent_folds:
        mask = out["fold_id"].eq(int(fold_id))
        if not mask.any():
            continue
        fold_max = int(pd.to_numeric(out.loc[mask, "validation_rows"], errors="coerce").max())
        threshold = min(int(configured_rows), max(int(row_floor), int(math.ceil(float(row_fraction) * fold_max))))
        out.loc[mask, "adaptive_min_validation_rows"] = threshold
        train_ok = (
            (out.loc[mask, "train_rows"] >= int(min_train_rows))
            & (out.loc[mask, "train_positive"] >= int(min_train_pos))
            & (out.loc[mask, "train_negative"] >= int(min_train_neg))
        )
        valid_ok = (
            (out.loc[mask, "validation_rows"] >= threshold)
            & (out.loc[mask, "validation_positive"] >= int(min_pos))
            & (out.loc[mask, "validation_negative"] >= int(min_neg))
        )
        out.loc[mask, "eligible_model_v10_2"] = (train_ok & valid_ok).to_numpy()
    return out


def _subset_ticker_index(index: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]], fold_ids: Sequence[int]) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    ids = {int(x) for x in fold_ids}
    return {t: {f: v for f, v in folds.items() if int(f) in ids} for t, folds in index.items()}


def recompute_recent_base_oof(
    frame: pd.DataFrame,
    features: Sequence[str],
    folds: Sequence[Any],
    ticker_index: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    eligibility_v10_2: pd.DataFrame,
    roles: Mapping[str, Sequence[int]],
    recent_folds: Sequence[int],
    v10args: argparse.Namespace,
    work_output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recent_set = {int(x) for x in recent_folds}
    recent_specs = [f for f in folds if int(f.fold_id) in recent_set]
    if not recent_specs:
        return pd.DataFrame(), pd.DataFrame()
    e = eligibility_v10_2.loc[eligibility_v10_2["fold_id"].isin(recent_set)].copy()
    e["eligible_model"] = e["eligible_model_v10_2"].astype(bool)
    old_output = v10args.output
    old_resume = getattr(v10args, "resume", False)
    v10args.output = work_output / "recent_recalc_cache"
    v10args.resume = bool(getattr(v10args, "resume", False))
    try:
        oof, manifest = v10.build_ticker_base_oof(
            frame, features, recent_specs, _subset_ticker_index(ticker_index, recent_folds), e, roles, v10args
        )
    finally:
        v10args.output = old_output
        v10args.resume = old_resume
    return oof, manifest


def combine_base_oof(old_oof: pd.DataFrame, recent: pd.DataFrame, recent_folds: Sequence[int]) -> pd.DataFrame:
    ids = {int(x) for x in recent_folds}
    keep = old_oof.loc[~pd.to_numeric(old_oof["fold_id"], errors="coerce").isin(ids)].copy()
    out = pd.concat([keep, recent], ignore_index=True, sort=False)
    return out.sort_values(["fold_id", "ticker", "row_index"], kind="mergesort").reset_index(drop=True)


def compute_recent_stage0_rows(
    frame: pd.DataFrame,
    oof_groups: pd.DataFrame,
    features: Sequence[str],
    matched_ab: pd.DataFrame,
    matched_cd: pd.DataFrame,
    recent_folds: Sequence[int],
    v10args: argparse.Namespace,
) -> pd.DataFrame:
    ids = {int(x) for x in recent_folds}
    groups = oof_groups.loc[pd.to_numeric(oof_groups["fold_id"], errors="coerce").isin(ids)].copy()
    source = frame[["source_row_id", *features]].copy()
    combined = groups.merge(source, on="source_row_id", how="left", validate="one_to_one")
    ab_lookup = {(str(t), int(f)): p for (t, f), p in matched_ab.groupby(["ticker", "fold_id"], sort=False)} if not matched_ab.empty else {}
    cd_lookup = {(str(t), int(f)): p for (t, f), p in matched_cd.groupby(["ticker", "fold_id"], sort=False)} if not matched_cd.empty else {}
    rows: list[dict[str, Any]] = []
    for (ticker, fold_id), part in combined.groupby([v10args.ticker_column, "fold_id"], sort=True):
        codes = part["error_code"].to_numpy(dtype=np.int8)
        y = part[v10args.target_column].to_numpy(dtype=np.int8)
        ab_pairs = ab_lookup.get((str(ticker), int(fold_id)), pd.DataFrame())
        cd_pairs = cd_lookup.get((str(ticker), int(fold_id)), pd.DataFrame())
        for feature in features:
            values = part[feature].to_numpy(float)
            valid = np.isfinite(values)
            yy = y[valid]
            pos = int(np.sum(yy == 1)); neg = int(np.sum(yy == 0))
            pearson, pn = safe_pearson(values, y, minimum_rows=int(v10args.minimum_map_valid_rows))
            spearman, sn = safe_spearman(values, y, minimum_rows=int(v10args.minimum_map_valid_rows))
            rows.append({
                "ticker": str(ticker), "fold_id": int(fold_id), "role": str(part["role"].iloc[0]),
                "axis": "TARGET", "feature": feature, "transform": "raw",
                "raw_auc": safe_roc_auc(yy, values[valid]) if pos > 0 and neg > 0 else math.nan,
                "valid_rows": int(valid.sum()), "positive_n": pos, "negative_n": neg,
                "pearson": pearson, "pearson_n": pn, "spearman": spearman, "spearman_n": sn,
                "coverage": float(valid.mean()) if len(valid) else math.nan,
            })
            for axis, pc, nc, pairs in [("AB", ERROR_A_TOP_TP, ERROR_B_TOP_FP, ab_pairs), ("CD", ERROR_C_LOW_TP, ERROR_D_LOW_TN, cd_pairs)]:
                m = separation_metrics(values, codes, pc, nc, pairs, row_indices=part["row_index"].to_numpy(np.int64))
                rows.append({
                    "ticker": str(ticker), "fold_id": int(fold_id), "role": str(part["role"].iloc[0]),
                    "axis": axis, "feature": feature, "transform": "raw",
                    "valid_rows": int(m["positive_valid_n"] + m["negative_valid_n"]),
                    "positive_n": int(m["positive_valid_n"]), "negative_n": int(m["negative_valid_n"]), **m,
                })
        log(f"recent raw map ticker={ticker} fold={fold_id}")
    return pd.DataFrame(rows)


def load_matrix_payload(v10_output: Path, tickers: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    payload: dict[str, dict[str, np.ndarray]] = {}
    root = v10_output / "ticker_correlation_matrices"
    for ticker in tickers:
        path = root / f"{ticker}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            payload[str(ticker)] = {"feature_names": z["feature_names"], "correlation": z["correlation"], "valid_n": z["valid_n"] if "valid_n" in z else np.empty((0,0))}
    return payload


def compute_recent_extended_rows(
    transformed: pd.DataFrame,
    oof_groups: pd.DataFrame,
    selected_sources: pd.DataFrame,
    transform_manifest: Mapping[str, Mapping[str, Sequence[str]]],
    matched_ab: pd.DataFrame,
    matched_cd: pd.DataFrame,
    recent_folds: Sequence[int],
    v10args: argparse.Namespace,
) -> pd.DataFrame:
    ids = {int(x) for x in recent_folds}
    groups = oof_groups.loc[pd.to_numeric(oof_groups["fold_id"], errors="coerce").isin(ids)].copy()
    source_columns = sorted({c for t in transform_manifest.values() for cols in t.values() for c in cols})
    if not source_columns:
        return pd.DataFrame()
    merged = groups.merge(transformed[["source_row_id", *source_columns]], on="source_row_id", how="left", validate="one_to_one")
    ab_lookup = {(str(t), int(f)): p for (t, f), p in matched_ab.groupby(["ticker", "fold_id"], sort=False)} if not matched_ab.empty else {}
    cd_lookup = {(str(t), int(f)): p for (t, f), p in matched_cd.groupby(["ticker", "fold_id"], sort=False)} if not matched_cd.empty else {}
    axis_lookup = selected_sources.groupby(["ticker", "feature"])["axis"].apply(lambda x: sorted(set(x))).to_dict() if not selected_sources.empty else {}
    rows: list[dict[str, Any]] = []
    for (ticker, fold_id), part in merged.groupby([v10args.ticker_column, "fold_id"], sort=True):
        y = part[v10args.target_column].to_numpy(np.int8); codes = part["error_code"].to_numpy(np.int8)
        ab_pairs = ab_lookup.get((str(ticker), int(fold_id)), pd.DataFrame()); cd_pairs = cd_lookup.get((str(ticker), int(fold_id)), pd.DataFrame())
        ticker_manifest = transform_manifest.get(str(ticker), {})
        for source_feature, columns in ticker_manifest.items():
            axes = axis_lookup.get((str(ticker), str(source_feature)), ["TARGET", "AB", "CD"])
            for column in columns:
                if column not in part.columns:
                    continue
                values = part[column].to_numpy(float)
                transform = column.split("__", 1)[1] if "__" in column else "unknown"
                for axis in sorted(set(axes) | {"TARGET"}):
                    if axis == "TARGET":
                        valid = np.isfinite(values); yy = y[valid]
                        pos = int(np.sum(yy == 1)); neg = int(np.sum(yy == 0))
                        rows.append({"ticker":str(ticker),"fold_id":int(fold_id),"role":str(part["role"].iloc[0]),"axis":axis,"feature":str(source_feature),"transformed_feature":column,"transform":transform,"raw_auc":safe_roc_auc(yy, values[valid]) if pos and neg else math.nan,"valid_rows":int(valid.sum()),"positive_n":pos,"negative_n":neg})
                    else:
                        pc, nc, pairs = (ERROR_A_TOP_TP, ERROR_B_TOP_FP, ab_pairs) if axis == "AB" else (ERROR_C_LOW_TP, ERROR_D_LOW_TN, cd_pairs)
                        m = separation_metrics(values, codes, pc, nc, pairs, row_indices=part["row_index"].to_numpy(np.int64))
                        rows.append({"ticker":str(ticker),"fold_id":int(fold_id),"role":str(part["role"].iloc[0]),"axis":axis,"feature":str(source_feature),"transformed_feature":column,"transform":transform,"valid_rows":int(m["positive_valid_n"]+m["negative_valid_n"]),"positive_n":int(m["positive_valid_n"]),"negative_n":int(m["negative_valid_n"]),**m})
        log(f"recent extended map ticker={ticker} fold={fold_id}")
    return pd.DataFrame(rows)


def summarize_all_maps(by_fold: pd.DataFrame, v10args: argparse.Namespace) -> pd.DataFrame:
    roles = {"selection": parse_ints(v10args.selection_folds), "confirmation": parse_ints(v10args.confirmation_folds), "recent_audit": parse_ints(v10args.recent_folds)}
    parts: list[pd.DataFrame] = []
    normalized = by_fold.copy()
    if "transformed_feature" not in normalized.columns:
        normalized["transformed_feature"] = np.nan
    raw_mask = normalized["transformed_feature"].isna()
    for is_raw, frame in [(True, normalized.loc[raw_mask]), (False, normalized.loc[~raw_mask])]:
        if frame.empty:
            continue
        group_cols = ["ticker", "axis", "feature", "transform"] if is_raw else ["ticker", "axis", "feature", "transformed_feature", "transform"]
        for axis in ["TARGET", "AB", "CD"]:
            subset = frame.loc[frame["axis"].eq(axis)].copy()
            if subset.empty:
                continue
            s = summarize_fixed_direction(
                subset, group_columns=group_cols,
                selection_fold_ids=roles["selection"], confirmation_fold_ids=roles["confirmation"], recent_fold_ids=roles["recent_audit"],
                raw_auc_column="raw_auc", valid_rows_column="valid_rows", positive_n_column="positive_n", negative_n_column="negative_n",
                prior_strength=float(v10args.auc_prior_strength),
            )
            parts.append(s)
    out = pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()
    return v10.assign_map_evidence_grades(out, v10args) if not out.empty else out


def build_conservative_maps(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    target_summary: pd.DataFrame,
    selection_folds: Sequence[int],
    config: ConservativeHierarchyConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = prepare_base_effect_frame(summary, by_fold, target_summary, selection_folds)
    rings = build_disjoint_peer_rings(base)
    long_parts = [build_hierarchy_for_strength(base, rings, selection_folds, strength, config) for strength in config.prior_strength_grid]
    long = pd.concat(long_parts, ignore_index=True, sort=False)
    robust = aggregate_hierarchy_sensitivity(long, config)
    precision = build_precision_separator_map(robust, config)
    return long, robust, precision


def profile_discovery(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    target_summary: pd.DataFrame,
    discovery_folds: Sequence[int],
    config: ConservativeHierarchyConfig,
    top_tickers: int,
    max_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, robust, precision = build_conservative_maps(summary, by_fold, target_summary, discovery_folds, config)
    reps = deduplicate_precision_sources(precision, max_per_ticker=max_features)
    if reps.empty:
        return robust, precision, pd.DataFrame()
    ticker_rank = reps.groupby("ticker", as_index=False).agg(
        candidate_sources=("source_feature", "nunique"),
        selection_score_sum=("precision_separator_score_v10_2", "sum"),
        median_effective_n=("selection_effective_n", "median"),
        median_auc=("selection_mean_fixed_auc", "median"),
    ).sort_values(["candidate_sources", "selection_score_sum", "median_effective_n", "ticker"], ascending=[False, False, False, True], kind="mergesort")
    ticker_rank["probe_rank"] = np.arange(1, len(ticker_rank)+1)
    ticker_rank["selected_for_probe"] = ticker_rank["probe_rank"] <= int(top_tickers)
    return robust, precision, ticker_rank


def build_probe_profiles(
    discovery_robust: pd.DataFrame,
    discovery_precision: pd.DataFrame,
    ticker_rank: pd.DataFrame,
    max_ab_features: int,
    max_target_features: int,
) -> dict[str, dict[str, list[str]]]:
    if ticker_rank.empty or "selected_for_probe" not in ticker_rank.columns:
        return {}
    selected_tickers = set(ticker_rank.loc[ticker_rank["selected_for_probe"].astype(bool), "ticker"].astype(str))
    reps = deduplicate_precision_sources(discovery_precision, max_per_ticker=max_ab_features)
    profiles: dict[str, dict[str, list[str]]] = {}
    for ticker in sorted(selected_tickers):
        p = reps.loc[reps["ticker"].astype(str).eq(ticker)].copy()
        ab_nodes = p.sort_values("precision_separator_score_v10_2", ascending=False)["node_id"].astype(str).tolist()[:max_ab_features]
        target = discovery_robust.loc[(discovery_robust["ticker"].astype(str).eq(ticker)) & discovery_robust["axis"].eq("TARGET")].copy()
        target = target.sort_values(["ticker_specific_robust", "ticker_specificity_score_v10_2"], ascending=[False, False], kind="mergesort")
        target_nodes = target["node_id"].astype(str).drop_duplicates().tolist()[:max_target_features]
        profiles[ticker] = {
            "AB_CONSERVATIVE": ab_nodes,
            "AB_PLUS_TARGET": list(dict.fromkeys([*ab_nodes, *target_nodes])),
        }
    return profiles


def run_probe(
    frame: pd.DataFrame,
    transformed: pd.DataFrame,
    profiles: Mapping[str, Mapping[str, Sequence[str]]],
    combined_base_oof: pd.DataFrame,
    folds: Sequence[Any],
    ticker_index: Mapping[str, Mapping[int, Mapping[str, np.ndarray]]],
    eligibility_v10_2: pd.DataFrame,
    v10args: argparse.Namespace,
    eval_folds: Sequence[int],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_frame = v10.build_model_frame(frame, transformed, v10args)
    all_columns = set(model_frame.columns)
    e_lookup = eligibility_v10_2.set_index(["ticker", "fold_id"])
    base_lookup = combined_base_oof.set_index("row_index")
    y_all = model_frame[v10args.target_column].to_numpy(np.int8)
    device = v10.resolve_xgboost_device(v10args)
    pred_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    eval_set = {int(x) for x in eval_folds}
    backends = v10.parse_str_tokens(args.probe_backends)
    for ticker, ticker_profiles in profiles.items():
        for fold in folds:
            if int(fold.fold_id) not in eval_set:
                continue
            validation = np.asarray(ticker_index[ticker][fold.fold_id]["validation"], np.int64)
            train = np.asarray(ticker_index[ticker][fold.fold_id]["train"], np.int64)
            if not len(validation):
                continue
            yy = y_all[validation]
            for profile_name in ["BASE_TICKER", *ticker_profiles.keys()]:
                if profile_name == "BASE_TICKER":
                    score = base_lookup.reindex(validation)["base_score_raw"].to_numpy(float)
                    status = "OK" if np.isfinite(score).any() else "NO_BASE_SCORE"
                    used: list[str] = []
                else:
                    requested = [c for c in ticker_profiles[profile_name] if c in all_columns]
                    row = e_lookup.loc[(str(ticker), int(fold.fold_id))]
                    eligible = bool(row["eligible_model_v10_2"])
                    if not eligible or len(requested) < int(args.minimum_probe_features):
                        score = np.full(len(validation), np.nan); status = "LOW_EVIDENCE_NO_MODEL"; used = requested
                    else:
                        xtr = model_frame.iloc[train][requested].to_numpy(np.float32)
                        xva = model_frame.iloc[validation][requested].to_numpy(np.float32)
                        ytr = y_all[train]
                        def fit_probe_backend(backend: str) -> np.ndarray:
                            return v10.train_backend_predict(backend, xtr, ytr, xva, args=v10args, device=device, seed=int(args.seed)+int(fold.fold_id)*1009+int(hashlib.sha256((ticker+profile_name+backend).encode()).hexdigest()[:8],16)%100000)

                        scores = []
                        if bool(getattr(args, "parallel_backends", False)) and len(backends) > 1:
                            with ThreadPoolExecutor(max_workers=len(backends), thread_name_prefix="probe-backend") as pool:
                                backend_results = list(pool.map(fit_probe_backend, backends))
                        else:
                            backend_results = []
                            for backend in backends:
                                try:
                                    backend_results.append(fit_probe_backend(backend))
                                except Exception:
                                    if args.strict_backend:
                                        raise
                                    log(f"probe backend failed {ticker} fold={fold.fold_id} profile={profile_name} backend={backend}")
                        for backend, backend_score in zip(backends, backend_results):
                            try:
                                scores.append(np.asarray(backend_score, dtype=np.float64))
                            except Exception:
                                if args.strict_backend:
                                    raise
                                log(f"probe backend failed {ticker} fold={fold.fold_id} profile={profile_name} backend={backend}")
                        score = np.nanmean(np.vstack(scores), axis=0) if scores else np.full(len(validation), np.nan)
                        status = "OK" if scores else "ALL_BACKENDS_FAILED"; used = requested
                metrics = safe_binary_metrics(yy, score); prob = probability_metrics(yy, score)
                policy = select_threshold_for_precision(yy, score, target_precision=float(args.target_precision), minimum_alerts=int(args.minimum_alerts_per_ticker), minimum_recall=0.0, minimum_wilson_lcb=0.0)
                metric_rows.append({"ticker":ticker,"fold_id":int(fold.fold_id),"role":v10.role_for_fold(fold.fold_id,{"selection":[0,1,2,3,4],"confirmation":[5,6],"recent_audit":[7]}),"profile":profile_name,"status":status,"feature_count":len(used),**metrics,**prob,**{f"policy_{k}":v for k,v in policy.items()}})
                for local, idx in enumerate(validation):
                    pred_rows.append({"row_index":int(idx),"source_row_id":int(model_frame.at[idx,"source_row_id"]),"ticker":ticker,"fold_id":int(fold.fold_id),"profile":profile_name,"target":int(yy[local]),"score":float(score[local]) if np.isfinite(score[local]) else math.nan,"status":status})
    preds = pd.DataFrame(pred_rows); metrics = pd.DataFrame(metric_rows)
    # Choose each ticker champion only on selection validation folds 3-4.
    val = metrics.loc[metrics["fold_id"].isin([3,4]) & metrics["status"].eq("OK")].copy()
    champions: list[dict[str, Any]] = []
    for ticker, part in val.groupby("ticker", sort=True):
        agg = part.groupby("profile", as_index=False).agg(mean_pr_auc=("pr_auc","mean"),worst_pr_auc=("pr_auc","min"),mean_best_precision=("policy_best_practical_precision","mean"),folds=("fold_id","nunique"))
        agg = agg.loc[agg["folds"] >= 2]
        if agg.empty:
            champion = "BASE_TICKER"
        else:
            agg["is_compact"] = ~agg["profile"].eq("BASE_TICKER")
            agg = agg.sort_values(["mean_best_precision","worst_pr_auc","mean_pr_auc","is_compact","profile"], ascending=[False,False,False,False,True], kind="mergesort")
            champion = str(agg.iloc[0]["profile"])
        champions.append({"ticker":str(ticker),"champion_profile":champion})
    champion_df = pd.DataFrame(champions)
    routed = preds.merge(champion_df, on="ticker", how="left")
    routed = routed.loc[routed["profile"].eq(routed["champion_profile"])].copy()
    portfolio_rows: list[dict[str, Any]] = []
    for fold_id, part in routed.groupby("fold_id", sort=True):
        y = part["target"].to_numpy(np.int8); score = part["score"].to_numpy(float)
        m = safe_binary_metrics(y, score)
        pol = select_threshold_for_precision(y, score, target_precision=float(args.target_precision), minimum_alerts=int(args.minimum_portfolio_alerts), minimum_recall=0.0, minimum_wilson_lcb=0.0)
        portfolio_rows.append({"fold_id":int(fold_id),"role":v10.role_for_fold(int(fold_id),{"selection":[0,1,2,3,4],"confirmation":[5,6],"recent_audit":[7]}),**m,**{f"policy_{k}":v for k,v in pol.items()}})
    return preds, metrics, champion_df, pd.DataFrame(portfolio_rows)



def audit_industry_metadata(target_summary: pd.DataFrame) -> dict[str, Any]:
    values = target_summary.get("industry", pd.Series(["UNKNOWN"] * len(target_summary))).astype(str).fillna("UNKNOWN")
    names = target_summary.get("name", pd.Series(["UNKNOWN"] * len(target_summary))).astype(str).fillna("UNKNOWN")
    unknown = values.str.upper().isin({"UNKNOWN", "NAN", "NONE", "NULL", ""})
    non = values.loc[~unknown]
    unknown_ratio = float(unknown.mean()) if len(values) else 1.0
    unique_ratio = float(non.nunique() / max(len(non), 1))
    name_identity = float((non.reset_index(drop=True) == names.loc[~unknown].reset_index(drop=True)).mean()) if len(non) else 0.0
    group_sizes = non.value_counts()
    multi_values = set(group_sizes.loc[group_sizes >= 2].index.astype(str))
    multi_coverage = float(non.astype(str).isin(multi_values).mean()) if len(non) else 0.0
    valid = bool(unknown_ratio <= 0.35 and unique_ratio <= 0.70 and name_identity <= 0.50 and multi_coverage >= 0.50 and non.nunique() >= 2)
    return {"valid": valid, "unknown_ratio": unknown_ratio, "unique_ratio": unique_ratio, "name_identity_ratio": name_identity, "multi_member_coverage": multi_coverage}


def drop_invalid_industry_transforms(by_fold: pd.DataFrame, summary: pd.DataFrame, valid: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if valid:
        return by_fold, summary
    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        if "transform" not in frame.columns:
            return frame
        mask = ~frame["transform"].astype(str).str.contains("date_industry_rank", case=False, na=False)
        return frame.loc[mask].copy()
    return keep(by_fold), keep(summary)

def main() -> None:
    p = argparse.ArgumentParser(description="V10.2: conservative ticker-specific hierarchy, true Recent recalculation, and small-ticker precision probe.")
    p.add_argument("--package-root", default=".")
    p.add_argument("--v10-output", default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--target-sidecar", default=None)
    p.add_argument("--folds", default=None)
    p.add_argument("--feature-profile-manifest", default=None)
    p.add_argument("--feature-profile", default="P0_ALL_VALID")
    p.add_argument("--v10-1-output", default=None)
    p.add_argument("--output", default="outputs/surge_tickerwise_correlation_map_v10_2")
    p.add_argument("--prior-strength-grid", default="20,40,80,120")
    p.add_argument("--specific-fdr-ticker-axis", type=float, default=0.10)
    p.add_argument("--specific-fdr-global-axis", type=float, default=0.20)
    p.add_argument("--precision-min-selection-auc", type=float, default=0.57)
    p.add_argument("--precision-min-selection-min-auc", type=float, default=0.52)
    p.add_argument("--precision-min-direction-consistency", type=float, default=0.80)
    p.add_argument("--precision-min-effective-n", type=float, default=12.0)
    p.add_argument("--precision-min-specific-z", type=float, default=1.25)
    p.add_argument("--precision-min-matched-concordance", type=float, default=0.54)
    p.add_argument("--precision-confirm-auc", type=float, default=0.53)
    p.add_argument("--precision-recent-auc", type=float, default=0.53)
    p.add_argument("--configured-min-validation-rows", type=int, default=60)
    p.add_argument("--adaptive-recent-row-fraction", type=float, default=0.85)
    p.add_argument("--minimum-recent-rows-floor", type=int, default=45)
    p.add_argument("--minimum-validation-positive", type=int, default=3)
    p.add_argument("--minimum-validation-negative", type=int, default=20)
    p.add_argument("--base-backends", default="lightgbm_cpu,xgboost_gpu")
    p.add_argument("--probe-backends", default="lightgbm_cpu,xgboost_gpu")
    p.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    p.add_argument("--allow-cpu-fallback", action="store_true")
    p.add_argument("--strict-backend", action="store_true")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--xgboost-threads", type=int, default=8)
    p.add_argument("--parallel-backends", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--require-full-439", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--base-feature-count", type=int, default=64)
    p.add_argument("--minimum-base-feature-count", type=int, default=12)
    p.add_argument("--base-dedup-threshold", type=float, default=0.98)
    p.add_argument("--run-probe", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--probe-top-tickers", type=int, default=8)
    p.add_argument("--probe-discovery-folds", default="0,1,2")
    p.add_argument("--probe-evaluation-folds", default="3,4,5,6,7")
    p.add_argument("--probe-max-ab-features", type=int, default=20)
    p.add_argument("--probe-max-target-features", type=int, default=8)
    p.add_argument("--minimum-probe-features", type=int, default=4)
    p.add_argument("--minimum-alerts-per-ticker", type=int, default=3)
    p.add_argument("--minimum-portfolio-alerts", type=int, default=30)
    p.add_argument("--target-precision", type=float, default=0.70)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.package_root = Path(args.package_root).expanduser().resolve()
    output = Path(args.output); output = output if output.is_absolute() else (args.package_root / output).resolve(); output.mkdir(parents=True, exist_ok=True)
    status_path = output / "RUN_STATUS.json"
    atomic_json(status_path, {"schema":SCHEMA_VERSION,"status":"RUNNING"})
    try:
        v10_output = find_output(args.package_root, args.v10_output, "outputs/surge_tickerwise_correlation_map_v10")
        v10_1_output = find_output(args.package_root, args.v10_1_output, "outputs/surge_tickerwise_correlation_map_v10_1")
        v10args = build_v10_args(args, output)
        features = v10.load_feature_universe(v10args, v10.table_columns(Path(v10args.dataset)))
        frame = v10.load_dataset_frame(v10args, features)
        folds = v10.load_folds(Path(v10args.folds)); roles = v10.resolve_roles(v10args, folds)
        global_index = v10.build_global_fold_index(frame, folds, v10args.date_column)
        ticker_index, eligibility = v10.build_ticker_fold_index(frame, folds, global_index, v10args)
        recent_folds = parse_ints(v10args.recent_folds)
        eligibility2 = adaptive_eligibility(eligibility, recent_folds, args.configured_min_validation_rows, args.adaptive_recent_row_fraction, args.minimum_recent_rows_floor, args.minimum_validation_positive, args.minimum_validation_negative, v10args.minimum_train_rows_model, v10args.minimum_train_positive_model, v10args.minimum_train_negative_model)
        atomic_csv(output / "ticker_probe_eligibility_v10_2.csv", eligibility2)

        # Korean ticker codes are identifiers, not numbers.  Explicit string
        # dtypes preserve leading zeroes when V10 artifacts are reloaded; a
        # numeric inference here breaks the OOF metadata join (e.g. 000100 -> 100).
        ticker_dtype = {v10args.ticker_column: str, "ticker": str}
        old_oof = pd.read_csv(v10_output / "ticker_base_oof_predictions.csv", dtype=ticker_dtype)
        recent_oof, recent_feature_manifest = recompute_recent_base_oof(frame, features, folds, ticker_index, eligibility2, roles, recent_folds, v10args, output)
        combined_oof = combine_base_oof(old_oof, recent_oof, recent_folds)
        atomic_csv(output / "ticker_base_oof_predictions_v10_2.csv", combined_oof)
        atomic_csv(output / "recent_base_feature_manifest_v10_2.csv", recent_feature_manifest)
        oof_meta = v10.attach_oof_metadata(frame, combined_oof, v10args)
        groups, counts = v10.add_ticker_error_groups(oof_meta, v10args)
        atomic_csv(output / "ticker_error_group_oof_v10_2.csv", groups); atomic_csv(output / "ticker_error_group_counts_v10_2.csv", counts)
        matched_ab, matched_cd = v10.build_all_ticker_matched_pairs(frame, groups, features, v10args)
        atomic_csv(output / "ticker_matched_pairs_ab_v10_2.csv", matched_ab); atomic_csv(output / "ticker_matched_pairs_cd_v10_2.csv", matched_cd)

        recent_raw = compute_recent_stage0_rows(frame, groups, features, matched_ab, matched_cd, recent_folds, v10args)
        old_raw = pd.read_csv(v10_output / "ticker_feature_map_by_fold.csv", dtype=ticker_dtype)
        raw_all = pd.concat([old_raw.loc[~old_raw["fold_id"].isin(recent_folds)], recent_raw], ignore_index=True, sort=False)
        raw_all["transformed_feature"] = np.nan

        selected_sources = pd.read_csv(v10_output / "ticker_selected_source_features.csv", dtype=ticker_dtype)
        clusters = pd.read_csv(v10_output / "ticker_cluster_assignments.csv", dtype=ticker_dtype)
        tickers = sorted(frame[v10args.ticker_column].astype(str).unique())
        matrix_payload = load_matrix_payload(v10_output, tickers)
        transformed, transform_manifest = v10.build_selected_transforms(frame, selected_sources, clusters, matrix_payload, v10args)
        recent_ext = compute_recent_extended_rows(transformed, groups, selected_sources, transform_manifest, matched_ab, matched_cd, recent_folds, v10args)
        old_ext = pd.read_csv(v10_output / "ticker_extended_map_by_fold.csv", dtype=ticker_dtype)
        ext_all = pd.concat([old_ext.loc[~old_ext["fold_id"].isin(recent_folds)], recent_ext], ignore_index=True, sort=False)
        all_by_fold = pd.concat([raw_all, ext_all], ignore_index=True, sort=False)
        all_summary = summarize_all_maps(all_by_fold, v10args)

        target_summary = pd.read_csv(v10_output / "ticker_target_summary.csv", dtype=ticker_dtype)
        industry_audit = audit_industry_metadata(target_summary)
        atomic_json(output / "INDUSTRY_AUDIT_V10_2.json", industry_audit)
        all_by_fold, all_summary = drop_invalid_industry_transforms(all_by_fold, all_summary, bool(industry_audit["valid"]))
        atomic_csv(output / "ticker_map_by_fold_v10_2.csv", all_by_fold); atomic_csv(output / "ticker_map_summary_v10_2.csv", all_summary)
        config = ConservativeHierarchyConfig(
            prior_strength_grid=parse_floats(args.prior_strength_grid),
            specific_fdr_ticker_axis=float(args.specific_fdr_ticker_axis),
            specific_fdr_global_axis=float(args.specific_fdr_global_axis),
            precision_min_selection_auc=float(args.precision_min_selection_auc),
            precision_min_selection_min_auc=float(args.precision_min_selection_min_auc),
            precision_min_direction_consistency=float(args.precision_min_direction_consistency),
            precision_min_effective_n=float(args.precision_min_effective_n),
            precision_min_specific_z=float(args.precision_min_specific_z),
            precision_min_matched_concordance=float(args.precision_min_matched_concordance),
            precision_confirm_auc=float(args.precision_confirm_auc),
            precision_recent_auc=float(args.precision_recent_auc),
        )
        long, robust, precision = build_conservative_maps(all_summary, all_by_fold, target_summary, parse_ints(v10args.selection_folds), config)
        atomic_csv(output / "ticker_hierarchy_sensitivity_long_v10_2.csv", long)
        atomic_csv(output / "ticker_hierarchical_effect_map_v10_2.csv", robust)
        atomic_csv(output / "ticker_precision_separator_map_v10_2.csv", precision)
        reps = deduplicate_precision_sources(precision, max_per_ticker=args.probe_max_ab_features)
        atomic_csv(output / "ticker_precision_separator_source_representatives_v10_2.csv", reps)
        sim, edges, clusters2, signature = build_residual_similarity(robust, config)
        sim.to_csv(output / "ticker_residual_similarity_matrix_v10_2.csv")
        atomic_csv(output / "ticker_residual_similarity_edges_v10_2.csv", edges); atomic_csv(output / "ticker_residual_clusters_v10_2.csv", clusters2); atomic_csv(output / "ticker_residual_signature_nodes_v10_2.csv", signature)

        discovery_folds = parse_ints(args.probe_discovery_folds)
        d_robust, d_precision, ticker_rank = profile_discovery(all_summary, all_by_fold, target_summary, discovery_folds, config, args.probe_top_tickers, args.probe_max_ab_features)
        atomic_csv(output / "probe_discovery_precision_map_v10_2.csv", d_precision); atomic_csv(output / "probe_ticker_ranking_v10_2.csv", ticker_rank)
        profiles = build_probe_profiles(d_robust, d_precision, ticker_rank, args.probe_max_ab_features, args.probe_max_target_features)
        atomic_json(output / "PROBE_PROFILES_V10_2.json", profiles)

        probe_portfolio = pd.DataFrame(); champions = pd.DataFrame()
        if args.run_probe and profiles:
            preds, metrics, champions, probe_portfolio = run_probe(frame, transformed, profiles, combined_oof, folds, ticker_index, eligibility2, v10args, parse_ints(args.probe_evaluation_folds), args)
            atomic_csv(output / "ticker_probe_predictions_v10_2.csv", preds); atomic_csv(output / "ticker_probe_metrics_v10_2.csv", metrics); atomic_csv(output / "ticker_probe_champions_v10_2.csv", champions); atomic_csv(output / "ticker_probe_portfolio_by_fold_v10_2.csv", probe_portfolio)

        recent_precision = precision.loc[precision["precision_separator_selection_candidate"]].copy()
        recent_tested = int(recent_precision["recent_status"].ne("UNTESTED").sum()) if not recent_precision.empty else 0
        final = {
            "schema": SCHEMA_VERSION,
            "status": "V10_2_COMPLETE",
            "rows": int(len(frame)), "tickers": int(frame[v10args.ticker_column].nunique()), "features": int(len(features)),
            "recent_recomputed_ticker_folds": int((eligibility2["role"].eq("recent_audit") & eligibility2["eligible_model_v10_2"]).sum()),
            "selection_precision_candidates": int(precision["precision_separator_selection_candidate"].sum()) if not precision.empty else 0,
            "confirmation_supported": int((precision["confirmation_status"].eq("SUPPORTED") & precision["precision_separator_selection_candidate"]).sum()) if not precision.empty else 0,
            "recent_tested_selection_candidates": recent_tested,
            "recent_supported": int((precision["recent_status"].eq("SUPPORTED") & precision["precision_separator_selection_candidate"]).sum()) if not precision.empty else 0,
            "robust_ticker_specific_nodes": int(robust["ticker_specific_robust"].sum()) if not robust.empty else 0,
            "probe_tickers": ticker_rank.loc[ticker_rank["selected_for_probe"].astype(bool), "ticker"].astype(str).tolist() if (not ticker_rank.empty and "selected_for_probe" in ticker_rank.columns) else [],
            "probe_run": bool(args.run_probe),
            "production_action": "REVIEW_PROBE_BEFORE_ANY_ALERT",
        }
        if not probe_portfolio.empty:
            final["probe_portfolio"] = probe_portfolio.to_dict(orient="records")
        atomic_json(output / "FINAL_RECOMMENDATION_V10_2.json", final)
        atomic_json(status_path, {"schema":SCHEMA_VERSION,"status":"SUCCESS","final":final})
        log(json.dumps(final, ensure_ascii=False, indent=2))
    except BaseException as exc:
        atomic_json(status_path, {"schema":SCHEMA_VERSION,"status":"FAILED","error_type":type(exc).__name__,"error":str(exc),"traceback":traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
