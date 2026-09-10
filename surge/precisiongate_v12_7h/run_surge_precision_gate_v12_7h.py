from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V10_CODE = HERE / "v10_2_repro"
for p in (HERE, V10_CODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import run_surge_tickerwise_validation_v10_2 as v102  # type: ignore
from run_surge_precision_gate_v12 import (
    build_model_frame,
    log as log_v12,
    parse_tokens,
    require_file,
)
from surge_precision_gate_v12 import (
    apply_frozen_threshold,
    build_discovery_separator_specs,
    fit_evidence_transformer,
    serialize_specs,
    transform_evidence,
    safe_pr_auc,
)
from surge_precision_gate_v12_7h import (
    SCHEMA,
    TrialConfig,
    add_discovery_ticker_prior,
    build_prediction_frame,
    config_for_family,
    dev_summary,
    family_sort_key,
    fit_trial_model,
    fold_metrics,
    gpu_preflight,
    predict_trial_model,
    rank_hard_fp_features,
    stratified_bootstrap_indices,
)


def log(message: str) -> None:
    print(f"[V12-7H] {message}", flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def system_snapshot() -> dict[str, Any]:
    snap: dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu_count": os.cpu_count(),
    }
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        snap["nvidia_smi"] = result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        snap["nvidia_smi"] = f"unavailable: {exc}"
    return snap


def prepare_full_meta_source(
    model_frame: pd.DataFrame,
    v10args: argparse.Namespace,
    v10_2_output: Path,
    specs: dict[str, list[Any]],
    raw_features: Sequence[str],
) -> pd.DataFrame:
    base_path = require_file(v10_2_output / "ticker_base_oof_predictions_v10_2.csv", "V10.2 combined base OOF")
    base = pd.read_csv(base_path, dtype={"ticker": str, v10args.ticker_column: str})
    base["ticker"] = base["ticker"].astype(str).str.zfill(6)
    base["fold_id"] = pd.to_numeric(base["fold_id"], errors="coerce").astype("Int64")
    base = base.loc[base["fold_id"].isin(range(8))].copy()
    base = base.loc[pd.to_numeric(base["base_score_raw"], errors="coerce").notna()].copy()
    if "model_status" in base.columns:
        base = base.loc[base["model_status"].astype(str).eq("OK")].copy()

    nodes = sorted({spec.node_id for values in specs.values() for spec in values})
    raw = [str(c) for c in raw_features if str(c) in model_frame.columns]
    extra = [c for c in nodes if c not in raw]
    keep = ["row_index", "source_row_id", v10args.ticker_column, v10args.target_column, *raw, *extra]
    missing = [c for c in keep if c not in model_frame.columns]
    if missing:
        raise RuntimeError(f"model frame missing V12-7H columns: {missing[:20]}")
    right = model_frame[keep].copy()
    right = right.rename(columns={v10args.ticker_column: "ticker", v10args.target_column: "target"})
    right["ticker"] = right["ticker"].astype(str).str.zfill(6)
    merged = base.merge(right, on=["row_index", "source_row_id", "ticker"], how="inner", validate="one_to_one")
    merged = merged.loc[merged["ticker"].isin(specs)].copy()
    merged["target"] = pd.to_numeric(merged["target"], errors="raise").astype(np.int8)
    merged["fold_id"] = merged["fold_id"].astype(int)
    return merged.sort_values(["fold_id", "ticker", "row_index"], kind="mergesort").reset_index(drop=True)


def select_raw_features(
    ranking_cache: dict[float, pd.DataFrame],
    discovery_meta: pd.DataFrame,
    raw_features: Sequence[str],
    config: TrialConfig,
    output: Path,
) -> list[str]:
    q = float(config.candidate_quantile)
    if q not in ranking_cache:
        ranking = rank_hard_fp_features(
            discovery_meta,
            raw_features,
            candidate_quantile=q,
            minimum_rows=40,
        )
        ranking_cache[q] = ranking
        atomic_csv(output / "feature_rankings" / f"hard_fp_feature_ranking_q{q:.2f}.csv", ranking)
    ranking = ranking_cache[q]
    if int(config.feature_k) <= 0 or ranking.empty:
        return []
    stable = ranking.loc[
        (ranking["sign_consistency"] >= (2.0 / 3.0))
        & (ranking["folds_evaluable"] >= 2)
    ].copy()
    if stable.empty:
        stable = ranking
    return stable.head(min(int(config.feature_k), len(stable)))["feature"].astype(str).tolist()


def family_aggregate(seed_rows: list[dict[str, Any]], config: TrialConfig) -> dict[str, Any]:
    frame = pd.DataFrame(seed_rows)
    def col_mean(name: str, default: float = float("nan")) -> float:
        if name not in frame or frame.empty:
            return default
        v = pd.to_numeric(frame[name], errors="coerce")
        return float(v.mean()) if v.notna().any() else default
    def col_min(name: str, default: float = float("nan")) -> float:
        if name not in frame or frame.empty:
            return default
        v = pd.to_numeric(frame[name], errors="coerce")
        return float(v.min()) if v.notna().any() else default
    return {
        "family_id": int(config.family_id),
        "config_key": config.key(),
        **{k: v for k, v in vars(config).items() if k != "family_id"},
        "seeds_completed": int(len(seed_rows)),
        "seed_safe_fraction": float(np.mean(frame["dev_safe"].astype(float))) if not frame.empty else 0.0,
        "dev_safe": bool(frame["dev_safe"].all()) if not frame.empty else False,
        "min_dev_precision": col_min("min_dev_precision", 0.0),
        "mean_dev_precision": col_mean("mean_dev_precision", 0.0),
        "min_dev_pr_auc_delta": col_min("min_dev_pr_auc_delta", -999.0),
        "mean_dev_pr_auc_delta": col_mean("mean_dev_pr_auc_delta", -999.0),
        "mean_seconds_per_seed": col_mean("seconds", float("nan")),
        "gpu_seed_fraction": col_mean("gpu_used", 0.0),
    }


def train_seed_prediction(
    discovery_meta: pd.DataFrame,
    evaluation_meta: pd.DataFrame,
    selected_raw: Sequence[str],
    config: TrialConfig,
    *,
    seed: int,
    cpu_threads: int,
    gpu_available: bool,
    require_gpu: bool,
    minimum_evidence_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    t0 = time.monotonic()
    bundle = fit_trial_model(
        discovery_meta,
        selected_raw,
        config,
        seed=int(seed),
        cpu_threads=int(cpu_threads),
        gpu_available=bool(gpu_available),
        require_gpu_for_xgb=bool(require_gpu),
    )
    gate = predict_trial_model(bundle, evaluation_meta)
    pred = build_prediction_frame(
        evaluation_meta,
        gate,
        config,
        minimum_evidence_count=int(minimum_evidence_count),
    )
    return pred, {
        "seconds": float(time.monotonic() - t0),
        "gpu_used": bool(bundle.gpu_used),
        "train_features": int(len(bundle.feature_columns) + (len(bundle.ticker_categories) if bundle.ticker_onehot else 0)),
    }


def ensemble_prediction(preds: Sequence[pd.DataFrame], config: TrialConfig) -> pd.DataFrame:
    if not preds:
        raise ValueError("no predictions")
    base = preds[0].copy()
    gates = np.vstack([p["gate_prob"].to_numpy(float) for p in preds])
    gate = np.nanmean(gates, axis=0)
    # rebuild score with mean gate probability
    from surge_precision_gate_v12 import combine_base_and_gate
    base["gate_prob"] = gate
    base["score"] = combine_base_and_gate(base["base_score_raw"], gate, float(config.alpha))
    return base



def screen_sort_key(row: dict[str, Any]) -> tuple[float, ...]:
    """Broad-search ranking uses fold 3 only and does not treat threshold safety as proof."""
    def f(name: str, fallback: float = -999.0) -> float:
        try:
            x=float(row.get(name, fallback)); return x if math.isfinite(x) else fallback
        except Exception:
            return fallback
    return (
        f("min_dev_pr_auc_delta"),
        f("mean_dev_pr_auc_delta"),
        f("min_dev_precision", 0.0),
        f("mean_dev_precision", 0.0),
        -float(row.get("feature_k", 9999))/10000.0,
    )

def main() -> None:
    p = argparse.ArgumentParser(description="CrashWatch Surge V12 7H: time-budgeted hard-FP precision-gate research; direct parent V10.2.")
    p.add_argument("--package-root", default=".")
    p.add_argument("--v10-output", required=True)
    p.add_argument("--v10-2-output", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--target-sidecar", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--feature-profile-manifest", required=True)
    p.add_argument("--feature-profile", default="P0_ALL_VALID")
    p.add_argument("--output", default="outputs/surge_precision_gate_v12_7h")
    p.add_argument("--require-full-439", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--target-hours", type=float, default=7.0)
    p.add_argument("--cpu-threads", type=int, default=24)
    p.add_argument("--require-gpu", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-ab-features-per-ticker", type=int, default=24)
    p.add_argument("--search-seeds-per-family", type=int, default=3)
    p.add_argument("--top-families", type=int, default=8)
    p.add_argument("--robust-seeds", type=int, default=13)
    p.add_argument("--minimum-evidence-count", type=int, default=1)
    p.add_argument("--minimum-alerts", type=int, default=30)
    p.add_argument("--target-precision", type=float, default=0.70)
    p.add_argument("--discovery-folds", default="0,1,2")
    p.add_argument("--development-folds", default="3,4")
    p.add_argument("--confirmation-folds", default="5,6")
    p.add_argument("--recent-folds", default="7")
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-search-families", type=int, default=20000)
    p.add_argument("--max-stability-refits", type=int, default=20000)
    args = p.parse_args()

    start = time.monotonic()
    budget_s = max(float(args.target_hours) * 3600.0, 300.0)
    search_deadline = start + budget_s * 0.62
    robust_deadline = start + budget_s * 0.86
    ablation_deadline = start + budget_s * 0.94
    work_deadline = start + budget_s * 0.975

    args.package_root = Path(args.package_root).expanduser().resolve()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = (args.package_root / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "RUN_STATUS.json", {"schema": SCHEMA, "status": "RUNNING", "target_hours": args.target_hours})
    atomic_json(output / "SYSTEM_SNAPSHOT.json", system_snapshot())

    try:
        gpu = gpu_preflight(int(args.seed))
        atomic_json(output / "GPU_PREFLIGHT.json", gpu)
        if bool(args.require_gpu) and not bool(gpu.get("available", False)):
            raise RuntimeError(f"CUDA XGBoost preflight failed: {gpu}")
        log(f"GPU preflight: {gpu}")

        v10_2_output = Path(args.v10_2_output).expanduser().resolve()
        discovery_map_path = require_file(v10_2_output / "probe_discovery_precision_map_v10_2.csv", "V10.2 discovery map")
        model_frame, v10args, raw_features = build_model_frame(args, output)
        if bool(args.require_full_439) and len(raw_features) != 439:
            raise RuntimeError(f"expected inherited 439 features, got {len(raw_features)}")
        discovery_map = pd.read_csv(discovery_map_path, dtype={"ticker": str})
        specs = build_discovery_separator_specs(
            discovery_map,
            model_frame.columns,
            max_features_per_ticker=int(args.max_ab_features_per_ticker),
        )
        if not specs:
            raise RuntimeError("no discovery-only V10.2 A/B specs")
        atomic_csv(output / "V12_7H_DISCOVERY_SEPARATOR_MANIFEST.csv", pd.DataFrame(serialize_specs(specs)))

        meta_source = prepare_full_meta_source(model_frame, v10args, v10_2_output, specs, raw_features)
        discovery_folds = [int(x) for x in parse_tokens(args.discovery_folds)]
        dev_folds = [int(x) for x in parse_tokens(args.development_folds)]
        if len(dev_folds) < 2:
            raise ValueError("V12-7H expects at least two development folds; default is 3,4")
        search_folds = [dev_folds[0]]  # broad search touches only fold 3; fold 4 is reserved for robust development validation.
        conf_folds = [int(x) for x in parse_tokens(args.confirmation_folds)]
        recent_folds = [int(x) for x in parse_tokens(args.recent_folds)]
        if set(discovery_folds) & set(dev_folds + conf_folds + recent_folds):
            raise ValueError("fold overlap")
        d_rows = meta_source.loc[meta_source["fold_id"].isin(discovery_folds)].copy()
        e_rows = meta_source.loc[meta_source["fold_id"].isin(dev_folds + conf_folds + recent_folds)].copy()
        if d_rows.empty or e_rows.empty:
            raise RuntimeError("missing discovery/evaluation rows")

        transformer = fit_evidence_transformer(
            d_rows,
            specs,
            ticker_column="ticker",
            base_score_column="base_score_raw",
            max_slots=int(args.max_ab_features_per_ticker),
        )
        d_evidence = transform_evidence(d_rows, transformer, ticker_column="ticker", base_score_column="base_score_raw")
        e_evidence = transform_evidence(e_rows, transformer, ticker_column="ticker", base_score_column="base_score_raw")
        d_meta = pd.concat([d_rows.reset_index(drop=True), d_evidence.reset_index(drop=True)], axis=1)
        e_meta = pd.concat([e_rows.reset_index(drop=True), e_evidence.reset_index(drop=True)], axis=1)
        d_meta, e_meta = add_discovery_ticker_prior(d_meta, e_meta, strength=80.0)
        d_meta["ticker"] = d_meta["ticker"].astype(str)
        e_meta["ticker"] = e_meta["ticker"].astype(str)
        atomic_json(output / "DATA_AUDIT.json", {
            "rows_meta": int(len(meta_source)), "discovery_rows": int(len(d_meta)), "evaluation_rows": int(len(e_meta)),
            "tickers": int(meta_source["ticker"].nunique()), "raw_features": int(len(raw_features)),
            "separator_tickers": int(len(specs)), "separator_nodes": int(sum(len(v) for v in specs.values())),
        })
        log(f"Prepared {len(d_meta):,} discovery + {len(e_meta):,} eval rows; {len(raw_features)} raw features")

        ranking_cache: dict[float, pd.DataFrame] = {}
        seed_result_path = output / "search_seed_results.csv"
        family_result_path = output / "search_family_results.csv"
        existing_seed = pd.read_csv(seed_result_path) if args.resume and seed_result_path.exists() else pd.DataFrame()
        existing_family = pd.read_csv(family_result_path) if args.resume and family_result_path.exists() else pd.DataFrame()
        completed_families = set(existing_family["family_id"].astype(int).tolist()) if not existing_family.empty else set()
        seed_records = existing_seed.to_dict(orient="records") if not existing_seed.empty else []
        family_records = existing_family.to_dict(orient="records") if not existing_family.empty else []

        # PHASE 1 — broad but seed-replicated search. Each architecture gets 3 seeds by default.
        family_id = 0
        while time.monotonic() < search_deadline and family_id < int(args.max_search_families):
            if family_id in completed_families:
                family_id += 1
                continue
            config = config_for_family(family_id, int(args.seed))
            selected_raw = select_raw_features(ranking_cache, d_meta, raw_features, config, output)
            this_seed_rows: list[dict[str, Any]] = []
            for j in range(int(args.search_seeds_per_family)):
                seed = int(args.seed) + family_id * 1009 + j * 7919
                try:
                    pred, meta = train_seed_prediction(
                        d_meta, e_meta.loc[e_meta["fold_id"].isin(search_folds)].reset_index(drop=True),
                        selected_raw, config, seed=seed, cpu_threads=int(args.cpu_threads),
                        gpu_available=bool(gpu["available"]), require_gpu=bool(args.require_gpu),
                        minimum_evidence_count=int(args.minimum_evidence_count),
                    )
                    summary = dev_summary(
                        pred, dev_folds=search_folds, minimum_alerts=int(args.minimum_alerts),
                        target_precision=float(args.target_precision),
                    )
                    row = {
                        "family_id": family_id, "config_key": config.key(), "seed": seed, "search_fold": search_folds[0],
                        "selected_raw_features": len(selected_raw), **vars(config), **meta,
                        **{k: v for k, v in summary.items() if k not in {"dev_policy", "dev_metrics"}},
                    }
                except Exception as exc:
                    row = {
                        "family_id": family_id, "config_key": config.key(), "seed": seed, "search_fold": search_folds[0],
                        "selected_raw_features": len(selected_raw), **vars(config),
                        "error": f"{type(exc).__name__}: {exc}", "dev_safe": False,
                        "min_dev_precision": 0.0, "mean_dev_precision": 0.0,
                        "min_dev_pr_auc_delta": -999.0, "mean_dev_pr_auc_delta": -999.0,
                        "seconds": 0.0, "gpu_used": False,
                    }
                this_seed_rows.append(row)
                seed_records.append(row)
            fam = family_aggregate(this_seed_rows, config)
            family_records.append(fam)
            completed_families.add(family_id)
            atomic_csv(seed_result_path, pd.DataFrame(seed_records))
            atomic_csv(family_result_path, pd.DataFrame(family_records))
            elapsed_h = (time.monotonic() - start) / 3600.0
            atomic_json(output / "PROGRESS_V12_7H.json", {
                "phase": "SEARCH", "elapsed_hours": elapsed_h, "target_hours": float(args.target_hours),
                "families_completed": len(completed_families), "last_family": fam,
            })
            if family_id % 5 == 0:
                log(f"search family={family_id} backend={config.backend} k={config.feature_k} q={config.candidate_quantile:.2f} safe={fam['dev_safe']} elapsed={elapsed_h:.2f}h")
            family_id += 1

        if not family_records:
            raise RuntimeError("no search families completed")
        fam_df = pd.DataFrame(family_records).drop_duplicates("family_id", keep="last")
        fam_sorted = sorted(fam_df.to_dict(orient="records"), key=screen_sort_key, reverse=True)
        # diversity: avoid spending robustness budget on identical backend/q/k corners only
        top: list[dict[str, Any]] = []
        seen_signature: set[tuple[Any, ...]] = set()
        for row in fam_sorted:
            sig = (row.get("backend"), row.get("candidate_quantile"), row.get("feature_k"), row.get("alpha"))
            if sig in seen_signature and len(top) < max(3, int(args.top_families) // 2):
                continue
            seen_signature.add(sig)
            top.append(row)
            if len(top) >= int(args.top_families):
                break
        atomic_csv(output / "TOP_SEARCH_FAMILIES.csv", pd.DataFrame(top))
        log(f"Search complete: {len(fam_df)} families; robustness candidates={len(top)}")

        # PHASE 2 — re-fit top architectures with many seeds and select only from dev 3-4.
        robust_rows: list[dict[str, Any]] = []
        robust_predictions: dict[int, pd.DataFrame] = {}
        for rank, row in enumerate(top, start=1):
            if time.monotonic() >= robust_deadline:
                break
            config = TrialConfig(**{k: row[k] for k in TrialConfig.__dataclass_fields__.keys()})
            selected_raw = select_raw_features(ranking_cache, d_meta, raw_features, config, output)
            preds: list[pd.DataFrame] = []
            seed_meta: list[dict[str, Any]] = []
            for j in range(int(args.robust_seeds)):
                if time.monotonic() >= robust_deadline and len(preds) >= 3:
                    break
                seed = int(args.seed) + 700000 + int(config.family_id) * 3001 + j * 104729
                pred, meta = train_seed_prediction(
                    d_meta, e_meta, selected_raw, config, seed=seed,
                    cpu_threads=int(args.cpu_threads), gpu_available=bool(gpu["available"]),
                    require_gpu=bool(args.require_gpu), minimum_evidence_count=int(args.minimum_evidence_count),
                )
                preds.append(pred)
                seed_meta.append(meta)
            if len(preds) < 2:
                continue
            ensemble = ensemble_prediction(preds, config)
            summary = dev_summary(
                ensemble, dev_folds=dev_folds, minimum_alerts=int(args.minimum_alerts),
                target_precision=float(args.target_precision),
            )
            rr = {
                "robust_rank_input": rank, "family_id": int(config.family_id), "config_key": config.key(),
                **{k: v for k, v in vars(config).items() if k != "family_id"},
                "selected_raw_features": len(selected_raw), "robust_seeds_completed": len(preds),
                "gpu_fraction": float(np.mean([m["gpu_used"] for m in seed_meta])),
                **{k: v for k, v in summary.items() if k not in {"dev_policy", "dev_metrics"}},
            }
            robust_rows.append(rr)
            robust_predictions[int(config.family_id)] = ensemble
            atomic_csv(output / "robust_family_results.csv", pd.DataFrame(robust_rows))
            log(f"robust family={config.family_id} seeds={len(preds)} safe={rr['dev_safe']} minP={rr['min_dev_precision']:.3f}")
        if not robust_rows:
            # fallback to best search family with a fresh 3-seed ensemble
            row = top[0]
            config = TrialConfig(**{k: row[k] for k in TrialConfig.__dataclass_fields__.keys()})
            selected_raw = select_raw_features(ranking_cache, d_meta, raw_features, config, output)
            preds = []
            for j in range(3):
                pred, _ = train_seed_prediction(
                    d_meta, e_meta, selected_raw, config, seed=int(args.seed) + 900000 + j,
                    cpu_threads=int(args.cpu_threads), gpu_available=bool(gpu["available"]), require_gpu=bool(args.require_gpu),
                    minimum_evidence_count=int(args.minimum_evidence_count),
                )
                preds.append(pred)
            ens = ensemble_prediction(preds, config)
            sm = dev_summary(ens, dev_folds=dev_folds, minimum_alerts=args.minimum_alerts, target_precision=args.target_precision)
            robust_rows.append({"family_id": config.family_id, **vars(config), **{k: v for k, v in sm.items() if k not in {"dev_policy", "dev_metrics"}}})
            robust_predictions[config.family_id] = ens

        robust_sorted = sorted(robust_rows, key=family_sort_key, reverse=True)
        champion_row = robust_sorted[0]
        champion = TrialConfig(**{k: champion_row[k] for k in TrialConfig.__dataclass_fields__.keys()})
        champion_selected_raw = select_raw_features(ranking_cache, d_meta, raw_features, champion, output)
        champion_pred = robust_predictions[int(champion.family_id)].copy()
        champion_dev = dev_summary(
            champion_pred, dev_folds=dev_folds, minimum_alerts=int(args.minimum_alerts), target_precision=float(args.target_precision)
        )
        frozen_threshold = float(champion_dev["threshold"])
        atomic_json(output / "CHAMPION_CONFIG_V12_7H.json", {**vars(champion), "selected_raw_features": champion_selected_raw, "dev": champion_dev})

        # PHASE 3 — focused component ablations; no confirmation/recent labels influence champion selection.
        ablation_rows: list[dict[str, Any]] = []
        ablation_variants = [
            ("EVIDENCE_ONLY", 0),
            ("TOP32_RAW", min(32, max(0, champion.feature_k))),
            ("TOP64_RAW", min(64, max(0, champion.feature_k))),
            ("CHAMPION_RAW_K", champion.feature_k),
        ]
        seen_k: set[int] = set()
        for name, k in ablation_variants:
            if time.monotonic() >= ablation_deadline:
                break
            if int(k) in seen_k:
                continue
            seen_k.add(int(k))
            variant = TrialConfig(**{**vars(champion), "feature_k": int(k)})
            selected = select_raw_features(ranking_cache, d_meta, raw_features, variant, output)
            preds = []
            for j in range(min(5, int(args.robust_seeds))):
                pred, _ = train_seed_prediction(
                    d_meta, e_meta.loc[e_meta["fold_id"].isin(dev_folds)].reset_index(drop=True), selected, variant,
                    seed=int(args.seed) + 1200000 + int(k) * 17 + j * 1009,
                    cpu_threads=int(args.cpu_threads), gpu_available=bool(gpu["available"]), require_gpu=bool(args.require_gpu),
                    minimum_evidence_count=int(args.minimum_evidence_count),
                )
                preds.append(pred)
            ens = ensemble_prediction(preds, variant)
            sm = dev_summary(ens, dev_folds=dev_folds, minimum_alerts=args.minimum_alerts, target_precision=args.target_precision)
            ablation_rows.append({"variant": name, "feature_k": int(k), "seeds": len(preds), **{x: y for x, y in sm.items() if x not in {"dev_policy", "dev_metrics"}}})
            atomic_csv(output / "champion_component_ablation.csv", pd.DataFrame(ablation_rows))

        # PHASE 4 — spend remaining useful budget on bootstrap-refit stability of the already selected champion.
        # This does not search new configurations; it measures how sensitive the chosen gate is to discovery sampling.
        stability_rows: list[dict[str, Any]] = []
        dev_eval = e_meta.loc[e_meta["fold_id"].isin(dev_folds)].reset_index(drop=True)
        rng = np.random.default_rng(int(args.seed) + 990001)
        rep = 0
        while time.monotonic() < work_deadline and rep < int(args.max_stability_refits):
            boot_idx = stratified_bootstrap_indices(d_meta, rng)
            boot = d_meta.loc[boot_idx].reset_index(drop=True)
            try:
                pred, meta = train_seed_prediction(
                    boot, dev_eval, champion_selected_raw, champion,
                    seed=int(args.seed) + 2000000 + rep * 9973,
                    cpu_threads=int(args.cpu_threads), gpu_available=bool(gpu["available"]), require_gpu=bool(args.require_gpu),
                    minimum_evidence_count=int(args.minimum_evidence_count),
                )
                sm = dev_summary(pred, dev_folds=dev_folds, minimum_alerts=args.minimum_alerts, target_precision=args.target_precision)
                stability_rows.append({
                    "replicate": rep, **meta,
                    **{k: v for k, v in sm.items() if k not in {"dev_policy", "dev_metrics"}},
                })
            except Exception as exc:
                stability_rows.append({"replicate": rep, "error": f"{type(exc).__name__}: {exc}"})
            rep += 1
            if rep % 10 == 0:
                atomic_csv(output / "champion_bootstrap_refit_stability.csv", pd.DataFrame(stability_rows))
                log(f"stability refits={rep} elapsed={(time.monotonic()-start)/3600:.2f}h")
        if stability_rows:
            atomic_csv(output / "champion_bootstrap_refit_stability.csv", pd.DataFrame(stability_rows))

        # FINAL — one frozen champion/threshold evaluation on folds 3-7.
        final_metrics = fold_metrics(champion_pred, dev_folds + conf_folds + recent_folds)
        final_policy = apply_frozen_threshold(
            champion_pred.rename(columns={"score": "v12_score"}), frozen_threshold,
            folds=dev_folds + conf_folds + recent_folds, score_column="v12_score",
        )
        final_policy["role"] = final_policy["fold_id"].map({
            **{x: "development" for x in dev_folds}, **{x: "confirmation" for x in conf_folds}, **{x: "recent_diagnostic" for x in recent_folds}
        })
        final_policy["gate_pass"] = (
            (final_policy["alerts"] >= int(args.minimum_alerts))
            & (final_policy["precision"] >= float(args.target_precision))
        )
        ticker_rows: list[dict[str, Any]] = []
        for (ticker, fold), g in champion_pred.groupby(["ticker", "fold_id"], sort=True):
            ticker_rows.append({
                "ticker": str(ticker), "fold_id": int(fold), "rows": int(len(g)), "positives": int(g["target"].sum()),
                "base_pr_auc": safe_pr_auc(g["target"], g["base_score_raw"]),
                "v12_7h_pr_auc": safe_pr_auc(g["target"], g["score"]),
                "eligible_rows": int(g["candidate_eligible"].sum()),
            })
        ticker_df = pd.DataFrame(ticker_rows)
        if not ticker_df.empty:
            ticker_df["pr_auc_delta"] = ticker_df["v12_7h_pr_auc"] - ticker_df["base_pr_auc"]

        atomic_csv(output / "v12_7h_champion_predictions.csv", champion_pred)
        atomic_csv(output / "v12_7h_metrics_by_fold.csv", final_metrics)
        atomic_csv(output / "v12_7h_frozen_policy_by_fold.csv", final_policy)
        atomic_csv(output / "v12_7h_ticker_metrics_by_fold.csv", ticker_df)

        stability = pd.DataFrame(stability_rows)
        stability_summary: dict[str, Any] = {"replicates": int(len(stability))}
        if not stability.empty and "min_dev_precision" in stability:
            good = stability.loc[pd.to_numeric(stability["min_dev_precision"], errors="coerce").notna()].copy()
            if not good.empty:
                stability_summary.update({
                    "safe_fraction": float(good["dev_safe"].astype(float).mean()),
                    "min_precision_p10": float(pd.to_numeric(good["min_dev_precision"], errors="coerce").quantile(0.10)),
                    "min_precision_median": float(pd.to_numeric(good["min_dev_precision"], errors="coerce").median()),
                    "mean_pr_auc_delta_p10": float(pd.to_numeric(good["mean_dev_pr_auc_delta"], errors="coerce").quantile(0.10)),
                })

        conf_passes = int(final_policy.loc[final_policy["fold_id"].isin(conf_folds), "gate_pass"].sum())
        recent_passes = int(final_policy.loc[final_policy["fold_id"].isin(recent_folds), "gate_pass"].sum())
        dev_safe = bool(champion_dev["dev_safe"])
        if not dev_safe:
            research_status = "DEV_GATE_FAIL"
        elif conf_passes == len(conf_folds) and recent_passes == len(recent_folds):
            research_status = "DIAGNOSTICALLY_SUPPORTED_NOT_UNTOUCHED"
        else:
            research_status = "DEV_GATE_PASS_FORWARD_DIAGNOSTIC_MIXED"
        production_action = "NO_ALERT" if not dev_safe else "NO_ALERT_UNTIL_NEW_FUTURE_DATA"

        elapsed_h = (time.monotonic() - start) / 3600.0
        final = {
            "schema": SCHEMA,
            "status": "V12_7H_COMPLETE",
            "direct_parent": "V10.2",
            "lineage": "V1_TO_V10_2_TO_V12_7H",
            "v11_v11_1": "REFERENCE_ONLY_NOT_USED",
            "target_hours": float(args.target_hours),
            "elapsed_hours": float(elapsed_h),
            "search_families_completed": int(len(fam_df)),
            "robust_families_completed": int(len(robust_rows)),
            "stability_refits_completed": int(len(stability_rows)),
            "champion": {**vars(champion), "selected_raw_features": len(champion_selected_raw), "selected_raw_feature_names": champion_selected_raw},
            "champion_dev": champion_dev,
            "stability": stability_summary,
            "frozen_policy": final_policy.to_dict(orient="records"),
            "confirmation_gate_passes": conf_passes,
            "recent_gate_passes": recent_passes,
            "research_status": research_status,
            "production_action": production_action,
            "target_policy": {"precision": float(args.target_precision), "minimum_alerts": int(args.minimum_alerts)},
            "holdout_warning": "folds 5-7 were previously inspected; final truth requires new future data.",
        }
        atomic_json(output / "FINAL_RECOMMENDATION_V12_7H.json", final)
        atomic_json(output / "LEAKAGE_CONTRACT_V12_7H.json", {
            "parent": "V10.2",
            "v11_v11_1": "reference-only; no lead-lag feature enters V12-7H",
            "separator_identity_direction": "V10.2 discovery folds 0-2 only",
            "raw_hard_fp_feature_ranking": "computed only on discovery folds 0-2 high-base-score candidates",
            "model_fit": "discovery folds 0-2 only",
            "broad_architecture_screen": "fold 3 only; threshold safety here is screening-only",
            "configuration_and_threshold_selection": "top families are re-fit and final configuration + one threshold are selected jointly on development folds 3-4; fold 4 is not used in broad search",
            "confirmation_recent": "folds 5-7 evaluated after champion and threshold freeze; diagnostic, not untouched holdout",
            "base_rank": "empirical historical rank fitted on discovery rows only",
            "time_budget": "additional wall-clock budget is spent on champion bootstrap-refit stability rather than new dev configurations after champion selection",
        })
        atomic_json(output / "RUN_STATUS.json", {
            "schema": SCHEMA, "status": "SUCCESS", "final": final,
            "inputs": {"discovery_map_sha256": sha256(discovery_map_path), "v10_2_output": str(v10_2_output)},
        })
        log(json.dumps(final, ensure_ascii=False, indent=2, default=str))
    except BaseException as exc:
        atomic_json(output / "RUN_STATUS.json", {
            "schema": SCHEMA, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(), "elapsed_hours": (time.monotonic() - start) / 3600.0,
        })
        raise


if __name__ == "__main__":
    main()
