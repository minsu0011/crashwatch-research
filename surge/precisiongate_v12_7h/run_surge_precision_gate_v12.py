from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
V10_CODE = HERE / "v10_2_repro"
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(V10_CODE) not in sys.path:
    sys.path.insert(0, str(V10_CODE))

import run_surge_tickerwise_correlation_map_v10 as v10  # type: ignore
import run_surge_tickerwise_validation_v10_2 as v102  # type: ignore
from surge_precision_gate_v12 import (
    V12_SCHEMA,
    apply_frozen_threshold,
    build_discovery_separator_specs,
    combine_base_and_gate,
    config_sort_key,
    evaluate_scores,
    fit_evidence_transformer,
    fit_gate_bundle,
    predict_gate_bundle,
    safe_pr_auc,
    select_frozen_dev_threshold,
    serialize_specs,
    transform_evidence,
)


def log(message: str) -> None:
    print(f"[V12] {message}", flush=True)


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


def parse_tokens(value: str) -> list[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def parse_floats(value: str) -> list[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label}: {path}")
    return path


def build_v10_runtime_args(args: argparse.Namespace, scratch_output: Path) -> argparse.Namespace:
    v = v10.build_parser().parse_args([])
    v.package_root = Path(args.package_root).resolve()
    v.output = scratch_output
    v.dataset = Path(args.dataset).expanduser().resolve()
    v.target_sidecar = Path(args.target_sidecar).expanduser().resolve()
    v.folds = Path(args.folds).expanduser().resolve()
    v.feature_profile_manifest = Path(args.feature_profile_manifest).expanduser().resolve()
    v.feature_profile = str(args.feature_profile)
    v.require_full_439 = bool(args.require_full_439)
    v.device = "cpu"  # V12 does not retrain the V10 base model.
    v.resume = True
    return v


def build_model_frame(args: argparse.Namespace, output: Path) -> tuple[pd.DataFrame, argparse.Namespace, list[str]]:
    v10args = build_v10_runtime_args(args, output / "_v10_runtime")
    columns = v10.table_columns(Path(v10args.dataset))
    features = v10.load_feature_universe(v10args, columns)
    frame = v10.load_dataset_frame(v10args, features)
    v10_output = Path(args.v10_output).expanduser().resolve()
    ticker_dtype = {v10args.ticker_column: str, "ticker": str}
    selected_sources = pd.read_csv(require_file(v10_output / "ticker_selected_source_features.csv", "V10 selected sources"), dtype=ticker_dtype)
    clusters = pd.read_csv(require_file(v10_output / "ticker_cluster_assignments.csv", "V10 clusters"), dtype=ticker_dtype)
    tickers = sorted(frame[v10args.ticker_column].astype(str).str.zfill(6).unique())
    matrix_payload = v102.load_matrix_payload(v10_output, tickers)
    transformed, _manifest = v10.build_selected_transforms(frame, selected_sources, clusters, matrix_payload, v10args)
    model_frame = v10.build_model_frame(frame, transformed, v10args)
    model_frame = model_frame.copy()
    model_frame["row_index"] = model_frame.index.astype(int)
    model_frame[v10args.ticker_column] = model_frame[v10args.ticker_column].astype(str).str.zfill(6)
    return model_frame, v10args, features


def prepare_meta_source(
    model_frame: pd.DataFrame,
    v10args: argparse.Namespace,
    v10_2_output: Path,
    specs: dict[str, list[Any]],
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
    keep = [
        "row_index", "source_row_id", v10args.ticker_column, v10args.target_column, *nodes
    ]
    missing = [c for c in keep if c not in model_frame.columns]
    if missing:
        raise RuntimeError(f"model frame missing V12 columns: {missing[:20]}")
    right = model_frame[keep].copy()
    right = right.rename(columns={v10args.ticker_column: "ticker", v10args.target_column: "target"})
    right["ticker"] = right["ticker"].astype(str).str.zfill(6)
    merged = base.merge(right, on=["row_index", "source_row_id", "ticker"], how="inner", validate="one_to_one")
    merged = merged.loc[merged["ticker"].isin(specs)].copy()
    merged["target"] = pd.to_numeric(merged["target"], errors="raise").astype(np.int8)
    merged["fold_id"] = merged["fold_id"].astype(int)
    return merged.sort_values(["fold_id", "ticker", "row_index"], kind="mergesort").reset_index(drop=True)


def evaluate_base_matched(pred: pd.DataFrame, minimum_alerts: int) -> pd.DataFrame:
    work = pred.copy()
    work["base_eval_score"] = pd.to_numeric(work["base_score_raw"], errors="coerce")
    return evaluate_scores(
        work,
        score_column="base_eval_score",
        base_score_column="base_score_raw",
        target_column="target",
        minimum_alerts=minimum_alerts,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="CrashWatch Surge V12: V10.2-derived hard-FP precision gate. V11/V11.1 are reference-only and not inputs."
    )
    p.add_argument("--package-root", default=".")
    p.add_argument("--v10-output", required=True, help="Full V10 output directory (selected transforms/matrices).")
    p.add_argument("--v10-2-output", required=True, help="Full V10.2 output directory, not only compact compact result archive.")
    p.add_argument("--dataset", required=True)
    p.add_argument("--target-sidecar", required=True)
    p.add_argument("--folds", required=True)
    p.add_argument("--feature-profile-manifest", required=True)
    p.add_argument("--feature-profile", default="P0_ALL_VALID")
    p.add_argument("--output", default="outputs/surge_precision_gate_v12")
    p.add_argument("--require-full-439", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--discovery-folds", default="0,1,2")
    p.add_argument("--development-folds", default="3,4")
    p.add_argument("--confirmation-folds", default="5,6")
    p.add_argument("--recent-folds", default="7")
    p.add_argument("--max-ab-features-per-ticker", type=int, default=12)
    p.add_argument("--candidate-quantiles", default="0.50,0.65,0.80")
    p.add_argument("--alphas", default="0.50,0.75,1.00")
    p.add_argument("--backends", default="logit,lgbm")
    p.add_argument("--negative-weight", type=float, default=2.0)
    p.add_argument("--ticker-prior-strength", type=float, default=60.0)
    p.add_argument("--min-ticker-gate-rows", type=int, default=30)
    p.add_argument("--min-ticker-gate-class", type=int, default=5)
    p.add_argument("--minimum-evidence-count", type=int, default=1)
    p.add_argument("--minimum-alerts", type=int, default=30)
    p.add_argument("--target-precision", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=17)
    args = p.parse_args()

    args.package_root = Path(args.package_root).expanduser().resolve()
    output = Path(args.output).expanduser()
    if not output.is_absolute():
        output = (args.package_root / output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "RUN_STATUS.json"
    atomic_json(status_path, {"schema": V12_SCHEMA, "status": "RUNNING"})

    try:
        v10_2_output = Path(args.v10_2_output).expanduser().resolve()
        discovery_map_path = require_file(
            v10_2_output / "probe_discovery_precision_map_v10_2.csv",
            "V10.2 discovery-only precision map (folds 0-2)",
        )
        model_frame, v10args, features = build_model_frame(args, output)
        if bool(args.require_full_439) and len(features) != 439:
            raise RuntimeError(f"V12 requires the inherited 439-feature universe; got {len(features)}")
        discovery = pd.read_csv(discovery_map_path, dtype={"ticker": str})
        specs = build_discovery_separator_specs(
            discovery,
            model_frame.columns,
            max_features_per_ticker=int(args.max_ab_features_per_ticker),
        )
        if not specs:
            raise RuntimeError("No discovery-only A/B separator specs survived V10.2 candidate rules.")
        atomic_csv(output / "V12_DISCOVERY_SEPARATOR_MANIFEST.csv", pd.DataFrame(serialize_specs(specs)))
        meta_source = prepare_meta_source(model_frame, v10args, v10_2_output, specs)
        discovery_folds = [int(x) for x in parse_tokens(args.discovery_folds)]
        dev_folds = [int(x) for x in parse_tokens(args.development_folds)]
        conf_folds = [int(x) for x in parse_tokens(args.confirmation_folds)]
        recent_folds = [int(x) for x in parse_tokens(args.recent_folds)]
        if set(discovery_folds) & set(dev_folds + conf_folds + recent_folds):
            raise ValueError("discovery folds overlap evaluation folds")
        discovery_rows = meta_source.loc[meta_source["fold_id"].isin(discovery_folds)].copy()
        evaluation_rows = meta_source.loc[meta_source["fold_id"].isin(dev_folds + conf_folds + recent_folds)].copy()
        if discovery_rows.empty or evaluation_rows.empty:
            raise RuntimeError("missing discovery/evaluation OOF rows")

        # Critical V12 leakage fix: historical score percentile is fitted only on folds 0-2.
        # V10.2 validation-fold-wide base_rank is never used as a model input or policy input.
        transformer = fit_evidence_transformer(
            discovery_rows,
            specs,
            ticker_column="ticker",
            base_score_column="base_score_raw",
            max_slots=int(args.max_ab_features_per_ticker),
        )
        discovery_meta = pd.concat(
            [discovery_rows.reset_index(drop=True), transform_evidence(discovery_rows, transformer).reset_index(drop=True)], axis=1
        )
        eval_meta = pd.concat(
            [evaluation_rows.reset_index(drop=True), transform_evidence(evaluation_rows, transformer).reset_index(drop=True)], axis=1
        )
        # Duplicate base_score_raw/base_prob names are avoided by transform_evidence naming base_prob.
        discovery_meta["ticker"] = discovery_meta["ticker"].astype(str)
        eval_meta["ticker"] = eval_meta["ticker"].astype(str)

        backends = parse_tokens(args.backends)
        quantiles = parse_floats(args.candidate_quantiles)
        alphas = parse_floats(args.alphas)
        config_rows: list[dict[str, Any]] = []
        prediction_cache: dict[str, pd.DataFrame] = {}
        fit_errors: list[dict[str, str]] = []

        for backend in backends:
            for quantile in quantiles:
                try:
                    bundle = fit_gate_bundle(
                        discovery_meta,
                        backend=backend,
                        candidate_quantile=float(quantile),
                        prior_strength=float(args.ticker_prior_strength),
                        negative_weight=float(args.negative_weight),
                        min_ticker_rows=int(args.min_ticker_gate_rows),
                        min_ticker_class=int(args.min_ticker_gate_class),
                        max_slots=int(args.max_ab_features_per_ticker),
                        ticker_column="ticker",
                        target_column="target",
                        seed=int(args.seed),
                    )
                    gp = predict_gate_bundle(bundle, eval_meta, ticker_column="ticker")
                except Exception as exc:
                    fit_errors.append({"backend": backend, "candidate_quantile": str(quantile), "error": f"{type(exc).__name__}: {exc}"})
                    log(f"skip backend={backend} q={quantile}: {exc}")
                    continue
                base_prob = pd.to_numeric(eval_meta["base_score_raw"], errors="coerce").to_numpy(float)
                for alpha in alphas:
                    pred = eval_meta[["row_index", "source_row_id", "ticker", "fold_id", "target", "base_score_raw"]].copy()
                    pred["gate_prob"] = gp["gate_prob"].to_numpy(float)
                    pred["gate_global_prob"] = gp["gate_global_prob"].to_numpy(float)
                    pred["gate_source"] = gp["gate_source"].astype(str).to_numpy()
                    pred["base_hist_rank"] = eval_meta["base_hist_rank"].to_numpy(float)
                    pred["ab_evidence_count"] = eval_meta["ab_evidence_count"].to_numpy(float)
                    pred["candidate_eligible"] = (
                        gp["candidate_eligible"].astype(bool).to_numpy()
                        & (eval_meta["ab_evidence_count"].to_numpy(float) >= int(args.minimum_evidence_count))
                    )
                    pred["v12_score"] = combine_base_and_gate(base_prob, pred["gate_prob"].to_numpy(float), float(alpha))
                    config_id = f"{backend}__q{quantile:.2f}__a{alpha:.2f}"
                    threshold = select_frozen_dev_threshold(
                        pred,
                        dev_folds=dev_folds,
                        minimum_alerts=int(args.minimum_alerts),
                        target_precision=float(args.target_precision),
                    )
                    metrics = evaluate_scores(
                        pred,
                        score_column="v12_score",
                        base_score_column="base_score_raw",
                        target_column="target",
                        minimum_alerts=int(args.minimum_alerts),
                    )
                    dev_metrics = metrics.loc[metrics["fold_id"].isin(dev_folds)].copy()
                    config_summary = {
                        "config_id": config_id,
                        "backend": backend,
                        "candidate_quantile": float(quantile),
                        "alpha": float(alpha),
                        "negative_weight": float(args.negative_weight),
                        "ticker_prior_strength": float(args.ticker_prior_strength),
                        "dev_safe": bool(threshold["safe"]),
                        "frozen_threshold": float(threshold["threshold"]),
                        "threshold_reason": str(threshold["reason"]),
                        "min_dev_precision": float(threshold.get("min_dev_precision", float("nan"))),
                        "mean_dev_precision": float(threshold.get("mean_dev_precision", float("nan"))),
                        "min_dev_pr_auc_lift": float(dev_metrics["pr_auc_lift"].min()) if not dev_metrics.empty else float("nan"),
                        "mean_dev_pr_auc_lift": float(dev_metrics["pr_auc_lift"].mean()) if not dev_metrics.empty else float("nan"),
                        "ticker_models": int(len(bundle.ticker_models)),
                        "eligible_tickers": int(pred.loc[pred["candidate_eligible"], "ticker"].nunique()),
                    }
                    config_rows.append(config_summary)
                    prediction_cache[config_id] = pred
                    log(
                        f"config={config_id} safe={config_summary['dev_safe']} "
                        f"minP={config_summary['min_dev_precision']:.3f} meanLift={config_summary['mean_dev_pr_auc_lift']:.3f}"
                    )

        if not config_rows:
            raise RuntimeError(f"No V12 gate configuration could be fit. Errors: {fit_errors[:5]}")
        config_df = pd.DataFrame(config_rows)
        config_df["selection_key"] = config_df.apply(lambda r: str(config_sort_key(r.to_dict())), axis=1)
        ranked_records = sorted(config_rows, key=config_sort_key, reverse=True)
        champion = ranked_records[0]
        champion_id = str(champion["config_id"])
        champion_pred = prediction_cache[champion_id].copy()
        frozen_threshold = float(champion["frozen_threshold"])
        champion_metrics = evaluate_scores(
            champion_pred,
            score_column="v12_score",
            base_score_column="base_score_raw",
            target_column="target",
            minimum_alerts=int(args.minimum_alerts),
        )
        frozen_policy = apply_frozen_threshold(
            champion_pred,
            frozen_threshold,
            folds=dev_folds + conf_folds + recent_folds,
        )
        frozen_policy["role"] = frozen_policy["fold_id"].map(
            {**{x: "development" for x in dev_folds}, **{x: "confirmation" for x in conf_folds}, **{x: "recent_diagnostic" for x in recent_folds}}
        )
        frozen_policy["gate_pass"] = (
            (frozen_policy["alerts"] >= int(args.minimum_alerts))
            & (frozen_policy["precision"] >= float(args.target_precision))
        )
        baseline_metrics = evaluate_base_matched(champion_pred, int(args.minimum_alerts))
        ticker_metrics_rows: list[dict[str, Any]] = []
        for (ticker, fold_id), grp in champion_pred.groupby(["ticker", "fold_id"], sort=True):
            ticker_metrics_rows.append(
                {
                    "ticker": str(ticker),
                    "fold_id": int(fold_id),
                    "rows": int(len(grp)),
                    "positives": int((grp["target"] == 1).sum()),
                    "base_pr_auc": safe_pr_auc(grp["target"], grp["base_score_raw"]),
                    "v12_pr_auc": safe_pr_auc(grp["target"], grp["v12_score"]),
                    "eligible_rows": int(grp["candidate_eligible"].sum()),
                }
            )
        ticker_metrics = pd.DataFrame(ticker_metrics_rows)
        if not ticker_metrics.empty:
            ticker_metrics["pr_auc_delta"] = ticker_metrics["v12_pr_auc"] - ticker_metrics["base_pr_auc"]

        atomic_csv(output / "v12_config_search.csv", config_df.sort_values("config_id"))
        atomic_csv(output / "v12_champion_predictions.csv", champion_pred)
        atomic_csv(output / "v12_metrics_by_fold.csv", champion_metrics)
        atomic_csv(output / "v12_frozen_policy_by_fold.csv", frozen_policy)
        atomic_csv(output / "v10_2_baseline_matched_by_fold.csv", baseline_metrics)
        atomic_csv(output / "v12_ticker_metrics_by_fold.csv", ticker_metrics)
        if fit_errors:
            atomic_json(output / "V12_BACKEND_WARNINGS.json", fit_errors)

        confirmation_passes = int(frozen_policy.loc[frozen_policy["fold_id"].isin(conf_folds), "gate_pass"].sum())
        recent_passes = int(frozen_policy.loc[frozen_policy["fold_id"].isin(recent_folds), "gate_pass"].sum())
        dev_safe = bool(champion["dev_safe"])
        production_action = "NO_ALERT"
        research_status = "DEV_GATE_FAIL"
        if dev_safe:
            research_status = "DEV_GATE_PASS_CONFIRMATION_REQUIRED"
            if confirmation_passes == len(conf_folds) and recent_passes == len(recent_folds):
                research_status = "DIAGNOSTICALLY_SUPPORTED_NOT_UNTOUCHED"
                # Fold 5-7 were already inspected in prior versions; never upgrade to production here.
                production_action = "NO_ALERT_UNTIL_NEW_FUTURE_DATA"

        leakage_contract = {
            "parent_lineage": "V1_TO_V10_2",
            "direct_parent": "V10.2",
            "v11_v11_1_status": "REFERENCE_ONLY_NOT_USED_AS_MODEL_INPUT",
            "discovery_model_fit_folds": discovery_folds,
            "development_selection_folds": dev_folds,
            "frozen_evaluation_folds": conf_folds + recent_folds,
            "base_rank_rule": "V10.2 fold-wide base_rank is forbidden; V12 uses empirical rank against discovery folds 0-2 only.",
            "separator_rule": "A/B node identities and directions come only from V10.2 probe_discovery_precision_map_v10_2.csv rebuilt on folds 0-2.",
            "threshold_rule": "one score threshold is selected on folds 3-4 jointly, then frozen unchanged for folds 5-7.",
            "holdout_warning": "folds 5-7 are diagnostics because prior project versions already inspected them; final truth requires new future data.",
        }
        atomic_json(output / "LEAKAGE_CONTRACT_V12.json", leakage_contract)

        final = {
            "schema": V12_SCHEMA,
            "status": "V12_COMPLETE",
            "parent": "V10.2",
            "v11_status": "REFERENCE_ONLY",
            "rows_used": int(len(meta_source)),
            "feature_universe": int(len(features)),
            "separator_tickers": int(len(specs)),
            "separator_nodes": int(sum(len(v) for v in specs.values())),
            "champion": champion,
            "frozen_policy": frozen_policy.to_dict(orient="records"),
            "confirmation_gate_passes": confirmation_passes,
            "recent_gate_passes": recent_passes,
            "research_status": research_status,
            "production_action": production_action,
            "target_policy": {"precision": float(args.target_precision), "minimum_alerts": int(args.minimum_alerts)},
            "interpretation": (
                "V12 does not replace the 439-feature V10.2 base model. It uses discovery-only A/B separator evidence "
                "as a second-stage hard-false-positive gate/rescorer on top of V10.2 OOF scores."
            ),
        }
        atomic_json(output / "FINAL_RECOMMENDATION_V12.json", final)
        atomic_json(
            status_path,
            {
                "schema": V12_SCHEMA,
                "status": "SUCCESS",
                "final": final,
                "inputs": {
                    "v10_output": str(Path(args.v10_output).resolve()),
                    "v10_2_output": str(v10_2_output),
                    "dataset": str(Path(args.dataset).resolve()),
                    "folds": str(Path(args.folds).resolve()),
                    "discovery_map_sha256": sha256(discovery_map_path),
                },
            },
        )
        log(json.dumps(final, ensure_ascii=False, indent=2, default=str))
    except BaseException as exc:
        atomic_json(
            status_path,
            {
                "schema": V12_SCHEMA,
                "status": "FAILED",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
