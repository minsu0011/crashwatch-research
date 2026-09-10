from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from surge_alert_budget_common_v5 import (
    DailyBudgetPolicy,
    apply_allocation_biases,
    apply_rank_ensemble_spec,
    daily_budget_selection,
    load_meta_model,
    predict_hard_negative_meta_lgb,
)
from surge_model_zoo_common import (
    DATE_COLUMN,
    TARGET_COLUMN,
    TARGET_VALID_COLUMN,
    TICKER_COLUMN,
    FileLock,
    RunStatus,
    _selected_metrics,
    atomic_write_csv,
    atomic_write_json,
    datewise_rank_normalize,
    join_source_and_target,
    parse_bool_series,
    payload_checksum_is_valid,
    read_table,
    safe_binary_metrics,
    table_columns,
    utc_now,
)
from surge_model_zoo_deployment import (
    load_registry,
    predict_recipe_record,
    predict_single_saved_model,
)


def parse_columns(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def load_json_checked(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload_checksum_is_valid(payload):
        raise ValueError(f"payload checksum mismatch: {path}")
    return payload


def v4_required_features(registry: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for record in registry.get("models", []):
        values.extend(str(value) for value in record.get("features", []))
        values.extend(str(value) for value in record.get("direction_features", []))
    return list(dict.fromkeys(values))


def v5_required_features(registry: Mapping[str, Any]) -> list[str]:
    values: list[str] = []
    for record in registry.get("models", []):
        values.extend(str(value) for value in record.get("features", []))
    return list(dict.fromkeys(values))


def frame_feature_values(frame: pd.DataFrame, features: list[str]) -> dict[str, np.ndarray]:
    return {
        feature: pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float32)
        for feature in features
    }


def predict_v4_signals(
    registry: Mapping[str, Any],
    frame: pd.DataFrame,
    registry_root: Path,
) -> pd.DataFrame:
    values = frame_feature_values(frame, v4_required_features(registry))
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in registry.get("models", []):
        recipe = str(record["recipe"])
        grouped[recipe].append(predict_recipe_record(record, values, registry_root))
    expected = [str(value) for value in registry.get("recipes", [])]
    missing = [recipe for recipe in expected if recipe not in grouped]
    if missing:
        raise KeyError(f"V4 production recipe 누락: {missing}")
    result = pd.DataFrame(index=frame.index)
    for recipe in expected:
        result[recipe] = np.mean(np.vstack(grouped[recipe]), axis=0)
    return result


def predict_v5_rank_signals(
    registry: Mapping[str, Any],
    frame: pd.DataFrame,
    registry_root: Path,
) -> pd.DataFrame:
    values = frame_feature_values(frame, v5_required_features(registry))
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in registry.get("models", []):
        recipe = str(record["recipe"])
        features = [str(value) for value in record["features"]]
        matrix = np.column_stack([values[feature] for feature in features]).astype(np.float32)
        grouped[recipe].append(predict_single_saved_model(record["model"], matrix, registry_root))
    expected = [str(value) for value in registry.get("recipes", [])]
    missing = [recipe for recipe in expected if recipe not in grouped]
    if missing:
        raise KeyError(f"V5 ranker production recipe 누락: {missing}")
    result = pd.DataFrame(index=frame.index)
    for recipe in expected:
        result[recipe] = np.mean(np.vstack(grouped[recipe]), axis=0)
    return result


def apply_frozen_method(
    freeze: Mapping[str, Any],
    signal_frame: pd.DataFrame,
    model_dir: Path,
) -> np.ndarray:
    spec = dict(freeze["method_spec"])
    method = str(spec["method"])
    if method != "hard_negative_meta_lgb":
        return apply_rank_ensemble_spec(spec, signal_frame)
    model_record = spec.get("model")
    if not isinstance(model_record, dict):
        raise ValueError("hard-negative meta model record 누락")
    raw_path = Path(str(model_record["path"]))
    path = raw_path if raw_path.is_absolute() else model_dir / raw_path
    if not path.exists():
        raise FileNotFoundError(path)
    from surge_model_zoo_common import sha256_file

    if model_record.get("sha256") != sha256_file(path):
        raise ValueError("hard-negative meta model SHA-256 mismatch")
    booster = load_meta_model(path)
    return predict_hard_negative_meta_lgb(booster, spec, signal_frame)


def evaluate_target(
    source: pd.DataFrame,
    target_path: Path,
    score: np.ndarray,
    alert: np.ndarray,
    date_column: str,
    ticker_column: str,
    target_column: str,
    target_valid_column: str,
) -> dict[str, Any]:
    side_columns = table_columns(target_path)
    requested = [
        "source_row_id",
        date_column,
        ticker_column,
        target_column,
        target_valid_column,
    ]
    sidecar = read_table(target_path, [value for value in requested if value in side_columns])
    source_keys = [value for value in ["source_row_id", date_column, ticker_column] if value in source.columns]
    joined = join_source_and_target(
        source[source_keys],
        sidecar,
        target_column=target_column,
        target_valid_column=target_valid_column,
        date_column=date_column,
        ticker_column=ticker_column,
    )
    valid = parse_bool_series(joined[target_valid_column]) & joined[target_column].isin([0, 1])
    mask = valid.to_numpy(dtype=bool)
    y = pd.to_numeric(joined.loc[valid, target_column], errors="coerce").to_numpy(dtype=np.uint8)
    monotonic = 1.0 / (1.0 + np.exp(-np.clip(score[mask], -30.0, 30.0)))
    ranking_all = safe_binary_metrics(y, monotonic)
    ranking = {
        "rows": ranking_all.get("rows"),
        "positives": ranking_all.get("positives"),
        "negatives": ranking_all.get("negatives"),
        "positive_rate": ranking_all.get("positive_rate"),
        "pr_auc": ranking_all.get("pr_auc"),
        "pr_auc_lift": ranking_all.get("pr_auc_lift"),
        "roc_auc": ranking_all.get("roc_auc"),
        "mean_score": float(np.nanmean(score[mask])) if np.isfinite(score[mask]).any() else float("nan"),
        "score_is_calibrated_probability": False,
    }
    operating = _selected_metrics(y, alert[mask].astype(bool))
    return {"ranking": ranking, "budget_policy": operating}


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch Surge V5 frozen daily-budget scorer")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/surge_alert_budget_v5"))
    parser.add_argument("--v4-model-dir", type=Path, default=Path("outputs/surge_model_zoo_v4"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--date-column", default=DATE_COLUMN)
    parser.add_argument("--ticker-column", default=TICKER_COLUMN)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--target-valid-column", default=TARGET_VALID_COLUMN)
    parser.add_argument("--metadata-columns", default="industry_name,market,bucket")
    parser.add_argument("--allow-gate-failed", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    v4_dir = args.v4_model_dir.resolve()
    dataset_path = args.dataset.resolve()
    output = args.output.resolve() if args.output else model_dir / "scored" / dataset_path.stem
    target_path = args.target_sidecar.resolve() if args.target_sidecar else None
    output.mkdir(parents=True, exist_ok=True)
    status = RunStatus(output)
    status.payload["schema"] = "crashwatch_surge_alert_budget_scorer_v5"
    atomic_write_json(status.path, status.payload)

    with FileLock(output / ".score_surge_alert_budget_v5.lock"):
        try:
            freeze = load_json_checked(model_dir / "ALERT_BUDGET_FREEZE_V5.json")
            recommendation = load_json_checked(model_dir / "FINAL_RECOMMENDATION_V5.json")
            gate_status = str(recommendation.get("status", "UNKNOWN"))
            if gate_status == "STOP_BUDGET_RECALL_GATE" and not args.allow_gate_failed:
                raise RuntimeError(
                    "V5 개발 Gate가 실패한 후보입니다. 진단 점수화만 하려면 --allow-gate-failed를 명시하십시오."
                )
            v4_registry = load_registry(v4_dir / "PRODUCTION_MODEL_REGISTRY.json")
            if not payload_checksum_is_valid(v4_registry):
                raise ValueError("V4 production registry checksum mismatch")
            v5_registry = load_json_checked(model_dir / "V5_RANKER_PRODUCTION_REGISTRY.json")
            if v5_registry.get("schema") != "crashwatch_surge_ranker_production_registry_v5":
                raise ValueError("V5 ranker registry schema mismatch")

            columns = table_columns(dataset_path)
            needed_features = list(
                dict.fromkeys(v4_required_features(v4_registry) + v5_required_features(v5_registry))
            )
            missing = [feature for feature in needed_features if feature not in columns]
            if missing:
                raise KeyError(f"점수화 dataset 피처 누락 {len(missing)}개: {missing[:30]}")
            metadata = parse_columns(args.metadata_columns)
            requested = [
                "source_row_id",
                args.date_column,
                args.ticker_column,
                *metadata,
                *needed_features,
            ]
            requested = [value for value in dict.fromkeys(requested) if value in columns]
            frame = read_table(dataset_path, requested)
            if "source_row_id" not in frame.columns:
                frame.insert(0, "source_row_id", np.arange(len(frame), dtype=np.int64))
            frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
            if frame[args.date_column].isna().any():
                raise ValueError("점수화 dataset 날짜 오류")

            v4_signals = predict_v4_signals(v4_registry, frame, v4_dir)
            v5_signals = predict_v5_rank_signals(v5_registry, frame, model_dir)
            signals = pd.concat([v4_signals, v5_signals], axis=1)
            expected_signals = [str(value) for value in freeze["signals"]]
            missing_signals = [signal for signal in expected_signals if signal not in signals.columns]
            if missing_signals:
                raise KeyError(f"freeze signal 누락: {missing_signals}")
            signal_frame = pd.concat(
                [
                    frame[[args.date_column, args.ticker_column] + [column for column in metadata if column in frame.columns]]
                    .rename(columns={args.date_column: "date", args.ticker_column: "ticker"})
                    .reset_index(drop=True),
                    signals[expected_signals].reset_index(drop=True),
                ],
                axis=1,
            )
            raw_score = apply_frozen_method(freeze, signal_frame, model_dir)
            policy = DailyBudgetPolicy.from_dict(freeze["daily_budget_policy"])
            groups = (
                signal_frame[policy.allocation_column].to_numpy(dtype=object)
                if policy.allocation_column and policy.allocation_column in signal_frame.columns
                else None
            )
            score = apply_allocation_biases(raw_score, groups, policy.allocation_biases)
            alert = daily_budget_selection(
                score,
                signal_frame["date"].to_numpy(dtype="datetime64[ns]"),
                policy.daily_fraction,
                max_alerts_per_day=policy.max_alerts_per_day,
            )
            daily_rank = datewise_rank_normalize(
                score,
                signal_frame["date"].to_numpy(dtype="datetime64[ns]"),
            )
            result_columns = [
                "source_row_id",
                args.date_column,
                args.ticker_column,
                *[column for column in metadata if column in frame.columns],
            ]
            result = frame[result_columns].copy()
            result["surge_raw_ensemble_score"] = raw_score
            result["surge_budget_score"] = score
            result["surge_daily_percentile_rank"] = daily_rank
            result["surge_alert"] = alert.astype(np.uint8)
            result["frozen_daily_fraction"] = float(policy.daily_fraction)
            result["frozen_max_alerts_per_day"] = (
                int(policy.max_alerts_per_day) if policy.max_alerts_per_day is not None else np.nan
            )
            atomic_write_csv(output / "surge_alert_budget_scores.csv", result)

            evaluation = None
            if target_path is not None:
                evaluation = evaluate_target(
                    frame,
                    target_path,
                    score,
                    alert,
                    args.date_column,
                    args.ticker_column,
                    args.target_column,
                    args.target_valid_column,
                )
                atomic_write_json(
                    output / "SCORING_EVALUATION.json",
                    {
                        "schema": "crashwatch_surge_alert_budget_scoring_evaluation_v5",
                        "gate_status": gate_status,
                        "policy": policy.to_dict(),
                        **evaluation,
                    },
                )
            status.success(
                rows=len(result),
                alerts=int(alert.sum()),
                alert_rate=float(np.mean(alert)) if len(alert) else float("nan"),
                gate_status=gate_status,
                completed_at=utc_now(),
            )
            print(f"Scored rows          : {len(result):,}")
            print(f"Alerts               : {int(alert.sum()):,}")
            print(f"Alert rate           : {float(np.mean(alert)):.2%}")
            print(f"Daily fraction       : {policy.daily_fraction:.1%}")
            print(f"Output               : {output}")
        except Exception as exc:
            import traceback

            status.failure(exc)
            raise


if __name__ == "__main__":
    main()
