from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from run_surge_model_zoo_v4 import apply_ensemble_spec
from surge_model_zoo_common import (
    DATE_COLUMN,
    TARGET_COLUMN,
    TARGET_VALID_COLUMN,
    TICKER_COLUMN,
    FileLock,
    RunStatus,
    ThresholdPolicy,
    apply_calibrator,
    apply_threshold_policy,
    atomic_write_csv,
    atomic_write_json,
    datewise_rank_normalize,
    evaluate_prediction_metrics,
    join_source_and_target,
    parse_bool_series,
    payload_checksum_is_valid,
    read_table,
    sha256_file,
    table_columns,
    utc_now,
)
from surge_model_zoo_deployment import load_registry, predict_recipe_record


def parse_columns(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def resolve_path(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else (root / path).resolve()


def load_freeze(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    required = {"spec", "calibrator", "threshold_policy", "recipes"}
    missing = required - set(payload)
    if missing:
        raise ValueError(f"ENSEMBLE_FREEZE 필드 누락: {sorted(missing)}")
    if not payload_checksum_is_valid(payload):
        raise ValueError("ENSEMBLE_FREEZE payload checksum mismatch")
    return payload


def required_features(registry: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for record in registry.get("models", []):
        values.extend(str(feature) for feature in record.get("features", []))
        values.extend(str(feature) for feature in record.get("direction_features", []))
    return list(dict.fromkeys(values))


def model_predictions(
    registry: dict[str, Any],
    frame: pd.DataFrame,
    registry_root: Path,
) -> pd.DataFrame:
    feature_values = {
        feature: pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float32)
        for feature in required_features(registry)
    }
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    expected_recipes = [str(value) for value in registry.get("recipes", [])]
    for record in registry.get("models", []):
        recipe = str(record["recipe"])
        prediction = predict_recipe_record(record, feature_values, registry_root)
        if len(prediction) != len(frame):
            raise ValueError(f"production prediction 길이 불일치: {recipe}")
        grouped[recipe].append(np.asarray(prediction, dtype=np.float64))
    missing = [recipe for recipe in expected_recipes if recipe not in grouped]
    if missing:
        raise ValueError(f"production model recipe 누락: {missing}")
    result = pd.DataFrame(index=frame.index)
    for recipe in expected_recipes:
        result[recipe] = np.mean(np.vstack(grouped[recipe]), axis=0)
    return result


def evaluate_with_target(
    source: pd.DataFrame,
    target_path: Path,
    score: np.ndarray,
    alert: np.ndarray,
    dates: np.ndarray,
    target_column: str,
    target_valid_column: str,
    date_column: str,
    ticker_column: str,
    target_recall: float,
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
    joined = join_source_and_target(
        source[[value for value in ["source_row_id", date_column, ticker_column] if value in source.columns]],
        sidecar,
        target_column=target_column,
        target_valid_column=target_valid_column,
        date_column=date_column,
        ticker_column=ticker_column,
    )
    valid = parse_bool_series(joined[target_valid_column]) & joined[target_column].isin([0, 1])
    y = pd.to_numeric(joined.loc[valid, target_column], errors="coerce").to_numpy(dtype=np.uint8)
    ranking = evaluate_prediction_metrics(y, score[valid.to_numpy()], dates[valid.to_numpy()], target_recall)
    # Alert is already frozen-policy output; compute directly without inventing a threshold.
    selected = alert[valid.to_numpy()].astype(bool)
    positives = int(np.sum(y == 1))
    selected_count = int(selected.sum())
    true_positive = int(np.sum((y == 1) & selected))
    positive_rate = float(np.mean(y == 1)) if len(y) else float("nan")
    precision = float(true_positive / selected_count) if selected_count else float("nan")
    recall = float(true_positive / positives) if positives else float("nan")
    operating = {
        "rows": int(len(y)),
        "positives": positives,
        "positive_rate": positive_rate,
        "alerts": selected_count,
        "true_positives": true_positive,
        "precision": precision,
        "recall": recall,
        "lift": float(precision / positive_rate) if positive_rate > 0 and np.isfinite(precision) else float("nan"),
        "alert_rate": float(selected_count / len(y)) if len(y) else float("nan"),
    }
    return {"ranking": ranking, "frozen_policy": operating}


def main() -> None:
    parser = argparse.ArgumentParser(description="CrashWatch Surge V4 frozen production ensemble scorer")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/surge_model_zoo_v4"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--date-column", default=DATE_COLUMN)
    parser.add_argument("--ticker-column", default=TICKER_COLUMN)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--target-valid-column", default=TARGET_VALID_COLUMN)
    parser.add_argument("--metadata-columns", default="industry_name,market,bucket")
    parser.add_argument("--target-recall", type=float, default=0.70)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    dataset_path = args.dataset.resolve()
    output = args.output.resolve() if args.output else model_dir / "scored" / dataset_path.stem
    target_path = args.target_sidecar.resolve() if args.target_sidecar else None
    output.mkdir(parents=True, exist_ok=True)
    status = RunStatus(output)
    with FileLock(output / ".score_surge_frozen_v4.lock"):
        try:
            registry_path = model_dir / "PRODUCTION_MODEL_REGISTRY.json"
            freeze_path = model_dir / "ENSEMBLE_FREEZE.json"
            registry = load_registry(registry_path)
            if not payload_checksum_is_valid(registry):
                raise ValueError("PRODUCTION_MODEL_REGISTRY payload checksum mismatch")
            freeze = load_freeze(freeze_path)
            registry_recipes = [str(value) for value in registry.get("recipes", [])]
            freeze_recipes = [str(value) for value in freeze.get("recipes", [])]
            if registry_recipes != freeze_recipes:
                raise ValueError("production registry와 ensemble freeze recipe 순서가 다릅니다")

            columns = table_columns(dataset_path)
            needed_features = required_features(registry)
            missing = [feature for feature in needed_features if feature not in columns]
            if missing:
                raise KeyError(f"점수화 dataset 피처 누락 {len(missing)}개: {missing[:20]}")
            metadata_columns = parse_columns(args.metadata_columns)
            requested = [
                "source_row_id",
                args.date_column,
                args.ticker_column,
                *metadata_columns,
                *needed_features,
            ]
            requested = [value for value in dict.fromkeys(requested) if value in columns]
            frame = read_table(dataset_path, requested)
            if "source_row_id" not in frame.columns:
                frame.insert(0, "source_row_id", np.arange(len(frame), dtype=np.int64))
            frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
            if frame[args.date_column].isna().any():
                raise ValueError("점수화 dataset에 잘못된 날짜가 있습니다")
            if args.ticker_column not in frame.columns:
                raise KeyError(args.ticker_column)

            recipe_frame = model_predictions(registry, frame, model_dir)
            ensemble_input = pd.concat(
                [
                    frame[[args.date_column]].rename(columns={args.date_column: "date"}).reset_index(drop=True),
                    recipe_frame.reset_index(drop=True),
                ],
                axis=1,
            )
            raw = apply_ensemble_spec(freeze["spec"], ensemble_input)
            calibrated = apply_calibrator(freeze["calibrator"], raw)
            policy = ThresholdPolicy(**freeze["threshold_policy"])
            dates = frame[args.date_column].to_numpy(dtype="datetime64[ns]")
            alert = apply_threshold_policy(policy, calibrated, dates)
            result_columns = [
                value
                for value in ["source_row_id", args.date_column, args.ticker_column, *metadata_columns]
                if value in frame.columns
            ]
            result = frame[result_columns].copy()
            for recipe in freeze_recipes:
                result[f"raw_{recipe}"] = recipe_frame[recipe].to_numpy(dtype=np.float32)
            result["surge_probability"] = calibrated.astype(np.float64)
            result["surge_daily_rank_pct"] = datewise_rank_normalize(calibrated, dates)
            result["surge_alert"] = alert.astype(np.uint8)
            result["threshold_policy_kind"] = policy.kind
            atomic_write_csv(output / "surge_scores.csv", result)

            training_cutoff = pd.Timestamp(registry["training_cutoff"])
            date_values = frame[args.date_column]
            audit: dict[str, Any] = {
                "schema": "crashwatch_surge_scoring_audit_v4",
                "created_at": utc_now(),
                "dataset_path": str(dataset_path),
                "dataset_sha256": sha256_file(dataset_path),
                "rows": int(len(frame)),
                "date_min": str(date_values.min().date()),
                "date_max": str(date_values.max().date()),
                "training_cutoff": str(training_cutoff.date()),
                "strict_future_rows": int((date_values > training_cutoff).sum()),
                "in_sample_or_historical_rows": int((date_values <= training_cutoff).sum()),
                "alert_rows": int(alert.sum()),
                "alert_rate": float(np.mean(alert)) if len(alert) else float("nan"),
                "threshold_policy": policy.to_dict(),
                "model_registry_sha256": sha256_file(registry_path),
                "ensemble_freeze_sha256": sha256_file(freeze_path),
            }
            if target_path is not None:
                evaluation = evaluate_with_target(
                    frame,
                    target_path,
                    calibrated,
                    alert,
                    dates,
                    args.target_column,
                    args.target_valid_column,
                    args.date_column,
                    args.ticker_column,
                    args.target_recall,
                )
                audit["evaluation"] = evaluation
                atomic_write_json(output / "SCORING_EVALUATION.json", evaluation)
            atomic_write_json(output / "SCORING_AUDIT.json", audit)
            status.success(rows=len(frame), alerts=int(alert.sum()), output=str(output / "surge_scores.csv"))
            print("=" * 72)
            print("CrashWatch Surge Frozen Ensemble Scoring V4")
            print(f"Rows             : {len(frame):,}")
            print(f"Alerts           : {int(alert.sum()):,}")
            print(f"Alert rate       : {float(np.mean(alert)):.2%}")
            print(f"Training cutoff  : {training_cutoff.date()}")
            print(f"Strict future    : {int((date_values > training_cutoff).sum()):,}")
            print(f"Output           : {output / 'surge_scores.csv'}")
            print("=" * 72)
        except BaseException as exc:
            status.failure(exc)
            raise


if __name__ == "__main__":
    main()
