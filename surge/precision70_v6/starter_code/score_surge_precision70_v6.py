from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from surge_model_zoo_common import (
    DATE_COLUMN,
    TARGET_COLUMN,
    TARGET_VALID_COLUMN,
    TICKER_COLUMN,
    FileLock,
    RunStatus,
    atomic_write_csv,
    atomic_write_json,
    join_source_and_target,
    parse_bool_series,
    payload_checksum_is_valid,
    read_table,
    sha256_file,
    table_columns,
)
from surge_model_zoo_deployment import (
    load_registry,
    predict_recipe_record,
    predict_single_saved_model,
)
from surge_precision_common_v6 import (
    CalibratorSpec,
    PrecisionPolicy,
    apply_calibrator,
    apply_meta_feature_spec,
    apply_precision_policy,
    datewise_rank_normalize,
    ensure_unit_interval,
    evaluate_alerts,
    sigmoid,
)


def parse_columns(text: str) -> list[str]:
    return [value.strip() for value in text.split(",") if value.strip()]


def load_json_checked(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload_checksum_is_valid(payload):
        raise ValueError(f"payload checksum mismatch: {path}")
    return payload


def required_features(registry: Mapping[str, Any], *, v4: bool = False) -> list[str]:
    values: list[str] = []
    for record in registry.get("models", []):
        values.extend(str(value) for value in record.get("features", []))
        if v4:
            values.extend(str(value) for value in record.get("direction_features", []))
    return list(dict.fromkeys(values))


def frame_feature_values(frame: pd.DataFrame, features: Sequence[str]) -> dict[str, np.ndarray]:
    return {
        feature: pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=np.float32)
        for feature in features
    }


def _agreement(std: np.ndarray) -> np.ndarray:
    return np.clip(1.0 - np.asarray(std, dtype=np.float64), 0.0, 1.0)


def predict_v4_signals(
    registry: Mapping[str, Any],
    frame: pd.DataFrame,
    registry_root: Path,
) -> pd.DataFrame:
    values = frame_feature_values(frame, required_features(registry, v4=True))
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in registry.get("models", []):
        grouped[str(record["recipe"])].append(predict_recipe_record(record, values, registry_root))
    result = pd.DataFrame(index=frame.index)
    for recipe in [str(value) for value in registry.get("recipes", [])]:
        predictions = grouped.get(recipe, [])
        if not predictions:
            raise KeyError(f"V4 production recipe 누락: {recipe}")
        matrix = np.vstack(predictions).astype(np.float64)
        result[recipe] = np.nanmean(matrix, axis=0)
        result[f"{recipe}__seed_agreement"] = _agreement(np.nanstd(matrix, axis=0))
    return result


def _predict_single_registry_record(
    record: Mapping[str, Any],
    values: Mapping[str, np.ndarray],
    registry_root: Path,
    row_count: int,
) -> np.ndarray:
    model = record.get("model")
    if not isinstance(model, Mapping):
        raise ValueError(f"model record 누락: {record.get('recipe')}")

    features = [str(value) for value in record.get("features", [])]
    if features:
        missing = [feature for feature in features if feature not in values]
        if missing:
            raise KeyError(f"production model 피처 누락: {record.get('recipe')}: {missing[:20]}")
        matrix = np.column_stack([values[feature] for feature in features]).astype(np.float32)
    elif str(model.get("format", "")) == "constant_probability":
        # Synthetic and emergency constant models have no numerical inputs.
        # Preserve the scoring row count without inventing a feature column.
        matrix = np.empty((int(row_count), 0), dtype=np.float32)
    else:
        raise ValueError(f"production model feature 목록이 비어 있음: {record.get('recipe')}")
    return predict_single_saved_model(model, matrix, registry_root)


def predict_rank_seed_signals(
    registry: Mapping[str, Any],
    frame: pd.DataFrame,
    registry_root: Path,
    dates: np.ndarray,
) -> pd.DataFrame:
    values = frame_feature_values(frame, required_features(registry))
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in registry.get("models", []):
        grouped[str(record["recipe"])].append(_predict_single_registry_record(record, values, registry_root, len(frame)))
    result = pd.DataFrame(index=frame.index)
    for recipe in [str(value) for value in registry.get("recipes", [])]:
        predictions = grouped.get(recipe, [])
        if not predictions:
            raise KeyError(f"V5 production recipe 누락: {recipe}")
        ranked = np.vstack([datewise_rank_normalize(prediction, dates) for prediction in predictions])
        result[recipe] = np.nanmean(ranked, axis=0)
        result[f"{recipe}__seed_agreement"] = _agreement(np.nanstd(ranked, axis=0))
    return result


def predict_v6_specialist_signals(
    registry: Mapping[str, Any],
    frame: pd.DataFrame,
    registry_root: Path,
    dates: np.ndarray,
) -> pd.DataFrame:
    values = frame_feature_values(frame, required_features(registry))
    grouped: dict[str, list[tuple[np.ndarray, str]]] = defaultdict(list)
    for record in registry.get("models", []):
        prediction = _predict_single_registry_record(record, values, registry_root, len(frame))
        grouped[str(record["recipe"])].append((prediction, str(record.get("seed_aggregation", "probability_mean"))))
    result = pd.DataFrame(index=frame.index)
    for recipe in [str(value) for value in registry.get("recipes", [])]:
        records = grouped.get(recipe, [])
        if not records:
            raise KeyError(f"V6 production specialist 누락: {recipe}")
        modes = {mode for _, mode in records}
        if len(modes) != 1:
            raise ValueError(f"V6 seed aggregation 불일치: {recipe}: {modes}")
        mode = next(iter(modes))
        predictions = [prediction for prediction, _ in records]
        if mode == "date_rank_mean":
            matrix = np.vstack([datewise_rank_normalize(prediction, dates) for prediction in predictions])
        else:
            matrix = np.vstack(predictions).astype(np.float64)
        result[recipe] = np.nanmean(matrix, axis=0)
        result[f"{recipe}__seed_agreement"] = _agreement(np.nanstd(matrix, axis=0))
    return result


def calibrate_signals(
    raw: pd.DataFrame,
    freeze: Mapping[str, Any],
    dates: np.ndarray,
) -> pd.DataFrame:
    calibrated = pd.DataFrame(index=raw.index)
    calibrators = freeze.get("signal_calibrators", {})
    for signal in [str(value) for value in freeze.get("signals", [])]:
        if signal not in raw.columns:
            raise KeyError(f"frozen signal 누락: {signal}")
        if signal not in calibrators:
            raise KeyError(f"frozen signal calibrator 누락: {signal}")
        base = ensure_unit_interval(raw[signal].to_numpy(dtype=np.float64), dates, "auto")
        calibrated[signal] = apply_calibrator(CalibratorSpec.from_dict(calibrators[signal]), base)
    return calibrated


def _resolve_model_path(root: Path, record: Mapping[str, Any]) -> Path:
    raw = Path(str(record["path"]))
    path = raw if raw.is_absolute() else root / raw
    if not path.exists():
        raise FileNotFoundError(path)
    expected = record.get("sha256")
    if isinstance(expected, str) and sha256_file(path) != expected:
        raise ValueError(f"meta model SHA-256 mismatch: {path}")
    return path


def apply_frozen_method(
    freeze: Mapping[str, Any],
    calibrated: pd.DataFrame,
    dates: np.ndarray,
    model_dir: Path,
) -> np.ndarray:
    spec = dict(freeze["method_spec"])
    method_frame = calibrated.copy()
    date_column = str(spec.get("feature_spec", {}).get("date_column", "date"))
    if date_column not in method_frame.columns:
        method_frame[date_column] = np.asarray(dates, dtype="datetime64[ns]")
    meta = apply_meta_feature_spec(method_frame, spec["feature_spec"])
    kind = str(spec["model_kind"])
    parameters = dict(spec.get("model_parameters", {}))
    if kind == "recipe_mean":
        raw = meta["ensemble_mean"].to_numpy(dtype=np.float64)
    elif kind == "family_mean":
        columns = [str(value) for value in parameters.get("family_columns", [])]
        raw = meta[columns].mean(axis=1).to_numpy(dtype=np.float64)
    elif kind == "safety_product":
        raw = meta["surge_safety_product"].to_numpy(dtype=np.float64)
    elif kind == "conservative_q25":
        raw = meta["conservative_q25_safety"].to_numpy(dtype=np.float64)
    elif kind == "strong_safety":
        raw = meta["strong_safety_product"].to_numpy(dtype=np.float64)
    elif kind == "consensus_lcb":
        z = float(parameters.get("uncertainty_z", 1.0))
        raw = meta["ensemble_mean"].to_numpy(dtype=np.float64) - z * meta["ensemble_std"].to_numpy(dtype=np.float64)
        raw = ensure_unit_interval(raw, dates, "rank")
    elif kind == "logistic":
        record = dict(spec.get("model") or {})
        coef = np.asarray(record.get("coef", []), dtype=np.float64)
        intercept = np.asarray(record.get("intercept", []), dtype=np.float64)
        if coef.ndim != 2 or coef.shape[0] != 1 or coef.shape[1] != meta.shape[1]:
            raise ValueError("logistic meta coefficient shape mismatch")
        raw = sigmoid(meta.to_numpy(dtype=np.float64) @ coef[0] + float(intercept[0]))
    elif kind == "lightgbm":
        import lightgbm as lgb

        record = dict(spec.get("model") or {})
        path = _resolve_model_path(model_dir, record)
        booster = lgb.Booster(model_file=str(path))
        raw = booster.predict(meta.to_numpy(dtype=np.float32), num_iteration=int(record.get("iterations", parameters.get("iterations", 0)) or 0))
    elif kind == "xgboost":
        import xgboost as xgb

        record = dict(spec.get("model") or {})
        path = _resolve_model_path(model_dir, record)
        booster = xgb.Booster()
        booster.load_model(path)
        matrix = xgb.DMatrix(meta.to_numpy(dtype=np.float32))
        iterations = int(record.get("iterations", parameters.get("iterations", 0)) or 0)
        raw = booster.predict(matrix, iteration_range=(0, iterations)) if iterations > 0 else booster.predict(matrix)
    else:
        raise ValueError(f"지원하지 않는 frozen method: {kind}")
    return apply_calibrator(CalibratorSpec.from_dict(spec["calibrator"]), np.asarray(raw, dtype=np.float64))


def population_stability_index(reference: Mapping[str, Any], current: np.ndarray) -> float:
    inner = np.asarray(reference.get("psi_inner_edges", []), dtype=np.float64)
    expected = np.asarray(reference.get("psi_reference_proportions", []), dtype=np.float64)
    values = np.asarray(current, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values) or not len(expected):
        return float("nan")
    edges = np.concatenate(([-np.inf], inner, [np.inf]))
    actual = np.histogram(values, bins=edges)[0].astype(np.float64)
    actual /= max(1.0, float(actual.sum()))
    if len(actual) != len(expected):
        return float("nan")
    eps = 1e-6
    return float(np.sum((actual - expected) * np.log((actual + eps) / (expected + eps))))


def evaluate_target(
    source: pd.DataFrame,
    target_path: Path,
    probability: np.ndarray,
    alert: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    side_columns = table_columns(target_path)
    requested = ["source_row_id", args.date_column, args.ticker_column, args.target_column, args.target_valid_column]
    sidecar = read_table(target_path, [value for value in requested if value in side_columns])
    source_keys = [value for value in ["source_row_id", args.date_column, args.ticker_column] if value in source.columns]
    joined = join_source_and_target(
        source[source_keys],
        sidecar,
        target_column=args.target_column,
        target_valid_column=args.target_valid_column,
        date_column=args.date_column,
        ticker_column=args.ticker_column,
    )
    valid = parse_bool_series(joined[args.target_valid_column]) & joined[args.target_column].isin([0, 1])
    mask = valid.to_numpy(dtype=bool)
    y = pd.to_numeric(joined.loc[valid, args.target_column], errors="coerce").to_numpy(dtype=np.uint8)
    return evaluate_alerts(
        y,
        probability[mask],
        alert[mask],
        source.loc[mask, args.date_column].to_numpy(dtype="datetime64[ns]"),
        source.loc[mask, args.ticker_column].to_numpy(dtype=object),
        confidence_level=args.confidence_level,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score frozen CrashWatch Surge Precision 70 V6 ensemble")
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/surge_precision70_v6"))
    parser.add_argument("--v4-model-dir", type=Path, default=Path("outputs/surge_model_zoo_v4"))
    parser.add_argument("--v5-model-dir", type=Path, default=Path("outputs/surge_alert_budget_v5"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target-sidecar", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--date-column", default=DATE_COLUMN)
    parser.add_argument("--ticker-column", default=TICKER_COLUMN)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--target-valid-column", default=TARGET_VALID_COLUMN)
    parser.add_argument("--metadata-columns", default="industry_name,market,bucket")
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--allow-gate-failed", action="store_true")
    parser.add_argument("--allow-score-drift", action="store_true")
    parser.add_argument(
        "--controller-state",
        type=Path,
        help="update_surge_precision_controller_v6.py가 만든 ACTIVE delayed-label controller state",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    v4_dir = args.v4_model_dir.resolve()
    v5_dir = args.v5_model_dir.resolve()
    dataset = args.dataset.resolve()
    target = args.target_sidecar.resolve() if args.target_sidecar else None
    output = args.output.resolve() if args.output else model_dir / "scored" / dataset.stem
    output.mkdir(parents=True, exist_ok=True)
    status = RunStatus(output)
    status.payload["schema"] = "crashwatch_surge_precision70_scorer_v6"
    atomic_write_json(status.path, status.payload)

    with FileLock(output / ".score_surge_precision70_v6.lock"):
        try:
            freeze = load_json_checked(model_dir / "PRECISION_FREEZE_V6.json")
            recommendation = load_json_checked(model_dir / "FINAL_RECOMMENDATION_V6.json")
            gate_status = str(recommendation.get("status", "UNKNOWN"))
            if gate_status != "READY_FOR_NEW_FUTURE_HOLDOUT" and not args.allow_gate_failed:
                raise RuntimeError("V6 Precision Gate 실패 후보입니다. 진단 점수화에는 --allow-gate-failed가 필요합니다.")
            v4_registry = load_registry(v4_dir / "PRODUCTION_MODEL_REGISTRY.json")
            if not payload_checksum_is_valid(v4_registry):
                raise ValueError("V4 production registry checksum mismatch")
            v5_registry = load_json_checked(v5_dir / "V5_RANKER_PRODUCTION_REGISTRY.json")
            v6_registry = load_json_checked(model_dir / "V6_SPECIALIST_PRODUCTION_REGISTRY.json")

            columns = table_columns(dataset)
            features = list(dict.fromkeys(
                required_features(v4_registry, v4=True)
                + required_features(v5_registry)
                + required_features(v6_registry)
            ))
            missing = [feature for feature in features if feature not in columns]
            if missing:
                raise KeyError(f"점수화 데이터 피처 누락 {len(missing)}개: {missing[:40]}")
            metadata = parse_columns(args.metadata_columns)
            requested = ["source_row_id", args.date_column, args.ticker_column, *metadata, *features]
            requested = [value for value in dict.fromkeys(requested) if value in columns]
            frame = read_table(dataset, requested)
            if "source_row_id" not in frame.columns:
                frame.insert(0, "source_row_id", np.arange(len(frame), dtype=np.int64))
            frame[args.date_column] = pd.to_datetime(frame[args.date_column], errors="coerce")
            if frame[args.date_column].isna().any():
                raise ValueError("점수화 데이터 날짜 오류")
            dates = frame[args.date_column].to_numpy(dtype="datetime64[ns]")

            v4 = predict_v4_signals(v4_registry, frame, v4_dir)
            v5 = predict_rank_seed_signals(v5_registry, frame, v5_dir, dates)
            v6 = predict_v6_specialist_signals(v6_registry, frame, model_dir, dates)
            raw = pd.concat([v4, v5, v6], axis=1)
            raw = raw.loc[:, ~raw.columns.duplicated(keep="last")]
            calibrated = calibrate_signals(raw, freeze, dates)
            calibrated["date"] = frame[args.date_column].to_numpy()
            feature_spec = freeze.get("method_spec", {}).get("feature_spec", {})
            for scope_column in feature_spec.get("scope_categories", {}).keys():
                if scope_column not in frame.columns:
                    raise KeyError(f"meta scope column 누락: {scope_column}")
                calibrated[str(scope_column)] = frame[str(scope_column)].to_numpy()
            probability = apply_frozen_method(freeze, calibrated, dates, model_dir)
            policy = PrecisionPolicy.from_dict(freeze["precision_policy"])
            policy_source = "frozen_selection_policy"
            controller_state = None
            if args.controller_state is not None:
                controller_path = args.controller_state.resolve()
                controller_state = load_json_checked(controller_path)
                if str(controller_state.get("status")) != "ACTIVE":
                    raise RuntimeError(
                        f"precision controller가 ACTIVE가 아닙니다: {controller_state.get('status')}"
                    )
                expected_freeze_sha = controller_state.get(
                    "base_freeze_sha256", controller_state.get("freeze_sha256")
                )
                actual_freeze_sha = sha256_file(model_dir / "PRECISION_FREEZE_V6.json")
                if isinstance(expected_freeze_sha, str) and expected_freeze_sha != actual_freeze_sha:
                    raise ValueError("controller와 현재 PRECISION_FREEZE_V6.json SHA-256이 다릅니다")
                policy_payload = controller_state.get(
                    "current_policy", controller_state.get("active_policy")
                )
                if not isinstance(policy_payload, Mapping):
                    raise ValueError("controller current_policy/active_policy 누락")
                policy = PrecisionPolicy.from_dict(policy_payload)
                policy_source = "matured_online_controller"
            scopes = (
                frame[policy.scope_column].astype("string").fillna("__MISSING__").to_numpy(dtype=object)
                if policy.scope_column and policy.scope_column in frame.columns
                else None
            )
            alert = apply_precision_policy(policy, probability, scopes)

            psi = population_stability_index(freeze.get("score_reference", {}), probability)
            warning_threshold = float(freeze.get("score_reference", {}).get("psi_warning_threshold", 0.25))
            drift_status = "WARN_SCORE_DRIFT" if np.isfinite(psi) and psi > warning_threshold else "PASS"
            if drift_status != "PASS" and not args.allow_score_drift:
                raise RuntimeError(
                    f"Frozen score distribution drift detected: PSI={psi:.4f} > {warning_threshold:.4f}. "
                    "진단 점수화에는 --allow-score-drift가 필요합니다."
                )
            result = frame[[value for value in ["source_row_id", args.date_column, args.ticker_column, *metadata] if value in frame.columns]].copy()
            result["surge_probability_3d5_v6"] = probability
            result["surge_alert_v6"] = alert.astype(np.uint8)
            result["precision_policy_kind"] = policy.kind
            result["global_threshold"] = policy.threshold
            result["gate_status"] = gate_status
            result["policy_source"] = policy_source
            result["controller_matured_through"] = (
                controller_state.get("matured_through") if controller_state else None
            )
            result["score_drift_status"] = drift_status
            result["controller_state_used"] = (
                str(args.controller_state.resolve()) if args.controller_state is not None else None
            )
            atomic_write_csv(output / "surge_precision70_scores.csv", result)

            evaluation = None
            if target is not None:
                evaluation = evaluate_target(frame, target, probability, alert, args)
                atomic_write_json(output / "PRECISION_EVALUATION.json", evaluation)
            audit = {
                "schema": "crashwatch_surge_precision70_scoring_audit_v6",
                "rows": int(len(frame)),
                "alerts": int(alert.sum()),
                "alert_rate": float(alert.mean()) if len(alert) else float("nan"),
                "score_mean": float(np.mean(probability)) if len(probability) else float("nan"),
                "score_std": float(np.std(probability)) if len(probability) else float("nan"),
                "population_stability_index": psi,
                "psi_warning_threshold": warning_threshold,
                "score_drift_status": drift_status,
                "gate_status": gate_status,
                "policy_source": policy_source,
                "controller_matured_through": (
                    controller_state.get("matured_through") if controller_state else None
                ),
                "no_alert_count_limit": True,
                "controller_state_used": (
                    str(args.controller_state.resolve()) if args.controller_state is not None else None
                ),
                "controller_status": controller_state.get("status") if controller_state else None,
                "active_policy": policy.to_dict(),
                "evaluation": evaluation,
            }
            atomic_write_json(output / "SCORING_AUDIT_V6.json", audit)
            status.success(alerts=int(alert.sum()), gate_status=gate_status, score_drift_status=drift_status)
            print(json.dumps(audit, ensure_ascii=False, indent=2))
        except Exception as exc:
            status.failure(exc)
            raise


if __name__ == "__main__":
    main()
