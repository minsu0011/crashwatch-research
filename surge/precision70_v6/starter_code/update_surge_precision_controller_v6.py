from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from surge_model_zoo_common import (
    DATE_COLUMN,
    TARGET_COLUMN,
    TARGET_VALID_COLUMN,
    TICKER_COLUMN,
    atomic_write_json,
    join_source_and_target,
    parse_bool_series,
    payload_checksum_is_valid,
    read_table,
    sha256_file,
    table_columns,
    utc_now,
    with_payload_checksum,
)
from surge_precision_common_v6 import (
    PrecisionPolicy,
    apply_precision_policy,
    evaluate_alerts,
    select_global_precision_policy,
    select_scope_precision_policy,
)


SCORE_COLUMN = "surge_probability_3d5_v6"


def load_json_checked(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not payload_checksum_is_valid(payload):
        raise ValueError(f"payload checksum mismatch: {path}")
    return payload


def mature_cutoff(dates: pd.Series, delay_trading_days: int) -> pd.Timestamp:
    unique = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce").dropna().unique()).sort_values()
    delay = max(0, int(delay_trading_days))
    if len(unique) <= delay:
        raise ValueError("라벨 지연 이후 사용할 성숙 거래일이 없습니다")
    return pd.Timestamp(unique[-delay - 1])


def _safe_abstain_policy(base: PrecisionPolicy, source: str) -> PrecisionPolicy:
    return PrecisionPolicy(
        kind="global_threshold",
        target_precision=base.target_precision,
        threshold=float("inf"),
        minimum_alerts=base.minimum_alerts,
        minimum_alert_days=base.minimum_alert_days,
        minimum_precision_lcb=base.minimum_precision_lcb,
        confidence_level=base.confidence_level,
        required_fold_pass_rate=base.required_fold_pass_rate,
        achieved_fold_pass_rate=0.0,
        achieved_alerts=0,
        gate_pass=False,
        source=source,
    )


def _conservative_threshold(base: float | None, candidate: float | None, allow_decrease: bool) -> float:
    base_value = float(base) if base is not None else float("-inf")
    candidate_value = float(candidate) if candidate is not None else float("inf")
    return candidate_value if allow_decrease else max(base_value, candidate_value)


def clamp_policy_to_base(
    base: PrecisionPolicy,
    candidate: PrecisionPolicy,
    allow_decrease: bool,
) -> PrecisionPolicy:
    """Do not lower a frozen threshold unless the operator explicitly allows it."""

    if candidate.kind == "global_threshold":
        return dataclasses.replace(
            candidate,
            threshold=_conservative_threshold(base.threshold, candidate.threshold, allow_decrease),
        )
    if candidate.kind != "scope_threshold":
        raise ValueError(f"지원하지 않는 controller policy kind: {candidate.kind}")

    base_scopes = base.scope_thresholds or {}
    candidate_scopes = candidate.scope_thresholds or {}
    labels = sorted(set(base_scopes) | set(candidate_scopes))
    thresholds: dict[str, float] = {}
    for label in labels:
        base_value = base_scopes.get(
            label,
            base.fallback_threshold if base.fallback_threshold is not None else base.threshold,
        )
        candidate_value = candidate_scopes.get(
            label,
            candidate.fallback_threshold if candidate.fallback_threshold is not None else candidate.threshold,
        )
        thresholds[label] = _conservative_threshold(base_value, candidate_value, allow_decrease)
    fallback = _conservative_threshold(
        base.fallback_threshold if base.fallback_threshold is not None else base.threshold,
        candidate.fallback_threshold if candidate.fallback_threshold is not None else candidate.threshold,
        allow_decrease,
    )
    return dataclasses.replace(
        candidate,
        threshold=fallback,
        fallback_threshold=fallback,
        scope_thresholds=thresholds,
    )


def evaluate_policy(
    frame: pd.DataFrame,
    policy: PrecisionPolicy,
    confidence_level: float,
) -> dict[str, Any]:
    scopes = (
        frame[policy.scope_column].astype("string").fillna("__MISSING__").to_numpy(dtype=object)
        if policy.scope_column and policy.scope_column in frame.columns
        else None
    )
    alert = apply_precision_policy(policy, frame["score"].to_numpy(dtype=np.float64), scopes)
    return evaluate_alerts(
        frame["target"].to_numpy(dtype=np.int8),
        frame["score"].to_numpy(dtype=np.float64),
        alert,
        frame["date"].to_numpy(dtype="datetime64[ns]"),
        frame[TICKER_COLUMN].to_numpy(dtype=object),
        confidence_level,
    )


def policy_is_active(
    metrics: Mapping[str, Any],
    target_precision: float,
    minimum_precision_lcb: float,
    minimum_alerts: int,
    minimum_alert_days: int,
) -> bool:
    return bool(
        float(metrics.get("precision", float("nan"))) >= float(target_precision)
        and float(metrics.get("precision_wilson_lcb", float("nan"))) >= float(minimum_precision_lcb)
        and int(metrics.get("alerts", 0)) >= int(minimum_alerts)
        and int(metrics.get("alert_days", 0)) >= int(minimum_alert_days)
    )


def _load_history(
    scores_path: Path,
    target_path: Path,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.Series]:
    scores = pd.read_csv(scores_path)
    required = {"source_row_id", args.date_column, args.ticker_column, args.score_column}
    missing = required - set(scores.columns)
    if missing:
        raise KeyError(f"score 파일 컬럼 누락: {sorted(missing)}")
    scores[args.date_column] = pd.to_datetime(scores[args.date_column], errors="coerce")
    if scores[args.date_column].isna().any():
        raise ValueError("score 파일 날짜 파싱 실패")

    side_columns = table_columns(target_path)
    requested = [
        "source_row_id",
        args.date_column,
        args.ticker_column,
        args.target_column,
        args.target_valid_column,
    ]
    sidecar = read_table(target_path, [value for value in requested if value in side_columns])
    joined = join_source_and_target(
        scores,
        sidecar,
        target_column=args.target_column,
        target_valid_column=args.target_valid_column,
        date_column=args.date_column,
        ticker_column=args.ticker_column,
    )
    joined[args.date_column] = pd.to_datetime(joined[args.date_column], errors="coerce")
    target = pd.to_numeric(joined[args.target_column], errors="coerce")
    score = pd.to_numeric(joined[args.score_column], errors="coerce")
    joined["_target_valid"] = parse_bool_series(joined[args.target_valid_column]).to_numpy(dtype=bool)
    joined["target"] = target
    joined["score"] = score
    finite = joined[args.date_column].notna() & np.isfinite(score)
    joined = joined.loc[finite].copy()
    return joined, scores[args.date_column].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Update Precision 70 V6 threshold using only matured, genuinely out-of-sample D+3 labels. "
            "No alert-count/rate cap is introduced."
        )
    )
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/surge_precision70_v6"))
    parser.add_argument("--scores", type=Path, required=True, help="누적 surge_precision70_scores.csv")
    parser.add_argument("--target-sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--date-column", default=DATE_COLUMN)
    parser.add_argument("--ticker-column", default=TICKER_COLUMN)
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--target-valid-column", default=TARGET_VALID_COLUMN)
    parser.add_argument("--score-column", default=SCORE_COLUMN)
    parser.add_argument("--label-delay-trading-days", type=int, default=3)
    parser.add_argument("--rolling-trading-days", type=int, default=504)
    parser.add_argument("--precision-buffer", type=float, default=0.03)
    parser.add_argument("--minimum-alerts", type=int, default=100)
    parser.add_argument("--minimum-alert-days", type=int, default=20)
    parser.add_argument("--minimum-precision-lcb", type=float, default=0.60)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    parser.add_argument("--scope-column")
    parser.add_argument("--minimum-alerts-per-scope", type=int, default=30)
    parser.add_argument("--minimum-alert-days-per-scope", type=int, default=10)
    parser.add_argument("--maximum-threshold-candidates", type=int, default=500)
    parser.add_argument("--allow-threshold-decrease", action="store_true")
    parser.add_argument(
        "--allow-in-sample-diagnostics",
        action="store_true",
        help="학습 종료일 이전 재점수화 자료도 허용. 실전 controller에는 사용하지 말 것.",
    )
    args = parser.parse_args()

    if args.label_delay_trading_days < 0:
        raise ValueError("label-delay-trading-days는 0 이상이어야 합니다")
    if args.rolling_trading_days <= 0:
        raise ValueError("rolling-trading-days는 양수여야 합니다")
    if not (0.0 <= args.precision_buffer < 0.30):
        raise ValueError("precision-buffer는 0~0.30 범위여야 합니다")

    model_dir = args.model_dir.resolve()
    scores_path = args.scores.resolve()
    target_path = args.target_sidecar.resolve()
    output_path = args.output.resolve() if args.output else model_dir / "PRECISION_CONTROLLER_STATE_V6.json"
    freeze_path = model_dir / "PRECISION_FREEZE_V6.json"
    freeze = load_json_checked(freeze_path)
    base_policy = PrecisionPolicy.from_dict(dict(freeze["precision_policy"]))

    history, all_score_dates = _load_history(scores_path, target_path, args)
    training_end_raw = freeze.get("training_data_end")
    training_end = pd.Timestamp(training_end_raw) if training_end_raw else None
    if training_end is not None and not args.allow_in_sample_diagnostics:
        in_sample = history[args.date_column] <= training_end
        if bool(in_sample.any()):
            first = history.loc[in_sample, args.date_column].min()
            last = history.loc[in_sample, args.date_column].max()
            raise ValueError(
                "controller에는 동결 모델 학습 종료일 이후 실시간 저장 점수만 사용할 수 있습니다. "
                f"training_end={training_end.isoformat()}, in_sample_range={first}~{last}. "
                "과거 재점수화는 --allow-in-sample-diagnostics에서만 허용됩니다."
            )

    cutoff = mature_cutoff(all_score_dates, args.label_delay_trading_days)
    matured_mask = (
        (history[args.date_column] <= cutoff)
        & history["_target_valid"].astype(bool)
        & pd.to_numeric(history["target"], errors="coerce").isin([0, 1])
    )
    matured = history.loc[matured_mask].copy()
    matured["target"] = pd.to_numeric(matured["target"], errors="coerce").astype(np.uint8)
    unique_dates = pd.DatetimeIndex(matured[args.date_column].dropna().unique()).sort_values()
    if len(unique_dates) > int(args.rolling_trading_days):
        start_date = pd.Timestamp(unique_dates[-int(args.rolling_trading_days)])
        matured = matured.loc[matured[args.date_column] >= start_date].copy()
    if matured.empty or matured["target"].nunique() < 2:
        raise ValueError("controller 학습용 성숙 OOS history가 비어 있거나 단일 class입니다")

    # Chronological blocks prevent one pooled lucky threshold from activating the controller.
    ordered_dates = pd.DatetimeIndex(matured[args.date_column].unique()).sort_values()
    block_count = min(4, max(2, len(ordered_dates) // 40))
    date_blocks = np.array_split(ordered_dates, block_count)
    fold_map: dict[pd.Timestamp, int] = {}
    for fold_id, block in enumerate(date_blocks):
        for value in block:
            fold_map[pd.Timestamp(value)] = int(fold_id)
    matured["fold_id"] = matured[args.date_column].map(fold_map).astype(int)
    matured["date"] = matured[args.date_column]
    if args.ticker_column != TICKER_COLUMN:
        matured[TICKER_COLUMN] = matured[args.ticker_column].astype(str)
    else:
        matured[TICKER_COLUMN] = matured[TICKER_COLUMN].astype(str)

    development_target = min(
        0.999999,
        float(base_policy.target_precision) + float(args.precision_buffer),
    )
    policy_error = None
    search = pd.DataFrame()
    try:
        scope_column = args.scope_column or base_policy.scope_column
        if scope_column:
            if scope_column not in matured.columns:
                raise KeyError(f"controller scope column 누락: {scope_column}")
            candidate, search = select_scope_precision_policy(
                matured,
                scope_column,
                target_precision=development_target,
                minimum_precision_lcb=args.minimum_precision_lcb,
                minimum_alerts_per_scope=args.minimum_alerts_per_scope,
                minimum_alert_days_per_scope=args.minimum_alert_days_per_scope,
                confidence_level=args.confidence_level,
                source="matured_oos_controller",
            )
        else:
            per_fold_alerts = max(3, int(np.ceil(args.minimum_alerts / max(1, block_count))))
            per_fold_days = max(2, int(np.ceil(args.minimum_alert_days / max(1, block_count))))
            candidate, search, _ = select_global_precision_policy(
                matured,
                target_precision=development_target,
                minimum_precision_lcb=args.minimum_precision_lcb,
                minimum_alerts_per_fold=per_fold_alerts,
                minimum_alert_days_per_fold=per_fold_days,
                confidence_level=args.confidence_level,
                required_fold_pass_rate=1.0,
                source="matured_oos_controller",
                maximum_threshold_candidates=args.maximum_threshold_candidates,
            )
        updated_policy = clamp_policy_to_base(base_policy, candidate, args.allow_threshold_decrease)
    except Exception as exc:
        candidate = _safe_abstain_policy(base_policy, "matured_oos_controller_failed")
        updated_policy = candidate
        policy_error = f"{type(exc).__name__}: {exc}"

    metrics = evaluate_policy(matured, updated_policy, args.confidence_level)
    active = bool(
        candidate.gate_pass
        and policy_is_active(
            metrics,
            development_target,
            args.minimum_precision_lcb,
            args.minimum_alerts,
            args.minimum_alert_days,
        )
    )
    active_policy = updated_policy if active else _safe_abstain_policy(base_policy, "matured_oos_controller_abstain")
    status = "ACTIVE" if active else "STOP_ONLINE_PRECISION70_GATE"

    state = with_payload_checksum(
        {
            "schema": "crashwatch_surge_precision70_controller_state_v6",
            "created_at": utc_now(),
            "status": status,
            "label_delay_trading_days": int(args.label_delay_trading_days),
            "rolling_trading_days": int(args.rolling_trading_days),
            "matured_through": cutoff.isoformat(),
            "history_start": pd.Timestamp(matured[args.date_column].min()).isoformat(),
            "history_end": pd.Timestamp(matured[args.date_column].max()).isoformat(),
            "history_rows": int(len(matured)),
            "history_positives": int(matured["target"].sum()),
            "target_precision": float(base_policy.target_precision),
            "development_target_precision": float(development_target),
            "minimum_precision_lcb": float(args.minimum_precision_lcb),
            "minimum_alerts": int(args.minimum_alerts),
            "minimum_alert_days": int(args.minimum_alert_days),
            "allow_threshold_decrease": bool(args.allow_threshold_decrease),
            "no_alert_count_limit": True,
            "base_policy": base_policy.to_dict(),
            "candidate_policy": candidate.to_dict(),
            "active_policy": active_policy.to_dict(),
            "current_policy": active_policy.to_dict(),
            "active_policy_metrics": metrics,
            "policy_error": policy_error,
            "training_data_end": training_end.isoformat() if training_end is not None else None,
            "scores_path": str(scores_path),
            "scores_sha256": sha256_file(scores_path),
            "target_sidecar": str(target_path),
            "target_sha256": sha256_file(target_path),
            "freeze_path": str(freeze_path),
            "freeze_sha256": sha256_file(freeze_path),
            "base_freeze_sha256": sha256_file(freeze_path),
            "search_top": search.head(50).to_dict(orient="records") if not search.empty else [],
            "warning": (
                "D+3 라벨이 확정된 OOS 점수만 사용한다. ACTIVE가 아니면 무경보 정책을 유지한다."
            ),
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_path, state)
    print(json.dumps(state, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
