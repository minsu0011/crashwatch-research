from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_estimators: int = 800
    max_depth: int = 7
    learning_rate: float = 0.035
    subsample: float = 0.90
    colsample_bytree: float = 0.78
    min_child_weight: float = 5.0
    reg_alpha: float = 0.15
    reg_lambda: float = 1.5
    max_bin: int = 256
    grow_policy: str = "depthwise"
    max_leaves: int = 0

    def payload(self) -> dict:
        return asdict(self)


CANDIDATES = [
    ModelConfig("depth6_fast", n_estimators=650, max_depth=6, learning_rate=0.045, colsample_bytree=0.80),
    ModelConfig("depth7_balanced", n_estimators=850, max_depth=7, learning_rate=0.032, colsample_bytree=0.76),
    ModelConfig("depth8_slow", n_estimators=1000, max_depth=8, learning_rate=0.025, min_child_weight=7.0, colsample_bytree=0.72),
    ModelConfig("lossguide64", n_estimators=900, max_depth=0, max_leaves=64, grow_policy="lossguide", learning_rate=0.03, colsample_bytree=0.76),
]


def fit_xgb(x_train: np.ndarray, y_train: np.ndarray, seed: int, config: ModelConfig, *, threads: int = 16, prefer_gpu: bool = True):
    from xgboost import XGBClassifier

    x_train = np.ascontiguousarray(x_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int8)
    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))
    scale_pos_weight = max(1.0, negatives / max(1, positives))
    base_params = dict(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        max_bin=config.max_bin,
        grow_policy=config.grow_policy,
        max_leaves=config.max_leaves,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=threads,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        device="cuda" if prefer_gpu else "cpu",
        verbosity=0,
        validate_parameters=True,
    )
    attempts: list[dict] = []

    def train(label: str, params: dict, rows: np.ndarray | None = None):
        fit_x = x_train if rows is None else np.ascontiguousarray(x_train[rows], dtype=np.float32)
        fit_y = y_train if rows is None else y_train[rows]
        model = XGBClassifier(**params)
        try:
            model.fit(fit_x, fit_y)
            attempts.append({
                "step": label, "status": "success", "rows": int(len(fit_y)),
                "device": params["device"], "max_bin": int(params["max_bin"]),
                "max_depth": int(params["max_depth"]), "max_leaves": int(params["max_leaves"]),
                "colsample_bytree": float(params["colsample_bytree"]),
            })
            return model
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            attempts.append({
                "step": label, "status": "failed", "rows": int(len(fit_y)),
                "device": params["device"], "max_bin": int(params["max_bin"]),
                "max_depth": int(params["max_depth"]), "max_leaves": int(params["max_leaves"]),
                "colsample_bytree": float(params["colsample_bytree"]),
                "error": message,
            })
            raise

    def is_oom(exc: Exception) -> bool:
        message = str(exc).lower()
        return any(token in message for token in ("out of memory", "cuda_error_out_of_memory", "std::bad_alloc", "bad allocation"))

    if prefer_gpu:
        gpu_params = dict(base_params)
        gpu_steps: list[tuple[str, dict, np.ndarray | None]] = [
            ("cuda_default", dict(gpu_params), None),
        ]
        reduced_bin = dict(gpu_params)
        reduced_bin["max_bin"] = min(128, int(reduced_bin["max_bin"]))
        gpu_steps.append(("cuda_max_bin_128", reduced_bin, None))

        reduced_depth = dict(reduced_bin)
        if int(reduced_depth["max_depth"]) > 0:
            reduced_depth["max_depth"] = max(4, int(reduced_depth["max_depth"]) - 1)
        elif int(reduced_depth["max_leaves"]) > 0:
            reduced_depth["max_leaves"] = max(32, int(reduced_depth["max_leaves"]) // 2)
        gpu_steps.append(("cuda_reduced_depth", reduced_depth, None))

        reduced_columns = dict(reduced_depth)
        reduced_columns["colsample_bytree"] = min(0.65, float(reduced_columns["colsample_bytree"]))
        gpu_steps.append(("cuda_reduced_columns", reduced_columns, None))

        sample_count = max(1000, int(len(y_train) * 0.80))
        sampled_rows = None
        if sample_count < len(y_train):
            rng = np.random.default_rng(seed)
            sampled_rows = np.sort(rng.choice(len(y_train), size=sample_count, replace=False))
        gpu_steps.append(("cuda_row_sample_80pct", reduced_columns, sampled_rows))

        last_gpu_error: Exception | None = None
        for label, params, rows in gpu_steps:
            try:
                model = train(label, params, rows)
                diagnostics = {"fallback": label != "cuda_default", "oom_detected": bool(last_gpu_error), "attempts": attempts}
                return model, "xgboost_cuda" if label == "cuda_default" else f"xgboost_{label}", diagnostics
            except Exception as exc:  # noqa: BLE001
                last_gpu_error = exc
                if not is_oom(exc):
                    LOGGER.warning("CUDA 일반 오류, CPU로 전환: %s", exc)
                    break
                LOGGER.warning("CUDA OOM, 다음 축소 단계로 재시도: step=%s error=%s", label, exc)

    cpu_params = dict(base_params)
    cpu_params["device"] = "cpu"
    try:
        model = train("cpu_fallback" if prefer_gpu else "cpu_requested", cpu_params)
    except Exception:
        LOGGER.exception("XGBoost CPU 학습도 실패했습니다.")
        raise
    diagnostics = {
        "fallback": bool(prefer_gpu),
        "oom_detected": any(
            any(token in str(item.get("error", "")).lower() for token in ("out of memory", "cuda_error_out_of_memory", "bad_alloc"))
            for item in attempts
        ),
        "attempts": attempts,
    }
    return model, "xgboost_cpu_fallback" if prefer_gpu else "xgboost_cpu", diagnostics
