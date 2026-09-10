from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ROUTE = {
    "CRASH_STRESS": "P2_XGB",
    "BULL_HIGH_VOL": "P2_XGB",
    "SIDEWAYS_LOW_VOL": "P2_XGB",
    "BEAR_LOW_VOL": "P2_XGB",
    "REBOUND": "P7_LGB",
    "BULL_LOW_VOL": "P7_LGB",
    "SIDEWAYS_HIGH_VOL": "P7_LGB",
    "BEAR_HIGH_VOL": "P7_LGB",
}

REGIME_KO_V2 = {
    "CRASH_STRESS": "급락·스트레스",
    "REBOUND": "급락 후 반등",
    "BULL_LOW_VOL": "상승·저변동성",
    "BULL_HIGH_VOL": "강상승·고변동성",
    "SIDEWAYS_LOW_VOL": "횡보·저변동성",
    "SIDEWAYS_HIGH_VOL": "강횡보·고변동성",
    "BEAR_LOW_VOL": "하락·저변동성",
    "BEAR_HIGH_VOL": "강하락·고변동성",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_model_path(artifact_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = artifact_path.parent / path
    return path.resolve()


def _validate_one_regime_per_date(frame: pd.DataFrame) -> None:
    counts = frame.groupby("date", sort=False)["regime"].nunique(dropna=False)
    bad = counts[counts != 1]
    if not bad.empty:
        examples = ", ".join(str(value) for value in bad.index[:5])
        raise ValueError(f"a market date must have exactly one regime; invalid dates: {examples}")


def attach_regime_calendar(frame: pd.DataFrame, calendar: pd.DataFrame | None) -> pd.DataFrame:
    out = frame.copy()
    if "date" not in out.columns:
        raise ValueError("input feature frame is missing date")
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()
    if "regime" not in out.columns:
        if calendar is None:
            raise ValueError("regime is missing; pass a trailing-information regime calendar")
        required = {"date", "regime"}
        missing = required - set(calendar.columns)
        if missing:
            raise ValueError(f"regime calendar is missing columns: {sorted(missing)}")
        cal = calendar[["date", "regime"]].copy()
        cal["date"] = pd.to_datetime(cal["date"], errors="raise").dt.normalize()
        if cal.duplicated("date").any():
            raise ValueError("regime calendar contains duplicate dates")
        out = out.merge(cal, on="date", how="left", validate="many_to_one")
    if out["regime"].isna().any():
        missing_dates = out.loc[out["regime"].isna(), "date"].drop_duplicates().head(5).tolist()
        raise ValueError(f"regime calendar has no value for dates: {missing_dates}")
    _validate_one_regime_per_date(out)
    return out


def add_daily_alerts(frame: pd.DataFrame, fraction: float = 0.03) -> pd.DataFrame:
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("alert fraction must be in (0, 1]")
    out = frame.copy()
    out["daily_rank"] = out.groupby("date", sort=False)["prediction"].rank(
        method="first", ascending=False
    ).astype(np.int32)
    sizes = out.groupby("date", sort=False)["prediction"].transform("size")
    thresholds = np.maximum(1, np.ceil(sizes.to_numpy(dtype=float) * float(fraction))).astype(np.int32)
    out["alert_top_3pct"] = out["daily_rank"].to_numpy(dtype=np.int32) <= thresholds
    return out


def route_prediction_columns(
    frame: pd.DataFrame,
    mapping: dict[str, str] | None = None,
    *,
    fallback_model: str = "P2_XGB",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """Apply a frozen market-regime route to existing expert prediction columns."""
    mapping = dict(mapping or DEFAULT_ROUTE)
    missing_models = sorted(set(mapping.values()) - set(frame.columns))
    if fallback_model not in frame.columns:
        missing_models.append(fallback_model)
    if missing_models:
        raise ValueError(f"expert prediction columns are missing: {sorted(set(missing_models))}")
    out = frame.copy()
    selected = out["regime"].map(mapping).fillna(fallback_model)
    out["selected_submodel"] = selected
    out["route_was_fallback"] = ~out["regime"].isin(mapping)
    prediction = np.full(len(out), np.nan, dtype=np.float64)
    for model in sorted(set(mapping.values()) | {fallback_model}):
        mask = selected.eq(model).to_numpy()
        if mask.any():
            prediction[mask] = out.loc[mask, model].to_numpy(dtype=np.float64)
    if not np.isfinite(prediction).all():
        raise ValueError("routed expert predictions contain NaN or infinity")
    out[prediction_column] = np.clip(prediction, 1e-7, 1.0 - 1e-7)
    return out


@dataclass(frozen=True)
class ExpertDefinition:
    name: str
    algorithm: str
    profile: str
    features: tuple[str, ...]
    model_paths: tuple[Path, ...]
    model_hashes: tuple[str, ...]


class RegimeExpertRouter:
    """Production inference router for date-level market regimes.

    The gate never uses target columns.  Every date is assigned one market regime,
    and every ticker on that date is scored by the same frozen expert ensemble.
    """

    def __init__(
        self,
        artifact_path: str | os.PathLike[str],
        *,
        total_threads: int = 32,
        use_gpu: bool = True,
        verify_hashes: bool = True,
    ) -> None:
        self.artifact_path = Path(artifact_path).expanduser().resolve()
        with self.artifact_path.open("r", encoding="utf-8") as handle:
            self.artifact = json.load(handle)
        if self.artifact.get("schema") != "crashwatch_regime_expert_router_v2":
            raise ValueError("unsupported regime expert router artifact")
        self.mapping = dict(self.artifact["regime_mapping"])
        self.fallback_model = str(self.artifact["fallback_model"])
        self.total_threads = max(1, int(total_threads))
        self.use_gpu = bool(use_gpu)
        self.verify_hashes = bool(verify_hashes)
        self._loaded: dict[str, list[Any]] = {}
        self.last_diagnostics: dict[str, Any] = {}
        self.experts = self._parse_experts()
        route_models = set(self.mapping.values()) | {self.fallback_model}
        missing = route_models - set(self.experts)
        if missing:
            raise ValueError(f"router artifact has no definitions for experts: {sorted(missing)}")

    def _parse_experts(self) -> dict[str, ExpertDefinition]:
        out: dict[str, ExpertDefinition] = {}
        for name, item in self.artifact["experts"].items():
            models = item.get("models", [])
            paths = tuple(_resolve_model_path(self.artifact_path, model["path"]) for model in models)
            hashes = tuple(str(model["sha256"]) for model in models)
            if not paths:
                raise ValueError(f"expert {name} has no model files")
            for path, expected in zip(paths, hashes, strict=True):
                if not path.exists():
                    raise FileNotFoundError(path)
                if self.verify_hashes and _sha256(path) != expected:
                    raise ValueError(f"model hash mismatch: {path}")
            out[name] = ExpertDefinition(
                name=str(name),
                algorithm=str(item["algorithm"]),
                profile=str(item["profile"]),
                features=tuple(str(value) for value in item["features"]),
                model_paths=paths,
                model_hashes=hashes,
            )
        return out

    def required_features(self) -> list[str]:
        selected = set(self.mapping.values()) | {self.fallback_model}
        return sorted(set().union(*(set(self.experts[name].features) for name in selected)))

    def _load_models(self, expert: ExpertDefinition) -> list[Any]:
        if expert.name in self._loaded:
            return self._loaded[expert.name]
        models: list[Any] = []
        if expert.algorithm == "xgboost":
            import xgboost as xgb

            for path in expert.model_paths:
                model = xgb.Booster()
                model.load_model(path)
                model.set_param({
                    "device": "cuda" if self.use_gpu else "cpu",
                    "nthread": self.total_threads,
                })
                models.append(model)
        elif expert.algorithm == "lightgbm":
            import lightgbm as lgb

            # LightGBM's Windows native file loader fails on some Korean paths.
            # Python can read the same files correctly, so pass their contents
            # through memory and keep model hashes tied to the original files.
            models = [
                lgb.Booster(model_str=path.read_text(encoding="utf-8"))
                for path in expert.model_paths
            ]
        else:
            raise ValueError(f"unsupported expert algorithm: {expert.algorithm}")
        self._loaded[expert.name] = models
        return models

    @staticmethod
    def _matrix(frame: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
        missing = [name for name in features if name not in frame.columns]
        if missing:
            raise ValueError(f"input is missing {len(missing)} required features; examples={missing[:10]}")
        matrix = frame.loc[:, list(features)].replace([np.inf, -np.inf], np.nan).to_numpy(
            dtype=np.float32, copy=True
        )
        return matrix

    def _predict_expert(self, expert_name: str, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        expert = self.experts[expert_name]
        matrix = self._matrix(frame, expert.features)
        models = self._load_models(expert)
        used_gpu = False
        fallback_reason: str | None = None
        if expert.algorithm == "xgboost":
            import xgboost as xgb

            def predict(device: str) -> np.ndarray:
                for model in models:
                    model.set_param({"device": device, "nthread": self.total_threads})
                data = xgb.DMatrix(matrix, missing=np.nan, nthread=self.total_threads)
                return np.vstack([np.asarray(model.predict(data), dtype=np.float32) for model in models])

            if self.use_gpu:
                try:
                    predictions = predict("cuda")
                    used_gpu = True
                except xgb.core.XGBoostError as exc:
                    fallback_reason = repr(exc)
                    predictions = predict("cpu")
            else:
                predictions = predict("cpu")
        else:
            per_model_threads = max(1, self.total_threads // max(1, len(models)))

            def predict_lgb(model: Any) -> np.ndarray:
                return np.asarray(model.predict(matrix, num_threads=per_model_threads), dtype=np.float32)

            with cf.ThreadPoolExecutor(max_workers=min(len(models), self.total_threads)) as executor:
                predictions = np.vstack(list(executor.map(predict_lgb, models)))
        if predictions.shape != (len(models), len(frame)):
            raise RuntimeError(f"unexpected prediction shape for {expert_name}: {predictions.shape}")
        if not np.isfinite(predictions).all():
            raise RuntimeError(f"non-finite predictions from {expert_name}")
        mean = np.clip(predictions.mean(axis=0), 1e-7, 1.0 - 1e-7)
        std = predictions.std(axis=0, ddof=0)
        diagnostics = {
            "expert": expert_name,
            "algorithm": expert.algorithm,
            "rows": int(len(frame)),
            "features": int(len(expert.features)),
            "models": int(len(models)),
            "gpu_requested": bool(self.use_gpu and expert.algorithm == "xgboost"),
            "gpu_used": bool(used_gpu),
            "gpu_fallback_reason": fallback_reason,
            "prediction_min": float(mean.min()) if len(mean) else None,
            "prediction_max": float(mean.max()) if len(mean) else None,
            "prediction_mean": float(mean.mean()) if len(mean) else None,
            "mean_seed_std": float(std.mean()) if len(std) else None,
        }
        return mean, std, diagnostics

    def predict_frame(
        self,
        features: pd.DataFrame,
        *,
        regime_calendar: pd.DataFrame | None = None,
        alert_fraction: float = 0.03,
    ) -> pd.DataFrame:
        frame = attach_regime_calendar(features, regime_calendar)
        selected = frame["regime"].map(self.mapping).fillna(self.fallback_model)
        frame["selected_submodel"] = selected
        frame["route_was_fallback"] = ~frame["regime"].isin(self.mapping)
        prediction = np.full(len(frame), np.nan, dtype=np.float64)
        uncertainty = np.full(len(frame), np.nan, dtype=np.float64)
        diagnostics: list[dict[str, Any]] = []

        # Keep XGBoost CUDA and LightGBM OpenMP calls in one process but do not
        # enter their native runtimes concurrently.  On Windows, simultaneous
        # CUDA/OpenMP initialization can terminate the interpreter without a
        # catchable Python exception.  Each expert still uses all configured
        # hardware internally, and LightGBM seed models run in parallel.
        for expert_name in sorted(selected.unique()):
            indices = np.flatnonzero(selected.eq(expert_name).to_numpy())
            mean, std, diag = self._predict_expert(str(expert_name), frame.iloc[indices])
            prediction[indices] = mean
            uncertainty[indices] = std
            diagnostics.append(diag)

        if not np.isfinite(prediction).all() or not np.isfinite(uncertainty).all():
            raise RuntimeError("router did not score every row")
        result_columns = [name for name in ("row_id", "date", "ticker", "regime") if name in frame.columns]
        result = frame[result_columns].copy()
        result["regime_ko"] = result["regime"].map(REGIME_KO_V2).fillna("알 수 없는 국면")
        result["selected_submodel"] = selected.to_numpy()
        result["route_was_fallback"] = frame["route_was_fallback"].to_numpy(dtype=bool)
        result["prediction"] = prediction
        result["seed_uncertainty"] = uncertainty
        result["gate_hash"] = str(self.artifact["gate_hash"])
        result = add_daily_alerts(result, alert_fraction)
        self.last_diagnostics = {
            "rows": int(len(result)),
            "dates": int(result["date"].nunique()),
            "tickers": int(result["ticker"].nunique()) if "ticker" in result else None,
            "route_counts": result["selected_submodel"].value_counts().to_dict(),
            "fallback_rows": int(result["route_was_fallback"].sum()),
            "experts": sorted(diagnostics, key=lambda item: item["expert"]),
        }
        return result


def build_router_artifact(
    gate: dict[str, Any],
    model_root: Path,
    *,
    route_mapping: dict[str, str] | None = None,
    artifact_root: Path | None = None,
) -> dict[str, Any]:
    model_root = Path(model_root).resolve()
    output_root = Path(artifact_root).resolve() if artifact_root is not None else model_root.parent
    mapping = dict(route_mapping or DEFAULT_ROUTE)
    definitions = {
        "P2_XGB": ("xgboost", "P2", "*.ubj"),
        "P7_LGB": ("lightgbm", "P7", "*.txt"),
    }
    experts: dict[str, Any] = {}
    for name, (algorithm, profile, pattern) in definitions.items():
        paths = sorted((model_root / name).glob(pattern))
        if len(paths) != 5:
            raise RuntimeError(f"expected five seed models for {name}, found {len(paths)}")
        experts[name] = {
            "algorithm": algorithm,
            "profile": profile,
            "features": list(gate["profile_features"][profile]),
            "feature_hash": gate["feature_hashes"][profile],
            "aggregation": "arithmetic mean across five frozen seeds",
            "models": [
                {
                    "path": Path(os.path.relpath(path, output_root)).as_posix(),
                    "sha256": _sha256(path),
                    "seed": int(path.stem.split("_")[-1]),
                }
                for path in paths
            ],
        }
    artifact = {
        "schema": "crashwatch_regime_expert_router_v2",
        "target": gate["target"],
        "development_max": gate["development_max"],
        "gate_hash": gate["artifact_hash"],
        "gate_frozen": True,
        "regime_mapping": mapping,
        "regime_names_ko": REGIME_KO_V2,
        "fallback_model": "P2_XGB",
        "fallback_policy": "unknown regime only; known regimes are always hard-routed",
        "route_granularity": "one market regime and one expert ensemble per trading date",
        "leakage_policy": "regime uses contemporaneous/trailing market data only; no target is accepted by the router",
        "alert_policy": "rank all tickers per date and alert max(1, ceil(3% of rows))",
        "experts": experts,
    }
    identity = dict(artifact)
    artifact["router_hash"] = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:32]
    return artifact
