from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import xgboost as xgb


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(data: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def feature_hash(features: list[str]) -> str:
    return hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()


def save_native_model(model: object, destination: Path, tag: str) -> None:
    """Save through an ASCII temporary path for Windows native libraries.

    LightGBM's native writer can fail when the destination contains Korean
    characters even though Python can access the same directory normally.
    XGBoost uses the same path to keep packaging behavior consistent.
    """
    suffix = destination.suffix
    temporary = Path(tempfile.gettempdir()) / f"cwdeploy_{os.getpid()}_{tag}{suffix}"
    try:
        if temporary.exists():
            temporary.unlink()
        model.save_model(str(temporary))
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def copy_release_sources(source_dir: Path, output_dir: Path) -> None:
    for name in [
        "crashwatch_predict.py",
        "verify_package.py",
        "README_KO.md",
        "MODEL_CARD_KO.md",
        "DEPLOYMENT_TEST_REPORT.md",
        "requirements.txt",
        "INSTALL_WINDOWS.bat",
        "PREDICT_LATEST_WINDOWS.bat",
        "NOTICE.md",
    ]:
        source = source_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        shutil.copy2(source, output_dir / name)


def write_checksums(output: Path) -> None:
    checksums: dict[str, str] = {}
    for path in sorted(output.rglob("*")):
        relative = path.relative_to(output)
        if (
            path.is_file()
            and path.name != "SHA256SUMS.json"
            and path.suffix.lower() != ".pyc"
            and "__pycache__" not in relative.parts
        ):
            checksums[relative.as_posix()] = sha256_file(path)
    atomic_json(checksums, output / "SHA256SUMS.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the frozen CrashWatch NormalMarket 3D/-5% deployment ensemble")
    parser.add_argument("--runtime-manifest", required=True)
    parser.add_argument("--freeze-manifest", required=True)
    parser.add_argument("--blend-selection", required=True)
    parser.add_argument("--validation-results", required=True)
    parser.add_argument("--release-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    args = parser.parse_args()

    started = time.time()
    runtime_path = Path(args.runtime_manifest).resolve()
    freeze_path = Path(args.freeze_manifest).resolve()
    blend_path = Path(args.blend_selection).resolve()
    validation_dir = Path(args.validation_results).resolve()
    release_source = Path(args.release_source).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    models_dir = output / "models"
    config_dir = output / "config"
    evidence_dir = output / "validation_evidence"
    models_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)

    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze["final_candidate"] != "C0_P2_ALLTRAIN":
        raise RuntimeError(f"unexpected frozen candidate: {freeze['final_candidate']}")
    if freeze["sector_expert"]["enabled"] or float(freeze["sector_expert"]["weight"]) != 0.0:
        raise RuntimeError("sector overlay must remain disabled in this release")
    candidate = freeze["candidate"]
    if candidate["profile"] != "P2_DEDUP_CLEAN" or candidate["train_policy"] != "all":
        raise RuntimeError(f"unexpected frozen candidate spec: {candidate}")

    features = [str(item) for item in freeze["profile_features"]]
    if len(features) != 371 or len(set(features)) != len(features):
        raise RuntimeError(f"invalid frozen feature list: count={len(features)} unique={len(set(features))}")
    matrix_path = Path(runtime["matrix_paths"][candidate["profile"]])
    target_path = Path(runtime["target_path"])
    valid_path = Path(runtime["valid_path"])
    for path in [matrix_path, target_path, valid_path]:
        if not path.exists():
            raise FileNotFoundError(path)

    X_all = np.load(matrix_path, mmap_mode="r")
    y_all = np.load(target_path, mmap_mode="r")
    valid = np.load(valid_path, mmap_mode="r").astype(bool)
    train_idx = np.flatnonzero(valid)
    X = np.asarray(X_all[train_idx], dtype=np.float32)
    y = np.asarray(y_all[train_idx], dtype=np.uint8)
    if X.shape != (len(y), len(features)):
        raise RuntimeError(f"training shape mismatch X={X.shape} y={y.shape} features={len(features)}")
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if not positives or not negatives:
        raise RuntimeError("training target is single-class")

    seeds = [int(seed) for seed in freeze["seeds"]]
    lgb_cfg = dict(freeze["model_params"]["lightgbm"])
    xgb_cfg = dict(freeze["model_params"]["xgboost"])
    lgb_rounds = int(lgb_cfg.pop("rounds"))
    xgb_rounds = int(xgb_cfg.pop("rounds"))
    lgb_multiplier = float(lgb_cfg.pop("class_weight_multiplier", 1.0))
    xgb_multiplier = float(xgb_cfg.pop("class_weight_multiplier", 1.0))

    # The blend was fixed during selection. It is not re-selected on full data.
    import csv

    with blend_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("candidate") == "C0_P2_ALLTRAIN"]
    if not rows:
        raise RuntimeError("C0 blend rows missing")
    selected = max(rows, key=lambda row: float(row["selection_score"]))
    xgb_weight = float(selected["xgb_weight"])
    if xgb_weight != 0.75:
        raise RuntimeError(f"expected frozen xgboost weight 0.75, got {xgb_weight}")

    artifacts: list[dict[str, object]] = []
    sample_idx = train_idx[-min(2048, len(train_idx)) :]
    sample_X = np.asarray(X_all[sample_idx], dtype=np.float32)
    lgb_sample_predictions: list[np.ndarray] = []
    xgb_sample_predictions: list[np.ndarray] = []

    lgb_base = {
        "objective": "binary",
        "metric": "None",
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "num_threads": int(args.threads),
        "scale_pos_weight": (negatives / positives) * lgb_multiplier,
    }
    lgb_base.update(lgb_cfg)
    lgb_dataset = lgb.Dataset(
        X,
        label=y,
        free_raw_data=False,
        feature_name=features,
        params={"max_bin": int(lgb_base.get("max_bin", 255)), "feature_pre_filter": False},
    )
    for seed in seeds:
        params = dict(lgb_base)
        params.update({"seed": seed, "feature_fraction_seed": seed, "bagging_seed": seed})
        model = lgb.train(params, lgb_dataset, num_boost_round=lgb_rounds, callbacks=[lgb.log_evaluation(0)])
        relative = Path("models") / f"lightgbm_seed_{seed}.txt"
        destination = output / relative
        save_native_model(model, destination, f"lgb_{seed}")
        prediction = np.asarray(model.predict(sample_X), dtype=np.float64)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"non-finite LightGBM prediction seed={seed}")
        lgb_sample_predictions.append(prediction)
        artifacts.append({"family": "lightgbm", "seed": seed, "path": relative.as_posix(), "sha256": sha256_file(destination), "bytes": destination.stat().st_size})

    max_bin = int(xgb_cfg.get("max_bin", 256))
    dtrain = xgb.QuantileDMatrix(X, label=y, max_bin=max_bin, nthread=int(args.threads), feature_names=features)
    dsample = xgb.DMatrix(sample_X, feature_names=features, nthread=int(args.threads))
    for seed in seeds:
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "device": "cuda",
            "tree_method": "hist",
            "scale_pos_weight": (negatives / positives) * xgb_multiplier,
            "seed": seed,
            "verbosity": 0,
            "nthread": int(args.threads),
        }
        params.update(xgb_cfg)
        model = xgb.train(params, dtrain, num_boost_round=xgb_rounds, verbose_eval=False)
        relative = Path("models") / f"xgboost_seed_{seed}.ubj"
        destination = output / relative
        save_native_model(model, destination, f"xgb_{seed}")
        prediction = np.asarray(model.predict(dsample), dtype=np.float64)
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"non-finite XGBoost prediction seed={seed}")
        xgb_sample_predictions.append(prediction)
        artifacts.append({"family": "xgboost", "seed": seed, "path": relative.as_posix(), "sha256": sha256_file(destination), "bytes": destination.stat().st_size})

    lgb_mean = np.vstack(lgb_sample_predictions).mean(axis=0)
    xgb_mean = np.vstack(xgb_sample_predictions).mean(axis=0)
    blended = (1.0 - xgb_weight) * lgb_mean + xgb_weight * xgb_mean
    if not np.isfinite(blended).all() or np.any((blended < 0.0) | (blended > 1.0)):
        raise RuntimeError("invalid final ensemble sample predictions")

    deployment_manifest = {
        "schema": "crashwatch_normal_market_3d5_deployment_v1",
        "release_version": "1.0.0",
        "model_id": "C0_P2_ALLTRAIN_P2_DEDUP_CLEAN_3D5_NORMAL_ONLY",
        "score_semantics": "ranking risk score; not a calibrated probability",
        "target": freeze["target"],
        "training": {
            "policy": "all development market regimes; target-valid rows only; frozen after model selection",
            "dataset_signature": freeze["dataset_signature"],
            "date_min": runtime["date_min"],
            "date_max": runtime["date_max"],
            "rows": int(len(y)),
            "positives": positives,
            "positive_rate": float(y.mean()),
        },
        "inference": {
            "market_policy": "NORMAL_CORE_V1",
            "active_regimes": freeze["market_gate"]["active_regimes"],
            "otherwise": "ABSTAIN_ABNORMAL_MARKET",
            "market_proxy_feature": runtime["market_feature"],
            "minimum_history_dates": 252,
            "daily_alert_policy": "top 3 percent among scored rows, minimum one row",
            "sector_overlay_enabled": False,
        },
        "features": {"profile": candidate["profile"], "count": len(features), "sha256": feature_hash(features), "path": "config/feature_list.json"},
        "ensemble": {"seed_aggregation": "mean", "lightgbm_weight": 1.0 - xgb_weight, "xgboost_weight": xgb_weight, "seeds": seeds},
        "artifacts": artifacts,
        "runtime_versions": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__, "lightgbm": lgb.__version__, "xgboost": xgb.__version__},
        "build": {"elapsed_seconds": time.time() - started, "sample_rows_checked": int(len(sample_idx)), "sample_score_min": float(blended.min()), "sample_score_mean": float(blended.mean()), "sample_score_max": float(blended.max())},
    }
    atomic_json(features, config_dir / "feature_list.json")
    atomic_json(deployment_manifest, config_dir / "deployment_manifest.json")
    atomic_json(freeze, config_dir / "MODEL_FREEZE_MANIFEST.json")

    evidence_names = [
        "FINAL_RECOMMENDATION.json",
        "TARGET_3D5_AUDIT.json",
        "NORMAL_GATE_AUDIT.json",
        "candidate_comparison.csv",
        "blend_selection.csv",
        "ticker_metrics_normal.csv",
        "sector_metrics_normal.csv",
        "recent_normal_market_audit.csv",
        "permutation_sanity.json",
        "winner_curse_normal_market.csv",
    ]
    for name in evidence_names:
        source = validation_dir / name
        if source.exists():
            shutil.copy2(source, evidence_dir / name)
    copy_release_sources(release_source, output)

    write_checksums(output)
    print(json.dumps(deployment_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
