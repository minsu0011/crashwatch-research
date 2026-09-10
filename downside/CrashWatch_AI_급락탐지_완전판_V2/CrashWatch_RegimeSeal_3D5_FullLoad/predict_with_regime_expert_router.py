from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from cw7h.utils import atomic_json
from cwfull.common import nvml_snapshot, set_full_load_mode
from cwregime.expert_router import RegimeExpertRouter


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(handle)
    try:
        frame.to_parquet(temp_name, engine="pyarrow", index=False, compression="zstd")
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def run(args: argparse.Namespace) -> dict:
    artifact = Path(args.artifact).expanduser().resolve()
    feature_path = Path(args.features).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {output}")
    set_full_load_mode(int(args.threads), "high")
    router = RegimeExpertRouter(
        artifact,
        total_threads=int(args.threads),
        use_gpu=not args.cpu_only,
        verify_hashes=not args.skip_hash_check,
    )
    schema = set(pq.ParquetFile(feature_path).schema_arrow.names)
    identifiers = [name for name in ("row_id", "date", "ticker") if name in schema]
    if "date" not in identifiers or "ticker" not in identifiers:
        raise ValueError("feature parquet must contain date and ticker")
    required = router.required_features()
    missing = sorted(set(required) - schema)
    if missing:
        raise ValueError(f"feature parquet is missing {len(missing)} required features: {missing[:10]}")
    read_columns = [*identifiers, *required]
    if "regime" in schema:
        read_columns.append("regime")
    # Targets may exist in the source parquet, but are deliberately not read.
    frame = pd.read_parquet(feature_path, columns=list(dict.fromkeys(read_columns)))
    calendar = None
    if "regime" not in frame.columns:
        if not args.regime_calendar:
            raise ValueError("pass --regime-calendar when the feature parquet has no regime column")
        calendar = pd.read_csv(args.regime_calendar, usecols=["date", "regime"])
    started = time.time()
    prediction = router.predict_frame(frame, regime_calendar=calendar, alert_fraction=float(args.alert_fraction))
    atomic_parquet(prediction, output)
    summary = {
        "status": "COMPLETE",
        "artifact": str(artifact),
        "router_hash": router.artifact["router_hash"],
        "input": str(feature_path),
        "output": str(output),
        "rows": int(len(prediction)),
        "dates": int(prediction["date"].nunique()),
        "tickers": int(prediction["ticker"].nunique()),
        "alerts": int(prediction["alert_top_3pct"].sum()),
        "elapsed_seconds": float(time.time() - started),
        "target_columns_read": False,
        "diagnostics": router.last_diagnostics,
        "gpu_after": nvml_snapshot(),
    }
    atomic_json(summary, output.with_suffix(output.suffix + ".summary.json"))
    return summary


def parse_args() -> argparse.Namespace:
    package = Path(__file__).resolve().parent
    project = package.parent
    parser = argparse.ArgumentParser(description="Predict with the frozen P2/P7 market-regime expert router")
    parser.add_argument("--features", required=True, help="Feature parquet containing date/ticker and frozen features")
    parser.add_argument("--regime-calendar", help="CSV containing date,regime when features have no regime column")
    parser.add_argument(
        "--artifact",
        default=str(project / "crashwatch_ai_data/regime_expert_router_v2/REGIME_EXPERT_ROUTER_V2.json"),
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--alert-fraction", type=float, default=0.03)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))
