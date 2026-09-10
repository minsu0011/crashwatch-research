from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cw7h.data import PreparedData
from cw7h.folds import FoldSlice
from cwfull.aggregate import direct_compare
from cwfull.lgb_engine import run_profiles
from cwfull.profiles import build_experiment_profiles


def main() -> int:
    package_root = Path(__file__).resolve().parent
    base_manifest = json.load((package_root / "seed_results" / "profile_manifest.json").open(encoding="utf-8"))
    all_features = list(base_manifest["profiles"]["P0_FULL_439"]["features"])
    with tempfile.TemporaryDirectory(prefix="cw-final-smoke-") as temp:
        root = Path(temp)
        manifest = build_experiment_profiles(package_root, all_features, root)
        assert manifest["base_profiles"]["P0_FULL_439"]["count"] == 439
        assert manifest["base_profiles"]["P2_DEDUP_CLEAN"]["count"] == 371
        assert manifest["base_profiles"]["P7_CORR095_PLUS_CONDITIONAL"]["count"] == 341
        assert len(manifest["factor_bits"]) == 32

        cache = root / "cache"
        cache.mkdir()
        rng = np.random.default_rng(17)
        rows, features = 480, 12
        X = rng.normal(size=(rows, features)).astype(np.float32)
        logits = 0.8 * X[:, 0] - 0.5 * X[:, 1] + 0.2 * rng.normal(size=rows)
        y = (logits > np.quantile(logits, 0.72)).astype(np.uint8)
        dates = np.repeat(np.arange(120, dtype=np.int64), 4) * 86_400_000_000_000
        np.save(cache / "X_all_valid.npy", X)
        np.save(cache / "target.npy", y)
        np.save(cache / "dates_ns.npy", dates)
        np.save(cache / "tickers.npy", np.array(["000001"] * rows))
        np.save(cache / "buckets.npy", np.array(["test"] * rows))
        np.save(cache / "original_row_id.npy", np.arange(rows))
        names = [f"f{i}" for i in range(features)]
        (cache / "feature_names.json").write_text(json.dumps(names), encoding="utf-8")
        prepared = PreparedData(cache, root / "synthetic.parquet", "synthetic-signature", {"rows": rows, "features": features}, names, {})
        folds = [
            FoldSlice(0, 0, 240, 280, 360, "", "", "", "", 60, 20, 240, 80, True, ""),
            FoldSlice(1, 0, 320, 360, 440, "", "", "", "", 80, 20, 320, 80, True, ""),
        ]
        profiles = {"A": names, "B": names[:-1]}
        config = {
            "learning_rate": 0.08, "num_leaves": 15, "max_depth": -1, "min_data_in_leaf": 10,
            "feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0,
            "lambda_l1": 0.0, "lambda_l2": 0.1, "max_bin": 63, "min_gain_to_split": 1e-12,
        }
        summary = run_profiles(
            prepared=prepared, folds=folds, profiles=profiles, seeds=[17, 43], best_iterations={0: 8, 1: 8},
            config=config, config_name="smoke", task_dir=root / "tasks", workers=2,
            threads_per_worker=1, min_free_ram_gb=0.1, priority="normal", retries=0,
        )
        assert summary["failed_last_pass"] == 0, summary
        assert summary["expected_model_count"] == 8, summary
        task_rows = []
        for file in (root / "tasks" / "smoke").glob("*.json"):
            payload = json.load(file.open(encoding="utf-8"))
            if payload.get("status") == "completed":
                payload["profile"] = "A" if payload["profile_hash"] == __import__("cw7h.utils", fromlist=["hash_strings"]).hash_strings(names) else "B"
                task_rows.append(payload)
        frame = pd.DataFrame(task_rows)
        comparison, paired = direct_compare(frame, "B", "A", [1])
        assert comparison and len(paired) == 4
    print("SMOKE_TEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
