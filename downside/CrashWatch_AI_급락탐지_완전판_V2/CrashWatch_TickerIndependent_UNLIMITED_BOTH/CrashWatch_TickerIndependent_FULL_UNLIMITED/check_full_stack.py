from __future__ import annotations

import json
import subprocess
import traceback
from pathlib import Path

import numpy as np


def attempt(name, function):
    try:
        return {"component": name, "status": "ok", "detail": function()}
    except Exception as exc:
        return {"component": name, "status": "failed", "detail": str(exc), "traceback": traceback.format_exc()[-3000:]}


def main() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(1500, 64)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] + rng.normal(size=1500) > 0).astype(np.int8)

    def gpu():
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"], text=True, timeout=8).strip().splitlines()[0]
        return out

    def xgb():
        from xgboost import XGBClassifier, __version__
        model = XGBClassifier(n_estimators=40, max_depth=4, tree_method="hist", device="cuda", n_jobs=3, verbosity=0)
        model.fit(x, y)
        return {"version": __version__, "prediction_mean": float(model.predict_proba(x[:20])[:, 1].mean())}

    def cat():
        from catboost import CatBoostClassifier, __version__
        model = CatBoostClassifier(iterations=40, depth=5, task_type="GPU", devices="0", thread_count=3, verbose=False)
        model.fit(x, y)
        return {"version": __version__, "prediction_mean": float(model.predict_proba(x[:20])[:, 1].mean())}

    def lgbm():
        from lightgbm import LGBMClassifier, __version__
        model = LGBMClassifier(n_estimators=40, num_leaves=15, n_jobs=4, verbosity=-1)
        model.fit(x, y)
        return {"version": __version__, "prediction_mean": float(model.predict_proba(x[:20])[:, 1].mean())}

    rows = [attempt("nvidia_smi", gpu), attempt("xgboost_cuda", xgb), attempt("catboost_cuda", cat), attempt("lightgbm_cpu", lgbm)]
    payload = {"success": all(r["status"] == "ok" for r in rows), "checks": rows}
    path = Path(__file__).resolve().parent / "full_stack_check.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()
