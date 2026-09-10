from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["CRASHWATCH_BACKEND_MODE"] = "cpu_only"


def attempt(name, function):
    try:
        return {"component": name, "status": "ok", "detail": function()}
    except Exception as exc:
        return {"component": name, "status": "failed", "detail": str(exc), "traceback": traceback.format_exc()[-3000:]}


def main() -> None:
    rng = np.random.default_rng(17)
    x = rng.normal(size=(1000, 48)).astype(np.float32)
    y = (x[:, 0] + 0.4 * x[:, 1] + rng.normal(size=1000) > 0).astype(np.int8)

    def xgb():
        from xgboost import XGBClassifier, __version__
        model = XGBClassifier(n_estimators=30, max_depth=3, tree_method="hist", device="cpu", n_jobs=4, verbosity=0)
        model.fit(x, y)
        return {"version": __version__, "prediction_mean": float(model.predict_proba(x[:20])[:, 1].mean())}

    def lgbm():
        from lightgbm import LGBMClassifier, __version__
        model = LGBMClassifier(n_estimators=30, num_leaves=15, n_jobs=4, verbosity=-1)
        model.fit(x, y)
        return {"version": __version__, "prediction_mean": float(model.predict_proba(x[:20])[:, 1].mean())}

    def cat():
        from catboost import CatBoostClassifier, __version__
        model = CatBoostClassifier(iterations=30, depth=4, task_type="CPU", thread_count=4, verbose=False)
        model.fit(x, y)
        return {"version": __version__, "prediction_mean": float(model.predict_proba(x[:20])[:, 1].mean())}

    rows = [attempt("xgboost_cpu", xgb), attempt("lightgbm_cpu", lgbm), attempt("catboost_cpu", cat)]
    payload = {"success": all(r["status"] == "ok" for r in rows), "checks": rows}
    path = Path(__file__).resolve().parent / "cpu_stack_check.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()
