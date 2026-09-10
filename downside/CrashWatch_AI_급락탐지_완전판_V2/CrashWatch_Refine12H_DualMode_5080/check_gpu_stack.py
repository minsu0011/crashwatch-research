from __future__ import annotations

import json
import traceback
from pathlib import Path

import numpy as np


def attempt(name, function):
    try:
        detail = function()
        return {"component": name, "status": "ok", "detail": detail}
    except Exception as exc:
        return {"component": name, "status": "failed", "detail": str(exc), "traceback": traceback.format_exc()[-3000:]}


def main() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(512, 24)).astype(np.float32)
    y = (x[:, 0] + rng.normal(size=512) > 0).astype(np.int8)

    def torch_check():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda.is_available() is False")
        device = torch.cuda.get_device_properties(0)
        a = torch.randn(2048, 2048, device="cuda")
        b = a @ a.T
        torch.cuda.synchronize()
        del a, b
        return {"version": torch.__version__, "cuda": torch.version.cuda, "gpu": device.name, "vram_gb": device.total_memory / 1024**3}

    def xgb_check():
        from xgboost import XGBClassifier, __version__
        model = XGBClassifier(n_estimators=20, max_depth=4, tree_method="hist", device="cuda", n_jobs=4)
        model.fit(x, y)
        model.predict_proba(x[:10])
        config = json.loads(model.get_booster().save_config())
        actual_device = str(config.get("learner", {}).get("generic_param", {}).get("device", "unknown"))
        if not actual_device.startswith("cuda"):
            raise RuntimeError(f"XGBoost requested CUDA but actual device is {actual_device}")
        return {"version": __version__, "backend": actual_device}

    def cat_check():
        from catboost import CatBoostClassifier, __version__
        model = CatBoostClassifier(iterations=20, depth=6, task_type="GPU", devices="0", verbose=False, allow_writing_files=False)
        model.fit(x, y)
        model.predict_proba(x[:10])
        return {"version": __version__, "backend": "cuda"}

    def lgbm_check():
        import lightgbm
        from lightgbm import LGBMClassifier
        model = LGBMClassifier(n_estimators=20, num_leaves=31, device_type="gpu", verbosity=-1)
        model.fit(x, y)
        model.predict_proba(x[:10])
        return {"version": lightgbm.__version__, "backend": "gpu_opencl"}

    rows = [
        attempt("pytorch", torch_check),
        attempt("xgboost", xgb_check),
        attempt("catboost", cat_check),
        attempt("lightgbm", lgbm_check),
    ]
    output = Path(__file__).resolve().parent / "gpu_stack_check.json"
    output.write_text(json.dumps({"checks": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checks": rows}, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
