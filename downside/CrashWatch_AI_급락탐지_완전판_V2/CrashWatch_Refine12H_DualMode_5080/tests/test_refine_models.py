from __future__ import annotations

import numpy as np
import pytest

from dual_ablation.refine12h.models import TREE_CONFIGS, fit_tree_model, release_tree


@pytest.mark.parametrize("family", ["xgboost", "lightgbm", "catboost"])
def test_tree_family_cpu_smoke(family: str) -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 12)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(scale=0.8, size=300) > 0).astype(np.int8)
    model = fit_tree_model(
        family, TREE_CONFIGS[family][0], x[:240], y[:240], seed=17,
        requested_backend="cpu", threads=2, eval_set=(x[240:], y[240:]), force_backend="cpu",
    )
    prediction = model.predict(x[240:])
    assert prediction.shape == (60,)
    assert np.all((prediction >= 0) & (prediction <= 1))
    release_tree(model)
