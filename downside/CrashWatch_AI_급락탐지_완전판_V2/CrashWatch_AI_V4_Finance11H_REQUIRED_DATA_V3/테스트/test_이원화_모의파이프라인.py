from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dual_ablation.experiment.metrics import add_daily_alerts, safe_metrics
from dual_ablation.features.catalogs import resolve_catalog


def test_daily_top3_always_selects_at_least_one():
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-02")] * 2,
        "ticker": ["000001", "000002"],
        "target": [0, 1],
        "prediction": [.1, .2],
    })
    out = add_daily_alerts(frame, .03)
    assert out["alert"].sum() == 1
    assert out.loc[out["alert"].eq(1), "ticker"].iloc[0] == "000002"


def test_single_class_auc_is_nan_but_calibration_exists():
    frame = pd.DataFrame({
        "date": pd.bdate_range("2026-01-01", periods=5),
        "target": [0, 0, 0, 0, 0],
        "prediction": [.1, .2, .1, .05, .2],
        "alert": [0, 1, 0, 0, 0],
    })
    result = safe_metrics(frame)
    assert np.isnan(result["pr_auc"])
    assert np.isnan(result["roc_auc"])
    assert np.isfinite(result["brier"])
    assert np.isfinite(result["logloss"])


def test_catalog_namespaces_do_not_overlap():
    columns = ["date", "u_market_x", "u_flow_y", "t_price_x", "t_event_y"]
    u = resolve_catalog(columns, {"u_market": ["u_market_"], "u_flow": ["u_flow_"]})
    t = resolve_catalog(columns, {"t_price": ["t_price_"], "t_event": ["t_event_"]})
    assert not set(sum(u.values(), [])).intersection(sum(t.values(), []))
