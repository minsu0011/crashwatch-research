from __future__ import annotations

import numpy as np
import pandas as pd

from cwregime.gating import SUBMODELS, apply_gate, fit_gate, model_metrics


def fixture() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows = []
    regimes = ["BULL_LOW_VOL", "BEAR_HIGH_VOL"]
    for block in ("B1", "B2"):
        for day in range(12):
            for ticker in range(40):
                regime = regimes[day % 2]
                y = int(rng.random() < (0.08 if regime == "BULL_LOW_VOL" else 0.18))
                row = {
                    "date": f"2025-01-{day + 1:02d}-{block}",
                    "ticker": f"{ticker:06d}",
                    "target": y,
                    "regime": regime,
                    "temporal_block": block,
                }
                for model in SUBMODELS:
                    edge = 0.28 if (regime == "BULL_LOW_VOL" and model == "P2_LGB") or (regime == "BEAR_HIGH_VOL" and model == "P7_XGB") else 0.04
                    row[model] = np.clip(0.05 + edge * y + rng.normal(0, 0.08), 0.001, 0.999)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    frame = fixture()
    gate = fit_gate(frame, method="hard", shrink_rows=0, n_clusters=2)
    assert gate.mapping["BULL_LOW_VOL"] == "P2_LGB"
    assert gate.mapping["BEAR_HIGH_VOL"] == "P7_XGB"
    out = apply_gate(frame, gate)
    assert out["prediction"].between(0, 1).all()
    assert set(out["selected_submodel"]) == {"P2_LGB", "P7_XGB"}
    one_class = frame.assign(target=0)
    metrics = model_metrics(one_class, "P2_LGB")
    assert np.isnan(metrics["pr_auc"])
    assert np.isnan(metrics["roc_auc"])
    assert np.isfinite(metrics["brier"])
    alerts = model_metrics(frame, "P2_LGB")
    assert alerts["alert_count"] >= frame["date"].nunique()
    clustered = fit_gate(frame, method="cluster", shrink_rows=200, n_clusters=2)
    assert set(clustered.mapping) >= {"BULL_LOW_VOL", "BEAR_HIGH_VOL"}
    print("regime submodel gate synthetic tests: PASS")


if __name__ == "__main__":
    main()

