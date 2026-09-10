from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from cwregime.router_followup import (
    TrainOnlyCalibrator,
    apply_calibration_artifact,
    fit_conservative_pair_route,
)


def fixture() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    rows = []
    for regime, better in (("BULL_LOW_VOL", "P2_XGB"), ("BEAR_HIGH_VOL", "P7_LGB")):
        for day in range(20):
            for ticker in range(30):
                y = int(rng.random() < 0.15)
                p2_edge = 0.35 if better == "P2_XGB" else 0.03
                p7_edge = 0.35 if better == "P7_LGB" else 0.03
                rows.append({
                    "date": f"2026-{1 if regime == 'BULL_LOW_VOL' else 2:02d}-{day + 1:02d}",
                    "ticker": f"{ticker:06d}",
                    "regime": regime,
                    "target": y,
                    "P2_XGB": np.clip(0.35 + p2_edge * y + rng.normal(0, 0.08), 0.001, 0.999),
                    "P7_LGB": np.clip(0.45 + p7_edge * y + rng.normal(0, 0.08), 0.001, 0.999),
                })
    return pd.DataFrame(rows)


def main() -> None:
    frame = fixture()
    route = fit_conservative_pair_route(frame, shrink_rows=0, p7_margin=0.0)
    assert route.mapping["BULL_LOW_VOL"] == "P2_XGB"
    assert route.mapping["BEAR_HIGH_VOL"] == "P7_LGB"
    assert route.mapping["CRASH_STRESS"] == "P2_XGB"

    raw = np.clip(0.30 + 0.55 * frame["target"].to_numpy() + np.random.default_rng(1).normal(0, 0.05, len(frame)), 0.01, 0.99)
    calibrator = TrainOnlyCalibrator.fit(
        raw,
        frame["target"].to_numpy(),
        frame["regime"],
        method="platt_regime",
        c_value=1.0,
    )
    calibrated = calibrator.predict(raw, frame["regime"])
    restored = apply_calibration_artifact(raw, frame["regime"], calibrator.to_artifact())
    assert np.allclose(calibrated, restored, atol=1e-10)
    assert brier_score_loss(frame["target"], calibrated) < brier_score_loss(frame["target"], raw)
    identity = TrainOnlyCalibrator.fit(raw, frame["target"], frame["regime"], method="none", c_value=1.0)
    assert np.allclose(identity.predict(raw, frame["regime"]), raw)
    print("regime router follow-up tests: PASS")


if __name__ == "__main__":
    main()
