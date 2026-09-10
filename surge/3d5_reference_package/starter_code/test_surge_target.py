from __future__ import annotations

import numpy as np
import pandas as pd

from build_surge_target import TARGET_NAME, build_target


def main() -> None:
    dates = pd.date_range("2026-01-01", periods=6, freq="B")
    frame = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "ticker": ["005930"] * 6 + ["000660"] * 6,
            "t_price_ret_1": [0.0, 0.03, 0.02, 0.0, 0.0, 0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
        }
    )
    target, audit = build_target(frame, horizon=3, threshold=0.05)
    assert bool(target.loc[0, TARGET_NAME]) is True
    assert int(target.loc[0, "first_hit_day"]) == 2
    assert bool(target.loc[6, TARGET_NAME]) is True
    assert int(target.loc[6, "first_hit_day"]) == 1
    assert int(target["target_valid"].sum()) == 6
    assert not target.loc[[3, 4, 5, 9, 10, 11], "target_valid"].any()
    assert audit["positives"] >= 2
    assert np.isfinite(target.loc[target["target_valid"], "best_forward_return_3d"]).all()
    print("SURGE_TARGET_TEST_PASS")


if __name__ == "__main__":
    main()
