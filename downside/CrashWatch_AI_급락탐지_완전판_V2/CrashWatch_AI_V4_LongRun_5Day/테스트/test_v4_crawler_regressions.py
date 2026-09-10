from __future__ import annotations

import numpy as np
import pandas as pd

from dual_ablation.crawlers import etf_pressure


def test_etf_pressure_missing_trading_value_is_a_series(monkeypatch, tmp_path) -> None:
    dates = pd.bdate_range("2024-01-02", periods=30)
    monkeypatch.setattr(etf_pressure, "_pykrx_etf", lambda *args: pd.DataFrame({
        "date": dates, "close": np.arange(30, dtype=float) + 100, "volume": 1000,
    }))
    config = tmp_path / "configs"
    config.mkdir()
    pd.DataFrame([{"ticker": "069500", "alias": "kospi200", "role": "long", "enabled": 1}]).to_csv(
        config / "etf_pressure_universe.csv", index=False
    )
    from dual_ablation.config import ProjectPaths
    paths = ProjectPaths(
        project=tmp_path, data_root=tmp_path / "data", raw_dual=tmp_path / "data" / "raw",
        feature_dual=tmp_path / "data" / "features", result_dual=tmp_path / "data" / "results",
        cache_dual=tmp_path / "data" / "cache", configs=config,
    )
    paths.raw_dual.mkdir(parents=True)
    result = etf_pressure.collect_etf_pressure(paths, "2024-01-01", "2024-03-01")
    assert len(result) == 30
    assert "u_etf_kospi200_value_z20" in result.columns
