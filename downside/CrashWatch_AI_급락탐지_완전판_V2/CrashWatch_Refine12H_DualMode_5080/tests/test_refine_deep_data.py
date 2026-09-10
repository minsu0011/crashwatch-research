from __future__ import annotations

import numpy as np

from dual_ablation.refine12h.deep_models import DeepPreprocessor, build_window_indices


def test_windows_never_cross_tickers_or_future() -> None:
    tickers = np.array(["A"] * 5 + ["B"] * 5)
    dates = np.array(list(range(5)) + list(range(5)))
    windows = build_window_indices(tickers, dates, 3)
    for row in range(len(tickers)):
        if windows[row, 0] < 0:
            continue
        selected = windows[row]
        assert np.all(tickers[selected] == tickers[row])
        assert selected[-1] == row
        assert np.all(dates[selected] <= dates[row])


def test_preprocessor_selects_feature_axis_for_3d_windows() -> None:
    matrix = np.arange(40, dtype=np.float32).reshape(10, 4)
    windows = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    pre = DeepPreprocessor.fit(matrix, np.arange(8), np.array([1, 3], dtype=np.int32))
    output = pre.transform_windows(matrix, windows)
    assert output.shape == (2, 3, 3)  # 2 selected features + missing-ratio channel
