"""Tests for macrocast.execution.adapters.sequence.

Pure-numpy test file — runs in every CI matrix row (no deep extra needed).
"""
from __future__ import annotations

import numpy as np
import pytest

from macrocast.execution.adapters.sequence import reshape_for_sequence


def test_shapes_horizon_1():
    series = np.arange(20, dtype=float)
    X, y = reshape_for_sequence(series=series, lookback=12, horizon=1)
    assert X.shape == (8, 12, 1)
    assert y.shape == (8,)


def test_shapes_horizon_3():
    series = np.arange(20, dtype=float)
    X, y = reshape_for_sequence(series=series, lookback=12, horizon=3)
    assert X.shape == (6, 12, 1)
    assert y.shape == (6,)


def test_target_alignment():
    series = np.arange(20, dtype=float)
    X, y = reshape_for_sequence(series=series, lookback=12, horizon=1)
    for i in range(len(y)):
        assert y[i] == series[i + 12 + 1 - 1], (i, y[i], series[i + 12])
        assert np.array_equal(X[i, :, 0], series[i : i + 12])


def test_target_alignment_h3():
    series = np.arange(20, dtype=float)
    X, y = reshape_for_sequence(series=series, lookback=12, horizon=3)
    for i in range(len(y)):
        assert y[i] == series[i + 12 + 3 - 1]


def test_too_short_raises():
    series = np.arange(12, dtype=float)
    with pytest.raises(ValueError, match="too short"):
        reshape_for_sequence(series=series, lookback=12, horizon=1)


def test_minimum_valid_window():
    """T == lookback + horizon produces exactly one window."""
    series = np.arange(13, dtype=float)
    X, y = reshape_for_sequence(series=series, lookback=12, horizon=1)
    assert X.shape == (1, 12, 1)
    assert y.shape == (1,)
    assert y[0] == series[12]


def test_rejects_2d_series():
    series = np.arange(20, dtype=float).reshape(4, 5)
    with pytest.raises(ValueError, match="1-D"):
        reshape_for_sequence(series=series, lookback=3, horizon=1)


def test_rejects_nonpositive_lookback():
    series = np.arange(20, dtype=float)
    with pytest.raises(ValueError, match="lookback"):
        reshape_for_sequence(series=series, lookback=0, horizon=1)


def test_rejects_nonpositive_horizon():
    series = np.arange(20, dtype=float)
    with pytest.raises(ValueError, match="horizon"):
        reshape_for_sequence(series=series, lookback=12, horizon=0)
