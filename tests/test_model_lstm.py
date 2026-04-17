"""Unit tests for LSTMModel (Phase 5)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.deep

from macrocast.execution.models.deep.lstm import LSTMModel
from macrocast.execution.models.deep._base import DeepModelConfig


def _synthetic_ar1(n: int = 100, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    for i in range(1, n):
        x[i] = 0.7 * x[i - 1] + rng.standard_normal() * 0.5
    return x


def _windows(series: np.ndarray, lookback: int, horizon: int):
    from macrocast.execution.adapters.sequence import reshape_for_sequence
    return reshape_for_sequence(series=series, lookback=lookback, horizon=horizon)


def test_fit_predict_smoke():
    series = _synthetic_ar1()
    X, y = _windows(series, 12, 1)
    cfg = DeepModelConfig(seed=42, max_epochs=5)
    model = LSTMModel(config=cfg).fit(X, y)
    yhat = model.predict_next(series[-12:])
    assert np.isfinite(yhat)
    assert isinstance(yhat, float)


def test_seed_reproducibility():
    series = _synthetic_ar1()
    X, y = _windows(series, 12, 1)
    cfg = DeepModelConfig(seed=7, max_epochs=3)
    yhat1 = LSTMModel(config=cfg).fit(X, y).predict_next(series[-12:])
    yhat2 = LSTMModel(config=cfg).fit(X, y).predict_next(series[-12:])
    assert yhat1 == yhat2


def test_seed_divergence():
    series = _synthetic_ar1()
    X, y = _windows(series, 12, 1)
    yhat_a = LSTMModel(config=DeepModelConfig(seed=1, max_epochs=3)).fit(X, y).predict_next(series[-12:])
    yhat_b = LSTMModel(config=DeepModelConfig(seed=2, max_epochs=3)).fit(X, y).predict_next(series[-12:])
    assert abs(yhat_a - yhat_b) > 1e-6
