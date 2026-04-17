"""Sanity bound: each deep family stays within 5x of AR(2) val-MSE on AR(2) data.

This catches structurally broken architectures (e.g., everything constant)
without claiming deep models beat AR(2) on synthetic data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytestmark = pytest.mark.deep

from statsmodels.tsa.ar_model import AutoReg

from macrocast.execution.adapters.sequence import reshape_for_sequence
from macrocast.execution.models.deep._base import DeepModelConfig
from macrocast.execution.models.deep.gru import GRUModel
from macrocast.execution.models.deep.lstm import LSTMModel
from macrocast.execution.models.deep.tcn import TCNModel


def _synthetic_ar2(n: int = 200, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=float)
    for i in range(2, n):
        x[i] = 0.6 * x[i - 1] - 0.2 * x[i - 2] + rng.standard_normal() * 0.3
    return x


def _ar2_rolling_val_mse(series: np.ndarray, val_start: int, val_end: int) -> float:
    errs = []
    for i in range(val_start, val_end):
        s = pd.Series(series[:i])
        fit = AutoReg(s, lags=2, trend="c", old_names=False).fit()
        pred = float(fit.predict(start=i, end=i).iloc[0])
        errs.append((pred - series[i]) ** 2)
    return float(np.mean(errs))


@pytest.mark.parametrize("Cls", [LSTMModel, GRUModel, TCNModel])
def test_deep_within_5x_of_ar2_on_ar2_signal(Cls):
    series = _synthetic_ar2(n=200, seed=1)
    lookback = 12
    cfg = DeepModelConfig(seed=11, max_epochs=20, lookback=lookback)
    X, y = reshape_for_sequence(series=series, lookback=lookback, horizon=1)

    # Hold out the last 40 points for rolling one-step val.
    n_val = 40
    train_end = len(series) - n_val
    X_train = X[: train_end - lookback]
    y_train = y[: train_end - lookback]

    model = Cls(config=cfg).fit(X_train, y_train)

    val_errs = []
    for i in range(train_end, len(series)):
        window = series[i - lookback : i]
        yhat = model.predict_next(window)
        val_errs.append((yhat - series[i]) ** 2)
    deep_val_mse = float(np.mean(val_errs))

    ar_val_mse = _ar2_rolling_val_mse(series, train_end, len(series))
    assert deep_val_mse < 5.0 * ar_val_mse, (
        f"{Cls.__name__}: deep val MSE {deep_val_mse:.4f} exceeds 5x AR(2) MSE {ar_val_mse:.4f}"
    )
