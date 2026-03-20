"""Tests for evaluation/mcs.py — Model Confidence Set."""

import numpy as np
import pandas as pd
import pytest

from macrocast.evaluation.mcs import mcs


def _make_loss_df(seed: int = 0) -> pd.DataFrame:
    """Synthetic loss table: 5 models, 60 dates, model 0 is best."""
    rng = np.random.default_rng(seed)
    T = 60
    dates = pd.date_range("2010-01", periods=T, freq="MS")
    models = ["ar", "ardi", "krr", "rf", "xgb"]
    rows = []
    # ar: smallest squared error
    base_errors = [0.5, 0.8, 1.0, 1.2, 1.5]
    for i, m in enumerate(models):
        se = (rng.standard_normal(T) + base_errors[i]) ** 2
        for t, d in enumerate(dates):
            rows.append({"model_id": m, "forecast_date": d, "squared_error": se[t]})
    return pd.DataFrame(rows)


def test_mcs_returns_result():
    df = _make_loss_df()
    res = mcs(df, alpha=0.10, block_size=4, n_bootstrap=100, seed=0)
    assert len(res.included) >= 1
    assert len(res.included) + len(res.excluded) == 5


def test_mcs_best_model_survives():
    """The best model (ar) should survive into the MCS."""
    df = _make_loss_df(seed=0)
    res = mcs(df, alpha=0.10, block_size=4, n_bootstrap=200, seed=0)
    assert "ar" in res.included


def test_mcs_p_values_keys():
    df = _make_loss_df()
    res = mcs(df, alpha=0.10, block_size=4, n_bootstrap=50, seed=1)
    assert set(res.p_values.keys()) == {"ar", "ardi", "krr", "rf", "xgb"}
