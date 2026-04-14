"""Tests for evaluation/decomposition.py."""

import numpy as np
import pandas as pd
import pytest

from macrocast.evaluation.decomposition import decompose_treatment_effects


def _make_result_df(seed: int = 0) -> pd.DataFrame:
    """Synthetic result table with AR benchmark and a few nonlinear models."""
    rng = np.random.default_rng(seed)
    T = 40
    dates = pd.date_range("2010-01", periods=T, freq="MS")
    y_true = rng.standard_normal(T)

    rows = []
    configs = [
        # (model_id, nonlinearity, regularization, cv_scheme, loss_function, noise_std)
        ("linear__none__bic__l2",       "linear", "none",   "_BICScheme()",  "l2",   0.5),
        ("linear__factors__bic__l2",    "linear", "factors","_BICScheme()",  "l2",   0.4),
        ("krr__factors__kfold5__l2",    "krr",    "factors","_KFoldCV(k=5)", "l2",   0.3),
        ("krr__factors__kfold5__epsi",  "krr",    "factors","_KFoldCV(k=5)", "epsilon_insensitive", 0.35),
        ("rf__none__kfold5__l2",        "random_forest", "none", "_KFoldCV(k=5)", "l2", 0.45),
    ]
    for model_id, nonlin, reg, cv, loss, noise in configs:
        y_hat = y_true + rng.standard_normal(T) * noise
        for i, d in enumerate(dates):
            rows.append({
                "model_id":       model_id,
                "nonlinearity":   nonlin,
                "regularization": reg,
                "cv_scheme":      cv,
                "loss_function":  loss,
                "horizon":        1,
                "forecast_date":  d,
                "y_hat":          y_hat[i],
                "y_true":         y_true[i],
            })
    return pd.DataFrame(rows)


def test_decompose_returns_four_components():
    df = _make_result_df()
    res = decompose_treatment_effects(df, benchmark_model_id="linear__none__bic__l2")
    assert "d_nonlinear" in res.coef
    assert "d_data_rich" in res.coef
    assert "d_kfold"     in res.coef
    assert "d_l2"        in res.coef


def test_decompose_summary_df_shape():
    df = _make_result_df()
    res = decompose_treatment_effects(df)
    assert res.summary_df.shape[0] == 5  # intercept + 4 components
    assert "coef" in res.summary_df.columns


def test_decompose_raises_missing_benchmark():
    df = _make_result_df()
    with pytest.raises(ValueError, match="Benchmark"):
        decompose_treatment_effects(df, benchmark_model_id="nonexistent_model")


def test_decompose_r_squared_in_unit_interval():
    df = _make_result_df(seed=1)
    res = decompose_treatment_effects(df)
    assert 0 <= res.r_squared <= 1 or np.isnan(res.r_squared)
