"""Tests for the R model bridge: macrocast/pipeline/r_models.py.

Tests are organised by dependency:

1. _write_feather round-trip   — pure Python, always runs
2. RModelEstimator._call_r     — requires Rscript; skipped when absent
3. Concrete model wrappers     — parametrised, requires Rscript
4. ARModel special interface   — requires Rscript
5. ForecastExperiment integration — requires Rscript
6. Graceful failure when R missing — monkeypatched
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macrocast.pipeline.r_models import (
    ARDIModel,
    ARModel,
    AdaptiveLassoModel,
    BoogingModel,
    ElasticNetModel,
    GroupLassoModel,
    LassoModel,
    RidgeModel,
    TVPRidgeModel,
    _write_feather,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

R_AVAILABLE: bool = shutil.which("Rscript") is not None

requires_r = pytest.mark.skipif(not R_AVAILABLE, reason="Rscript not found on PATH")


def _r_package_available(pkg: str) -> bool:
    """Return True if an R package is installed on the system."""
    if not R_AVAILABLE:
        return False
    result = subprocess.run(
        ["Rscript", "-e", f"cat(requireNamespace('{pkg}', quietly=TRUE))"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout.strip().upper() == "TRUE"


GRPREG_AVAILABLE: bool = _r_package_available("grpreg")


def _synthetic_data(
    T: int = 60, N: int = 10, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Small synthetic (X, y) for unit tests."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, N))
    y = X[:, 0] + 0.3 * X[:, 1] + rng.standard_normal(T) * 0.1
    return X.astype(float), y.astype(float)


# ---------------------------------------------------------------------------
# 1. _write_feather round-trip (always runs)
# ---------------------------------------------------------------------------


class TestWriteFeather:
    def test_2d_round_trip(self) -> None:
        """Write a 2-D array and read it back; values must match."""
        import pyarrow.feather as pf

        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.feather"
            _write_feather(arr, path)
            table = pf.read_table(str(path))
            recovered = np.column_stack([table[c].to_pylist() for c in table.column_names])
        np.testing.assert_allclose(recovered, arr)

    def test_1d_reshaped(self) -> None:
        """1-D input should be reshaped to (T, 1) and round-trip correctly."""
        import pyarrow.feather as pf

        arr = np.array([1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test1d.feather"
            _write_feather(arr, path)
            table = pf.read_table(str(path))
            assert len(table.column_names) == 1
            recovered = np.array(table[table.column_names[0]].to_pylist())
        np.testing.assert_allclose(recovered, arr)

    def test_large_array(self) -> None:
        """Smoke test: write/read a (200, 130) array."""
        import pyarrow.feather as pf

        arr = np.random.default_rng(0).standard_normal((200, 130))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.feather"
            _write_feather(arr, path)
            table = pf.read_table(str(path))
            recovered = np.column_stack(
                [table[c].to_pylist() for c in table.column_names]
            )
        np.testing.assert_allclose(recovered, arr, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. R bridge — basic smoke tests
# ---------------------------------------------------------------------------


@requires_r
class TestRBridge:
    def test_ridge_returns_scalar(self) -> None:
        """RidgeModel.predict returns shape (1,) finite scalar."""
        X, y = _synthetic_data()
        model = RidgeModel(cv_folds=5, nlambda=10)
        model.fit(X, y)
        y_hat = model.predict(X[:1])
        assert y_hat.shape == (1,)
        assert np.isfinite(y_hat[0])

    def test_lasso_stores_hp(self) -> None:
        """LassoModel stores selected lambda in best_params_ after predict."""
        X, y = _synthetic_data()
        model = LassoModel(cv_folds=5, nlambda=10)
        model.fit(X, y)
        model.predict(X[:1])
        assert "lambda" in model.best_params_
        assert model.best_params_["lambda"] > 0

    def test_bridge_raises_on_unknown_model(self) -> None:
        """An unrecognised model name causes RuntimeError from the bridge."""
        from macrocast.pipeline.r_models import RModelEstimator

        model = RModelEstimator("not_a_real_model")
        X, y = _synthetic_data()
        model.fit(X, y)
        with pytest.raises(RuntimeError, match="R bridge failed"):
            model.predict(X[:1])


# ---------------------------------------------------------------------------
# 3. All concrete model classes (parametrised)
# ---------------------------------------------------------------------------

_STANDARD_MODELS = [
    RidgeModel(cv_folds=5, nlambda=10),
    LassoModel(cv_folds=5, nlambda=10),
    AdaptiveLassoModel(cv_folds=5, nlambda=10),
    ElasticNetModel(cv_folds=5, nlambda=10, alpha=0.5),
    ARDIModel(intercept=True),
    TVPRidgeModel(n_poly=2, cv_folds=5),
    BoogingModel(n_boot=20, prune_quantile=0.5),
]


@requires_r
@pytest.mark.parametrize(
    "model",
    _STANDARD_MODELS,
    ids=[type(m).__name__ for m in _STANDARD_MODELS],
)
def test_model_produces_finite_forecast(model: object) -> None:
    """Each concrete model returns a finite scalar forecast."""
    X, y = _synthetic_data()
    model.fit(X, y)  # type: ignore[attr-defined]
    y_hat = model.predict(X[:1])  # type: ignore[attr-defined]
    assert y_hat.shape == (1,)
    assert np.isfinite(y_hat[0])


@requires_r
@pytest.mark.skipif(not GRPREG_AVAILABLE, reason="R package 'grpreg' not installed")
def test_group_lasso_with_explicit_groups() -> None:
    """GroupLassoModel with explicit group vector produces finite forecast."""
    X, y = _synthetic_data(N=8)
    # 8 features → 4 groups of 2
    groups = [1, 1, 2, 2, 3, 3, 4, 4]
    model = GroupLassoModel(groups=groups, cv_folds=5)
    model.fit(X, y)
    y_hat = model.predict(X[:1])
    assert y_hat.shape == (1,)
    assert np.isfinite(y_hat[0])


# ---------------------------------------------------------------------------
# 4. ARModel special interface
# ---------------------------------------------------------------------------


@requires_r
class TestARModel:
    def test_ar_with_full_series(self) -> None:
        """ARModel returns finite forecast when _y_train_full is provided."""
        rng = np.random.default_rng(0)
        T = 60
        y_full = rng.standard_normal(T)
        X = rng.standard_normal((T, 5))

        model = ARModel(h=1, max_lag=6)
        model._y_train_full = y_full
        model._y_test_lags = y_full[-6:]
        model.fit(X, y_full[1:])  # h-shifted target
        y_hat = model.predict(X[:1])
        assert y_hat.shape == (1,)
        assert np.isfinite(y_hat[0])

    def test_ar_stores_selected_lag(self) -> None:
        """ARModel stores the BIC-selected lag order in best_params_."""
        rng = np.random.default_rng(1)
        T = 60
        y_full = rng.standard_normal(T)
        X = rng.standard_normal((T, 5))

        model = ARModel(h=1, max_lag=6)
        model._y_train_full = y_full
        model._y_test_lags = y_full[-6:]
        model.fit(X, y_full[1:])
        model.predict(X[:1])
        assert "p" in model.best_params_
        assert 1 <= model.best_params_["p"] <= 6


# ---------------------------------------------------------------------------
# 5. ForecastExperiment integration
# ---------------------------------------------------------------------------


@requires_r
def test_ridge_in_forecast_experiment() -> None:
    """RidgeModel runs through ForecastExperiment end-to-end."""
    from macrocast.pipeline.components import CVScheme, LossFunction, Regularization
    from macrocast.pipeline.experiment import FeatureSpec, ForecastExperiment, ModelSpec

    rng = np.random.default_rng(10)
    # 120 months: 2005-01 through 2014-12; oos window within this range
    dates = pd.date_range("2005-01", periods=120, freq="MS")
    X = rng.standard_normal((120, 8))
    y = X[:, 0] + rng.standard_normal(120) * 0.1
    panel = pd.DataFrame(X, index=dates, columns=[f"x{i}" for i in range(8)])
    target = pd.Series(y, index=dates, name="target")

    ridge_spec = ModelSpec(
        model_cls=RidgeModel,
        regularization=Regularization.RIDGE,
        cv_scheme=CVScheme.KFOLD(k=5),
        loss_function=LossFunction.L2,
        model_kwargs={"cv_folds": 5, "nlambda": 10},
    )
    exp = ForecastExperiment(
        panel=panel,
        target=target,
        horizons=[1],
        model_specs=[ridge_spec],
        feature_spec=FeatureSpec(n_factors=2, n_lags=2, use_factors=True),
        oos_start="2013-01-01",
        oos_end="2013-03-01",
        n_jobs=1,
    )
    rs = exp.run()
    assert len(rs) > 0, "Expected at least one forecast record"
    assert all(np.isfinite(r.y_hat) for r in rs.records)


# ---------------------------------------------------------------------------
# 6. Graceful failure when R is not available
# ---------------------------------------------------------------------------


def test_r_bridge_raises_when_r_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """predict() raises RuntimeError when Rscript is not on PATH."""
    monkeypatch.setattr(shutil, "which", lambda _: None)

    X, y = _synthetic_data()
    model = RidgeModel(cv_folds=2)
    model.fit(X, y)

    with pytest.raises(RuntimeError, match="Rscript not found"):
        model.predict(X[:1])


def test_fit_before_predict_raises() -> None:
    """predict() without prior fit() raises RuntimeError."""
    model = RidgeModel()
    X, _ = _synthetic_data()

    with pytest.raises(RuntimeError, match="Call fit\\(\\) before predict"):
        model.predict(X[:1])
