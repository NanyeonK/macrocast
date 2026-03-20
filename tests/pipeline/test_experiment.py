"""Integration tests for pipeline/experiment.py — ForecastExperiment.

Uses a small synthetic panel so tests run in seconds.
"""

import numpy as np
import pandas as pd
import pytest

from macrocast.pipeline.components import (
    CVScheme,
    LossFunction,
    Nonlinearity,
    Regularization,
    Window,
)
from macrocast.pipeline.experiment import FeatureSpec, ForecastExperiment, ModelSpec
from macrocast.pipeline.models import KRRModel, RFModel
from macrocast.pipeline.results import ResultSet


@pytest.fixture()
def synthetic_panel():
    """Monthly panel: 120 observations, 10 predictors."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2005-01", periods=120, freq="MS")
    X = rng.standard_normal((120, 10))
    y = X[:, 0] + 0.3 * X[:, 1] + rng.standard_normal(120) * 0.2
    panel = pd.DataFrame(X, index=dates, columns=[f"x{i}" for i in range(10)])
    target = pd.Series(y, index=dates, name="target")
    return panel, target


@pytest.fixture()
def krr_spec():
    return ModelSpec(
        model_cls=KRRModel,
        regularization=Regularization.FACTORS,
        cv_scheme=CVScheme.KFOLD(k=2),
        loss_function=LossFunction.L2,
        model_kwargs={"alpha_grid": [0.1, 1.0], "gamma_grid": [0.1], "cv_folds": 2},
    )


@pytest.fixture()
def rf_spec():
    return ModelSpec(
        model_cls=RFModel,
        regularization=Regularization.NONE,
        cv_scheme=CVScheme.KFOLD(k=2),
        loss_function=LossFunction.L2,
        model_kwargs={
            "n_estimators": 5,
            "max_depth_grid": [3],
            "min_samples_leaf_grid": [5],
            "cv_folds": 2,
        },
    )


class TestModelSpec:
    def test_auto_model_id(self, krr_spec):
        # model_id should be non-empty and contain component values
        assert "krr" in krr_spec.model_id
        assert "factors" in krr_spec.model_id

    def test_build_returns_new_instance(self, krr_spec):
        m1 = krr_spec.build()
        m2 = krr_spec.build()
        assert m1 is not m2


class TestForecastExperiment:
    def test_run_returns_result_set(self, synthetic_panel, krr_spec):
        panel, target = synthetic_panel
        exp = ForecastExperiment(
            panel=panel,
            target=target,
            horizons=[1],
            model_specs=[krr_spec],
            feature_spec=FeatureSpec(n_factors=2, n_lags=2, use_factors=True),
            window=Window.EXPANDING,
            oos_start="2014-01-01",
            oos_end="2014-03-01",
            n_jobs=1,
        )
        rs = exp.run()
        assert isinstance(rs, ResultSet)
        assert len(rs) > 0

    def test_multiple_horizons(self, synthetic_panel, krr_spec):
        panel, target = synthetic_panel
        exp = ForecastExperiment(
            panel=panel,
            target=target,
            horizons=[1, 3],
            model_specs=[krr_spec],
            feature_spec=FeatureSpec(n_factors=2, n_lags=2, use_factors=True),
            oos_start="2014-01-01",
            oos_end="2014-02-01",
            n_jobs=1,
        )
        rs = exp.run()
        horizons_found = set(r.horizon for r in rs.records)
        assert horizons_found == {1, 3}

    def test_multiple_models(self, synthetic_panel, krr_spec, rf_spec):
        panel, target = synthetic_panel
        exp = ForecastExperiment(
            panel=panel,
            target=target,
            horizons=[1],
            model_specs=[krr_spec, rf_spec],
            feature_spec=FeatureSpec(n_factors=2, n_lags=2, use_factors=True),
            oos_start="2014-01-01",
            oos_end="2014-02-01",
            n_jobs=1,
        )
        rs = exp.run()
        model_ids = {r.model_id for r in rs.records}
        assert len(model_ids) == 2

    def test_rolling_window_requires_size(self, synthetic_panel, krr_spec):
        panel, target = synthetic_panel
        with pytest.raises(ValueError, match="rolling_size"):
            ForecastExperiment(
                panel=panel,
                target=target,
                horizons=[1],
                model_specs=[krr_spec],
                window=Window.ROLLING,
                rolling_size=None,
            )

    def test_parquet_output(self, synthetic_panel, krr_spec, tmp_path):
        panel, target = synthetic_panel
        exp = ForecastExperiment(
            panel=panel,
            target=target,
            horizons=[1],
            model_specs=[krr_spec],
            feature_spec=FeatureSpec(n_factors=2, n_lags=2),
            oos_start="2014-01-01",
            oos_end="2014-02-01",
            n_jobs=1,
            output_dir=tmp_path,
        )
        rs = exp.run()
        parquet_files = list(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 1
