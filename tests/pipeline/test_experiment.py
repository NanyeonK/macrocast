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


class TestPathAverageTarget:
    """Tests for target_scheme='path_average' in ForecastExperiment."""

    def test_path_avg_h1_equals_direct(self, synthetic_panel, krr_spec):
        """At h=1, path_average and direct must produce identical y_true values."""
        panel, target = synthetic_panel

        common = dict(
            panel=panel,
            target=target,
            horizons=[1],
            model_specs=[krr_spec],
            oos_start="2014-01-01",
            oos_end="2014-03-01",
            n_jobs=1,
        )
        rs_direct = ForecastExperiment(
            **common,
            feature_spec=FeatureSpec(n_factors=2, n_lags=2, target_scheme="direct"),
        ).run()
        rs_path = ForecastExperiment(
            **common,
            feature_spec=FeatureSpec(n_factors=2, n_lags=2, target_scheme="path_average"),
        ).run()

        y_true_direct = sorted(r.y_true for r in rs_direct.records)
        y_true_path = sorted(r.y_true for r in rs_path.records)
        np.testing.assert_allclose(y_true_direct, y_true_path, rtol=1e-10)

    def test_path_avg_h3_manual_verify(self, synthetic_panel, krr_spec):
        """At h=3, y_true for path_average must equal mean of 3 adjacent target values."""
        panel, target = synthetic_panel

        rs = ForecastExperiment(
            panel=panel,
            target=target,
            horizons=[3],
            model_specs=[krr_spec],
            feature_spec=FeatureSpec(n_factors=2, n_lags=2, target_scheme="path_average"),
            oos_start="2014-01-01",
            oos_end="2014-02-01",
            n_jobs=1,
        ).run()

        assert len(rs.records) > 0
        for r in rs.records:
            t_star = r.forecast_date
            t_idx = target.index.get_loc(t_star)
            expected_avg = float(np.mean(target.iloc[t_idx - 3 + 1 : t_idx + 1]))
            np.testing.assert_allclose(r.y_true, expected_avg, rtol=1e-10)

    def test_target_scheme_recorded_in_results(self, synthetic_panel, krr_spec):
        """target_scheme value is stored in each ForecastRecord."""
        panel, target = synthetic_panel
        rs = ForecastExperiment(
            panel=panel,
            target=target,
            horizons=[1],
            model_specs=[krr_spec],
            feature_spec=FeatureSpec(n_factors=2, n_lags=2, target_scheme="path_average"),
            oos_start="2014-01-01",
            oos_end="2014-02-01",
            n_jobs=1,
        ).run()
        assert all(r.target_scheme == "path_average" for r in rs.records)

    def test_invalid_target_scheme_raises(self, synthetic_panel, krr_spec):
        """Invalid target_scheme value should raise ValueError at construction time."""
        panel, target = synthetic_panel
        with pytest.raises(ValueError, match="target_scheme"):
            ForecastExperiment(
                panel=panel,
                target=target,
                horizons=[1],
                model_specs=[krr_spec],
                feature_spec=FeatureSpec(target_scheme="iterated"),
            )


class TestCLSS2021Integration:
    """End-to-end integration tests for CLSS 2021 feature modes."""

    def test_marx_maf_path_average_runs_without_error(self, synthetic_panel, krr_spec):
        """FeatureSpec(use_maf=True, target_scheme='path_average') runs full experiment."""
        panel, target = synthetic_panel
        rs = ForecastExperiment(
            panel=panel,
            target=target,
            horizons=[3],
            model_specs=[krr_spec],
            feature_spec=FeatureSpec(
                n_factors=2,
                n_lags=2,
                use_maf=True,
                p_marx=4,
                target_scheme="path_average",
            ),
            oos_start="2014-01-01",
            oos_end="2014-03-01",
            n_jobs=1,
        ).run()
        assert isinstance(rs, ResultSet)
        assert len(rs) > 0

    def test_include_levels_runs_without_error(self, synthetic_panel, krr_spec):
        """include_levels=True with panel_levels passes through experiment."""
        panel, target = synthetic_panel
        rng = np.random.default_rng(42)
        levels = pd.DataFrame(
            rng.standard_normal((len(panel), 5)),
            index=panel.index,
            columns=[f"lev{i}" for i in range(5)],
        )
        rs = ForecastExperiment(
            panel=panel,
            target=target,
            horizons=[1],
            model_specs=[krr_spec],
            feature_spec=FeatureSpec(
                n_factors=2, n_lags=2, use_factors=True, include_levels=True
            ),
            panel_levels=levels,
            oos_start="2014-01-01",
            oos_end="2014-02-01",
            n_jobs=1,
        ).run()
        assert len(rs) > 0

    def test_include_levels_requires_panel_levels(self, synthetic_panel, krr_spec):
        """include_levels=True without panel_levels raises ValueError."""
        panel, target = synthetic_panel
        with pytest.raises(ValueError, match="panel_levels"):
            ForecastExperiment(
                panel=panel,
                target=target,
                horizons=[1],
                model_specs=[krr_spec],
                feature_spec=FeatureSpec(include_levels=True),
            )
