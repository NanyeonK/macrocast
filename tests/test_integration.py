"""End-to-end integration test: synthetic MacroFrame → ForecastExperiment → ResultSet → evaluation.

This test exercises the full Layer 2 + Layer 3 stack without network access
and without R (R-side models are excluded from this test).  It verifies that:

  1. A synthetic MacroFrame (mimicking FRED-MD structure) can be created.
  2. ForecastExperiment produces a non-empty ResultSet.
  3. ResultSet can be written to and read from parquet.
  4. Evaluation metrics (MSFE, Relative MSFE, DM test) run without error.
  5. Treatment effect decomposition produces the four component estimates.
  6. PBSV runs for a small number of groups and test dates.

Tests are marked `not network` and use only in-memory synthetic data.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macrocast.data.schema import MacroFrame, MacroFrameMetadata, VariableMetadata
from macrocast.evaluation.decomposition import decompose_treatment_effects
from macrocast.evaluation.dm import dm_test
from macrocast.evaluation.metrics import msfe, relative_msfe
from macrocast.evaluation.pbsv import oshapley_vi
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_macro_frame() -> MacroFrame:
    """Synthetic FRED-MD-like MacroFrame with 120 monthly observations."""
    rng = np.random.default_rng(2024)
    T, N = 120, 30
    dates = pd.date_range("2005-01-01", periods=T, freq="MS")

    # Simulate stationary-transformed panel (already tcode-applied)
    X = rng.standard_normal((T, N))
    col_names = [f"X{i:03d}" for i in range(N)]
    df = pd.DataFrame(X, index=dates, columns=col_names)

    # Variable metadata for 7 FRED groups
    groups = ["output_income", "labor", "housing", "prices",
              "money", "interest_rates", "stock_market"]
    var_meta = {}
    for i, col in enumerate(col_names):
        var_meta[col] = VariableMetadata(
            name=col,
            description=f"Synthetic variable {i}",
            group=groups[i % len(groups)],
            tcode=5,  # log difference (already applied)
            frequency="monthly",
        )

    meta = MacroFrameMetadata(
        dataset="FRED-MD",
        vintage=None,
        frequency="monthly",
        variables=var_meta,
        groups={g: g.replace("_", " ").title() for g in groups},
        is_transformed=True,
    )
    return MacroFrame(df, meta)


@pytest.fixture(scope="module")
def synthetic_target(synthetic_macro_frame: MacroFrame) -> pd.Series:
    """Synthetic IP growth target derived from the first two factors."""
    rng = np.random.default_rng(99)
    X = synthetic_macro_frame.data.values
    y = 0.5 * X[:, 0] - 0.3 * X[:, 2] + rng.standard_normal(len(X)) * 0.2
    return pd.Series(y, index=synthetic_macro_frame.data.index, name="INDPRO")


@pytest.fixture(scope="module")
def small_experiment_result(
    synthetic_macro_frame: MacroFrame, synthetic_target: pd.Series
) -> ResultSet:
    """Run a small experiment: KRR and RF, h=1, 6 OOS dates."""
    panel = synthetic_macro_frame.data

    krr_spec = ModelSpec(
        model_cls=KRRModel,
        regularization=Regularization.FACTORS,
        cv_scheme=CVScheme.KFOLD(k=3),
        loss_function=LossFunction.L2,
        model_kwargs={"alpha_grid": [0.1, 1.0], "gamma_grid": [0.1], "cv_folds": 3},
    )
    rf_spec = ModelSpec(
        model_cls=RFModel,
        regularization=Regularization.NONE,
        cv_scheme=CVScheme.KFOLD(k=3),
        loss_function=LossFunction.L2,
        model_kwargs={
            "n_estimators": 10,
            "max_depth_grid": [3],
            "min_samples_leaf_grid": [5],
            "cv_folds": 3,
        },
    )

    exp = ForecastExperiment(
        panel=panel,
        target=synthetic_target,
        horizons=[1],
        model_specs=[krr_spec, rf_spec],
        feature_spec=FeatureSpec(n_factors=3, n_lags=3, use_factors=True),
        window=Window.EXPANDING,
        oos_start="2014-07-01",
        oos_end="2014-12-01",
        n_jobs=1,
    )
    return exp.run()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMacroFrameCreation:
    def test_frame_shape(self, synthetic_macro_frame):
        assert synthetic_macro_frame.data.shape == (120, 30)

    def test_index_is_datetime(self, synthetic_macro_frame):
        assert isinstance(synthetic_macro_frame.data.index, pd.DatetimeIndex)

    def test_metadata_populated(self, synthetic_macro_frame):
        meta = synthetic_macro_frame.metadata
        assert meta.dataset == "FRED-MD"
        assert meta.is_transformed is True
        assert len(meta.variables) == 30


class TestExperimentRunsEndToEnd:
    def test_result_set_non_empty(self, small_experiment_result):
        assert len(small_experiment_result) > 0

    def test_result_set_has_two_models(self, small_experiment_result):
        df = small_experiment_result.to_dataframe()
        assert df["model_id"].nunique() == 2

    def test_all_records_have_finite_y_hat(self, small_experiment_result):
        df = small_experiment_result.to_dataframe()
        assert df["y_hat"].notna().all()
        assert np.isfinite(df["y_hat"].values).all()

    def test_horizon_column_correct(self, small_experiment_result):
        df = small_experiment_result.to_dataframe()
        assert set(df["horizon"].unique()) == {1}

    def test_experiment_id_consistent(self, small_experiment_result):
        df = small_experiment_result.to_dataframe()
        assert df["experiment_id"].nunique() == 1


class TestParquetRoundTrip:
    def test_write_read_preserves_rows(self, small_experiment_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_results.parquet"
            small_experiment_result.to_parquet(path)
            df_read = pd.read_parquet(path)
            assert len(df_read) == len(small_experiment_result)

    def test_write_read_preserves_columns(self, small_experiment_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_results.parquet"
            small_experiment_result.to_parquet(path)
            df_read = pd.read_parquet(path)
            for col in ["model_id", "nonlinearity", "regularization", "y_hat", "y_true"]:
                assert col in df_read.columns

    def test_msfe_from_loaded_parquet(self, small_experiment_result):
        """MSFE computed from loaded parquet matches in-memory calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_results.parquet"
            small_experiment_result.to_parquet(path)
            df_memory = small_experiment_result.to_dataframe()
            df_disk   = pd.read_parquet(path)
            # MSFE of the KRR model
            for model_id in df_memory["model_id"].unique():
                m_mem  = df_memory[df_memory["model_id"] == model_id]
                m_disk = df_disk[df_disk["model_id"] == model_id]
                msfe_mem  = msfe(m_mem["y_true"].values, m_mem["y_hat"].values)
                msfe_disk = msfe(m_disk["y_true"].values, m_disk["y_hat"].values)
                assert abs(msfe_mem - msfe_disk) < 1e-10


class TestEvaluationMetrics:
    def test_relative_msfe_finite(self, small_experiment_result):
        df = small_experiment_result.to_dataframe()
        model_ids = df["model_id"].unique()
        # Use first model as pseudo-benchmark
        bench_id = model_ids[0]
        bench = df[df["model_id"] == bench_id]
        for mid in model_ids:
            m = df[df["model_id"] == mid]
            r = relative_msfe(m["y_true"].values, m["y_hat"].values,
                              bench["y_hat"].values)
            assert np.isfinite(r)

    def test_dm_test_runs(self, small_experiment_result):
        df = small_experiment_result.to_dataframe()
        model_ids = df["model_id"].unique()
        m1 = df[df["model_id"] == model_ids[0]].sort_values("forecast_date")
        m2 = df[df["model_id"] == model_ids[1]].sort_values("forecast_date")
        # Align dates
        common = m1.merge(m2[["forecast_date", "y_hat"]], on="forecast_date",
                          suffixes=("_1", "_2"))
        res = dm_test(
            common["y_true"].values,
            common["y_hat_1"].values,
            common["y_hat_2"].values,
            h=1,
        )
        assert np.isfinite(res.dm_stat) or np.isnan(res.dm_stat)

    def test_msfe_by_model_summary(self, small_experiment_result):
        summary = small_experiment_result.msfe_by_model(horizon=1)
        assert len(summary) == 2
        assert "msfe" in summary.columns
        assert (summary["msfe"] >= 0).all()


class TestDecomposition:
    def test_treatment_effect_four_components(self, small_experiment_result):
        """Decomposition should yield four component estimates (+ intercept)."""
        df = small_experiment_result.to_dataframe()

        # Build a richer synthetic result table with more model configurations
        # for the regression to have variation in all four dummies
        rng = np.random.default_rng(42)
        T = 30
        dates = pd.date_range("2010-01", periods=T, freq="MS")
        y_true = rng.standard_normal(T)

        extra_rows = []
        configs = [
            ("linear__none__bic__l2",      "linear", "none",   "_BICScheme()",   "l2",  0.8),
            ("linear__factors__bic__l2",   "linear", "factors","_BICScheme()",   "l2",  0.6),
            ("krr__factors__kfold5__l2",   "krr",    "factors","_KFoldCV(k=5)",  "l2",  0.4),
            ("svr__factors__kfold5__epsi", "svr_rbf","factors","_KFoldCV(k=5)",  "epsilon_insensitive", 0.45),
            ("rf__none__kfold5__l2",       "random_forest","none","_KFoldCV(k=5)","l2", 0.55),
        ]
        for mid, nonlin, reg, cv, loss, noise in configs:
            y_hat = y_true + rng.standard_normal(T) * noise
            for i, d in enumerate(dates):
                extra_rows.append({
                    "model_id": mid, "nonlinearity": nonlin,
                    "regularization": reg, "cv_scheme": cv,
                    "loss_function": loss, "horizon": 1,
                    "forecast_date": d, "y_hat": y_hat[i], "y_true": y_true[i],
                })
        result_df = pd.DataFrame(extra_rows)

        res = decompose_treatment_effects(
            result_df, benchmark_model_id="linear__none__bic__l2"
        )
        assert len(res.coef) == 5  # intercept + 4
        assert "d_nonlinear" in res.coef


class TestPBSV:
    def test_oshapley_basic(self):
        """oShapley-VI runs end-to-end with a tiny dataset."""
        rng = np.random.default_rng(77)
        T_tr, T_te, N = 40, 5, 4
        X_tr = rng.standard_normal((T_tr, N))
        y_tr = X_tr[:, 0] + rng.standard_normal(T_tr) * 0.1
        X_te = rng.standard_normal((T_te, N))
        y_te = X_te[:, 0] + rng.standard_normal(T_te) * 0.1

        def ols_fn(X_train, y_train, X_test):
            Xb = np.column_stack([np.ones(len(X_train)), X_train])
            coef = np.linalg.lstsq(Xb, y_train, rcond=None)[0]
            Xt = np.column_stack([np.ones(len(X_test)), X_test])
            return Xt @ coef

        groups = [[0, 1], [2, 3]]
        phi = oshapley_vi(ols_fn, X_tr, y_tr, X_te, y_te, groups)
        assert phi.shape == (2,)
        assert np.all(np.isfinite(phi))
