"""Tests for pipeline/results.py — ForecastRecord and ResultSet."""

import tempfile
from pathlib import Path

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
from macrocast.pipeline.results import ForecastRecord, ResultSet


def _make_record(y_hat: float = 1.0, y_true: float = 1.5, horizon: int = 1) -> ForecastRecord:
    return ForecastRecord(
        experiment_id="test-exp-001",
        model_id="krr__factors__KFold(k=5)__l2",
        nonlinearity=Nonlinearity.KRR,
        regularization=Regularization.FACTORS,
        cv_scheme=CVScheme.KFOLD(k=5),
        loss_function=LossFunction.L2,
        window=Window.EXPANDING,
        horizon=horizon,
        train_end=pd.Timestamp("2010-01-01"),
        forecast_date=pd.Timestamp("2010-02-01"),
        y_hat=y_hat,
        y_true=y_true,
        n_train=100,
        n_factors=8,
        n_lags=4,
    )


@pytest.fixture
def sample_record() -> ForecastRecord:
    return _make_record()


class TestForecastRecord:
    def test_feature_set_default(self, sample_record: ForecastRecord):
        assert sample_record.feature_set == ""
        assert "feature_set" in sample_record.to_dict()

    def test_error(self):
        r = _make_record(y_hat=1.0, y_true=2.0)
        assert r.error == pytest.approx(1.0)

    def test_squared_error(self):
        r = _make_record(y_hat=1.0, y_true=3.0)
        assert r.squared_error == pytest.approx(4.0)

    def test_to_dict_keys(self):
        r = _make_record()
        d = r.to_dict()
        assert "model_id" in d
        assert "nonlinearity" in d
        assert d["nonlinearity"] == "krr"


class TestResultSet:
    def test_add_and_len(self):
        rs = ResultSet()
        rs.add(_make_record())
        rs.add(_make_record())
        assert len(rs) == 2

    def test_to_dataframe_shape(self):
        rs = ResultSet()
        for _ in range(5):
            rs.add(_make_record())
        df = rs.to_dataframe()
        assert df.shape[0] == 5

    def test_to_dataframe_empty(self):
        rs = ResultSet()
        df = rs.to_dataframe()
        assert df.empty

    def test_msfe_by_model(self):
        rs = ResultSet()
        rs.add(_make_record(y_hat=1.0, y_true=2.0))  # se=1
        rs.add(_make_record(y_hat=1.0, y_true=3.0))  # se=4
        summary = rs.msfe_by_model()
        assert summary["msfe"].iloc[0] == pytest.approx(2.5)  # mean of 1,4

    def test_parquet_roundtrip(self):
        rs = ResultSet(experiment_id="round-trip-test")
        rs.add(_make_record(horizon=1))
        rs.add(_make_record(horizon=3))
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.parquet"
            rs.to_parquet(path)
            assert path.exists()
            df = pd.read_parquet(path)
            assert len(df) == 2
            assert set(df["horizon"]) == {1, 3}


class TestWithCombination:
    """Tests for ResultSet.with_combination()."""

    def _make_multi_model_rs(self) -> ResultSet:
        """Two models × 3 forecast dates × horizon 1."""
        rs = ResultSet(experiment_id="combo-test")
        dates = pd.date_range("2010-01", periods=3, freq="MS")
        for date in dates:
            for model_id, y_hat in [("AR", 1.0), ("LASSO", 1.5)]:
                rs.add(ForecastRecord(
                    experiment_id="combo-test",
                    model_id=model_id,
                    nonlinearity=Nonlinearity.LINEAR,
                    regularization=Regularization.NONE,
                    cv_scheme=CVScheme.BIC,
                    loss_function=LossFunction.L2,
                    window=Window.EXPANDING,
                    horizon=1,
                    train_end=date,
                    forecast_date=date + pd.offsets.MonthBegin(1),
                    y_hat=y_hat,
                    y_true=2.0,
                    n_train=50,
                    n_factors=None,
                    n_lags=2,
                ))
        return rs

    def test_returns_new_result_set(self) -> None:
        rs = self._make_multi_model_rs()
        rs_ext = rs.with_combination("mean")
        assert isinstance(rs_ext, ResultSet)
        assert rs_ext is not rs

    def test_original_unchanged(self) -> None:
        rs = self._make_multi_model_rs()
        rs_ext = rs.with_combination("mean")
        df_orig = rs.to_dataframe()
        assert set(df_orig["model_id"].unique()) == {"AR", "LASSO"}

    def test_combo_mean_present_in_dataframe(self) -> None:
        rs = self._make_multi_model_rs()
        rs_ext = rs.with_combination("mean")
        model_ids = set(rs_ext.to_dataframe()["model_id"].unique())
        assert "COMBO_MEAN" in model_ids
        assert "AR" in model_ids
        assert "LASSO" in model_ids

    def test_combo_mean_value_correct(self) -> None:
        """COMBO_MEAN y_hat should be mean(1.0, 1.5) = 1.25."""
        rs = self._make_multi_model_rs()
        rs_ext = rs.with_combination("mean")
        df = rs_ext.to_dataframe()
        combo_yhat = df[df["model_id"] == "COMBO_MEAN"]["y_hat"].values
        np.testing.assert_allclose(combo_yhat, 1.25)

    def test_multiple_methods(self) -> None:
        rs = self._make_multi_model_rs()
        rs_ext = rs.with_combination(["mean", "median"])
        model_ids = set(rs_ext.to_dataframe()["model_id"].unique())
        assert "COMBO_MEAN" in model_ids
        assert "COMBO_MEDIAN" in model_ids

    def test_string_method_accepted(self) -> None:
        rs = self._make_multi_model_rs()
        rs_ext = rs.with_combination("median")
        assert "COMBO_MEDIAN" in rs_ext.to_dataframe()["model_id"].values

    def test_combination_methods_list(self) -> None:
        rs = self._make_multi_model_rs()
        rs_ext = rs.with_combination(["mean", "inv_msfe"])
        methods = rs_ext.combination_methods()
        assert "COMBO_MEAN" in methods
        assert "COMBO_INV_MSFE" in methods

    def test_combination_methods_empty_on_base(self) -> None:
        rs = self._make_multi_model_rs()
        assert rs.combination_methods() == []

    def test_chaining_adds_both(self) -> None:
        """Chaining two with_combination calls accumulates both combos."""
        rs = self._make_multi_model_rs()
        rs_ext = rs.with_combination("mean").with_combination("median")
        model_ids = set(rs_ext.to_dataframe()["model_id"].unique())
        assert "COMBO_MEAN" in model_ids
        assert "COMBO_MEDIAN" in model_ids

    def test_row_count(self) -> None:
        """Original 6 rows + 3 combo rows = 9."""
        rs = self._make_multi_model_rs()
        rs_ext = rs.with_combination("mean")
        assert len(rs_ext.to_dataframe()) == 9

    def test_empty_rs_raises(self) -> None:
        rs = ResultSet()
        with pytest.raises(ValueError, match="empty"):
            rs.with_combination("mean")
