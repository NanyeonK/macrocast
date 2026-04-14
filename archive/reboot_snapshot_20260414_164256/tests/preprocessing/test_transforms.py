"""Tests for macrocast.data.transforms.

Numerical correctness is verified against hand-computed values based on
McCracken & Ng (2016) Table 1 definitions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from macrocast.preprocessing.transforms import (
    TransformCode,
    apply_hamilton_filter,
    apply_pca,
    apply_tcode,
    apply_tcodes,
    apply_x_factors,
)


@pytest.fixture
def level_series() -> pd.Series:
    """Simple level series: 1, 2, 4, 8, 16."""
    return pd.Series([1.0, 2.0, 4.0, 8.0, 16.0], name="X")


@pytest.fixture
def log_series() -> pd.Series:
    """Exponential series suitable for log transforms."""
    return pd.Series(
        [np.exp(1), np.exp(2), np.exp(3), np.exp(4), np.exp(5)], name="Y"
    )


class TestTransformCode:
    def test_enum_values(self) -> None:
        assert TransformCode.LEVEL == 1
        assert TransformCode.DIFF == 2
        assert TransformCode.DIFF2 == 3
        assert TransformCode.LOG == 4
        assert TransformCode.LOG_DIFF == 5
        assert TransformCode.LOG_DIFF2 == 6
        assert TransformCode.DELTA_RATIO == 7


class TestApplyTcode:
    def test_tcode1_level(self, level_series: pd.Series) -> None:
        result = apply_tcode(level_series, 1)
        pd.testing.assert_series_equal(result, level_series)

    def test_tcode2_diff(self, level_series: pd.Series) -> None:
        result = apply_tcode(level_series, 2)
        expected = pd.Series([np.nan, 1.0, 2.0, 4.0, 8.0], name="X")
        pd.testing.assert_series_equal(result, expected)

    def test_tcode3_diff2(self, level_series: pd.Series) -> None:
        result = apply_tcode(level_series, 3)
        # First diff: NaN, 1, 2, 4, 8
        # Second diff: NaN, NaN, 1, 2, 4
        expected = pd.Series([np.nan, np.nan, 1.0, 2.0, 4.0], name="X")
        pd.testing.assert_series_equal(result, expected)

    def test_tcode4_log(self, log_series: pd.Series) -> None:
        result = apply_tcode(log_series, 4)
        expected = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], name="Y")
        pd.testing.assert_series_equal(result, expected, atol=1e-10)

    def test_tcode5_log_diff(self, log_series: pd.Series) -> None:
        # ln(e^k) - ln(e^(k-1)) = k - (k-1) = 1 for all k
        result = apply_tcode(log_series, 5)
        expected = pd.Series([np.nan, 1.0, 1.0, 1.0, 1.0], name="Y")
        pd.testing.assert_series_equal(result, expected, atol=1e-10)

    def test_tcode6_log_diff2(self, log_series: pd.Series) -> None:
        # Δ²ln = Δ(1,1,1,1) = NaN, NaN, 0, 0, 0
        result = apply_tcode(log_series, 6)
        expected = pd.Series([np.nan, np.nan, 0.0, 0.0, 0.0], name="Y")
        pd.testing.assert_series_equal(result, expected, atol=1e-10)

    def test_tcode7_delta_ratio(self, level_series: pd.Series) -> None:
        # x = 1,2,4,8,16
        # ratio = x_t/x_{t-1} = NaN, 2, 2, 2, 2
        # Δratio = NaN, NaN, 0, 0, 0
        result = apply_tcode(level_series, 7)
        expected = pd.Series([np.nan, np.nan, 0.0, 0.0, 0.0], name="X")
        pd.testing.assert_series_equal(result, expected, atol=1e-10)

    def test_invalid_tcode_raises(self) -> None:
        with pytest.raises(ValueError, match="tcode must be 1-7"):
            apply_tcode(pd.Series([1.0, 2.0]), 8)

    def test_log_of_nonpositive_produces_nan_and_warns(self) -> None:
        s = pd.Series([-1.0, 0.0, 1.0, 2.0], name="Z")
        with pytest.warns(UserWarning, match="non-positive"):
            result = apply_tcode(s, 4)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert not np.isnan(result.iloc[2])

    def test_preserves_index(self) -> None:
        idx = pd.date_range("2000-01", periods=5, freq="MS")
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=idx, name="A")
        result = apply_tcode(s, 2)
        assert list(result.index) == list(idx)


class TestApplyTcodes:
    def test_applies_different_codes_per_column(self) -> None:
        df = pd.DataFrame(
            {"A": [1.0, 2.0, 3.0, 4.0], "B": [10.0, 20.0, 30.0, 40.0]}
        )
        tcodes = {"A": 1, "B": 2}
        result = apply_tcodes(df, tcodes)
        # A: level, no change
        pd.testing.assert_series_equal(result["A"], df["A"])
        # B: first diff
        expected_b = pd.Series([np.nan, 10.0, 10.0, 10.0], name="B")
        pd.testing.assert_series_equal(result["B"], expected_b)

    def test_missing_column_uses_level(self) -> None:
        df = pd.DataFrame({"X": [1.0, 2.0, 3.0]})
        result = apply_tcodes(df, {})
        pd.testing.assert_series_equal(result["X"], df["X"])

    def test_same_shape_as_input(self) -> None:
        df = pd.DataFrame(np.random.randn(20, 5), columns=list("ABCDE"))
        tcodes = {c: i + 1 for i, c in enumerate("ABCDE")}
        result = apply_tcodes(df, tcodes)
        assert result.shape == df.shape


class TestInverseTcode:
    def test_raises_not_implemented(self) -> None:
        from macrocast.preprocessing.transforms import inverse_tcode

        with pytest.raises(NotImplementedError):
            inverse_tcode(pd.Series([1.0]), 1, pd.Series([1.0]))


class TestApplyPca:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 10))
        F = apply_pca(X, k=3)
        assert F.shape == (50, 3)

    def test_agrees_with_apply_x_factors(self) -> None:
        rng = np.random.default_rng(1)
        X = rng.standard_normal((40, 8))
        np.testing.assert_array_equal(apply_pca(X, k=4), apply_x_factors(X, k=4))

    def test_k_capped_at_min_dim(self) -> None:
        rng = np.random.default_rng(2)
        X = rng.standard_normal((5, 3))
        F = apply_pca(X, k=10)
        # n_components capped at min(K, T-1) = min(3, 4) = 3
        assert F.shape[1] <= 3


class TestApplyHamiltonFilter:
    @pytest.fixture
    def quarterly_series(self) -> pd.Series:
        rng = np.random.default_rng(42)
        trend = np.cumsum(rng.standard_normal(120))
        cycle = rng.standard_normal(120) * 0.5
        idx = pd.period_range("1990Q1", periods=120, freq="Q")
        return pd.Series(trend + cycle, index=idx)

    def test_output_shape(self, quarterly_series: pd.Series) -> None:
        trend, cycle = apply_hamilton_filter(quarterly_series, h=8, p=4)
        assert len(trend) == len(quarterly_series)
        assert len(cycle) == len(quarterly_series)

    def test_preserves_index(self, quarterly_series: pd.Series) -> None:
        trend, cycle = apply_hamilton_filter(quarterly_series, h=8, p=4)
        assert trend.index.equals(quarterly_series.index)
        assert cycle.index.equals(quarterly_series.index)

    def test_leading_nans(self, quarterly_series: pd.Series) -> None:
        h, p = 8, 4
        trend, cycle = apply_hamilton_filter(quarterly_series, h=h, p=p)
        # First h+p-1 values must be NaN
        assert np.all(np.isnan(trend.values[: h + p - 1]))
        assert np.all(np.isnan(cycle.values[: h + p - 1]))
        # Values from h+p-1 onwards must be finite
        assert np.all(np.isfinite(trend.values[h + p - 1 :]))

    def test_trend_plus_cycle_equals_y(self, quarterly_series: pd.Series) -> None:
        h, p = 8, 4
        trend, cycle = apply_hamilton_filter(quarterly_series, h=h, p=p)
        y = quarterly_series.values
        start = h + p - 1
        # trend + cycle == y_{t+h} at valid positions
        np.testing.assert_allclose(
            trend.values[start:] + cycle.values[start:],
            y[start:],
            rtol=1e-10,
        )

    def test_ndarray_input(self) -> None:
        rng = np.random.default_rng(7)
        y = rng.standard_normal(100)
        trend, cycle = apply_hamilton_filter(y, h=8, p=4)
        assert isinstance(trend, np.ndarray)
        assert isinstance(cycle, np.ndarray)
        assert trend.shape == (100,)

    def test_too_short_raises(self) -> None:
        y = np.arange(10.0)
        with pytest.raises(ValueError, match="too short"):
            apply_hamilton_filter(y, h=8, p=4)
