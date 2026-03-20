"""Tests for evaluation/metrics.py."""

import numpy as np
import pytest

from macrocast.evaluation.metrics import csfe, mae, msfe, oos_r2, relative_msfe


def test_msfe_perfect():
    y = np.array([1.0, 2.0, 3.0])
    assert msfe(y, y) == pytest.approx(0.0)


def test_msfe_constant_error():
    y_true = np.ones(5)
    y_hat  = np.zeros(5)
    assert msfe(y_true, y_hat) == pytest.approx(1.0)


def test_mae_basic():
    y_true = np.array([1.0, 2.0, 3.0])
    y_hat  = np.array([0.0, 2.0, 5.0])
    assert mae(y_true, y_hat) == pytest.approx((1 + 0 + 2) / 3)


def test_relative_msfe_equal_models():
    y = np.random.default_rng(0).standard_normal(20)
    y_hat = y + 0.1
    # Any model vs itself → rel MSFE = 1
    assert relative_msfe(y, y_hat, y_hat) == pytest.approx(1.0)


def test_relative_msfe_improvement():
    y      = np.array([1.0, 2.0, 3.0])
    y_bench = np.zeros(3)   # MSFE = (1+4+9)/3 = 14/3
    y_model = y             # perfect → MSFE = 0 → rel = 0
    assert relative_msfe(y, y_model, y_bench) == pytest.approx(0.0)


def test_relative_msfe_zero_benchmark():
    y = np.ones(5)
    assert np.isnan(relative_msfe(y, y, y))


def test_oos_r2_perfect():
    y      = np.array([1.0, 2.0])
    y_bench = np.zeros(2)
    assert oos_r2(y, y, y_bench) == pytest.approx(1.0)  # perfect model


def test_csfe_shape_and_monotone():
    y_true = np.ones(10)
    y_hat  = np.zeros(10)
    c = csfe(y_true, y_hat)
    assert c.shape == (10,)
    assert np.all(np.diff(c) >= 0)  # cumulative — non-decreasing
