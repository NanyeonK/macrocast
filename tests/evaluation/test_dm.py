"""Tests for evaluation/dm.py — Diebold-Mariano test."""

import numpy as np
import pytest

from macrocast.evaluation.dm import dm_test


def test_dm_equal_forecasts():
    """DM statistic should be 0 when both forecasts are identical."""
    rng = np.random.default_rng(5)
    y = rng.standard_normal(100)
    y_hat = rng.standard_normal(100)
    res = dm_test(y, y_hat, y_hat, h=1, hln_adjust=False)
    assert res.dm_stat == pytest.approx(0.0, abs=1e-10)
    assert res.p_value == pytest.approx(1.0, abs=1e-6)


def test_dm_superior_model():
    """A clearly better model should yield p < 0.05."""
    rng = np.random.default_rng(6)
    y = rng.standard_normal(200)
    y_hat_good = y + rng.standard_normal(200) * 0.01   # near-perfect
    y_hat_bad  = rng.standard_normal(200)               # random noise
    res = dm_test(y, y_hat_bad, y_hat_good, h=1, hln_adjust=True)
    assert res.p_value < 0.05


def test_dm_loss_diff_mean_sign():
    """loss_diff_mean > 0 means model 1 has higher loss than model 2."""
    rng = np.random.default_rng(7)
    y      = rng.standard_normal(50)
    y_hat1 = rng.standard_normal(50)  # bad
    y_hat2 = y                         # perfect
    res = dm_test(y, y_hat1, y_hat2, h=1)
    assert res.loss_diff_mean > 0


def test_dm_mae_loss():
    rng = np.random.default_rng(8)
    y = rng.standard_normal(100)
    y1, y2 = rng.standard_normal(100), rng.standard_normal(100)
    res = dm_test(y, y1, y2, h=1, loss="mae")
    assert np.isfinite(res.dm_stat)


def test_dm_hln_adjust_flag():
    rng = np.random.default_rng(9)
    y = rng.standard_normal(100)
    y1, y2 = rng.standard_normal(100), rng.standard_normal(100)
    res = dm_test(y, y1, y2, hln_adjust=True)
    assert res.hln_adjusted is True
