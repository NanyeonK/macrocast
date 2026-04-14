"""Tests for macrocast/evaluation/cw.py — Clark-West (2007) test."""

from __future__ import annotations

import numpy as np
import pytest

from macrocast.evaluation.cw import CWResult, cw_test


def _make_forecasts(
    T: int = 200, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(T)
    f_bench = rng.standard_normal(T) * 0.5   # noisy benchmark
    f_model = y + rng.standard_normal(T) * 0.1  # model close to truth
    return y, f_bench, f_model


class TestCWTestBasic:
    def test_returns_cw_result(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = cw_test(y, f1, f2)
        assert isinstance(result, CWResult)

    def test_fields_finite(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = cw_test(y, f1, f2)
        assert np.isfinite(result.statistic)
        assert np.isfinite(result.p_value)
        assert np.isfinite(result.adj_loss_diff_mean)

    def test_p_value_in_unit_interval(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = cw_test(y, f1, f2)
        assert 0.0 <= result.p_value <= 1.0

    def test_reject_consistent_with_p_value(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = cw_test(y, f1, f2, alpha=0.10)
        assert result.reject == (result.p_value < 0.10)

    def test_alpha_stored(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = cw_test(y, f1, f2, alpha=0.05)
        assert result.alpha == 0.05


class TestCWTestDirectionality:
    def test_better_model_has_positive_adj_loss_diff(self) -> None:
        """When f_model is much closer to y than f_bench, d_bar should be > 0."""
        rng = np.random.default_rng(0)
        T = 500
        y = rng.standard_normal(T)
        f_bench = rng.standard_normal(T) * 2.0      # very noisy
        f_model = y + rng.standard_normal(T) * 0.05  # nearly perfect
        result = cw_test(y, f_bench, f_model)
        assert result.adj_loss_diff_mean > 0

    def test_better_model_rejects_h0(self) -> None:
        """A clearly better model should reject H0 at 10%."""
        rng = np.random.default_rng(1)
        T = 500
        y = rng.standard_normal(T)
        f_bench = rng.standard_normal(T) * 2.0
        f_model = y + rng.standard_normal(T) * 0.05
        result = cw_test(y, f_bench, f_model, alpha=0.10)
        assert result.reject

    def test_statistic_negative_when_benchmark_dominates(self) -> None:
        """When the benchmark clearly beats the model on raw loss (before CW
        adjustment), the statistic should be lower than in the reverse case.
        CW is one-sided: it detects when the LARGER model is better."""
        rng = np.random.default_rng(2)
        T = 500
        y = rng.standard_normal(T)
        # case A: model ≈ truth (model wins)
        f_bench_a = rng.standard_normal(T) * 2.0
        f_model_a = y + rng.standard_normal(T) * 0.05
        stat_a = cw_test(y, f_bench_a, f_model_a).statistic
        # case B: roles reversed (benchmark wins)
        stat_b = cw_test(y, f_model_a, f_bench_a).statistic
        assert stat_a > stat_b

    def test_identical_forecasts_zero_statistic(self) -> None:
        """If both forecasts are identical, statistic should be ~0."""
        rng = np.random.default_rng(3)
        T = 200
        y = rng.standard_normal(T)
        f = rng.standard_normal(T)
        result = cw_test(y, f, f)
        assert result.statistic == pytest.approx(0.0, abs=1e-10)
        assert not result.reject

    def test_one_sided_test(self) -> None:
        """p-value should be one-sided (0.5 when statistic=0)."""
        rng = np.random.default_rng(4)
        y = rng.standard_normal(200)
        f = rng.standard_normal(200)
        result = cw_test(y, f, f)
        assert result.p_value == pytest.approx(0.5, abs=1e-10)


class TestCWTestHorizons:
    def test_h1_default_bw(self) -> None:
        """At h=1, Newey-West bandwidth defaults to 1."""
        y, f1, f2 = _make_forecasts()
        result = cw_test(y, f1, f2, h=1)
        assert np.isfinite(result.statistic)

    def test_longer_horizon(self) -> None:
        """h=12 should use larger HAC bandwidth and still produce finite result."""
        y, f1, f2 = _make_forecasts(T=300)
        result = cw_test(y, f1, f2, h=12)
        assert np.isfinite(result.statistic)
        assert 0.0 <= result.p_value <= 1.0

    def test_custom_bw_override(self) -> None:
        """nw_bw override should be respected."""
        y, f1, f2 = _make_forecasts()
        r_default = cw_test(y, f1, f2, h=6)
        r_custom = cw_test(y, f1, f2, h=6, nw_bw=0)
        # Different bandwidth → different statistic
        assert r_default.statistic != r_custom.statistic


class TestCWTestInputValidation:
    def test_length_mismatch_raises(self) -> None:
        y = np.ones(100)
        f1 = np.ones(100)
        f2 = np.ones(50)
        with pytest.raises(ValueError, match="same length"):
            cw_test(y, f1, f2)

    def test_1d_arrays_accepted(self) -> None:
        rng = np.random.default_rng(5)
        y = rng.standard_normal(100)
        f1 = rng.standard_normal(100)
        f2 = rng.standard_normal(100)
        result = cw_test(y, f1, f2)
        assert isinstance(result, CWResult)


class TestCWAdjustmentTerm:
    def test_adj_correction_always_non_negative(self) -> None:
        """The CW adjustment term (f1-f2)^2 is always >= 0, so d_bar >= DM d_bar."""
        rng = np.random.default_rng(10)
        T = 200
        y = rng.standard_normal(T)
        f1 = rng.standard_normal(T)
        f2 = rng.standard_normal(T)

        e1 = y - f1
        e2 = y - f2
        dm_d_bar = float((e1**2 - e2**2).mean())
        cw_d_bar = cw_test(y, f1, f2).adj_loss_diff_mean
        # CW d_bar = DM d_bar + mean((f1-f2)^2) >= DM d_bar
        assert cw_d_bar >= dm_d_bar - 1e-10


class TestCWExport:
    def test_importable_from_evaluation(self) -> None:
        from macrocast.evaluation.cw import CWResult, cw_test  # noqa: F401
