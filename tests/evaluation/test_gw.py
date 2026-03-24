"""Tests for macrocast/evaluation/gw.py — Giacomini-White (2006) test."""

from __future__ import annotations

import numpy as np
import pytest

from macrocast.evaluation.gw import GWResult, gw_test


def _make_forecasts(
    T: int = 200, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(T)
    f1 = rng.standard_normal(T) * 0.5
    f2 = y + rng.standard_normal(T) * 0.1
    return y, f1, f2


class TestGWTestBasic:
    def test_returns_gw_result(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = gw_test(y, f1, f2)
        assert isinstance(result, GWResult)

    def test_fields_finite(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = gw_test(y, f1, f2)
        assert np.isfinite(result.statistic)
        assert np.isfinite(result.p_value)

    def test_p_value_in_unit_interval(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = gw_test(y, f1, f2)
        assert 0.0 <= result.p_value <= 1.0

    def test_reject_consistent_with_p_value(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = gw_test(y, f1, f2, alpha=0.10)
        assert result.reject == (result.p_value < 0.10)

    def test_alpha_stored(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = gw_test(y, f1, f2, alpha=0.05)
        assert result.alpha == 0.05

    def test_df_equals_one_unconditional(self) -> None:
        """Without instruments, df should be 1 (constant instrument)."""
        y, f1, f2 = _make_forecasts()
        result = gw_test(y, f1, f2)
        assert result.df == 1

    def test_mean_interaction_shape_unconditional(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = gw_test(y, f1, f2)
        assert result.mean_interaction.shape == (1,)


class TestGWTestUnconditionalEquivalence:
    def test_unconditional_rejects_when_model2_better(self) -> None:
        """Unconditional GW should reject H0 when f2 is clearly better than f1."""
        rng = np.random.default_rng(1)
        T = 500
        y = rng.standard_normal(T)
        f1 = rng.standard_normal(T) * 2.0
        f2 = y + rng.standard_normal(T) * 0.05
        result = gw_test(y, f1, f2, alpha=0.10)
        assert result.reject

    def test_identical_forecasts_degenerate(self) -> None:
        """Identical forecasts → d_t = 0 → Z = 0 → singular S → degenerate NaN path."""
        rng = np.random.default_rng(2)
        T = 200
        y = rng.standard_normal(T)
        f = rng.standard_normal(T)
        result = gw_test(y, f, f)
        # Degenerate: S is all-zero, inversion fails → NaN statistic
        assert not result.reject

    def test_statistic_symmetric_in_models(self) -> None:
        """GW is two-sided: swapping f1/f2 should give same statistic."""
        y, f1, f2 = _make_forecasts()
        r12 = gw_test(y, f1, f2)
        r21 = gw_test(y, f2, f1)
        # Loss differential flips sign, but Wald stat uses Z_bar' S^{-1} Z_bar
        # which is a quadratic form — stays the same magnitude up to sign of Z_bar
        assert r12.statistic == pytest.approx(r21.statistic, rel=1e-6)
        assert r12.p_value == pytest.approx(r21.p_value, rel=1e-6)


class TestGWTestConditional:
    def test_single_instrument_df_one(self) -> None:
        rng = np.random.default_rng(3)
        T = 200
        y = rng.standard_normal(T)
        f1 = rng.standard_normal(T)
        f2 = rng.standard_normal(T)
        instr = rng.integers(0, 2, size=T).astype(float)
        result = gw_test(y, f1, f2, instruments=instr)
        assert result.df == 1
        assert result.mean_interaction.shape == (1,)

    def test_two_instruments_df_two(self) -> None:
        rng = np.random.default_rng(4)
        T = 200
        y = rng.standard_normal(T)
        f1 = rng.standard_normal(T)
        f2 = rng.standard_normal(T)
        instr = rng.standard_normal((T, 2))
        result = gw_test(y, f1, f2, instruments=instr)
        assert result.df == 2
        assert result.mean_interaction.shape == (2,)

    def test_1d_instrument_accepted(self) -> None:
        """A 1-D instrument array should work (reshaped to (T, 1) internally)."""
        rng = np.random.default_rng(5)
        T = 200
        y = rng.standard_normal(T)
        f1 = rng.standard_normal(T)
        f2 = rng.standard_normal(T)
        instr = rng.standard_normal(T)
        result = gw_test(y, f1, f2, instruments=instr)
        assert result.df == 1

    def test_recession_indicator_instrument(self) -> None:
        """Binary recession dummy instrument should work correctly."""
        rng = np.random.default_rng(6)
        T = 300
        y = rng.standard_normal(T)
        # f2 beats f1 in recessions, roughly equal otherwise
        rec = (rng.uniform(size=T) < 0.2).astype(float)
        f1 = rng.standard_normal(T)
        f2 = y + rng.standard_normal(T) * 0.2
        result = gw_test(y, f1, f2, instruments=rec)
        assert isinstance(result, GWResult)
        assert np.isfinite(result.statistic)

    def test_p_value_in_unit_interval_conditional(self) -> None:
        rng = np.random.default_rng(7)
        T = 200
        y = rng.standard_normal(T)
        f1 = rng.standard_normal(T)
        f2 = rng.standard_normal(T)
        instr = np.column_stack([
            rng.integers(0, 2, size=T).astype(float),
            rng.standard_normal(T),
        ])
        result = gw_test(y, f1, f2, instruments=instr)
        assert 0.0 <= result.p_value <= 1.0


class TestGWTestLossFunction:
    def test_mse_loss(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = gw_test(y, f1, f2, loss="mse")
        assert np.isfinite(result.statistic)

    def test_mae_loss(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = gw_test(y, f1, f2, loss="mae")
        assert np.isfinite(result.statistic)

    def test_mse_mae_differ(self) -> None:
        """MSE and MAE should generally produce different statistics."""
        y, f1, f2 = _make_forecasts()
        r_mse = gw_test(y, f1, f2, loss="mse")
        r_mae = gw_test(y, f1, f2, loss="mae")
        assert r_mse.statistic != r_mae.statistic

    def test_unknown_loss_raises(self) -> None:
        y, f1, f2 = _make_forecasts()
        with pytest.raises(ValueError, match="Unknown loss"):
            gw_test(y, f1, f2, loss="huber")  # type: ignore[arg-type]


class TestGWTestHorizons:
    def test_h1_default_bw(self) -> None:
        y, f1, f2 = _make_forecasts()
        result = gw_test(y, f1, f2, h=1)
        assert np.isfinite(result.statistic)

    def test_longer_horizon(self) -> None:
        y, f1, f2 = _make_forecasts(T=300)
        result = gw_test(y, f1, f2, h=12)
        assert np.isfinite(result.statistic)
        assert 0.0 <= result.p_value <= 1.0

    def test_custom_bw_override(self) -> None:
        y, f1, f2 = _make_forecasts()
        r_default = gw_test(y, f1, f2, h=6)
        r_custom = gw_test(y, f1, f2, h=6, nw_bw=0)
        assert r_default.statistic != r_custom.statistic


class TestGWTestInputValidation:
    def test_length_mismatch_raises(self) -> None:
        y = np.ones(100)
        f1 = np.ones(100)
        f2 = np.ones(50)
        with pytest.raises(ValueError, match="same length"):
            gw_test(y, f1, f2)

    def test_instrument_row_mismatch_raises(self) -> None:
        y = np.ones(100)
        f1 = np.ones(100)
        f2 = np.ones(100)
        instr = np.ones((50, 2))
        with pytest.raises(ValueError, match="instruments must have"):
            gw_test(y, f1, f2, instruments=instr)

    def test_1d_arrays_accepted(self) -> None:
        rng = np.random.default_rng(8)
        y = rng.standard_normal(100)
        f1 = rng.standard_normal(100)
        f2 = rng.standard_normal(100)
        result = gw_test(y, f1, f2)
        assert isinstance(result, GWResult)


class TestGWExport:
    def test_importable_from_evaluation(self) -> None:
        from macrocast.evaluation.gw import GWResult, gw_test  # noqa: F401
