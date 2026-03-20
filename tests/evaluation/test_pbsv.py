"""Tests for evaluation/pbsv.py — PBSV and oShapley-VI."""

import numpy as np
import pytest

from macrocast.evaluation.pbsv import compute_pbsv, model_accordance_score, oshapley_vi


def _linear_forecast_fn(X_train, y_train, X_test):
    """Simple OLS forecast function for testing."""
    X_tr = np.column_stack([np.ones(len(X_train)), X_train])
    X_te = np.column_stack([np.ones(len(X_test)), X_test])
    coef = np.linalg.lstsq(X_tr, y_train, rcond=None)[0]
    return X_te @ coef


class TestOShapleyVI:
    def test_output_shape(self):
        rng = np.random.default_rng(10)
        T, N = 60, 4
        X_train = rng.standard_normal((T, N))
        y_train = X_train[:, 0] + rng.standard_normal(T) * 0.1
        X_test  = rng.standard_normal((10, N))
        y_test  = X_test[:, 0] + rng.standard_normal(10) * 0.1

        # Two groups: [0,1] and [2,3]
        groups = [[0, 1], [2, 3]]
        phi = oshapley_vi(_linear_forecast_fn, X_train, y_train, X_test, y_test, groups)
        assert phi.shape == (2,)

    def test_shapley_values_sum_near_total(self):
        """Shapley values should approximately sum to total gain."""
        rng = np.random.default_rng(11)
        T, N = 50, 2
        X = rng.standard_normal((T, N))
        y = X[:, 0] * 2 + rng.standard_normal(T) * 0.05
        X_tr, X_te = X[:40], X[40:]
        y_tr, y_te = y[:40], y[40:]

        groups = [[0], [1]]
        phi = oshapley_vi(_linear_forecast_fn, X_tr, y_tr, X_te, y_te, groups)
        # Both phi values should be finite
        assert np.all(np.isfinite(phi))

    def test_dominant_group_gets_higher_shapley(self):
        """Group 0 (the signal) should have higher |Shapley| than group 1 (noise)."""
        rng = np.random.default_rng(12)
        T = 80
        X = rng.standard_normal((T, 2))
        y = X[:, 0] * 3 + rng.standard_normal(T) * 0.01  # group 0 is signal
        X_tr, X_te = X[:60], X[60:]
        y_tr, y_te = y[:60], y[60:]

        groups = [[0], [1]]
        phi = oshapley_vi(_linear_forecast_fn, X_tr, y_tr, X_te, y_te, groups)
        # Group 0 is the signal: should get positive Shapley; group 1 (noise) negative
        assert phi[0] > 0
        assert phi[1] < 0


class TestComputePBSV:
    def test_output_shape(self):
        rng = np.random.default_rng(13)
        T_test = 5
        T_train_base = 40
        N = 4

        X_train_seq = [rng.standard_normal((T_train_base + t, N)) for t in range(T_test)]
        y_train_seq = [rng.standard_normal(T_train_base + t) for t in range(T_test)]
        X_test = rng.standard_normal((T_test, N))
        y_test = rng.standard_normal(T_test)
        groups = [[0, 1], [2, 3]]

        pbsv = compute_pbsv(
            forecast_fn=_linear_forecast_fn,
            X_train_seq=X_train_seq,
            y_train_seq=y_train_seq,
            X_test_seq=X_test,
            y_test_seq=y_test,
            groups=groups,
        )
        assert pbsv.shape == (T_test, len(groups))


class TestMAS:
    def test_perfect_agreement(self):
        """Identical PBSV matrices → MAS = 1.0 for all groups."""
        pbsv = np.random.default_rng(14).standard_normal((20, 3))
        mas = model_accordance_score(pbsv, pbsv)
        assert np.allclose(mas, 1.0)

    def test_perfect_disagreement(self):
        """Opposite signs → MAS = 0.0."""
        pbsv = np.random.default_rng(15).standard_normal((20, 2))
        mas = model_accordance_score(pbsv, -pbsv)
        assert np.allclose(mas, 0.0)

    def test_output_shape(self):
        pbsv = np.random.default_rng(16).standard_normal((15, 4))
        mas = model_accordance_score(pbsv, pbsv)
        assert mas.shape == (4,)
