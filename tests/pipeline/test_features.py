"""Tests for pipeline/features.py — FeatureBuilder."""

import numpy as np
import pytest

from macrocast.pipeline.features import FeatureBuilder


@pytest.fixture()
def synthetic_data():
    """Return (X_panel, y) with T=100 observations and N=20 predictors."""
    rng = np.random.default_rng(0)
    T, N = 100, 20
    X = rng.standard_normal((T, N))
    y = rng.standard_normal(T)
    return X, y


class TestFeatureBuilderFactorsMode:
    def test_fit_transform_shape(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder(n_factors=4, n_lags=3, use_factors=True)
        Z = builder.fit_transform(X, y)
        # Output rows: T - n_lags
        assert Z.shape == (100 - 3, 4 + 3)

    def test_n_features_property(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder(n_factors=4, n_lags=3, use_factors=True)
        builder.fit(X, y)
        assert builder.n_features == 4 + 3

    def test_transform_test_row(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder(n_factors=4, n_lags=3, use_factors=True)
        builder.fit(X, y)
        X_test = X[-1:, :]
        y_lags = y[-3:]
        Z_test = builder.transform(X_test, y_lags)
        assert Z_test.shape == (1 - 3 + 3, 4 + 3)  # 1 row after lag trimming

    def test_no_look_ahead_pca(self, synthetic_data):
        """Fitting on train split must NOT use test rows for PCA."""
        X, y = synthetic_data
        split = 80
        X_tr, X_te = X[:split], X[split:]
        y_tr, y_te = y[:split], y[split:]

        builder = FeatureBuilder(n_factors=4, n_lags=2, use_factors=True)
        builder.fit(X_tr, y_tr)
        # Transform of test rows should not raise and use train-fitted PCA.
        # Pass full y_tr so there are enough values to build AR lags for all X_te rows.
        Z_te = builder.transform(X_te, y_tr)
        assert Z_te.shape[1] == builder.n_features


class TestFeatureBuilderAROnly:
    def test_ar_only_shape(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder(n_lags=5, use_factors=False)
        Z = builder.fit_transform(X, y)
        assert Z.shape == (100 - 5, 5)

    def test_n_features_no_factors(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder(n_lags=5, use_factors=False)
        builder.fit(X, y)
        assert builder.n_features == 5


class TestFeatureBuilderEdgeCases:
    def test_raises_before_fit(self, synthetic_data):
        X, y = synthetic_data
        builder = FeatureBuilder()
        with pytest.raises(RuntimeError):
            builder.transform(X, y)

    def test_n_factors_clamped_to_min_dim(self):
        """n_factors should be clamped when panel has fewer columns."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 3))  # only 3 columns
        y = rng.standard_normal(50)
        builder = FeatureBuilder(n_factors=10, n_lags=2, use_factors=True)
        Z = builder.fit_transform(X, y)
        # n_factors clamped to 3 (or less if T is limiting)
        assert Z.shape[1] <= 3 + 2
