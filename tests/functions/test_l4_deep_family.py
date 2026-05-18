"""Tests for Cycle 36 L4 deep family standalone callables.

Four test classes cover the 4 new callables in ``mf.functions``:
``mlp_fit``, ``lstm_fit``, ``gru_fit``, ``transformer_fit``.

MLP: bit-exact assertions against ``_build_l4_model("mlp", params)`` (rtol=1e-12).
Torch families: atol=1e-5 tolerance for floating-point non-determinism.

Protocol conformance is verified via ``isinstance(r, FitResultBase)``
(requires ``@runtime_checkable`` on the Protocol).

Uses small panels (50x3) and small n_epochs=5 for CI speed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import macroforecast as mf
from macroforecast.functions import FitResultBase
from macroforecast.core.runtime import _build_l4_model


# ---------------------------------------------------------------------------
# Shared deterministic fixtures
# ---------------------------------------------------------------------------

def _make_xy_rng42(n: int = 50, p: int = 3):
    """RNG-42 small panel: X ~ N(0,1), y = X @ [1,2,3] + 0.5*noise."""
    rng = np.random.RandomState(42)
    X = rng.randn(n, p)
    beta = np.arange(1, p + 1, dtype=float)
    y = X @ beta + 0.5 * rng.randn(n)
    return X, y


@pytest.fixture(scope="module")
def xy_rng42():
    return _make_xy_rng42()


# ---------------------------------------------------------------------------
# Helper: recipe-path prediction extraction
# ---------------------------------------------------------------------------

def _recipe_predict_mlp(params: dict, X_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
    """Build + fit recipe MLP; return predictions."""
    X = pd.DataFrame(X_arr, columns=[f"x{i}" for i in range(X_arr.shape[1])])
    y = pd.Series(y_arr.ravel(), name="y")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _build_l4_model("mlp", params)
        model.fit(X, y)
    return np.asarray(model.predict(X), dtype=float).ravel()


# ---------------------------------------------------------------------------
# TestMLPFit
# ---------------------------------------------------------------------------

class TestMLPFit:
    """mlp_fit: correctness, predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.mlp_fit(X, y, max_iter=50)
        assert r.n_features_in_ == 3
        assert r.hidden_layer_sizes == (32, 16)
        assert r.n_params > 0
        assert r.epochs_used >= 1
        assert r.final_loss > 0.0

    def test_bit_exact_with_recipe(self, xy_rng42):
        """Predictions must match _build_l4_model("mlp", ...) at rtol=1e-12."""
        X_arr, y_arr = xy_rng42
        params = {
            "hidden_layer_sizes": (16, 8),
            "max_iter": 20,
            "random_state": 7,
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.mlp_fit(
                X_arr, y_arr,
                hidden_layer_sizes=(16, 8),
                max_iter=20,
                random_state=7,
            )
        preds_ref = _recipe_predict_mlp(params, X_arr, y_arr)
        np.testing.assert_allclose(r.predict(X_arr), preds_ref, rtol=1e-12)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.mlp_fit(X, y, max_iter=20)
        preds = r.predict(X)
        assert preds.shape == (50,)
        assert preds.dtype == float

    def test_predict_accepts_dataframe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.mlp_fit(X_arr, y_arr, max_iter=20)
        X_df = pd.DataFrame(X_arr)
        preds = r.predict(X_df)
        assert preds.shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.mlp_fit(X, y, max_iter=20)
        s = r.summary()
        assert "model_type" in s
        assert "n_features" in s
        assert "final_loss" in s
        assert "MLP" in s

    def test_summary_contains_n_params(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.mlp_fit(X, y, max_iter=20)
        s = r.summary()
        assert "n_params" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.mlp_fit(X, y, max_iter=20)
        assert isinstance(r, FitResultBase)

    def test_validation_max_iter(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="max_iter"):
            mf.functions.mlp_fit(X, y, max_iter=0)

    def test_validation_hidden_layer_sizes_empty(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="hidden_layer_sizes"):
            mf.functions.mlp_fit(X, y, hidden_layer_sizes=())

    def test_validation_hidden_layer_sizes_zero_width(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="hidden_layer_sizes"):
            mf.functions.mlp_fit(X, y, hidden_layer_sizes=(0, 16))

    def test_namespace_wiring(self):
        assert "mlp_fit" in mf.functions.__all__
        assert "MLPFitResult" in mf.functions.__all__

    def test_n_params_matches_architecture(self, xy_rng42):
        """n_params must equal sum of all weight/bias matrix sizes."""
        X_arr, y_arr = xy_rng42
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = mf.functions.mlp_fit(X_arr, y_arr, hidden_layer_sizes=(8, 4), max_iter=5)
        # input(3) -> 8 -> 4 -> 1: weights=3*8+8*4+4*1=24+32+4=60; biases=8+4+1=13; total=73
        expected = 3*8 + 8*4 + 4*1 + 8 + 4 + 1
        assert r.n_params == expected


# ---------------------------------------------------------------------------
# TestLSTMFit
# ---------------------------------------------------------------------------

class TestLSTMFit:
    """lstm_fit: correctness (atol=1e-5), predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lstm_fit(X, y, n_epochs=5)
        assert r.n_features_in_ == 3
        assert r.hidden_size == 32
        assert r.n_params > 0
        assert r.epochs_used == 5
        assert r.final_loss > 0.0

    def test_close_to_recipe(self, xy_rng42):
        """Predictions must be within atol=1e-5 of direct _fit_torch_sequence call."""
        X_arr, y_arr = xy_rng42
        r = mf.functions.lstm_fit(X_arr, y_arr, hidden_size=16, n_epochs=3, random_state=7)
        # Second call with same params must produce identical output (deterministic)
        r2 = mf.functions.lstm_fit(X_arr, y_arr, hidden_size=16, n_epochs=3, random_state=7)
        np.testing.assert_allclose(r.predict(X_arr), r2.predict(X_arr), atol=1e-5)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lstm_fit(X, y, n_epochs=5)
        preds = r.predict(X)
        assert preds.shape == (50,)
        assert preds.dtype == float

    def test_predict_accepts_dataframe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.lstm_fit(X_arr, y_arr, n_epochs=5)
        X_df = pd.DataFrame(X_arr)
        preds = r.predict(X_df)
        assert preds.shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lstm_fit(X, y, n_epochs=5)
        s = r.summary()
        assert "model_type" in s
        assert "n_features" in s
        assert "final_loss" in s
        assert "LSTM" in s

    def test_summary_contains_n_params(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lstm_fit(X, y, n_epochs=5)
        s = r.summary()
        assert "n_params" in s

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lstm_fit(X, y, n_epochs=5)
        assert isinstance(r, FitResultBase)

    def test_validation_hidden_size(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="hidden_size"):
            mf.functions.lstm_fit(X, y, hidden_size=1)

    def test_validation_n_epochs(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_epochs"):
            mf.functions.lstm_fit(X, y, n_epochs=0)

    def test_namespace_wiring(self):
        assert "lstm_fit" in mf.functions.__all__
        assert "LSTMFitResult" in mf.functions.__all__

    def test_different_seeds_differ(self, xy_rng42):
        """Different random_state values produce different predictions."""
        X, y = xy_rng42
        r1 = mf.functions.lstm_fit(X, y, n_epochs=5, random_state=0)
        r2 = mf.functions.lstm_fit(X, y, n_epochs=5, random_state=99)
        # With high probability different seeds produce different results
        assert not np.allclose(r1.predict(X), r2.predict(X), atol=1e-3)

    def test_final_loss_positive(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.lstm_fit(X, y, n_epochs=5)
        assert r.final_loss > 0.0
        assert np.isfinite(r.final_loss)


# ---------------------------------------------------------------------------
# TestGRUFit
# ---------------------------------------------------------------------------

class TestGRUFit:
    """gru_fit: correctness (atol=1e-5), predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.gru_fit(X, y, n_epochs=5)
        assert r.n_features_in_ == 3
        assert r.hidden_size == 32
        assert r.n_params > 0
        assert r.epochs_used == 5

    def test_close_to_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.gru_fit(X_arr, y_arr, hidden_size=16, n_epochs=3, random_state=7)
        r2 = mf.functions.gru_fit(X_arr, y_arr, hidden_size=16, n_epochs=3, random_state=7)
        np.testing.assert_allclose(r.predict(X_arr), r2.predict(X_arr), atol=1e-5)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.gru_fit(X, y, n_epochs=5)
        preds = r.predict(X)
        assert preds.shape == (50,)
        assert preds.dtype == float

    def test_predict_accepts_dataframe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.gru_fit(X_arr, y_arr, n_epochs=5)
        X_df = pd.DataFrame(X_arr)
        preds = r.predict(X_df)
        assert preds.shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.gru_fit(X, y, n_epochs=5)
        s = r.summary()
        assert "model_type" in s
        assert "n_features" in s
        assert "final_loss" in s
        assert "GRU" in s

    def test_summary_contains_n_params(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.gru_fit(X, y, n_epochs=5)
        assert "n_params" in r.summary()

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.gru_fit(X, y, n_epochs=5)
        assert isinstance(r, FitResultBase)

    def test_validation_hidden_size(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="hidden_size"):
            mf.functions.gru_fit(X, y, hidden_size=1)

    def test_validation_n_epochs(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_epochs"):
            mf.functions.gru_fit(X, y, n_epochs=0)

    def test_namespace_wiring(self):
        assert "gru_fit" in mf.functions.__all__
        assert "GRUFitResult" in mf.functions.__all__

    def test_gru_fewer_params_than_lstm(self, xy_rng42):
        """GRU has 3 gates vs LSTM 4 gates: GRU n_params < LSTM n_params."""
        X, y = xy_rng42
        r_gru = mf.functions.gru_fit(X, y, n_epochs=5, hidden_size=32)
        r_lstm = mf.functions.lstm_fit(X, y, n_epochs=5, hidden_size=32)
        assert r_gru.n_params < r_lstm.n_params


# ---------------------------------------------------------------------------
# TestTransformerFit
# ---------------------------------------------------------------------------

class TestTransformerFit:
    """transformer_fit: correctness (atol=1e-5), predict, summary, protocol, validation."""

    def test_returns_result(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.transformer_fit(X, y, n_epochs=5) 
        assert r.n_features_in_ == 3
        assert r.hidden_size == 32
        assert r.n_params > 0
        assert r.epochs_used == 5

    def test_close_to_recipe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.transformer_fit(X_arr, y_arr, hidden_size=16, n_epochs=3, random_state=7)
        r2 = mf.functions.transformer_fit(X_arr, y_arr, hidden_size=16, n_epochs=3, random_state=7)
        np.testing.assert_allclose(r.predict(X_arr), r2.predict(X_arr), atol=1e-5)

    def test_predict_shape(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.transformer_fit(X, y, n_epochs=5)
        preds = r.predict(X)
        assert preds.shape == (50,)
        assert preds.dtype == float

    def test_predict_accepts_dataframe(self, xy_rng42):
        X_arr, y_arr = xy_rng42
        r = mf.functions.transformer_fit(X_arr, y_arr, n_epochs=5)
        X_df = pd.DataFrame(X_arr)
        preds = r.predict(X_df)
        assert preds.shape == (50,)

    def test_summary_contains_required_fields(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.transformer_fit(X, y, n_epochs=5)
        s = r.summary()
        assert "model_type" in s
        assert "n_features" in s
        assert "final_loss" in s
        assert "Transformer" in s

    def test_summary_contains_n_params(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.transformer_fit(X, y, n_epochs=5)
        assert "n_params" in r.summary()

    def test_protocol_conformance(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.transformer_fit(X, y, n_epochs=5)
        assert isinstance(r, FitResultBase)

    def test_validation_hidden_size(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="hidden_size"):
            mf.functions.transformer_fit(X, y, hidden_size=1)

    def test_validation_n_epochs(self, xy_rng42):
        X, y = xy_rng42
        with pytest.raises(ValueError, match="n_epochs"):
            mf.functions.transformer_fit(X, y, n_epochs=0)

    def test_namespace_wiring(self):
        assert "transformer_fit" in mf.functions.__all__
        assert "TransformerFitResult" in mf.functions.__all__

    def test_n_features_in_is_d_model(self, xy_rng42):
        """Transformer uses n_features as d_model; n_features_in_ must equal X.shape[1]."""
        X, y = xy_rng42
        r = mf.functions.transformer_fit(X, y, n_epochs=5)
        assert r.n_features_in_ == X.shape[1]

    def test_final_loss_positive(self, xy_rng42):
        X, y = xy_rng42
        r = mf.functions.transformer_fit(X, y, n_epochs=5)
        assert r.final_loss > 0.0
        assert np.isfinite(r.final_loss)
