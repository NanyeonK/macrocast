"""Tests for macrocast/config.py — YAML config loader."""

import tempfile
from pathlib import Path

import pytest
import yaml

from macrocast.config import (
    DEFAULT_CONFIG_YAML,
    ExperimentConfig,
    load_config,
    load_config_from_dict,
)
from macrocast.pipeline.components import (
    CVScheme,
    LossFunction,
    Nonlinearity,
    Regularization,
    Window,
)
from macrocast.pipeline.models import KRRModel, RFModel


# ---------------------------------------------------------------------------
# Minimal valid config dict
# ---------------------------------------------------------------------------


MINIMAL_CONFIG = {
    "experiment": {
        "id": "test-exp-001",
        "output_dir": "/tmp/macrocast_test",
        "horizons": [1, 3],
        "window": "expanding",
        "n_jobs": 1,
    },
    "data": {
        "dataset": "fred_md",
        "target": "INDPRO",
    },
    "features": {
        "n_factors": 4,
        "n_lags": 2,
        "use_factors": True,
    },
    "models": [
        {
            "name": "krr",
            "regularization": "factors",
            "cv_scheme": "kfold",
            "kfold_k": 3,
            "loss_function": "l2",
            "kwargs": {"alpha_grid": [0.1, 1.0], "gamma_grid": [0.1]},
        },
        {
            "name": "rf",
            "regularization": "none",
            "cv_scheme": "kfold",
            "kfold_k": 3,
            "loss_function": "l2",
            "kwargs": {"n_estimators": 10},
        },
    ],
}


class TestLoadConfigFromDict:
    def test_returns_experiment_config(self):
        cfg = load_config_from_dict(MINIMAL_CONFIG)
        assert isinstance(cfg, ExperimentConfig)

    def test_experiment_id(self):
        cfg = load_config_from_dict(MINIMAL_CONFIG)
        assert cfg.experiment_id == "test-exp-001"

    def test_horizons(self):
        cfg = load_config_from_dict(MINIMAL_CONFIG)
        assert cfg.horizons == [1, 3]

    def test_window_expanding(self):
        cfg = load_config_from_dict(MINIMAL_CONFIG)
        assert cfg.window == Window.EXPANDING

    def test_window_rolling(self):
        raw = {**MINIMAL_CONFIG, "experiment": {**MINIMAL_CONFIG["experiment"], "window": "rolling"}}
        cfg = load_config_from_dict(raw)
        assert cfg.window == Window.ROLLING

    def test_model_count(self):
        cfg = load_config_from_dict(MINIMAL_CONFIG)
        assert len(cfg.model_specs) == 2

    def test_krr_model_class(self):
        cfg = load_config_from_dict(MINIMAL_CONFIG)
        krr = next(s for s in cfg.model_specs if s.model_cls is KRRModel)
        assert krr.regularization == Regularization.FACTORS
        assert krr.loss_function == LossFunction.L2

    def test_rf_model_class(self):
        cfg = load_config_from_dict(MINIMAL_CONFIG)
        rf = next(s for s in cfg.model_specs if s.model_cls is RFModel)
        assert rf.regularization == Regularization.NONE

    def test_kfold_cv_scheme(self):
        cfg = load_config_from_dict(MINIMAL_CONFIG)
        for spec in cfg.model_specs:
            assert spec.cv_scheme == CVScheme.KFOLD(k=3)

    def test_feature_spec(self):
        cfg = load_config_from_dict(MINIMAL_CONFIG)
        assert cfg.feature_spec.n_factors == 4
        assert cfg.feature_spec.n_lags == 2
        assert cfg.feature_spec.use_factors is True

    def test_auto_experiment_id_when_null(self):
        raw = {**MINIMAL_CONFIG, "experiment": {**MINIMAL_CONFIG["experiment"], "id": None}}
        cfg = load_config_from_dict(raw)
        assert cfg.experiment_id  # should not be None or empty

    def test_unknown_model_raises(self):
        raw = {
            **MINIMAL_CONFIG,
            "models": [{"name": "nonexistent_model"}]
        }
        with pytest.raises(ValueError, match="Unknown model"):
            load_config_from_dict(raw)

    def test_bic_cv_scheme(self):
        raw = {
            **MINIMAL_CONFIG,
            "models": [{"name": "krr", "cv_scheme": "bic"}]
        }
        cfg = load_config_from_dict(raw)
        assert cfg.model_specs[0].cv_scheme == CVScheme.BIC

    def test_poos_cv_scheme(self):
        raw = {
            **MINIMAL_CONFIG,
            "models": [{"name": "krr", "cv_scheme": "poos"}]
        }
        cfg = load_config_from_dict(raw)
        assert cfg.model_specs[0].cv_scheme == CVScheme.POOS


class TestLoadConfigFromFile:
    def test_load_from_yaml_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(MINIMAL_CONFIG, f)
            tmp_path = Path(f.name)

        try:
            cfg = load_config(tmp_path)
            assert isinstance(cfg, ExperimentConfig)
            assert len(cfg.model_specs) == 2
        finally:
            tmp_path.unlink()

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.yaml")

    def test_default_config_yaml_is_parseable(self):
        """The bundled default template must be valid YAML and loadable."""
        raw = yaml.safe_load(DEFAULT_CONFIG_YAML)
        cfg = load_config_from_dict(raw)
        assert isinstance(cfg, ExperimentConfig)
        assert len(cfg.model_specs) >= 4  # krr, rf, xgboost, nn, lstm


class TestModelSpecKwargs:
    def test_kwargs_forwarded(self):
        cfg = load_config_from_dict(MINIMAL_CONFIG)
        krr = next(s for s in cfg.model_specs if s.model_cls is KRRModel)
        assert "alpha_grid" in krr.model_kwargs
        assert krr.model_kwargs["alpha_grid"] == [0.1, 1.0]

    def test_model_can_be_instantiated(self):
        cfg = load_config_from_dict(MINIMAL_CONFIG)
        for spec in cfg.model_specs:
            model = spec.build()
            assert model is not None
