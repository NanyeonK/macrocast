"""Tests for macrocast/pipeline/presets.py."""

from __future__ import annotations

import pytest

from macrocast.pipeline.components import LossFunction, Regularization
from macrocast.pipeline.experiment import FeatureSpec
from macrocast.replication.clss2021 import CLSS2021, get_preset


# ---------------------------------------------------------------------------
# CLSS2021.info_sets
# ---------------------------------------------------------------------------


class TestCLSS2021InfoSets:
    def test_returns_16_sets(self) -> None:
        sets = CLSS2021.info_sets()
        assert len(sets) == 16

    def test_keys_match_table1_labels(self) -> None:
        sets = CLSS2021.info_sets()
        assert set(sets.keys()) == set(CLSS2021.TABLE1_LABELS)

    def test_all_values_are_feature_spec(self) -> None:
        sets = CLSS2021.info_sets()
        for label, spec in sets.items():
            assert isinstance(spec, FeatureSpec), f"{label} is not FeatureSpec"

    def test_default_params(self) -> None:
        sets = CLSS2021.info_sets()
        # F uses n_factors=8, n_lags=12, p_marx=12 by default
        f = sets["F"]
        assert f.n_factors == CLSS2021.DEFAULT_K
        assert f.n_lags == CLSS2021.DEFAULT_P_Y
        assert f.p_marx == CLSS2021.DEFAULT_P_MARX

    def test_custom_params_propagate(self) -> None:
        sets = CLSS2021.info_sets(P_Y=6, K=4, P_MARX=6)
        for spec in sets.values():
            assert spec.n_factors == 4
            assert spec.n_lags == 6
            assert spec.p_marx == 6

    # --- F-type specs ---

    def test_F_factor_type(self) -> None:
        spec = CLSS2021.info_sets()["F"]
        assert spec.factor_type == "X"
        assert not spec.append_marx
        assert not spec.append_raw_x
        assert not spec.append_levels

    def test_F_X(self) -> None:
        spec = CLSS2021.info_sets()["F-X"]
        assert spec.factor_type == "X"
        assert spec.append_raw_x
        assert not spec.append_marx

    def test_F_MARX(self) -> None:
        spec = CLSS2021.info_sets()["F-MARX"]
        assert spec.factor_type == "X"
        assert spec.append_marx
        assert not spec.append_raw_x

    def test_F_MAF(self) -> None:
        spec = CLSS2021.info_sets()["F-MAF"]
        assert spec.factor_type == "MARX"
        assert spec.append_x_factors

    def test_F_Level(self) -> None:
        spec = CLSS2021.info_sets()["F-Level"]
        assert spec.factor_type == "X"
        assert spec.append_levels
        assert not spec.append_marx

    def test_F_X_MARX_Level(self) -> None:
        spec = CLSS2021.info_sets()["F-X-MARX-Level"]
        assert spec.factor_type == "X"
        assert spec.append_raw_x
        assert spec.append_marx
        assert spec.append_levels

    # --- No-factor specs ---

    def test_X(self) -> None:
        spec = CLSS2021.info_sets()["X"]
        assert spec.factor_type == "none"
        assert spec.append_raw_x
        assert not spec.append_marx

    def test_MARX(self) -> None:
        spec = CLSS2021.info_sets()["MARX"]
        assert spec.factor_type == "none"
        assert spec.append_marx
        assert not spec.append_raw_x

    def test_MAF(self) -> None:
        spec = CLSS2021.info_sets()["MAF"]
        assert spec.factor_type == "MARX"
        assert not spec.append_x_factors
        assert not spec.append_raw_x

    def test_X_MARX_Level(self) -> None:
        spec = CLSS2021.info_sets()["X-MARX-Level"]
        assert spec.factor_type == "none"
        assert spec.append_raw_x
        assert spec.append_marx
        assert spec.append_levels

    # --- Auto-labels ---

    def test_auto_labels_match_dict_keys(self) -> None:
        """Each FeatureSpec's label should equal its dict key."""
        sets = CLSS2021.info_sets()
        for key, spec in sets.items():
            assert spec.label == key, f"label mismatch: key={key!r}, label={spec.label!r}"


# ---------------------------------------------------------------------------
# CLSS2021 model specs
# ---------------------------------------------------------------------------


class TestCLSS2021ModelSpecs:
    def test_rf_spec(self) -> None:
        spec = CLSS2021.rf_spec()
        assert spec.regularization == Regularization.NONE
        assert spec.loss_function == LossFunction.L2
        assert spec.model_kwargs["n_estimators"] == 200
        assert spec.model_kwargs["min_samples_leaf"] == 5

    def test_en_spec(self) -> None:
        spec = CLSS2021.en_spec()
        assert spec.regularization == Regularization.ELASTIC_NET
        assert spec.model_id == "EN"

    def test_al_spec(self) -> None:
        spec = CLSS2021.al_spec()
        assert spec.regularization == Regularization.ADAPTIVE_LASSO
        assert spec.model_id == "AL"

    def test_ardi_spec(self) -> None:
        from macrocast.pipeline.components import CVScheme
        spec = CLSS2021.ardi_spec()
        assert spec.cv_scheme == CVScheme.BIC
        assert spec.model_id == "FM"

    def test_all_model_specs_returns_four(self) -> None:
        specs = CLSS2021.all_model_specs()
        assert len(specs) == 4
        ids = {s.model_id for s in specs}
        assert ids == {"RF", "EN", "AL", "FM"}

    def test_rf_spec_custom_params(self) -> None:
        spec = CLSS2021.rf_spec(n_estimators=100, min_samples_leaf=10)
        assert spec.model_kwargs["n_estimators"] == 100
        assert spec.model_kwargs["min_samples_leaf"] == 10


# ---------------------------------------------------------------------------
# get_preset
# ---------------------------------------------------------------------------


class TestGetPreset:
    def test_returns_correct_spec(self) -> None:
        spec = get_preset("F-MARX")
        assert spec.factor_type == "X"
        assert spec.append_marx

    def test_custom_params_forwarded(self) -> None:
        spec = get_preset("F", K=4, P_Y=6)
        assert spec.n_factors == 4
        assert spec.n_lags == 6

    def test_unknown_name_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown info set"):
            get_preset("NOT_A_REAL_SET")

    def test_unknown_study_raises_key_error(self) -> None:
        with pytest.raises(KeyError, match="Unknown study"):
            get_preset("F", study="unknown_study")

    def test_all_table1_labels_accessible(self) -> None:
        for label in CLSS2021.TABLE1_LABELS:
            spec = get_preset(label)
            assert isinstance(spec, FeatureSpec)


# ---------------------------------------------------------------------------
# FeatureSpec.from_name (classmethod bridge)
# ---------------------------------------------------------------------------


class TestFeatureSpecFromName:
    def test_from_name_basic(self) -> None:
        spec = FeatureSpec.from_name("F-MARX")
        assert spec.factor_type == "X"
        assert spec.append_marx

    def test_from_name_with_params(self) -> None:
        spec = FeatureSpec.from_name("F", K=4, P_Y=6)
        assert spec.n_factors == 4
        assert spec.n_lags == 6

    def test_from_name_unknown_raises(self) -> None:
        with pytest.raises(KeyError):
            FeatureSpec.from_name("NONEXISTENT")

    def test_from_name_maf(self) -> None:
        spec = FeatureSpec.from_name("MAF")
        assert spec.factor_type == "MARX"
        assert not spec.append_x_factors


# ---------------------------------------------------------------------------
# HORIZONS and TABLE1_LABELS class attributes
# ---------------------------------------------------------------------------


def test_horizons() -> None:
    assert CLSS2021.HORIZONS == [1, 3, 6, 9, 12, 24]


def test_table1_labels_count() -> None:
    assert len(CLSS2021.TABLE1_LABELS) == 16
