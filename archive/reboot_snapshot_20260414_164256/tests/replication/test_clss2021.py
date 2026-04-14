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
        assert spec.model_kwargs["min_samples_leaf_grid"] == [5]

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

    def test_all_model_specs_returns_six(self) -> None:
        specs = CLSS2021.all_model_specs()
        assert len(specs) == 6
        ids = {s.model_id for s in specs}
        assert ids == {"RF", "EN", "AL", "FM", "KRR", "SVR"}

    def test_rf_spec_custom_params(self) -> None:
        spec = CLSS2021.rf_spec(n_estimators=100, min_samples_leaf=10)
        assert spec.model_kwargs["n_estimators"] == 100
        assert spec.model_kwargs["min_samples_leaf_grid"] == [10]


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


# ---------------------------------------------------------------------------
# End-to-end smoke test: ForecastExperiment with CLSS2021 presets
# ---------------------------------------------------------------------------


def _make_synthetic_panel(
    T: int = 200, N: int = 20, seed: int = 42
) -> tuple["pd.DataFrame", "pd.Series"]:
    """Synthetic FRED-MD-like panel for CI-speed replication smoke tests."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01", periods=T, freq="MS")
    X = rng.standard_normal((T, N))
    df = pd.DataFrame(X, index=dates, columns=[f"X{i:02d}" for i in range(N)])
    target = pd.Series(
        0.5 * X[:, 0] - 0.3 * X[:, 2] + rng.standard_normal(T) * 0.2,
        index=dates,
        name="INDPRO",
    )
    return df, target


class TestCLSS2021EndToEnd:
    """Smoke tests: full ForecastExperiment run using CLSS2021 presets."""

    def test_rf_e2e_produces_records(self) -> None:
        """RF with F and F-MARX info sets returns non-empty ResultSet."""
        import pandas as pd
        from macrocast.pipeline.components import Window
        from macrocast.pipeline.experiment import ForecastExperiment

        df, target = _make_synthetic_panel()
        info_sets = CLSS2021.info_sets(P_Y=2, K=3, P_MARX=2)
        rf_spec = CLSS2021.rf_spec(n_estimators=20, min_samples_leaf=5)

        results = []
        for label in ["F", "F-MARX"]:
            exp = ForecastExperiment(
                panel=df,
                target=target,
                horizons=[1],
                model_specs=[rf_spec],
                feature_spec=info_sets[label],
                window=Window.EXPANDING,
                oos_start="2012-01",
                n_jobs=1,
            )
            rs = exp.run()
            df_r = rs.to_dataframe()
            assert len(df_r) > 0, f"Empty ResultSet for feature set {label!r}"
            results.append(df_r)

    def test_feature_set_labels_in_output(self) -> None:
        """ResultSet feature_set column matches the FeatureSpec label."""
        from macrocast.pipeline.components import Window
        from macrocast.pipeline.experiment import ForecastExperiment

        df, target = _make_synthetic_panel()
        info_sets = CLSS2021.info_sets(P_Y=2, K=3, P_MARX=2)
        rf_spec = CLSS2021.rf_spec(n_estimators=20, min_samples_leaf=5)

        for label in ["F", "F-MARX"]:
            exp = ForecastExperiment(
                panel=df,
                target=target,
                horizons=[1],
                model_specs=[rf_spec],
                feature_spec=info_sets[label],
                window=Window.EXPANDING,
                oos_start="2012-01",
                n_jobs=1,
            )
            rs = exp.run()
            df_r = rs.to_dataframe()
            assert (df_r["feature_set"] == label).all(), (
                f"Expected feature_set={label!r}, got {df_r['feature_set'].unique()}"
            )

    def test_finite_forecasts(self) -> None:
        """All y_hat values are finite (no NaN/Inf from RF)."""
        import numpy as np
        from macrocast.pipeline.components import Window
        from macrocast.pipeline.experiment import ForecastExperiment

        df, target = _make_synthetic_panel()
        info_sets = CLSS2021.info_sets(P_Y=2, K=3, P_MARX=2)
        rf_spec = CLSS2021.rf_spec(n_estimators=20, min_samples_leaf=5)

        exp = ForecastExperiment(
            panel=df,
            target=target,
            horizons=[1],
            model_specs=[rf_spec],
            feature_spec=info_sets["F-MARX"],
            window=Window.EXPANDING,
            oos_start="2012-01",
            n_jobs=1,
        )
        rs = exp.run()
        df_r = rs.to_dataframe()
        assert np.isfinite(df_r["y_hat"].values).all(), "Non-finite forecasts in y_hat"


class TestCLSS2021NewSpecs:
    """Tests for krr_spec, svr_spec, and all_model_specs completeness."""

    def test_krr_spec_kwargs(self) -> None:
        spec = CLSS2021.krr_spec()
        assert spec.model_id == "KRR"
        assert spec.model_kwargs["alpha_grid"] == [0.001, 0.01, 0.1, 1.0, 10.0]
        assert spec.model_kwargs["gamma_grid"] == [0.001, 0.01, 0.1, 1.0]
        assert spec.model_kwargs["cv_folds"] == 5

    def test_krr_spec_custom_grids(self) -> None:
        spec = CLSS2021.krr_spec(alpha_grid=[0.1, 1.0], gamma_grid=[0.01])
        assert spec.model_kwargs["alpha_grid"] == [0.1, 1.0]
        assert spec.model_kwargs["gamma_grid"] == [0.01]

    def test_svr_spec_kwargs(self) -> None:
        spec = CLSS2021.svr_spec()
        assert spec.model_id == "SVR"
        assert spec.model_kwargs["C_grid"] == [0.1, 1.0, 10.0, 100.0]
        assert spec.model_kwargs["gamma_grid"] == [0.001, 0.01, 0.1, 1.0]
        assert spec.model_kwargs["epsilon_grid"] == [0.01, 0.1]

    def test_svr_spec_custom_grids(self) -> None:
        spec = CLSS2021.svr_spec(C_grid=[1.0, 10.0], epsilon_grid=[0.05])
        assert spec.model_kwargs["C_grid"] == [1.0, 10.0]
        assert spec.model_kwargs["epsilon_grid"] == [0.05]

    def test_all_model_specs_count(self) -> None:
        specs = CLSS2021.all_model_specs()
        assert len(specs) == 6

    def test_all_model_specs_ids(self) -> None:
        ids = {s.model_id for s in CLSS2021.all_model_specs()}
        assert ids == {"RF", "EN", "AL", "FM", "KRR", "SVR"}
