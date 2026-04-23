"""v0.9.2 Group B: preprocessing contract relaxations + runtime branches.

This batch adds two genuine runtime implementations. More Group-B axes
(tcode_application_scope, representation_policy:tcode_only, cv_select_lags)
require deeper infrastructure and land in a follow-up batch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.execution.build import (
    _apply_additional_preprocessing,
    _apply_x_lag_creation,
    _build_raw_panel_training_data,
)
from macrocast.preprocessing.build import PreprocessContract


def _contract(**overrides) -> PreprocessContract:
    base = dict(
        target_transform_policy="raw_level",
        x_transform_policy="raw_level",
        tcode_policy="extra_preprocess_without_tcode",
        target_missing_policy="none",
        x_missing_policy="none",
        target_outlier_policy="none",
        x_outlier_policy="none",
        scaling_policy="none",
        dimensionality_reduction_policy="none",
        feature_selection_policy="none",
        preprocess_order="extra_only",
        preprocess_fit_scope="train_only",
        inverse_transform_policy="none",
        evaluation_scale="raw_level",
    )
    base.update(overrides)
    return PreprocessContract(**base)


def test_fixed_x_lags_adds_lag1_columns():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
    Xp = pd.DataFrame({"a": [6.0], "b": [60.0]})
    c = _contract(x_lag_creation="fixed_x_lags")
    Xt, Xp2 = _apply_x_lag_creation(X, Xp, c)
    assert {"a", "b", "a__lag1", "b__lag1"}.issubset(set(Xt.columns))
    assert Xt["a__lag1"].iloc[1] == 1.0
    assert Xt["a__lag1"].iloc[0] == 0.0  # filled NaN
    assert Xt.shape == (5, 4)
    assert Xp2.shape == (1, 4)


def test_raw_panel_fixed_x_lag_prediction_uses_origin_history():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    c = _contract(x_lag_creation="fixed_x_lags")

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=3,
        contract=c,
        predictor_family="all_macro_vars",
    )

    assert X_train.tolist() == [[1.0, 0.0], [2.0, 1.0], [3.0, 2.0]]
    assert y_train.tolist() == [11.0, 12.0, 13.0]
    assert X_pred.tolist() == [[4.0, 3.0]]


def test_raw_panel_target_level_addback_uses_origin_target_history():
    frame = pd.DataFrame(
        {
            "target": [10.0, 11.0, 12.0, 13.0, 14.0],
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    c = _contract()

    X_train, y_train, X_pred = _build_raw_panel_training_data(
        frame,
        "target",
        horizon=1,
        start_idx=0,
        origin_idx=3,
        contract=c,
        predictor_family="all_macro_vars",
        level_feature_block="target_level_addback",
    )

    assert X_train.tolist() == [[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]]
    assert y_train.tolist() == [11.0, 12.0, 13.0]
    assert X_pred.tolist() == [[4.0, 13.0]]


def test_no_x_lags_is_identity():
    X = pd.DataFrame({"a": [1.0, 2.0]})
    Xp = pd.DataFrame({"a": [3.0]})
    c = _contract()
    Xt, Xp2 = _apply_x_lag_creation(X, Xp, c)
    assert Xt.equals(X)
    assert Xp2.equals(Xp)


def test_hp_filter_shifts_column_mean_toward_zero():
    rng = np.random.default_rng(0)
    # Construct a series with a strong trend — HP filter removes trend
    trend = np.linspace(0, 100, 60)
    noise = rng.standard_normal(60) * 0.5
    X = pd.DataFrame({"a": trend + noise, "b": trend * 0.5 + noise})
    Xp = pd.DataFrame({"a": [50.0], "b": [25.0]})
    c = _contract(additional_preprocessing="hp_filter")
    Xt, _ = _apply_additional_preprocessing(X, Xp, c)
    # cycle component has mean near zero (vs original ~50)
    assert abs(Xt["a"].mean()) < 1.0
    assert abs(X["a"].mean()) > 40.0  # sanity: original had big mean


def test_hp_filter_none_is_identity():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    Xp = pd.DataFrame({"a": [4.0]})
    c = _contract()
    Xt, Xp2 = _apply_additional_preprocessing(X, Xp, c)
    assert Xt.equals(X)
    assert Xp2.equals(Xp)


def test_registry_promotions_are_operational():
    from macrocast.registry.build import _discover_axis_definitions

    defs = _discover_axis_definitions()

    def _status(axis, value):
        return next(e.status for e in defs[axis].entries if e.id == value)

    assert _status("additional_preprocessing", "hp_filter") == "operational"
    assert _status("x_lag_creation", "fixed_x_lags") == "operational"


def test_contract_accepts_hp_filter():
    from macrocast.preprocessing.build import is_operational_preprocess_contract

    c = _contract(additional_preprocessing="hp_filter")
    assert is_operational_preprocess_contract(c)


def test_contract_accepts_fixed_x_lags():
    from macrocast.preprocessing.build import is_operational_preprocess_contract

    c = _contract(x_lag_creation="fixed_x_lags")
    assert is_operational_preprocess_contract(c)
