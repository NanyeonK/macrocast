"""macrocast.pipeline — Layer 2: Forecasting experiment and model grid."""

from macrocast.pipeline.components import (
    CVScheme,
    CVSchemeType,
    LossFunction,
    Nonlinearity,
    Regularization,
    Window,
)
from macrocast.pipeline.estimator import MacrocastEstimator, SequenceEstimator
from macrocast.pipeline.experiment import FeatureSpec, ForecastExperiment, ModelSpec
from macrocast.pipeline.features import FeatureBuilder
from macrocast.pipeline.models import (
    KRRModel,
    LSTMModel,
    NNModel,
    RFModel,
    SVRLinearModel,
    SVRRBFModel,
    XGBoostModel,
)
from macrocast.pipeline.results import ForecastRecord, ResultSet

__all__ = [
    # components
    "CVScheme",
    "CVSchemeType",
    "LossFunction",
    "Nonlinearity",
    "Regularization",
    "Window",
    # estimator ABCs
    "MacrocastEstimator",
    "SequenceEstimator",
    # features
    "FeatureBuilder",
    # results
    "ForecastRecord",
    "ResultSet",
    # models
    "KRRModel",
    "SVRRBFModel",
    "SVRLinearModel",
    "RFModel",
    "XGBoostModel",
    "NNModel",
    "LSTMModel",
    # experiment
    "ModelSpec",
    "FeatureSpec",
    "ForecastExperiment",
]
