# macrocast.pipeline

`macrocast.pipeline` contains the public API for model components, estimators, model classes, feature construction, experiment orchestration, and run results.

## Import

```python
from macrocast.pipeline import ForecastExperiment, FeatureBuilder, RFModel
```

## Components

Core component exports:
- `CVScheme`
- `CVSchemeType`
- `LossFunction`
- `Nonlinearity`
- `Regularization`
- `Window`

Purpose:
- define the high-level modeling choices used in forecasting experiments

## Estimator base classes

Estimator exports:
- `MacrocastEstimator`
- `SequenceEstimator`

Purpose:
- provide common interfaces for standard and sequence models

## Feature and experiment objects

Experiment-surface exports:
- `FeatureBuilder`
- `ModelSpec`
- `FeatureSpec`
- `ForecastExperiment`
- `HorseRaceGrid`
- `ForecastRecord`
- `ResultSet`

Purpose:
- define model inputs, experiment configuration, execution, and recorded outputs

## Python-side model classes

Python model exports:
- `KRRModel`
- `SVRRBFModel`
- `SVRLinearModel`
- `RFModel`
- `XGBoostModel`
- `GBModel`
- `NNModel`
- `LSTMModel`

Purpose:
- provide the main nonlinear / ML model implementations used by the package

## R-side and classical model classes

R-backed and classical exports:
- `RModelEstimator`
- `ARModel`
- `ARDIModel`
- `RidgeModel`
- `LassoModel`
- `AdaptiveLassoModel`
- `GroupLassoModel`
- `ElasticNetModel`
- `TVPRidgeModel`
- `BoogingModel`
- `BVARModel`

Purpose:
- expose additional linear, regularized, and classical benchmark families

## Registry helpers

Pipeline-registry exports:
- `get_feature_defaults`
- `get_model_defaults`
- `load_feature_registry`
- `load_model_registry`
- `validate_feature_model_compatibility`
- `validate_feature_registry`
- `validate_model_registry`

Purpose:
- expose canonical model/feature registry behavior and compatibility checks

## Related pages

- `User Guide > Stage 3`
- `Examples > Recipes & Runs`
