from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TargetTransformPolicy = Literal[
    "raw_level",
    "tcode_transformed",
    "custom_target_transform",
]

XTransformPolicy = Literal[
    "raw_level",
    "dataset_tcode_transformed",
    "custom_x_transform",
]

TcodePolicy = Literal[
    "raw_only",
    "tcode_only",
    "tcode_then_extra_preprocess",
    "extra_preprocess_without_tcode",
    "extra_then_tcode",
    "custom_transform_pipeline",
]

MissingPolicy = Literal[
    "none",
    "drop",
    "em_impute",
    "custom",
]

OutlierPolicy = Literal[
    "none",
    "clip",
    "outlier_to_nan",
    "custom",
]

ScalingPolicy = Literal[
    "none",
    "standard",
    "robust",
    "minmax",
    "custom",
]

DimensionalityReductionPolicy = Literal[
    "none",
    "pca",
    "ipca",
    "custom",
]

FeatureSelectionPolicy = Literal[
    "none",
    "correlation_filter",
    "lasso_select",
    "custom",
]

PreprocessOrder = Literal[
    "none",
    "tcode_only",
    "extra_only",
    "tcode_then_extra",
    "extra_then_tcode",
    "custom",
]

PreprocessFitScope = Literal[
    "not_applicable",
    "train_only",
    "expanding_train_only",
    "rolling_train_only",
]

InverseTransformPolicy = Literal[
    "none",
    "target_only",
    "forecast_scale_only",
    "custom",
]

EvaluationScale = Literal[
    "raw_level",
    "transformed_scale",
]


@dataclass(frozen=True)
class PreprocessContract:
    target_transform_policy: str
    x_transform_policy: str
    tcode_policy: str
    target_missing_policy: str
    x_missing_policy: str
    target_outlier_policy: str
    x_outlier_policy: str
    scaling_policy: str
    dimensionality_reduction_policy: str
    feature_selection_policy: str
    preprocess_order: str
    preprocess_fit_scope: str
    inverse_transform_policy: str
    evaluation_scale: str
