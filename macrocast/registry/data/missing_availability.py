from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='missing_availability',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='require_complete_rows',
            description='drop rows with any missing values',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='keep_available_rows',
            description='keep rows with available cases only (per-series)',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='impute_predictors_only',
            description='impute predictors only and keep target missingness strict',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='zero_fill_leading_predictor_gaps',
            description='fill predictor leading missing values with zero and report availability gaps',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
