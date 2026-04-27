from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='contemporaneous_x_rule',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='allow_same_period_predictors',
            description='allow same-period predictors at forecast origin',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='forbid_same_period_predictors',
            description='forbid same-period predictors at forecast origin',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
