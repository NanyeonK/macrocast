from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='y_lag_count',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='fixed',
            description='fixed',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='cv_select',
            description='cv select',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='IC_select',
            description='IC select',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='model_specific',
            description='model specific',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
