from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='validation_size_rule',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='ratio',
            description='ratio',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fixed_n',
            description='fixed n',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fixed_years',
            description='fixed years',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fixed_dates',
            description='fixed dates',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='horizon_specific_n',
            description='horizon specific n',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
