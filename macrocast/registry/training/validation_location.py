from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='validation_location',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='last_block',
            description='last block',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='rolling_blocks',
            description='rolling blocks',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='expanding_validation',
            description='expanding validation',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='blocked_cv',
            description='blocked cv',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='nested_time_cv',
            description='nested time cv',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
