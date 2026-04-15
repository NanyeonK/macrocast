from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='split_family',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='simple_holdout',
            description='simple holdout',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='time_split',
            description='time split',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='blocked_kfold',
            description='blocked kfold',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='expanding_cv',
            description='expanding cv',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='rolling_cv',
            description='rolling cv',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
