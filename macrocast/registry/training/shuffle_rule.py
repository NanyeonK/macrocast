from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='shuffle_rule',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='no_shuffle',
            description='no shuffle',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='restricted_shuffle_for_iid_only',
            description='restricted shuffle for iid only',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='groupwise_shuffle',
            description='groupwise shuffle',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='forbidden_for_time_series',
            description='forbidden for time series',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
