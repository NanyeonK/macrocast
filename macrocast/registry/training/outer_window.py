from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='outer_window',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='expanding',
            description='expanding',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='rolling',
            description='rolling',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='anchored_rolling',
            description='anchored rolling',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='hybrid_expanding_rolling',
            description='hybrid expanding rolling',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='recursive_reestimation',
            description='recursive reestimation',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
