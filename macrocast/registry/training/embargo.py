from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='embargo_gap',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='none',
            description='none',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='fixed_gap',
            description='fixed gap',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='horizon_gap',
            description='horizon gap',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='publication_gap',
            description='publication gap',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='custom_gap',
            description='custom gap',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
