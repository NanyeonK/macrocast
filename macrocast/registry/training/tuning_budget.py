from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='tuning_budget',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='max_trials',
            description='max trials',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='max_time',
            description='max time',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='max_epochs',
            description='max epochs',
            status='future',
            priority='B',
        ),
        EnumRegistryEntry(
            id='max_models',
            description='max models',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='early_stop_trials',
            description='early stop trials',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
