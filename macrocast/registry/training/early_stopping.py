from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='early_stopping',
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
            id='validation_patience',
            description='validation patience',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='loss_plateau',
            description='loss plateau',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='time_budget_stop',
            description='time budget stop',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='trial_pruning',
            description='trial pruning',
            status='registry_only',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
