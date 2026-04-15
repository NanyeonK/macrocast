from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='tuning_objective',
    layer='3_training',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='validation_mse',
            description='validation mse',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='validation_rmse',
            description='validation rmse',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='validation_mae',
            description='validation mae',
            status='planned',
            priority='A',
        ),
        EnumRegistryEntry(
            id='validation_mape',
            description='validation mape',
            status='registry_only',
            priority='B',
        ),
        EnumRegistryEntry(
            id='validation_quantile_loss',
            description='validation quantile loss',
            status='future',
            priority='B',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
