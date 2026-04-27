from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='variable_universe',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='all_variables',
            description='use all variables in the dataset',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='core_variables',
            description='preselected core macro indicators',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='category_variables',
            description='subset by FRED category',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='target_specific_variables',
            description='subset chosen per target',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='explicit_variable_list',
            description='user-supplied column list via leaf_config.variable_universe_columns',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
