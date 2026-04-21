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
            id='preselected_core',
            description='preselected core macro indicators',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='category_subset',
            description='subset by FRED category',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='target_specific_subset',
            description='subset chosen per target',
            status='operational',
            priority='B',
        ),
        EnumRegistryEntry(
            id='handpicked_set',
            description='user-supplied column list via leaf_config.variable_universe_columns',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
