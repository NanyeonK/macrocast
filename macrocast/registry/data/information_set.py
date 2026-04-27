from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='information_set_type',
    layer='1_data_task',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='final_revised_data',
            description='use final revised data',
            status='operational',
            priority='A',
        ),
        EnumRegistryEntry(
            id='pseudo_oos_on_revised_data',
            description='simulate pseudo out-of-sample splits on final revised data',
            status="operational",
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
