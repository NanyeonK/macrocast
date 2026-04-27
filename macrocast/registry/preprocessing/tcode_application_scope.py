from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name='tcode_application_scope',
    layer='2_preprocessing',
    axis_type='enum',
    default_policy='fixed',
    entries=(
        EnumRegistryEntry(
            id='target_only',
            description='t-code on target only',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='predictors_only',
            description='t-code on predictors only',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='target_and_predictors',
            description='t-code on target and X',
            status="operational",
            priority='A',
        ),
        EnumRegistryEntry(
            id='none',
            description='no t-code application',
            status='operational',
            priority='A',
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
