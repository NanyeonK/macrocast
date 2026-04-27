from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="official_transform_scope",
    layer="1_data_task",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="target_only",
            description="apply official transformation codes to target only",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="predictors_only",
            description="apply official transformation codes to predictors only",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="target_and_predictors",
            description="apply official transformation codes to target and predictors",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="none",
            description="do not apply official transformation codes",
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
