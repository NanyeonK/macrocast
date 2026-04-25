from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


AXIS_DEFINITION = AxisDefinition(
    axis_name="overlap_handling",
    layer="6_stat_tests",
    axis_type="enum",
    default_policy="fixed",
    entries=(
        EnumRegistryEntry(
            id="allow_overlap",
            description="Allow overlapping forecast-error rows without forcing HAC correction",
            status="operational",
            priority="A",
        ),
        EnumRegistryEntry(
            id="evaluate_with_hac",
            description="Require HAC-capable Layer 6 tests for overlapping long-horizon errors",
            status="operational",
            priority="A",
        ),
    ),
    compatible_with={},
    incompatible_with={},
)
