from __future__ import annotations

from ..base import AxisDefinition, EnumRegistryEntry


FAILURE_POLICY_ENTRIES: tuple[EnumRegistryEntry, ...] = (
    EnumRegistryEntry(id="fail_fast", description="Abort immediately on execution failure.", status="operational", priority="A"),
    EnumRegistryEntry(id="skip_failed_cell", description="Skip failed cell and continue.", status="operational", priority="A"),
    EnumRegistryEntry(id="skip_failed_model", description="Skip failed model and continue.", status="operational", priority="A"),
    EnumRegistryEntry(id="save_partial_results", description="Persist partial results after failure.", status="operational", priority="A"),
    EnumRegistryEntry(id="warn_only", description="Record the failure in the manifest and emit a RuntimeWarning per failed unit; continue execution (both sweep and recipe layers).", status="operational", priority="A"),
    )

AXIS_DEFINITION = AxisDefinition(
    axis_name="failure_policy",
    layer="0_meta",
    axis_type="enum",
    entries=FAILURE_POLICY_ENTRIES,
    compatible_with={},
    incompatible_with={},
    default_policy="fixed",
)
