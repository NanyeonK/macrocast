from __future__ import annotations

from .derive import derive_design_shape, derive_execution_posture, derive_experiment_unit
from .errors import Stage0CompletenessError, Stage0RoutingError
from .normalize import (
    normalize_comparison_contract,
    normalize_fixed_design,
    normalize_replication_input,
    normalize_study_mode,
    normalize_varying_design,
)
from .types import ComparisonContract, FixedDesign, ReplicationInput, Stage0Frame, VaryingDesign
from .validate import validate_stage0_frame


def build_stage0_frame(
    *,
    study_mode: str,
    fixed_design: FixedDesign | dict,
    comparison_contract: ComparisonContract | dict,
    varying_design: VaryingDesign | dict | None = None,
    replication_input: ReplicationInput | dict | None = None,
) -> Stage0Frame:
    normalized_study_mode = normalize_study_mode(study_mode)
    normalized_fixed_design = normalize_fixed_design(fixed_design)
    normalized_comparison_contract = normalize_comparison_contract(comparison_contract)
    normalized_varying_design = normalize_varying_design(varying_design)
    normalized_replication_input = normalize_replication_input(replication_input)

    design_shape = derive_design_shape(
        normalized_study_mode,
        normalized_varying_design,
    )
    execution_posture = derive_execution_posture(
        normalized_study_mode,
        design_shape,
        normalized_replication_input,
    )
    experiment_unit = derive_experiment_unit(normalized_study_mode, execution_posture)

    stage0 = Stage0Frame(
        study_mode=normalized_study_mode,
        fixed_design=normalized_fixed_design,
        comparison_contract=normalized_comparison_contract,
        varying_design=normalized_varying_design,
        execution_posture=execution_posture,
        design_shape=design_shape,
        replication_input=normalized_replication_input,
        experiment_unit=experiment_unit,
    )
    validate_stage0_frame(stage0)
    return stage0


def resolve_route_owner(stage0: Stage0Frame) -> str:
    if stage0.execution_posture == "wrapper_bundle_plan":
        return "wrapper"
    if stage0.execution_posture == "replication_locked_plan":
        return "replication"
    if stage0.execution_posture in {"single_run_recipe", "single_run_with_internal_sweep"}:
        return "single_run"
    raise Stage0RoutingError(f"unknown execution_posture={stage0.execution_posture!r}")


def check_stage0_completeness(stage0: Stage0Frame) -> None:
    if stage0.execution_posture in {"single_run_recipe", "single_run_with_internal_sweep"}:
        if not stage0.varying_design.model_families:
            raise Stage0CompletenessError(
                "stage0 requires at least one model family for single-run execution"
            )


def stage0_summary(stage0: Stage0Frame) -> str:
    models = ", ".join(stage0.varying_design.model_families) or "none"
    horizons = ", ".join(stage0.varying_design.horizons) or "none"
    return (
        f"study_mode={stage0.study_mode}; "
        f"dataset={stage0.fixed_design.dataset_adapter}; "
        f"route={resolve_route_owner(stage0)}; "
        f"execution_posture={stage0.execution_posture}; "
        f"design_shape={stage0.design_shape}; "
        f"models=[{models}]; horizons=[{horizons}]"
    )
