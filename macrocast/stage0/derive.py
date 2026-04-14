from __future__ import annotations

from .types import ComparisonContract, DesignShape, ExecutionPosture, ReplicationInput, VaryingDesign


def derive_design_shape(
    study_mode: str,
    varying_design: VaryingDesign,
) -> str:
    if study_mode == "orchestrated_bundle_study":
        return "wrapper_managed_multi_run_bundle"

    n_models = len(varying_design.model_families)
    n_control_axes = sum(
        bool(values)
        for values in (
            varying_design.feature_recipes,
            varying_design.preprocess_variants,
            varying_design.tuning_variants,
        )
    )

    if study_mode == "controlled_variation_study" or n_control_axes > 0:
        return "one_fixed_env_controlled_axis_variation"
    if n_models <= 1:
        return "one_fixed_env_one_tool_surface"
    return "one_fixed_env_multi_tool_surface"


def derive_execution_posture(
    study_mode: str,
    design_shape: str,
    replication_input: ReplicationInput | None,
) -> str:
    if replication_input is not None or study_mode == "replication_override_study":
        return "replication_locked_plan"
    if study_mode == "orchestrated_bundle_study" or design_shape == "wrapper_managed_multi_run_bundle":
        return "wrapper_bundle_plan"
    if design_shape == "one_fixed_env_controlled_axis_variation":
        return "single_run_with_internal_sweep"
    return "single_run_recipe"


def derive_experiment_unit(study_mode: str, execution_posture: str) -> str | None:
    if execution_posture == "wrapper_bundle_plan":
        return "orchestrated_bundle"
    if execution_posture == "replication_locked_plan":
        return "replication_override"
    if study_mode == "controlled_variation_study":
        return "controlled_variation"
    return "single_path"
