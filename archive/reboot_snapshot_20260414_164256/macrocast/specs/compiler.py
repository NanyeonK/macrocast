from __future__ import annotations

from pathlib import Path
from typing import Any

from macrocast.design.resolver import ResolvedExperimentSpec
from macrocast.design.resolver import resolve_experiment_spec as _resolve_experiment_spec
from macrocast.design.resolver import resolve_experiment_spec_from_dict as _resolve_experiment_spec_from_dict
from macrocast.design.resolver import resolve_experiment_spec_from_experiment_config as _resolve_experiment_spec_from_experiment_config
from macrocast.meta import load_axes_registry
from macrocast.recipes import (
    build_recipe_experiment_overrides,
    build_recipe_resolution_context,
    load_recipe,
    load_recipe_schema,
    recipe_to_experiment_config,
    recipe_to_runtime_config,
    validate_recipe,
    validate_recipe_schema,
)
from macrocast.tree_context import derive_tree_context_from_compiled_spec

CompiledExperimentSpec = ResolvedExperimentSpec


def validate_compiled_experiment_spec(spec: CompiledExperimentSpec) -> CompiledExperimentSpec:
    return spec.validate_compiled_spec()


def compile_experiment_spec(path: str | Path, *, preset_id: str | None = None, experiment_overrides: dict[str, Any] | None = None) -> CompiledExperimentSpec:
    return validate_compiled_experiment_spec(_resolve_experiment_spec(path, preset_id=preset_id, experiment_overrides=experiment_overrides))


def compile_experiment_spec_from_dict(raw: dict[str, Any], *, preset_id: str | None = None, experiment_overrides: dict[str, Any] | None = None) -> CompiledExperimentSpec:
    return validate_compiled_experiment_spec(_resolve_experiment_spec_from_dict(raw, preset_id=preset_id, experiment_overrides=experiment_overrides))


def compile_experiment_spec_from_recipe(recipe_path: str, *, preset_id: str | None = None, experiment_overrides: dict[str, Any] | None = None) -> CompiledExperimentSpec:
    schema = validate_recipe_schema(load_recipe_schema())
    recipe = validate_recipe(load_recipe(recipe_path), schema)
    resolution_context = build_recipe_resolution_context(recipe)
    exp_cfg = recipe_to_experiment_config(recipe)
    direct_overrides = build_recipe_experiment_overrides(recipe)
    merged_overrides = dict(direct_overrides)
    merged_overrides.update(experiment_overrides or {})
    compiled = validate_compiled_experiment_spec(_resolve_experiment_spec_from_experiment_config(exp_cfg, preset_id=preset_id, experiment_overrides=merged_overrides))
    compiled.meta_config['recipe_id'] = recipe['recipe_id']
    compiled.meta_config['recipe_kind'] = recipe['kind']
    compiled.meta_config['taxonomy_path'] = recipe['taxonomy_path']
    compiled.meta_config['numeric_params'] = recipe['numeric_params']
    compiled.meta_config['output_prefs'] = recipe['outputs']
    compiled.meta_config['recipe_resolution_context'] = resolution_context
    compiled.meta_config['compile_path'] = 'recipe_native_experiment_config'
    compiled.meta_config['tree_context'] = derive_tree_context_from_compiled_spec(compiled.meta_config, load_axes_registry())
    return compiled

__all__ = ['CompiledExperimentSpec','validate_compiled_experiment_spec','compile_experiment_spec','compile_experiment_spec_from_dict','compile_experiment_spec_from_recipe']
