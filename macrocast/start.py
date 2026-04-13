from __future__ import annotations

from typing import Any, Iterable

from macrocast.meta import load_axes_registry
from macrocast.output import build_run_manifest, ensure_output_dirs
from macrocast.registries import load_registry_bundle
from macrocast.specs.compiler import compile_experiment_spec_from_recipe
from macrocast.tree_context import derive_tree_context_from_compiled_spec

_AVAILABLE_STAGES = (
    'axes',
    'registries',
    'compile',
    'tree_context',
    'runs_preview',
    'manifest_preview',
)


def _normalize_stages(stages: str | Iterable[str] | None) -> list[str]:
    if stages is None or stages == 'all':
        return list(_AVAILABLE_STAGES)
    if isinstance(stages, str):
        stages = [stages]
    out = []
    for stage in stages:
        if stage not in _AVAILABLE_STAGES:
            raise ValueError(f'unknown stage {stage!r}; valid stages: {_AVAILABLE_STAGES}')
        out.append(stage)
    return out


def macrocast_start(*, recipe_path: str = 'baselines/minimal_fred_md.yaml', preset_id: str = 'researcher_explicit', stages: str | Iterable[str] | None = None, output_root: str = '/tmp/macrocast_start_preview') -> dict[str, Any]:
    selected = _normalize_stages(stages)
    out: dict[str, Any] = {'selected_stages': selected}

    axes = None
    compiled = None
    tree_context = None

    if 'axes' in selected:
        axes = load_axes_registry()
        out['axes'] = {
            'unit_of_run': axes.get('unit_of_run', {}),
            'fixed_axes_summary': axes.get('fixed_axes_summary', {}),
            'sweep_axes_summary': axes.get('sweep_axes_summary', {}),
        }

    if 'registries' in selected:
        bundle = load_registry_bundle()
        out['registries'] = {layer: sorted(files.keys()) for layer, files in bundle.items()}

    if any(stage in selected for stage in ('compile', 'tree_context', 'runs_preview', 'manifest_preview')):
        compiled = compile_experiment_spec_from_recipe(recipe_path, preset_id=preset_id)

    if 'compile' in selected and compiled is not None:
        out['compile'] = compiled.to_contract_dict() | {
            'recipe_id': compiled.meta_config.get('recipe_id'),
            'recipe_kind': compiled.meta_config.get('recipe_kind'),
            'taxonomy_path': compiled.meta_config.get('taxonomy_path', {}),
            'compile_path': compiled.meta_config.get('compile_path'),
        }

    if any(stage in selected for stage in ('tree_context', 'runs_preview', 'manifest_preview')) and compiled is not None:
        axes = axes or load_axes_registry()
        tree_context = derive_tree_context_from_compiled_spec(compiled.meta_config, axes)

    if 'tree_context' in selected and tree_context is not None:
        out['tree_context'] = tree_context

    if 'runs_preview' in selected and compiled is not None:
        dirs = ensure_output_dirs(output_root, compiled.experiment_config.experiment_id, recipe_id=compiled.meta_config.get('recipe_id'), taxonomy_path=compiled.meta_config.get('taxonomy_path'))
        out['runs_preview'] = {k: str(v) for k, v in dirs.items()}

    if 'manifest_preview' in selected and compiled is not None:
        tree_context = tree_context or derive_tree_context_from_compiled_spec(compiled.meta_config, axes or load_axes_registry())
        manifest = build_run_manifest(
            run_id='macrocast-start-preview',
            experiment_id=compiled.experiment_config.experiment_id,
            recipe_id=compiled.meta_config.get('recipe_id'),
            taxonomy_path=compiled.meta_config.get('taxonomy_path'),
            tree_context=tree_context,
            config_hash='preview',
            code_version='preview',
            dataset_ids=[compiled.meta_config.get('dataset')],
            benchmark_ids=[compiled.meta_config.get('benchmark_id')],
            artifact_paths={},
        )
        out['manifest_preview'] = manifest

    return out
