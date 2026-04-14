from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from macrocast.choice_stack import STAGE0_META_KEYS, build_choice_stack, build_yaml_preview
from macrocast.meta import load_axes_registry
from macrocast.output import build_run_manifest, ensure_output_dirs
from macrocast.recipes import load_recipe_schema, validate_recipe_schema, validate_recipe
from macrocast.registries import load_registry_bundle
from macrocast.specs.compiler import compile_experiment_spec_from_recipe
from macrocast.tree_context import derive_tree_context_from_compiled_spec

_AVAILABLE_STAGES = (
    'axes',
    'registries',
    'choice_stack',
    'yaml_preview',
    'compile',
    'tree_context',
    'runs_preview',
    'manifest_preview',
)

_EXPERIMENT_UNIT_ROUTES: dict[str, dict[str, Any]] = {
    'single_target_single_model': {
        'owner': 'single_run',
        'status': 'implemented',
        'shape': 'true_single_path',
        'compile_allowed': True,
        'continue_in_single_run': True,
        'message': 'Current macrocast_single_run path is fully aligned to one target + one model.',
    },
    'single_target_model_grid': {
        'owner': 'single_run',
        'status': 'planned_single_run_extension',
        'shape': 'single_target_model_sweep',
        'compile_allowed': False,
        'continue_in_single_run': False,
        'message': 'Model-grid branching belongs inside macrocast_single_run, but current downstream branching is not implemented yet.',
    },
    'single_target_full_sweep': {
        'owner': 'single_run',
        'status': 'planned_single_run_extension',
        'shape': 'single_target_multi_axis_sweep',
        'compile_allowed': False,
        'continue_in_single_run': False,
        'message': 'Full single-target sweep still belongs to macrocast_single_run, but current downstream branching is not implemented yet.',
    },
    'multi_target_separate_runs': {
        'owner': 'wrapper_orchestrator',
        'status': 'wrapper_required',
        'shape': 'multi_run_bundle',
        'compile_allowed': False,
        'continue_in_single_run': False,
        'message': 'Multi-target separate runs must fan out into multiple single-run YAMLs and belong in a wrapper/orchestrator.',
    },
    'multi_target_shared_design': {
        'owner': 'wrapper_orchestrator',
        'status': 'wrapper_required',
        'shape': 'shared_design_multi_run_bundle',
        'compile_allowed': False,
        'continue_in_single_run': False,
        'message': 'Multi-target shared-design execution belongs in a wrapper/orchestrator that manages shared fixed axes across emitted runs.',
    },
    'replication_recipe': {
        'owner': 'wrapper_orchestrator',
        'status': 'wrapper_required',
        'shape': 'prebuilt_recipe_bundle',
        'compile_allowed': False,
        'continue_in_single_run': False,
        'message': 'Replication recipes should load pre-authored package recipes or recipe bundles, not branch the generic single-run wizard.',
    },
    'benchmark_suite': {
        'owner': 'wrapper_orchestrator',
        'status': 'wrapper_required',
        'shape': 'benchmark_comparison_suite',
        'compile_allowed': False,
        'continue_in_single_run': False,
        'message': 'Benchmark suites compare multiple runs and belong in a future suite/orchestrator entry point.',
    },
    'ablation_study': {
        'owner': 'wrapper_orchestrator',
        'status': 'wrapper_required',
        'shape': 'controlled_comparison_suite',
        'compile_allowed': False,
        'continue_in_single_run': False,
        'message': 'Ablation studies manage coordinated counterfactual runs and belong in a wrapper/orchestrator.',
    },
}


def _normalize_stages(stages: str | Iterable[str] | None, *, default: list[str] | None = None) -> list[str]:
    if stages is None:
        return list(default or ['choice_stack'])
    if stages == 'all':
        return list(_AVAILABLE_STAGES)
    if isinstance(stages, str):
        stages = [stages]
    out = []
    for stage in stages:
        if stage not in _AVAILABLE_STAGES:
            raise ValueError(f'unknown stage {stage!r}; valid stages: {_AVAILABLE_STAGES}')
        out.append(stage)
    return out


def _ensure_meta(recipe: dict[str, Any]) -> dict[str, Any]:
    meta = recipe.setdefault('meta', {})
    if not isinstance(meta, dict):
        raise ValueError('recipe meta block must be a dict when present')
    meta.setdefault('experiment_unit', 'single_target_single_model')
    return meta


def _route_for_experiment_unit(experiment_unit: str | None) -> dict[str, Any]:
    unit = experiment_unit or 'single_target_single_model'
    base = _EXPERIMENT_UNIT_ROUTES.get(
        unit,
        {
            'owner': 'wrapper_orchestrator',
            'status': 'unknown_requires_design',
            'shape': 'unknown',
            'compile_allowed': False,
            'continue_in_single_run': False,
            'message': f'Unknown experiment_unit={unit!r}; treat as wrapper/orchestrator work until explicitly designed.',
        },
    )
    return {'experiment_unit': unit, **base}


def _route_for_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    meta = _ensure_meta(recipe)
    return _route_for_experiment_unit(meta.get('experiment_unit'))


def _load_user_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        recipe = yaml.safe_load(f) or {}
    schema = validate_recipe_schema(load_recipe_schema())
    recipe = validate_recipe(recipe, schema)
    _ensure_meta(recipe)
    return recipe


def _write_recipe_yaml(recipe: dict[str, Any], yaml_path: str | Path) -> Path:
    path = Path(yaml_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding='utf-8')
    return path


def _get_current_value(recipe: dict[str, Any], key: str) -> Any:
    if key in STAGE0_META_KEYS:
        return _ensure_meta(recipe).get(key)
    if key in recipe.get('taxonomy_path', {}):
        return recipe['taxonomy_path'][key]
    if key in recipe.get('numeric_params', {}):
        return recipe['numeric_params'][key]
    if key in recipe.get('outputs', {}):
        return recipe['outputs'][key]
    return recipe.get('custom_selections', {}).get(key)


def _set_value(recipe: dict[str, Any], key: str, value: Any) -> None:
    if key in STAGE0_META_KEYS:
        _ensure_meta(recipe)[key] = value
    elif key in recipe.get('taxonomy_path', {}):
        recipe['taxonomy_path'][key] = value
    elif key in recipe.get('numeric_params', {}):
        recipe['numeric_params'][key] = value
    elif key in recipe.get('outputs', {}):
        recipe['outputs'][key] = value
    else:
        recipe.setdefault('custom_selections', {})[key] = value


def _collect_custom_value(choice: dict[str, Any]) -> Any:
    kind = choice.get('kind')
    print('Custom selection mode:')
    if kind in {'enum', 'taxonomy_path'}:
        print('  1. literal value')
        print('  2. notebook function/reference name')
        print('  3. python file path (+ optional callable name)')
        mode = input('Choose custom mode [1/2/3]: ').strip() or '1'
        if mode == '2':
            name = input('Notebook reference name: ').strip()
            return {'custom_mode': 'notebook_ref', 'name': name}
        if mode == '3':
            path = input('Python file path: ').strip()
            callable_name = input('Callable name (optional): ').strip()
            payload = {'custom_mode': 'python_path', 'path': path}
            if callable_name:
                payload['callable'] = callable_name
            return payload
        return input('Literal custom value: ').strip()

    raw = input('Enter custom YAML value: ').strip()
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def _interactive_wizard(*, recipe_path: str, recipe_id: str, kind: str, yaml_path: str | None, selections: dict[str, Any] | None = None, max_steps: int | None = None) -> dict[str, Any]:
    stack = build_choice_stack()
    print('macrocast_single_run: single-path guided start')
    print('This mode helps you build one YAML path step-by-step.')

    if not yaml_path:
        yaml_name = input('YAML file path to write [custom_recipe.yaml]: ').strip() or 'custom_recipe.yaml'
        yaml_path = yaml_name
    recipe = build_yaml_preview(selections=selections, base_recipe_path=recipe_path, recipe_id=recipe_id, kind=kind)
    recipe['recipe_id'] = Path(yaml_path).stem
    write_path = _write_recipe_yaml(recipe, yaml_path)
    print(f'Writing selections to: {write_path}')

    completed = []
    route = _route_for_recipe(recipe)
    stop_reason = None
    limit = len(stack) if max_steps is None else min(max_steps, len(stack))
    for idx, choice in enumerate(stack[:limit], 1):
        options = list(choice.get('options', [])) + ['custom']
        current = _get_current_value(recipe, choice['key'])
        print()
        print(f'Step {idx}/{len(stack)} — {choice["key"]}')
        if current is not None:
            print(f'Current/default: {current}')
        for i, option in enumerate(options, 1):
            print(f'  {i}. {option}')
        answer = input('Select number/name, Enter=keep current, q=stop: ').strip()
        if answer.lower() == 'q':
            break
        if answer == '':
            selected = current
        else:
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                selected = options[int(answer) - 1]
            elif answer in options:
                selected = answer
            else:
                selected = answer
            if selected == 'custom':
                selected = _collect_custom_value(choice)
        if selected is not None:
            _set_value(recipe, choice['key'], selected)
        _write_recipe_yaml(recipe, write_path)
        completed.append({'key': choice['key'], 'value': _get_current_value(recipe, choice['key'])})

        if choice['key'] == 'experiment_unit':
            route = _route_for_recipe(recipe)
            print(f"Route owner: {route['owner']} | status: {route['status']}")
            print(route['message'])
            if not route['continue_in_single_run']:
                stop_reason = route['message']
                break

    next_choice = stack[len(completed)] if len(completed) < len(stack) and stop_reason is None else None
    return {
        'selected_stages': ['wizard'],
        'interactive': True,
        'yaml_path': str(write_path),
        'recipe_dict': recipe,
        'recipe_yaml': yaml.safe_dump(recipe, sort_keys=False),
        'completed_choices': completed,
        'current_choice': next_choice,
        'choice_stack': stack,
        'route': route,
        'stop_reason': stop_reason,
    }


def macrocast_single_run(*, recipe_path: str = 'baselines/minimal_fred_md.yaml', yaml_path: str | None = None, preset_id: str = 'researcher_explicit', stages: str | Iterable[str] | None = None, output_root: str = '/tmp/macrocast_single_run_preview', selections: dict[str, Any] | None = None, recipe_id: str = 'custom_recipe', kind: str = 'baseline', max_steps: int | None = None) -> dict[str, Any]:
    if stages is None and yaml_path is None:
        return _interactive_wizard(recipe_path=recipe_path, recipe_id=recipe_id, kind=kind, yaml_path=yaml_path, selections=selections, max_steps=max_steps)

    if stages is None and yaml_path is not None:
        stages = ['compile', 'tree_context', 'runs_preview', 'manifest_preview']

    selected = _normalize_stages(stages)
    out: dict[str, Any] = {'selected_stages': selected}

    axes = None
    compiled = None
    tree_context = None
    route = None

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

    if 'choice_stack' in selected:
        out['choice_stack'] = build_choice_stack()

    preview = None
    if 'yaml_preview' in selected:
        preview = build_yaml_preview(selections=selections, base_recipe_path=recipe_path, recipe_id=recipe_id, kind=kind)
        route = _route_for_recipe(preview)
        out['yaml_preview'] = {
            'recipe_dict': preview,
            'recipe_yaml': yaml.safe_dump(preview, sort_keys=False),
        }
        out['route'] = route

    effective_recipe_path = recipe_path
    if yaml_path is not None:
        user_recipe = _load_user_yaml(yaml_path)
        route = _route_for_recipe(user_recipe)
        out['input_yaml_path'] = str(yaml_path)
        out['input_yaml_recipe'] = user_recipe
        out['route'] = route
        effective_recipe_path = str(Path(yaml_path))
    elif route is None and any(stage in selected for stage in ('compile', 'tree_context', 'runs_preview', 'manifest_preview')):
        preview = preview or build_yaml_preview(selections=selections, base_recipe_path=recipe_path, recipe_id=recipe_id, kind=kind)
        route = _route_for_recipe(preview)
        out['route'] = route

    if any(stage in selected for stage in ('compile', 'tree_context', 'runs_preview', 'manifest_preview')):
        route = route or _route_for_experiment_unit('single_target_single_model')
        if not route['compile_allowed']:
            out['compile_blocked_reason'] = route['message']
            out['blocked_stages'] = [stage for stage in selected if stage in ('compile', 'tree_context', 'runs_preview', 'manifest_preview')]
            return out
        compiled = compile_experiment_spec_from_recipe(effective_recipe_path, preset_id=preset_id)

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
            run_id='macrocast-single-run-preview',
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
