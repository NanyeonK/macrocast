from pathlib import Path

import yaml

from macrocast import macrocast_single_run
from macrocast.tree_context import derive_tree_context_from_compiled_spec
from macrocast.specs.compiler import compile_experiment_spec_from_recipe
from macrocast.meta import load_axes_registry


def test_macrocast_single_run_default_interactive_yes(monkeypatch, tmp_path: Path) -> None:
    answers = iter([str(tmp_path / 'wizard.yaml'), '1'])
    monkeypatch.setattr('builtins.input', lambda _='': next(answers))
    out = macrocast_single_run(max_steps=1)
    assert out['interactive'] is True
    assert out['yaml_path'].endswith('wizard.yaml')
    assert out['completed_choices'][0]['key'] == 'experiment_unit'
    assert out['completed_choices'][0]['value'] == 'single_target_single_model'
    assert out['recipe_dict']['meta']['experiment_unit'] == 'single_target_single_model'
    assert out['route']['status'] == 'implemented'
    assert out['current_choice']['key'] == 'reproducibility_mode'


def test_macrocast_single_run_choice_stack_stage() -> None:
    out = macrocast_single_run(stages=['choice_stack'])
    first = out['choice_stack'][0]
    assert first['key'] == 'experiment_unit'
    assert first['options'] == [
        'single_target_single_model',
        'single_target_model_grid',
        'single_target_full_sweep',
        'multi_target_separate_runs',
        'multi_target_shared_design',
        'replication_recipe',
        'benchmark_suite',
        'ablation_study',
    ]


def test_macrocast_single_run_yaml_preview_override() -> None:
    out = macrocast_single_run(stages=['yaml_preview'], selections={'data': 'fred_qd', 'target': 'UNRATE'}, recipe_id='my_recipe')
    assert out['yaml_preview']['recipe_dict']['recipe_id'] == 'my_recipe'
    assert out['yaml_preview']['recipe_dict']['taxonomy_path']['data'] == 'fred_qd'
    assert out['yaml_preview']['recipe_dict']['taxonomy_path']['target'] == 'UNRATE'
    assert out['yaml_preview']['recipe_dict']['meta']['experiment_unit'] == 'single_target_single_model'
    assert out['route']['status'] == 'implemented'


def test_macrocast_single_run_yaml_preview_wrapper_route() -> None:
    out = macrocast_single_run(stages=['yaml_preview'], selections={'experiment_unit': 'benchmark_suite'})
    assert out['yaml_preview']['recipe_dict']['meta']['experiment_unit'] == 'benchmark_suite'
    assert out['route']['owner'] == 'wrapper_orchestrator'
    assert out['route']['status'] == 'wrapper_required'


def test_macrocast_single_run_yaml_path_auto_run(tmp_path: Path) -> None:
    recipe = {
        'recipe_id': 'user_recipe',
        'kind': 'baseline',
        'taxonomy_path': {
            'domain': 'macro', 'data': 'fred_md', 'frequency': 'monthly', 'info_set': 'revised',
            'sample': 'full_sample', 'oos': 'short_smoke_oos', 'task': 'point_direct', 'target': 'INDPRO',
            'x_map': 'all_minus_target', 'target_prep': 'basic_none', 'x_prep': 'basic_none', 'features': 'factors_x',
            'framework': 'expanding', 'validation': 'last_block', 'model': 'random_forest', 'tuning': 'grid_search',
            'metric': 'point_default', 'benchmark': 'ar', 'stat': 'forecast_comparison_default', 'importance': 'tree_native_importance', 'output': 'parquet_manifest_bundle'
        },
        'numeric_params': {'horizons': [1], 'n_factors': 4, 'n_lags': 2, 'oos_start': '2012-01-01', 'oos_end': '2012-03-01'},
        'outputs': {'output_dir': '/tmp/macrocast_results'}
    }
    path = tmp_path / 'user_recipe.yaml'
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding='utf-8')
    out = macrocast_single_run(yaml_path=str(path))
    assert out['input_yaml_recipe']['recipe_id'] == 'user_recipe'
    assert out['input_yaml_recipe']['meta']['experiment_unit'] == 'single_target_single_model'
    assert out['manifest_preview']['recipe_id'] == 'user_recipe'


def test_macrocast_single_run_blocks_wrapper_compile(tmp_path: Path) -> None:
    recipe = {
        'recipe_id': 'wrapper_recipe',
        'kind': 'benchmark',
        'meta': {'experiment_unit': 'benchmark_suite'},
        'taxonomy_path': {
            'domain': 'macro', 'data': 'fred_md', 'frequency': 'monthly', 'info_set': 'revised',
            'sample': 'full_sample', 'oos': 'short_smoke_oos', 'task': 'point_direct', 'target': 'INDPRO',
            'x_map': 'all_minus_target', 'target_prep': 'basic_none', 'x_prep': 'basic_none', 'features': 'factors_x',
            'framework': 'expanding', 'validation': 'last_block', 'model': 'random_forest', 'tuning': 'grid_search',
            'metric': 'point_default', 'benchmark': 'ar', 'stat': 'forecast_comparison_default', 'importance': 'tree_native_importance', 'output': 'parquet_manifest_bundle'
        },
        'numeric_params': {'horizons': [1], 'n_factors': 4, 'n_lags': 2, 'oos_start': '2012-01-01', 'oos_end': '2012-03-01'},
        'outputs': {'output_dir': '/tmp/macrocast_results'}
    }
    path = tmp_path / 'wrapper_recipe.yaml'
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding='utf-8')
    out = macrocast_single_run(yaml_path=str(path))
    assert out['route']['status'] == 'wrapper_required'
    assert 'compile' not in out
    assert out['blocked_stages'] == ['compile', 'tree_context', 'runs_preview', 'manifest_preview']


def test_derive_tree_context_from_compiled_spec() -> None:
    compiled = compile_experiment_spec_from_recipe('baselines/minimal_fred_md.yaml', preset_id='researcher_explicit')
    ctx = derive_tree_context_from_compiled_spec(compiled.meta_config, load_axes_registry())
    assert 'dataset' in ctx['fixed_axes']
    assert 'horizon' in ctx['sweep_axes']
    assert ctx['fixed_values']['dataset'] == 'fred_md'
    assert ctx['sweep_values']['horizon'] == [1]
