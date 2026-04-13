from macrocast.specs.compiler import compile_experiment_spec_from_recipe


def test_compile_experiment_spec_from_recipe_baseline() -> None:
    compiled = compile_experiment_spec_from_recipe('baselines/minimal_fred_md.yaml', preset_id='researcher_explicit')
    assert compiled.meta_config['dataset'] == 'fred_md'
    assert compiled.meta_config['target'] == 'INDPRO'
    assert compiled.meta_config['horizon'] == [1]
    assert compiled.meta_config['outer_window'] == 'expanding'
    assert compiled.meta_config['benchmark_family'] == 'ar'
    assert compiled.meta_config['benchmark_id'] == 'ar_bic_expanding'
    assert compiled.meta_config['recipe_id'] == 'minimal_fred_md'
    assert compiled.meta_config['recipe_kind'] == 'baseline'
    assert compiled.meta_config['taxonomy_path']['model'] == 'random_forest'
    assert compiled.meta_config['numeric_params']['n_factors'] == 4
    assert compiled.meta_config['recipe_resolution_context']['resolved_model_key'] == 'random_forest'
    assert compiled.meta_config['recipe_resolution_context']['benchmark_family'] == 'ar'
    assert compiled.meta_config['recipe_resolution_context']['direct_experiment_overrides']['dataset'] == 'fred_md'
    assert compiled.meta_config['recipe_resolution_context']['direct_experiment_overrides']['outer_window'] == 'expanding'
    assert compiled.meta_config['compile_path'] == 'recipe_native_experiment_config'
    assert compiled.meta_config['model_family'] == 'tree_ensemble'


def test_compile_experiment_spec_includes_tree_context() -> None:
    compiled = compile_experiment_spec_from_recipe('baselines/minimal_fred_md.yaml', preset_id='researcher_explicit')
    assert compiled.meta_config['tree_context']['fixed_values']['dataset'] == 'fred_md'
    assert 'horizon' in compiled.meta_config['tree_context']['sweep_axes']
    assert compiled.to_contract_dict()['tree_context']['compile_path'] == 'recipe_native_experiment_config'
