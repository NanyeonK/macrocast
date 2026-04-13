from __future__ import annotations

from copy import deepcopy
from typing import Any


def recipe_to_runtime_config(recipe: dict[str, Any]) -> dict[str, Any]:
    path = recipe['taxonomy_path']
    nums = recipe['numeric_params']
    outputs = recipe['outputs']

    model_key = path['model']
    model_map = {
        'random_forest': 'RF',
        'kernel_ridge': 'KRR',
        'elastic_net': 'EN',
        'adaptive_lasso': 'AL',
        'ar': 'AR',
    }
    feature_map = {
        'factors_x': 'X',
    }
    raw = {
        'experiment_id': recipe['recipe_id'],
        'dataset': path['data'],
        'target': path['target'],
        'horizons': nums.get('horizons', [1]),
        'window': path['framework'],
        'oos_start': nums.get('oos_start'),
        'oos_end': nums.get('oos_end'),
        'models': [model_map.get(model_key, model_key)],
        'n_factors': nums.get('n_factors', 4),
        'n_lags': nums.get('n_lags', 2),
        'factor_type': feature_map.get(path['features'], 'X'),
        'benchmark_family': path.get('benchmark'),
        'benchmark_options': recipe.get('benchmark_options', {}),
    }
    if 'output_dir' in outputs:
        raw['output_dir'] = outputs['output_dir']
    return raw


def build_recipe_resolution_context(recipe: dict[str, Any]) -> dict[str, Any]:
    path = recipe['taxonomy_path']
    nums = recipe['numeric_params']
    outputs = recipe['outputs']
    return {
        'recipe_id': recipe['recipe_id'],
        'recipe_kind': recipe['kind'],
        'taxonomy_path': deepcopy(path),
        'numeric_params': deepcopy(nums),
        'output_prefs': deepcopy(outputs),
        'benchmark_family': path.get('benchmark'),
        'benchmark_options': deepcopy(recipe.get('benchmark_options', {})),
        'resolved_run_horizons': nums.get('horizons', [1]),
        'resolved_framework': path.get('framework'),
        'resolved_feature_key': path.get('features'),
        'resolved_model_key': path.get('model'),
        'direct_experiment_overrides': build_recipe_experiment_overrides(recipe),
    }


def build_recipe_experiment_overrides(recipe: dict[str, Any]) -> dict[str, Any]:
    path = recipe['taxonomy_path']
    nums = recipe['numeric_params']
    overrides: dict[str, Any] = {
        'dataset': path['data'],
        'target': path['target'],
        'horizon': nums.get('horizons', [1]),
        'outer_window': path['framework'],
        'benchmark_family': path.get('benchmark'),
        'benchmark_options': deepcopy(recipe.get('benchmark_options', {})),
        'recipe_feature_key': path.get('features'),
        'recipe_model_key': path.get('model'),
        'recipe_output_key': path.get('output'),
        'recipe_task_key': path.get('task'),
    }
    if nums.get('oos_start') or nums.get('oos_end'):
        overrides['oos_period'] = {
            'start': nums.get('oos_start'),
            'end': nums.get('oos_end'),
        }
    return overrides

def recipe_to_experiment_config(recipe):
    from macrocast.config import build_experiment_config_from_components

    path = recipe["taxonomy_path"]
    nums = recipe["numeric_params"]
    outputs = recipe["outputs"]

    feature_section = {
        "factor_type": "X" if path.get("features") == "factors_x" else "none",
        "n_factors": nums.get("n_factors", 4),
        "n_lags": nums.get("n_lags", 2),
        "p_marx": nums.get("p_marx", 12),
    }
    model_name_map = {
        "random_forest": "rf",
        "kernel_ridge": "krr",
        "elastic_net": "elastic_net",
        "adaptive_lasso": "adaptive_lasso",
        "ar": "ar",
    }
    model_section = [{"name": model_name_map.get(path["model"], path["model"])}]

    return build_experiment_config_from_components(
        experiment_id=recipe["recipe_id"],
        output_dir=outputs.get("output_dir", "~/.macrocast/results"),
        dataset=path["data"],
        target=path["target"],
        vintage=nums.get("vintage"),
        sample_start=nums.get("sample_start"),
        feature_section=feature_section,
        model_section=model_section,
        horizons=nums.get("horizons", [1]),
        window=path["framework"],
        rolling_size=nums.get("rolling_size"),
        oos_start=nums.get("oos_start"),
        oos_end=nums.get("oos_end"),
        n_jobs=int(nums.get("n_jobs", 1)),
    )
