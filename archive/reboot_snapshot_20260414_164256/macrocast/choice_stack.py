from __future__ import annotations

from copy import deepcopy
from typing import Any

import yaml

from macrocast.recipes import load_recipe, load_recipe_schema
from macrocast.taxonomy.loaders import load_taxonomy_file

STAGE0_META_KEYS = {
    'experiment_unit',
    'axis_type',
    'registry_type',
    'reproducibility_mode',
    'failure_policy',
    'compute_mode',
}

ORDERED_CHOICE_STACK = [
    {'key': 'experiment_unit', 'layer': '0_meta', 'source': 'macrocast/taxonomy/0_meta/experiment_unit.yaml', 'kind': 'enum', 'extract_path': ['experiment_units', '*', 'id']},
    {'key': 'reproducibility_mode', 'layer': '0_meta', 'source': 'macrocast/taxonomy/0_meta/reproducibility_mode.yaml', 'kind': 'enum', 'extract_path': ['reproducibility_modes']},
    {'key': 'failure_policy', 'layer': '0_meta', 'source': 'macrocast/taxonomy/0_meta/failure_policy.yaml', 'kind': 'enum', 'extract_path': ['failure_policies']},
    {'key': 'compute_mode', 'layer': '0_meta', 'source': 'macrocast/taxonomy/0_meta/compute_mode.yaml', 'kind': 'enum', 'extract_path': ['compute_modes']},
    {'key': 'domain', 'layer': '1_data', 'source': 'macrocast/taxonomy/1_data/source.yaml', 'kind': 'taxonomy_path', 'extract_path': ['data_domains']},
    {'key': 'data', 'layer': '1_data', 'source': 'macrocast/taxonomy/1_data/source.yaml', 'kind': 'taxonomy_path', 'extract_path': ['dataset_sources']},
    {'key': 'frequency', 'layer': '1_data', 'source': 'macrocast/taxonomy/1_data/frequency.yaml', 'kind': 'taxonomy_path', 'extract_path': ['frequencies']},
    {'key': 'info_set', 'layer': '1_data', 'source': 'macrocast/taxonomy/1_data/info_set.yaml', 'kind': 'taxonomy_path', 'extract_path': ['information_set_types']},
    {'key': 'sample', 'layer': '1_data', 'source': 'recipes/baselines/minimal_fred_md.yaml', 'kind': 'taxonomy_path', 'extract_path': ['taxonomy_path', 'sample']},
    {'key': 'oos', 'layer': '1_data', 'source': 'recipes/baselines/minimal_fred_md.yaml', 'kind': 'taxonomy_path', 'extract_path': ['taxonomy_path', 'oos']},
    {'key': 'task', 'layer': '2_target_x', 'source': 'recipes/schema/recipe_schema.yaml', 'kind': 'taxonomy_path', 'extract_path': ['recipe_schema', 'required_taxonomy_path_keys']},
    {'key': 'target', 'layer': '2_target_x', 'source': 'recipes/baselines/minimal_fred_md.yaml', 'kind': 'taxonomy_path', 'extract_path': ['taxonomy_path', 'target']},
    {'key': 'x_map', 'layer': '2_target_x', 'source': 'macrocast/taxonomy/2_target_x/x_map_policy.yaml', 'kind': 'taxonomy_path', 'extract_path': ['x_map_policies']},
    {'key': 'target_prep', 'layer': '3_preprocess', 'source': 'recipes/baselines/minimal_fred_md.yaml', 'kind': 'taxonomy_path', 'extract_path': ['taxonomy_path', 'target_prep']},
    {'key': 'x_prep', 'layer': '3_preprocess', 'source': 'recipes/baselines/minimal_fred_md.yaml', 'kind': 'taxonomy_path', 'extract_path': ['taxonomy_path', 'x_prep']},
    {'key': 'features', 'layer': '4_training', 'source': 'recipes/baselines/minimal_fred_md.yaml', 'kind': 'taxonomy_path', 'extract_path': ['taxonomy_path', 'features']},
    {'key': 'framework', 'layer': '4_training', 'source': 'macrocast/taxonomy/4_training/framework.yaml', 'kind': 'taxonomy_path', 'extract_path': ['forecast_frameworks', 'outer_window']},
    {'key': 'validation', 'layer': '4_training', 'source': 'macrocast/taxonomy/4_training/split.yaml', 'kind': 'taxonomy_path', 'extract_path': ['validation_locations']},
    {'key': 'model', 'layer': '4_training', 'source': 'recipes/baselines/minimal_fred_md.yaml', 'kind': 'taxonomy_path', 'extract_path': ['taxonomy_path', 'model']},
    {'key': 'tuning', 'layer': '4_training', 'source': 'macrocast/taxonomy/4_training/tuning_registry.yaml', 'kind': 'taxonomy_path', 'extract_path': ['search_algorithms']},
    {'key': 'metric', 'layer': '5_evaluation', 'source': 'macrocast/taxonomy/5_evaluation/metric_registry.yaml', 'kind': 'taxonomy_path', 'extract_path': ['metric_registry', 'point_forecast']},
    {'key': 'benchmark', 'layer': '5_evaluation', 'source': 'macrocast/taxonomy/5_evaluation/benchmark_registry.yaml', 'kind': 'taxonomy_path', 'extract_path': ['benchmark_families']},
    {'key': 'stat', 'layer': '6_stat_tests', 'source': 'macrocast/taxonomy/6_stat_tests/test_registry.yaml', 'kind': 'taxonomy_path', 'extract_path': ['statistical_tests', 'equal_predictive_ability']},
    {'key': 'importance', 'layer': '7_importance', 'source': 'macrocast/taxonomy/7_importance/importance_registry.yaml', 'kind': 'taxonomy_path', 'extract_path': ['importance_methods', 'model_native']},
    {'key': 'output', 'layer': '8_output_provenance', 'source': 'recipes/baselines/minimal_fred_md.yaml', 'kind': 'taxonomy_path', 'extract_path': ['taxonomy_path', 'output']},
    {'key': 'numeric_params', 'layer': 'numeric', 'source': 'recipes/baselines/minimal_fred_md.yaml', 'kind': 'numeric', 'extract_path': ['numeric_params']},
    {'key': 'outputs', 'layer': 'output', 'source': 'recipes/baselines/minimal_fred_md.yaml', 'kind': 'dict', 'extract_path': ['outputs']},
]


def _repo_root_source(source: str) -> dict[str, Any]:
    if source.startswith('recipes/schema/'):
        return load_recipe_schema()
    if source.startswith('recipes/'):
        return load_recipe(source.removeprefix('recipes/'))
    rel = source.removeprefix('macrocast/taxonomy/')
    return load_taxonomy_file(rel)


def _extract_by_path(obj: Any, path: list[str]) -> Any:
    cur = obj
    for step in path:
        if step == '*':
            if not isinstance(cur, list):
                return []
            return cur
        if not isinstance(cur, dict) or step not in cur:
            return None
        cur = cur[step]
    return cur


def _options_for_item(item: dict[str, Any]) -> list[str]:
    data = _repo_root_source(item['source'])
    path = item.get('extract_path', [])
    if '*' in path:
        star = path.index('*')
        prefix = path[:star]
        suffix = path[star + 1:]
        base = _extract_by_path(data, prefix)
        if not isinstance(base, list):
            return []
        out = []
        for elem in base:
            val = _extract_by_path(elem, suffix) if suffix else elem
            if isinstance(val, str):
                out.append(val)
        return out
    val = _extract_by_path(data, path)
    if isinstance(val, list):
        return [x for x in val if isinstance(x, str)]
    if isinstance(val, dict):
        return sorted([k for k in val.keys()])
    if isinstance(val, str):
        return [val]
    return []


def build_choice_stack() -> list[dict[str, Any]]:
    stack: list[dict[str, Any]] = []
    for item in ORDERED_CHOICE_STACK:
        entry = dict(item)
        entry['options'] = _options_for_item(item)
        stack.append(entry)
    return stack


def build_yaml_preview(*, selections: dict[str, Any] | None = None, base_recipe_path: str = 'baselines/minimal_fred_md.yaml', recipe_id: str = 'custom_recipe', kind: str = 'baseline') -> dict[str, Any]:
    recipe = deepcopy(load_recipe(base_recipe_path))
    recipe['recipe_id'] = recipe_id
    recipe['kind'] = kind
    recipe.setdefault('meta', {})
    recipe['meta'].setdefault('experiment_unit', 'single_target_single_model')
    selections = selections or {}
    for key, value in selections.items():
        if key in STAGE0_META_KEYS:
            recipe.setdefault('meta', {})[key] = value
        elif key in recipe.get('taxonomy_path', {}):
            recipe['taxonomy_path'][key] = value
        elif key in recipe.get('numeric_params', {}):
            recipe['numeric_params'][key] = value
        elif key in recipe.get('outputs', {}):
            recipe['outputs'][key] = value
        else:
            recipe.setdefault('custom_selections', {})[key] = value
    return recipe
