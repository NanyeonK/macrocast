from __future__ import annotations

from typing import Any


def derive_tree_context_from_compiled_spec(meta_config: dict[str, Any], axes_registry: dict[str, Any] | None = None) -> dict[str, Any]:
    taxonomy_path = meta_config.get('taxonomy_path', {}) or {}
    fixed_summary = ((axes_registry or {}).get('fixed_axes_summary') or {}).get('includes', [])
    sweep_summary = ((axes_registry or {}).get('sweep_axes_summary') or {}).get('includes', [])
    fixed_values = {
        key: meta_config.get(key)
        for key in fixed_summary
        if key in meta_config
    }
    sweep_values = {
        key: meta_config.get(key)
        for key in sweep_summary
        if key in meta_config
    }
    if 'horizon' in meta_config:
        sweep_values.setdefault('horizon', meta_config.get('horizon'))
    return {
        'compile_path': meta_config.get('compile_path'),
        'recipe_id': meta_config.get('recipe_id'),
        'recipe_kind': meta_config.get('recipe_kind'),
        'taxonomy_path': taxonomy_path,
        'fixed_axes': list(fixed_summary),
        'sweep_axes': list(sweep_summary),
        'fixed_values': fixed_values,
        'sweep_values': sweep_values,
    }
