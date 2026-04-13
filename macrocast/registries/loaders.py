from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REGISTRY_LAYERS = ['meta', 'data', 'training', 'evaluation', 'output']


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _registries_root() -> Path:
    return _repo_root() / 'registries'


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f'registry YAML at {path} must decode to dict')
    return data


def load_registry_file(relative_path: str) -> dict[str, Any]:
    return _load_yaml(_registries_root() / relative_path)


def load_registry_layer(layer: str) -> dict[str, dict[str, Any]]:
    layer_dir = _registries_root() / layer
    if not layer_dir.exists():
        raise FileNotFoundError(f'unknown registry layer: {layer}')
    bundle: dict[str, dict[str, Any]] = {}
    for path in sorted(layer_dir.glob('*.yaml')):
        bundle[path.stem] = _load_yaml(path)
    return bundle


def load_registry_bundle() -> dict[str, dict[str, dict[str, Any]]]:
    return {layer: load_registry_layer(layer) for layer in REGISTRY_LAYERS}


def resolve_registry_source(metadata: dict[str, Any]) -> dict[str, Any]:
    reg = metadata.get('registry')
    if not isinstance(reg, dict):
        raise ValueError('registry metadata must include registry block')
    inline_payload = {k: v for k, v in metadata.items() if k != 'registry'}
    if inline_payload:
        return inline_payload
    if 'source' not in reg:
        raise ValueError('registry metadata must include source path when no inline payload exists')
    return _load_yaml(_repo_root() / reg['source'])


def load_operational_registry(relative_path: str) -> dict[str, Any]:
    return resolve_registry_source(load_registry_file(relative_path))


def load_operational_global_defaults_registry() -> dict[str, Any]:
    return load_operational_registry('meta/global_defaults.yaml')


def load_operational_dataset_registry() -> dict[str, Any]:
    return load_operational_registry('data/datasets.yaml')


def load_operational_target_registry() -> dict[str, Any]:
    return load_operational_registry('data/targets.yaml')


def load_operational_data_task_registry() -> dict[str, Any]:
    return load_operational_registry('data/data_tasks.yaml')


def load_operational_model_registry() -> dict[str, Any]:
    return load_operational_registry('training/models.yaml')


def load_operational_feature_registry() -> dict[str, Any]:
    return load_operational_registry('training/features.yaml')


def load_operational_benchmark_registry() -> dict[str, Any]:
    return load_operational_registry('evaluation/benchmarks.yaml')
