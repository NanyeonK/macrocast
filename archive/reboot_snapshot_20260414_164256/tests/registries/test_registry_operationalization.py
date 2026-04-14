from macrocast.registries import (
    load_operational_benchmark_registry,
    load_operational_data_task_registry,
    load_operational_dataset_registry,
    load_operational_global_defaults_registry,
    load_operational_model_registry,
    load_operational_target_registry,
    load_registry_file,
)


def test_operational_registry_loaders_return_live_sources() -> None:
    assert 'global_defaults' in load_operational_global_defaults_registry()
    assert 'datasets' in load_operational_dataset_registry()
    assert 'targets' in load_operational_target_registry()
    assert 'data_tasks' in load_operational_data_task_registry()
    assert 'models' in load_operational_model_registry()
    assert 'benchmark_families' in load_operational_benchmark_registry()


def test_registry_files_now_embed_inline_payload() -> None:
    assert 'global_defaults' in load_registry_file('meta/global_defaults.yaml')
    assert 'datasets' in load_registry_file('data/datasets.yaml')
    assert 'benchmark_families' in load_registry_file('evaluation/benchmarks.yaml')
