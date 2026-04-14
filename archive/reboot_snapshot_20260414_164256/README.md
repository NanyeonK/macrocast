# macrocast

Decomposing ML Forecast Gains in Macroeconomic Forecasting.

`macrocast` is a generic forecasting research package organized around a tree-path architecture:
- `taxonomy/` = selectable choice universe
- `registries/` = backing defaults/adapters/contracts
- `recipes/` = named studies, baselines, benchmarks, ablations
- `runs/` = realized outputs from resolved paths

The package goal is not to hardcode individual papers.
Paper studies such as CLSS 2021 should appear as one recipe/path through the package, not as the package's organizing identity.

## Current migration status

Implemented migration layers:
- taxonomy bundle
- registries layer skeleton
- recipe layer skeleton
- recipe-aware compiled spec path
- benchmark redesign toward family + options
- runs layer skeleton and path-aware output layout

Still in migration:
- some current `config/*.yaml` files remain transitional operational truth
- some paper-specific helper code remains for compatibility/verification scaffolding

## Quick start

```python
from macrocast.specs.compiler import compile_experiment_spec_from_recipe

compiled = compile_experiment_spec_from_recipe(
    'baselines/minimal_fred_md.yaml',
    preset_id='researcher_explicit',
)

print(compiled.to_contract_dict())
```

## Architecture buckets

- `macrocast/taxonomy/`
- `registries/`
- `recipes/`
- `runs/`
- engine modules under `macrocast/data`, `macrocast/pipeline`, `macrocast/evaluation`, `macrocast/interpretation`, `macrocast/output`

## CLSS 2021 status

CLSS 2021 is being migrated toward a recipe/path representation:
- canonical recipe artifact: `recipes/papers/clss2021.yaml`
- existing `macrocast.replication.*` helpers are migration scaffolding only

## Repository

- GitHub: https://github.com/NanyeonK/macrocast

## License

MIT
