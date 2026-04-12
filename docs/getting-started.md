# Getting Started

## 1. Baseline recipe path

The package is moving toward recipe-first compilation.
A minimal baseline recipe already exists:
- `recipes/baselines/minimal_fred_md.yaml`

Compile it into a package contract:

```python
from macrocast.specs.compiler import compile_experiment_spec_from_recipe

compiled = compile_experiment_spec_from_recipe(
    'baselines/minimal_fred_md.yaml',
    preset_id='researcher_explicit',
)

print(compiled.to_contract_dict())
```

## 2. What is stable now

Stable migration pieces:
- recipe schema
- recipe loaders/validators
- recipe-aware compiled spec entry path
- benchmark family/options resolution
- path-aware output layout

## 3. What is still transitional

Still transitional:
- some direct `config/*.yaml` operational registries
- some paper-specific replication helpers
- some older docs/tutorials that predate tree-path migration

## 4. CLSS example status

CLSS 2021 currently exists in two forms:
- future target form: `recipes/papers/clss2021.yaml`
- temporary migration scaffolding: `macrocast.replication.clss2021*`

Use the recipe artifact as the architectural reference.
