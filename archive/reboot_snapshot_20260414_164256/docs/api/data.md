# macrocast.data

`macrocast.data` contains the public API for raw data access, dataset metadata, registry defaults, merging, and vintage-aware panels.

## Import

```python
from macrocast.data import load_fred_md, load_fred_qd, load_fred_sd
```

## Data loading

Main loaders:
- `load_fred_md`
- `load_fred_qd`
- `load_fred_sd`

Purpose:
- fetch and normalize the supported macro datasets into package data objects

## Schema objects

Core schema exports:
- `MacroFrame`
- `MacroFrameMetadata`
- `VariableMetadata`

Purpose:
- represent the package's structured macro data surface and attached metadata

## Vintage access

Vintage-aware exports:
- `list_available_vintages`
- `load_vintage_panel`
- `RealTimePanel`

Purpose:
- inspect available vintages and build vintage-aware panels for pseudo-real-time work

## Registry helpers

Dataset and target registry exports:
- `load_dataset_registry`
- `load_target_registry`
- `load_data_task_registry`
- `validate_dataset_registry`
- `validate_target_registry`
- `validate_data_task_registry`
- `get_dataset_defaults`
- `get_target_defaults`
- `get_data_task_defaults`

Purpose:
- expose canonical defaults and validation logic for data/task-level registries

## Merge helpers

Merge exports:
- `merge_macro_frames`
- `MergeResult`

Purpose:
- combine aligned macro data objects while preserving merge metadata

## Related pages

- `User Guide > Stage 1`
- `User Guide > Stage 2`
- `Examples > Recipes & Runs`
