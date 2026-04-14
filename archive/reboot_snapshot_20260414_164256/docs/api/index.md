# API Reference

The public `macrocast` API is organized around a small set of import surfaces.

## Main package API

Canonical entry points from the top-level package:

```python
import macrocast as mc
from macrocast import macrocast_single_run, compile_experiment_spec_from_recipe
```

Most users should know these first:
- `macrocast_single_run()` — guided path construction and preview entry point
- `compile_experiment_spec_from_recipe()` — recipe-native compile entry point
- `load_config()` / `load_config_from_dict()` — legacy config loading helpers

## API structure

The API is split into package surfaces rather than one giant flat namespace.

- `macrocast.data`
  - raw data acquisition, schema objects, registry defaults, vintage access
- `macrocast.preprocessing`
  - transforms, missing handling, panel transforms, preprocessing registries
- `macrocast.pipeline`
  - forecasting components, estimators, models, experiment orchestration
- `macrocast.evaluation`
  - metrics, tests, decomposition, combination, evaluation registries
- `macrocast.interpretation`
  - dual-weight interpretation, marginal effects, PBSV, variable importance
- `macrocast.viz`
  - plotting helpers for forecast and interpretation outputs
- `macrocast.utils`
  - cache helpers, registry utilities, LaTeX export helpers

## Import paths and usage pattern

Recommended import style:

```python
from macrocast import macrocast_single_run
from macrocast.data import load_fred_md
from macrocast.preprocessing import apply_tcodes
from macrocast.pipeline import ForecastExperiment, RFModel
from macrocast.evaluation import relative_msfe, dm_test
```

Interpretation rule:
- import from the package surface when you want stable public entry points
- import from deep internal modules only when working on package internals

## API groups

### Top-level workflow
- `macrocast_single_run()`
- `compile_experiment_spec_from_recipe()`
- `CompiledExperimentSpec`

### Data and preprocessing
- `macrocast.data`
- `macrocast.preprocessing`

### Modeling and execution
- `macrocast.pipeline`

### Evaluation and interpretation
- `macrocast.evaluation`
- `macrocast.interpretation`
- `macrocast.viz`

### Utilities
- `macrocast.utils`

## Detail pages

Use the package detail pages for grouped public objects:
- `macrocast.data`
- `macrocast.preprocessing`
- `macrocast.pipeline`
- `macrocast.evaluation`
- `macrocast.interpretation`
- `macrocast.viz`
- `macrocast.utils`

These pages are organized by object family so they read more like a package reference and less like raw source dumps.
