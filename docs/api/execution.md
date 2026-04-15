# Execution API Reference

## Import surface

```python
from macrocast import (
    ExecutionSpec,
    ExecutionResult,
    build_execution_spec,
    execute_recipe,
)
```

## Objects

### `ExecutionSpec`

Fields:
- `recipe`
- `run`
- `preprocess`

### `ExecutionResult`

Fields:
- `spec`
- `run`
- `raw_result`
- `artifact_dir`

## Functions

### `build_execution_spec()`
Build the minimal execution-facing contract object.

### `execute_recipe()`
Run the current thin runtime slice and write deterministic artifacts.

Key current behavior:
- validates that preprocessing belongs to the operational runtime subset
- uses separate internal model and benchmark executors
- reads model family, benchmark family, and benchmark config from recipe grammar
- supports executable model families `ar`, `ridge`, `lasso`, `elasticnet`, and `randomforest`
- supports executable benchmarks `historical_mean`, `zero_change`, and `ar_bic`
- writes benchmark predictions and benchmark-aware metrics
- records model spec, benchmark spec, preprocessing semantics, and compiler provenance in the manifest

## Notes

The execution layer is intentionally narrower than the full registry/compiler choice space.
Unsupported runtime choices, including unsupported benchmark families, unsupported model families, or unsupported feature builders, should be filtered at compile time rather than silently ignored during execution.
