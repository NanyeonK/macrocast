# Compiler API Reference

## Import surface

```python
from macrocast import (
    CompileError,
    CompileValidationError,
    CompiledRecipeSpec,
    CompileResult,
    load_recipe_yaml,
    compile_recipe_dict,
    compile_recipe_yaml,
    compiled_spec_to_dict,
    run_compiled_recipe,
)
```

## Objects

### `CompiledRecipeSpec`

Fields:
- `recipe_id`
- `layer_order`
- `axis_selections`
- `leaf_config`
- `preprocess_contract`
- `stage0`
- `recipe_spec`
- `run_spec`
- `execution_status`
- `warnings`
- `blocked_reasons`

### `CompileResult`

Fields:
- `compiled`
- `manifest`

## Functions

### `load_recipe_yaml()`
Load one YAML recipe file into a Python dict.

### `compile_recipe_dict()` / `compile_recipe_yaml()`
Validate recipe grammar, build canonical selections, build preprocessing contract, derive Stage 0 and recipe/run specs, derive `model_spec` and `benchmark_spec`, and assign compile status.

### `compiled_spec_to_dict()`
Serialize the compiled spec into a provenance-safe dict.

### `run_compiled_recipe()`
Run only compiled specs with `execution_status='executable'`.
Raises if the grammar path is only representable or blocked.

## Notes

The compiler exists so macrocast can support a wider long-run taxonomy than the current runtime slice without losing explicitness.
Model family and benchmark family remain separate grammar objects all the way into compiled provenance.
The current executable model set includes `ar`, `ridge`, `lasso`, `elasticnet`, and `randomforest`.
The current executable benchmark set includes `historical_mean`, `zero_change`, and `ar_bic`.
There is no silent fallback from unsupported model or benchmark choices to whatever currently runs.
