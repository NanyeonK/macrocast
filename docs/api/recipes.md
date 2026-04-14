# Recipes and execution API Reference

## Import surface

```python
from macrocast import (
    RecipeSpec,
    RunSpec,
    build_recipe_spec,
    build_run_spec,
    check_recipe_completeness,
    recipe_summary,
)
```

## Objects

### `RecipeSpec`

Fields:
- `recipe_id`
- `stage0`
- `target`
- `horizons`
- `raw_dataset`

Purpose:
- minimal declarative forecasting-study object for v1

### `RunSpec`

Fields:
- `run_id`
- `recipe_id`
- `route_owner`
- `artifact_subdir`

Purpose:
- minimal execution-facing run object derived from a recipe

## Functions

### `build_recipe_spec()`

```python
def build_recipe_spec(
    *,
    recipe_id: str,
    stage0: Stage0Frame,
    target: str,
    horizons: tuple[int, ...],
    raw_dataset: str,
) -> RecipeSpec:
    ...
```

### `check_recipe_completeness()`

```python
def check_recipe_completeness(recipe: RecipeSpec) -> None:
    ...
```

Raises `RecipeValidationError` when the recipe is structurally incomplete.

### `build_run_spec()`

```python
def build_run_spec(recipe: RecipeSpec) -> RunSpec:
    ...
```

Behavior:
- resolves route owner from Stage 0
- constructs deterministic `run_id`
- constructs default `artifact_subdir`

### `recipe_summary()`

```python
def recipe_summary(recipe: RecipeSpec) -> str:
    ...
```

Returns a compact human-readable summary.

## Example

```python
from macrocast import build_recipe_spec, build_run_spec

recipe = build_recipe_spec(
    recipe_id="fred_md_baseline",
    stage0=stage0,
    target="INDPRO",
    horizons=(1, 3, 6, 12),
    raw_dataset="fred_md",
)

run = build_run_spec(recipe)
```

## Notes

- This is the minimal v1 recipe/execution contract.
- It intentionally sits above Stage 0 and raw data but below actual forecasting execution.
- The goal is to make one recipe correspond to one well-defined study declaration and one deterministic run identity.