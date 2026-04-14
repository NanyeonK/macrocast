# Stage 0.1 experiment_unit — function design

Goal:
Define the concrete function family that should operationalize each `experiment_unit` route.

Design rule:
- use one shared route-decision layer
- use one shared bundle-plan contract
- let each experiment-unit-specific function return a normalized plan object
- keep actual execution separate from planning/bundle construction

This keeps Stage 0.1 responsible for route and plan construction, not for full execution side effects.

## Core design decision

Do not create eight unrelated entry functions with unrelated payloads.

Also do not force the user to choose `own_analysis` explicitly as a first-class Stage 0 value.
For the main package flow, `own_analysis` should be the implicit default.
Replication should be treated as an explicit special input path: if the user supplies a replication recipe or replication bundle reference, the package diverts into replication mode; otherwise it proceeds as ordinary own-analysis design construction.

Instead, use three layers:
1. route resolution
2. route-specific plan construction
3. later execution/orchestration over the normalized plan

## Dataset-driven implications for Stage 0

For the current package scope, dataset choice already fixes more than one later design field.

Current practical rule:
- `fred_md` implies monthly analysis
- `fred_qd` implies quarterly analysis
- `fred_sd` is mixed-frequency and should not be treated like a simple single-target monthly/quarterly panel

Important consequence:
- dataset and frequency should not be modeled as two equally free top-level choices in the default path
- frequency should usually be derived from dataset choice unless the user is in a custom-data branch
- if `fred_sd` is combined with other datasets, the package should assume a mixed-frequency / multi-target or bundle-style design rather than an ordinary one-path setup

Recommended Stage 0 redesign implication:
- move `dataset` into the earliest fixed-design choices
- derive `frequency` from dataset for standard FRED paths
- open an explicit mixed-frequency / multi-target branch when `fred_sd` enters a combined design

## Default-flow rule

Recommended Stage 0.1 behavior:
- if no explicit replication input is supplied, proceed as ordinary own-analysis design construction
- benchmark comparison and ablation are not separate top-level study intents in the default flow; they are comparison/variation structures attached to own analysis
- replication is the exceptional path because it starts from a locked recipe, recipe bundle, or paper-specific preset

Recommended implication for public UX:
- do not ask the user to choose `own_analysis` explicitly
- ask for replication only when a replication recipe / bundle / preset is supplied or explicitly requested
- otherwise start from fixed-vs-varying design choices for the baseline horse-race analysis

## Shared contracts

### 1. ExperimentUnitRouteDecision

Purpose:
- classify the selected `experiment_unit`
- record whether the current public entry point may continue
- provide the canonical downstream builder name

Proposed fields:
- `experiment_unit: str`
- `owner: Literal["single_run", "wrapper_orchestrator"]`
- `status: Literal["implemented", "planned_single_run_extension", "wrapper_required", "unknown_requires_design"]`
- `shape: str`
- `compile_allowed: bool`
- `continue_in_single_run: bool`
- `builder_function: str | None`
- `message: str`

Recommended function:

```python
def resolve_experiment_unit_route(
    experiment_unit: str,
) -> ExperimentUnitRouteDecision:
    ...
```

Input:
- one `experiment_unit` string

Output:
- normalized route-decision object

Notes:
- this is the publicized version of the current `_route_for_experiment_unit()` logic
- current private implementation already exists conceptually in `macrocast/start.py`

### 2. ChildRecipePlan

Purpose:
- represent one child recipe emitted by a route-specific builder

Proposed fields:
- `recipe_id: str`
- `recipe_dict: dict[str, Any]`
- `recipe_path_hint: str | None`
- `role: str`
- `tags: list[str]`
- `comparison_group: str | None`

Recommended use:
- one-path routes emit one child recipe
- bundle routes emit multiple child recipes

### 3. ExperimentUnitBundlePlan

Purpose:
- normalize outputs from all Stage 0.1 builders

Proposed fields:
- `experiment_unit: str`
- `route: ExperimentUnitRouteDecision`
- `bundle_id: str`
- `kind: str`
- `fixed_selections: dict[str, Any]`
- `sweep_spec: dict[str, Any]`
- `shared_design: dict[str, Any]`
- `child_recipes: list[ChildRecipePlan]`
- `comparison_spec: dict[str, Any]`
- `compile_ready: bool`
- `next_action: str`
- `warnings: list[str]`

Recommended function consumers:
- preview layer
- YAML writer layer
- future wrapper/orchestrator execution layer
- manifest/bundle summary writer

## Shared helper inputs

All route-specific builders should use the same broad input pattern.

### Base builder inputs

```python
def build_<route_name>_plan(
    *,
    base_recipe_path: str,
    recipe_id: str,
    kind: str,
    fixed_selections: dict[str, Any] | None = None,
    numeric_params: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    custom_selections: dict[str, Any] | None = None,
) -> ExperimentUnitBundlePlan:
    ...
```

Why this input pattern:
- aligns with current recipe contract
- keeps route-specific additions explicit
- lets route-specific builders remain close to `build_yaml_preview()` semantics

## Route-specific builders

### A. single_target_single_model

Recommended function:

```python
def build_single_target_single_model_plan(
    *,
    base_recipe_path: str,
    recipe_id: str,
    kind: str = "baseline",
    fixed_selections: dict[str, Any] | None = None,
    numeric_params: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    custom_selections: dict[str, Any] | None = None,
) -> ExperimentUnitBundlePlan:
    ...
```

What it takes as input:
- one base recipe
- one fully specified set of fixed selections
- numeric/output/custom selections

What it returns:
- bundle plan with exactly one `ChildRecipePlan`
- `compile_ready=True` if required recipe fields are valid

What changes downstream:
- nothing special; this is the canonical one-path case

### B. single_target_model_grid

Recommended function:

```python
def build_single_target_model_grid_plan(
    *,
    base_recipe_path: str,
    recipe_id: str,
    fixed_selections: dict[str, Any],
    model_candidates: list[str],
    tuning_candidates: list[str] | None = None,
    per_model_numeric_overrides: dict[str, dict[str, Any]] | None = None,
    numeric_params: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    custom_selections: dict[str, Any] | None = None,
) -> ExperimentUnitBundlePlan:
    ...
```

What it takes as input:
- one fixed single-target design block
- explicit `model_candidates`
- optional tuning candidates and per-model numeric overrides

What it returns:
- bundle plan with one child recipe per model candidate
- `sweep_spec={"axis": "model", ...}`
- `comparison_spec` describing model-grid comparison
- `compile_ready=False` for current implementation stage unless child-recipe fanout is actually wired

Additional parameters needed:
- `model_candidates`
- optional `tuning_candidates`
- optional `per_model_numeric_overrides`

### C. single_target_full_sweep

Recommended function:

```python
def build_single_target_full_sweep_plan(
    *,
    base_recipe_path: str,
    recipe_id: str,
    fixed_selections: dict[str, Any],
    sweep_axes: dict[str, list[Any]],
    nested_sweep_axes: dict[str, dict[str, list[Any]]] | None = None,
    conditional_rules: dict[str, Any] | None = None,
    numeric_params: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    custom_selections: dict[str, Any] | None = None,
) -> ExperimentUnitBundlePlan:
    ...
```

What it takes as input:
- one fixed design block
- explicit sweep axes
- optional nested sweep axes
- optional conditional rules

What it returns:
- bundle plan with emitted child recipes or an unresolved sweep bundle description
- `sweep_spec` carrying axis definitions
- `comparison_spec` for sweep comparisons

Additional parameters needed:
- `sweep_axes`
- `nested_sweep_axes`
- `conditional_rules`

### D. multi_target_separate_runs

Recommended function:

```python
def build_multi_target_separate_runs_plan(
    *,
    base_recipe_path: str,
    recipe_id: str,
    shared_selections: dict[str, Any],
    target_list: list[str],
    per_target_overrides: dict[str, dict[str, Any]] | None = None,
    numeric_params: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    custom_selections: dict[str, Any] | None = None,
) -> ExperimentUnitBundlePlan:
    ...
```

What it takes as input:
- shared design selections
- explicit `target_list`
- optional target-specific overrides

What it returns:
- bundle plan with one child recipe per target
- `shared_design` block plus emitted child recipes
- `comparison_spec` for cross-target organization

Additional parameters needed:
- `target_list`
- `per_target_overrides`

### E. multi_target_shared_design

Recommended function:

```python
def build_multi_target_shared_design_plan(
    *,
    base_recipe_path: str,
    recipe_id: str,
    shared_selections: dict[str, Any],
    target_list: list[str],
    x_map_policy: str | dict[str, Any] | None = None,
    target_specific_maps: dict[str, Any] | None = None,
    numeric_params: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
    custom_selections: dict[str, Any] | None = None,
) -> ExperimentUnitBundlePlan:
    ...
```

What it takes as input:
- shared design block
- target list
- optional cross-target mapping rules

What it returns:
- bundle plan expressing one shared design with target-specific child recipes or target-specific mapping payloads

Additional parameters needed:
- `target_list`
- `x_map_policy`
- `target_specific_maps`

### F. replication_recipe

Recommended function:

```python
def build_replication_recipe_plan(
    *,
    bundle_id: str,
    recipe_paths: list[str] | None = None,
    paper_id: str | None = None,
    locked_overrides: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
) -> ExperimentUnitBundlePlan:
    ...
```

What it takes as input:
- a recipe bundle identifier or explicit recipe paths
- optional paper identifier
- optional locked overrides

What it returns:
- bundle plan describing pre-authored child recipes
- `compile_ready` depends on existence/validity of child recipe files

Additional parameters needed:
- `bundle_id` or `recipe_paths`
- `paper_id`
- `locked_overrides`

### G. benchmark_suite

Recommended function:

```python
def build_benchmark_suite_plan(
    *,
    base_recipe_path: str,
    recipe_id: str,
    shared_selections: dict[str, Any],
    benchmark_candidates: list[str],
    model_candidates: list[str] | None = None,
    numeric_params: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> ExperimentUnitBundlePlan:
    ...
```

What it takes as input:
- shared design selections
- explicit benchmark candidates
- optional model candidates if suite spans benchmark x model cells

What it returns:
- bundle plan with benchmark comparison structure
- child recipes organized for benchmark-oriented comparison

Additional parameters needed:
- `benchmark_candidates`
- optional `model_candidates`

### H. ablation_study

Recommended function:

```python
def build_ablation_study_plan(
    *,
    base_recipe_path: str,
    recipe_id: str,
    baseline_selections: dict[str, Any],
    ablation_axes: dict[str, list[Any]],
    comparison_label: str | None = None,
    numeric_params: dict[str, Any] | None = None,
    outputs: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> ExperimentUnitBundlePlan:
    ...
```

What it takes as input:
- a baseline design
- explicit ablation axes and candidate toggles
- optional comparison label

What it returns:
- bundle plan with baseline + ablated child recipes
- comparison spec centered on controlled counterfactual differences

Additional parameters needed:
- `baseline_selections`
- `ablation_axes`
- `comparison_label`

## Output contract summary

All route-specific builders should return the same top-level object:
- `ExperimentUnitBundlePlan`

This is important because later code should not need completely different control flow for every route type.

A good normalized output should always tell the caller:
- what route was selected
- whether the bundle is compile-ready
- how many child recipes exist
- what is fixed vs swept
- whether the next action is compile, emit YAMLs, or hand off to wrapper/orchestrator

## Recommended first implementation order

1. public route decision object
- `resolve_experiment_unit_route()`

2. one-path normalized builder
- `build_single_target_single_model_plan()`

3. model-grid normalized builder
- `build_single_target_model_grid_plan()`

4. full-sweep normalized builder
- `build_single_target_full_sweep_plan()`

5. wrapper-owned bundle builders
- multi-target / benchmark / ablation / replication

## Why this design fits the current codebase

It aligns with what already exists:
- `build_yaml_preview()` already builds recipe-shaped dictionaries
- `_route_for_experiment_unit()` already performs route classification
- `compile_experiment_spec_from_recipe()` already consumes one recipe path
- `build_run_manifest()` already expects one compiled run-level identity

So the missing layer is not generic compile logic.
The missing layer is the Stage 0.1 plan-builder layer that converts one route selection into:
- one child recipe, or
- a normalized bundle of child recipes

## Suggested next step

Implement these in two phases:

Phase 1:
- `ExperimentUnitRouteDecision`
- `ChildRecipePlan`
- `ExperimentUnitBundlePlan`
- `resolve_experiment_unit_route()`
- `build_single_target_single_model_plan()`

Phase 2:
- `build_single_target_model_grid_plan()`
- `build_single_target_full_sweep_plan()`
- wrapper-owned bundle builders
