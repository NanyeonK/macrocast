# Stage 0 — Meta Layer

Stage 0 is the control layer that decides what kind of forecasting design the package is building before it spends effort on data selection, preprocessing, estimation, evaluation, or interpretation.

This stage is not cosmetic metadata.
It is the stage that fixes the high-level structure of the exercise.

In forecasting-paper language, Stage 0 fixes the design frame before the package starts filling in the empirical specification.

## What Stage 0 fixes

Stage 0 fixes the choices that determine the shape of the rest of the workflow.

At minimum, it fixes:
- what one experiment unit is
- whether the current request is still one executable single-run path
- whether later stages should be interpreted as fixed-path choices or sweep-aware choices
- whether the package can continue into compile / preview immediately
- whether the request must stop and hand off to a future wrapper/orchestrator

More concretely, Stage 0 fixes five kinds of things:

1. Design identity
- is this one target + one model?
- one target + model grid?
- one target + full sweep?
- or already a multi-run comparison object?

2. Route ownership
- does the request still belong to `macrocast_single_run()`?
- or has it crossed into wrapper/orchestrator territory?

3. Sweep semantics
- should later choices be treated as fixed design choices?
- or do they need sweep / nested-sweep behavior?

4. Execution posture
- how strict is reproducibility?
- how should failure be handled?
- what compute mode is intended?

5. Metadata contract
- which values belong in `recipe.meta`
- which values do not belong in `taxonomy_path`

That is why Stage 0 must run before the rest of the wizard.

## Why Stage 0 exists

Without Stage 0, the wizard looks like a flat menu of choices.
That is misleading.
Different `experiment_unit` values imply different downstream shapes:

- one fully fixed path
- one target plus model sweep
- one target plus many sweep axes
- many emitted runs under one shared design
- a benchmark-comparison suite
- an ablation bundle

Those are not small parameter tweaks. They are different execution graphs.

So the package needs one explicit routing layer that decides whether the current call is:
- still a true `macrocast_single_run()` path, or
- already asking for multi-run orchestration behavior

## What Stage 0 does not fix

Stage 0 does not fix the substantive empirical specification yet.
It does not choose:
- which dataset is used
- which target variable is forecast
- which predictor pool is used
- which preprocessing recipe is applied
- which model is estimated
- which benchmark wins

Those are later-stage choices.

So Stage 0 fixes the frame of the exercise, not the full content of the exercise.

## Stage 0 subprocesses

Stage 0 has six subprocesses or substages in the current design.

### 0.1 experiment_unit
This is the primary routing decision.
It fixes what counts as one experiment unit and therefore changes the meaning of the rest of the workflow.

### 0.2 axis_type
This classifies whether a choice should later be treated as:
- fixed
- sweep
- nested_sweep
- conditional
- derived
- eval_only
- report_only

This is where the package starts deciding which later choices belong to the fixed design and which belong to research variation.

### 0.3 registry_type
This identifies what kind of source defines a choice:
- enum registry
- numeric registry
- callable registry
- plugin-like extension
- user YAML
- external adapter

This matters because not all later choices should be collected or validated in the same way.

### 0.4 reproducibility_mode
This fixes how strict the run should be about replication and seed control.

### 0.5 failure_policy
This fixes how the workflow should behave when one step, one model, or one cell fails.

### 0.6 compute_mode
This fixes the intended execution posture:
- serial
- parallel by model/horizon/trial
- GPU
- distributed execution

## Stage 0 responsibilities

Stage 0 owns:
- route classification
- Stage 0 metadata persistence in recipe YAML
- early stop / continue decision
- compile-preview eligibility
- later-stage visibility rules
- fixed-vs-sweep framing for later stages

Stage 0 does not own:
- dataset loading
- preprocessing execution
- model fitting
- statistical testing
- output rendering

Those belong to later layers. Stage 0 only decides what kind of object those later layers are being asked to build.

## Stage 0 code path

Current code entry points:

- `macrocast/start.py`
  - `macrocast_single_run()`
  - `_interactive_wizard()`
  - `_ensure_meta()`
  - `_route_for_experiment_unit()`
  - `_route_for_recipe()`
- `macrocast/choice_stack.py`
  - `STAGE0_META_KEYS`
  - `build_choice_stack()`
  - `build_yaml_preview()`

Current execution logic:

1. `macrocast_single_run()` starts the wizard or loads a user YAML.
2. `_ensure_meta()` guarantees the recipe has a `meta` block.
3. Stage 0 keys are treated specially through `STAGE0_META_KEYS`.
4. `_route_for_experiment_unit()` maps the selected unit into one of the routing classes.
5. `_route_for_recipe()` applies that routing to the current recipe.
6. `_interactive_wizard()` asks `experiment_unit` first and can stop immediately if the route is not executable inside current single-run flow.
7. `macrocast_single_run()` blocks compile/tree/runs/manifest preview when the selected route is wrapper-owned or still unimplemented.

That is the current package contract.

## Stage 0 YAML contract

Stage 0 values belong in the recipe `meta` block.

Current contract:

```yaml
meta:
  experiment_unit: single_target_single_model
  reproducibility_mode: strict_reproducible
  failure_policy: fail_fast
  compute_mode: serial
```

Rule:
- Stage 0 values should not be stored as ordinary `taxonomy_path` leaves.
- Stage 0 values should not silently fall into `custom_selections`.

Reason:
- `taxonomy_path` describes the selected path within one experiment design.
- Stage 0 decides what kind of experiment design object is being built in the first place.

So Stage 0 is above the ordinary taxonomy path.

## How Stage 0 changes the wizard

If Stage 0 says the run is `single_target_single_model`, the rest of the wizard can proceed as a normal one-path builder.

If Stage 0 says the run is `single_target_model_grid`, the wizard still belongs to `macrocast_single_run()`, but the downstream stack must branch into:
- fixed single-target choices
- model sweep choices
- tuning/grid-specific visibility rules

If Stage 0 says the run is `single_target_full_sweep`, the wizard still belongs to `macrocast_single_run()`, but the downstream stack must branch into:
- fixed axes
- sweep axes
- nested or conditional sweep axes

If Stage 0 says the run is `multi_target_*`, `benchmark_suite`, `replication_recipe`, or `ablation_study`, then the package is no longer describing one executable single path. It is describing a multi-run object. In that case, `macrocast_single_run()` should stop and hand off to a future wrapper/orchestrator.

That is the key structural boundary.

## 0.1 experiment_unit

Question:
- What is one experiment unit?

This is the most important choice in the entire wizard because it decides whether all later choices describe:
- one run,
- one sweep family, or
- one orchestration bundle.

## 0.1 source-plan universe

Options from source plan:
- `single_target_single_model`
- `single_target_model_grid`
- `single_target_full_sweep`
- `multi_target_separate_runs`
- `multi_target_shared_design`
- `multi_output_joint_model`
- `hierarchical_forecasting_run`
- `panel_forecasting_run`
- `state_space_run`
- `replication_recipe`
- `benchmark_suite`
- `ablation_study`

Current implemented taxonomy file:
- `macrocast/taxonomy/0_meta/experiment_unit.yaml`

Current exposed set in code:
- `single_target_single_model`
- `single_target_model_grid`
- `single_target_full_sweep`
- `multi_target_separate_runs`
- `multi_target_shared_design`
- `replication_recipe`
- `benchmark_suite`
- `ablation_study`

Not yet exposed in the active current route table:
- `multi_output_joint_model`
- `hierarchical_forecasting_run`
- `panel_forecasting_run`
- `state_space_run`

So the current package state is a reduced operational subset of the larger source-plan universe.

## 0.1 current package interpretation

Current decision:
- `macrocast_single_run()` owns Stage 0 routing plus the single-target family.
- near-term executable route is only `single_target_single_model`.
- future single-run extensions still owned by `macrocast_single_run()` are:
  - `single_target_model_grid`
  - `single_target_full_sweep`
- wrapper/orchestrator-owned families are:
  - `multi_target_separate_runs`
  - `multi_target_shared_design`
  - `replication_recipe`
  - `benchmark_suite`
  - `ablation_study`

This means the ownership split is not “simple versus complex.”
It is:
- one-target family stays in single-run entry point
- multi-run comparison / fan-out family moves to wrapper/orchestrator

## 0.1 route table

### A. Owned by `macrocast_single_run()`

#### `single_target_single_model`
- status: implemented
- meaning: one target, one model family, one compiled single path
- downstream structure: ordinary one-path wizard
- compile preview: allowed
- current role: canonical current single-run mode

#### `single_target_model_grid`
- status: planned single-run extension
- meaning: one target, fixed design, model family becomes a sweep axis
- downstream structure: fixed path + model/tuning branch
- compile preview: currently blocked
- reason blocked: downstream branch semantics are not implemented yet

#### `single_target_full_sweep`
- status: planned single-run extension
- meaning: one target, but several later choices become sweep axes
- downstream structure: fixed axes + multi-axis sweep branch
- compile preview: currently blocked
- reason blocked: downstream branch semantics are not implemented yet

### B. Owned by future wrapper/orchestrator

#### `multi_target_separate_runs`
- status: wrapper-required
- meaning: emit multiple single-target runs
- downstream structure: one shared high-level request becomes many single-run YAMLs
- compile preview in `macrocast_single_run()`: blocked

#### `multi_target_shared_design`
- status: wrapper-required
- meaning: many targets under one shared fixed design
- downstream structure: shared fixed axes + emitted per-target runs
- compile preview in `macrocast_single_run()`: blocked

#### `replication_recipe`
- status: wrapper-required
- meaning: load a pre-authored recipe bundle or paper-specific run family
- downstream structure: authored bundle, not generic stepwise single-path branching
- compile preview in `macrocast_single_run()`: blocked

#### `benchmark_suite`
- status: wrapper-required
- meaning: benchmark-centric comparison object across multiple runs
- downstream structure: coordinated benchmark comparison suite
- compile preview in `macrocast_single_run()`: blocked

#### `ablation_study`
- status: wrapper-required
- meaning: controlled counterfactual run family
- downstream structure: coordinated comparison bundle
- compile preview in `macrocast_single_run()`: blocked

## 0.1 function mapping

This section should be read alongside `macrocast/start.py`.

### `_EXPERIMENT_UNIT_ROUTES`
Location:
- `macrocast/start.py`

Role:
- central route table for current package behavior
- maps each exposed `experiment_unit` to:
  - `owner`
  - `status`
  - `shape`
  - `compile_allowed`
  - `continue_in_single_run`
  - `message`

Interpretation:
- this is the operational truth for current Stage 0 routing
- docs should stay aligned with this mapping

### `_ensure_meta(recipe)`
Location:
- `macrocast/start.py`

Role:
- guarantees the recipe has a `meta` block
- seeds `meta.experiment_unit = single_target_single_model` when absent

Interpretation:
- Stage 0 is no longer optional hidden state
- every recipe gets an explicit routing context

### `STAGE0_META_KEYS`
Location:
- `macrocast/choice_stack.py`

Role:
- declares which keys must be written into `recipe.meta`
- currently includes:
  - `experiment_unit`
  - `axis_type`
  - `registry_type`
  - `reproducibility_mode`
  - `failure_policy`
  - `compute_mode`

Interpretation:
- these keys sit above ordinary taxonomy-path selection

### `_route_for_experiment_unit(experiment_unit)`
Location:
- `macrocast/start.py`

Role:
- converts the raw selected unit into route metadata
- decides whether the choice belongs to:
  - implemented single-run path
  - planned single-run extension
  - wrapper-required handoff

Interpretation:
- this is the actual classification step for 0.1

### `_route_for_recipe(recipe)`
Location:
- `macrocast/start.py`

Role:
- reads the recipe meta block and applies `_route_for_experiment_unit()`

Interpretation:
- the route is a property of the recipe, not only of the live wizard state

### `_interactive_wizard(...)`
Location:
- `macrocast/start.py`

Role:
- asks Stage 0 first
- writes YAML immediately after the answer
- after `experiment_unit`, checks the route
- if `continue_in_single_run` is false, it stops and returns routing metadata instead of pretending the full wizard is executable

Interpretation:
- this is the current runtime enforcement of the Stage 0 boundary

### `macrocast_single_run(...)`
Location:
- `macrocast/start.py`

Role:
- public entry point
- can run in interactive mode or yaml-path preview mode
- blocks compile/tree/runs/manifest preview when `compile_allowed` is false

Interpretation:
- Stage 0 does not only affect docs wording
- it directly changes runtime behavior

## 0.1 design rule

Design rule for now:
- `macrocast_single_run()` may classify many route types
- but it should only execute the route families whose downstream semantics are already implemented

That prevents a fake single-run surface that silently accepts multi-run intent without a real execution contract.

## 0.1 what is still missing

Current gaps after the routing model lock:

1. downstream branch spec for `single_target_model_grid`
- which later choices stay fixed?
- which choices become sweep-aware?
- how should model-grid YAML be serialized?

2. downstream branch spec for `single_target_full_sweep`
- which axes are eligible for sweep?
- how should nested sweep and conditional sweep be represented?
- how should manifests record sweep semantics?

3. wrapper/orchestrator handoff contract
- how does a wrapper emit or reference multiple child YAMLs?
- what is the bundle manifest?
- what shared fixed axes belong at wrapper level?

4. source-plan expansion policy
- when and how should currently unexposed units like `panel_forecasting_run` or `state_space_run` enter `_EXPERIMENT_UNIT_ROUTES`?

Until those are designed, Stage 0 should remain explicit about what is owned, what is blocked, and why.

## 0.2 axis_type

Question:
- Is a choice fixed, swept, conditional, or derived?

Options:
- `fixed`
- `sweep`
- `nested_sweep`
- `conditional`
- `derived`
- `eval_only`
- `report_only`

Current note:
- this concept is present in taxonomy and docs but not yet fully activated as a first-class Stage 0 runtime branching rule.

## 0.3 registry_type

Question:
- What kind of registry or source defines this choice?

Options:
- `enum_registry`
- `numeric_registry`
- `callable_registry`
- `custom_plugin`
- `user_defined_yaml`
- `external_adapter`

Current note:
- this remains mostly descriptive planning state; it is not yet a strong runtime branch in the wizard.

## 0.4 reproducibility_mode

Question:
- How strict should reproducibility be?

Options:
- `strict_reproducible`
- `seeded_reproducible`
- `best_effort`
- `exploratory`

Current note:
- stored under `meta`
- conceptually should affect seeding, logging, and preview / execution guarantees later

## 0.5 failure_policy

Question:
- What should happen when a step fails?

Options:
- `fail_fast`
- `skip_failed_cell`
- `skip_failed_model`
- `retry_then_skip`
- `fallback_to_default_hp`
- `save_partial_results`
- `warn_only`
- `hard_error`

Current note:
- stored under `meta`
- runtime semantics still need deeper integration in execution layer

## 0.6 compute_mode

Question:
- How should execution resources be used?

Options:
- `serial`
- `parallel_by_model`
- `parallel_by_horizon`
- `parallel_by_oos_date`
- `parallel_by_trial`
- `gpu_single`
- `gpu_multi`
- `distributed_cluster`

Current note:
- stored under `meta`
- runtime semantics still need deeper integration in execution layer

## Stage 0 summary

Current Stage 0 truth is:
- Stage 0 is the routing gate, not metadata decoration.
- `experiment_unit` is the decisive choice.
- Stage 0 values live in `recipe.meta`.
- `macrocast_single_run()` owns Stage 0 and the single-target family.
- only `single_target_single_model` is executable end-to-end today.
- route-aware blocking is intentional and correct until branch-specific downstream semantics are implemented.

## What should happen next

Before expanding the rest of the wizard, the package should decide:
1. exact downstream branch spec for `single_target_model_grid`
2. exact downstream branch spec for `single_target_full_sweep`
3. minimum handoff contract for wrapper/orchestrator-owned route families
4. exposure policy for currently unimplemented source-plan units
