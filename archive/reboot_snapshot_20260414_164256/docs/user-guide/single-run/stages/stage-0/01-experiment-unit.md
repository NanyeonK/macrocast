# 0.1 experiment_unit

`experiment_unit` is the primary Stage 0 routing choice.

It answers one question first:
- what counts as one forecasting exercise in the current request?

This choice matters because it changes the meaning of all later stages.
A later choice like `model`, `target`, or `benchmark` does not mean the same thing if the package is building:
- one fully fixed path,
- one target plus model grid,
- one broader sweep,
- or one multi-run comparison bundle.

## Current design caveat

The current taxonomy exposes eight flat `experiment_unit` values, but conceptually they are not all at the same layer.

More natural package interpretation:
- ordinary own analysis should be the implicit default path
- benchmark comparison is part of the normal horse-race design, not a completely separate first choice
- ablation is a controlled variation on top of a baseline design, not a separate first choice at the same conceptual level as one-path design
- replication is the true exceptional path because it begins from a locked recipe, bundle, or paper-specific preset

So in a cleaner Stage 0 redesign, replication would likely be handled as an explicit special input path, while ordinary analysis would proceed directly into fixed-vs-varying design choices without asking the user to choose `own_analysis` by name.

## What this choice fixes

Selecting `experiment_unit` fixes three things immediately.

1. Route ownership
- does the request remain inside `macrocast_single_run()`?
- or should it move to a future wrapper/orchestrator?

2. Downstream shape
- do later stages behave like ordinary single-path specification choices?
- or do they need sweep semantics or bundle semantics?

3. Compile / preview eligibility
- can the current public entry point continue directly into compile/tree/runs/manifest preview?
- or must it stop because downstream semantics are not implemented yet?

## Current selectable values

Current exposed values in `macrocast/taxonomy/0_meta/experiment_unit.yaml` are:
- `single_target_single_model`
- `single_target_model_grid`
- `single_target_full_sweep`
- `multi_target_separate_runs`
- `multi_target_shared_design`
- `replication_recipe`
- `benchmark_suite`
- `ablation_study`

## Route table and current implementation status

### `single_target_single_model`
What this selects:
- one target variable
- one model family
- one compiled single path

How later stages change:
- later stages behave like ordinary one-path specification choices
- dataset, target, preprocessing, framework, model, benchmark, and output are interpreted as one executable path

Extra parameters implied:
- no extra Stage 0-only parameters beyond the normal later-stage choices
- later numeric parameters such as horizons, lag counts, factor counts, and sample boundaries still live outside `experiment_unit`

Current function support:
- supported by `macrocast_single_run()`
- compile / preview allowed
- implemented in route table through `_EXPERIMENT_UNIT_ROUTES`
- classified by `_route_for_experiment_unit()`

### `single_target_model_grid`
What this selects:
- one target variable
- one fixed design frame
- a model-family sweep rather than one fixed model

How later stages change:
- later stages must split into:
  - fixed single-target choices
  - model/tuning sweep choices
- the meaning of `model` changes from one chosen model to a sweep dimension

Extra parameters implied:
- model-grid membership or model allowlist
- possibly tuning-grid scope per model family
- possibly comparison/reporting controls across model candidates

Current function support:
- value exists in taxonomy and route table
- `_route_for_experiment_unit()` recognizes it
- `macrocast_single_run()` classifies it as `planned_single_run_extension`
- compile / preview currently blocked
- there is no dedicated downstream implementation function yet

### `single_target_full_sweep`
What this selects:
- one target variable
- one exercise where multiple later axes may vary jointly

How later stages change:
- later stages must distinguish:
  - fixed axes
  - sweep axes
  - nested or conditional sweep axes
- this is broader than a model-only grid

Extra parameters implied:
- explicit sweep-axis declarations
- nested-sweep structure
- conditional sweep rules
- possibly cross-product / restricted-grid controls

Current function support:
- value exists in taxonomy and route table
- `_route_for_experiment_unit()` recognizes it
- `macrocast_single_run()` classifies it as `planned_single_run_extension`
- compile / preview currently blocked
- there is no dedicated downstream implementation function yet

### `multi_target_separate_runs`
What this selects:
- multiple target variables
- each target emitted as a separate child run

How later stages change:
- target becomes a fan-out dimension
- later stages no longer describe one path; they describe a shared request that emits many one-path runs

Extra parameters implied:
- target list
- emission rule for child YAMLs
- per-target naming / output organization
- aggregation or comparison policy across emitted runs

Current function support:
- value exists in taxonomy and route table
- `_route_for_experiment_unit()` recognizes it
- `macrocast_single_run()` classifies it as `wrapper_required`
- compile / preview blocked in current public entry point
- no wrapper/orchestrator function exists yet in the active public API

### `multi_target_shared_design`
What this selects:
- multiple targets under one shared fixed design frame

How later stages change:
- some later choices remain shared across all targets
- some target-specific mappings may still vary
- the workflow becomes a structured multi-run bundle, not one path

Extra parameters implied:
- target set
- shared fixed-axis block
- target-specific mapping rules where allowed
- bundle-level result organization

Current function support:
- value exists in taxonomy and route table
- `_route_for_experiment_unit()` recognizes it
- `macrocast_single_run()` classifies it as `wrapper_required`
- compile / preview blocked in current public entry point
- no dedicated wrapper/orchestrator function exists yet

### `replication_recipe`
What this selects:
- a pre-authored replication bundle rather than a generic choice-by-choice design

How later stages change:
- later choices should not be interpreted as a normal interactive one-path build
- the package should instead load or reference authored replication recipes

Extra parameters implied:
- recipe-bundle identifier
- paper-study selection metadata
- possibly locked benchmark / target / evaluation conventions

Current function support:
- value exists in taxonomy and route table
- `_route_for_experiment_unit()` recognizes it
- `macrocast_single_run()` classifies it as `wrapper_required`
- compile / preview blocked in current public entry point
- no dedicated replication-bundle entry function exists yet in the public API

### `benchmark_suite`
What this selects:
- a benchmark-centric comparison suite across multiple runs or candidates

How later stages change:
- benchmark specification becomes bundle-level logic rather than one path detail
- later stages must support comparison across multiple candidate runs

Extra parameters implied:
- benchmark family set
- suite membership
- comparison summary and reporting structure

Current function support:
- value exists in taxonomy and route table
- `_route_for_experiment_unit()` recognizes it
- `macrocast_single_run()` classifies it as `wrapper_required`
- compile / preview blocked in current public entry point
- no dedicated benchmark-suite entry function exists yet

### `ablation_study`
What this selects:
- a controlled comparison bundle where one or more ingredients are systematically turned on/off

How later stages change:
- later choices must support baseline-vs-variant comparisons
- the workflow becomes a coordinated bundle, not one ordinary path

Extra parameters implied:
- baseline specification
- ablation dimensions
- comparison rules and reporting structure

Current function support:
- value exists in taxonomy and route table
- `_route_for_experiment_unit()` recognizes it
- `macrocast_single_run()` classifies it as `wrapper_required`
- compile / preview blocked in current public entry point
- no dedicated ablation-bundle entry function exists yet

## Actual functions that currently exist

The following functions or structures currently exist and are active:
- `macrocast_single_run()`
- `_route_for_experiment_unit()`
- `_route_for_recipe()`
- `_interactive_wizard()`
- `_EXPERIMENT_UNIT_ROUTES`
- `macrocast/taxonomy/0_meta/experiment_unit.yaml`

These currently do exist and are the real operational machinery behind `0.1 experiment_unit`.

## Functions that do not currently exist

There are currently no dedicated public execution functions such as:
- `macrocast_model_grid_run()`
- `macrocast_full_sweep_run()`
- `macrocast_multi_target_runs()`
- `macrocast_benchmark_suite()`
- `macrocast_ablation_study()`
- `macrocast_replication_bundle()`

So for most nontrivial `experiment_unit` values, the package currently has:
- taxonomy support
- route classification support
- blocking / handoff messaging

but not a fully implemented downstream execution path.

## Present implementation summary

Implemented end-to-end enough for current public entry point:
- `single_target_single_model`

Recognized but not yet downstream-implemented inside `macrocast_single_run()`:
- `single_target_model_grid`
- `single_target_full_sweep`

Recognized and intentionally deferred to a future wrapper/orchestrator:
- `multi_target_separate_runs`
- `multi_target_shared_design`
- `replication_recipe`
- `benchmark_suite`
- `ablation_study`

## Read next

After `0.1 experiment_unit`, the next relevant pages are:
- [Choice Stack](choice-stack.md)
- [Stage 1](../stage-1/index.md)
