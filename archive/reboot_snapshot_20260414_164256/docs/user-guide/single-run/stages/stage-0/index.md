# Stage 0

Stage 0 is the stage that fixes the design frame of the forecasting exercise before the package starts filling in the empirical specification.

In practical terms, Stage 0 answers the questions that must be settled before later stages can be interpreted correctly:
- what counts as one experiment unit?
- is the current request still one executable single-run path?
- which later choices should be treated as fixed design choices and which should later be allowed to vary?
- can the package continue directly into compile / preview, or should it stop and hand off to a future wrapper/orchestrator?

## Dataset-first implication

In the current package scope, some Stage 1-looking choices are already partially constrained by the dataset family.

Practical package rule:
- `fred_md` implies monthly work
- `fred_qd` implies quarterly work
- `fred_sd` is mixed-frequency and becomes structurally different from the ordinary single-target path, especially when combined with other datasets

So in the next Stage 0 redesign, dataset choice should move earlier in the design frame and frequency should often be derived rather than chosen independently for the standard FRED paths.

## What Stage 0 fixes

Stage 0 fixes the high-level structure of the exercise.

It fixes five kinds of things.

### 1. Design identity
Stage 0 fixes what the package is building at the highest level.

Examples:
- one target + one model
- one target + model grid
- one target + broader sweep
- a multi-run comparison object

### 2. Route ownership
Stage 0 fixes whether the request still belongs to `macrocast_single_run()` or whether it has already crossed into a future wrapper/orchestrator problem.

### 3. Sweep semantics
Stage 0 fixes whether later choices should be interpreted as:
- fixed design choices
- sweep choices
- nested sweep choices
- conditional choices

Without this step, later options look flat even when they are not.

### 4. Execution posture
Stage 0 fixes how strict and operational the exercise should be.

Examples:
- strict vs exploratory reproducibility
- fail-fast vs skip / retry behavior
- serial vs parallel / distributed execution intent

### 5. Metadata contract
Stage 0 fixes which values belong in `recipe.meta` rather than in the ordinary taxonomy path.

This matters because Stage 0 choices describe the structure of the exercise, not only one later empirical selection inside the exercise.

## What Stage 0 does not fix

Stage 0 does not yet fix the substantive empirical specification.

It does not choose:
- dataset
- target variable
- predictor set
- preprocessing recipe
- model class
- benchmark winner

Those belong to later stages.

So Stage 0 fixes the frame of the forecasting exercise, not its full empirical content.

## Stage 0 subprocesses

Stage 0 currently has six subprocesses.

### 0.1 experiment_unit
This is the primary routing decision.
It fixes what one experiment unit means and therefore changes the meaning of the rest of the workflow.

This is the most important Stage 0 choice because it decides whether the later stages describe:
- one path
- one sweep family
- or one multi-run comparison object

### 0.2 axis_type
This classifies whether a later choice should be treated as:
- fixed
- sweep
- nested_sweep
- conditional
- derived
- eval_only
- report_only

This is where the package starts separating fixed design from research variation.

### 0.3 registry_type
This classifies what kind of source defines a choice.

Examples:
- enum registry
- numeric registry
- callable registry
- user YAML
- external adapter

This matters because not all choices should be collected, validated, or serialized in the same way.

### 0.4 reproducibility_mode
This fixes how strict the exercise should be about replication, seeds, and controlled reruns.

### 0.5 failure_policy
This fixes what the package should do when one step, one model, or one cell fails.

### 0.6 compute_mode
This fixes the intended execution posture.

Examples:
- serial
- parallel by model
- parallel by horizon
- GPU-backed execution
- distributed execution

## Why Stage 0 comes first

The package workflow is hierarchical, not flat.

If Stage 0 changes, the meaning of later stages changes.
For example:
- if `experiment_unit` is a true one-path object, later stages behave like ordinary specification choices
- if `experiment_unit` is a model-grid object, later stages must distinguish fixed choices from sweep choices
- if `experiment_unit` is a wrapper-owned object, the package should not pretend that later choices still describe one executable single-run path

So Stage 0 must come before Stage 1, not after it.

## Current package interpretation

At the moment, the package draws the following practical boundary:
- `macrocast_single_run()` is the owner of the single-target family
- true multi-run comparison objects belong to a future wrapper/orchestrator layer

This means Stage 0 is also the place where the package decides whether it can continue with the current public entry point.

## Read next

After this page, the next two pages to read are:
- [Choice Stack](choice-stack.md)
- [Stage 1](../stage-1/index.md)
