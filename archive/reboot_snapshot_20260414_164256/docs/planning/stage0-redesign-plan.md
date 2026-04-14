# Stage 0 redesign plan

Goal:
Replace the current flat `experiment_unit`-first Stage 0 with a dataset-aware, horse-race-first design frame that matches the package's actual intended workflow.

## Design objective

Stage 0 should no longer begin by asking the user to choose among flat values like:
- `single_target_single_model`
- `benchmark_suite`
- `ablation_study`
- `replication_recipe`

Instead, Stage 0 should:
1. detect whether the request is a replication override
2. infer structural implications from dataset choice
3. fix the baseline design spine
4. declare which dimensions vary
5. declare how the horse race is compared and reported
6. declare execution posture

## Target Stage 0 schema

Recommended order:
1. `replication_input`
2. `dataset_spec`
3. `derived_design_shape`
4. `fixed_design`
5. `varying_design`
6. `comparison_contract`
7. `execution_posture`

## Planning principles

- ordinary own analysis is the implicit default
- replication is the explicit exceptional path
- benchmark comparison is normal, not exceptional
- ablation is one variation structure, not a top-level study intent
- dataset choice comes early because it already constrains frequency and shape
- FRED-MD -> monthly
- FRED-QD -> quarterly
- FRED-SD -> mixed frequency
- FRED-SD combined with others -> bundle / multi-target style branch

## Planned implementation order

### Task 1: Freeze the Stage 0 contract in docs

Objective:
Lock the new Stage 0 meaning before touching runtime logic.

Files:
- Modify: `docs/user-guide/single-run/stages/stage-0/index.md`
- Modify: `docs/user-guide/single-run/stages/stage-0/01-experiment-unit.md`
- Modify: `docs/planning/stage0-redesign-schema.md`

Deliverable:
- public Stage 0 docs stop centering flat `experiment_unit`
- docs explain new order: replication -> dataset -> fixed -> varying -> comparison -> posture

Verification:
- `uv run mkdocs build --strict`

### Task 2: Introduce Stage 0 typed builders and derivation helpers

Objective:
Create the normalized internal contract for the new Stage 0 schema.

Files:
- Create: `macrocast/stage0.py`
- Modify: `macrocast/__init__.py` (only if public export is needed)

Functions to add:
- `resolve_replication_input(...)`
- `derive_dataset_spec(...)`
- `derive_design_shape(...)`
- `build_fixed_design(...)`
- `build_varying_design(...)`
- `build_comparison_contract(...)`
- `build_execution_posture(...)`
- `build_stage0_design_frame(...)`

Core outputs:
- `ReplicationInputDecision`
- `DatasetSpec`
- `DerivedDesignShape`
- `Stage0DesignFrame`

Verification:
- unit tests for dataset derivation and replication detection

### Task 3: Wire dataset-driven derivation rules

Objective:
Implement the current package-specific dataset logic explicitly.

Files:
- Modify: `macrocast/stage0.py`
- Create: `tests/stage0/test_dataset_derivation.py`

Rules to lock:
- `['fred_md']` -> `single_frequency_macro`, `monthly`
- `['fred_qd']` -> `single_frequency_macro`, `quarterly`
- `['fred_sd']` -> `mixed_frequency_state_macro`, `mixed_frequency`
- any set containing `fred_sd` and another dataset -> `mixed_source_bundle`

Verification:
- dataset derivation tests pass

### Task 4: Replace flat first-question logic in the preview builder

Objective:
Move the public flow away from flat `experiment_unit`-first routing.

Files:
- Modify: `macrocast/start.py`
- Modify: `macrocast/choice_stack.py`
- Modify: `tests/start/test_macrocast_start.py`

Changes:
- stop using flat `experiment_unit` as the first public control
- first detect replication override input
- then derive dataset family from selected dataset(s)
- then build a `Stage0DesignFrame`

Compatibility rule:
- existing `meta.experiment_unit` may remain as a derived/internal compatibility field during transition
- do not break existing recipe compile path immediately

Verification:
- existing start tests updated
- new Stage 0 flow tests added

### Task 5: Introduce fixed/varying/comparison grammar

Objective:
Make the normal own-analysis flow package-native.

Files:
- Modify: `macrocast/stage0.py`
- Create: `tests/stage0/test_stage0_design_frame.py`
- Modify: `docs/user-guide/single-run/stages/stage-0/index.md`

What must be representable:
- fixed baseline design
- model horse race
- design robustness
- ablation
- multi-target expansion
- comparison contract with benchmark and tests

Verification:
- design-frame tests pass
- docs explain each block clearly

### Task 6: Keep replication as explicit override path

Objective:
Handle replication without polluting the normal own-analysis front door.

Files:
- Modify: `macrocast/start.py`
- Modify: `macrocast/stage0.py`
- Create: `tests/stage0/test_replication_override.py`

Behavior:
- if replication recipe / bundle / preset is supplied, skip ordinary own-analysis flow
- otherwise do not ask for `own_analysis`

Verification:
- replication override tests pass

### Task 7: Decide whether `experiment_unit` survives as internal derived field

Objective:
Avoid breaking downstream code too early while removing it as the public primary abstraction.

Files:
- Modify: `macrocast/stage0.py`
- Modify: `macrocast/start.py`
- Possibly modify: `macrocast/specs/compiler.py`

Decision target:
- keep `meta.experiment_unit` temporarily as a derived compatibility mirror
- or replace it with a richer derived structure from `Stage0DesignFrame`

Verification:
- recipe compile path still works for compatibility cases

### Task 8: Only after contracts stabilize, design route-specific execution builders

Objective:
Build route-specific bundle planning on top of the new Stage 0 schema, not before it.

Files:
- Modify: `macrocast/stage0.py`
- Possibly create: `macrocast/stage0_builders.py`

Candidates:
- `build_single_path_bundle(...)`
- `build_model_horserace_bundle(...)`
- `build_robustness_bundle(...)`
- `build_ablation_bundle(...)`
- `build_multi_target_bundle(...)`
- `build_replication_bundle(...)`

Verification:
- bundle-plan tests
- manifest contract tests

## Immediate recommendation

Do Tasks 1-3 first.
Do not start execution-builder implementation before the Stage 0 schema and dataset derivation rules are frozen.

## Short rationale

The current problem is not missing execution code first.
The current problem is that the package still lacks the right Stage 0 grammar.
If that grammar is wrong, every later function will encode the wrong abstraction.
