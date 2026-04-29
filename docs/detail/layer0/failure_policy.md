# 4.0.2 Failure Handling

- Parent: [4.0 Layer 0: Study Scope](index.md)
- Previous: [4.0.1 Study Scope](study_scope.md)
- Current: `failure_policy`
- Next: [4.0.3 Reproducibility](reproducibility_mode.md)

`failure_policy` controls what happens when a recipe, sweep variant, model branch, target branch, or cell fails.

This is a runtime discipline axis. It does not change the statistical model. It changes whether the runner stops, skips, warns, or preserves partial artifacts.

The default is `fail_fast`. Users do not need to choose this axis for ordinary runs; the compiler and runtime treat an omitted `failure_policy` as `fail_fast`.

## Where It Lives In Code

| Purpose | Function or object |
|---|---|
| Registry entries | `macrocast.registry.stage0.failure_policy.FAILURE_POLICY_ENTRIES` |
| Compiler runtime support gate | `macrocast.compiler.build._execution_status` |
| Manifest payload | `macrocast.compiler.build.compiled_spec_to_dict` |
| Execution payload reader | `macrocast.execution.build._failure_policy_spec` |
| Direct recipe runtime | `macrocast.execution.build.execute_recipe` |
| Sweep parent policy reader | `macrocast.execution.sweep_runner._extract_parent_failure_policy` |
| Sweep runtime | `macrocast.execution.sweep_runner.execute_sweep` |
| Ablation default helper | `macrocast.studies.ablation._ensure_baseline_failure_policy` |

## Choices

Read this axis as the run's error-handling policy. It does not change the model; it changes whether the runtime stops or records failed units and continues.

Only executable policies are exposed as public choices.

### Quick Map

| Choice | State | Best Use |
|---|---|---|
| `fail_fast` | runnable, default | ordinary runs, debugging, replication |
| `skip_failed_cell` | runnable | large sweeps |
| `skip_failed_model` | runnable | multi-model direct runs |
| `save_partial_results` | runnable | long runs where partial artifacts matter |
| `warn_only` | runnable | exploratory runs |

### `fail_fast`

Use this when a failure should stop the run immediately. This is the package default.

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: fail_fast
```

Runtime behavior:

```text
direct recipe = re-raise first error
sweep runner  = stop at first failed variant
```

If the recipe omits `failure_policy`, the runtime behaves as if this value were selected:

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_one_method
```

### `skip_failed_cell`

Use this for large controlled sweeps where some cells may be invalid.

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_compare_methods
      failure_policy: skip_failed_cell
```

Runtime behavior:

```text
sweep runner = record failed variant in study_manifest.json
next step    = continue with remaining variants
```

### `skip_failed_model`

Use this when one model branch may fail but other target/model branches should still run.

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: skip_failed_model
```

Runtime behavior:

```text
direct recipe = continue past recoverable model/prediction failures
artifact      = record failed components
```

### `save_partial_results`

Use this when completed artifacts are valuable even if a later component fails.

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: save_partial_results
```

Runtime behavior:

```text
direct recipe = preserve completed outputs where supported
sweep runner  = preserve completed variants and failure metadata
```

### `warn_only`

Use this for exploratory runs where recoverable failures should be visible but not fatal.

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: warn_only
```

Runtime behavior:

```text
warning type = RuntimeWarning
artifact     = failed units recorded
run status   = continues where recoverable
```

## Failure Scope

The same axis is read at two levels:

- Sweep level: `execute_sweep()` reads it from the parent recipe with `_extract_parent_failure_policy()`.
- Direct recipe level: `execute_recipe()` reads it from compiler provenance with `_failure_policy_spec()`.

This matters because `skip_failed_cell` is mainly a sweep-cell policy, while `skip_failed_model`, `save_partial_results`, and `warn_only` are also meaningful inside direct recipe execution.

## YAML

Default behavior can be left implicit:

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_one_method
```

Or written explicitly:

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: fail_fast
```

For a controlled sweep:

```yaml
path:
  0_meta:
    fixed_axes:
      study_scope: one_target_compare_methods
      failure_policy: skip_failed_cell
```

For exploratory runs where recoverable failures should be visible but not fatal:

```yaml
path:
  0_meta:
    fixed_axes:
      failure_policy: warn_only
```

## Runtime Artifacts

The compiler writes:

```json
"failure_policy_spec": {
  "failure_policy": "fail_fast"
}
```

Sweep execution writes variant status and error text into `study_manifest.json`. Direct recipe execution records failed components and can still write partial outputs when the selected policy allows continuation.

## Guidance

Use `fail_fast` for debugging and replication.

Use `skip_failed_cell` for large sweeps where invalid combinations are expected.

Use `warn_only` for exploratory work where warnings should be visible in logs but the run should continue.

The public axis intentionally contains only executable policies. Retry and
hyperparameter-fallback behavior should be added later only when the runtime
executor exists.
