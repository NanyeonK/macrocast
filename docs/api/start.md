# Start / single-run inspection API Reference

## Import surface

```python
from macrocast import macrocast_single_run
```

## Purpose

`macrocast_single_run()` is the minimal public entry point for inspecting one live recipe path through the rebuilt compiler surface.

This slice is intentionally narrow:
- it inspects route ownership
- it inspects compile status
- it exposes `tree_context` provenance
- it previews run/manifest paths only when the route is executable
- it does not perform hidden execution
- it does not pretend wrapper-owned or internal-sweep routes are runnable single paths

## Function

```python
def macrocast_single_run(
    *,
    yaml_path: str,
    stages: str | Iterable[str] | None = None,
    output_root: str = "/tmp/macrocast_single_run_preview",
) -> dict:
    ...
```

## Stages

Available stages:
- `route_preview`
- `compile_preview`
- `tree_context`
- `runs_preview`
- `manifest_preview`

Default behavior includes all five stages.

Important rule:
- if the compiled route is not executable, `runs_preview` and `manifest_preview` are blocked explicitly
- compile/tree inspection still remains available so the user can see why the route is blocked

## Output semantics

### `route_preview`
Returns:
- `route_owner`
- `execution_status`
- `wizard_status`
- `continue_in_single_run`
- `message`
- `warnings`
- `blocked_reasons`
- compact `tree_context_summary`
- optional `wrapper_handoff`

### `compile_preview`
Returns the serialized compiler manifest from `compiled_spec_to_dict()`.

### `tree_context`
Returns the top-level tree-context provenance payload.

### `runs_preview`
Only for executable routes. Returns deterministic output-path previews without executing the run.

### `manifest_preview`
Only for executable routes. Returns the expected manifest skeleton and expected artifact file set without executing the run.

## Example

```python
out = macrocast_single_run(yaml_path="examples/recipes/model-benchmark.yaml")

out["route_preview"]
out["tree_context"]
```

## Notes

This is a route-inspection slice, not a full guided YAML-building wizard restore.
The live repo no longer contains the archived pre-reboot wizard support stack, so this surface stays aligned to the current compiler/runtime architecture rather than recreating obsolete helper layers.
