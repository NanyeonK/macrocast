# Stage 0 API Reference

## Import surface

```python
from macrocast.stage0 import (
    FixedDesign,
    VaryingDesign,
    ComparisonContract,
    ReplicationInput,
    Stage0Frame,
    build_stage0_frame,
    resolve_route_owner,
    check_stage0_completeness,
    stage0_summary,
    stage0_to_dict,
    stage0_from_dict,
)
```

## Object reference

### `FixedDesign`

Fields:
- `dataset_adapter`
- `information_set`
- `sample_split`
- `benchmark`
- `evaluation_protocol`
- `forecast_task`

Purpose:
- hold the fairness-defining common environment of the study

### `VaryingDesign`

Fields:
- `model_families`
- `feature_recipes`
- `preprocess_variants`
- `tuning_variants`
- `horizons`

Purpose:
- hold explicitly allowed study variation

### `ComparisonContract`

Fields:
- `information_set_policy`
- `sample_split_policy`
- `benchmark_policy`
- `evaluation_policy`

Purpose:
- define the fairness conditions that comparisons must satisfy

### `ReplicationInput`

Fields:
- `source_type`
- `source_id`
- `locked_constraints`
- `override_reason`

Purpose:
- represent an explicit replication-driven override path

### `Stage0Frame`

Fields:
- `study_mode`
- `fixed_design`
- `comparison_contract`
- `varying_design`
- `execution_posture`
- `design_shape`
- `replication_input`
- `experiment_unit`

Purpose:
- canonical Stage 0 output consumed by downstream layers

## Function reference

### `build_stage0_frame()`

Signature:

```python
def build_stage0_frame(
    *,
    study_mode: str,
    fixed_design: FixedDesign | dict,
    comparison_contract: ComparisonContract | dict,
    varying_design: VaryingDesign | dict | None = None,
    replication_input: ReplicationInput | dict | None = None,
) -> Stage0Frame:
    ...
```

Behavior:
- normalizes inputs
- validates required structure
- derives `design_shape`
- derives `execution_posture`
- derives compatibility mirror `experiment_unit`
- returns immutable `Stage0Frame`

### `resolve_route_owner()`

```python
def resolve_route_owner(stage0: Stage0Frame) -> str:
    ...
```

Returns one of:
- `single_run`
- `wrapper`
- `replication`

### `check_stage0_completeness()`

```python
def check_stage0_completeness(stage0: Stage0Frame) -> None:
    ...
```

Raises explicit Stage 0 completeness errors if the frame is not ready for execution.

### `stage0_summary()`

```python
def stage0_summary(stage0: Stage0Frame) -> str:
    ...
```

Returns a compact human-readable summary string.

### `stage0_to_dict()` / `stage0_from_dict()`

```python
def stage0_to_dict(stage0: Stage0Frame) -> dict:
    ...


def stage0_from_dict(payload: dict) -> Stage0Frame:
    ...
```

Purpose:
- dict serialization and reconstruction

## Example

```python
from macrocast.stage0 import build_stage0_frame

stage0 = build_stage0_frame(
    study_mode="single_path_benchmark_study",
    fixed_design={
        "dataset_adapter": "fred_md",
        "information_set": "revised_monthly",
        "sample_split": "expanding_window_oos",
        "benchmark": "ar_bic",
        "evaluation_protocol": "point_forecast_core",
        "forecast_task": "single_target_point_forecast",
    },
    comparison_contract={
        "information_set_policy": "identical",
        "sample_split_policy": "identical",
        "benchmark_policy": "identical",
        "evaluation_policy": "identical",
    },
    varying_design={
        "model_families": ("ar", "ridge", "rf"),
        "horizons": ("h1", "h3", "h6", "h12"),
    },
)
```

## Notes

- Stage 0 is grammar-first, not registry-first.
- The public API is intentionally compact so the package stays pythonic.
- Large conceptual Stage 0 categories remain architecture guidance unless runtime use truly requires them.


## Error classes

The Stage 0 layer uses explicit errors rather than generic failures.

- `Stage0Error`
- `Stage0NormalizationError`
- `Stage0ValidationError`
- `Stage0CompletenessError`
- `Stage0RoutingError`

## Round-trip helpers

`stage0_to_dict()` and `stage0_from_dict()` support serialization and reconstruction of the canonical Stage 0 frame.

This is useful for:
- recipe persistence
- config export
- deterministic reconstruction from saved payloads

## Replication behavior

When `replication_input` is supplied or `study_mode="replication_override_study"`, the current v1 implementation derives:
- `execution_posture="replication_locked_plan"`
- route owner `replication`
- compatibility mirror `experiment_unit="replication_override"`
