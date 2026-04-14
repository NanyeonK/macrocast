# Stage 0 redesign schema

Goal:
Replace the flat `experiment_unit`-first framing with a dataset-aware, horse-race-first Stage 0 design.

Core package assumption:
- ordinary own analysis is the implicit default
- replication is an explicit exceptional path
- benchmark comparison is part of the normal horse-race flow
- ablation and robustness are variation structures on top of a baseline design, not top-level intents

## Stage 0 redesign overview

Recommended Stage 0 order:
1. `replication_input`
2. `dataset_spec`
3. `derived_design_shape`
4. `fixed_design`
5. `varying_design`
6. `comparison_contract`
7. `execution_posture`

This means Stage 0 no longer begins by asking for one flat `experiment_unit` label.
Instead, it begins by determining:
- whether this is a replication override path
- what dataset family implies about frequency and structural complexity
- what stays fixed
- what is allowed to vary
- how the horse race is defined and reported

## 0.1 replication_input

Purpose:
- detect whether the user is entering through a locked replication path
- if not supplied, proceed automatically as ordinary own-analysis design construction

Fields:
- `replication_recipe_path: str | null`
- `replication_bundle_id: str | null`
- `replication_preset_id: str | null`

Rule:
- all null -> ordinary own-analysis flow
- any non-null -> replication flow

## 0.2 dataset_spec

Purpose:
- capture the raw dataset selection before later design choices
- derive family and implied frequency rules from the selected dataset set

Fields:
- `datasets: list[str]`
- `dataset_family: str`  # derived, not manually chosen in the standard path
- `derived_frequency: str`
- `requires_bundle_mode: bool`
- `requires_mixed_frequency_handling: bool`
- `requires_multi_target_handling: bool`

### Recommended dataset-family derivation

#### Case A: FRED-MD
Input:
```yaml
datasets:
  - fred_md
```

Derived:
```yaml
dataset_family: single_frequency_macro
derived_frequency: monthly
requires_bundle_mode: false
requires_mixed_frequency_handling: false
requires_multi_target_handling: false
```

#### Case B: FRED-QD
Input:
```yaml
datasets:
  - fred_qd
```

Derived:
```yaml
dataset_family: single_frequency_macro
derived_frequency: quarterly
requires_bundle_mode: false
requires_mixed_frequency_handling: false
requires_multi_target_handling: false
```

#### Case C: FRED-SD alone
Input:
```yaml
datasets:
  - fred_sd
```

Derived:
```yaml
dataset_family: mixed_frequency_state_macro
derived_frequency: mixed_frequency
requires_bundle_mode: false
requires_mixed_frequency_handling: true
requires_multi_target_handling: true
```

#### Case D: FRED-SD combined with MD or QD
Input:
```yaml
datasets:
  - fred_md
  - fred_sd
```

Derived:
```yaml
dataset_family: mixed_source_bundle
derived_frequency: mixed_frequency
requires_bundle_mode: true
requires_mixed_frequency_handling: true
requires_multi_target_handling: true
```

### Why dataset comes this early

For current package scope:
- `fred_md` effectively implies monthly work
- `fred_qd` effectively implies quarterly work
- `fred_sd` is mixed-frequency by construction
- `fred_sd` combined with other datasets pushes the design toward bundle or multi-target handling

So dataset choice already constrains part of the later design and should not be treated as just another late empirical detail.

## 0.3 derived_design_shape

Purpose:
- summarize the high-level structural consequences of replication and dataset choice before detailed fixed/varying selections begin

Fields:
- `analysis_path: str`  # ordinary_analysis | replication
- `structure_class: str`  # one_path | mixed_frequency_path | bundle_path
- `target_mode: str`  # single_target | multi_target | state_panel_like
- `frequency_mode: str`  # monthly | quarterly | mixed_frequency

Interpretation examples:
- `fred_md` + no replication -> `ordinary_analysis`, `one_path`, `single_target`, `monthly`
- `fred_qd` + no replication -> `ordinary_analysis`, `one_path`, `single_target`, `quarterly`
- `fred_sd` alone -> `ordinary_analysis`, `mixed_frequency_path`, `state_panel_like`, `mixed_frequency`
- any replication input -> `replication`, structure depends on the locked bundle/preset

## 0.4 fixed_design

Purpose:
- define the baseline design spine that stays fixed across the horse race unless later variation explicitly changes it

Recommended fields:
- `target_spec`
- `predictor_map`
- `sample_spec`
- `benchmark_family`
- `evaluation_scale`
- `forecast_framework`
- `validation_design`
- `preprocessing_baseline`

Example:
```yaml
fixed_design:
  target_spec:
    target: INDPRO
    horizon_grid: [1, 3, 12]
  predictor_map:
    x_map: all_minus_target
  sample_spec:
    sample: full_sample
    oos: short_smoke_oos
  benchmark_family: ar
  evaluation_scale: point_forecast
  forecast_framework:
    outer_window: expanding
    refit_policy: recursive
  validation_design:
    validation: last_block
  preprocessing_baseline:
    target_prep: basic_none
    x_prep: basic_none
    features: factors_x
```

## 0.5 varying_design

Purpose:
- define which dimensions are intentionally varied in the horse race, robustness analysis, or ablation study

Primary field:
- `variation_mode`

Recommended values:
- `none`
- `model_horserace`
- `design_robustness`
- `ablation`
- `multi_target_expansion`
- `combined`

### model_horserace
Use when the main comparison is across model families under a fixed design.

Fields:
- `model_candidates`
- `tuning_candidates`
- `per_model_overrides`

### design_robustness
Use when the main comparison is across specification variants.

Fields:
- `robustness_axes`
- `grid_type`  # one_at_a_time | restricted | full_cross

### ablation
Use when the main comparison is baseline versus controlled variants.

Fields:
- `baseline_label`
- `ablation_axes`
- `ablation_rule`  # drop_one | toggle_block | replace_component

### multi_target_expansion
Use when the fixed baseline design is applied to a target set.

Fields:
- `target_list`
- `target_specific_overrides`
- `shared_design_only`

## 0.6 comparison_contract

Purpose:
- define how results are compared and reported

Fields:
- `benchmark_candidates`
- `primary_benchmark`
- `comparison_unit`
- `statistical_test_plan`
- `report_focus`

Example:
```yaml
comparison_contract:
  benchmark_candidates:
    - ar
  primary_benchmark: ar
  comparison_unit: model
  statistical_test_plan:
    - dm
    - cw
    - mcs
  report_focus:
    - relative_msfe
    - oos_r2
    - regime_eval
```

Interpretation:
- benchmark comparison is normal, not exceptional
- the package should assume some comparison contract exists in ordinary own-analysis mode

## 0.7 execution_posture

Purpose:
- define reproducibility, failure behavior, and compute posture

Fields:
- `reproducibility_mode`
- `failure_policy`
- `compute_mode`

Example:
```yaml
execution_posture:
  reproducibility_mode: strict_reproducible
  failure_policy: fail_fast
  compute_mode: serial
```

## Full YAML skeleton

```yaml
meta:
  replication_input:
    replication_recipe_path: null
    replication_bundle_id: null
    replication_preset_id: null

  dataset_spec:
    datasets:
      - fred_md
    dataset_family: single_frequency_macro
    derived_frequency: monthly
    requires_bundle_mode: false
    requires_mixed_frequency_handling: false
    requires_multi_target_handling: false

  derived_design_shape:
    analysis_path: ordinary_analysis
    structure_class: one_path
    target_mode: single_target
    frequency_mode: monthly

  fixed_design:
    target_spec:
      target: INDPRO
      horizon_grid: [1, 3, 12]
    predictor_map:
      x_map: all_minus_target
    sample_spec:
      sample: full_sample
      oos: short_smoke_oos
    benchmark_family: ar
    evaluation_scale: point_forecast
    forecast_framework:
      outer_window: expanding
      refit_policy: recursive
    validation_design:
      validation: last_block
    preprocessing_baseline:
      target_prep: basic_none
      x_prep: basic_none
      features: factors_x

  varying_design:
    variation_mode: model_horserace
    model_horserace:
      model_candidates:
        - ar
        - random_forest
        - elastic_net
        - factors
      tuning_candidates:
        - grid_search
      per_model_overrides: {}
    design_robustness: {}
    ablation: {}
    multi_target_expansion: {}

  comparison_contract:
    benchmark_candidates:
      - ar
    primary_benchmark: ar
    comparison_unit: model
    statistical_test_plan:
      - dm
      - cw
      - mcs
    report_focus:
      - relative_msfe
      - oos_r2

  execution_posture:
    reproducibility_mode: strict_reproducible
    failure_policy: fail_fast
    compute_mode: serial
```

## Proposed function family

### 1. replication detection
```python
def resolve_replication_input(
    *,
    replication_recipe_path: str | None = None,
    replication_bundle_id: str | None = None,
    replication_preset_id: str | None = None,
) -> dict[str, Any]:
    ...
```

Output:
- `is_replication`
- `source_type`
- `source_ref`
- `message`

### 2. dataset-family derivation
```python
def derive_dataset_spec(
    datasets: list[str],
) -> dict[str, Any]:
    ...
```

Output:
- `dataset_family`
- `derived_frequency`
- `requires_bundle_mode`
- `requires_mixed_frequency_handling`
- `requires_multi_target_handling`

### 3. design-shape derivation
```python
def derive_design_shape(
    *,
    replication_decision: dict[str, Any],
    dataset_spec: dict[str, Any],
) -> dict[str, Any]:
    ...
```

Output:
- `analysis_path`
- `structure_class`
- `target_mode`
- `frequency_mode`

### 4. fixed-design builder
```python
def build_fixed_design(
    *,
    selections: dict[str, Any],
) -> dict[str, Any]:
    ...
```

### 5. varying-design builder
```python
def build_varying_design(
    *,
    variation_mode: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    ...
```

### 6. comparison-contract builder
```python
def build_comparison_contract(
    *,
    benchmark_candidates: list[str],
    primary_benchmark: str,
    comparison_unit: str,
    statistical_test_plan: list[str],
    report_focus: list[str],
) -> dict[str, Any]:
    ...
```

### 7. execution-posture builder
```python
def build_execution_posture(
    *,
    reproducibility_mode: str,
    failure_policy: str,
    compute_mode: str,
) -> dict[str, Any]:
    ...
```

### 8. top-level Stage 0 compiler
```python
def build_stage0_design_frame(
    *,
    replication_input: dict[str, Any],
    dataset_spec: dict[str, Any],
    derived_design_shape: dict[str, Any],
    fixed_design: dict[str, Any],
    varying_design: dict[str, Any],
    comparison_contract: dict[str, Any],
    execution_posture: dict[str, Any],
) -> dict[str, Any]:
    ...
```

## Practical consequence

Under this redesign:
- ordinary own analysis is the default
- benchmark comparison is assumed as part of normal horse-race design
- ablation and robustness are chosen through `varying_design`
- replication is the explicit exceptional path
- dataset choice becomes an early structural decision because it already determines frequency and possibly bundle/multi-target handling
