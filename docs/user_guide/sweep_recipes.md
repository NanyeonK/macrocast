# Sweep Recipe Grammar

A sweep recipe is a normal macrocast recipe that declares at least one
`sweep_axes` entry on one or more layers. The compiler
(`compile_sweep_plan`) expands those entries into a Cartesian product of
fully-specified variant recipes, each compilable by
`compile_recipe_dict` as a single-path recipe.

## Per-layer axes

Every layer under `path` (0_meta, 1_data_task, ..., 7_importance) can
hold three kinds of entries:

| Key | Meaning |
|---|---|
| `fixed_axes` | Values held constant across all variants |
| `sweep_axes` | Values expanded Cartesian-wise; **must be a list** |
| `leaf_config` | Non-axis details (target, horizons, benchmark_config) |

## Validation rules

- Declaring the same axis in both `fixed_axes` and `sweep_axes` on the
  same layer raises `SweepPlanError`.
- An empty sweep list (`sweep_axes: {model_family: []}`) raises
  `SweepPlanError`.
- A recipe with no `sweep_axes` at all is not a sweep recipe — use
  `compile_recipe_dict` instead.
- The Cartesian size must not exceed `max_variants` (default 1000). Pass
  a larger cap explicitly if you know what you are doing.

## Variant identity

Each variant receives two stable identifiers derived from canonical JSON
hashes:

- `variant_id` — `v-<8-hex>` from the variant's axis values; the same
  sweep combination always produces the same id.
- `study_id` — `sha256-<16-hex>` from the parent recipe id + sorted
  axes + variant axis values.

These flow into every variant's provenance payload and into the
`study_manifest.json`. Re-running a plan with the same inputs produces
the same ids, which makes replication and downstream joins deterministic.

## Example: two axes

```yaml
2_preprocessing:
  fixed_axes:
    # ... 13 other preprocessing axes
  sweep_axes:
    scaling_policy: [none, standard]
3_training:
  fixed_axes:
    framework: rolling
    benchmark_family: zero_change
    feature_builder: raw_feature_panel
  sweep_axes:
    model_family: [ridge, lasso, elasticnet]
```

This produces 2 * 3 = 6 variants. Every variant fixes a specific
(`scaling_policy`, `model_family`) pair and inherits everything else
from `fixed_axes`.

## Out of scope in v0.3

The following features are reserved for later phases:

- `conditional_axes` — axis A only swept when axis B equals X (Phase 10)
- `derived_axes` — axis value computed from other axes (Phase 10)
- Parallel variant execution — serial only in v0.3, opt-in in v1.1
  (ADR-003)
- Multi-target joint sweeps (Phase 5a)
