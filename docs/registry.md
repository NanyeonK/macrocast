# Registry and grammar guide

## Purpose

macrocast separates two ideas explicitly:
- full choice-space representation
- current executable runtime support

The registry layer is the canonical home for that distinction.
It also now treats benchmark design as a first-class grammar concern rather than a hardcoded runtime detail.

## Canonical layer order

Every study path must follow this order:
- `0_meta`
- `1_data_task`
- `2_preprocessing`
- `3_training`
- `4_evaluation`
- `5_output_provenance`
- `6_stat_tests`
- `7_importance`

This order is fixed by `get_canonical_layer_order()`.

## Registry objects

The public registry surface exposes:
- `AxisRegistryEntry`
- `AxisSelection`
- `get_canonical_layer_order()`
- `get_axis_registry()`
- `get_axis_registry_entry()`
- `axis_governance_table()`

## Support-status split

Each enumerated option has one explicit support state:
- `operational`
- `registry_only`
- `planned`
- `external_plugin`
- `not_supported_yet`

This means macrocast can represent more choices than it can execute today.
That is intentional.

## Fixed / sweep / conditional semantics

The registry does not only list admissible values.
It also fixes the intended default policy per axis:
- `fixed`
- `sweep`
- `conditional`

This is how macrocast keeps hidden defaults out of benchmarking studies.

## Benchmark grammar

Benchmark design now has two layers:
- enum choice in path: `benchmark_family`
- numeric/free benchmark details in `leaf_config.benchmark_config`

Current benchmark-family vocabulary:
- `historical_mean`
- `ar_bic`
- `zero_change`
- `custom_benchmark`

Interpretation:
- path records the benchmark family as a discrete study-design choice
- `benchmark_config` records free parameters such as lag grid, minimum train size, or custom benchmark notes

This keeps benchmark design aligned with the package rule:
- enum choice in registry/path
- numeric or free design in leaf config

## Current v1 intent

The current runtime is intentionally narrower than the long-run registry:
- datasets are operational for `fred_md`, `fred_qd`, `fred_sd`
- revised information set is operational
- single-target point forecast is operational
- raw-only preprocessing contract is operational
- expanding-window AR engine is operational
- benchmark execution is operational for `historical_mean` and `ar_bic`
- custom benchmark families are already representable in grammar, but not executable in the current runtime slice

That distinction is part of the package grammar, not an informal note.
