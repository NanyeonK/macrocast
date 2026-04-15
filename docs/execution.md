# Execution pipeline

## Purpose

The execution layer consumes current package contracts and emits deterministic run artifacts.
It sits behind an explicit compiler boundary, preserves preprocessing semantics in output provenance, and treats model execution and benchmark execution as separate runtime components.

## Current role

The current runtime now supports a first importance layer in addition to frameworks, preprocessing, DM testing, CW testing, a plugin-ready custom benchmark bridge, a baseline comparison summary artifact, a first explicit-vintage real-time slice, and a first narrow multi-target slice.
It executes a benchmark-respecting slice with:
- revised-data or explicit-vintage real-time single-target point forecast
- first narrow multi-target point-forecast route with one shared study environment across explicit targets
- explicit benchmark family from recipe grammar
- deterministic prediction, metric, and comparison artifacts
- two operational feature-builder families
- one narrow family of train-only raw-panel preprocessing paths
- operational statistical tests: DM and CW
- first operational custom benchmark bridge via local Python callable loading
- first operational importance layer: minimal importance

## Current operational frameworks

- `expanding`
- `rolling`

## Current operational statistical tests

- `stat_test = dm`
- `stat_test = cw`
- baseline comparison artifact is always written as `comparison_summary.json`
- DM writes `stat_test_dm.json`
- CW writes `stat_test_cw.json`
- CW first slice uses the benchmark-vs-model forecast-gap adjustment on the existing prediction table and reports a simple normal-approximation statistic
- manifest records `comparison_file`, `stat_test_spec`, and `stat_test_file`

## Current operational importance layer

The current runtime can execute:
- `importance_method = minimal_importance`

Current behavior:
- writes `importance_minimal.json`
- manifest records `importance_spec` and `importance_file`
- currently implemented for:
  - non-AR linear routes: `ridge`, `lasso`
  - tree route: `randomforest`
- current minimal importance requires `feature_builder='raw_feature_panel'`
- unsupported importance requests fail explicitly at runtime

Current implementation semantics:
- ridge/lasso importance = absolute coefficient magnitude from the final fitted training window
- randomforest importance = `feature_importances_` from the final fitted training window

## Current operational feature builders

- `autoreg_lagged_target`
- `raw_feature_panel`

## Current operational preprocessing paths

- explicit `raw_only`
- train-only raw-panel extra-preprocess path:
  - `tcode_policy = extra_preprocess_without_tcode`
  - `x_missing_policy = em_impute`
  - `scaling_policy = standard` or `robust`
  - `preprocess_order = extra_only`
  - `preprocess_fit_scope = train_only`

## Current model executors

- `ar`
- `ridge`
- `lasso`
- `elasticnet`
- `randomforest`

## Current benchmark executors

- `historical_mean`
- `zero_change`
- `ar_bic`
- `custom_benchmark`

Current custom benchmark bridge contract:
- benchmark family stays `custom_benchmark` in grammar/provenance
- runtime loads a local Python file from `benchmark_config.plugin_path`
- runtime resolves `benchmark_config.callable_name`
- callable signature is `custom_benchmark(train, horizon, benchmark_config) -> float`
- callable must return one numeric forecast

## Provenance behavior

The manifest preserves:
- `preprocess_summary`
- full `preprocess_contract`
- `execution_architecture`
- full `model_spec`
- full `benchmark_spec`
- `target` for single-target runs
- `targets` for multi-target runs
- `comparison_file`
- `stat_test_spec`
- `importance_spec`
- optional compiler provenance payload
- top-level `tree_context` payload when compiler provenance is passed through execution
- summary text can include a compact `tree_context=` line for fixed-vs-sweep route inspection

## Current limitation

Even though the execution surface is broader than before, the current runtime still has explicit boundaries:
- only `minimal_importance` is operational in the importance layer
- current importance support is intentionally limited to `ridge`, `lasso`, and `randomforest` on the raw-panel path
- `custom_benchmark` currently uses only the first plugin-ready local Python bridge, not a broader package/plugin registry
- `real_time` currently means one explicit vintage per run, not a rolling historical real-time evaluation engine
- multi-target execution currently means one shared model/benchmark/preprocess environment across explicit targets, not a target-specific orchestration framework
- SHAP remains future work
