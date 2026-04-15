# macrocast

> Given a standardized macro dataset adapter and a fixed forecasting recipe, compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol.

macrocast is being rebuilt as an architecture-first forecasting package.
The package goal is not to make every long-run choice executable immediately.
The goal is to make the full research choice space explicit in package grammar while systematically promoting registry-defined choices into operational support.

## Rebuild status

The rebuilt package currently has seven public layers/surfaces wired in order:
- `macrocast.stage0`
- `macrocast.raw`
- `macrocast.recipes`
- `macrocast.preprocessing`
- `macrocast.registry`
- `macrocast.compiler`
- `macrocast.execution`

Current operational subset
- operational frameworks: `expanding`, `rolling`
- operational benchmark families: `historical_mean`, `zero_change`, `ar_bic`, `custom_benchmark`
- operational model families: `ar`, `ridge`, `lasso`, `elasticnet`, `randomforest`
- operational feature builders: `autoreg_lagged_target`, `raw_feature_panel`
- operational preprocessing paths:
  - explicit `raw_only`
  - train-only raw-panel path with `x_missing_policy=em_impute` and `scaling_policy=standard`
  - train-only raw-panel path with `x_missing_policy=em_impute` and `scaling_policy=robust`
- operational statistical tests:
  - `dm`
  - `cw`
- operational importance methods:
  - `minimal_importance`
  - current supported routes: `ridge`, `lasso`, `randomforest` on `raw_feature_panel`

Current roadmap focus
- post-wrapper provenance slice now records deterministic `tree_context` payloads in compile/run artifacts so fixed-vs-sweep semantics remain explicit.
- next major widening target after that is wizard/runtime UX around tree-path selection rather than another hidden execution fallback.
