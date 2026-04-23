# Layer 3 Training Audit

Date: 2026-04-24

Layer 3 is the forecast-generator layer. It consumes the feature matrix created
by Layer 2 and produces forecasts.

## Canonical Role

Layer 3 owns estimator and training protocol choices:

- model family;
- benchmark family;
- direct versus iterated forecast generation;
- forecast object, such as mean, median, or quantile;
- training window, minimum training size, training-start rule, and refit policy;
- model-order choices that are estimator behavior, such as AR lag selection;
- validation split, hyperparameter search, tuning objective, and tuning budget;
- model seed, early stopping, convergence handling, cache, checkpointing, and
  execution backend.

Layer 3 does not own the research feature representation grammar. It should
receive the Layer 2 representation payload, then fit and predict.

## Boundary With Layer 2

Canonical Layer 2 ownership now includes:

- `feature_builder`;
- `predictor_family`;
- `data_richness_mode`;
- `factor_count`;
- `feature_block_set`;
- `target_lag_block`;
- `x_lag_feature_block`;
- `factor_feature_block`;
- `level_feature_block`;
- `rotation_feature_block`;
- `temporal_feature_block`;
- `feature_block_combination`.

These axes decide how `H`, `X`, and target history become `Z`; they are not
model estimator choices.

Legacy runtime code still uses these names for executor dispatch. That is a
compatibility shape, not the canonical boundary. Future implementation should
split the current coarse names into explicit Layer 2 feature blocks, lower
generic `Z` unification into Layer 2, and leave Layer 3 with only
model/training execution.

The detailed Layer 2 x Layer 3 sweep contract is in
`layer2_layer3_sweep_contract.md`. That document is the operational reference
for freely sweeping research representations with forecast generators.

## Consumption Contract

Layer 3 should consume the output of Layer 2 through the unified Layer 2
representation interface:

```text
fit(model_family, Z_train, y_train, Z_pred, training_spec) -> y_pred
```

The unification itself is not a Layer 3 ownership item. Layer 3 may validate
that the selected forecast type and model family can consume the shape of `Z`,
but it must not decide how `Z` was built. The following are Layer 2 facts, not
Layer 3 facts:

- whether target lags are included;
- whether X lags are included;
- whether PCA/static factors are included;
- whether level add-backs are included;
- whether temporal or rotation blocks are included;
- whether X-side scaling, missing handling, outlier handling, or feature
  selection was applied.

Layer 3 owns how the estimator is fit on the supplied matrix:

- direct versus iterated forecast-generation logic;
- estimator family and estimator-specific hyperparameters;
- validation split and search;
- training window and refit policy;
- convergence and failure handling;
- model-order selection when the order is part of estimator behavior.

The important split is target-lag feature construction versus AR model-order
selection. `target_lag_block=fixed_target_lags` is a Layer 2 feature block. AR
BIC lag selection is a Layer 3 estimator behavior. A direct ridge model with
`target_lag_block=fixed_target_lags` is not an AR-BIC model; it is a direct
ridge model whose `Z` contains target-history columns.

## Layer 2 x Layer 3 Sweeps

The full grammar may sweep Layer 2 representation axes and Layer 3 training
axes in the same study. The sweep runner expands the Cartesian product, then
compiles each cell as a concrete recipe. This means all compatibility decisions
are made after expansion, when both the representation and the forecast
generator are known.

Layer 3 must therefore provide clear cell-level outcomes:

- `executable` when the selected model can consume the selected `Z`;
- `not_supported` when the required runtime composer or forecast generator is
  not implemented;
- `blocked_by_incompatibility` when the combination is semantically invalid,
  such as iterated raw-panel forecasting without an exogenous-X path contract.

The current important operational path is direct raw-panel forecasting over a
generic 2-D `Z`. This now includes fixed target lags concatenated with raw X,
fixed X lags, and static PCA factor scores. Autoregressive target-lag-only
forecasting remains a separate iterated path until Layer 2 finishes unifying
the representation handoff contract.

## Current Layer 3 Axes

The canonical Layer 3 registry surface is:

| Group | Axes |
|---|---|
| Forecast generator | `model_family`, `benchmark_family`, `forecast_type`, `forecast_object`, `horizon_modelization` |
| Training window | `min_train_size`, `training_start_rule`, `outer_window`, `refit_policy`, `lookback` |
| Model order | legacy `y_lag_count` for AR/model-order selection; target-lag feature construction is Layer 2 `target_lag_selection` / `target_lag_block` provenance |
| Validation/search | `validation_size_rule`, `validation_location`, `embargo_gap`, `split_family`, `shuffle_rule`, `alignment_fairness`, `search_algorithm`, `tuning_objective`, `tuning_budget`, `hp_space_style` |
| Runtime discipline | `seed_policy`, `early_stopping`, `convergence_handling`, `logging_level`, `checkpointing`, `cache_policy`, `execution_backend` |

## Compatibility Items

- `feature_builder`, `predictor_family`, `data_richness_mode`, and
  `factor_count`: canonical Layer 2 feature-representation axes. Legacy paths
  remain accepted as recipe/manifest compatibility and provenance, while
  runtime dispatch uses Layer 2 block-derived feature runtime where supported.
- `factor_ar_lags`: legacy runtime key remains accepted; target-lag feature
  count next to factor blocks is recorded as Layer 2 `target_lag_count`
  provenance, while model-specific lag-order selection remains Layer 3.
