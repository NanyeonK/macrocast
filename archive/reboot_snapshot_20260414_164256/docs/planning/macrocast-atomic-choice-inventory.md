# macrocast atomic choice inventory

Source of record:
- `archive/legacy-plans/source/plan_2026_04_09_2358.md`

Interpretation rule:
- main_stage = top-level layer for user wizard
- substage = grouped section inside a main stage
- atomic_choice = smallest user-selectable choice unit
- wizard_status = one of implemented | partial | not_started
- docs_status = one of documented | partial | missing

## Stage 0 — Meta Layer

### Substage 0.1 experiment unit
- 0.1 experiment_unit
  - yaml_target: primary = `meta.experiment_unit`; deprecated fallback = `taxonomy_path.experiment_unit`
  - wizard_status: partial
  - docs_status: documented
  - note: current package now treats this as a routing decision, not a normal taxonomy leaf; only single-target family belongs under `macrocast_single_run()`

### Substage 0.2 axis type
- 0.2 axis_type
  - yaml_target: meta.axis_type
  - wizard_status: not_started
  - docs_status: missing

### Substage 0.3 registry type
- 0.3 registry_type
  - yaml_target: meta.registry_type
  - wizard_status: not_started
  - docs_status: missing

### Substage 0.4 reproducibility mode
- 0.4 reproducibility_mode
  - yaml_target: meta.reproducibility_mode
  - wizard_status: partial
  - docs_status: partial

### Substage 0.5 failure policy
- 0.5 failure_policy
  - yaml_target: meta.failure_policy
  - wizard_status: partial
  - docs_status: partial

### Substage 0.6 compute mode
- 0.6 compute_mode
  - yaml_target: meta.compute_mode
  - wizard_status: partial
  - docs_status: partial

## Stage 1 — Data / Task Definition Layer

### Substage 1.1 data definition
- 1.1.1 data_domain
- 1.1.2 dataset_source
- 1.1.3 frequency
- 1.1.4 information_set_type
- 1.1.5 vintage_policy
- 1.1.6 alignment_rule
- 1.1.7 release_lag_rule
- 1.1.8 missing_availability_or_ragged_edge
- 1.1.9 variable_universe
  - wizard_status: partial
  - docs_status: partial

### Substage 1.2 period definition
- 1.2.1 full_sample_period
- 1.2.2 training_start_rule
- 1.2.3 oos_period
- 1.2.4 minimum_train_size
- 1.2.5 warm_up_rule
- 1.2.6 structural_break_segmentation
  - wizard_status: partial
  - docs_status: missing

### Substage 1.3 horizon definition
- 1.3.1 horizon_list
- 1.3.2 forecast_type
- 1.3.3 forecast_object
- 1.3.4 horizon_target_construction
- 1.3.5 overlap_handling
  - wizard_status: partial
  - docs_status: missing

### Substage 1.4 target and predictor definition
- 1.4.1 target_family
- 1.4.2 predictor_family
- 1.4.3 contemporaneous_X_rule
- 1.4.4 own_target_lags
- 1.4.5 deterministic_components
- 1.4.6 exogenous_block
  - wizard_status: partial
  - docs_status: missing

### Substage 1.5 multi-target mapping
- 1.5.1 X_map_policy
- 1.5.2 target_to_target_inclusion
- 1.5.3 multi_target_architecture
  - wizard_status: not_started
  - docs_status: missing
  - note: wrapper/orchestrator scope after single-run path is stable

### Substage 1.6 extra data/task layer
- 1.6.1 scale_at_evaluation
- 1.6.2 benchmark_family_task_level
- 1.6.3 regime_or_conditional_task
  - wizard_status: not_started
  - docs_status: missing

## Stage 2 — Preprocessing Layer
- 2.0.1 separation_rule
- 2.0.2 preprocessing_fit_scope
- 2.1.1 target_missing
- 2.1.2 target_outlier
- 2.1.3 target_definition_transform
- 2.1.4 transform_timing
- 2.1.5 inverse_transform
- 2.1.6 target_normalization
- 2.1.7 target_domain_restriction
- 2.1.8 target_class_handling
- 2.2.1 X_missing
- 2.2.2 X_outlier
- 2.2.3 standardize_scale
- 2.2.4 scaling_scope
- 2.2.5 additional_preprocessing
- 2.2.6 lag_creation
- 2.2.7 dimensionality_reduction
- 2.2.8 feature_selection
- 2.2.9 feature_grouping
- 2.3.1 execution_order
- 2.3.2 recipe_mode

## Stage 3 — Forecasting / Training Layer
- 3.1.1 outer_window
- 3.1.2 refit_policy
- 3.1.3 data_rich_vs_data_poor_mode
- 3.1.4 sequence_framework
- 3.1.5 horizon_modelization
- 3.2.1 validation_size_rule
- 3.2.2 validation_location
- 3.2.3 embargo_gap
- 3.3.1 split_family
- 3.3.2 shuffle_rule
- 3.3.3 alignment_fairness
- 3.4.1 naive_benchmark_models
- 3.4.2 linear_ml_regularized_models
- 3.4.3 kernel_margin_models
- 3.4.4 tree_ensemble_models
- 3.4.5 neural_models
- 3.4.6 panel_spatial_hierarchical_models
- 3.4.7 probabilistic_quantile_models
- 3.4.8 custom_plugin_models
- 3.5.1 search_algorithm
- 3.5.2 tuning_objective
- 3.5.3 tuning_budget
- 3.5.4 hp_space_style
- 3.5.5 seed_policy
- 3.5.6 early_stopping
- 3.5.7 convergence_handling
- 3.6.1 feature_builder_type
- 3.6.2 y_lag_count
- 3.6.3 factor_count
- 3.6.4 lookback
- 3.7.1 logging_level
- 3.7.2 checkpointing
- 3.7.3 cache
- 3.7.4 execution_backend

## Stage 4 — Evaluation Layer
- 4.1.1 point_forecast_metrics
- 4.1.2 relative_metrics
- 4.1.3 direction_event_metrics
- 4.1.4 quantile_interval_density_metrics
- 4.1.5 economic_decision_metrics
- 4.2.1 benchmark_model
- 4.2.2 benchmark_estimation_window
- 4.2.3 benchmark_by_target_horizon
- 4.3.1 aggregation_over_time
- 4.3.2 aggregation_over_horizons
- 4.3.3 aggregation_over_targets
- 4.3.4 ranking_rule
- 4.3.5 report_style
- 4.4.1 regime_definition
- 4.4.2 regime_use
- 4.4.3 regime_metrics
- 4.5.1 decomposition_target
- 4.5.2 decomposition_order

## Stage 5 — Output / Provenance Layer
- 5.1 saved_objects
- 5.2 provenance_fields
- 5.3 export_format
- 5.4 artifact_granularity

## Stage 6 — Statistical Test Layer
- 6.1 equal_predictive_ability_tests
- 6.2 nested_model_tests
- 6.3 conditional_predictive_ability_instability
- 6.4 multiple_model_data_snooping_tests
- 6.5 density_interval_forecast_tests
- 6.6 direction_classification_tests
- 6.7 residual_calibration_diagnostics
- 6.8 dependence_correction
- 6.9 test_scope

## Stage 7 — Variable Importance / Interpretability Layer
- 7.1 importance_scope
- 7.2 model_native_importance
- 7.3 model_agnostic_importance
- 7.4 shap_family
- 7.5 gradient_path_methods
- 7.6 local_surrogate_perturbation
- 7.7 partial_dependence_style
- 7.8 grouped_importance
- 7.9 sequence_temporal_importance
- 7.10 stability_of_importance
- 7.11 importance_aggregation
- 7.12 output_style

## Summary
- main_stages: 8
- atomic_choices: 136 approx
- current wizard covers only a small subset and still lacks branch-aware downstream traversal
- next requirement is a docs-linked hierarchical wizard specification with explicit single-run vs wrapper ownership
