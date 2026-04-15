# Preprocessing contract

## Purpose

macrocast no longer treats preprocessing as one hidden default pipeline.
The package now separates:
- representation / transform choices
- extra preprocessing choices
- execution semantics

## Current public surface

The preprocessing layer now exposes:
- `PreprocessContract`
- `build_preprocess_contract()`
- `check_preprocess_governance()`
- `is_operational_preprocess_contract()`
- `preprocess_summary()`
- `preprocess_to_dict()`

## Representation / transform axes

The contract records:
- `target_transform_policy`
- `x_transform_policy`
- `tcode_policy`

Current t-code policy vocabulary:
- `raw_only`
- `tcode_only`
- `tcode_then_extra_preprocess`
- `extra_preprocess_without_tcode`
- `extra_then_tcode`
- `custom_transform_pipeline`

These are intentionally distinct design choices.
macrocast does not treat them as the same path.

## Extra preprocessing axes

The contract also records:
- `target_missing_policy`
- `x_missing_policy`
- `target_outlier_policy`
- `x_outlier_policy`
- `scaling_policy`
- `dimensionality_reduction_policy`
- `feature_selection_policy`

## Execution semantics

The contract records:
- `preprocess_order`
- `preprocess_fit_scope`
- `inverse_transform_policy`
- `evaluation_scale`

This is the minimum needed to keep model effects separate from preprocessing effects in later benchmarking interpretation.

## Current executable subset

The current runtime is intentionally honest and narrow.
Only the explicit raw-only contract is operational today:
- raw target representation
- raw x representation
- no t-code transform
- no extra preprocessing
- no inverse transform
- raw-level evaluation scale

Other preprocessing choices are already representable in package grammar, but not yet executable.
That distinction is explicit through registry/compiler status rather than hidden behavior.
