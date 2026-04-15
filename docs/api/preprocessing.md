# Preprocessing API Reference

## Import surface

```python
from macrocast import (
    PreprocessContract,
    build_preprocess_contract,
    check_preprocess_governance,
    is_operational_preprocess_contract,
    preprocess_summary,
    preprocess_to_dict,
)
```

## `PreprocessContract`

Fields:
- `target_transform_policy`
- `x_transform_policy`
- `tcode_policy`
- `target_missing_policy`
- `x_missing_policy`
- `target_outlier_policy`
- `x_outlier_policy`
- `scaling_policy`
- `dimensionality_reduction_policy`
- `feature_selection_policy`
- `preprocess_order`
- `preprocess_fit_scope`
- `inverse_transform_policy`
- `evaluation_scale`

## Functions

### `build_preprocess_contract()`
Construct a fully explicit preprocessing contract.

### `check_preprocess_governance()`
Apply governance rules around representation choice, extra preprocessing, fit scope, and fixed-vs-sweep semantics.

### `is_operational_preprocess_contract()`
Return whether the contract is runnable in the current execution slice.

### `preprocess_to_dict()` / `preprocess_summary()`
Serialize the contract for manifests and inspection.

## Notes

The preprocessing layer is now designed so manifests can preserve:
- whether t-code was applied
- whether extra preprocessing was applied
- exact order
- fit scope
- evaluation scale
