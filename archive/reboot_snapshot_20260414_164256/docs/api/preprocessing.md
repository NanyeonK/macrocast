# macrocast.preprocessing

`macrocast.preprocessing` contains the public API for transforms, missing-value handling, panel transforms, and preprocessing registries.

## Import

```python
from macrocast.preprocessing import apply_tcodes, handle_missing, PanelTransformer
```

## Transform code functions

Transform exports:
- `TransformCode`
- `apply_tcode`
- `apply_tcodes`
- `inverse_tcode`
- `apply_marx`
- `apply_maf`
- `apply_x_factors`
- `apply_pca`
- `apply_hamilton_filter`

Purpose:
- convert raw macro series into transformed modeling inputs

## Missing and outlier handling

Missing-data exports:
- `detect_missing_type`
- `classify_missing`
- `handle_missing`
- `remove_outliers_iqr`
- `prepare_fredmd`
- `em_factor`

Purpose:
- diagnose and repair missingness / outlier problems before feature construction

## Panel transforms

Panel-transform exports:
- `PanelTransformer`
- `WinsorizeTransform`
- `DemeanTransform`
- `HPFilterTransform`
- `StandardizeTransform`
- `CustomTransform`
- `DropTransform`

Purpose:
- build reusable preprocessing pipelines for target and predictor panels

## Registry helpers

Preprocessing-registry exports:
- `get_target_recipe`
- `get_x_recipe`
- `load_preprocessing_registry`
- `validate_preprocessing_registry`

Purpose:
- expose canonical preprocessing recipes and validation rules

## Related pages

- `User Guide > Stage 2`
- `User Guide > Stage 3`
