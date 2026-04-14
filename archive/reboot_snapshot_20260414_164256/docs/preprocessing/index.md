# Preprocessing Layer

The preprocessing layer (`macrocast.preprocessing`) handles stationarity transformations,
missing value treatment, and panel construction from raw FRED data. It provides the
building blocks that `MacroFrame.transform()` and `FeatureBuilder` rely on internally.

## Modules

| Module | Purpose |
|--------|---------|
| `transforms` | Seven McCracken-Ng (2016) transformation codes; MARX, MAF, X-factors, PCA, Hamilton filter |
| `missing` | Missing value classification, interpolation, and trimming |
| `panel` | Panel alignment, frequency conversion, and stacking utilities |

## Quick Start

```python
from macrocast.preprocessing.transforms import apply_tcodes, TransformCode
from macrocast.preprocessing.missing import handle_missing

# Apply transformation codes from FRED-MD header row
df_stationary = apply_tcodes(df_raw, tcode_series)

# Handle missing values with default strategy
df_clean = handle_missing(df_stationary)
```
