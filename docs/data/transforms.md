# Transformations

macrocast implements the seven stationarity transformations defined in McCracken and Ng (2016, Table 1). These codes are embedded in the FRED-MD and FRED-QD CSV files (row 1) and are applied by `MacroFrame.transform()`.

---

## Transformation Codes

| tcode | Name | Formula | Typical use |
|-------|------|---------|-------------|
| 1 | Level | $x_t$ | Interest rates (already stationary) |
| 2 | First difference | $x_t - x_{t-1}$ | Employment in levels |
| 3 | Second difference | $\Delta^2 x_t$ | Rarely used; I(2) series |
| 4 | Log level | $\ln x_t$ | Ratios, index levels |
| 5 | Log first difference | $\Delta \ln x_t$ | Real activity (INDPRO, PAYEMS) |
| 6 | Log second difference | $\Delta^2 \ln x_t$ | Rarely used |
| 7 | Change in ratio | $\Delta(x_t / x_{t-1} - 1)$ | Spreads, percentage changes |

The `TransformCode` enum provides named constants:

```python
from macrocast.data.transforms import TransformCode

TransformCode.LEVEL        # 1
TransformCode.DIFF         # 2
TransformCode.DIFF2        # 3
TransformCode.LOG          # 4
TransformCode.LOG_DIFF     # 5
TransformCode.LOG_DIFF2    # 6
TransformCode.DELTA_RATIO  # 7
```

---

## Using Transformations

### Via MacroFrame (recommended)

```python
import macrocast as mc

md = mc.load_fred_md()

# Apply default tcodes from the FRED-MD spec
md_t = md.transform()

# Override specific variables
md_t = md.transform(override={"INDPRO": 5, "UNRATE": 2})
```

### Via load functions

```python
md_t = mc.load_fred_md(
    transform=True,
    tcode_override={"INDPRO": 5},
)
```

### Low-level functions

```python
import pandas as pd
from macrocast.data.transforms import apply_tcode, apply_tcodes

# Single series
series = pd.Series([100.0, 102.0, 101.5, 103.0])
log_diff = apply_tcode(series, 5)   # Δln(x_t)

# Full DataFrame
tcodes = {"INDPRO": 5, "UNRATE": 2, "FEDFUNDS": 1}
df_t = apply_tcodes(df, tcodes)
```

---

## Notes on Leading NaN Values

Differencing operations introduce leading NaN values:

- tcode 2 (DIFF): 1 leading NaN
- tcode 3 (DIFF2): 2 leading NaNs
- tcode 5 (LOG\_DIFF): 1 leading NaN
- tcode 6 (LOG\_DIFF2): 2 leading NaNs
- tcode 7 (DELTA\_RATIO): 2 leading NaNs

These are preserved in the output. To remove them, call `handle_missing("trim_start")` after transformation:

```python
md_t = md.transform().handle_missing("trim_start")
```

---

## Non-Positive Values

For log-based transformations (tcode 4, 5, 6), non-positive values are set to NaN and a warning is issued. This can occur in series such as interest rate spreads that cross zero.

---

## Inverse Transformation

`inverse_tcode` is not implemented in v0.1.

!!! note "Not yet implemented in v0.1"
    `macrocast.data.transforms.inverse_tcode` raises `NotImplementedError`. Inversion (reconstructing levels from differences) is deferred to v0.2.

---

## References

McCracken, M.W. and Ng, S. (2016). "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business and Economic Statistics*, 34(4), 574-589. Table 1 defines the seven transformation codes.
