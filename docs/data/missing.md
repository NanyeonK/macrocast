# Missing Values

FRED-MD and FRED-QD panels contain three structurally distinct types of missing observations. macrocast distinguishes among them and provides targeted treatment methods for each.

---

## Missing Value Types

| Type | Definition | Common cause |
|------|-----------|--------------|
| **Leading** | NaN at the start of a series | Series not yet published at the sample start date |
| **Trailing** | NaN at the end of a series | Series discontinued or publication lag |
| **Intermittent** | NaN in the interior of a series | Data revisions, survey gaps, outlier removal |

The `classify_missing` function reports all three counts for each variable:

```python
import macrocast as mc

md = mc.load_fred_md()
report = md.missing_report()
# Returns a DataFrame with columns:
# n_total, n_leading, n_trailing, n_intermittent, pct_missing,
# first_valid_idx, last_valid_idx

# Sort by total missing fraction
print(report.sort_values("pct_missing", ascending=False).head(10))
```

---

## Treatment Methods

`handle_missing` dispatches to one of five methods:

### `"trim_start"` (recommended for FRED-MD/QD)

Advances the sample start to the latest `first_valid_index` across all columns, ensuring no series has leading NaN values.

```python
md_clean = md.handle_missing("trim_start")
```

This is the standard approach in the FRED-MD literature. It preserves the full cross-section at the cost of a shorter sample.

### `"drop_vars"`

Drops variables whose missing fraction exceeds a threshold (default 50%).

```python
# Drop variables with more than 30% missing
md_clean = md.handle_missing("drop_vars", max_missing_pct=0.3)
```

### `"interpolate"`

Linearly interpolates only intermittent (interior) NaN gaps. Leading and trailing NaNs are left intact.

```python
md_interp = md.handle_missing("interpolate")
```

### `"forward_fill"`

Last-observation-carried-forward (LOCF) for all NaN cells.

```python
md_ff = md.handle_missing("forward_fill")
```

### `"em"` (not yet implemented)

!!! note "Not yet implemented in v0.1"
    EM-based imputation (`method="em"`) raises `NotImplementedError`. This method is deferred to v0.2.

---

## Via MacroFrame

All treatment methods are accessible through `MacroFrame.handle_missing()`, which returns a new MacroFrame:

```python
md = mc.load_fred_md()
md_ready = (
    md
    .trim(start="1970-01", end="2023-12")
    .handle_missing("trim_start")
    .transform()
)
```

---

## Low-Level Functions

```python
from macrocast.data.missing import classify_missing, handle_missing

# Report
report = classify_missing(df)

# Treat
df_clean = handle_missing(df, "trim_start")
df_clean = handle_missing(df, "drop_vars", max_missing_pct=0.5)
df_interp = handle_missing(df, "interpolate")
df_ff     = handle_missing(df, "forward_fill")
```

---

## Outlier Detection

`MacroFrame.outlier_flag()` returns a boolean mask of outlier cells using IQR-based detection:

```python
flags = md.outlier_flag(method="iqr", threshold=10.0)
# flags is a DataFrame with True where |x - median| > threshold * IQR

n_flags = flags.sum()
print(n_flags[n_flags > 0])   # variables with at least one outlier
```

The threshold of 10 IQR units follows the recommendation in McCracken and Ng (2016).
