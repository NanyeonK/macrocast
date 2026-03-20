# MacroFrame

`MacroFrame` is the central data container in macrocast. It wraps a `pandas.DataFrame` with dataset metadata (variable descriptions, transformation codes, groupings, vintage identifier) and exposes a fluent interface for pre-processing.

MacroFrame is **immutable**: every method that transforms data returns a new instance. The original is never modified.

---

## Construction

MacroFrame objects are normally created by the loader functions, not constructed directly:

```python
import macrocast as mc

md = mc.load_fred_md()          # returns MacroFrame
qd = mc.load_fred_qd()
sd = mc.load_fred_sd()
```

---

## Properties

```python
md.data        # pd.DataFrame — the underlying panel (read-only view)
md.metadata    # MacroFrameMetadata — dataset-level metadata
md.tcodes      # dict[str, int] — tcode per variable
md.vintage     # str or None — vintage identifier ("YYYY-MM") or None for current
```

### `metadata` attributes

```python
md.metadata.dataset         # 'FRED-MD'
md.metadata.vintage         # None or '2020-01'
md.metadata.frequency       # 'monthly'
md.metadata.is_transformed  # bool
md.metadata.groups          # dict[str, str] — group key -> display label
md.metadata.variables       # dict[str, VariableMetadata]
```

### `VariableMetadata` attributes

```python
vmeta = md.metadata.variables["INDPRO"]
vmeta.name           # 'INDPRO'
vmeta.description    # 'Industrial Production Index'
vmeta.group          # 'output_income'
vmeta.tcode          # 5
vmeta.frequency      # 'monthly'
```

---

## Methods

### `.transform(override=None)`

Apply stationarity transformations using stored tcodes.

```python
md_t = md.transform()

# Override specific variables
md_t = md.transform(override={"INDPRO": 5, "UNRATE": 2})
```

Returns a new MacroFrame with `metadata.is_transformed = True`.

---

### `.group(group_name)`

Return a MacroFrame restricted to variables in the specified group.

```python
output  = md.group("output_income")
labor   = md.group("labor")
prices  = md.group("prices")
```

Raises `KeyError` if no variables belong to the group.

---

### `.trim(start=None, end=None, min_obs_pct=None)`

Restrict the sample period and optionally drop sparse variables.

```python
md_sub = md.trim(start="1970-01", end="2019-12")

# Drop variables with fewer than 90% non-missing observations
md_sub = md.trim(start="1970-01", min_obs_pct=0.9)
```

---

### `.handle_missing(method, **kwargs)`

Apply a missing-value treatment. Returns a new MacroFrame.

```python
md_clean = md.handle_missing("trim_start")
md_clean = md.handle_missing("drop_vars", max_missing_pct=0.3)
md_interp = md.handle_missing("interpolate")
md_ff     = md.handle_missing("forward_fill")
```

See [Missing Values](missing.md) for method details.

---

### `.missing_report()`

Return a per-variable missing value summary as a DataFrame.

```python
report = md.missing_report()
print(report.columns.tolist())
# ['n_total', 'n_leading', 'n_trailing', 'n_intermittent',
#  'pct_missing', 'first_valid_idx', 'last_valid_idx']
```

---

### `.outlier_flag(method="iqr", threshold=10.0)`

Return a boolean DataFrame marking outlier cells.

```python
flags = md.outlier_flag()                    # default: IQR, threshold=10
flags = md.outlier_flag(threshold=5.0)
```

---

### `.to_numpy()`

Return the data as a 2-D NumPy array of shape `(T, N)` in float64.

```python
arr = md.to_numpy()
print(arr.shape)   # (790, 128)
```

---

## Column Selection

MacroFrame supports direct column indexing via `__getitem__`, forwarded to the underlying DataFrame:

```python
indpro = md["INDPRO"]              # pd.Series
subset = md[["INDPRO", "PAYEMS"]]  # pd.DataFrame
```

---

## Method Chaining

Because every method returns a new MacroFrame, operations can be chained:

```python
md_ready = (
    mc.load_fred_md()
    .trim(start="1970-01", end="2023-12")
    .handle_missing("trim_start")
    .transform()
)
```

---

## Representation

```python
print(md)
# MacroFrame(dataset='FRED-MD', vintage='current', T=790, N=128,
#            period=1959-01-01 to 2024-10-01, status=levels)

print(md_t)
# MacroFrame(dataset='FRED-MD', vintage='current', T=790, N=128,
#            period=1959-01-01 to 2024-10-01, status=transformed)
```

---

## Metadata Classes

### `MacroFrameMetadata`

Dataset-level metadata attached to every MacroFrame.

| Attribute | Type | Description |
|-----------|------|-------------|
| `dataset` | `str` | Source dataset (`"FRED-MD"`, `"FRED-QD"`, `"FRED-SD"`) |
| `vintage` | `str` or `None` | Vintage identifier or `None` for current release |
| `frequency` | `str` | `"monthly"`, `"quarterly"`, `"state_monthly"` |
| `variables` | `dict[str, VariableMetadata]` | Per-variable metadata |
| `groups` | `dict[str, str]` | Group key to display label mapping |
| `is_transformed` | `bool` | Whether stationarity transformations have been applied |

### `VariableMetadata`

Per-variable metadata.

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | FRED mnemonic (e.g. `"INDPRO"`) |
| `description` | `str` | Human-readable label |
| `group` | `str` | Group key (e.g. `"output_income"`) |
| `tcode` | `int` | McCracken-Ng transformation code (1-7) |
| `frequency` | `str` | Observation frequency |
