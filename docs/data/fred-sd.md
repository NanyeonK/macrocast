# FRED-SD

FRED-SD is a state-level macroeconomic dataset containing monthly and quarterly series for all 50 U.S. states plus the District of Columbia. The source is an Excel workbook where each sheet corresponds to a macroeconomic variable and columns correspond to states.

**Reference:** McCracken, M.W. and Owyang, M.T. (2021). "The St. Louis Fed's Financial Stress Index, Version 2." Federal Reserve Bank of St. Louis Working Paper 2021-016.

---

## Loading

```python
import macrocast as mc

# Load all variables for all states (large: ~28 × 51 = ~1428 columns)
sd = mc.load_fred_sd()

# Specific vintage
sd = mc.load_fred_sd(vintage="2020-01")

# Subset by state
sd = mc.load_fred_sd(states=["CA", "TX", "NY"])

# Subset by variable
sd = mc.load_fred_sd(variables=["UR", "EMPL"])

# Both filters
sd = mc.load_fred_sd(
    states=["CA", "TX"],
    variables=["UR"],
    start="2000-01",
    end="2023-12",
)
```

---

## Function Reference

### `load_fred_sd`

```python
macrocast.load_fred_sd(
    vintage=None,
    states=None,
    variables=None,
    start=None,
    end=None,
    cache_dir=None,
    force_download=False,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vintage` | `str` or `None` | `None` | Vintage in `"YYYY-MM"` format. `None` fetches the latest release. |
| `states` | `list[str]` or `None` | `None` | Two-letter state codes (e.g. `["CA", "TX"]`). `None` retains all 51. |
| `variables` | `list[str]` or `None` | `None` | Variable (sheet) names to include. `None` loads all sheets. |
| `start` | `str` or `None` | `None` | Sample start date. |
| `end` | `str` or `None` | `None` | Sample end date (inclusive). |
| `cache_dir` | `str` or `Path` or `None` | `None` | Override for cache directory. |
| `force_download` | `bool` | `False` | Force re-download. |

**Returns:** `MacroFrame`

---

## Column Naming Convention

FRED-SD columns follow the `{variable}_{state}` convention:

```python
sd = mc.load_fred_sd(states=["CA", "TX"], variables=["UR"])
print(sd.data.columns.tolist())
# ['UR_CA', 'UR_TX']
```

---

## File Format Details

The FRED-SD workbook is an `.xlsx` file with one sheet per variable. Each sheet has:

- Row 1: header with state codes as column names
- Subsequent rows: monthly observations indexed by date

The loader uses `openpyxl` to parse the workbook, normalizes the index to a `DatetimeIndex`, and concatenates all selected sheets into a single wide DataFrame.

Vintage files follow the same `YYYY-MM.xlsx` naming convention and are available from 2005-01 onward:

```python
sd_2020 = mc.load_fred_sd(vintage="2020-01", states=["CA"])
print(sd_2020.vintage)   # '2020-01'

# Enumerate expected vintage identifiers (no network call)
vintages = mc.list_available_vintages("fred_sd", start="2010-01", end="2020-12")
```

---

## Caching

The FRED-SD workbook is cached at `~/.macrocast/cache/fred_sd/FRED_SD.xlsx` and refreshed after 30 days:

```python
# Force re-download
sd = mc.load_fred_sd(force_download=True)

# Custom cache directory
sd = mc.load_fred_sd(cache_dir="/data/fred")
```

---

## Usage Example

```python
import macrocast as mc

# Load unemployment rates for all states
ur = mc.load_fred_sd(variables=["UR"])
print(ur.data.shape)   # (T, 51)

# Compute cross-state average
ur_avg = ur.data.mean(axis=1)

# Check missing patterns
report = ur.missing_report()
print(report[["n_leading", "n_trailing", "pct_missing"]].head())
```
