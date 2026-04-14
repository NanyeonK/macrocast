# FRED-MD

FRED-MD is a balanced monthly panel of ~128 macroeconomic series maintained by the Federal Reserve Bank of St. Louis. It is the primary dataset for the macrocast empirical analysis.

**Reference:** McCracken, M.W. and Ng, S. (2016). "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business and Economic Statistics*, 34(4), 574-589.

---

## Loading

```python
import macrocast as mc

# Latest vintage (cached, refreshed after 30 days)
md = mc.load_fred_md()

# Specific vintage
md = mc.load_fred_md(vintage="2020-01")

# With sample trimming and transformation applied immediately
md = mc.load_fred_md(
    start="1970-01",
    end="2023-12",
    transform=True,
)

# Override a transformation code
md = mc.load_fred_md(
    transform=True,
    tcode_override={"INDPRO": 5, "UNRATE": 2},
)
```

---

## Function Reference

### `load_fred_md`

```python
macrocast.load_fred_md(
    vintage=None,
    start=None,
    end=None,
    transform=False,
    tcode_override=None,
    cache_dir=None,
    force_download=False,
)
```

**Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vintage` | `str` or `None` | `None` | Vintage in `"YYYY-MM"` format. `None` fetches the latest release. |
| `start` | `str` or `None` | `None` | Sample start date, e.g. `"1970-01"`. |
| `end` | `str` or `None` | `None` | Sample end date (inclusive), e.g. `"2023-12"`. |
| `transform` | `bool` | `False` | Apply McCracken-Ng stationarity transformations immediately. |
| `tcode_override` | `dict[str, int]` or `None` | `None` | Per-variable tcode overrides. Only used when `transform=True`. |
| `cache_dir` | `str` or `Path` or `None` | `None` | Override for the default cache directory. |
| `force_download` | `bool` | `False` | Force re-download even if a valid cache exists. |

**Returns:** `MacroFrame`

---

## File Format

FRED-MD CSV files have a non-standard structure that the loader handles explicitly:

- **Row 1:** transformation codes (tcode 1-7) for each variable
- **Row 2:** column header with variable mnemonics (first column is `sasdate`)
- **Rows 3+:** monthly observations

The loader reads the tcode row before parsing the data, then merges these with the bundled spec file for additional metadata.

---

## Vintage Availability

Monthly vintages are available from 1999-01 onward. The `list_available_vintages` function enumerates expected vintage identifiers without making a network request:

```python
vintages = mc.list_available_vintages("fred_md", start="2010-01", end="2020-12")
```

Not every generated vintage identifier has a corresponding file on the FRED server; discontinued months are silently absent.

---

## Available Groups

FRED-MD variables are organized into eight groups. Use `MacroFrame.group()` to subset:

```python
md = mc.load_fred_md()
output = md.group("output_income")   # INDPRO, RPI, ...
labor  = md.group("labor")           # PAYEMS, UNRATE, ...
prices = md.group("prices")          # CPIAUCSL, PPIACO, ...
```
