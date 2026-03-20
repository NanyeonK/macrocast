# FRED-QD

FRED-QD is the quarterly counterpart to FRED-MD, containing ~248 macroeconomic series at quarterly frequency. It includes GDP-side variables and additional financial and international series not available at the monthly frequency.

**Reference:** McCracken, M.W. and Ng, S. (2016). "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business and Economic Statistics*, 34(4), 574-589. (FRED-QD follows the same methodology.)

---

## Loading

```python
import macrocast as mc

# Latest vintage
qd = mc.load_fred_qd()

# Specific vintage
qd = mc.load_fred_qd(vintage="2020-Q1")

# With transformation
qd_t = mc.load_fred_qd(
    start="1970-Q1",
    end="2023-Q4",
    transform=True,
)
```

---

## Function Reference

### `load_fred_qd`

```python
macrocast.load_fred_qd(
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
| `vintage` | `str` or `None` | `None` | Vintage identifier. `None` fetches the latest release. |
| `start` | `str` or `None` | `None` | Sample start date. |
| `end` | `str` or `None` | `None` | Sample end date (inclusive). |
| `transform` | `bool` | `False` | Apply stationarity transformations immediately. |
| `tcode_override` | `dict[str, int]` or `None` | `None` | Per-variable tcode overrides. |
| `cache_dir` | `str` or `Path` or `None` | `None` | Override for cache directory. |
| `force_download` | `bool` | `False` | Force re-download. |

**Returns:** `MacroFrame`

---

## Differences from FRED-MD

| Feature | FRED-MD | FRED-QD |
|---------|---------|---------|
| Frequency | Monthly | Quarterly |
| Series count | ~128 | ~248 |
| GDP components | No | Yes |
| International series | Limited | More complete |
| Date format in CSV | `mm/dd/YYYY` | `mm/dd/YYYY` |
| Coverage start | 1959:01 | 1959:Q1 |

The file format is identical to FRED-MD: row 1 contains transformation codes, row 2 contains column headers.

---

## Usage Example

```python
import macrocast as mc

qd = mc.load_fred_qd()
qd_t = qd.transform()

# Subset to output and income variables
output = qd_t.group("output_income")
print(output.data.shape)

# Check missing patterns
report = qd.missing_report()
print(report.sort_values("pct_missing", ascending=False).head(10))
```
