# Getting Started

This page covers installation and a short walkthrough of the data layer (v0.1). The pipeline and evaluation layers are not yet available.

---

## Installation

**From PyPI:**

```bash
pip install macrocast
```

**From source (recommended for development):**

```bash
git clone https://github.com/macrocast/macrocast.git
cd macrocast
uv sync --all-extras
```

**Python version:** 3.10 or later.

---

## Data Layer Walkthrough

### Loading FRED-MD

```python
import macrocast as mc

# Load the latest vintage (cached under ~/.macrocast/cache/fred_md/)
md = mc.load_fred_md()
print(md)
# MacroFrame(dataset='FRED-MD', vintage='current', T=790, N=128, ...)

# Load a specific vintage
md_2020 = mc.load_fred_md(vintage="2020-01")

# Trim sample and transform in one call
md_t = mc.load_fred_md(
    start="1970-01",
    end="2023-12",
    transform=True,
)
```

### Inspecting the data

```python
# Access the underlying DataFrame
df = md.data
print(df.shape)          # (T, N)
print(df.index[:3])      # DatetimeIndex

# Transformation codes (1-7, McCracken & Ng 2016)
print(md.tcodes["INDPRO"])   # 5 = log-difference

# Variable metadata
vmeta = md.metadata.variables["INDPRO"]
print(vmeta.group)            # 'output_income'
print(vmeta.description)      # 'Industrial Production Index'
```

### Applying transformations

Transformations convert level series to approximately stationary series following the McCracken-Ng (2016) codes.

```python
# Apply default tcodes from the spec
md_t = md.transform()

# Override a specific variable's tcode
md_t = md.transform(override={"INDPRO": 5, "UNRATE": 2})
```

### Subsetting by variable group

```python
# Available groups: output_income, labor, housing, prices,
#                   money_credit, interest_rates, stock_market, other
output = md.group("output_income")
labor  = md.group("labor")
```

### Missing value handling

```python
# View the missing value report
report = md.missing_report()
print(report[["n_leading", "n_trailing", "n_intermittent", "pct_missing"]])

# Advance start date to eliminate leading NaNs (standard FRED-MD approach)
md_clean = md.handle_missing("trim_start")

# Interpolate interior gaps
md_interp = md.handle_missing("interpolate")

# Drop variables with more than 30% missing
md_drop = md.handle_missing("drop_vars", max_missing_pct=0.3)
```

### Trimming the sample

```python
md_sub = md.trim(start="1970-01", end="2019-12")

# Drop sparse variables during trimming
md_sub = md.trim(start="1970-01", min_obs_pct=0.9)
```

### Method chaining

MacroFrame methods return new objects, so they can be chained:

```python
md_ready = (
    mc.load_fred_md()
    .trim(start="1970-01", end="2023-12")
    .handle_missing("trim_start")
    .transform()
)
```

### FRED-QD (quarterly)

```python
qd = mc.load_fred_qd()
qd_t = qd.transform()
```

### FRED-SD (state-level)

```python
# Load unemployment rates for California and Texas
sd = mc.load_fred_sd(states=["CA", "TX"], variables=["UR"])
print(sd.data.shape)
```

### Vintage management

```python
# List available vintage identifiers (no network call)
vintages = mc.list_available_vintages("fred_md", start="2010-01", end="2020-12")

# Load multiple vintages
panel = mc.load_vintage_panel("fred_md", vintages=["2019-01", "2020-01"])
rt = mc.RealTimePanel(panel)
print(rt)
# RealTimePanel(n_vintages=2, range=2019-01 to 2020-01)
```

---

## Next Steps

- See the [Data Layer overview](data/index.md) for a comparison of FRED-MD, FRED-QD, and FRED-SD.
- See [MacroFrame](data/macroframe.md) for the full API reference of the core container.
- See [Transformations](data/transforms.md) for the McCracken-Ng tcode reference.

!!! note "Pipeline and Evaluation layers"
    The forecasting pipeline (Layer 2) and evaluation layer (Layer 3) are not yet implemented in v0.1. See the [Roadmap](roadmap.md) for the planned timeline.
