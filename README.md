# macrocast

Decomposing ML Forecast Gains in Macroeconomic Forecasting.

An open-source Python (+ R) framework for systematic evaluation of machine learning methods in macroeconomic forecasting, with built-in support for the FRED-MD, FRED-QD, and FRED-SD database ecosystem.

[![CI](https://github.com/macrocast/macrocast/actions/workflows/ci.yml/badge.svg)](https://github.com/macrocast/macrocast/actions/workflows/ci.yml)
[![Docs](https://github.com/macrocast/macrocast/actions/workflows/docs.yml/badge.svg)](https://macrocast.github.io/macrocast)
[![PyPI](https://img.shields.io/pypi/v/macrocast)](https://pypi.org/project/macrocast/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Status

| Layer | Version | Status |
|-------|---------|--------|
| Data (FRED-MD/QD/SD) | v0.1.0 | Complete |
| Forecasting Pipeline | v0.2.0 | Complete |
| Evaluation | v0.3.0 | Complete |

---

## Installation

```bash
pip install macrocast
# or with all extras
pip install macrocast[all]
```

---

## Quick Start

```python
import macrocast as mc

# Load and transform FRED-MD (latest vintage, cached locally)
md = mc.load_fred_md()
md_t = md.transform()

print(md_t)
# MacroFrame(dataset='FRED-MD', vintage='current', T=790, N=128,
#            period=1959-01-01 to 2024-10-01, status=transformed)

# Subset by variable group
output = md_t.group("output_income")   # INDPRO, RPI, ...
prices = md_t.group("prices")          # CPI, PPI, ...

# Check missing values
report = md.missing_report()
print(report[["n_leading", "n_trailing", "n_intermittent"]].head())

# Method chaining
md_ready = (
    mc.load_fred_md()
    .trim(start="1970-01", end="2023-12")
    .handle_missing("trim_start")
    .transform()
)

# Load a specific vintage
md_2020 = mc.load_fred_md(vintage="2020-01")

# FRED-QD (quarterly)
qd = mc.load_fred_qd()

# FRED-SD (state-level)
sd = mc.load_fred_sd(states=["CA", "TX"], variables=["UR"])
```

---

## Documentation

Full documentation is available at [macrocast.github.io/macrocast](https://macrocast.github.io/macrocast).

---

## License

MIT
