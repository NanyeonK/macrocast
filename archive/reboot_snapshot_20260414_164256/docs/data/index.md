# Data Layer Overview

The data layer (Layer 1) provides loaders and pre-processing utilities for the three main FRED datasets.

---

## Supported Databases

| Database | Frequency | Coverage | Series | Format |
|----------|-----------|----------|--------|--------|
| FRED-MD | Monthly | 1959:01 to present | ~128 | CSV |
| FRED-QD | Quarterly | 1959:Q1 to present | ~248 | CSV |
| FRED-SD | Monthly / Quarterly | Varies by state | ~28 × 51 | XLSX |

---

## Architecture

All three loaders return a `MacroFrame` object, which wraps a `pandas.DataFrame` with dataset metadata: vintage identifier, transformation codes, variable descriptions, and group assignments. MacroFrame is immutable — every method that modifies data returns a new instance.

```
FRED source (CSV / XLSX)
        │
        ▼
  download + cache
  (~/.macrocast/cache/)
        │
        ▼
  parse raw data
  (tcode row, date row)
        │
        ▼
  build MacroFrame
  (data + metadata + tcodes)
        │
        ▼
  .transform()  .group()  .trim()  .handle_missing()
```

---

## Entry Points

```python
import macrocast as mc

md = mc.load_fred_md()                        # FRED-MD monthly
qd = mc.load_fred_qd()                        # FRED-QD quarterly
sd = mc.load_fred_sd(states=["CA", "TX"])     # FRED-SD state-level
```

---

## Variable Groups (FRED-MD and FRED-QD)

Variables are organized into thematic groups following McCracken and Ng (2016):

| Group key | Description |
|-----------|-------------|
| `output_income` | Industrial production, GDP, income aggregates |
| `labor` | Employment, hours, wages, unemployment |
| `housing` | Permits, starts, sales, prices |
| `prices` | CPI, PPI, commodity prices |
| `money_credit` | M1, M2, credit aggregates |
| `interest_rates` | Federal funds rate, Treasury yields, spreads |
| `stock_market` | S&P 500, dividend yield, volatility (VXO) |
| `other` | Exchange rates, consumer sentiment, miscellaneous |

Access a group subset via `MacroFrame.group()`:

```python
prices = md.group("prices")
labor  = md.group("labor")
```

---

## Caching

All loaders cache downloaded files under `~/.macrocast/cache/{dataset}/`. Current vintage files expire after 30 days; historical vintage files never expire. Pass `force_download=True` to bypass the cache, or `cache_dir` to override the directory.

```python
md = mc.load_fred_md(force_download=True)
md = mc.load_fred_md(cache_dir="/data/fred")
```

---

## References

McCracken, M.W. and Ng, S. (2016). "FRED-MD: A Monthly Database for Macroeconomic Research." *Journal of Business and Economic Statistics*, 34(4), 574-589.

McCracken, M.W. and Owyang, M.T. (2021). "The St. Louis Fed's Financial Stress Index, Version 2." Federal Reserve Bank of St. Louis Working Paper 2021-016.
