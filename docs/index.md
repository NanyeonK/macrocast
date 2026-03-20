# macrocast

**Decomposing ML Forecast Gains in Macroeconomic Forecasting**

macrocast is an open-source Python framework for systematic evaluation of machine learning methods in macroeconomic forecasting. It provides unified access to the FRED-MD, FRED-QD, and FRED-SD database ecosystem and implements the four-component decomposition framework of Coulombe et al. (2020, JBES).

---

## Current Status

| Layer | Status | Description |
|-------|--------|-------------|
| **Data (v0.1)** | Complete | FRED-MD, FRED-QD, FRED-SD loaders, MacroFrame, transformations, missing value handling, vintage management |
| **Pipeline (v0.2)** | Planned | ForecastExperiment, four-component decomposition, model zoo |
| **Evaluation (v0.3)** | Planned | MSFE, MCS, regime-conditional evaluation, decomposition tables |

---

## Installation

```bash
pip install macrocast
```

With optional extras:

```bash
pip install macrocast[ml]    # LightGBM + PyTorch
pip install macrocast[viz]   # matplotlib + seaborn
pip install macrocast[all]   # all extras
```

For development:

```bash
git clone https://github.com/macrocast/macrocast.git
cd macrocast
uv sync --all-extras
```

---

## Quick Example

```python
import macrocast as mc

# Load and transform FRED-MD (latest vintage)
md = mc.load_fred_md()
md_t = md.transform()

# Inspect
print(md_t)
# MacroFrame(dataset='FRED-MD', vintage='current', T=790, N=128,
#            period=1959-01-01 to 2024-10-01, status=transformed)

# Subset to output and income variables
output = md_t.group("output_income")

# Check missing values
report = md_t.missing_report()
print(report[["n_leading", "n_trailing", "n_intermittent"]].head())
```

---

## Design Principles

macrocast is built around three principles:

**Decomposability first.** Every design decision serves the goal of isolating individual sources of forecast improvement. The pipeline is not optimized for raw predictive accuracy but for transparent attribution of performance gains.

**FRED-native.** The data layer treats FRED-MD, FRED-QD, and FRED-SD as first-class citizens. Transformation codes, vintage structure, and variable groupings are built into the schema.

**Minimal core, extensible surface.** The package ships with a small set of well-tested models. External models enter through a standard scikit-learn-compatible interface.

---

## Citation

If you use macrocast in your research, please cite:

```bibtex
@article{macrocast2026,
  title   = {macrocast: An Open-Source Framework for Decomposing
             Machine Learning Gains in Macroeconomic Forecasting},
  author  = {Chan},
  journal = {International Journal of Forecasting},
  year    = {2026}
}
```

The decomposition methodology follows Coulombe et al. (2020):

```bibtex
@article{coulombe2020,
  title   = {How is Machine Learning Useful for Macroeconomic Forecasting?},
  author  = {Coulombe, Philippe Goulet and Leroux, Maxime and
             Stevanovic, Dalibor and Surprenant, St{\'e}phane},
  journal = {Journal of Business \& Economic Statistics},
  year    = {2020}
}
```
