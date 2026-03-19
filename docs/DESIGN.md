# macrocast: Package Design Document (v0.1)

## 1. Overview

**Package name**: `macrocast`
**Tagline**: Decomposing ML Forecast Gains in Macroeconomics
**Languages**: Python (primary), R (companion package)
**License**: MIT
**Target venue**: IJF Special Issue "Advances in Open Source Forecasting Software" (deadline: 2026-08-31)

### 1.1 Problem Statement

Coulombe et al. (2020, JBES) decomposed ML forecasting gains into four treatment effects: nonlinearity, regularization, cross-validation, and loss function. This decomposition revealed that nonlinearity is the primary driver of ML gains in macroeconomic forecasting, and that these gains concentrate in periods of high uncertainty and financial stress.

However, replicating and extending this decomposition across different datasets, target variables, forecast horizons, and sample periods requires substantial bespoke coding. No existing open-source tool provides:

- Unified access to the FRED-MD/QD/SD database ecosystem with automatic transformation and vintage management
- A modular pipeline where each ML component can be independently toggled for controlled experiments
- Regime-conditional forecast evaluation tied to macroeconomic state variables

### 1.2 Design Philosophy

Three principles guide the architecture:

1. **Decomposability first**: Every design decision serves the goal of isolating individual sources of forecast improvement. The pipeline is not optimized for raw predictive accuracy but for transparent attribution of performance gains.
2. **FRED-native**: The data layer treats FRED-MD/QD/SD as first-class citizens, not generic CSV inputs. Transformation codes, vintage structure, and variable groupings (output, labor, housing, prices, money, interest rates, stock market) are built into the schema.
3. **Minimal core, extensible surface**: The package ships with a small set of well-tested models (AR, Ridge, Random Forest, feedforward NN). External models enter through a standard interface compatible with scikit-learn (Python) and tidymodels (R).


---

## 2. Architecture

```
macrocast/
├── data/                    # Layer 1: Data
│   ├── fred_md.py           # FRED-MD downloader & processor
│   ├── fred_qd.py           # FRED-QD downloader & processor
│   ├── fred_sd.py           # FRED-SD downloader & processor
│   ├── transforms.py        # Transformation code engine (tcodes 1-7)
│   ├── vintages.py          # Vintage management & real-time alignment
│   └── schema.py            # Variable metadata, groups, frequencies
│
├── pipeline/                # Layer 2: Forecasting Pipeline
│   ├── experiment.py        # ForecastExperiment orchestrator
│   ├── components.py        # Pluggable component definitions
│   │   ├── nonlinearity     #   linear vs. nonlinear toggle
│   │   ├── regularization   #   none / L1 / L2 / factor / etc.
│   │   ├── cv_scheme        #   rolling / expanding / K-fold / BIC
│   │   └── loss_function    #   L2 / L1 / epsilon-insensitive / quantile
│   ├── models.py            # Built-in model zoo (AR, Ridge, RF, NN)
│   ├── direct_multi.py      # Direct vs. iterated multi-step forecasting
│   └── adapters.py          # sklearn / tidymodels adapter interface
│
├── evaluation/              # Layer 3: Evaluation
│   ├── metrics.py           # MSFE, MAE, MAFE ratios, CSFE
│   ├── regime.py            # Regime-conditional evaluation
│   ├── mcs.py               # Model Confidence Set (Hansen et al., 2011)
│   ├── decomposition.py     # Treatment-effect decomposition engine
│   └── visualization.py     # Forecast comparison plots
│
├── utils/
│   ├── parallel.py          # Joblib/multiprocessing wrapper
│   ├── cache.py             # Local caching for downloaded data
│   └── config.py            # YAML-based experiment configuration
│
└── cli.py                   # Command-line interface
```


---

## 3. Layer 1: Data

### 3.1 Supported Databases

| Database | Frequency  | Coverage           | Series    | Source format |
|----------|------------|--------------------|-----------|---------------|
| FRED-MD  | Monthly    | 1959:01 - present  | ~130      | CSV           |
| FRED-QD  | Quarterly  | 1959:Q1 - present  | ~250      | CSV           |
| FRED-SD  | Mixed (M/Q)| Varies by state    | ~28 × 51  | XLSX          |

### 3.2 Core Data API

```python
import macrocast as mc

# Download latest vintage
md = mc.load_fred_md()                       # returns MacroFrame
qd = mc.load_fred_qd()
sd = mc.load_fred_sd(states=["CA", "TX"])    # subset by state

# Download specific vintage
md_202001 = mc.load_fred_md(vintage="2020-01")

# Automatic transformation (stationarity)
md_transformed = md.transform()              # applies tcodes from row 1
md_transformed = md.transform(override={"INDPRO": 5})  # override specific

# Variable grouping
prices = md.group("prices")                  # subset by category
real = md.group("output_income")

# Real-time alignment for pseudo-out-of-sample
rt_panel = mc.RealTimePanel(
    database="fred_md",
    vintages=("2010-01", "2020-12"),         # range of vintages
    target="INDPRO"
)
```

### 3.3 MacroFrame Object

`MacroFrame` extends pandas DataFrame with:

- `.tcode`: transformation code vector
- `.group_map`: variable-to-category mapping
- `.vintage_date`: vintage identifier
- `.transform(method="default")`: apply tcodes
- `.factors(n_factors=8, method="PCA")`: extract static factors
- `.to_panel()`: convert to long-format panel (for SD cross-state analysis)
- `.missing_report()`: summary of missing values by variable and date
- `.outlier_flag(method="iqr", threshold=10)`: detect and flag outliers

### 3.4 FRED-SD Specifics

FRED-SD presents unique data engineering challenges:

- Source files are .xlsx with tabs per series and columns per state
- Mixed frequencies (monthly labor data, quarterly income/housing)
- State-level variable availability differs by state industrial composition
- Vintages available from 2005 onward

The data layer handles these by normalizing to a unified `StateMacroFrame` with explicit frequency tags and interpolation options for mixed-frequency analysis.


---

## 4. Layer 2: Forecasting Pipeline

### 4.1 The Decomposition Framework

Following Coulombe et al. (2020), the pipeline decomposes ML forecasting into four orthogonal "treatment" dimensions:

| Dimension       | Baseline (off)              | Treatment (on)                    |
|-----------------|-----------------------------|-----------------------------------|
| Nonlinearity    | Linear model                | Nonlinear model (RF, NN, etc.)    |
| Regularization  | OLS / no penalty            | Ridge, Lasso, Factors, Elastic Net|
| CV scheme       | BIC / fixed window          | K-fold, rolling-origin, expanding |
| Loss function   | L2 (squared error)          | L1, epsilon-insensitive, quantile |

The key design insight: each dimension is a **pluggable component**, not a model hyperparameter. This means the user constructs experiments by mixing and matching components, and the pipeline evaluates all requested combinations.

### 4.2 Experiment API

```python
from macrocast.pipeline import ForecastExperiment
from macrocast.pipeline.components import (
    Nonlinearity, Regularization, CVScheme, LossFunction
)

exp = ForecastExperiment(
    data=md_transformed,
    target="INDPRO",
    horizons=[1, 3, 6, 12],
    eval_start="1990-01",
    eval_end="2019-12",
    expanding=True,                          # expanding window
    min_train_size=120,                      # minimum training window (months)
)

# Define treatment grid
exp.set_components(
    nonlinearity=[
        Nonlinearity.LINEAR,                 # OLS / Ridge
        Nonlinearity.RANDOM_FOREST,
        Nonlinearity.NEURAL_NET,
    ],
    regularization=[
        Regularization.NONE,
        Regularization.RIDGE,
        Regularization.FACTORS(n=8),         # PCA factor augmentation
    ],
    cv_scheme=[
        CVScheme.BIC,
        CVScheme.KFOLD(k=5),
        CVScheme.ROLLING_ORIGIN(window=60),
    ],
    loss_function=[
        LossFunction.L2,
        LossFunction.L1,
    ],
)

# Run all combinations (3 × 3 × 3 × 2 = 54 configurations)
results = exp.run(n_jobs=-1)
```

### 4.3 Built-in Models

| Model           | Nonlinearity | Implementation         |
|-----------------|-------------|------------------------|
| AR(p)           | Linear      | statsmodels / custom   |
| Ridge           | Linear      | sklearn                |
| Lasso           | Linear      | sklearn                |
| Elastic Net     | Linear      | sklearn                |
| Factor + OLS    | Linear      | PCA + statsmodels      |
| Random Forest   | Nonlinear   | sklearn                |
| Gradient Boost  | Nonlinear   | lightgbm               |
| Neural Net (FF) | Nonlinear   | pytorch (1-2 hidden)   |
| SVR             | Nonlinear   | sklearn                |

### 4.4 Custom Model Interface

```python
from macrocast.pipeline.adapters import MacrocastEstimator

class MyModel(MacrocastEstimator):
    """User-defined model must implement fit() and predict()."""

    def fit(self, X, y, **kwargs):
        # X: (T_train, N) array
        # y: (T_train,) array
        ...
        return self

    def predict(self, X, **kwargs):
        # X: (T_test, N) array
        # returns: (T_test,) array
        ...
        return y_hat

    @property
    def nonlinearity_type(self):
        return "nonlinear"  # or "linear"
```

### 4.5 Data Environment Modes

Coulombe et al. distinguish between data-rich (N >> 1) and data-poor (small N) environments. The pipeline supports both:

- **Data-rich**: Use full FRED-MD/QD panel (~130 or ~250 predictors)
- **Data-poor**: Use only lags of the target variable + a few selected predictors
- **Factor-augmented**: Extract k factors from the panel, use as predictors

```python
exp.set_data_environment(
    mode="data_rich",           # "data_rich" | "data_poor" | "factor_augmented"
    n_factors=8,                # for factor_augmented mode
    factor_method="PCA",        # "PCA" | "sparse_PCA" | "targeted"
)
```

### 4.6 Multi-step Forecasting

```python
exp.set_multistep(
    method="direct",            # "direct" | "iterated" | "both"
)
```

- **Direct**: Estimate separate model for each horizon h
- **Iterated**: One-step model iterated forward
- Comparison of direct vs. iterated is itself a design question for macro forecasting


---

## 5. Layer 3: Evaluation

### 5.1 Core Metrics

- MSFE (Mean Squared Forecast Error)
- Relative MSFE (vs. AR benchmark)
- MAE, MAFE
- Cumulative Squared Forecast Error (CSFE) over time
- Directional accuracy

### 5.2 Treatment-Effect Decomposition

This is the core methodological contribution of the package. For any pair of configurations that differ in exactly one component, the difference in forecast performance is attributed to that component.

```python
from macrocast.evaluation import decompose

decomp = decompose(results)

# Returns a DecompositionTable with:
# - marginal effect of nonlinearity (averaging over other dimensions)
# - marginal effect of regularization
# - marginal effect of CV scheme
# - marginal effect of loss function
# - interaction terms (optional)

decomp.summary()
decomp.plot_waterfall(target="INDPRO", horizon=12)
```

Formally, define MSFE(n, r, c, l) as the forecast error for configuration (nonlinearity n, regularization r, CV scheme c, loss l). The marginal effect of nonlinearity is:

```
ΔNL = E_{r,c,l}[MSFE(nonlinear, r, c, l)] - E_{r,c,l}[MSFE(linear, r, c, l)]
```

where the expectation is taken over all combinations of the other three dimensions.

### 5.3 Regime-Conditional Evaluation

A key finding of Coulombe et al. is that nonlinear gains concentrate in periods of macroeconomic uncertainty and financial stress. The package makes this analysis first-class:

```python
from macrocast.evaluation import RegimeEvaluator

regime_eval = RegimeEvaluator(
    results=results,
    regime_indicators={
        "uncertainty": "VXOCLSX",          # VXO (from FRED-MD)
        "financial_stress": "STLFSI",      # St. Louis Financial Stress Index
        "recession": "USREC",              # NBER recession indicator
    },
    threshold_method="quantile",           # "quantile" | "fixed" | "ms" (Markov switching)
    quantile=0.75,                         # high regime = top 25%
)

regime_eval.compare(metric="relative_msfe")
regime_eval.plot_csfe_by_regime()
```

### 5.4 Model Confidence Set

```python
from macrocast.evaluation import model_confidence_set

mcs = model_confidence_set(
    results,
    alpha=0.10,
    statistic="T_max",                     # "T_max" | "T_R"
    bootstrap="block",
    block_length="auto",
)

mcs.superior_set                           # models not rejected
mcs.p_values                               # MCS p-values per model
```

### 5.5 Visualization Suite

| Plot type                | Description                                       |
|--------------------------|---------------------------------------------------|
| `waterfall`              | Decomposition of ML gain by component             |
| `csfe_path`              | Cumulative forecast error over time                |
| `regime_heatmap`         | Performance matrix: model × regime                 |
| `horizon_profile`        | Relative MSFE across forecast horizons             |
| `variable_dashboard`     | Multi-target comparison in a single view           |
| `vintage_stability`      | How results change across data vintages            |


---

## 6. Configuration & Reproducibility

### 6.1 YAML Experiment Config

All experiments can be defined in a YAML file for full reproducibility:

```yaml
# experiment_config.yaml
name: "coulombe_replication_extended"
seed: 42

data:
  database: "fred_md"
  vintage: "latest"
  target: ["INDPRO", "PAYEMS", "CPIAUCSL", "CPILFESL"]
  transform: "default"
  outlier_treatment: "iqr_10"

pipeline:
  horizons: [1, 3, 6, 12]
  eval_start: "1990-01"
  eval_end: "2023-12"
  min_train_size: 120
  expanding: true
  multistep: "direct"

  data_environment: "data_rich"
  n_factors: 8

  nonlinearity: ["linear", "random_forest", "neural_net"]
  regularization: ["none", "ridge", "factors"]
  cv_scheme: ["bic", "kfold_5", "rolling_60"]
  loss_function: ["l2", "l1"]

evaluation:
  metrics: ["msfe", "relative_msfe", "mae", "direction"]
  benchmark: "ar_bic"
  mcs:
    alpha: 0.10
    bootstrap: "block"
  regime:
    indicators: ["VXOCLSX", "STLFSI", "USREC"]
    method: "quantile"
    quantile: 0.75
  decomposition:
    include_interactions: false

output:
  path: "./results/"
  format: ["csv", "latex", "plots"]
```

### 6.2 CLI

```bash
# Run experiment from config
macrocast run experiment_config.yaml

# Quick single-target experiment
macrocast forecast --target INDPRO --horizon 12 --database fred_md

# Download and cache data
macrocast data download --database fred_md --vintage latest
macrocast data download --database fred_sd --states all

# Generate decomposition report
macrocast report --results ./results/ --format latex
```


---

## 7. R Companion Package

The R package (`macrocastR`) mirrors the Python API with idiomatic R syntax:

```r
library(macrocastR)

# Data
md <- load_fred_md()
md_t <- transform_fred(md)

# Experiment
exp <- forecast_experiment(
  data = md_t,
  target = "INDPRO",
  horizons = c(1, 3, 6, 12),
  eval_start = "1990-01",
  eval_end = "2019-12"
)

exp <- exp |>
  set_nonlinearity(c("linear", "random_forest")) |>
  set_regularization(c("none", "ridge", "factors")) |>
  set_cv_scheme(c("bic", "kfold_5")) |>
  set_loss(c("l2"))

results <- run_experiment(exp, cores = 4)
decompose(results) |> plot_waterfall()
```

R implementation uses:
- `data.table` / `tibble` for data handling
- `ranger` (RF), `glmnet` (regularized linear), `nnet`/`torch` (NN)
- `tidymodels` adapter for external models
- `ggplot2` for visualization


---

## 8. Scope Boundaries

### In scope (v1.0 for IJF submission)
- FRED-MD/QD/SD data pipeline with vintage management
- Four-component decomposition framework
- Core models: AR, Ridge, Lasso, RF, NN (feedforward)
- Regime-conditional evaluation with 2-3 indicators
- MCS testing
- YAML config + CLI
- Python package with R companion

### Out of scope (future versions)
- Pre-trained foundation models (Chronos, TimesFM, etc.)
- Bayesian methods (BVAR, BSTS)
- Mixed-frequency / nowcasting (MIDAS, DFM)
- International macro databases (ECB SDW, BOE, etc.)
- Dashboard / web UI
- GPU acceleration for large-scale NN experiments


---

## 9. Development Timeline

| Phase         | Period              | Deliverable                           |
|---------------|---------------------|---------------------------------------|
| Phase 1       | 2026-03 ~ 04        | Data layer (Layer 1) complete         |
| Phase 2       | 2026-04 ~ 05        | Pipeline core (Layer 2) complete      |
| Phase 3       | 2026-05 ~ 06        | Evaluation (Layer 3) + decomposition  |
| Phase 4       | 2026-06 ~ 07        | R companion + CLI + documentation     |
| Paper draft   | 2026-06 ~ 07        | IJF paper writing                     |
| Testing       | 2026-07 ~ 08        | Replication of Coulombe et al. (2020) |
| Submission    | 2026-08 (mid)       | Submit to IJF                         |


---

## 10. Paper Outline (Tentative)

**Title**: macrocast: An Open-Source Framework for Decomposing Machine Learning Gains in Macroeconomic Forecasting

1. **Introduction** (2 pp)
   - ML in macro forecasting: growing adoption, limited understanding of "why"
   - Gap: no unified software for controlled decomposition experiments
   - Contribution: open-source tool bridging FRED data ecosystem and decomposition methodology

2. **The Decomposition Framework** (3 pp)
   - Coulombe et al. (2020) methodology
   - Formalization of treatment effects in a factorial experiment design
   - Extension: regime-conditional decomposition

3. **Software Design** (4 pp)
   - Data layer: challenges of FRED-MD/QD/SD integration
   - Pipeline layer: component-based architecture for controlled experiments
   - Evaluation layer: regime-aware metrics and MCS
   - Design trade-offs specific to macroeconomic forecasting

4. **Empirical Illustration** (6 pp)
   - Replication of Coulombe et al. on FRED-MD (updated sample)
   - Extension to FRED-QD (quarterly frequency)
   - Extension to FRED-SD (cross-state heterogeneity in ML gains)
   - Regime analysis: do nonlinear gains still concentrate in crisis periods?

5. **Comparison with Existing Tools** (2 pp)
   - BVAR (R), FredMD (Python), prosper_nn, TSLib
   - What macrocast adds: decomposition-first design, FRED-native, regime evaluation

6. **Conclusion** (1 pp)

Appendix: API reference, replication instructions, YAML config examples

**Target length**: ~20 pages (as specified by CFP)


---

## 11. Open Design Questions

1. **Factor extraction method**: PCA is standard, but targeted PCA (Bai and Ng, 2008) or sparse PCA could be options. Include as a component or keep PCA-only for v1?

2. **FRED-SD aggregation**: Should the package support bottom-up national forecasting (forecast each state, then aggregate)? This is interesting but may expand scope.

3. **Interaction terms in decomposition**: Coulombe et al. focus on marginal effects. Should v1 include two-way and three-way interaction analysis?

4. **Real-time vs. pseudo-out-of-sample**: Real-time evaluation using vintages is more rigorous but computationally much heavier. Default to pseudo-OOS with real-time as an option?

5. **Package name**: `macrocast` is clean but generic. Alternatives: `fredcast`, `mlmacro`, `decompcast`. Preferences?
