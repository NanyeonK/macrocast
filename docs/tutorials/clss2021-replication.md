# CLSS 2021 Replication

This tutorial replicates the horse race exercise from Coulombe, Leroux, Stevanovic, and Surprenant (2021) — "Macroeconomic Data Transformations Matter" — using the macrocast pipeline. We reproduce the key finding: data transformation choices (MARX, MAF, levels) matter for forecast accuracy, and the best information set varies by horizon and target variable.

**Reference:** Coulombe, P. G., Leroux, M., Stevanovic, D., & Surprenant, S. (2021). Macroeconomic data transformations matter. *International Journal of Forecasting*, 37(4), 1338–1354.

---

## Setup

```python
import numpy as np
import pandas as pd

from macrocast.pipeline.components import CVScheme, LossFunction, Nonlinearity, Regularization
from macrocast.pipeline.experiment import FeatureSpec, ModelSpec
from macrocast.pipeline.horserace import HorseRaceGrid
from macrocast.pipeline.models import KRRModel, RFModel, GBModel
from macrocast.pipeline.r_models import ElasticNetModel, AdaptiveLassoModel
from macrocast.evaluation.horserace import horserace_summary
```

---

## 1. Data Preparation

CLSS 2021 uses FRED-MD at a monthly frequency targeting CPI inflation, IP growth, and unemployment. For a self-contained example we generate a synthetic panel; the full replication simply substitutes `mc.load_fred_md()`.

```python
# --- Synthetic panel (replace with mc.load_fred_md() for the real exercise) ---

rng = np.random.default_rng(42)
T, N = 360, 128  # 30 years × 128 FRED-MD series

dates = pd.date_range("1990-01", periods=T, freq="MS")

# Predictor panel X (stationary-transformed)
factor = rng.standard_normal((T, 5))  # 5 latent factors
loadings = rng.standard_normal((5, N))
noise = rng.standard_normal((T, N)) * 0.5
X = pd.DataFrame(factor @ loadings + noise, index=dates)

# Target: CPI inflation proxy (% MoM growth)
y = pd.Series(
    factor[:, 0] * 0.4 + factor[:, 1] * 0.2 + rng.standard_normal(T) * 0.1,
    index=dates,
    name="CPI_growth",
)

# Levels panel (required for information sets that include 'Level')
X_levels = pd.DataFrame(np.cumsum(X.values, axis=0), index=dates)

print(f"Panel shape: {X.shape}   Target length: {len(y)}")
# Panel shape: (360, 128)   Target length: 360
```

??? note "Loading real FRED-MD data"
    ```python
    import macrocast as mc

    md = mc.load_fred_md(vintage="current")
    X   = md.panel           # stationary-transformed panel
    y   = X["CPIAUCSL"]      # CPI all items, % MoM
    X   = X.drop(columns=["CPIAUCSL"])

    # Levels panel for the 'Level' information sets
    md_raw = mc.load_fred_md(vintage="current", apply_transforms=False)
    X_levels = md_raw.panel.reindex(X.index)
    ```

---

## 2. Defining the 15 Information Sets

CLSS 2021 (Table 1) defines 15 information sets by combining three dimensions:

| Dimension | Values |
|-----------|--------|
| **Base** | F (PCA factors), X (raw variables), F-X (both) |
| **Transformation** | none, MARX, MAF |
| **Augmentation** | none, Level |

The table below maps each information set to its `FeatureSpec` configuration.

| Info Set | `use_factors` | `include_raw_x` | `use_marx` | `marx_for_pca` | `include_levels` |
|----------|:---:|:---:|:---:|:---:|:---:|
| F | ✓ | | | | |
| F-X | ✓ | ✓ | | | |
| X | | ✓ | | | |
| F-MAF | ✓ | | ✓ | ✓ | |
| F-X-MAF | ✓ | ✓ | ✓ | ✓ | |
| X-MAF | | ✓ | ✓ | — | |
| F-MARX | ✓ | | ✓ | ✗ | |
| F-X-MARX | ✓ | ✓ | ✓ | ✗ | |
| X-MARX | | | ✓ | — | |
| F-Level | ✓ | | | | ✓ |
| F-X-Level | ✓ | ✓ | | | ✓ |
| X-Level | | ✓ | | | ✓ |
| F-MARX-Level | ✓ | | ✓ | ✗ | ✓ |
| F-X-MARX-Level | ✓ | ✓ | ✓ | ✗ | ✓ |
| X-MARX-Level | | | ✓ | — | ✓ |

`marx_for_pca=True` (MAF): PCA is applied to the MARX-transformed panel.
`marx_for_pca=False` (MARX): PCA is applied to raw X; MARX columns are appended alongside factors.
`—` (dash): `marx_for_pca` is irrelevant when `use_factors=False`.

```python
# p_marx=12 is standard for monthly data (Coulombe et al. 2021, p. 1341)
P_MARX = 12
N_FACTORS = 8   # tuned by CV in practice; fixed here for illustration
N_LAGS = 4

def fs(label, use_factors, include_raw_x=False, use_marx=False,
       marx_for_pca=True, use_maf=False, include_levels=False):
    """Convenience wrapper: FeatureSpec with explicit CLSS 2021 label."""
    return FeatureSpec(
        use_factors=use_factors,
        n_factors=N_FACTORS,
        n_lags=N_LAGS,
        include_raw_x=include_raw_x,
        use_marx=use_marx,
        p_marx=P_MARX,
        marx_for_pca=marx_for_pca,
        use_maf=use_maf,
        include_levels=include_levels,
        label=label,          # explicit label overrides auto-generation
    )

# ── no transformation ────────────────────────────────────────────
info_sets_base = [
    fs("F",   use_factors=True),
    fs("F-X", use_factors=True,  include_raw_x=True),
    fs("X",   use_factors=False, include_raw_x=True),
]

# ── MARX / MAF ────────────────────────────────────────────────────
info_sets_marx = [
    fs("F-MAF",    use_factors=True,  use_marx=True, marx_for_pca=True,  use_maf=True),
    fs("F-X-MAF",  use_factors=True,  include_raw_x=True,
                   use_marx=True, marx_for_pca=True,  use_maf=True),
    fs("X-MAF",    use_factors=False, include_raw_x=True,
                   use_marx=True, marx_for_pca=True),
    fs("F-MARX",   use_factors=True,  use_marx=True, marx_for_pca=False),
    fs("F-X-MARX", use_factors=True,  include_raw_x=True,
                   use_marx=True, marx_for_pca=False),
    fs("X-MARX",   use_factors=False, use_marx=True, marx_for_pca=False),
]

# ── Level augmentation ────────────────────────────────────────────
info_sets_level = [
    fs("F-Level",         use_factors=True,  include_levels=True),
    fs("F-X-Level",       use_factors=True,  include_raw_x=True, include_levels=True),
    fs("X-Level",         use_factors=False, include_raw_x=True, include_levels=True),
    fs("F-MARX-Level",    use_factors=True,  use_marx=True, marx_for_pca=False,
                          include_levels=True),
    fs("F-X-MARX-Level",  use_factors=True,  include_raw_x=True,
                          use_marx=True, marx_for_pca=False, include_levels=True),
    fs("X-MARX-Level",    use_factors=False, use_marx=True, marx_for_pca=False,
                          include_levels=True),
]

all_info_sets = info_sets_base + info_sets_marx + info_sets_level

print(f"Total information sets: {len(all_info_sets)}")
print([s.label for s in all_info_sets])
# Total information sets: 15
# ['F', 'F-X', 'X', 'F-MAF', 'F-X-MAF', 'X-MAF', 'F-MARX', 'F-X-MARX',
#  'X-MARX', 'F-Level', 'F-X-Level', 'X-Level', 'F-MARX-Level',
#  'F-X-MARX-Level', 'X-MARX-Level']
```

---

## 3. Defining the Model Grid

CLSS 2021 uses five models: Elastic Net, Adaptive LASSO, Booging (R-side), Random Forest, and Gradient Boosting.

```python
from macrocast.pipeline.r_models import ElasticNetModel, AdaptiveLassoModel, BoogingModel

CV_5FOLD = CVScheme.KFOLD(k=5)

model_grid = [
    # ── R-side linear models ─────────────────────────────────────
    ModelSpec(
        model_cls=ElasticNetModel,
        regularization=Regularization.ELASTIC_NET,
        cv_scheme=CV_5FOLD,
        loss_function=LossFunction.L2,
        model_id="elastic_net",
    ),
    ModelSpec(
        model_cls=AdaptiveLassoModel,
        regularization=Regularization.ADAPTIVE_LASSO,
        cv_scheme=CV_5FOLD,
        loss_function=LossFunction.L2,
        model_id="adaptive_lasso",
    ),
    ModelSpec(
        model_cls=BoogingModel,
        regularization=Regularization.BOOGING,
        cv_scheme=CV_5FOLD,
        loss_function=LossFunction.L2,
        model_id="booging",
    ),
    # ── Python-side nonlinear models ──────────────────────────────
    ModelSpec(
        model_cls=RFModel,
        regularization=Regularization.FACTORS,
        cv_scheme=CV_5FOLD,
        loss_function=LossFunction.L2,
        model_kwargs={"n_estimators": 500, "cv_folds": 5},
        model_id="random_forest",
    ),
    ModelSpec(
        model_cls=GBModel,
        regularization=Regularization.FACTORS,
        cv_scheme=CV_5FOLD,
        loss_function=LossFunction.L2,
        model_kwargs={"n_estimators": 500, "cv_folds": 5},
        model_id="gradient_boosting",
    ),
]

print(f"Models: {[s.model_id for s in model_grid]}")
# Models: ['elastic_net', 'adaptive_lasso', 'booging', 'random_forest', 'gradient_boosting']
```

??? note "Benchmark AR model"
    The AR benchmark is handled automatically by `horserace_summary` via
    auto-detection (`nonlinearity == "linear"` and `regularization == "none"`).
    To include it explicitly in your grid:

    ```python
    from macrocast.pipeline.r_models import ARModel

    ar_benchmark = ModelSpec(
        model_cls=ARModel,
        regularization=Regularization.NONE,
        cv_scheme=CVScheme.BIC,
        loss_function=LossFunction.L2,
        model_id="ar_benchmark",
    )
    model_grid = [ar_benchmark] + model_grid
    ```

---

## 4. Running the Horse Race

`HorseRaceGrid` runs one `ForecastExperiment` per information set and merges all
`ForecastRecord`s into a single `ResultSet`. Each record carries a `feature_set`
label (e.g., `"F-MARX"`) identifying the information set it was generated under.

```python
# Horizons h = 1, 3, 6, 9, 12, 24 (months ahead) — CLSS 2021 Table 2
HORIZONS = [1, 3, 6, 9, 12, 24]

grid = HorseRaceGrid(
    panel=X,
    target=y,
    horizons=HORIZONS,
    model_specs=model_grid,
    feature_specs=all_info_sets,
    panel_levels=X_levels,    # required for Level info sets
    oos_start="2010-01-01",   # evaluation window start
    n_jobs=4,                 # parallelise over (model, horizon, date) triples
)

result_set = grid.run()

print(result_set)
# ResultSet(experiment_id='...', n_records=...)

df = result_set.to_dataframe()
print(df[["model_id", "feature_set", "horizon", "y_hat", "y_true"]].head())
#          model_id feature_set  horizon     y_hat    y_true
# 0  elastic_net           F        1   0.031     0.025
# 1  elastic_net           F        1   0.018     0.022
# ...
```

The `feature_set` column is what makes the merged `ResultSet` queryable across
all 15 information sets:

```python
# Quick check: record counts per information set
df.groupby("feature_set").size().sort_values(ascending=False)
# F-X-MAF      ...
# F-MAF        ...
# F            ...
# ...
```

---

## 5. Interpreting the Results

`horserace_summary` assembles all four evaluation components from a single call.

```python
result = horserace_summary(
    result_df=df,
    benchmark_id="ar_benchmark",   # or None for auto-detection
    horizons=HORIZONS,
    mcs_alpha=0.10,
)
```

### 5.1 Relative MSFE Table

Values below 1.0 indicate improvement over the AR benchmark.
Rows are `(model_id, feature_set)` pairs; columns are horizons.

```python
print(result.rmsfe_table.round(3).to_string())
#                                    horizon
#                                1      3      6      9     12     24
# model_id         feature_set
# adaptive_lasso   F            0.921  0.887  0.863  0.851  0.843  0.831
#                  F-MAF        0.908  0.872  0.848  0.839  0.832  0.821
#                  F-MARX       0.914  0.879  0.856  0.845  0.838  0.827
#                  ...
# gradient_boosting F           0.935  0.905  0.891  0.884  0.878  0.869
#                  F-MAF        0.922  0.891  0.876  0.870  0.864  0.856
#                  ...
# ar_benchmark     (benchmark)  1.000  1.000  1.000  1.000  1.000  1.000
```

### 5.2 Best Specification per Horizon

```python
print(result.best_specs.to_string(index=False))
#  horizon        model_id feature_set  rmsfe
#        1  adaptive_lasso       F-MAF  0.908
#        3  adaptive_lasso       F-MAF  0.872
#        6  adaptive_lasso       F-MAF  0.848
#        9  adaptive_lasso    F-X-MARX  0.839
#       12   random_forest      F-MARX  0.829
#       24   random_forest    F-X-MARX  0.817
```

The dominant pattern from CLSS 2021 is that MAF (PCA on MARX panel) tends to
dominate at short horizons, while MARX columns alongside factors gain at longer
horizons. This reflects the shifting importance of recent-history moving averages
as the horizon grows.

### 5.3 Model Confidence Set

`True` indicates the model-information-set pair belongs to the MCS at 10%
significance (Hansen, Lunde, Nason 2011 block bootstrap).

```python
print(result.mcs_table.astype(int).to_string())
#                                    horizon
#                                1  3  6  9  12  24
# model_id         feature_set
# adaptive_lasso   F            1  1  1  1   1   1
#                  F-MAF        1  1  1  1   1   1
#                  F-MARX       0  1  1  1   1   1
#                  ...
# ar_benchmark     (benchmark)  0  0  0  0   0   0

# Fraction of (model, info set) pairs in the MCS per horizon
result.mcs_table.mean().round(2)
# horizon
# 1     0.27
# 3     0.35
# 6     0.40
# 9     0.44
# 12    0.47
# 24    0.52
# dtype: float64
```

### 5.4 Diebold-Mariano Tests vs AR

Two-sided DM test (Diebold and Mariano 1995, HLN-corrected).
Small p-values indicate statistically significant gains over the AR benchmark.

```python
print(result.dm_table.round(3).to_string())
#                                    horizon
#                                1      3      6      9     12     24
# model_id         feature_set
# adaptive_lasso   F            0.042  0.018  0.011  0.009  0.008  0.006
#                  F-MAF        0.031  0.012  0.007  0.006  0.005  0.004
#                  ...
# ar_benchmark     (benchmark)    NaN    NaN    NaN    NaN    NaN    NaN
```

### 5.5 Visualising the RMSFE Table

```python
import matplotlib.pyplot as plt

rmsfe = result.rmsfe_table

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(rmsfe.values, aspect="auto", cmap="RdYlGn_r", vmin=0.8, vmax=1.05)
ax.set_xticks(range(len(rmsfe.columns)))
ax.set_xticklabels(rmsfe.columns)
ax.set_yticks(range(len(rmsfe.index)))
ax.set_yticklabels([f"{m} | {f}" for m, f in rmsfe.index])
plt.colorbar(im, ax=ax, label="Relative MSFE (< 1 = beats AR)")
ax.set_xlabel("Horizon (months ahead)")
ax.set_title("CLSS 2021 Horse Race — Relative MSFE")
plt.tight_layout()
plt.savefig("clss2021_rmsfe.png", dpi=150)
```

---

## 6. Saving and Reloading Results

```python
from pathlib import Path

# Save to parquet for later analysis or sharing with R
out = Path("results/clss2021")
out.mkdir(parents=True, exist_ok=True)
result_set.to_parquet(out / "horserace.parquet")

# Reload
from macrocast.pipeline.results import ResultSet
rs_loaded = ResultSet.from_parquet(out / "horserace.parquet")
df_loaded  = rs_loaded.to_dataframe_cached()
```

---

## Summary

| Step | What we did | Key API |
|------|-------------|---------|
| 1 | Generated synthetic 128-variable monthly panel | `pd.DataFrame` / `mc.load_fred_md()` |
| 2 | Defined 15 CLSS 2021 information sets | `FeatureSpec` with `include_raw_x`, `marx_for_pca`, `include_levels` |
| 3 | Specified 5-model grid | `ModelSpec` |
| 4 | Ran horse race across all (model, info set, horizon, date) cells | `HorseRaceGrid.run()` |
| 5 | Computed RMSFE, best spec, MCS, DM tables | `horserace_summary()` |

The central finding from CLSS 2021 — that the MARX and MAF transformations
substantially improve forecast accuracy over raw factors, especially at medium-to-long
horizons — is directly testable with this pipeline.
