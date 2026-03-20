# Roadmap

---

## v0.1.0 — Data Layer (Current)

Released: March 2026

**Completed:**

- `load_fred_md()` — FRED-MD monthly loader with caching, vintage support, and tcode parsing
- `load_fred_qd()` — FRED-QD quarterly loader
- `load_fred_sd()` — FRED-SD state-level loader (Excel, openpyxl)
- `MacroFrame` — immutable panel container with fluent API
- `TransformCode` enum and `apply_tcode` / `apply_tcodes` — all seven McCracken-Ng transformations
- `classify_missing` / `handle_missing` — three missing-type classification and five treatment methods
- `list_available_vintages` / `load_vintage_panel` / `RealTimePanel` — vintage enumeration and multi-vintage loading
- `macrocast.utils.cache` — local file caching with age-based expiry

**Test coverage:** 107 tests passing. All Layer 1 modules covered.

---

## v0.2.0 — Forecasting Pipeline (Planned)

Target: April-May 2026

The forecasting pipeline implements the four-component decomposition framework of Coulombe et al. (2020).

**Planned:**

- `ForecastExperiment` — main orchestrator for factorial forecast experiments
- `MacrocastEstimator` interface — scikit-learn-compatible model adapter
- Component definitions: `Nonlinearity`, `Regularization`, `CVScheme`, `LossFunction`
- Built-in model zoo: AR(p), Ridge, Lasso, Elastic Net, Random Forest, feedforward NN
- Direct and iterated multi-step forecasting
- Expanding and rolling window evaluation schemes
- Parallel execution via `joblib`
- YAML-based experiment configuration

---

## v0.3.0 — Evaluation Layer (Planned)

Target: May-June 2026

**Planned:**

- `decompose()` — treatment-effect decomposition engine
- `RegimeEvaluator` — regime-conditional MSFE with quantile/Markov-switching thresholds
- `model_confidence_set()` — Hansen, Lunde, and Nason (2011) MCS with block bootstrap
- MSFE, relative MSFE, MAE, directional accuracy metrics
- Cumulative squared forecast error (CSFE) over time
- Visualization: waterfall plots, CSFE paths, regime heatmaps, horizon profiles

---

## v0.4.0 — R Companion and CLI (Planned)

Target: June-July 2026

**Planned:**

- `macrocastR` R package mirroring the Python API
- Command-line interface (`macrocast run`, `macrocast data download`, `macrocast report`)
- Full documentation and empirical illustration replicating Coulombe et al. (2020)

---

## Paper Submission

Target: August 2026

Submission to *International Journal of Forecasting* Special Issue "Advances in Open Source Forecasting Software".

---

## Known Limitations in v0.1

- EM-based imputation (`method="em"`) raises `NotImplementedError`. Not required for the core decomposition analysis and will not be implemented.
- `RealTimePanel` provides basic access only. Pseudo-out-of-sample alignment utilities are planned for v0.2.
- The `target` parameter in `load_vintage_panel` is accepted but not yet applied.
