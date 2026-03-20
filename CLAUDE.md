# macrocast

Decomposing ML Forecast Gains in Macroeconomic Forecasting.
Python (primary) + R companion package. Target: IJF Special Issue (deadline 2026-08-31).

## Architecture

Three-layer design: Data → Pipeline → Evaluation.

```
macrocast/
├── macrocast/
│   ├── data/           # Layer 1: FRED-MD/QD/SD download, transform, vintage mgmt
│   ├── pipeline/       # Layer 2: Forecasting experiment with 4-component decomposition
│   ├── evaluation/     # Layer 3: Metrics, regime-conditional eval, MCS, decomposition
│   └── utils/          # Parallel, cache, config
├── macrocastR/         # R companion (separate package)
├── tests/
├── examples/
├── docs/
└── paper/              # LaTeX manuscript
```

## Tech Stack

- Python: pandas, numpy, scikit-learn, statsmodels, lightgbm, pytorch, joblib
- R: data.table, glmnet, ranger, tidymodels, ggplot2
- Build: `uv` for Python env, `renv` for R
- Test: pytest (Python), testthat (R)
- Lint: ruff (Python), lintr (R)
- Docs: mkdocs-material

## Development Environment Versions (as of 2026-03-19)

Exact pins are in `uv.lock`. The versions below were used during Layer 1 development.

| Package | Version |
|---------|---------|
| Python | 3.11.13 |
| pandas | 3.0.1 |
| numpy | 2.4.3 |
| scikit-learn | 1.8.0 |
| statsmodels | 0.14.6 |
| scipy | 1.17.1 |
| requests | 2.32.5 |
| openpyxl | 3.1.5 |
| pyyaml | 6.0.3 |
| joblib | 1.5.3 |
| tqdm | 4.67.3 |

## Key Commands

```bash
# Python
uv run pytest tests/ -v                    # run tests
uv run ruff check macrocast/              # lint
uv run ruff format macrocast/             # format
uv run python -m macrocast.cli --help     # CLI

# R
Rscript -e "devtools::test('macrocastR')"
Rscript -e "lintr::lint_package('macrocastR')"
```

## Code Conventions

- Python: type hints on all public functions. Docstrings in NumPy style.
- Each module file: top-level docstring explaining purpose.
- No wildcard imports. Explicit is better than implicit.
- pandas: prefer `.loc[]` over chained indexing.
- Tests mirror source structure: `macrocast/data/fred_md.py` → `tests/data/test_fred_md.py`
- R: roxygen2 documentation. snake_case for functions.

## Data Layer Rules

- FRED-MD/QD CSVs: first row = transformation codes, second row = dates header. Handle this explicitly.
- FRED-SD: xlsx format with tabs per series. Must parse with openpyxl.
- All download functions must cache locally under `~/.macrocast/cache/` by default.
- Vintage identifiers follow YYYY-MM format (e.g., "2020-01").
- Transformation codes (tcode 1-7): implement exactly as McCracken & Ng (2016) specify.

## Pipeline Layer Rules

- The four decomposition components (nonlinearity, regularization, cv_scheme, loss_function) are enum-like objects, NOT string flags.
- ForecastExperiment is the main orchestrator. It takes a MacroFrame + config and produces a ResultSet.
- Two window concepts are distinct: inner CV loop (K-fold, POOS-CV, BIC) vs outer evaluation loop (expanding/rolling). Do not conflate them.
- All Python models implement MacrocastEstimator: `.fit(X, y)`, `.predict(X)`, `.nonlinearity_type`.
- LSTM and other sequence models implement SequenceEstimator: input X is (T, L, N) where L is lookback window.
- Multi-step forecasting: "direct" only in v1. Each horizon h trains a separate model on y_{t+h}. Iterated is out of scope.
- Expanding window is default for the outer evaluation loop. Rolling window is an option.
- R/Python split: linear regularized models (Ridge, LASSO, Adaptive LASSO, Group LASSO, Elastic Net, ARDI) run in macrocastR. Nonlinear models (KRR, SVR, RF, XGBoost, NN, LSTM) run in Python. Results shared via parquet under ~/.macrocast/results/{experiment_id}/.
- FeatureBuilder constructs Z_t from MacroFrame. For FACTORS/ARDI: Z_t = [PCA factors, AR lags of target]. For others: Z_t = [AR lags only]. p_y and p_f are tuning parameters selected by CVScheme.
- Benchmark is AR(p) with p selected by BIC. Data-rich linear baseline is ARDI.
- Group LASSO uses FRED variable groups (output_income, labor, housing, prices, money, interest_rates, stock_market) as the group structure.

## Evaluation Layer Rules

- Benchmark model is always AR(p) selected by BIC.
- Relative MSFE = model MSFE / AR MSFE. Values < 1 indicate improvement.
- MCS: implement Hansen, Lunde, Nason (2011) with block bootstrap.
- Regime indicators come from the same FRED dataset (e.g., VXOCLSX for uncertainty).

## Writing Style (for docstrings, README, paper)

- Academic econometrics tone. No marketing language.
- Minimal transitions. Direct statements.
- "We find that" for results. "Table N reports" for tables.
- End-of-sentence citations.
- No m-dashes, colons, or semicolons in extended prose.

## Git Workflow

- Branch naming: `feature/layer1-fred-md`, `fix/tcode-transform`, `docs/api-reference`
- Commit messages: imperative mood, max 72 chars first line
- No force-push to main
