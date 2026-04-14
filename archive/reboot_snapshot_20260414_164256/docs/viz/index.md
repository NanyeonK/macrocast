# Visualization

The `macrocast.viz` module provides publication-quality plots for forecast evaluation
and model interpretation results. All functions return `matplotlib` figure objects.

## Available Plots

| Function | Purpose |
|----------|---------|
| `marginal_effect_plot` | Bar chart of marginal OOS-R² contributions by information set |
| `variable_importance_plot` | Grouped variable importance from tree models |
| `cumulative_squared_error_plot` | Cumulative squared error difference over time |
| `plot_rmsfe_heatmap` | RMSFE heatmap across models and horizons |
| `plot_horizon_lines` | Line plot of relative MSFE by forecast horizon |
| `plot_mcs_membership` | MCS membership indicators across models |

## Quick Start

```python
from macrocast.viz.plots import plot_rmsfe_heatmap

fig = plot_rmsfe_heatmap(results, benchmark="AR")
fig.savefig("rmsfe_heatmap.pdf", bbox_inches="tight")
```
