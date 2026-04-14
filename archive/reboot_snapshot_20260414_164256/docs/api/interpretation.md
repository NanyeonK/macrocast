# macrocast.interpretation

`macrocast.interpretation` contains the public API for dual-weight analysis, marginal contribution analysis, PBSV-style attribution, and grouped variable-importance summaries.

## Import

```python
from macrocast.interpretation import krr_dual_weights, compute_pbsv, vi_by_group
```

## Dual-weight interpretation

Dual-weight exports:
- `krr_dual_weights`
- `tree_dual_weights`
- `nn_dual_weights`
- `effective_history_length`
- `top_analogies`

Purpose:
- analyze which historical observations dominate a forecast under dual-weight views

## Marginal contribution tools

Marginal-contribution exports:
- `oos_r2_panel`
- `MarginalEffect`
- `marginal_contribution`
- `marginal_contribution_all`

Purpose:
- measure contribution changes across model ingredients and information sets

## PBSV and Shapley-style tools

PBSV exports:
- `oshapley_vi`
- `compute_pbsv`
- `model_accordance_score`

Purpose:
- attribute out-of-sample predictive contribution in a performance-based way

## Variable-importance summaries

Variable-importance exports:
- `CLSS_VI_GROUPS`
- `extract_vi_dataframe`
- `vi_by_group`
- `average_vi_by_horizon`

Purpose:
- summarize model-native importance outputs into grouped and horizon-level reporting objects

## Related pages

- `User Guide > Stage 7`
- `API Reference > macrocast.viz`
