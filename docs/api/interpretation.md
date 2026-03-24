# macrocast.interpretation

Full API reference for the interpretation layer modules.

---

## macrocast.interpretation.dual

::: macrocast.interpretation.dual
    options:
      members:
        - krr_dual_weights
        - tree_dual_weights
        - nn_dual_weights
        - effective_history_length
        - top_analogies
      show_root_heading: true

---

## macrocast.interpretation.pbsv

::: macrocast.interpretation.pbsv
    options:
      members:
        - oshapley_vi
        - compute_pbsv
        - model_accordance_score
      show_root_heading: true

---

## macrocast.interpretation.marginal

::: macrocast.interpretation.marginal
    options:
      members:
        - marginal_contribution
        - marginal_contribution_all
        - MarginalEffect
        - oos_r2_panel
      show_root_heading: true

---

## macrocast.interpretation.variable_importance

::: macrocast.interpretation.variable_importance
    options:
      members:
        - extract_vi_dataframe
        - vi_by_group
        - average_vi_by_horizon
      show_root_heading: true
