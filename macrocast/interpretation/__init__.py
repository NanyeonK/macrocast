"""macrocast.interpretation — Model interpretation and explainability."""

from macrocast.interpretation.dual import (
    effective_history_length,
    krr_dual_weights,
    nn_dual_weights,
    top_analogies,
    tree_dual_weights,
)
from macrocast.interpretation.marginal import (
    MarginalEffect,
    marginal_contribution,
    marginal_contribution_all,
    oos_r2_panel,
)
from macrocast.interpretation.pbsv import (
    compute_pbsv,
    model_accordance_score,
    oshapley_vi,
)
from macrocast.interpretation.variable_importance import (
    CLSS_VI_GROUPS,
    average_vi_by_horizon,
    extract_vi_dataframe,
    vi_by_group,
)

__all__ = [
    # dual weights
    "krr_dual_weights",
    "tree_dual_weights",
    "nn_dual_weights",
    "effective_history_length",
    "top_analogies",
    # marginal contribution
    "oos_r2_panel",
    "MarginalEffect",
    "marginal_contribution",
    "marginal_contribution_all",
    # PBSV
    "oshapley_vi",
    "compute_pbsv",
    "model_accordance_score",
    # variable importance
    "CLSS_VI_GROUPS",
    "extract_vi_dataframe",
    "vi_by_group",
    "average_vi_by_horizon",
]
