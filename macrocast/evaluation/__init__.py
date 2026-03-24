"""macrocast.evaluation — Statistical forecast evaluation."""

from macrocast.evaluation.combination import combine_forecasts
from macrocast.evaluation.cw import CWResult, cw_test
from macrocast.evaluation.decomposition import (
    DecompositionResult,
    decompose_treatment_effects,
)
from macrocast.evaluation.dm import DMResult, dm_test
from macrocast.evaluation.gw import GWResult, gw_test
from macrocast.evaluation.horserace import (
    HorseRaceResult,
    best_spec_table,
    dm_vs_benchmark_table,
    horserace_summary,
    mcs_membership_table,
    relative_msfe_table,
)
from macrocast.evaluation.mcs import MCSResult, mcs
from macrocast.evaluation.metrics import csfe, mae, msfe, oos_r2, relative_msfe
from macrocast.evaluation.regime import (
    RegimeResult,
    load_nber_recessions,
    regime_conditional_msfe,
)

__all__ = [
    # forecast combination
    "combine_forecasts",
    # metrics
    "msfe",
    "mae",
    "relative_msfe",
    "csfe",
    "oos_r2",
    # decomposition
    "decompose_treatment_effects",
    "DecompositionResult",
    # statistical tests
    "cw_test",
    "CWResult",
    "dm_test",
    "DMResult",
    "gw_test",
    "GWResult",
    # MCS
    "mcs",
    "MCSResult",
    # regime
    "regime_conditional_msfe",
    "RegimeResult",
    "load_nber_recessions",
    # horse race
    "HorseRaceResult",
    "relative_msfe_table",
    "best_spec_table",
    "mcs_membership_table",
    "dm_vs_benchmark_table",
    "horserace_summary",
]
