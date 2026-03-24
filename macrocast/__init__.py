"""macrocast: Decomposing ML Forecast Gains in Macroeconomic Forecasting.

Modules
-------
* ``macrocast.data``           — FRED-MD/QD/SD loaders + MacroFrame
* ``macrocast.preprocessing``  — tcode transforms, MARX/MAF, panel preprocessing
* ``macrocast.pipeline``       — ForecastExperiment, models, features
* ``macrocast.evaluation``     — MSFE, MCS, DM, CW statistical tests
* ``macrocast.interpretation`` — dual weights, PBSV, variable importance
* ``macrocast.viz``            — visualization
* ``macrocast.utils``          — registry, LaTeX export, cache
* ``macrocast.replication``    — paper-specific helpers (CLSS 2021 etc.)

Quick start::

    from macrocast import load_fred_md, ForecastExperiment
"""

__version__ = "0.1.0"

from macrocast.data import (
    MacroFrame,
    MacroFrameMetadata,
    MergeResult,
    RealTimePanel,
    VariableMetadata,
    list_available_vintages,
    load_fred_md,
    load_fred_qd,
    load_fred_sd,
    load_vintage_panel,
    merge_macro_frames,
)
from macrocast.pipeline import (
    FeatureSpec,
    ForecastExperiment,
    ForecastRecord,
    ModelSpec,
    ResultSet,
)
from macrocast.preprocessing import (
    TransformCode,
    apply_hamilton_filter,
    apply_maf,
    apply_marx,
    apply_pca,
    apply_tcode,
    apply_tcodes,
    apply_x_factors,
    classify_missing,
    handle_missing,
)

__all__ = [
    "__version__",
    # data
    "load_fred_md",
    "load_fred_qd",
    "load_fred_sd",
    "MacroFrame",
    "MacroFrameMetadata",
    "VariableMetadata",
    "list_available_vintages",
    "load_vintage_panel",
    "RealTimePanel",
    "merge_macro_frames",
    "MergeResult",
    # preprocessing
    "TransformCode",
    "apply_tcode",
    "apply_tcodes",
    "apply_marx",
    "apply_maf",
    "apply_x_factors",
    "apply_pca",
    "apply_hamilton_filter",
    "classify_missing",
    "handle_missing",
    # pipeline
    "ForecastExperiment",
    "ModelSpec",
    "FeatureSpec",
    "ResultSet",
    "ForecastRecord",
]
