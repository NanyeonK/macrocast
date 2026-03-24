"""macrocast.data — Raw data acquisition (FRED-MD, FRED-QD, FRED-SD)."""

from macrocast.data.fred_md import load_fred_md
from macrocast.data.fred_qd import load_fred_qd
from macrocast.data.fred_sd import load_fred_sd
from macrocast.data.merge import MergeResult, merge_macro_frames
from macrocast.data.schema import MacroFrame, MacroFrameMetadata, VariableMetadata
from macrocast.data.vintages import (
    RealTimePanel,
    list_available_vintages,
    load_vintage_panel,
)

__all__ = [
    # Loaders
    "load_fred_md",
    "load_fred_qd",
    "load_fred_sd",
    # Multi-dataset merge
    "merge_macro_frames",
    "MergeResult",
    # Core container
    "MacroFrame",
    "MacroFrameMetadata",
    "VariableMetadata",
    # Vintages
    "list_available_vintages",
    "load_vintage_panel",
    "RealTimePanel",
]
