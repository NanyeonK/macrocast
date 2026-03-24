"""macrocast.preprocessing — Data transforms and panel preprocessing."""

from macrocast.preprocessing.missing import (
    classify_missing,
    detect_missing_type,
    handle_missing,
)
from macrocast.preprocessing.panel import (
    CustomTransform,
    DemeanTransform,
    DropTransform,
    HPFilterTransform,
    PanelTransformer,
    StandardizeTransform,
    WinsorizeTransform,
)
from macrocast.preprocessing.transforms import (
    TransformCode,
    apply_hamilton_filter,
    apply_maf,
    apply_marx,
    apply_pca,
    apply_tcode,
    apply_tcodes,
    apply_x_factors,
    inverse_tcode,
)

__all__ = [
    # stationarity transforms (McCracken & Ng 2016)
    "TransformCode",
    "apply_tcode",
    "apply_tcodes",
    "inverse_tcode",
    # panel feature transforms (Coulombe et al. 2021)
    "apply_marx",
    "apply_maf",
    "apply_x_factors",
    "apply_pca",
    # cycle / trend decomposition
    "apply_hamilton_filter",
    # missing value handling
    "detect_missing_type",
    "classify_missing",
    "handle_missing",
    # panel preprocessing
    "PanelTransformer",
    "WinsorizeTransform",
    "DemeanTransform",
    "HPFilterTransform",
    "StandardizeTransform",
    "CustomTransform",
    "DropTransform",
]
