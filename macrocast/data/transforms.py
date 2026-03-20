"""Transformation codes (tcode) for FRED-MD/QD series.

Implements the seven stationarity transformations defined in
McCracken & Ng (2016), Table 1. Each transformation maps a raw
level series to a (approximately) stationary series.
"""

from __future__ import annotations

import warnings
from enum import IntEnum

import numpy as np
import pandas as pd


class TransformCode(IntEnum):
    """Enumeration of the seven McCracken-Ng transformation codes."""

    LEVEL = 1
    DIFF = 2
    DIFF2 = 3
    LOG = 4
    LOG_DIFF = 5
    LOG_DIFF2 = 6
    DELTA_RATIO = 7


def apply_tcode(series: pd.Series, tcode: int) -> pd.Series:
    """Apply a single transformation code to a pandas Series.

    Parameters
    ----------
    series : pd.Series
        Raw level series. Index is typically a DatetimeIndex.
    tcode : int
        Transformation code 1-7 as defined by McCracken & Ng (2016).

    Returns
    -------
    pd.Series
        Transformed series with the same index. Leading observations
        that cannot be computed (due to lags or log of non-positive
        values) are set to NaN.

    Raises
    ------
    ValueError
        If *tcode* is not in 1-7.
    """
    tc = int(tcode)
    if tc not in range(1, 8):
        raise ValueError(f"tcode must be 1-7, got {tcode}")

    x = series.copy().astype(float)

    if tc == TransformCode.LEVEL:
        # y_t = x_t
        return x

    if tc == TransformCode.DIFF:
        # y_t = x_t - x_{t-1}
        return x.diff()

    if tc == TransformCode.DIFF2:
        # y_t = x_t - 2*x_{t-1} + x_{t-2}
        return x.diff().diff()

    if tc == TransformCode.LOG:
        # y_t = ln(x_t)
        if (x <= 0).any():
            warnings.warn(
                f"Series '{series.name}' contains non-positive values; "
                "setting to NaN before log transform.",
                stacklevel=2,
            )
        log_x = x.copy()
        log_x[log_x <= 0] = np.nan
        return np.log(log_x)

    if tc == TransformCode.LOG_DIFF:
        # y_t = ln(x_t) - ln(x_{t-1}) = Δln(x_t)
        if (x <= 0).any():
            warnings.warn(
                f"Series '{series.name}' contains non-positive values; "
                "setting to NaN before log transform.",
                stacklevel=2,
            )
        log_x = x.copy()
        log_x[log_x <= 0] = np.nan
        return np.log(log_x).diff()

    if tc == TransformCode.LOG_DIFF2:
        # y_t = Δ²ln(x_t)
        if (x <= 0).any():
            warnings.warn(
                f"Series '{series.name}' contains non-positive values; "
                "setting to NaN before log transform.",
                stacklevel=2,
            )
        log_x = x.copy()
        log_x[log_x <= 0] = np.nan
        return np.log(log_x).diff().diff()

    if tc == TransformCode.DELTA_RATIO:
        # y_t = Δ(x_t / x_{t-1} - 1) = (x_t/x_{t-1}) - (x_{t-1}/x_{t-2})
        ratio = x / x.shift(1)
        return ratio.diff()

    # unreachable
    raise ValueError(f"Unhandled tcode: {tcode}")


def apply_tcodes(
    df: pd.DataFrame,
    tcodes: dict[str, int],
) -> pd.DataFrame:
    """Apply transformation codes to all columns of a DataFrame.

    Columns not present in *tcodes* are passed through unchanged
    (equivalent to tcode=1).

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with variable names as columns.
    tcodes : dict[str, int]
        Mapping from column name to transformation code (1-7).

    Returns
    -------
    pd.DataFrame
        Transformed DataFrame with the same shape and index.
    """
    result = df.copy()
    for col in df.columns:
        tc = tcodes.get(col, TransformCode.LEVEL)
        result[col] = apply_tcode(df[col], tc)
    return result


def inverse_tcode(series: pd.Series, tcode: int, initial: pd.Series) -> pd.Series:
    """Invert a transformation code to recover level series.

    Parameters
    ----------
    series : pd.Series
        Transformed series to invert.
    tcode : int
        The transformation code that was originally applied.
    initial : pd.Series
        Initial values needed for reconstruction (level at t=0 etc.).

    Returns
    -------
    pd.Series
        Reconstructed level series.

    Raises
    ------
    NotImplementedError
        Always. Inversion is deferred to v2.
    """
    raise NotImplementedError(
        "inverse_tcode is not implemented in v1. Deferred to a future release."
    )
