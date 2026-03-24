"""Data transformations for macrocast preprocessing.

Two families of transforms:

1. **Stationarity transforms** (McCracken & Ng 2016, Table 1)
   Applied series-by-series to convert raw FRED-MD/QD levels to
   (approximately) stationary series before any modelling step.

   Functions: :func:`apply_tcode`, :func:`apply_tcodes`

2. **Panel feature transforms** (Coulombe et al. 2021)
   Applied to the full stationary predictor panel X to build
   richer feature representations.  All functions operate on
   2-D NumPy arrays (T × K) and return NumPy arrays.

   - :func:`apply_marx`     — Moving Average Rotation of X (Eq. 7–9)
   - :func:`apply_maf`      — Moving Average Factors (PCA on MARX panel)
   - :func:`apply_x_factors` — Standard diffusion factors (PCA on X)
   - :func:`apply_pca`      — General-purpose PCA wrapper

3. **Cycle/trend decomposition**
   Applied to individual time series to separate trend from cycle.

   - :func:`apply_hamilton_filter` — Hamilton (2018) regression filter
"""

from __future__ import annotations

import warnings
from enum import IntEnum

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# 1. Stationarity transforms — McCracken & Ng (2016)
# ---------------------------------------------------------------------------


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
        return x

    if tc == TransformCode.DIFF:
        return x.diff()

    if tc == TransformCode.DIFF2:
        return x.diff().diff()

    if tc == TransformCode.LOG:
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
        ratio = x / x.shift(1)
        return ratio.diff()

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

    Raises
    ------
    NotImplementedError
        Deferred to a future release.
    """
    raise NotImplementedError(
        "inverse_tcode is not implemented in v1. Deferred to a future release."
    )


# ---------------------------------------------------------------------------
# 2. Panel feature transforms — Coulombe et al. (2021)
# ---------------------------------------------------------------------------


def apply_marx(
    X: NDArray[np.floating],
    p: int,
    scale: bool = True,
) -> NDArray[np.floating]:
    """Moving Average Rotation of X (MARX) — Coulombe et al. (2021) Eq. 7–9.

    For variable k and moving-average order p':

        MARX_{t,k,p'} = (1/p') * sum_{j=0}^{p'-1} X_{t-j, k}

    All p' = 1, ..., p moving-average orders are stacked column-wise,
    yielding a panel with K*p columns.  The first p-1 rows (where early
    lags are unavailable) are dropped, so the returned array has T-p+1 rows.

    Parameters
    ----------
    X : array of shape (T, K)
        Stationary predictor panel.  Should be standardised before calling
        (or pass ``scale=True`` to standardise internally).
    p : int
        Maximum moving-average order (P_MARX in the paper; default 12).
    scale : bool, default True
        Standardise each column of X to zero mean / unit variance before
        computing moving averages.  Recommended when columns have
        different scales.

    Returns
    -------
    X_marx : array of shape (T - p + 1, K * p)
        Columns ordered as [MA_1_var0, ..., MA_1_varK-1,
                             MA_2_var0, ..., MA_p_varK-1].
    """
    X = np.asarray(X, dtype=float)
    if scale:
        X = StandardScaler().fit_transform(X)

    T, K = X.shape
    cs = np.zeros((T + 1, K), dtype=float)
    cs[1:] = np.cumsum(X, axis=0)
    parts: list[NDArray[np.floating]] = []
    for lag in range(1, p + 1):
        ma = (cs[p : T + 1] - cs[p - lag : T + 1 - lag]) / lag
        parts.append(ma)
    return np.concatenate(parts, axis=1)  # (T - p + 1, K * p)


def apply_maf(
    X: NDArray[np.floating],
    k: int,
    p: int,
    scale: bool = True,
) -> NDArray[np.floating]:
    """Moving Average Factors (MAF) — PCA on the MARX-transformed panel.

    Computes the MARX panel then extracts the first *k* principal components.
    This is the ``factor_type="MARX"`` information set in CLSS 2021 Table 1.

    Parameters
    ----------
    X : array of shape (T, K)
        Stationary predictor panel.
    k : int
        Number of factors to extract (K in the paper; default 8).
    p : int
        MARX lag order (P_MARX in the paper; default 12).
    scale : bool, default True
        Standardise X before MARX computation.

    Returns
    -------
    F_maf : array of shape (T - p + 1, k)
        MAF factor scores.
    """
    X_marx = apply_marx(X, p=p, scale=scale)
    n_act = min(k, X_marx.shape[1], X_marx.shape[0] - 1)
    pca = PCA(n_components=n_act)
    return pca.fit_transform(X_marx)


def apply_x_factors(
    X: NDArray[np.floating],
    k: int,
    scale: bool = True,
) -> NDArray[np.floating]:
    """Standard diffusion factors — PCA on the stationary predictor panel.

    This is the ``factor_type="X"`` information set in CLSS 2021 Table 1.

    Parameters
    ----------
    X : array of shape (T, K)
        Stationary predictor panel.
    k : int
        Number of factors to extract.
    scale : bool, default True
        Standardise X before PCA.

    Returns
    -------
    F : array of shape (T, k)
        Factor scores.
    """
    X = np.asarray(X, dtype=float)
    if scale:
        X = StandardScaler().fit_transform(X)
    n_act = min(k, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_act)
    return pca.fit_transform(X)


def apply_pca(
    X: NDArray[np.floating],
    k: int,
    scale: bool = True,
) -> NDArray[np.floating]:
    """General-purpose PCA: extract the first *k* principal components.

    Thin wrapper around :func:`apply_x_factors` with a name that is not
    tied to any specific information-set labelling convention.

    Parameters
    ----------
    X : array of shape (T, K)
        Input panel.
    k : int
        Number of components to retain.
    scale : bool, default True
        Standardise each column before PCA.

    Returns
    -------
    F : array of shape (T, k)
        Principal component scores.
    """
    return apply_x_factors(X, k=k, scale=scale)


# ---------------------------------------------------------------------------
# 3. Cycle / trend decomposition
# ---------------------------------------------------------------------------


def apply_hamilton_filter(
    series: pd.Series | NDArray[np.floating],
    h: int = 8,
    p: int = 4,
) -> tuple[NDArray[np.floating] | pd.Series, NDArray[np.floating] | pd.Series]:
    """Hamilton (2018) regression-based trend/cycle decomposition.

    Decomposes *series* by regressing y_{t+h} on a constant and p lags
    [y_t, y_{t-1}, ..., y_{t-p+1}] via OLS.  The fitted value is the
    trend component; the residual is the cycle.

    Hamilton (2018) recommends h=8, p=4 for quarterly data (GDP-like
    series) and h=24, p=12 for monthly series to approximate a 2-year
    horizon.

    Parameters
    ----------
    series : pd.Series or array of shape (T,)
        Level or stationary series to filter.
    h : int, default 8
        Lead horizon: dependent variable is y_{t+h}.
    p : int, default 4
        Number of autoregressive lags used as regressors.

    Returns
    -------
    trend : same type as *series*
        OLS fitted values placed at the y_{t+h} positions.
        The first h+p-1 observations are NaN.
    cycle : same type as *series*
        OLS residuals (y_{t+h} - trend).
        The first h+p-1 observations are NaN.

    Raises
    ------
    ValueError
        If the series is too short to form any valid regression row.

    References
    ----------
    Hamilton, J. D. (2018). Why you should never use the Hodrick-Prescott
    filter. *Review of Economics and Statistics*, 100(5), 831-843.
    """
    is_series = isinstance(series, pd.Series)
    idx = series.index if is_series else None
    y = np.asarray(series, dtype=float)
    T = len(y)

    # Valid rows: t in [p-1, T-1-h], giving y_{t+h} in [p-1+h, T-1].
    n = T - h - p + 1
    if n <= p + 1:
        raise ValueError(
            f"Series length {T} is too short for h={h}, p={p}. "
            f"Need at least {h + p + p + 2} observations."
        )

    t_vals = np.arange(p - 1, T - h)          # shape (n,)
    Y = y[t_vals + h]                          # dependent variable
    lag_cols = np.stack([y[t_vals - j] for j in range(p)], axis=1)  # (n, p)
    Z = np.column_stack([np.ones(n), lag_cols])  # (n, p+1)

    beta, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
    fitted = Z @ beta
    residuals = Y - fitted

    # Place into full-length arrays; positions before p-1+h are NaN.
    trend_arr = np.full(T, np.nan)
    cycle_arr = np.full(T, np.nan)
    start = (p - 1) + h
    trend_arr[start:] = fitted
    cycle_arr[start:] = residuals

    if is_series:
        return (
            pd.Series(trend_arr, index=idx, name="trend"),
            pd.Series(cycle_arr, index=idx, name="cycle"),
        )
    return trend_arr, cycle_arr
