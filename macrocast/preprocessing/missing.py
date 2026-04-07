"""Missing value classification and handling for macroeconomic panels.

Distinguishes between three structural missing patterns common in
FRED-MD/QD:

- **leading**: NaN at the start of a series (series not yet published
  at sample start).
- **trailing**: NaN at the end (series discontinued or delayed).
- **intermittent**: NaN in the middle of a series.

The ``handle_missing`` function dispatches to the treatment method
requested by the user.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def remove_outliers_iqr(
    df: pd.DataFrame,
    threshold: float = 10.0,
) -> pd.DataFrame:
    """Replace outliers with NaN using the McCracken-Ng / fbi convention.

    A value is an outlier if it deviates from the column median by more
    than *threshold* times the interquartile range (IQR).  Matches the
    logic of ``fbi::rm_outliers.fredmd`` exactly.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data. Only numeric columns are processed.
    threshold : float, default 10.0
        Multiplier applied to IQR. The fbi / McCracken-Ng default is 10.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with outliers set to NaN.
    """
    out = df.copy()
    for col in df.select_dtypes(include="number").columns:
        s = df[col]
        median = s.median(skipna=True)
        q25 = s.quantile(0.25)
        q75 = s.quantile(0.75)
        iqr = q75 - q25
        if iqr == 0:
            continue
        outlier_mask = (s - median).abs() > threshold * iqr
        out.loc[outlier_mask, col] = np.nan
    return out


def detect_missing_type(series: pd.Series) -> dict[str, int | float]:
    """Classify missing observations in a single series."""
    n_total = len(series)
    n_missing = int(series.isna().sum())

    first_valid = series.first_valid_index()
    last_valid = series.last_valid_index()

    if first_valid is None:
        return {
            "n_total": n_total,
            "n_leading": n_total,
            "n_trailing": 0,
            "n_intermittent": 0,
            "pct_missing": 1.0,
            "first_valid_idx": None,
            "last_valid_idx": None,
        }

    first_pos = series.index.get_loc(first_valid)
    last_pos = series.index.get_loc(last_valid)

    n_leading = int(first_pos)
    n_trailing = int(n_total - last_pos - 1)
    n_intermittent = n_missing - n_leading - n_trailing

    return {
        "n_total": n_total,
        "n_leading": n_leading,
        "n_trailing": n_trailing,
        "n_intermittent": n_intermittent,
        "pct_missing": n_missing / n_total,
        "first_valid_idx": first_valid,
        "last_valid_idx": last_valid,
    }


def classify_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a missing-value summary report for a full panel."""
    records = {}
    for col in df.columns:
        records[col] = detect_missing_type(df[col])
    return pd.DataFrame(records).T


def handle_missing(
    df: pd.DataFrame,
    method: str,
    **kwargs: object,
) -> pd.DataFrame:
    """Apply a missing-value treatment to a DataFrame."""
    if method == "trim_start":
        return _trim_start(df)
    if method == "drop_vars":
        return _drop_vars(df, **kwargs)
    if method == "interpolate":
        return _interpolate_intermittent(df)
    if method == "forward_fill":
        return df.ffill()
    if method == "factor":
        from macroforecast.preprocessing.imputation import factor_impute

        return factor_impute(df, **kwargs)
    if method == "em":
        raise NotImplementedError(
            "EM imputation is not implemented in v1. Deferred to a future release."
        )
    raise ValueError(
        f"Unknown missing-value method: '{method}'. "
        "Choose from: trim_start, drop_vars, interpolate, forward_fill, factor, em."
    )


def _trim_start(df: pd.DataFrame) -> pd.DataFrame:
    """Advance start date so no column has leading NaNs."""
    latest_start = max(
        (s for s in (df[c].first_valid_index() for c in df.columns) if s is not None),
        default=None,
    )
    if latest_start is None:
        return df
    return df.loc[latest_start:]


def _drop_vars(df: pd.DataFrame, max_missing_pct: float = 0.5) -> pd.DataFrame:
    """Drop columns with fraction of NaN above *max_missing_pct*."""
    frac_missing = df.isna().mean()
    keep = frac_missing[frac_missing <= max_missing_pct].index
    n_dropped = len(df.columns) - len(keep)
    if n_dropped:
        import warnings

        warnings.warn(
            f"drop_vars: removed {n_dropped} variable(s) exceeding "
            f"{max_missing_pct:.0%} missing threshold.",
            stacklevel=3,
        )
    return df[keep]


def _interpolate_intermittent(df: pd.DataFrame) -> pd.DataFrame:
    """Linearly interpolate only internal (intermittent) NaN gaps.

    Leading and trailing NaN blocks are preserved.
    """
    result = df.copy()
    for col in df.columns:
        s = df[col]
        first_valid = s.first_valid_index()
        last_valid = s.last_valid_index()
        if first_valid is None:
            continue
        interior = s.loc[first_valid:last_valid].interpolate(method="linear")
        result.loc[first_valid:last_valid, col] = interior
    return result


# ---------------------------------------------------------------------------
# McCracken-Ng pipeline convenience function
# ---------------------------------------------------------------------------


def em_factor(
    X: np.ndarray,
    k: int = 8,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Stock & Watson (2002) EM algorithm for factor estimation with NaN.

    Parameters
    ----------
    X : (T, N) array
        Panel with missing values (NaN).
    k : int
        Number of factors to extract.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence tolerance (relative change in filled data).

    Returns
    -------
    F : (T, k) array
        Extracted factors.
    L : (N, k) array
        Factor loadings.
    X_filled : (T, N) array
        Filled panel (no NaN).
    n_iter : int
        Number of EM iterations performed.
    """
    T, N = X.shape
    nan_mask = np.isnan(X)

    # Init: fill NaN with column means
    X_fill = X.copy()
    col_means = np.nanmean(X_fill, axis=0)
    for j in range(N):
        if np.isnan(col_means[j]):
            col_means[j] = 0.0
        X_fill[nan_mask[:, j], j] = col_means[j]

    n_iter = 0
    for iteration in range(max_iter):
        n_iter = iteration + 1
        # Standardize: demean + scale (DEMEAN=2 in MATLAB)
        mu = np.mean(X_fill, axis=0)
        sigma = np.std(X_fill, axis=0, ddof=0)
        sigma[sigma == 0] = 1.0
        X_std = (X_fill - mu) / sigma

        # PCA via SVD
        U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
        F = U[:, :k] * S[:k]  # (T, k)
        L = Vt[:k, :].T        # (N, k)

        # Predict + un-standardize
        X_hat_std = F @ L.T
        X_hat = X_hat_std * sigma + mu

        # Fill NaN with predictions
        X_new = X_fill.copy()
        X_new[nan_mask] = X_hat[nan_mask]

        # Convergence check
        diff = np.sum((X_new - X_fill) ** 2) / max(np.sum(X_fill ** 2), 1e-10)
        X_fill = X_new
        if diff < tol:
            break

    # Final PCA on converged data
    mu = np.mean(X_fill, axis=0)
    sigma = np.std(X_fill, axis=0, ddof=0)
    sigma[sigma == 0] = 1.0
    X_std = (X_fill - mu) / sigma
    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    F = U[:, :k] * S[:k]
    L = Vt[:k, :].T

    return F, L, X_fill, n_iter


def prepare_fredmd(
    raw_df: pd.DataFrame,
    tcodes: dict[str, int],
    outlier_threshold: float = 10.0,
    tcode_override: dict[str, int] | None = None,
    em_k: int = 8,
) -> pd.DataFrame:
    """Apply the McCracken-Ng preprocessing pipeline in the correct order.

    The canonical FRED-MD pipeline (McCracken & Ng 2016; ``fbi`` R package)
    proceeds as:

    1. **Transform** raw levels to stationary series using tcodes.
    2. **Trim** the first 2 rows lost to differencing.
    3. **Remove outliers** on the *stationary* panel (IQR method).
    4. **EM imputation** using Stock & Watson (2002) factor-based algorithm.

    .. warning::
       A common mistake is to run outlier removal on raw levels *before*
       transforming.  This function enforces the correct order.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw-level FRED-MD panel (rows = dates, columns = series).
    tcodes : dict[str, int]
        Transformation codes for each column.
    outlier_threshold : float, default 10.0
        IQR multiplier for outlier detection (McCracken-Ng default: 10).
    tcode_override : dict[str, int], optional
        Per-variable tcode overrides.
    em_k : int, default 8
        Number of factors for EM imputation.

    Returns
    -------
    pd.DataFrame
        Stationary panel with outliers replaced by NaN, trimmed of the
        first 2 rows, and all NaN values filled via EM algorithm.
        No NaN cells remain in the returned DataFrame.
    """
    from macrocast.preprocessing.transforms import apply_tcodes

    effective_tcodes = dict(tcodes)
    if tcode_override:
        effective_tcodes.update(tcode_override)

    # Step 1: transform raw levels → stationary
    transformed = apply_tcodes(raw_df, effective_tcodes)

    # Step 2: trim first 2 rows (lost to differencing; MATLAB: yt(3:end,:))
    transformed = transformed.iloc[2:]

    # Step 3: outlier removal on stationary data
    cleaned = remove_outliers_iqr(transformed, threshold=outlier_threshold)

    # Step 4: EM factor imputation
    X = cleaned.values
    F, L, X_filled, n_iter = em_factor(X, k=em_k)
    result = pd.DataFrame(
        X_filled,
        index=cleaned.index,
        columns=cleaned.columns,
    )

    return result