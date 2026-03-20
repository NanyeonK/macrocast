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

import pandas as pd


def detect_missing_type(series: pd.Series) -> dict[str, int | float]:
    """Classify missing observations in a single series.

    Parameters
    ----------
    series : pd.Series
        Univariate time series with a DatetimeIndex.

    Returns
    -------
    dict
        Keys: ``n_total``, ``n_leading``, ``n_trailing``,
        ``n_intermittent``, ``pct_missing``, ``first_valid_idx``,
        ``last_valid_idx``.
    """
    n_total = len(series)
    n_missing = int(series.isna().sum())

    first_valid = series.first_valid_index()
    last_valid = series.last_valid_index()

    if first_valid is None:
        # Series is entirely missing
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
    """Produce a missing-value summary report for a full panel.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with DatetimeIndex rows and variable columns.

    Returns
    -------
    pd.DataFrame
        One row per variable; columns mirror the keys of
        ``detect_missing_type`` plus the variable name as the index.
    """
    records = {}
    for col in df.columns:
        records[col] = detect_missing_type(df[col])
    return pd.DataFrame(records).T


def handle_missing(
    df: pd.DataFrame,
    method: str,
    **kwargs: object,
) -> pd.DataFrame:
    """Apply a missing-value treatment to a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Panel data with DatetimeIndex rows.
    method : str
        Treatment method. Supported values:

        ``"trim_start"``
            Advance the start date to the latest ``first_valid_index``
            across all columns, ensuring no series has leading NaNs.
            This is the standard approach in the FRED-MD literature.
        ``"drop_vars"``
            Drop columns whose fraction of missing observations exceeds
            *max_missing_pct* (keyword argument, default 0.5).
        ``"interpolate"``
            Linearly interpolate only **intermittent** NaN values
            (i.e., internal gaps). Leading and trailing NaNs are left
            intact.
        ``"forward_fill"``
            Last-observation-carried-forward (LOCF) for all NaN cells.
        ``"em"``
            EM-based imputation. Not implemented in v1.
    **kwargs
        Extra keyword arguments forwarded to the selected method.

    Returns
    -------
    pd.DataFrame
        Treated DataFrame.

    Raises
    ------
    ValueError
        If *method* is unrecognised.
    NotImplementedError
        For ``"em"`` (deferred to v2).
    """
    if method == "trim_start":
        return _trim_start(df)
    if method == "drop_vars":
        return _drop_vars(df, **kwargs)
    if method == "interpolate":
        return _interpolate_intermittent(df)
    if method == "forward_fill":
        return df.ffill()
    if method == "em":
        raise NotImplementedError(
            "EM imputation is not implemented in v1. Deferred to a future release."
        )
    raise ValueError(
        f"Unknown missing-value method: '{method}'. "
        "Choose from: trim_start, drop_vars, interpolate, forward_fill, em."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
        # Interpolate only the interior slice
        interior = s.loc[first_valid:last_valid].interpolate(method="linear")
        result.loc[first_valid:last_valid, col] = interior
    return result
