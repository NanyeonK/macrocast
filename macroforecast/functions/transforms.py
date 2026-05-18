"""Standalone L3 panel-transform functions.

Cycle 30: L3 basic panel transforms standalone-ization (10 ops).

Each callable wraps the corresponding runtime primitive from
``macroforecast.core.runtime`` to preserve bit-exact results with
the recipe-path dispatch.  Import pattern follows C28/C29 (linear.py,
tests.py): runtime helpers are imported lazily inside each function
body to avoid circular imports and keep the module self-contained at
definition time.

Basic stationary / lag / aggregation / scale ops:
    diff_transform, log_transform, log_diff_transform,
    pct_change_transform, cumsum_transform, ma_window_transform,
    lag_matrix, seasonal_lag_matrix, ma_increasing_order_transform,
    scale_transform
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal validation helper
# ---------------------------------------------------------------------------

def _require_non_empty(panel: pd.DataFrame, *, name: str = "panel") -> None:
    """Raise ValueError when the DataFrame has zero rows or zero columns."""
    if panel.empty:
        raise ValueError(
            f"{name} must not be empty; got shape {panel.shape}"
        )


# ---------------------------------------------------------------------------
# 1. diff_transform
# ---------------------------------------------------------------------------

def diff_transform(panel: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
    """Compute a simple finite difference along the time axis.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel. Each column is a variable; rows are time periods.
        DataFrame or Series; Series is promoted to a single-column
        DataFrame internally.
    periods : int, default 1
        Number of lag periods to difference.  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Differenced panel of the same shape.  The first ``periods`` rows
        contain ``NaN``.

    Notes
    -----
    Calls ``_as_frame`` followed by ``_diff_like`` from
    ``macroforecast.core.runtime``.  Equivalent recipe configuration::

        op: diff
        params:
          n_diff: 1

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame(rng.randn(10, 2), columns=["a", "b"])
    >>> diff_transform(panel).shape
    (10, 2)
    >>> diff_transform(panel).iloc[0].isna().all()
    True

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD: A Monthly Database for Macroeconomic
    Research', JBES 34(4): 574-589. Transformation code 2 (first
    difference).
    """
    from macroforecast.core.runtime import _as_frame, _diff_like  # noqa: PLC0415

    if periods < 1:
        raise ValueError("diff_transform requires periods >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _diff_like(frame, periods=periods)


# ---------------------------------------------------------------------------
# 2. log_transform
# ---------------------------------------------------------------------------

def log_transform(panel: pd.DataFrame) -> pd.DataFrame:
    """Element-wise natural logarithm of a panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.  All values must be strictly positive.  NaN values
        are preserved.

    Returns
    -------
    pd.DataFrame
        Log-transformed panel with the same shape and index/columns.

    Notes
    -----
    Wraps ``_as_frame`` from ``macroforecast.core.runtime`` then applies
    ``np.log`` directly (via pandas ``applymap``-equivalent path).
    Values <= 0 are not silently coerced; callers must ensure positivity.
    The recipe-path uses a cell-by-cell guard (``pd.NA`` on <= 0 cells)
    whereas this standalone uses ``np.log`` directly to preserve NaN
    propagation. Equivalent recipe configuration::

        op: log

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": [1.0, 2.0, 4.0], "b": [2.0, 4.0, 8.0]})
    >>> log_transform(panel)
         a         b
    0  0.0  0.693147
    1  0.693147  1.386294
    2  1.386294  2.079442

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD', JBES 34(4). Transformation code 4
    (log level).
    """
    from macroforecast.core.runtime import _as_frame  # noqa: PLC0415

    frame = _as_frame(panel)
    _require_non_empty(frame)
    return np.log(frame)


# ---------------------------------------------------------------------------
# 3. log_diff_transform
# ---------------------------------------------------------------------------

def log_diff_transform(panel: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
    """Log then first-difference: ``ln(y_t) - ln(y_{t-periods})``.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.  All values must be strictly positive.
    periods : int, default 1
        Number of lag periods to difference.  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Log-differenced panel. The first ``periods`` rows contain NaN.

    Notes
    -----
    Applies ``np.log`` after ``_as_frame``, then calls ``_diff_like``
    from ``macroforecast.core.runtime``.  Equivalent recipe
    configuration::

        op: log_diff
        params:
          n_diff: 1

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": [1.0, np.e, np.e**2]})
    >>> log_diff_transform(panel)
         a
    0  NaN
    1  1.0
    2  1.0

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD', JBES 34(4). Transformation code 5
    (log first-difference -- monthly growth rate approximation).
    """
    from macroforecast.core.runtime import _as_frame, _diff_like  # noqa: PLC0415

    if periods < 1:
        raise ValueError("log_diff_transform requires periods >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    logged = np.log(frame)
    return _diff_like(logged, periods=periods)


# ---------------------------------------------------------------------------
# 4. pct_change_transform
# ---------------------------------------------------------------------------

def pct_change_transform(panel: pd.DataFrame, *, periods: int = 1) -> pd.DataFrame:
    """Percentage change along the time axis.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.
    periods : int, default 1
        Number of lag periods for the percentage change.  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Percentage-change panel: ``(y_t - y_{t-periods}) / |y_{t-periods}|``.
        The first ``periods`` rows contain NaN.

    Notes
    -----
    Calls ``_pct_change_like`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: pct_change
        params:
          n_periods: 1

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [100.0, 110.0, 121.0]})
    >>> pct_change_transform(panel)
              a
    0       NaN
    1  0.100000
    2  0.100000

    References
    ----------
    McCracken & Ng (2016) 'FRED-MD', JBES 34(4). Transformation code 3
    (percent change).
    """
    from macroforecast.core.runtime import _as_frame, _pct_change_like  # noqa: PLC0415

    if periods < 1:
        raise ValueError("pct_change_transform requires periods >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _pct_change_like(frame, periods=periods)


# ---------------------------------------------------------------------------
# 5. cumsum_transform
# ---------------------------------------------------------------------------

def cumsum_transform(panel: pd.DataFrame) -> pd.DataFrame:
    """Cumulative sum along the time axis.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.

    Returns
    -------
    pd.DataFrame
        Cumulative-sum panel of the same shape.  NaNs in the input are
        treated as 0 by pandas ``cumsum`` (NaN propagation is disabled).

    Notes
    -----
    Calls ``_as_frame(panel).cumsum()`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: cumsum

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    >>> cumsum_transform(panel)
         a
    0  1.0
    1  3.0
    2  6.0

    References
    ----------
    macroforecast design Part 2, L3: feature engineering DAG step library.
    """
    from macroforecast.core.runtime import _as_frame  # noqa: PLC0415

    frame = _as_frame(panel)
    _require_non_empty(frame)
    return frame.cumsum()


# ---------------------------------------------------------------------------
# 6. ma_window_transform
# ---------------------------------------------------------------------------

def ma_window_transform(panel: pd.DataFrame, *, window: int = 3) -> pd.DataFrame:
    """Centred rolling moving average with fixed window width.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.
    window : int, default 3
        Rolling window size in periods.  Must be >= 1.  The first
        ``window - 1`` rows will contain NaN (min_periods = window).

    Returns
    -------
    pd.DataFrame
        Rolling-mean panel of the same shape.

    Notes
    -----
    Equivalent to ``_as_frame(panel).rolling(window, min_periods=window).mean()``,
    matching the runtime dispatch for ``op: ma_window``.  Equivalent
    recipe configuration::

        op: ma_window
        params:
          window: 3

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    >>> ma_window_transform(panel, window=3)
              a
    0       NaN
    1       NaN
    2  2.000000
    3  3.000000
    4  4.000000

    References
    ----------
    macroforecast design Part 2, L3: step library, ``ma_window`` op.
    """
    from macroforecast.core.runtime import _as_frame  # noqa: PLC0415

    if window < 1:
        raise ValueError("ma_window_transform requires window >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return frame.rolling(window=window, min_periods=window).mean()


# ---------------------------------------------------------------------------
# 7. lag_matrix
# ---------------------------------------------------------------------------

def lag_matrix(
    panel: pd.DataFrame,
    *,
    n_lag: int = 4,
    include_contemporaneous: bool = False,
) -> pd.DataFrame:
    """Build a wide lag matrix from a panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.  Each column is lagged ``n_lag`` times.
    n_lag : int, default 4
        Number of lags.  Must be >= 1.
    include_contemporaneous : bool, default False
        If ``True``, also include lag 0 (the contemporaneous column),
        suffixed ``_lag0``.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns suffixed ``_lag1``, ``_lag2``, ...,
        ``_lag{n_lag}``.  If ``include_contemporaneous=True``, also
        includes ``_lag0``.  Shape: ``(T, K * n_lags)`` where K is the
        number of input columns.

    Notes
    -----
    Calls ``_lagged_predictors`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: lag
        params:
          n_lag: 4
          include_contemporaneous: false

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> panel = pd.DataFrame({"a": range(6), "b": range(6, 12)})
    >>> lag_matrix(panel, n_lag=2).columns.tolist()
    ['a_lag1', 'a_lag2', 'b_lag1', 'b_lag2']

    References
    ----------
    Stock & Watson (2002) 'Forecasting Using Principal Components from a
    Large Number of Predictors', JASA 97(460): 1167-1179.
    """
    from macroforecast.core.runtime import _as_frame, _lagged_predictors  # noqa: PLC0415

    if n_lag < 1:
        raise ValueError("lag_matrix requires n_lag >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _lagged_predictors(frame, n_lag, include_contemporaneous=include_contemporaneous)


# ---------------------------------------------------------------------------
# 8. seasonal_lag_matrix
# ---------------------------------------------------------------------------

def seasonal_lag_matrix(
    panel: pd.DataFrame,
    *,
    seasonal_period: int = 12,
    n_seasonal_lags: int = 1,
) -> pd.DataFrame:
    """Build a seasonal lag matrix from a panel.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.  Each column is seasonally lagged.
    seasonal_period : int, default 12
        Seasonal cycle length (e.g. 12 for monthly data, 4 for quarterly).
        Must be >= 2.
    n_seasonal_lags : int, default 1
        Number of seasonal lags to include.  Must be >= 1.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns suffixed ``_s{seasonal_period}_lag{i}``
        for ``i`` in ``1, ..., n_seasonal_lags``.  Each lag shifts by
        ``seasonal_period * i`` periods.

    Notes
    -----
    Calls ``_seasonal_lagged_predictors`` from
    ``macroforecast.core.runtime``.  Equivalent recipe configuration::

        op: seasonal_lag
        params:
          seasonal_period: 12
          n_seasonal_lags: 1

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.RandomState(42)
    >>> panel = pd.DataFrame({"a": rng.randn(24)})
    >>> seasonal_lag_matrix(panel, seasonal_period=12, n_seasonal_lags=1).shape
    (24, 1)

    References
    ----------
    Hylleberg, Engle, Granger & Yoo (1990) 'Seasonal Integration and
    Cointegration', Journal of Econometrics 44(1-2): 215-238.
    """
    from macroforecast.core.runtime import _as_frame, _seasonal_lagged_predictors  # noqa: PLC0415

    if seasonal_period < 2:
        raise ValueError("seasonal_lag_matrix requires seasonal_period >= 2")
    if n_seasonal_lags < 1:
        raise ValueError("seasonal_lag_matrix requires n_seasonal_lags >= 1")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _seasonal_lagged_predictors(
        frame,
        seasonal_period=seasonal_period,
        n_seasonal_lags=n_seasonal_lags,
    )


# ---------------------------------------------------------------------------
# 9. ma_increasing_order_transform
# ---------------------------------------------------------------------------

def ma_increasing_order_transform(
    panel: pd.DataFrame,
    *,
    max_order: int = 12,
) -> pd.DataFrame:
    """Compute moving averages of all orders from 2 to ``max_order``.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.
    max_order : int, default 12
        Maximum window order.  Must be >= 2.  Generates windows
        2, 3, ..., max_order.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with columns suffixed ``_ma{order}`` for each
        order from 2 to ``max_order``.  Shape:
        ``(T, K * (max_order - 1))`` where K is the input column count.

    Notes
    -----
    Calls ``_ma_increasing_order`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: ma_increasing_order
        params:
          max_order: 12

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": range(10)})
    >>> ma_increasing_order_transform(panel, max_order=3).columns.tolist()
    ['a_ma2', 'a_ma3']

    References
    ----------
    Coulombe, Leroux, Stevanovic & Surprenant (2021) 'Macroeconomic Data
    Transformations Matter', International Journal of Forecasting 37(4):
    1338-1354.
    """
    from macroforecast.core.runtime import _as_frame, _ma_increasing_order  # noqa: PLC0415

    if max_order < 2:
        raise ValueError("ma_increasing_order_transform requires max_order >= 2")
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _ma_increasing_order(frame, max_order=max_order)


# ---------------------------------------------------------------------------
# 10. scale_transform
# ---------------------------------------------------------------------------

def scale_transform(
    panel: pd.DataFrame,
    *,
    method: str = "zscore",
) -> pd.DataFrame:
    """Standardise a panel column-by-column using a named scale method.

    Parameters
    ----------
    panel : pd.DataFrame
        Input panel.
    method : str, default "zscore"
        Scaling method.  One of:

        * ``"zscore"`` / ``"standard"`` / ``"standardize"`` --
          ``(x - mean) / std`` (population std, ddof=0).
        * ``"robust"`` -- ``(x - median) / IQR`` where IQR is the
          75th minus 25th percentile gap.
        * ``"minmax"`` -- ``(x - min) / (max - min)``.

    Returns
    -------
    pd.DataFrame
        Scaled panel of the same shape.  Columns with zero spread are
        divided by ``pd.NA`` (result: all-NaN column).

    Notes
    -----
    Calls ``_scale_frame`` from ``macroforecast.core.runtime``.
    Equivalent recipe configuration::

        op: scale
        params:
          method: zscore

    Examples
    --------
    >>> import pandas as pd
    >>> panel = pd.DataFrame({"a": [0.0, 1.0, 2.0, 3.0, 4.0]})
    >>> scale_transform(panel)["a"].mean()  # doctest: +ELLIPSIS
    0.0

    References
    ----------
    macroforecast design Part 2, L3: step library, ``scale`` op.
    Matches sklearn ``StandardScaler`` (zscore), ``RobustScaler``
    (robust), and ``MinMaxScaler`` (minmax) column-by-column behaviour.
    """
    from macroforecast.core.runtime import _as_frame, _scale_frame  # noqa: PLC0415

    _VALID_METHODS = {"zscore", "standard", "standardize", "robust", "minmax"}
    if method not in _VALID_METHODS:
        raise NotImplementedError(
            f"scale_transform does not support method={method!r}; "
            f"choose from {sorted(_VALID_METHODS)}"
        )
    frame = _as_frame(panel)
    _require_non_empty(frame)
    return _scale_frame(frame, method=method)
