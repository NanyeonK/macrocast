"""Clark-West (2007) test for equal predictive accuracy in nested models.

Tests H0: the larger (nested) model has no predictive advantage over the
benchmark, against H1: the larger model is more accurate.

The DM test is size-distorted for nested models because the population
loss differential is non-positive under H0. Clark & West (2007) propose an
adjustment:

    d_t = e1_t^2 - [e2_t^2 - (f1_t - f2_t)^2]

where e_i = y_true - f_i are forecast errors.  Intuitively, the adjustment
(f1_t - f2_t)^2 penalises the larger model for the noise introduced by
estimating additional parameters.

The CW statistic is the t-ratio of d_bar from regressing d_t on a constant:

    CW = d_bar / se(d_bar)

with HAC standard errors (Newey-West, max(1, h-1) lags) when h > 1.

Under H0, CW is approximately N(0, 1).  The test is **one-sided**: reject H0
when CW > critical value (i.e., larger model is better).

Reference
---------
Clark, T.E. and West, K.D. (2007).
"Approximately Normal Tests for Equal Predictive Accuracy in Nested Models."
Journal of Econometrics, 138(1), 291–311.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class CWResult:
    """Result of the Clark-West test.

    Attributes
    ----------
    statistic : float
        CW test statistic (approximately N(0,1) under H0).
    p_value : float
        One-sided p-value for H1: larger model is more accurate.
    reject : bool
        Whether H0 is rejected at *alpha* significance level.
    alpha : float
        Significance level used to determine *reject*.
    adj_loss_diff_mean : float
        Mean of the adjusted loss differential d_bar.
        Positive values indicate that the larger model is better after
        the CW penalty for parameter estimation noise.
    """

    statistic: float
    p_value: float
    reject: bool
    alpha: float
    adj_loss_diff_mean: float


def cw_test(
    y_true: NDArray[np.floating],
    f_benchmark: NDArray[np.floating],
    f_model: NDArray[np.floating],
    h: int = 1,
    nw_bw: int | None = None,
    alpha: float = 0.10,
) -> CWResult:
    """Clark-West test for predictive accuracy in nested models.

    Parameters
    ----------
    y_true : array of shape (T,)
        Realised values of the target variable.
    f_benchmark : array of shape (T,)
        Forecasts from the benchmark (smaller/nested) model, e.g. AR(p).
    f_model : array of shape (T,)
        Forecasts from the larger model (nests the benchmark).
    h : int
        Forecast horizon.  Used to set the Newey-West bandwidth to
        ``max(1, h - 1)`` by default.
    nw_bw : int or None
        Override the Newey-West bandwidth.  ``None`` uses ``max(1, h - 1)``.
    alpha : float
        Significance level for the *reject* flag.  Default 0.10.

    Returns
    -------
    CWResult

    Notes
    -----
    The test is one-sided: H1 is that the larger model has better predictive
    accuracy.  Use a one-sided critical value (e.g. 1.282 at 10%, 1.645 at 5%).

    The test requires that *f_model* nests *f_benchmark* (i.e., the benchmark
    is a restricted version of the larger model).  For non-nested comparisons,
    use the standard :func:`dm_test` instead.
    """
    y_true = np.asarray(y_true, dtype=float)
    f_benchmark = np.asarray(f_benchmark, dtype=float)
    f_model = np.asarray(f_model, dtype=float)

    T = len(y_true)
    if not (T == len(f_benchmark) == len(f_model)):
        raise ValueError("y_true, f_benchmark, and f_model must have the same length.")

    e1 = y_true - f_benchmark   # benchmark errors
    e2 = y_true - f_model       # larger model errors

    # Clark-West adjusted loss differential
    # d_t = e1_t^2 - e2_t^2 + (f_benchmark_t - f_model_t)^2
    adj_correction = (f_benchmark - f_model) ** 2
    d = e1**2 - e2**2 + adj_correction
    d_bar = d.mean()

    # Degenerate case: zero variance in loss differentials
    if np.std(d) == 0.0:
        return CWResult(
            statistic=0.0,
            p_value=0.5,
            reject=False,
            alpha=alpha,
            adj_loss_diff_mean=float(d_bar),
        )

    # Newey-West HAC variance of d_bar
    bw = nw_bw if nw_bw is not None else max(1, h - 1)
    d_dm = d - d_bar
    gamma0 = np.dot(d_dm, d_dm) / T
    nw_var = gamma0
    for lag in range(1, bw + 1):
        gamma_l = np.dot(d_dm[lag:], d_dm[:-lag]) / T
        w = 1.0 - lag / (bw + 1)
        nw_var += 2 * w * gamma_l

    var_d_bar = nw_var / T

    if var_d_bar <= 0:
        return CWResult(
            statistic=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            adj_loss_diff_mean=float(d_bar),
        )

    statistic = float(d_bar / np.sqrt(var_d_bar))

    # One-sided p-value: H1 is that the larger model is better (d_bar > 0)
    p_value = float(stats.norm.sf(statistic))
    reject = p_value < alpha

    return CWResult(
        statistic=statistic,
        p_value=p_value,
        reject=reject,
        alpha=alpha,
        adj_loss_diff_mean=float(d_bar),
    )
