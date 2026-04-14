"""Giacomini-White (2006) test for conditional predictive ability.

Tests H0: equal conditional predictive accuracy between two forecasting models,
given a set of instruments h_t (e.g., recession indicator, lagged variables).

The GW test extends the Diebold-Mariano test to a conditional setting:

    H0: E[d_t | h_t] = 0  for all t

where d_t = L(e1_t) - L(e2_t) is the loss differential.

The test statistic is a Wald test of E[h_t * d_t] = 0:

    GW = T * Z_bar' * S_hat^{-1} * Z_bar

where Z_t = h_t * d_t and S_hat is a HAC (Newey-West) estimate of the
long-run covariance matrix of Z_t.

Under H0: GW ~ chi-squared(q), where q = dim(h_t).

Using h_t = 1 (constant) reduces the GW test to the standard DM test.

Key distinction from DM: the GW test is valid for both nested and non-nested
models, and detects whether predictive ability varies with economic conditions
(e.g., model 2 is better in recessions, worse in expansions).

Reference
---------
Giacomini, R. and White, H. (2006).
"Tests of Conditional Predictive Ability."
Econometrica, 74(6), 1545–1578.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy import stats


@dataclass
class GWResult:
    """Result of the Giacomini-White test.

    Attributes
    ----------
    statistic : float
        GW Wald statistic (approximately chi-squared(q) under H0).
    p_value : float
        p-value from the chi-squared(q) distribution.
    reject : bool
        Whether H0 is rejected at *alpha* significance level.
    alpha : float
        Significance level used for *reject*.
    df : int
        Degrees of freedom (= number of instruments q).
    mean_interaction : np.ndarray of shape (q,)
        Sample mean of Z_t = h_t * d_t.  Non-zero entries indicate which
        instruments drive the conditional predictive ability difference.
    """

    statistic: float
    p_value: float
    reject: bool
    alpha: float
    df: int
    mean_interaction: NDArray[np.floating]


def gw_test(
    y_true: NDArray[np.floating],
    y_hat_1: NDArray[np.floating],
    y_hat_2: NDArray[np.floating],
    instruments: NDArray[np.floating] | None = None,
    h: int = 1,
    loss: Literal["mse", "mae"] = "mse",
    nw_bw: int | None = None,
    alpha: float = 0.10,
) -> GWResult:
    """Giacomini-White test for conditional predictive ability.

    Parameters
    ----------
    y_true : array of shape (T,)
        Realised values.
    y_hat_1 : array of shape (T,)
        Forecasts from model 1 (benchmark).
    y_hat_2 : array of shape (T,)
        Forecasts from model 2.
    instruments : array of shape (T,) or (T, q), optional
        Conditioning instruments h_t.  A 1-D array is treated as a single
        instrument.  If ``None``, the constant instrument ``h_t = 1`` is used,
        which reduces the GW test to the unconditional DM test (chi-squared(1),
        equivalent to a two-sided normal test).
    h : int
        Forecast horizon.  Sets the Newey-West bandwidth to ``max(1, h-1)``
        by default.
    loss : str
        Loss function: ``"mse"`` (squared error) or ``"mae"`` (absolute error).
    nw_bw : int or None
        Override the Newey-West bandwidth.
    alpha : float
        Significance level.  Default 0.10.

    Returns
    -------
    GWResult

    Examples
    --------
    Unconditional test (equivalent to two-sided DM):

    >>> result = gw_test(y_true, f1, f2)

    Conditional on recession indicator:

    >>> result = gw_test(y_true, f1, f2, instruments=usrec)

    Conditional on recession + lagged loss diff (q=2):

    >>> instr = np.column_stack([usrec, lagged_loss_diff])
    >>> result = gw_test(y_true, f1, f2, instruments=instr)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_hat_1 = np.asarray(y_hat_1, dtype=float)
    y_hat_2 = np.asarray(y_hat_2, dtype=float)

    T = len(y_true)
    if not (T == len(y_hat_1) == len(y_hat_2)):
        raise ValueError("y_true, y_hat_1, and y_hat_2 must have the same length.")

    # Loss differential
    e1 = y_true - y_hat_1
    e2 = y_true - y_hat_2
    if loss == "mse":
        d = e1**2 - e2**2
    elif loss == "mae":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(f"Unknown loss: '{loss}'. Choose 'mse' or 'mae'.")

    # Instruments matrix H: shape (T, q)
    if instruments is None:
        H = np.ones((T, 1))  # constant → unconditional test
    else:
        H = np.asarray(instruments, dtype=float)
        if H.ndim == 1:
            H = H.reshape(-1, 1)
        if H.shape[0] != T:
            raise ValueError(
                f"instruments must have {T} rows (got {H.shape[0]})."
            )

    q = H.shape[1]

    # Interaction: Z_t = h_t * d_t, shape (T, q)
    Z = H * d[:, np.newaxis]
    Z_bar = Z.mean(axis=0)  # shape (q,)

    # HAC (Newey-West) covariance matrix of Z_bar
    bw = nw_bw if nw_bw is not None else max(1, h - 1)
    Z_dm = Z - Z_bar[np.newaxis, :]  # demeaned, shape (T, q)
    S = Z_dm.T @ Z_dm / T  # gamma_0
    for lag in range(1, bw + 1):
        gamma_l = Z_dm[lag:].T @ Z_dm[:-lag] / T
        w = 1.0 - lag / (bw + 1)
        S += w * (gamma_l + gamma_l.T)

    S_bar = S / T  # long-run variance of Z_bar

    # Wald statistic: T * Z_bar' * S^{-1} * Z_bar
    try:
        S_inv = np.linalg.inv(S_bar)
        statistic = float(T * Z_bar @ S_inv @ Z_bar)
    except np.linalg.LinAlgError:
        return GWResult(
            statistic=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            df=q,
            mean_interaction=Z_bar,
        )

    # Degenerate: all Z_t identical
    if statistic < 0 or not np.isfinite(statistic):
        return GWResult(
            statistic=float("nan"),
            p_value=float("nan"),
            reject=False,
            alpha=alpha,
            df=q,
            mean_interaction=Z_bar,
        )

    p_value = float(stats.chi2.sf(statistic, df=q))
    reject = p_value < alpha

    return GWResult(
        statistic=statistic,
        p_value=p_value,
        reject=reject,
        alpha=alpha,
        df=q,
        mean_interaction=Z_bar,
    )
