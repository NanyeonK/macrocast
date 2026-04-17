"""Flat 1-D time series → supervised (batch, time, feature) tensors.

The deep-model executors in ``macrocast.execution.models.deep`` all consume
the same ``(n_windows, lookback, n_features)`` layout; this adapter is the
single source of truth for how the flat training series is sliced into
windows and how the target index lines up for a given forecast horizon.

Only the univariate case is implemented in Phase 5 — the deep executors
feed the target series directly. Multivariate / raw-panel reshaping is a
Phase 10 deliverable.
"""
from __future__ import annotations

import numpy as np


def reshape_for_sequence(
    *,
    series: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build supervised sequence windows from a flat 1-D series.

    Parameters
    ----------
    series : ndarray, shape ``(T,)``
        The univariate training series, chronologically ordered.
    lookback : int
        Number of consecutive past observations that form one window's
        input tensor.
    horizon : int
        Forecast horizon. ``y_seq[i]`` is taken ``horizon`` steps past the
        end of the window: ``series[i + lookback + horizon - 1]``.

    Returns
    -------
    X_seq : ndarray, shape ``(n_windows, lookback, 1)``
        Sliding windows with a trailing feature axis so downstream torch
        layers see ``(batch, time, features)``.
    y_seq : ndarray, shape ``(n_windows,)``
        Aligned targets.

    Raises
    ------
    ValueError
        If ``series`` is not 1-D, if ``lookback`` or ``horizon`` are not
        positive integers, or if ``len(series) < lookback + horizon``
        (no valid window).
    """
    if not isinstance(lookback, int) or lookback < 1:
        raise ValueError(f"lookback must be a positive int, got {lookback!r}")
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError(f"horizon must be a positive int, got {horizon!r}")

    series = np.asarray(series, dtype=float)
    if series.ndim != 1:
        raise ValueError(
            f"series must be 1-D, got shape {series.shape}"
        )

    n = len(series)
    if n < lookback + horizon:
        raise ValueError(
            f"series of length {n} is too short for lookback={lookback}, "
            f"horizon={horizon} (need at least {lookback + horizon})"
        )

    n_windows = n - lookback - horizon + 1
    X_seq = np.empty((n_windows, lookback, 1), dtype=float)
    y_seq = np.empty(n_windows, dtype=float)
    for i in range(n_windows):
        X_seq[i, :, 0] = series[i : i + lookback]
        y_seq[i] = series[i + lookback + horizon - 1]
    return X_seq, y_seq


__all__ = ["reshape_for_sequence"]
