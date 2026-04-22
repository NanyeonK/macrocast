"""Target construction for 1.2.4 horizon_target_construction axis.

Provides forward-transform (build training target from the raw target series at
horizon h) and inverse-transform (convert model forecasts back to the raw
target scale so metrics can be computed on the original series).

Three constructions are operational in v1.0:

- ``future_target_level_t_plus_h``: target_{t+h} (default, identity inverse)
- ``future_diff``: target_{t+h} - target_t
- ``future_logdiff``: log(target_{t+h}) - log(target_t)

All constructions share a single vectorised forward implementation. The inverse
takes a scalar point forecast plus the anchor target level at the origin index
and returns the predicted target level at t+h.
"""
from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd


OPERATIONAL_CONSTRUCTIONS: Final[frozenset[str]] = frozenset({
    "future_target_level_t_plus_h",
    "future_diff",
    "future_logdiff",
})
LEGACY_CONSTRUCTION_ALIASES: Final[dict[str, str]] = {
    "future_level_y_t_plus_h": "future_target_level_t_plus_h",
}
SUPPORTED_CONSTRUCTIONS: Final[frozenset[str]] = (
    OPERATIONAL_CONSTRUCTIONS | frozenset(LEGACY_CONSTRUCTION_ALIASES)
)


def canonicalize_horizon_target_construction(construction: str) -> str:
    """Return the canonical target-construction id for legacy aliases."""
    return LEGACY_CONSTRUCTION_ALIASES.get(str(construction), str(construction))


def _log_or_raise(series: pd.Series, *, construction: str) -> pd.Series:
    """log(series) with strict-positivity check."""
    if (series <= 0).any():
        raise ValueError(
            f"horizon_target_construction={construction!r} requires strictly "
            f"positive target values (got min={float(series.min())})"
        )
    return np.log(series)


def build_horizon_target(target: pd.Series, horizon: int, construction: str) -> pd.Series:
    """Build the training target at ``horizon`` from the raw target series.

    Output is aligned to the target index with NaN at the trailing ``horizon``
    positions where target_{t+h} is not observed.
    """
    construction = canonicalize_horizon_target_construction(construction)
    if construction not in OPERATIONAL_CONSTRUCTIONS:
        raise ValueError(
            f"unknown horizon_target_construction={construction!r}; "
            f"operational set is {sorted(SUPPORTED_CONSTRUCTIONS)}"
        )
    target_future = target.shift(-horizon)
    if construction == "future_target_level_t_plus_h":
        return target_future
    if construction == "future_diff":
        return target_future - target
    log_target = _log_or_raise(target, construction=construction)
    log_target_future = _log_or_raise(target_future.dropna(), construction=construction).reindex(target.index)
    return log_target_future - log_target


def inverse_horizon_target(
    y_hat: float,
    y_anchor: float,
    construction: str,
) -> float:
    """Convert a forecast on construction scale back to raw target level."""
    construction = canonicalize_horizon_target_construction(construction)
    if construction not in OPERATIONAL_CONSTRUCTIONS:
        raise ValueError(
            f"unknown horizon_target_construction={construction!r}; "
            f"operational set is {sorted(SUPPORTED_CONSTRUCTIONS)}"
        )
    y_hat_f = float(y_hat)
    if construction == "future_target_level_t_plus_h":
        return y_hat_f
    if construction == "future_diff":
        return float(y_anchor) + y_hat_f
    if y_anchor <= 0:
        raise ValueError(
            f"horizon_target_construction={construction!r} inverse requires "
            f"strictly positive target_anchor (got {y_anchor!r})"
        )
    return float(y_anchor) * float(np.exp(y_hat_f))


def is_log_space(construction: str) -> bool:
    """True if the forecast scale is logarithmic."""
    return canonicalize_horizon_target_construction(construction) == "future_logdiff"


def forward_scalar(y_val: float, y_anchor: float, construction: str) -> float:
    """Apply forward transform to a single scalar forecast or actual value."""
    construction = canonicalize_horizon_target_construction(construction)
    if construction not in OPERATIONAL_CONSTRUCTIONS:
        raise ValueError(
            f"unknown horizon_target_construction={construction!r}; "
            f"operational set is {sorted(SUPPORTED_CONSTRUCTIONS)}"
        )
    y_val_f = float(y_val)
    if construction == "future_target_level_t_plus_h":
        return y_val_f
    if construction == "future_diff":
        return y_val_f - float(y_anchor)
    if y_val_f <= 0 or float(y_anchor) <= 0:
        raise ValueError(
            f"horizon_target_construction={construction!r} forward requires "
            f"strictly positive target and target_anchor (got target={y_val_f!r}, "
            f"target_anchor={float(y_anchor)!r})"
        )
    return float(np.log(y_val_f) - np.log(float(y_anchor)))
