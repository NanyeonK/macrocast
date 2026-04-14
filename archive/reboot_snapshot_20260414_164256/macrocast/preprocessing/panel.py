"""Panel preprocessing pipeline for macrocast.

PanelTransformer chains multiple named preprocessing steps in a
fit/transform pattern (sklearn-compatible interface).  Each step
targets a subset of columns defined by a TransformScope.

Usage
-----
>>> preprocessor = PanelTransformer([
...     WinsorizeTransform(scope="all", lower_pct=0.01),
...     HPFilterTransform(scope="group:prices", lambda_=1600),
...     CustomTransform(fn=np.log1p, scope=["INDPRO", "PAYEMS"]),
... ])
>>> X_train_clean = preprocessor.fit_transform(X_train, metadata=md.metadata)
>>> X_oos_clean   = preprocessor.transform(X_oos,   metadata=md.metadata)

Scope specification
-------------------
A scope can be one of:

``"all"``
    Every column in the DataFrame.
``list[str]``
    Explicit column names.
``"group:<key>"``
    Variables whose ``VariableMetadata.group`` equals ``<key>``
    (requires metadata).
``"tcode:<N>"``
    Variables whose tcode equals ``N`` (requires metadata).
``"re:<pattern>"``
    Columns matching a regex pattern.

OOS discipline
--------------
Transforms that estimate parameters from data (Winsorize, Demean) should be
fit on the *training* window only, then applied to the OOS window via
``transform()``.  Transforms with no parameters (HP filter, custom functions)
apply identically in both modes, but the HP filter is inherently non-causal
and should not be used in strict pseudo-OOS settings.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Callable

import pandas as pd

from macrocast.data.schema import MacroFrameMetadata

# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


def _resolve_scope(
    scope: str | list[str],
    df: pd.DataFrame,
    metadata: MacroFrameMetadata | None,
) -> list[str]:
    """Return the list of column names matching the given scope.

    Parameters
    ----------
    scope : str or list of str
        See module docstring for valid scope specifications.
    df : pd.DataFrame
        The panel being processed; only its column names are used.
    metadata : MacroFrameMetadata or None
        Required for group/tcode scope types.

    Returns
    -------
    list[str]
        Column names present in *df* that match the scope.

    Raises
    ------
    ValueError
        If scope type requires metadata but none is provided.
    """
    all_cols = list(df.columns)

    if isinstance(scope, list):
        return [c for c in scope if c in all_cols]

    if scope == "all":
        return all_cols

    if scope.startswith("group:"):
        group_key = scope[len("group:"):]
        if metadata is None:
            raise ValueError("Scope 'group:*' requires metadata.")
        return [
            c for c in all_cols
            if c in metadata.variables
            and metadata.variables[c].group == group_key
        ]

    if scope.startswith("tcode:"):
        try:
            tc = int(scope[len("tcode:"):])
        except ValueError as exc:
            raise ValueError(f"Invalid tcode scope: {scope!r}") from exc
        if metadata is None:
            raise ValueError("Scope 'tcode:*' requires metadata.")
        return [
            c for c in all_cols
            if c in metadata.variables
            and metadata.variables[c].tcode == tc
        ]

    if scope.startswith("re:"):
        pattern = scope[len("re:"):]
        rx = re.compile(pattern)
        return [c for c in all_cols if rx.search(c)]

    raise ValueError(
        f"Unrecognised scope {scope!r}. "
        "Use 'all', a list of column names, 'group:<key>', 'tcode:<N>', or 're:<pattern>'."
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseTransform(ABC):
    """Abstract base class for a single panel preprocessing step.

    Parameters
    ----------
    scope : str or list of str
        Which columns to transform.  Defaults to ``"all"``.
    """

    def __init__(self, scope: str | list[str] = "all") -> None:
        self.scope = scope
        self._fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> BaseTransform:
        """Estimate parameters from *df* (training window).

        Default implementation marks the transform as fitted without
        estimating any parameters (for stateless transforms).
        """
        self._fitted = True
        return self

    @abstractmethod
    def transform(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> pd.DataFrame:
        """Apply the transform to *df*.

        Must not modify *df* in-place; return a new DataFrame.
        """

    def fit_transform(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> pd.DataFrame:
        """Fit on *df* and return the transformed result."""
        self.fit(df, metadata)
        return self.transform(df, metadata)

    def _cols(self, df: pd.DataFrame, metadata: MacroFrameMetadata | None) -> list[str]:
        """Resolve the effective column list."""
        return _resolve_scope(self.scope, df, metadata)


# ---------------------------------------------------------------------------
# Concrete transforms
# ---------------------------------------------------------------------------


class WinsorizeTransform(BaseTransform):
    """Cap extreme values at empirical percentile thresholds.

    Thresholds are estimated on the *training* window via ``fit()``.
    The same thresholds are applied in ``transform()``, making this
    OOS-safe when used correctly.

    Parameters
    ----------
    lower_pct : float
        Lower tail percentile (e.g. 0.01 for 1st percentile).
    upper_pct : float or None
        Upper tail percentile.  Defaults to ``1 - lower_pct``.
    scope : str or list of str
    """

    def __init__(
        self,
        lower_pct: float = 0.01,
        upper_pct: float | None = None,
        scope: str | list[str] = "all",
    ) -> None:
        super().__init__(scope)
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct if upper_pct is not None else 1.0 - lower_pct
        self._lower_: dict[str, float] = {}
        self._upper_: dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> WinsorizeTransform:
        cols = self._cols(df, metadata)
        for c in cols:
            s = df[c].dropna()
            self._lower_[c] = float(s.quantile(self.lower_pct))
            self._upper_[c] = float(s.quantile(self.upper_pct))
        self._fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        out = df.copy()
        cols = self._cols(df, metadata)
        for c in cols:
            if c in self._lower_:
                out[c] = out[c].clip(lower=self._lower_[c], upper=self._upper_[c])
        return out


class DemeanTransform(BaseTransform):
    """Subtract the time-series mean from each selected column.

    The mean is estimated on the training window via ``fit()``.

    Parameters
    ----------
    scope : str or list of str
    """

    def __init__(self, scope: str | list[str] = "all") -> None:
        super().__init__(scope)
        self._means_: dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> DemeanTransform:
        cols = self._cols(df, metadata)
        for c in cols:
            self._means_[c] = float(df[c].mean())
        self._fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        out = df.copy()
        cols = self._cols(df, metadata)
        for c in cols:
            if c in self._means_:
                out[c] = out[c] - self._means_[c]
        return out


class HPFilterTransform(BaseTransform):
    """Hodrick-Prescott filter — returns the cycle component.

    **Warning**: the HP filter is a two-sided smoother and therefore
    non-causal.  It should not be used inside pseudo-OOS loops unless
    the contamination is accepted (e.g. for descriptive analysis).

    Parameters
    ----------
    lambda_ : float
        Smoothing parameter (1600 for monthly, 1600 for quarterly,
        100 for annual data are common choices).
    component : str
        ``"cycle"`` (default) returns the detrended residual;
        ``"trend"`` returns the smoothed trend component.
    scope : str or list of str

    Notes
    -----
    Requires ``statsmodels`` (already a package dependency).
    """

    def __init__(
        self,
        lambda_: float = 1600.0,
        component: str = "cycle",
        scope: str | list[str] = "all",
    ) -> None:
        super().__init__(scope)
        if component not in {"cycle", "trend"}:
            raise ValueError("component must be 'cycle' or 'trend'.")
        self.lambda_ = lambda_
        self.component = component

    def transform(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> pd.DataFrame:
        from statsmodels.tsa.filters.hp_filter import hpfilter  # lazy import

        out = df.copy()
        cols = self._cols(df, metadata)
        for c in cols:
            s = df[c].dropna()
            if len(s) < 4:
                continue
            cycle, trend = hpfilter(s, lamb=self.lambda_)
            result = cycle if self.component == "cycle" else trend
            out.loc[result.index, c] = result.values
        self._fitted = True
        return out


class StandardizeTransform(BaseTransform):
    """Subtract mean and divide by std (z-score) for each column.

    Estimated on the training window.

    Parameters
    ----------
    scope : str or list of str
    """

    def __init__(self, scope: str | list[str] = "all") -> None:
        super().__init__(scope)
        self._means_: dict[str, float] = {}
        self._stds_: dict[str, float] = {}

    def fit(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> StandardizeTransform:
        cols = self._cols(df, metadata)
        for c in cols:
            self._means_[c] = float(df[c].mean())
            self._stds_[c] = float(df[c].std())
        self._fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        out = df.copy()
        cols = self._cols(df, metadata)
        for c in cols:
            if c in self._means_ and self._stds_.get(c, 0) > 0:
                out[c] = (out[c] - self._means_[c]) / self._stds_[c]
        return out


class CustomTransform(BaseTransform):
    """Apply a user-supplied function element-wise or column-wise.

    Parameters
    ----------
    fn : callable
        Applied to each selected column as a pd.Series.
        Signature: ``fn(series: pd.Series) -> pd.Series``.
    scope : str or list of str
    """

    def __init__(
        self,
        fn: Callable[[pd.Series], pd.Series],
        scope: str | list[str] = "all",
    ) -> None:
        super().__init__(scope)
        self.fn = fn

    def transform(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> pd.DataFrame:
        out = df.copy()
        for c in self._cols(df, metadata):
            out[c] = self.fn(df[c])
        self._fitted = True
        return out


class DropTransform(BaseTransform):
    """Drop variables matched by scope from the panel.

    Parameters
    ----------
    scope : str or list of str
        Variables to remove.
    """

    def transform(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> pd.DataFrame:
        self._fitted = True
        return df.drop(columns=self._cols(df, metadata), errors="ignore")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class PanelTransformer:
    """Chain multiple preprocessing steps in a fit/transform pipeline.

    Each step is a :class:`BaseTransform` subclass.  ``fit()`` calls
    ``fit()`` on every step in order, passing the *output* of the
    previous step as input.  ``transform()`` applies all steps using
    the parameters estimated during fitting.

    Parameters
    ----------
    steps : list of BaseTransform
        Preprocessing steps to apply in sequence.
    metadata : MacroFrameMetadata or None
        Shared metadata used for scope resolution.  Can also be passed
        per-call to ``fit()`` / ``transform()``.

    Examples
    --------
    >>> preprocessor = PanelTransformer([
    ...     WinsorizeTransform(scope="all", lower_pct=0.01),
    ...     DemeanTransform(scope="group:prices"),
    ... ])
    >>> X_train_clean = preprocessor.fit_transform(X_train, metadata=md.metadata)
    >>> X_oos_clean   = preprocessor.transform(X_oos,   metadata=md.metadata)
    """

    def __init__(
        self,
        steps: list[BaseTransform],
        metadata: MacroFrameMetadata | None = None,
    ) -> None:
        self.steps = steps
        self.metadata = metadata
        self._fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> PanelTransformer:
        """Fit each step sequentially on the (transformed) training data."""
        meta = metadata or self.metadata
        current = df
        for step in self.steps:
            step.fit(current, meta)
            current = step.transform(current, meta)
        self._fitted = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> pd.DataFrame:
        """Apply all fitted steps to *df*."""
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        meta = metadata or self.metadata
        current = df
        for step in self.steps:
            current = step.transform(current, meta)
        return current

    def fit_transform(
        self,
        df: pd.DataFrame,
        metadata: MacroFrameMetadata | None = None,
    ) -> pd.DataFrame:
        """Fit on *df* and return the transformed result."""
        self.fit(df, metadata)
        return self.transform(df, metadata)

    def __repr__(self) -> str:
        step_names = [type(s).__name__ for s in self.steps]
        return f"PanelTransformer([{', '.join(step_names)}])"
