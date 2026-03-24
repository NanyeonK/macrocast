"""Named information set and model spec presets for replicated papers.

Each study class exposes:
- ``info_sets(**params)``  → dict[str, FeatureSpec]
- ``rf_spec(...)``         → ModelSpec (and other model specs as needed)

Usage
-----
>>> from macrocast.pipeline.presets import CLSS2021
>>> specs = CLSS2021.info_sets(P_Y=12, K=8, P_MARX=12)
>>> specs["F-MARX"]
FeatureSpec(factor_type='X', ..., append_marx=True, ...)
>>> FeatureSpec.from_name("F-MARX")
FeatureSpec(factor_type='X', ..., append_marx=True, ...)
"""

from __future__ import annotations

from typing import Any

from macrocast.pipeline.components import (
    CVScheme,
    LossFunction,
    Regularization,
)
from macrocast.pipeline.experiment import FeatureSpec, ModelSpec

# ---------------------------------------------------------------------------
# CLSS 2021: Coulombe, Leroux, Stevanovic, Surprenant (IJF 2021)
# "Macroeconomic data transformations matter"
# ---------------------------------------------------------------------------


class CLSS2021:
    """Information set and model spec presets for CLSS 2021.

    Reference
    ---------
    Goulet Coulombe, Leroux, Stevanovic, Surprenant (2021),
    *Macroeconomic data transformations matter*,
    IJF 37(4): 1338-1354.

    Paper parameters (Appendix B)
    ------------------------------
    P_Y = 12   — AR lags of target
    K   = 8    — number of PCA factors
    P_MARX = 12 — MARX lag order
    Horizons: h ∈ {1, 3, 6, 9, 12, 24}
    """

    #: Default paper parameters
    DEFAULT_P_Y: int = 12
    DEFAULT_K: int = 8
    DEFAULT_P_MARX: int = 12

    @classmethod
    def info_sets(
        cls,
        P_Y: int = 12,
        K: int = 8,
        P_MARX: int = 12,
    ) -> dict[str, FeatureSpec]:
        """Return all 16 Table 1 information sets as a dict[label, FeatureSpec].

        Parameters
        ----------
        P_Y : int
            AR lag order of the target (P_y in the paper).
        K : int
            Number of PCA factors.
        P_MARX : int
            MARX lag order.

        Returns
        -------
        dict[str, FeatureSpec]
            Keys are the canonical Table 1 labels (e.g. ``"F-MARX"``).
        """
        def _spec(**kwargs: Any) -> FeatureSpec:
            return FeatureSpec(n_factors=K, n_lags=P_Y, p_marx=P_MARX, **kwargs)

        return {
            # Factor-based (PCA on stationary X)
            "F":              _spec(factor_type="X"),
            "F-X":            _spec(factor_type="X",    append_raw_x=True),
            "F-MARX":         _spec(factor_type="X",    append_marx=True),
            "F-MAF":          _spec(factor_type="MARX", append_x_factors=True),
            "F-Level":        _spec(factor_type="X",    append_levels=True),
            "F-X-MARX":       _spec(factor_type="X",    append_raw_x=True, append_marx=True),
            "F-X-MAF":        _spec(factor_type="MARX", append_x_factors=True, append_raw_x=True),
            "F-X-Level":      _spec(factor_type="X",    append_raw_x=True, append_levels=True),
            "F-X-MARX-Level": _spec(factor_type="X",    append_raw_x=True, append_marx=True, append_levels=True),
            # No factor (raw X, MARX, MAF, or level columns only)
            "X":              _spec(factor_type="none", append_raw_x=True),
            "MARX":           _spec(factor_type="none", append_marx=True),
            "MAF":            _spec(factor_type="MARX"),
            "X-MARX":         _spec(factor_type="none", append_raw_x=True, append_marx=True),
            "X-MAF":          _spec(factor_type="MARX", append_raw_x=True),
            "X-Level":        _spec(factor_type="none", append_raw_x=True, append_levels=True),
            "X-MARX-Level":   _spec(factor_type="none", append_raw_x=True, append_marx=True, append_levels=True),
        }

    @classmethod
    def rf_spec(
        cls,
        n_estimators: int = 200,
        min_samples_leaf: int = 5,
        max_features: float = 1 / 3,
        model_id: str = "RF",
    ) -> ModelSpec:
        """Random Forest spec matching the paper (Appendix B).

        Parameters
        ----------
        n_estimators : int
            Number of trees (B = 200 in the paper).
        min_samples_leaf : int
            Minimum node size (5 in the paper).  Passed as a single-element
            grid so ``RFModel`` uses it as a fixed hyperparameter (no tuning).
        max_features : float
            Fraction of features considered per split.  The paper uses
            ``floor(p/3)``; passing ``1/3`` approximates this.
        model_id : str
        """
        from macrocast.pipeline.models import RFModel  # avoid circular import

        return ModelSpec(
            model_cls=RFModel,
            regularization=Regularization.NONE,
            cv_scheme=CVScheme.KFOLD(k=5),
            loss_function=LossFunction.L2,
            model_kwargs={
                "n_estimators": n_estimators,
                "min_samples_leaf_grid": [min_samples_leaf],
                "max_features": max_features,
            },
            model_id=model_id,
        )

    @classmethod
    def en_spec(cls, model_id: str = "EN") -> ModelSpec:
        """Elastic Net spec (estimated via R/glmnet)."""
        from macrocast.pipeline.r_models import ElasticNetModel  # noqa: PLC0415

        return ModelSpec(
            model_cls=ElasticNetModel,
            regularization=Regularization.ELASTIC_NET,
            cv_scheme=CVScheme.KFOLD(k=5),
            loss_function=LossFunction.L2,
            model_id=model_id,
        )

    @classmethod
    def al_spec(cls, model_id: str = "AL") -> ModelSpec:
        """Adaptive LASSO spec (estimated via R/glmnet)."""
        from macrocast.pipeline.r_models import AdaptiveLassoModel  # noqa: PLC0415

        return ModelSpec(
            model_cls=AdaptiveLassoModel,
            regularization=Regularization.ADAPTIVE_LASSO,
            cv_scheme=CVScheme.KFOLD(k=5),
            loss_function=LossFunction.L2,
            model_id=model_id,
        )

    @classmethod
    def ardi_spec(cls, model_id: str = "FM") -> ModelSpec:
        """ARDI (Factor Model) benchmark spec."""
        from macrocast.pipeline.r_models import ARDIModel  # noqa: PLC0415

        return ModelSpec(
            model_cls=ARDIModel,
            regularization=Regularization.NONE,
            cv_scheme=CVScheme.BIC,
            loss_function=LossFunction.L2,
            model_id=model_id,
        )

    @classmethod
    def all_model_specs(cls) -> list[ModelSpec]:
        """Return all five paper model specs: RF, EN, AL, FM, BT."""
        return [
            cls.rf_spec(),
            cls.en_spec(),
            cls.al_spec(),
            cls.ardi_spec(),
        ]

    #: All 16 info set labels in Table 1 order
    TABLE1_LABELS: list[str] = [
        "F", "F-X", "F-MARX", "F-MAF", "F-Level",
        "F-X-MARX", "F-X-MAF", "F-X-Level", "F-X-MARX-Level",
        "X", "MARX", "MAF", "X-MARX", "X-MAF", "X-Level", "X-MARX-Level",
    ]

    #: Default forecast horizons (paper Section 3)
    HORIZONS: list[int] = [1, 3, 6, 9, 12, 24]


# ---------------------------------------------------------------------------
# Registry: FeatureSpec.from_name() factory
# ---------------------------------------------------------------------------

# Maps preset name → factory function.  New study presets register here.
_PRESET_REGISTRY: dict[str, dict[str, FeatureSpec]] = {}


def _ensure_clss2021() -> None:
    if "clss2021" not in _PRESET_REGISTRY:
        _PRESET_REGISTRY["clss2021"] = CLSS2021.info_sets()


def get_preset(
    name: str,
    study: str = "clss2021",
    **params: Any,
) -> FeatureSpec:
    """Look up a named FeatureSpec preset.

    Parameters
    ----------
    name : str
        Information set label (e.g. ``"F-MARX"``).
    study : str
        Which study's preset registry to use.  Currently only
        ``"clss2021"`` is supported.
    **params
        Forwarded to the study's ``info_sets()`` factory when the
        registry needs to be (re-)generated with non-default parameters.

    Returns
    -------
    FeatureSpec

    Raises
    ------
    KeyError
        If *name* is not found in the registry for *study*.
    """
    if study == "clss2021":
        if params:
            presets = CLSS2021.info_sets(**params)
        else:
            _ensure_clss2021()
            presets = _PRESET_REGISTRY["clss2021"]
        if name not in presets:
            raise KeyError(
                f"Unknown info set {name!r} for study {study!r}. "
                f"Available: {sorted(presets)}"
            )
        return presets[name]
    raise KeyError(f"Unknown study preset: {study!r}. Available: {sorted(_PRESET_REGISTRY)}")
