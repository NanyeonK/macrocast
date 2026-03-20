"""Variable importance extraction and aggregation for CLSS 2021 Fig 3.

Extracts feature_importances from ForecastRecord objects and aggregates
by semantic group (AR, AR-MARX, Factors, MARX, X, Level) for stacked
bar visualisation.  Replicates Figure 3 of Coulombe, Leroux, Stevanovic,
and Surprenant (2021).
"""

from __future__ import annotations

from typing import Union

import pandas as pd

from macrocast.pipeline.results import ForecastRecord, ResultSet

# ---------------------------------------------------------------------------
# Group label mapping
# ---------------------------------------------------------------------------

CLSS_VI_GROUPS: dict[str, str] = {
    "ar":      "AR",
    "ar_marx": "AR-MARX",
    "factors": "Factors",
    "marx":    "MARX",
    "x":       "X",
    "levels":  "Level",
}

# ---------------------------------------------------------------------------
# Group inference from feature name
# ---------------------------------------------------------------------------


def _infer_group(name: str) -> str:
    """Infer semantic group from a feature column name.

    Parameters
    ----------
    name : str
        Feature name as produced by FeatureBuilder, e.g. ``"y_lag_1"``,
        ``"MARX_3"``, ``"MAF_factor_2"``.

    Returns
    -------
    str
        Group key: one of ``"ar"``, ``"ar_marx"``, ``"factors"``,
        ``"marx"``, ``"x"``, ``"levels"``, or ``"other"``.
    """
    # Order matters: more-specific prefixes must be checked before shorter ones.
    if name.startswith("y_marx_lag_"):
        return "ar_marx"
    if name.startswith("y_lag_"):
        return "ar"
    if name.startswith("MAF_factor_") or name.startswith("factor_"):
        return "factors"
    if name.startswith("MARX_"):
        return "marx"
    if name.startswith("X_"):
        return "x"
    if name.startswith("level_"):
        return "levels"
    return "other"


# ---------------------------------------------------------------------------
# Extract long-form VI DataFrame from records
# ---------------------------------------------------------------------------


def extract_vi_dataframe(
    result_source: Union[ResultSet, list[ForecastRecord]],
) -> pd.DataFrame:
    """Build a long-form DataFrame of per-feature importances from records.

    Records whose ``feature_importances`` is ``None`` are silently skipped.

    Parameters
    ----------
    result_source : ResultSet or list of ForecastRecord
        Source of forecast records.

    Returns
    -------
    pd.DataFrame
        Columns: ``model_id``, ``feature_set``, ``horizon``, ``date``,
        ``feature_name``, ``importance``, ``group``.
        ``date`` is the ``forecast_date`` of each record.
        ``group`` is inferred via :func:`_infer_group`.
    """
    # Normalise input to a list of records
    if isinstance(result_source, ResultSet):
        records: list[ForecastRecord] = result_source.records
    else:
        records = result_source

    rows: list[dict] = []
    for rec in records:
        if rec.feature_importances is None:
            continue
        for feat_name, importance in rec.feature_importances.items():
            rows.append(
                {
                    "model_id": rec.model_id,
                    "feature_set": rec.feature_set,
                    "horizon": rec.horizon,
                    "date": rec.forecast_date,
                    "feature_name": feat_name,
                    "importance": importance,
                    "group": _infer_group(feat_name),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "model_id",
                "feature_set",
                "horizon",
                "date",
                "feature_name",
                "importance",
                "group",
            ]
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregate by group
# ---------------------------------------------------------------------------


def vi_by_group(
    vi_df: pd.DataFrame,
    group_map: dict[str, str] | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Sum (and optionally normalise) importances by semantic group.

    Parameters
    ----------
    vi_df : pd.DataFrame
        Output of :func:`extract_vi_dataframe`.
    group_map : dict of str -> str, optional
        Override mapping from ``feature_name`` to group string.  When
        provided, it replaces the ``group`` column values for matching
        feature names before aggregation.
    normalize : bool, default True
        If True, divide each group's summed importance by the total
        summed importance in the same ``(model_id, feature_set, horizon,
        date)`` cell so that shares sum to 1.0.

    Returns
    -------
    pd.DataFrame
        Columns: ``model_id``, ``feature_set``, ``horizon``, ``date``,
        ``group``, ``importance_share``.
    """
    if vi_df.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "feature_set",
                "horizon",
                "date",
                "group",
                "importance_share",
            ]
        )

    # Apply optional group override
    df = vi_df.copy()
    if group_map is not None:
        mask = df["feature_name"].isin(group_map)
        df.loc[mask, "group"] = df.loc[mask, "feature_name"].map(group_map)

    cell_keys = ["model_id", "feature_set", "horizon", "date", "group"]
    agg = (
        df.groupby(cell_keys, sort=False)["importance"]
        .sum()
        .reset_index()
        .rename(columns={"importance": "importance_share"})
    )

    if normalize:
        # Compute total importance per (model_id, feature_set, horizon, date)
        total_keys = ["model_id", "feature_set", "horizon", "date"]
        totals = (
            agg.groupby(total_keys, sort=False)["importance_share"]
            .sum()
            .reset_index()
            .rename(columns={"importance_share": "_total"})
        )
        agg = agg.merge(totals, on=total_keys, how="left")
        agg["importance_share"] = agg["importance_share"] / agg["_total"]
        agg = agg.drop(columns=["_total"])

    return agg


# ---------------------------------------------------------------------------
# Average over OOS dates per horizon
# ---------------------------------------------------------------------------


def average_vi_by_horizon(
    vi_group_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """Average importance shares over OOS dates within each horizon.

    Parameters
    ----------
    vi_group_df : pd.DataFrame
        Output of :func:`vi_by_group`.
    horizons : list of int, optional
        If provided, only the listed horizons are retained before averaging.

    Returns
    -------
    pd.DataFrame
        Columns: ``model_id``, ``feature_set``, ``horizon``, ``group``,
        ``importance_share`` (mean over dates).
    """
    df = vi_group_df.copy()

    if horizons is not None:
        df = df[df["horizon"].isin(horizons)]

    if df.empty:
        return pd.DataFrame(
            columns=[
                "model_id",
                "feature_set",
                "horizon",
                "group",
                "importance_share",
            ]
        )

    agg_keys = ["model_id", "feature_set", "horizon", "group"]
    result = (
        df.groupby(agg_keys, sort=False)["importance_share"]
        .mean()
        .reset_index()
    )

    return result
