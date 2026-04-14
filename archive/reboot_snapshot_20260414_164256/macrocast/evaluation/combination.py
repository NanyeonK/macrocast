"""Forecast combination methods.

Combines point forecasts from multiple models into a single composite forecast
using equal-weight averages, median, trimmed mean, or inverse-MSFE weighting.

The returned DataFrame has the same column schema as the input result_df
(from ResultSet.to_dataframe()), with model_id set to
``"COMBO_{METHOD}"`` (e.g., ``"COMBO_MEAN"``).

Reference
---------
Stock, J. H. and Watson, M. W. (2004).
"Combination forecasts of output growth in a seven-country data set."
Journal of Forecasting, 23(6), 405–430.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

_METHOD = Literal["mean", "median", "trimmed_mean", "inv_msfe"]


def combine_forecasts(
    result_df: pd.DataFrame,
    method: _METHOD = "mean",
    trim_pct: float = 0.10,
    window: int | None = None,
) -> pd.DataFrame:
    """Combine forecasts from multiple models into a single composite.

    Parameters
    ----------
    result_df : pd.DataFrame
        Forecast result table (from ``ResultSet.to_dataframe()``).  Must
        contain columns ``model_id``, ``horizon``, ``forecast_date``,
        ``y_hat``, and ``y_true``.
    method : str
        Combination rule:

        ``"mean"``
            Simple equal-weight average.
        ``"median"``
            Cross-model median at each (horizon, date).
        ``"trimmed_mean"``
            Trimmed mean, removing *trim_pct* from each tail.
        ``"inv_msfe"``
            Inverse-MSFE weights.  Weights are proportional to 1/MSFE_m
            where MSFE_m is computed over an expanding or rolling window
            of past errors for model m.
    trim_pct : float
        Fraction to trim from each tail when ``method="trimmed_mean"``.
        Default 0.10 (10% trimming).
    window : int or None
        Rolling window size for ``method="inv_msfe"``.  ``None`` uses an
        expanding window (all past observations).  Ignored for other
        methods.

    Returns
    -------
    pd.DataFrame
        Combined forecast table with the same schema as *result_df*.
        ``model_id`` is set to ``"COMBO_{METHOD.upper()}"``.
        Non-forecast metadata columns (``nonlinearity``, ``regularization``,
        etc.) are set to ``"combo"`` or ``None`` as appropriate.

    Raises
    ------
    ValueError
        If *method* is not one of the supported options.
    ValueError
        If required columns are missing from *result_df*.
    """
    required = {"model_id", "horizon", "forecast_date", "y_hat", "y_true"}
    missing = required - set(result_df.columns)
    if missing:
        raise ValueError(f"result_df missing required columns: {sorted(missing)}")

    valid_methods = {"mean", "median", "trimmed_mean", "inv_msfe"}
    if method not in valid_methods:
        raise ValueError(
            f"Unknown method '{method}'. Choose one of {sorted(valid_methods)}."
        )

    df = result_df.copy()
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])

    if method == "inv_msfe":
        combined = _combine_inv_msfe(df, window=window)
    else:
        combined = _combine_simple(df, method=method, trim_pct=trim_pct)

    combo_id = f"COMBO_{method.upper()}"
    combined["model_id"] = combo_id

    # Fill metadata columns that don't apply to a combination
    meta_cols = ["nonlinearity", "regularization", "cv_scheme", "loss_function"]
    for col in meta_cols:
        if col in result_df.columns:
            combined[col] = "combo"

    optional_cols = ["experiment_id", "train_end", "n_train", "n_factors", "feature_set"]
    for col in optional_cols:
        if col in result_df.columns:
            combined[col] = None

    # Reorder columns to match input schema
    col_order = [c for c in result_df.columns if c in combined.columns]
    combined = combined[col_order]

    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _combine_simple(
    df: pd.DataFrame,
    method: str,
    trim_pct: float,
) -> pd.DataFrame:
    """Equal-weight combination: mean, median, or trimmed mean."""
    # Pivot: rows = (horizon, forecast_date), columns = model_id
    pivot = df.pivot_table(
        index=["horizon", "forecast_date"],
        columns="model_id",
        values="y_hat",
        aggfunc="first",
    )
    y_true = df.groupby(["horizon", "forecast_date"])["y_true"].first()

    if method == "mean":
        y_combo = pivot.mean(axis=1)
    elif method == "median":
        y_combo = pivot.median(axis=1)
    else:  # trimmed_mean
        def _trim(row: pd.Series) -> float:
            vals = row.dropna().values
            n = len(vals)
            k = max(0, int(np.floor(n * trim_pct)))
            if k == 0 or 2 * k >= n:
                return float(np.mean(vals))
            return float(np.mean(np.sort(vals)[k : n - k]))

        y_combo = pivot.apply(_trim, axis=1)

    combined = pd.DataFrame({"y_hat": y_combo, "y_true": y_true}).reset_index()
    return combined


def _combine_inv_msfe(
    df: pd.DataFrame,
    window: int | None,
) -> pd.DataFrame:
    """Inverse-MSFE weighted combination.

    For each (horizon h, forecast_date t), weights are proportional to
    1 / MSFE_m where MSFE_m is the mean squared error of model m over all
    (expanding) or last *window* (rolling) forecast dates strictly before t.
    Falls back to equal weights when fewer than 2 past observations are
    available for any model.
    """
    rows = []

    for horizon, h_df in df.groupby("horizon"):
        dates = sorted(h_df["forecast_date"].unique())
        models = h_df["model_id"].unique().tolist()

        # Pre-build a dict: model → sorted list of (forecast_date, squared_error)
        history: dict[str, list[tuple[pd.Timestamp, float]]] = {m: [] for m in models}
        for _, row in h_df.sort_values("forecast_date").iterrows():
            history[row["model_id"]].append(
                (row["forecast_date"], (row["y_true"] - row["y_hat"]) ** 2)
            )

        for t in dates:
            # Forecasts available at time t
            t_rows = h_df[h_df["forecast_date"] == t]
            if t_rows.empty:
                continue

            y_true = float(t_rows["y_true"].iloc[0])
            model_yhat: dict[str, float] = dict(
                zip(t_rows["model_id"], t_rows["y_hat"].astype(float))
            )

            # Compute MSFE per model using errors strictly before t
            msfe: dict[str, float] = {}
            for m in models:
                past = [se for (d, se) in history[m] if d < t]
                if window is not None:
                    past = past[-window:]
                if len(past) >= 2:
                    msfe[m] = float(np.mean(past))
                else:
                    msfe[m] = float("nan")

            # If any model lacks history, fall back to equal weights
            if any(np.isnan(v) or v <= 0 for v in msfe.values()):
                weights = {m: 1.0 / len(models) for m in models}
            else:
                inv = {m: 1.0 / v for m, v in msfe.items()}
                total = sum(inv.values())
                weights = {m: inv[m] / total for m in models}

            y_combo = sum(weights[m] * model_yhat.get(m, float("nan")) for m in models)

            rows.append(
                {
                    "forecast_date": t,
                    "horizon": horizon,
                    "y_hat": y_combo,
                    "y_true": y_true,
                }
            )

    return pd.DataFrame(rows)
