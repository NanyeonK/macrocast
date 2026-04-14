#!/usr/bin/env python
"""CLSS 2021 results analysis: relative RMSFE + marginal contribution + VI.

Run after clss2021_overnight_run.py completes.

Reads parquet files from ~/.macrocast/results/clss2021_overnight/ and produces:
  1. Relative RMSFE table (Table 2/3 structure from CLSS 2021)
  2. Marginal contribution analysis (Fig 1/2 style)
  3. Variable importance by group (Fig 3 style)
  4. Figures saved to ~/.macrocast/results/clss2021_overnight/figures/

Usage: uv run python scripts/clss2021_analyze_results.py
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import pandas as pd

from macrocast.evaluation import (
    marginal_contribution_all,
    marginal_effect_plot,
    oos_r2_panel,
    variable_importance_plot,
)
from macrocast.evaluation.variable_importance import (
    average_vi_by_horizon,
    extract_vi_dataframe,
    vi_by_group,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

RESULTS_DIR = Path.home() / ".macrocast" / "results" / "clss2021_overnight"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CLSS 2021 experiment constants (must match overnight_run.py)
# ---------------------------------------------------------------------------

TARGETS: list[str] = ["INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL"]
HORIZONS: list[int] = [1, 3, 6, 12]

# CLSS 2021 feature set presentation order (Table 2/3 row order)
FEATURE_SET_ORDER: list[str] = ["F", "F-X", "X-MARX", "F-MARX", "F-X-MARX", "MAF"]


# ---------------------------------------------------------------------------
# Step 1: Load parquet results
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> pd.DataFrame:
    """Load all *_results.parquet files and concatenate into one DataFrame.

    Prefers the pre-combined parquet written by the overnight script when
    available.  Falls back to loading individual per-target files and
    concatenating them in-process.

    Parameters
    ----------
    results_dir : Path
        Root directory containing the parquet files.

    Returns
    -------
    pd.DataFrame
        Combined results with at minimum columns:
        model_id, feature_set, horizon, forecast_date, y_hat, y_true, target.
    """
    combined_path = results_dir / "combined_results.parquet"
    if combined_path.exists():
        log.info("Loading pre-combined parquet: %s", combined_path)
        df = pd.read_parquet(combined_path)
        log.info("Loaded %d rows from combined parquet.", len(df))
        return df

    # Fall back to per-target files
    pattern = str(results_dir / "*_results.parquet")
    parquet_files = sorted(glob.glob(pattern))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {results_dir}. "
            "Run scripts/clss2021_overnight_run.py first."
        )

    log.info("Found %d per-target parquet files; concatenating.", len(parquet_files))
    dfs: list[pd.DataFrame] = []
    for fpath in parquet_files:
        chunk = pd.read_parquet(fpath)
        log.info("  %s: %d rows", os.path.basename(fpath), len(chunk))
        dfs.append(chunk)

    df = pd.concat(dfs, ignore_index=True)
    log.info("Combined: %d rows total.", len(df))
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Log a schema summary and warn about any expected-but-missing columns.

    Parameters
    ----------
    df : pd.DataFrame
        Combined results DataFrame.
    """
    log.info("Schema (%d rows x %d cols):", *df.shape)
    log.info("  columns : %s", df.columns.tolist())
    log.info("  dtypes  :\n%s", df.dtypes.to_string())

    # Columns required by the evaluation functions used below
    required = {
        "model_id", "feature_set", "horizon",
        "forecast_date", "y_hat", "y_true",
    }
    missing = required - set(df.columns)
    if missing:
        log.warning("Expected columns missing from results: %s", sorted(missing))
    else:
        log.info("All required columns present.")

    # Diagnostic prints for quick sanity check
    if "target" in df.columns:
        log.info("Targets      : %s", sorted(df["target"].unique().tolist()))
    log.info("Feature sets : %s", sorted(df["feature_set"].unique().tolist()))
    log.info("Models       : %s", sorted(df["model_id"].unique().tolist()))
    log.info("Horizons     : %s", sorted(df["horizon"].unique().tolist()))
    if "forecast_date" in df.columns:
        log.info(
            "OOS window   : %s  to  %s",
            df["forecast_date"].min(),
            df["forecast_date"].max(),
        )


# ---------------------------------------------------------------------------
# Step 2: Relative RMSFE table (Table 2/3 from CLSS 2021)
# ---------------------------------------------------------------------------


def relative_rmsfe_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute relative RMSFE per (model_id, feature_set) vs F baseline.

    RMSFE is computed per (model_id, feature_set, target, horizon), then
    averaged across targets before constructing the final pivot.  The F
    feature set (factors only, no MARX) serves as the per-horizon baseline,
    matching CLSS 2021 Table 2/3.

    Parameters
    ----------
    df : pd.DataFrame
        Combined results DataFrame.  Must contain model_id, feature_set,
        horizon, y_hat, y_true.  Optional target column is used when present.

    Returns
    -------
    pd.DataFrame
        Pivot table with MultiIndex rows (model_id, feature_set) and horizon
        columns.  Values are RMSFE relative to the F feature set within the
        same (model_id, horizon).  Values < 1 indicate improvement over F.
    """
    df = df.copy()
    df["sq_err"] = (df["y_hat"] - df["y_true"]) ** 2

    # Grouping keys: include target when available so averaging is correct
    group_keys: list[str] = ["model_id", "feature_set", "horizon"]
    if "target" in df.columns:
        group_keys = ["model_id", "feature_set", "target", "horizon"]

    rmsfe = (
        df.groupby(group_keys)["sq_err"]
        .mean()
        .pow(0.5)
        .reset_index()
        .rename(columns={"sq_err": "rmsfe"})
    )

    # If target dimension exists, average RMSFE across targets per
    # (model_id, feature_set, horizon) before computing relative values.
    if "target" in rmsfe.columns:
        rmsfe = (
            rmsfe.groupby(["model_id", "feature_set", "horizon"])["rmsfe"]
            .mean()
            .reset_index()
        )

    # Baseline: feature_set == "F" for each (model_id, horizon)
    baseline = (
        rmsfe.loc[rmsfe["feature_set"] == "F", ["model_id", "horizon", "rmsfe"]]
        .rename(columns={"rmsfe": "rmsfe_baseline"})
    )

    merged = rmsfe.merge(baseline, on=["model_id", "horizon"], how="left")
    merged["rel_rmsfe"] = merged["rmsfe"] / merged["rmsfe_baseline"]

    # Pivot to (model_id, feature_set) x horizon
    pivot = merged.pivot_table(
        index=["model_id", "feature_set"],
        columns="horizon",
        values="rel_rmsfe",
    )
    pivot.columns.name = "horizon"

    # Reorder feature_set rows to match CLSS 2021 presentation order
    ordered_fsets = [fs for fs in FEATURE_SET_ORDER if fs in pivot.index.get_level_values("feature_set")]
    if ordered_fsets:
        # Reindex only rows whose feature_set is in the canonical order list;
        # append remaining rows at the end so no data is lost.
        in_order_mask = pivot.index.get_level_values("feature_set").isin(ordered_fsets)
        pivot_ordered = pivot.loc[in_order_mask].sort_index(
            level="feature_set",
            key=lambda s: s.map({fs: i for i, fs in enumerate(ordered_fsets)}).fillna(999),
        )
        pivot_rest = pivot.loc[~in_order_mask]
        pivot = pd.concat([pivot_ordered, pivot_rest])

    return pivot


# ---------------------------------------------------------------------------
# Step 3: Marginal contribution (Fig 1/2 from CLSS 2021)
# ---------------------------------------------------------------------------


def compute_and_plot_marginal_contributions(
    df: pd.DataFrame,
    figures_dir: Path,
) -> pd.DataFrame:
    """Compute marginal contributions and produce Fig 1/2 style plots.

    The function first adds the oos_r2 column (CLSS 2021 Eq. 11) then calls
    marginal_contribution_all() for features MARX, MAF, and F (Eq. 12).

    Parameters
    ----------
    df : pd.DataFrame
        Combined results DataFrame with y_hat, y_true, feature_set, model_id,
        horizon, forecast_date.  Optional target column is used when present.
    figures_dir : Path
        Directory into which PNG files are written.

    Returns
    -------
    pd.DataFrame
        Stacked marginal contribution table (one row per feature/model/horizon).
    """
    log.info("Adding oos_r2 column (CLSS 2021 Eq. 11) ...")
    df_r2 = oos_r2_panel(df)

    log.info("Computing marginal contributions for MARX, MAF, F ...")
    mc_df = marginal_contribution_all(
        df_r2,
        features=["MARX", "MAF", "F"],
    )

    if mc_df.empty:
        log.warning(
            "marginal_contribution_all() returned an empty DataFrame.  "
            "This usually means the expected (with, without) feature-set pairs "
            "are not both present in the results.  "
            "Available feature sets: %s",
            sorted(df["feature_set"].unique().tolist()),
        )
        return mc_df

    log.info("Marginal contribution table (%d rows):", len(mc_df))
    log.info("\n%s", mc_df.to_string(index=False))

    # Save the marginal contribution table as a parquet artifact
    mc_path = RESULTS_DIR / "marginal_contributions.parquet"
    mc_df.to_parquet(mc_path, index=False)
    log.info("Saved marginal contribution table: %s", mc_path)

    # Detect which models are present so we can pass them explicitly
    available_models: list[str] = sorted(mc_df["model"].unique().tolist())

    # Fig 1: MARX marginal contribution
    features_to_plot = [
        ("MARX", "fig1_marx_marginal.png"),
        ("MAF",  "fig2_maf_marginal.png"),
        ("F",    "fig4_factors_marginal.png"),
    ]
    for feature_label, filename in features_to_plot:
        feat_rows = mc_df.loc[mc_df["feature"] == feature_label]
        if feat_rows.empty:
            log.info("No marginal contribution data for feature=%s; skipping plot.", feature_label)
            continue

        # Filter models to those with data for this feature
        models_with_data = sorted(feat_rows["model"].unique().tolist())
        horizons_with_data = sorted(feat_rows["horizon"].unique().tolist())

        try:
            fig = marginal_effect_plot(
                mc_df=mc_df,
                feature=feature_label,
                models=models_with_data,
                horizons=horizons_with_data,
                title=f"Marginal contribution of {feature_label} (CLSS 2021)",
            )
            out_path = figures_dir / filename
            fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
            plt.close(fig)
            log.info("Saved %s", out_path)
        except Exception as exc:
            log.warning("Could not produce %s: %s", filename, exc)

    return mc_df


# ---------------------------------------------------------------------------
# Step 4: Variable importance (Fig 3 from CLSS 2021)
# ---------------------------------------------------------------------------


def compute_and_plot_variable_importance(
    df: pd.DataFrame,
    figures_dir: Path,
) -> None:
    """Extract per-feature importances and produce Fig 3 style stacked-bar plot.

    feature_importances is stored as a JSON string in the parquet file.  This
    function reconstructs lightweight pseudo-ForecastRecord objects from the
    DataFrame rows so that extract_vi_dataframe() can be called.

    Parameters
    ----------
    df : pd.DataFrame
        Combined results DataFrame.  Must have a feature_importances column
        containing JSON strings (or None/NaN for records without VI).
    figures_dir : Path
        Directory into which PNG files are written.
    """
    # feature_importances is serialised as a JSON string by ForecastRecord.to_dict().
    # Build a plain list-of-dicts for extraction rather than deserialising back to
    # ForecastRecord objects (avoids reconstructing enum fields from parquet strings).

    if "feature_importances" not in df.columns:
        log.info("No feature_importances column in results; skipping VI analysis.")
        return

    vi_rows_available = df["feature_importances"].notna().sum()
    log.info(
        "Rows with feature_importances: %d / %d",
        vi_rows_available,
        len(df),
    )
    if vi_rows_available == 0:
        log.info("No records have feature importances; skipping VI analysis.")
        return

    # Build long-form VI DataFrame directly (bypassing extract_vi_dataframe
    # which expects ResultSet/ForecastRecord objects) so we can work from the
    # parquet-loaded DataFrame without reconstructing enum types.
    from macrocast.evaluation.variable_importance import _infer_group

    vi_rows: list[dict] = []
    for _, row in df.loc[df["feature_importances"].notna()].iterrows():
        try:
            importances: dict[str, float] = json.loads(row["feature_importances"])
        except (json.JSONDecodeError, TypeError):
            continue
        for feat_name, importance in importances.items():
            vi_rows.append(
                {
                    "model_id":    row["model_id"],
                    "feature_set": row["feature_set"],
                    "horizon":     row["horizon"],
                    "date":        row["forecast_date"],
                    "target":      row.get("target", "_all_"),
                    "feature_name": feat_name,
                    "importance":  float(importance),
                    "group":       _infer_group(feat_name),
                }
            )

    if not vi_rows:
        log.info("feature_importances column present but all values unparseable; skipping.")
        return

    vi_df = pd.DataFrame(vi_rows)
    log.info("VI DataFrame: %d rows, features=%d", len(vi_df), vi_df["feature_name"].nunique())

    # Save raw VI long-form table
    vi_path = RESULTS_DIR / "variable_importance.parquet"
    vi_df.to_parquet(vi_path, index=False)
    log.info("Saved variable importance long-form table: %s", vi_path)

    # Aggregate by semantic group and average over OOS dates
    vi_group_df = vi_by_group(vi_df)
    vi_avg_df   = average_vi_by_horizon(vi_group_df, horizons=HORIZONS)

    # Add target column if missing (single-target case)
    if "target" not in vi_avg_df.columns and "target" in vi_df.columns:
        # merge target back via group columns
        target_map = (
            vi_group_df[["model_id", "feature_set", "horizon", "date", "group"]]
            .merge(
                vi_df[["model_id", "feature_set", "horizon", "date", "target"]].drop_duplicates(),
                on=["model_id", "feature_set", "horizon", "date"],
                how="left",
            )
        )
        # average_vi_by_horizon already averaged over dates; add target by
        # most common assignment per (model_id, feature_set, horizon, group)
        target_lookup = (
            target_map.groupby(["model_id", "feature_set", "horizon", "group"])["target"]
            .agg(lambda s: s.mode().iloc[0] if len(s) > 0 else "_all_")
            .reset_index()
        )
        vi_avg_df = vi_avg_df.merge(
            target_lookup, on=["model_id", "feature_set", "horizon", "group"], how="left"
        )

    # Determine which targets are actually present in the VI data
    if "target" in vi_avg_df.columns:
        available_targets = [t for t in TARGETS if t in vi_avg_df["target"].unique()]
    else:
        available_targets = TARGETS[:1]  # single pool

    if not available_targets:
        log.warning("No VI data for any of the expected targets; skipping VI plot.")
        return

    try:
        fig3 = variable_importance_plot(
            vi_avg_df=vi_avg_df,
            targets=available_targets,
            title="Variable importance by group (CLSS 2021 Fig 3)",
        )
        out_path = figures_dir / "fig3_variable_importance.png"
        fig3.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig3)
        log.info("Saved %s", out_path)
    except Exception as exc:
        log.warning("Could not produce fig3_variable_importance.png: %s", exc)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _print_rmsfe_table(table: pd.DataFrame) -> None:
    """Pretty-print the relative RMSFE table to stdout."""
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print("\n" + "=" * 70)
    print("Relative RMSFE (vs F baseline, averaged over targets)")
    print("Values < 1 indicate improvement over F feature set")
    print("=" * 70)
    print(table.to_string())
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point: load results, compute tables, save figures."""
    log.info("=== CLSS 2021 results analysis ===")
    log.info("Results dir : %s", RESULTS_DIR)
    log.info("Figures dir : %s", FIGURES_DIR)

    # ------------------------------------------------------------------
    # Step 1: Load
    # ------------------------------------------------------------------
    try:
        result_df = load_results(RESULTS_DIR)
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    # Schema diagnostics — confirms column names before any downstream call
    validate_schema(result_df)

    # ------------------------------------------------------------------
    # Step 2: Relative RMSFE table
    # ------------------------------------------------------------------
    log.info("Computing relative RMSFE table ...")
    rmsfe_tbl = relative_rmsfe_table(result_df)

    if rmsfe_tbl.empty:
        log.warning(
            "RMSFE table is empty.  This may indicate that the F feature set "
            "is absent from the results, or that y_hat/y_true columns are missing."
        )
    else:
        _print_rmsfe_table(rmsfe_tbl)
        rmsfe_out = RESULTS_DIR / "rmsfe_table.parquet"
        rmsfe_tbl.to_parquet(rmsfe_out)
        log.info("Saved RMSFE table: %s", rmsfe_out)

        # Also write a human-readable CSV for inspection
        rmsfe_csv = RESULTS_DIR / "rmsfe_table.csv"
        rmsfe_tbl.to_csv(str(rmsfe_csv))
        log.info("Saved RMSFE table CSV: %s", rmsfe_csv)

    # ------------------------------------------------------------------
    # Step 3: Marginal contribution analysis
    # ------------------------------------------------------------------
    log.info("Computing marginal contributions ...")
    mc_df = compute_and_plot_marginal_contributions(result_df, FIGURES_DIR)

    # ------------------------------------------------------------------
    # Step 4: Variable importance
    # ------------------------------------------------------------------
    log.info("Computing variable importance ...")
    compute_and_plot_variable_importance(result_df, FIGURES_DIR)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    log.info("=== Analysis complete ===")
    log.info("Outputs written to: %s", RESULTS_DIR)
    log.info(
        "Figures: %s",
        [p.name for p in sorted(FIGURES_DIR.glob("*.png"))],
    )


if __name__ == "__main__":
    main()
