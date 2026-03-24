"""CLSS 2021 full production replication.

Runs RF horse race across 6 information sets, 11 targets, 4 horizons.
Upgrades overnight run: p_marx=4, n_estimators=200, 11 targets, n_jobs=8.

Key improvements over overnight script:
  - p_marx=4  (meaningful MARX/MAF; overnight used p_marx=1 which collapses to identity)
  - n_estimators=200  (more stable forests)
  - 11 targets  (all CLSS 2021 targets available in FRED-MD)
  - n_jobs=8  (full parallelisation)
  - Checkpointing: skips targets whose parquet already exists

Usage: uv run python scripts/clss2021_full_run.py
Results saved to: ~/.macrocast/results/clss2021_full/
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from macrocast.data import load_fred_md
from macrocast.pipeline import (
    CVScheme,
    FeatureSpec,
    HorseRaceGrid,
    LossFunction,
    ModelSpec,
    Regularization,
    RFModel,
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

RESULTS_DIR = Path.home() / ".macrocast" / "results" / "clss2021_full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# All 11 CLSS 2021 targets available in FRED-MD.
# "S&P 500" has a space in the column name — handled via safe_name() below.
TARGETS: list[str] = [
    "INDPRO",            # Industrial production
    "PAYEMS",            # Nonfarm payroll employment
    "UNRATE",            # Unemployment rate
    "CPIAUCSL",          # CPI all items
    "PCEPI",             # PCE price index
    "TB3MS",             # 3-month T-bill rate
    "GS10",              # 10-year Treasury yield
    "WPSFD49207",        # PPI finished goods
    "M2REAL",            # Real M2
    "DPCERA3M086SBEA",   # Real PCE
    "S&P 500",           # S&P 500 index
]

HORIZONS: list[int] = [1, 3, 6, 12]
OOS_START: str = "1999-01-01"
OOS_END: str = "2017-12-01"
N_JOBS: int = 8

# CLSS 2021 defaults: p_marx=4 is required for meaningful MARX/MAF.
# p_marx=1 (overnight) collapses MARX to raw differences — same as stationary X.
_FEAT_KWARGS: dict = dict(n_factors=4, n_lags=2, p_marx=4)

# ---------------------------------------------------------------------------
# 6 information sets (Table 1 / Fig 1 in CLSS 2021)
# ---------------------------------------------------------------------------

FEATURE_SPECS: list[FeatureSpec] = [
    # F: factors only, no raw X, no MARX
    FeatureSpec(
        use_factors=True,
        include_raw_x=False,
        use_marx=False,
        label="F",
        **_FEAT_KWARGS,
    ),
    # F-X: factors + raw predictors, no MARX
    FeatureSpec(
        use_factors=True,
        include_raw_x=True,
        use_marx=False,
        label="F-X",
        **_FEAT_KWARGS,
    ),
    # X-MARX: MARX features only (no PCA, no raw X)
    FeatureSpec(
        use_factors=False,
        include_raw_x=False,
        use_marx=True,
        marx_for_pca=False,
        label="X-MARX",
        **_FEAT_KWARGS,
    ),
    # F-MARX: factors + MARX features (PCA on raw X, MARX appended)
    FeatureSpec(
        use_factors=True,
        include_raw_x=False,
        use_marx=True,
        marx_for_pca=False,
        label="F-MARX",
        **_FEAT_KWARGS,
    ),
    # F-X-MARX: factors + raw X + MARX (dominant CLSS 2021 information set)
    FeatureSpec(
        use_factors=True,
        include_raw_x=True,
        use_marx=True,
        marx_for_pca=False,
        label="F-X-MARX",
        **_FEAT_KWARGS,
    ),
    # MAF: factors from MARX-transformed X (PCA on MARX panel)
    FeatureSpec(
        use_factors=True,
        include_raw_x=False,
        use_marx=True,
        marx_for_pca=True,
        use_maf=True,
        label="MAF",
        **_FEAT_KWARGS,
    ),
]

# ---------------------------------------------------------------------------
# Model spec: RF with production settings
# ---------------------------------------------------------------------------

RF_SPEC = ModelSpec(
    model_cls=RFModel,
    model_kwargs=dict(
        n_estimators=200,
        max_depth_grid=[3, 5, 7],
        min_samples_leaf_grid=[5, 10, 20],
        cv_folds=3,
    ),
    regularization=Regularization.NONE,
    cv_scheme=CVScheme.KFOLD(k=3),
    loss_function=LossFunction.L2,
    model_id="rf",
)

MODEL_SPECS: list[ModelSpec] = [RF_SPEC]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def safe_name(tgt: str) -> str:
    """Convert target column name to a filesystem-safe string."""
    return tgt.replace("&", "").replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FRED-MD current release and return (panel_stationary, panel_levels)."""
    log.info("Loading FRED-MD (current release) ...")
    mf_levels = load_fred_md(vintage=None)
    log.info("Loaded: %d obs x %d vars", *mf_levels.data.shape)

    panel_levels: pd.DataFrame = mf_levels.data.copy()

    mf_stat = mf_levels.transform()
    panel_stationary: pd.DataFrame = mf_stat.data.copy()

    log.info(
        "Stationary panel: %d obs x %d vars, %s to %s",
        *panel_stationary.shape,
        panel_stationary.index.min().strftime("%Y-%m"),
        panel_stationary.index.max().strftime("%Y-%m"),
    )
    return panel_stationary, panel_levels


# ---------------------------------------------------------------------------
# Relative RMSFE table
# ---------------------------------------------------------------------------


def compute_relative_rmsfe(df: pd.DataFrame) -> pd.DataFrame:
    """Compute relative RMSFE for each feature set vs benchmark F.

    Parameters
    ----------
    df : pd.DataFrame
        Combined results DataFrame with columns y_hat, y_true,
        feature_set, target, horizon.

    Returns
    -------
    pd.DataFrame
        Pivot: rows = feature_set, columns = horizon, values = rel_rmsfe.
    """
    if df.empty:
        log.warning("Results DataFrame is empty; cannot compute RMSFE table.")
        return pd.DataFrame()

    df = df.copy()
    df["sq_err"] = (df["y_hat"] - df["y_true"]) ** 2

    msfe = (
        df.groupby(["feature_set", "target", "horizon"])["sq_err"]
        .mean()
        .reset_index()
        .rename(columns={"sq_err": "msfe"})
    )

    bench = msfe[msfe["feature_set"] == "F"][["target", "horizon", "msfe"]].rename(
        columns={"msfe": "msfe_bench"}
    )
    msfe = msfe.merge(bench, on=["target", "horizon"], how="left")
    msfe["rmsfe"] = msfe["msfe"] ** 0.5
    msfe["rmsfe_bench"] = msfe["msfe_bench"] ** 0.5
    msfe["rel_rmsfe"] = msfe["rmsfe"] / msfe["rmsfe_bench"]

    summary = (
        msfe.groupby(["feature_set", "horizon"])["rel_rmsfe"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(index="feature_set", columns="horizon", values="rel_rmsfe")

    ordered_labels = [s.label for s in FEATURE_SPECS]
    pivot = pivot.reindex([lbl for lbl in ordered_labels if lbl in pivot.index])
    return pivot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    run_start = time.time()
    log.info("=== CLSS 2021 full replication started ===")
    log.info("p_marx=4  n_estimators=200  cv_folds=3")
    log.info("Targets (%d): %s", len(TARGETS), TARGETS)
    log.info("Horizons: %s", HORIZONS)
    log.info("Feature sets: %s", [s.label for s in FEATURE_SPECS])
    log.info("OOS window: %s — %s", OOS_START, OOS_END)
    log.info("n_jobs: %d", N_JOBS)
    log.info("Results dir: %s", RESULTS_DIR)

    panel_stat, panel_levels = load_data()

    all_result_dfs: list[pd.DataFrame] = []

    for tgt in TARGETS:
        fname = safe_name(tgt)
        out_path = RESULTS_DIR / f"{fname}_results.parquet"

        # Checkpointing: skip if result file already exists
        if out_path.exists():
            log.info("SKIP %s — %s already exists", tgt, out_path.name)
            existing = pd.read_parquet(out_path)
            all_result_dfs.append(existing)
            continue

        tgt_start = time.time()
        log.info("--- Target: %s  [%s] ---", tgt, datetime.now().strftime("%H:%M:%S"))

        if tgt not in panel_stat.columns:
            log.error("Target '%s' not in panel. Skipping.", tgt)
            continue

        target_series: pd.Series = panel_stat[tgt].dropna()
        predictor_panel: pd.DataFrame = panel_stat.drop(columns=[tgt])
        predictor_levels: pd.DataFrame = panel_levels.drop(columns=[tgt], errors="ignore")

        try:
            grid = HorseRaceGrid(
                panel=predictor_panel,
                target=target_series,
                horizons=HORIZONS,
                model_specs=MODEL_SPECS,
                feature_specs=FEATURE_SPECS,
                panel_levels=predictor_levels,
                oos_start=OOS_START,
                oos_end=OOS_END,
                n_jobs=N_JOBS,
            )

            result_set = grid.run()
            result_df = result_set.to_dataframe()

            if result_df.empty:
                log.warning("Empty results for target %s.", tgt)
            else:
                result_df["target"] = tgt
                result_df.to_parquet(out_path, index=False)
                log.info("Saved: %s  (%d rows)", out_path.name, len(result_df))
                all_result_dfs.append(result_df)

        except Exception:
            log.exception("ERROR processing target %s", tgt)
            log.info("Continuing with remaining targets ...")
            continue

        elapsed = time.time() - tgt_start
        log.info("Target %s done in %.1f min", tgt, elapsed / 60)

    # ------------------------------------------------------------------
    # Combine
    # ------------------------------------------------------------------

    if not all_result_dfs:
        log.error("No results collected. Exiting.")
        return

    log.info("Combining results from %d targets ...", len(all_result_dfs))
    combined = pd.concat(all_result_dfs, ignore_index=True)

    combined_path = RESULTS_DIR / "combined_results.parquet"
    combined.to_parquet(combined_path, index=False)
    log.info("Saved combined: %s  (%d rows)", combined_path, len(combined))

    # ------------------------------------------------------------------
    # Relative RMSFE table
    # ------------------------------------------------------------------

    log.info("\n=== Relative RMSFE (vs F benchmark, averaged over targets) ===")
    rmsfe_table = compute_relative_rmsfe(combined)

    if not rmsfe_table.empty:
        pd.set_option("display.float_format", "{:.4f}".format)
        pd.set_option("display.max_columns", None)
        print("\n" + rmsfe_table.to_string())
        print()

        rmsfe_path = RESULTS_DIR / "rmsfe_summary.parquet"
        rmsfe_table.to_parquet(rmsfe_path)
        rmsfe_table.to_csv(RESULTS_DIR / "rmsfe_summary.csv")
        log.info("Saved RMSFE summary: %s", rmsfe_path)
    else:
        log.warning("RMSFE table empty; check results.")

    total = time.time() - run_start
    log.info("=== Run complete in %.1f min ===", total / 60)


if __name__ == "__main__":
    main()
