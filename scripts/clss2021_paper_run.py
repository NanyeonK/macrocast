"""CLSS 2021 paper-faithful replication.

Reproduces Coulombe, Leroux, Stevanovic, Surprenant (2021)
"Macroeconomic Data Transformations Matter", IJF 37(4):1338-1354.

Key settings matched to the paper:
  - FRED-MD 2018-02 vintage
  - OOS evaluation 1980-01 to 2017-12 (38 years)
  - RF: fully grown trees, max_features=1/3, 75% subsampling, n_estimators=500
  - All 15 information sets (Table 1)
  - All 6 horizons: h = 1, 3, 6, 9, 12, 24
  - 11 targets (10 CLSS 2021 targets + S&P 500)
  - Checkpointing: skips targets whose parquet already exists

Usage: uv run python scripts/clss2021_paper_run.py
Results: ~/.macrocast/results/clss2021_paper/
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

RESULTS_DIR = Path.home() / ".macrocast" / "results" / "clss2021_paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment parameters — matched to CLSS 2021
# ---------------------------------------------------------------------------

TARGETS: list[str] = [
    "INDPRO",            # Industrial production
    "PAYEMS",            # Nonfarm payroll employment
    "UNRATE",            # Unemployment rate
    "CPIAUCSL",          # CPI all items
    "PCEPI",             # PCE price index
    "TB3MS",             # 3-month T-bill
    "GS10",              # 10-year Treasury yield
    "WPSFD49207",        # PPI finished goods
    "M2REAL",            # Real M2
    "DPCERA3M086SBEA",   # Real PCE
    "S&P 500",           # S&P 500 index
]

HORIZONS: list[int] = [1, 3, 6, 9, 12, 24]
OOS_START: str = "1980-01-01"
OOS_END: str = "2017-12-01"
N_JOBS: int = 8
_RF_N_JOBS: int = max(1, 48 // N_JOBS)  # 6 — avoids oversubscription on 48-core server
VINTAGE: str = "2018-02"

# CLSS 2021 defaults: p_marx=4, n_factors=4, n_lags=2
_FEAT_KWARGS: dict = dict(n_factors=4, n_lags=2, p_marx=4)

# ---------------------------------------------------------------------------
# All 15 information sets (Table 1, CLSS 2021)
# ---------------------------------------------------------------------------

FEATURE_SPECS: list[FeatureSpec] = [
    # --- Standard transformations (no MARX/MAF/Level) ---
    # F: factors from raw X
    FeatureSpec(use_factors=True,  include_raw_x=False, use_marx=False,
                label="F", **_FEAT_KWARGS),
    # F-X: factors + raw predictors
    FeatureSpec(use_factors=True,  include_raw_x=True,  use_marx=False,
                label="F-X", **_FEAT_KWARGS),
    # X: raw predictors only (no factors)
    FeatureSpec(use_factors=False, include_raw_x=True,  use_marx=False,
                label="X", **_FEAT_KWARGS),

    # --- MARX transformations ---
    # X-MARX: MARX features only
    FeatureSpec(use_factors=False, include_raw_x=False, use_marx=True,
                marx_for_pca=False, label="X-MARX", **_FEAT_KWARGS),
    # F-MARX: factors (from raw X) + MARX columns
    FeatureSpec(use_factors=True,  include_raw_x=False, use_marx=True,
                marx_for_pca=False, label="F-MARX", **_FEAT_KWARGS),
    # F-X-MARX: factors + raw X + MARX (dominant CLSS 2021 info set)
    FeatureSpec(use_factors=True,  include_raw_x=True,  use_marx=True,
                marx_for_pca=False, label="F-X-MARX", **_FEAT_KWARGS),

    # --- MAF (factors from MARX-transformed X) ---
    # MAF: factors from MARX panel only
    FeatureSpec(use_factors=True,  include_raw_x=False, use_marx=True,
                marx_for_pca=True, use_maf=True, label="MAF", **_FEAT_KWARGS),
    # F-X-MAF: F factors (from raw X PCA) + raw X + MAF factors (from MARX PCA)
    # Requires include_f_factors=True to run dual PCA (raw X and MARX panel).
    FeatureSpec(use_factors=True,  include_raw_x=True,  use_marx=True,
                marx_for_pca=True, use_maf=True, include_f_factors=True,
                label="F-X-MAF", **_FEAT_KWARGS),
    # X-MAF: raw X + MAF factors only (no F factors from raw X PCA)
    FeatureSpec(use_factors=True,  include_raw_x=True,  use_marx=True,
                marx_for_pca=True, use_maf=True, label="X-MAF", **_FEAT_KWARGS),

    # --- Level transformations ---
    # F-Level: factors + level variables
    FeatureSpec(use_factors=True,  include_raw_x=False, use_marx=False,
                include_levels=True, label="F-Level", **_FEAT_KWARGS),
    # F-X-Level: factors + raw X + levels
    FeatureSpec(use_factors=True,  include_raw_x=True,  use_marx=False,
                include_levels=True, label="F-X-Level", **_FEAT_KWARGS),
    # X-Level: raw X + levels
    FeatureSpec(use_factors=False, include_raw_x=True,  use_marx=False,
                include_levels=True, label="X-Level", **_FEAT_KWARGS),

    # --- MARX + Level combinations ---
    # F-MARX-Level: factors + MARX + levels
    FeatureSpec(use_factors=True,  include_raw_x=False, use_marx=True,
                marx_for_pca=False, include_levels=True,
                label="F-MARX-Level", **_FEAT_KWARGS),
    # F-X-MARX-Level: factors + raw X + MARX + levels
    FeatureSpec(use_factors=True,  include_raw_x=True,  use_marx=True,
                marx_for_pca=False, include_levels=True,
                label="F-X-MARX-Level", **_FEAT_KWARGS),
    # X-MARX-Level: MARX + levels (no factors)
    FeatureSpec(use_factors=False, include_raw_x=False, use_marx=True,
                marx_for_pca=False, include_levels=True,
                label="X-MARX-Level", **_FEAT_KWARGS),
]

# ---------------------------------------------------------------------------
# Model spec: RF calibrated to CLSS 2021
# ---------------------------------------------------------------------------

RF_SPEC = ModelSpec(
    model_cls=RFModel,
    model_kwargs=dict(
        n_estimators=500,
        min_samples_leaf_grid=[5],   # fixed=5, no CV (matches CLSS 2021 ranger min.node.size=5)
        max_features=1 / 3,
        max_samples=0.75,
        cv_folds=5,
        rf_n_jobs=_RF_N_JOBS,        # 6 per worker to avoid oversubscription on 48-core server
    ),
    regularization=Regularization.NONE,
    cv_scheme=CVScheme.KFOLD(k=5),
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
    """Load FRED-MD 2018-02 vintage and return (panel_stationary, panel_levels)."""
    log.info("Loading FRED-MD vintage=%s ...", VINTAGE)
    try:
        mf_levels = load_fred_md(vintage=VINTAGE)
        log.info(
            "Loaded vintage %s: %d obs x %d vars, %s to %s",
            VINTAGE,
            *mf_levels.data.shape,
            mf_levels.data.index.min().strftime("%Y-%m"),
            mf_levels.data.index.max().strftime("%Y-%m"),
        )
    except Exception as exc:
        log.warning(
            "Could not load vintage %s (%s). Falling back to current release.",
            VINTAGE, exc,
        )
        mf_levels = load_fred_md(vintage=None)
        log.info("Loaded current release: %d obs x %d vars", *mf_levels.data.shape)

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
    """Relative RMSFE per feature_set × horizon, averaged over targets.

    Benchmark = feature_set "F". Values < 1 beat the factor-only baseline.
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
    msfe["rel_rmsfe"] = (msfe["msfe"] / msfe["msfe_bench"]) ** 0.5

    summary = (
        msfe.groupby(["feature_set", "horizon"])["rel_rmsfe"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(index="feature_set", columns="horizon", values="rel_rmsfe")
    ordered = [s.label for s in FEATURE_SPECS]
    pivot = pivot.reindex([lbl for lbl in ordered if lbl in pivot.index])
    return pivot


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    run_start = time.time()
    log.info("=== CLSS 2021 paper-faithful replication ===")
    log.info("Vintage: %s | OOS: %s – %s | n_jobs: %d", VINTAGE, OOS_START, OOS_END, N_JOBS)
    log.info("Targets (%d): %s", len(TARGETS), TARGETS)
    log.info("Horizons: %s", HORIZONS)
    log.info("Feature sets (%d): %s", len(FEATURE_SPECS), [s.label for s in FEATURE_SPECS])
    log.info("RF: n_estimators=500, max_features=1/3, max_samples=0.75, fully grown")
    log.info("Results dir: %s", RESULTS_DIR)

    panel_stat, panel_levels = load_data()

    all_result_dfs: list[pd.DataFrame] = []

    for tgt in TARGETS:
        fname = safe_name(tgt)
        out_path = RESULTS_DIR / f"{fname}_results.parquet"

        # Checkpointing: skip if already done
        if out_path.exists():
            log.info("SKIP %s — already exists (%s)", tgt, out_path.name)
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
                log.warning("Empty results for %s.", tgt)
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
    # Combine and summarise
    # ------------------------------------------------------------------

    if not all_result_dfs:
        log.error("No results collected. Exiting.")
        return

    combined = pd.concat(all_result_dfs, ignore_index=True)
    combined_path = RESULTS_DIR / "combined_results.parquet"
    combined.to_parquet(combined_path, index=False)
    log.info("Saved combined: %s  (%d rows)", combined_path, len(combined))

    log.info("\n=== Relative RMSFE (vs F, averaged over targets) ===")
    rmsfe_table = compute_relative_rmsfe(combined)

    if not rmsfe_table.empty:
        pd.set_option("display.float_format", "{:.4f}".format)
        pd.set_option("display.max_columns", None)
        print("\n" + rmsfe_table.to_string())
        print()
        rmsfe_table.to_parquet(RESULTS_DIR / "rmsfe_summary.parquet")
        rmsfe_table.to_csv(RESULTS_DIR / "rmsfe_summary.csv")
        log.info("Saved RMSFE summary to %s", RESULTS_DIR)
    else:
        log.warning("RMSFE table empty.")

    total = time.time() - run_start
    log.info("=== Run complete in %.1f min ===", total / 60)


if __name__ == "__main__":
    main()
