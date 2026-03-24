"""AR(p) benchmark run for CLSS 2021 replication.

Runs direct h-step AR(p) forecasts with BIC-selected lag order for all
11 targets, 6 horizons, OOS 1980-01 to 2017-12.

Results are saved as parquet files in the same directory as the RF run so
that the comparison script can merge them.

Usage: uv run python scripts/clss2021_ar_benchmark.py
Expected runtime: ~10-20 minutes (AR is cheap).
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from macrocast.data import load_fred_md

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
# Parameters — must match clss2021_paper_run.py
# ---------------------------------------------------------------------------

RESULTS_DIR = Path.home() / ".macrocast" / "results" / "clss2021_paper"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TARGETS: list[str] = [
    "INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "PCEPI",
    "TB3MS", "GS10", "WPSFD49207", "M2REAL", "DPCERA3M086SBEA", "S&P 500",
]
HORIZONS: list[int] = [1, 3, 6, 9, 12, 24]
OOS_START: str = "1980-01-01"
OOS_END: str = "2017-12-01"
VINTAGE: str = "2018-02"
P_MAX: int = 12          # maximum AR lag order considered
FEATURE_SET: str = "AR"  # label used in parquet files


def safe_name(tgt: str) -> str:
    return tgt.replace("&", "").replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# BIC-selected direct AR(p) forecast
# ---------------------------------------------------------------------------

def _build_lag_matrix(y: np.ndarray, p: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Build design matrix for direct h-step AR(p) regression.

    Returns X (T-p-h+1, p) and y_lead (T-p-h+1,) where
    y_lead[i] = y[i+p+h-1] and X[i] = [y[i+p-1], ..., y[i]].
    """
    T = len(y)
    n_obs = T - p - h + 1
    if n_obs <= 0:
        return np.empty((0, p)), np.empty(0)

    X = np.column_stack([y[p - 1 + i: T - h - (p - 1 - i)] for i in range(p)])
    y_lead = y[p + h - 1: T]
    # Trim to matching length (safety)
    n = min(len(X), len(y_lead))
    return X[:n], y_lead[:n]


def ar_bic_select(y_train: np.ndarray, h: int, p_max: int = 12) -> int:
    """Select AR lag order p by BIC for direct h-step forecasting."""
    best_bic = np.inf
    best_p = 1
    n_total = len(y_train)

    for p in range(1, min(p_max + 1, n_total - h - 2)):
        X, y_lead = _build_lag_matrix(y_train, p, h)
        n = len(X)
        if n < p + 2:
            continue
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y_lead)
        resid = y_lead - model.predict(X)
        rss = np.sum(resid ** 2)
        if rss <= 0:
            continue
        k = p + 1  # lags + intercept
        bic = n * np.log(rss / n) + k * np.log(n)
        if bic < best_bic:
            best_bic = bic
            best_p = p

    return best_p


def ar_forecast_series(
    y: pd.Series,
    horizons: list[int],
    oos_start: str,
    oos_end: str,
    p_max: int = 12,
) -> pd.DataFrame:
    """Run expanding-window direct AR(p) forecasts for all horizons.

    Parameters
    ----------
    y : pd.Series
        Full stationary target series (DatetimeIndex).
    horizons : list[int]
        Forecast horizons in months.
    oos_start : str
        First OOS evaluation date (yyyy-mm-dd).
    oos_end : str
        Last OOS evaluation date (yyyy-mm-dd).
    p_max : int
        Maximum AR lag order.

    Returns
    -------
    pd.DataFrame with columns [date, horizon, y_hat, y_true, feature_set].
    """
    oos_idx = pd.date_range(oos_start, oos_end, freq="MS")
    y_vals = y.values
    y_dates = y.index

    records: list[dict] = []

    for h in horizons:
        for oos_date in oos_idx:
            # Training data: all observations up to and including oos_date
            train_mask = y_dates <= oos_date
            y_train = y_vals[train_mask]

            if len(y_train) < p_max + h + 5:
                continue

            # True value: h periods ahead of oos_date
            target_date = oos_date + pd.DateOffset(months=h)
            if target_date not in y.index:
                continue
            y_true = y.loc[target_date]

            # Select p and forecast
            try:
                p = ar_bic_select(y_train, h, p_max)
                X_last = y_train[-(p):][::-1].reshape(1, -1)  # most recent p lags
                X_fit, y_lead = _build_lag_matrix(y_train, p, h)
                if len(X_fit) < p + 2:
                    continue
                model = LinearRegression(fit_intercept=True)
                model.fit(X_fit, y_lead)
                y_hat = float(model.predict(X_last)[0])
            except Exception:
                continue

            records.append({
                "date": oos_date,
                "horizon": h,
                "y_hat": y_hat,
                "y_true": float(y_true),
                "feature_set": FEATURE_SET,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    run_start = time.time()
    log.info("=== AR benchmark run for CLSS 2021 ===")
    log.info("Vintage: %s | OOS: %s – %s", VINTAGE, OOS_START, OOS_END)
    log.info("Targets (%d): %s", len(TARGETS), TARGETS)
    log.info("Horizons: %s | p_max: %d", HORIZONS, P_MAX)

    # Load FRED-MD 2018-02 (cached from paper run)
    log.info("Loading FRED-MD vintage=%s ...", VINTAGE)
    try:
        mf_levels = load_fred_md(vintage=VINTAGE)
        log.info(
            "Loaded vintage %s: %d obs x %d vars",
            VINTAGE, *mf_levels.data.shape,
        )
    except Exception as exc:
        log.warning("Could not load vintage %s (%s). Falling back to current.", VINTAGE, exc)
        mf_levels = load_fred_md(vintage=None)

    mf_stat = mf_levels.transform()
    panel: pd.DataFrame = mf_stat.data.copy()
    log.info(
        "Stationary panel: %d obs x %d vars, %s to %s",
        *panel.shape,
        panel.index.min().strftime("%Y-%m"),
        panel.index.max().strftime("%Y-%m"),
    )

    all_dfs: list[pd.DataFrame] = []

    for tgt in TARGETS:
        fname = safe_name(tgt)
        out_path = RESULTS_DIR / f"{fname}_ar_benchmark.parquet"

        if out_path.exists():
            log.info("SKIP %s — AR benchmark already exists", tgt)
            all_dfs.append(pd.read_parquet(out_path))
            continue

        if tgt not in panel.columns:
            log.error("Target '%s' not in panel. Skipping.", tgt)
            continue

        tgt_start = time.time()
        log.info("--- AR benchmark: %s  [%s] ---", tgt, datetime.now().strftime("%H:%M:%S"))

        y = panel[tgt].dropna()

        try:
            result_df = ar_forecast_series(
                y=y,
                horizons=HORIZONS,
                oos_start=OOS_START,
                oos_end=OOS_END,
                p_max=P_MAX,
            )
        except Exception:
            log.exception("ERROR processing AR benchmark for %s", tgt)
            continue

        if result_df.empty:
            log.warning("Empty AR results for %s.", tgt)
            continue

        result_df["target"] = tgt
        result_df.to_parquet(out_path, index=False)
        elapsed = time.time() - tgt_start
        log.info(
            "Saved: %s  (%d rows, %.1f sec)",
            out_path.name, len(result_df), elapsed,
        )
        all_dfs.append(result_df)

    if not all_dfs:
        log.error("No AR results collected.")
        return

    combined = pd.concat(all_dfs, ignore_index=True)
    out = RESULTS_DIR / "ar_benchmark_combined.parquet"
    combined.to_parquet(out, index=False)
    total = time.time() - run_start
    log.info("Saved combined AR benchmark: %s  (%d rows)", out, len(combined))
    log.info("=== AR benchmark complete in %.1f min ===", total / 60)

    # Quick sanity check
    n_oos = 456
    expected = n_oos * len(HORIZONS) * len(TARGETS)
    log.info(
        "Records: %d / %d expected (%.1f%%)",
        len(combined), expected, 100 * len(combined) / expected,
    )


if __name__ == "__main__":
    main()
