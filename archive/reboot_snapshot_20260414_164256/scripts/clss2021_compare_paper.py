"""Compare CLSS 2021 replication results against paper benchmarks.

Loads completed RF parquets + AR benchmark, computes:
  1. Relative RMSFE (vs AR) per info set x horizon — averaged over completed targets
  2. Best spec per horizon (lowest relative RMSFE)
  3. Side-by-side comparison with paper's key finding:
     F-X-MARX dominates at short horizons; Level specs matter at long horizons

Usage: uv run python scripts/clss2021_compare_paper.py
Can run at any point during the paper run — partial results shown.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

RESULTS_DIR = Path.home() / ".macrocast" / "results" / "clss2021_paper"

TARGETS = [
    "INDPRO", "PAYEMS", "UNRATE", "CPIAUCSL", "PCEPI",
    "TB3MS", "GS10", "WPSFD49207", "M2REAL", "DPCERA3M086SBEA", "S&P 500",
]
FEATURE_SETS = [
    "F", "F-X", "X", "X-MARX", "F-MARX", "F-X-MARX",
    "MAF", "F-X-MAF", "X-MAF",
    "F-Level", "F-X-Level", "X-Level",
    "F-MARX-Level", "F-X-MARX-Level", "X-MARX-Level",
]
HORIZONS = [1, 3, 6, 9, 12, 24]


def safe_name(tgt: str) -> str:
    return tgt.replace("&", "").replace(" ", "_").replace("/", "_")


def load_rf_results() -> pd.DataFrame:
    """Load all completed RF parquets."""
    dfs = []
    for tgt in TARGETS:
        path = RESULTS_DIR / f"{safe_name(tgt)}_results.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if "target" not in df.columns:
                df["target"] = tgt
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def load_ar_results() -> pd.DataFrame:
    """Load AR benchmark parquet."""
    path = RESULTS_DIR / "ar_benchmark_combined.parquet"
    if path.exists():
        return pd.read_parquet(path)
    # Try per-target files
    dfs = []
    for tgt in TARGETS:
        p = RESULTS_DIR / f"{safe_name(tgt)}_ar_benchmark.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if "target" not in df.columns:
                df["target"] = tgt
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def relative_rmsfe_vs_ar(
    rf: pd.DataFrame,
    ar: pd.DataFrame,
) -> pd.DataFrame:
    """Compute relative RMSFE (RF info set / AR) averaged over shared targets.

    Returns DataFrame indexed by feature_set, columns = horizons.
    """
    rf = rf.copy()
    ar = ar.copy()

    # Align targets
    shared_targets = set(rf["target"].unique()) & set(ar["target"].unique())
    if not shared_targets:
        return pd.DataFrame()

    rf = rf[rf["target"].isin(shared_targets)]
    ar = ar[ar["target"].isin(shared_targets)]

    rf["sq_err"] = (rf["y_hat"] - rf["y_true"]) ** 2
    ar["sq_err"] = (ar["y_hat"] - ar["y_true"]) ** 2

    # MSFE per (feature_set, target, horizon)
    rf_msfe = (
        rf.groupby(["feature_set", "target", "horizon"])["sq_err"]
        .mean()
        .reset_index()
        .rename(columns={"sq_err": "msfe_rf"})
    )

    # AR MSFE per (target, horizon)
    ar_msfe = (
        ar.groupby(["target", "horizon"])["sq_err"]
        .mean()
        .reset_index()
        .rename(columns={"sq_err": "msfe_ar"})
    )

    merged = rf_msfe.merge(ar_msfe, on=["target", "horizon"], how="inner")
    merged["rel_rmsfe"] = (merged["msfe_rf"] / merged["msfe_ar"]) ** 0.5

    # Average over targets
    summary = (
        merged.groupby(["feature_set", "horizon"])["rel_rmsfe"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(index="feature_set", columns="horizon", values="rel_rmsfe")
    pivot = pivot.reindex([fs for fs in FEATURE_SETS if fs in pivot.index])
    return pivot


def relative_rmsfe_vs_f(rf: pd.DataFrame) -> pd.DataFrame:
    """Relative RMSFE vs F spec (no AR needed).

    Useful for intermediate check when AR run isn't complete yet.
    """
    rf = rf.copy()
    rf["sq_err"] = (rf["y_hat"] - rf["y_true"]) ** 2

    msfe = (
        rf.groupby(["feature_set", "target", "horizon"])["sq_err"]
        .mean()
        .reset_index()
        .rename(columns={"sq_err": "msfe"})
    )

    bench = (
        msfe[msfe["feature_set"] == "F"][["target", "horizon", "msfe"]]
        .rename(columns={"msfe": "msfe_f"})
    )
    if bench.empty:
        return pd.DataFrame()

    merged = msfe.merge(bench, on=["target", "horizon"], how="left")
    merged["rel_rmsfe"] = (merged["msfe"] / merged["msfe_f"]) ** 0.5

    summary = (
        merged.groupby(["feature_set", "horizon"])["rel_rmsfe"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(index="feature_set", columns="horizon", values="rel_rmsfe")
    pivot = pivot.reindex([fs for fs in FEATURE_SETS if fs in pivot.index])
    return pivot


def print_table(df: pd.DataFrame, title: str, benchmark_label: str) -> None:
    if df.empty:
        print(f"\n{title}: NO DATA")
        return
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"Relative RMSFE vs {benchmark_label}  (< 1.00 = better than benchmark)")
    print(f"Targets included: {df.attrs.get('n_targets', '?')}")
    print(f"{'='*70}")
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(df.to_string())


def print_best_specs(df: pd.DataFrame) -> None:
    if df.empty:
        return
    print("\n--- Best spec per horizon (lowest relative RMSFE) ---")
    for h in HORIZONS:
        if h not in df.columns:
            continue
        col = df[h].dropna()
        if col.empty:
            continue
        best = col.idxmin()
        val = col.min()
        print(f"  h={h:2d}: {best:20s}  {val:.4f}")


def print_paper_findings() -> None:
    """Print the qualitative findings from CLSS 2021 for comparison."""
    print("\n--- CLSS 2021 paper key findings (Table 2/3 summary) ---")
    print("  Short horizons (h=1,3):  MARX and Level transformations dominate")
    print("  F-X-MARX:                best or near-best spec in most cases")
    print("  Level specs:             gain importance at h=12, h=24")
    print("  MAF:                     consistently strong across horizons")
    print("  X alone (raw, no factors): typically worse than F or F-X")
    print("  Overall: transformations matter, especially for RF at short h")


def main() -> None:
    print("=" * 70)
    print("CLSS 2021 Mid-Run Comparison Report")
    print("=" * 70)

    # Load data
    rf = load_rf_results()
    ar = load_ar_results()

    completed_rf = sorted(rf["target"].unique().tolist()) if not rf.empty else []
    completed_ar = sorted(ar["target"].unique().tolist()) if not ar.empty else []

    print(f"\nRF results:  {len(completed_rf)}/{len(TARGETS)} targets")
    if completed_rf:
        print(f"  {completed_rf}")
    print(f"AR benchmark: {len(completed_ar)}/{len(TARGETS)} targets")
    if completed_ar:
        print(f"  {completed_ar}")

    if rf.empty:
        print("\nNo RF results yet. Re-run after first target completes (~2h).")
        return

    # --- Table 1: vs AR (if available) ---
    if not ar.empty:
        shared = set(completed_rf) & set(completed_ar)
        tbl_ar = relative_rmsfe_vs_ar(rf, ar)
        tbl_ar.attrs["n_targets"] = len(shared)
        print_table(
            tbl_ar,
            "TABLE: Relative RMSFE (RF info sets vs AR benchmark)",
            "AR(p)-BIC",
        )
        print_best_specs(tbl_ar)
    else:
        print("\n[AR benchmark not yet complete — skipping vs-AR table]")

    # --- Table 2: vs F (always available if any RF done) ---
    tbl_f = relative_rmsfe_vs_f(rf)
    tbl_f.attrs["n_targets"] = len(completed_rf)
    print_table(
        tbl_f,
        "TABLE: Relative RMSFE (RF info sets vs RF-F baseline)",
        "RF-F (factors only)",
    )
    print_best_specs(tbl_f)

    # --- Paper findings for comparison ---
    print_paper_findings()

    # --- Coverage summary ---
    print(f"\n{'='*70}")
    print("Coverage summary")
    print(f"{'='*70}")
    if not rf.empty:
        for tgt in completed_rf:
            sub = rf[rf["target"] == tgt]
            n_specs = sub["feature_set"].nunique() if "feature_set" in sub.columns else "?"
            n_rows = len(sub)
            expected = 15 * 6 * 456
            pct = 100 * n_rows / expected
            print(f"  {tgt:25s}: {n_rows:6d} rows ({n_specs} specs, {pct:.0f}% of expected)")


if __name__ == "__main__":
    main()
