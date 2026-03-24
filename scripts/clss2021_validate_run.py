"""Validate a CLSS 2021 paper run — detect silent failures and missing results.

Usage (can run while the main job is still in progress):
  uv run python scripts/clss2021_validate_run.py

Checks:
  1. Missing parquet files (target skipped entirely)
  2. Record count per (target, feature_set, horizon) — flag if < expected
  3. NaN rate in y_hat (silent prediction failures)
  4. ERROR/WARNING lines in run.log
  5. Summary: which targets/specs need re-running
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Expected run parameters (must match clss2021_paper_run.py)
# ---------------------------------------------------------------------------

RESULTS_DIR = Path.home() / ".macrocast" / "results" / "clss2021_paper"
LOG_FILE = RESULTS_DIR / "run.log"

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
OOS_DATES = 456   # 1980-01 to 2017-12 = 456 months
EXPECTED_PER_CELL = OOS_DATES  # 1 record per OOS date per (model, h)

# Tolerance: flag if record count < this fraction of expected
TOLERANCE = 0.95


def safe_name(tgt: str) -> str:
    return tgt.replace("&", "").replace(" ", "_").replace("/", "_")


# ---------------------------------------------------------------------------
# Check 1: missing parquet files
# ---------------------------------------------------------------------------

def check_missing_files() -> list[str]:
    missing = []
    for tgt in TARGETS:
        path = RESULTS_DIR / f"{safe_name(tgt)}_results.parquet"
        if not path.exists():
            missing.append(tgt)
    return missing


# ---------------------------------------------------------------------------
# Check 2: record counts per (feature_set, horizon)
# ---------------------------------------------------------------------------

def check_record_counts(df: pd.DataFrame, tgt: str) -> list[dict]:
    issues = []
    for fs in FEATURE_SETS:
        fs_df = df[df["feature_set"] == fs] if "feature_set" in df.columns else pd.DataFrame()
        if fs_df.empty:
            issues.append({"target": tgt, "feature_set": fs, "horizon": "ALL",
                           "found": 0, "expected": EXPECTED_PER_CELL * len(HORIZONS),
                           "issue": "MISSING spec entirely"})
            continue
        for h in HORIZONS:
            h_df = fs_df[fs_df["horizon"] == h]
            n = len(h_df)
            if n < EXPECTED_PER_CELL * TOLERANCE:
                issues.append({"target": tgt, "feature_set": fs, "horizon": h,
                               "found": n, "expected": EXPECTED_PER_CELL,
                               "issue": f"LOW ({n}/{EXPECTED_PER_CELL})"})
    return issues


# ---------------------------------------------------------------------------
# Check 3: NaN rate in predictions
# ---------------------------------------------------------------------------

def check_nan_rate(df: pd.DataFrame, tgt: str) -> list[dict]:
    issues = []
    if "y_hat" not in df.columns:
        return issues
    for fs in df["feature_set"].unique():
        fs_df = df[df["feature_set"] == fs]
        nan_rate = fs_df["y_hat"].isna().mean()
        if nan_rate > 0.01:  # > 1% NaN predictions is a problem
            issues.append({"target": tgt, "feature_set": fs,
                           "nan_pct": f"{nan_rate*100:.1f}%"})
    return issues


# ---------------------------------------------------------------------------
# Check 4: log errors
# ---------------------------------------------------------------------------

def check_log_errors() -> dict:
    if not LOG_FILE.exists():
        return {"status": "LOG NOT FOUND"}

    with open(LOG_FILE) as f:
        lines = f.readlines()

    errors = [l.strip() for l in lines if "ERROR" in l]
    cell_fails = [l.strip() for l in lines if "Cell failed" in l]
    warnings = [l.strip() for l in lines if "WARNING" in l and "UserWarning" not in l]
    completed = [l.strip() for l in lines if "done in" in l]

    return {
        "total_lines": len(lines),
        "errors": errors,
        "cell_failures": len(cell_fails),
        "warnings": warnings[:10],
        "completed_targets": len(completed),
        "last_line": lines[-1].strip() if lines else "",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("CLSS 2021 Paper Run Validation")
    print(f"Results dir: {RESULTS_DIR}")
    print("=" * 60)

    # --- Check 1: missing files ---
    missing = check_missing_files()
    completed_targets = [t for t in TARGETS if t not in missing]
    print(f"\n[1] Parquet files: {len(completed_targets)}/{len(TARGETS)} targets complete")
    if missing:
        print(f"    MISSING: {missing}")
    else:
        print("    All targets have results files.")

    # --- Check 4: log ---
    log_info = check_log_errors()
    print(f"\n[2] Log analysis ({log_info.get('total_lines', 0)} lines)")
    print(f"    Completed targets: {log_info.get('completed_targets', 0)}/{len(TARGETS)}")
    print(f"    Cell failures: {log_info.get('cell_failures', 0)}")
    print(f"    Last log line: {log_info.get('last_line', '')[:80]}")

    if log_info.get("errors"):
        print(f"    ERRORS ({len(log_info['errors'])}):")
        for e in log_info["errors"][:5]:
            print(f"      {e[:100]}")
    if log_info.get("warnings"):
        print(f"    Warnings (sample):")
        for w in log_info["warnings"][:3]:
            print(f"      {w[:100]}")

    # --- Check 2 & 3: record counts and NaN ---
    all_count_issues = []
    all_nan_issues = []

    print(f"\n[3] Record counts and NaN check (completed targets only)")
    for tgt in completed_targets:
        path = RESULTS_DIR / f"{safe_name(tgt)}_results.parquet"
        df = pd.read_parquet(path)

        count_issues = check_record_counts(df, tgt)
        nan_issues = check_nan_rate(df, tgt)

        all_count_issues.extend(count_issues)
        all_nan_issues.extend(nan_issues)

        total_records = len(df)
        expected_total = len(FEATURE_SETS) * len(HORIZONS) * EXPECTED_PER_CELL
        status = "✓" if not count_issues else f"ISSUES ({len(count_issues)})"
        print(f"    {tgt:25s}: {total_records:6d}/{expected_total} records  [{status}]")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if not all_count_issues and not all_nan_issues and not missing:
        print("All checks passed. Run looks healthy.")
    else:
        if all_count_issues:
            print(f"\nRecord count issues ({len(all_count_issues)}):")
            issue_df = pd.DataFrame(all_count_issues)
            # Group by target to show which ones need re-running
            for tgt, grp in issue_df.groupby("target"):
                print(f"  {tgt}: {len(grp)} spec-horizon cells short")
                for _, row in grp.head(3).iterrows():
                    print(f"    {row['feature_set']} h={row['horizon']}: {row['issue']}")

        if all_nan_issues:
            print(f"\nHigh NaN rate in predictions ({len(all_nan_issues)}):")
            for issue in all_nan_issues:
                print(f"  {issue['target']} / {issue['feature_set']}: {issue['nan_pct']} NaN")

        if missing:
            print(f"\nMissing targets (need full re-run): {missing}")

    # --- How to re-run specific targets ---
    bad_targets = list({i["target"] for i in all_count_issues} |
                       {i["target"] for i in all_nan_issues} |
                       set(missing))
    if bad_targets:
        print(f"\nTo re-run failing targets, delete their parquet files:")
        for tgt in bad_targets:
            p = RESULTS_DIR / f"{safe_name(tgt)}_results.parquet"
            print(f"  rm {p}")
        print("Then restart clss2021_paper_run.py — checkpointing will skip healthy targets.")


if __name__ == "__main__":
    main()
