"""CLSS 2021 paper-faithful figure reproduction.

Reproduces the key figures/tables from Coulombe, Leroux, Stevanovic, Surprenant
(2021), "Macroeconomic Data Transformations Matter", IJF 37(4):1338-1354.

Figures produced (saved to ~/.macrocast/results/clss2021_paper/figures/):
  fig1_rmsfe_vs_ar.png        — Table 3 equivalent: RF relative RMSFE vs AR,
                                 heatmap, 15 specs × 6 horizons
  fig2_rmsfe_vs_f.png         — Relative RMSFE vs RF-F baseline (transformation
                                 gains), heatmap, 15 specs × 6 horizons
  fig3_best_spec_by_horizon.png — Bar chart: best spec improvement per horizon
  fig4_rmsfe_by_target_h1.png  — Per-target relative RMSFE at h=1 (grouped bars)
  fig5_rmsfe_by_target_h12.png — Per-target relative RMSFE at h=12

Usage: uv run python scripts/clss2021_paper_figures.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

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
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Presentation order matching CLSS 2021 Table 1 / Table 3
FEATURE_ORDER: list[str] = [
    "F",
    "F-X",
    "X",
    "X-MARX",
    "F-MARX",
    "F-X-MARX",
    "MAF",
    "F-X-MAF",
    "X-MAF",
    "F-Level",
    "F-X-Level",
    "X-Level",
    "F-MARX-Level",
    "F-X-MARX-Level",
    "X-MARX-Level",
]

HORIZONS: list[int] = [1, 3, 6, 9, 12, 24]

# Colour palette consistent with IJF papers
PALETTE = {
    "red":   "#d62728",
    "green": "#2ca02c",
    "blue":  "#1f77b4",
    "gray":  "#7f7f7f",
    "orange": "#ff7f0e",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_rf(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "combined_results.parquet"
    if not path.exists():
        raise FileNotFoundError(f"combined_results.parquet not found at {path}")
    df = pd.read_parquet(path)
    log.info("RF results loaded: %d rows, %d targets", len(df), df["target"].nunique())
    return df


def load_ar(results_dir: Path) -> pd.DataFrame:
    path = results_dir / "ar_benchmark_combined.parquet"
    if not path.exists():
        raise FileNotFoundError(f"ar_benchmark_combined.parquet not found at {path}")
    df = pd.read_parquet(path)
    log.info("AR benchmark loaded: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------


def compute_rmsfe_vs_ar(
    rf: pd.DataFrame,
    ar: pd.DataFrame,
) -> pd.DataFrame:
    """Relative RMSFE: RF info sets vs AR(p)-BIC, averaged over targets.

    Returns pivot: feature_set (rows) x horizon (cols), values = sqrt(MSFE_rf / MSFE_ar).
    """
    # RF MSFE per (feature_set, target, horizon)
    rf = rf.copy()
    rf["sq_err"] = (rf["y_hat"] - rf["y_true"]) ** 2
    rf_msfe = (
        rf.groupby(["feature_set", "target", "horizon"])["sq_err"]
        .mean()
        .reset_index(name="msfe_rf")
    )

    # AR MSFE per (target, horizon)
    ar = ar.copy()
    ar["sq_err"] = (ar["y_hat"] - ar["y_true"]) ** 2
    ar_msfe = (
        ar.groupby(["target", "horizon"])["sq_err"]
        .mean()
        .reset_index(name="msfe_ar")
    )

    merged = rf_msfe.merge(ar_msfe, on=["target", "horizon"], how="left")
    merged["rel_rmsfe"] = np.sqrt(merged["msfe_rf"] / merged["msfe_ar"])

    # Average over targets
    avg = (
        merged.groupby(["feature_set", "horizon"])["rel_rmsfe"]
        .mean()
        .reset_index()
    )
    pivot = avg.pivot(index="feature_set", columns="horizon", values="rel_rmsfe")
    pivot = pivot.reindex([f for f in FEATURE_ORDER if f in pivot.index])
    pivot.columns.name = None
    return pivot


def compute_rmsfe_vs_f(rf: pd.DataFrame) -> pd.DataFrame:
    """Relative RMSFE: RF info sets vs RF-F baseline, averaged over targets.

    Returns pivot: feature_set (rows) x horizon (cols), values = sqrt(MSFE / MSFE_F).
    """
    rf = rf.copy()
    rf["sq_err"] = (rf["y_hat"] - rf["y_true"]) ** 2
    msfe = (
        rf.groupby(["feature_set", "target", "horizon"])["sq_err"]
        .mean()
        .reset_index(name="msfe")
    )
    bench = msfe[msfe["feature_set"] == "F"][["target", "horizon", "msfe"]].rename(
        columns={"msfe": "msfe_f"}
    )
    merged = msfe.merge(bench, on=["target", "horizon"], how="left")
    merged["rel_rmsfe"] = np.sqrt(merged["msfe"] / merged["msfe_f"])

    avg = (
        merged.groupby(["feature_set", "horizon"])["rel_rmsfe"]
        .mean()
        .reset_index()
    )
    pivot = avg.pivot(index="feature_set", columns="horizon", values="rel_rmsfe")
    pivot = pivot.reindex([f for f in FEATURE_ORDER if f in pivot.index])
    pivot.columns.name = None
    return pivot


# ---------------------------------------------------------------------------
# Figure 1 & 2: RMSFE heatmaps
# ---------------------------------------------------------------------------

_HEATMAP_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}

# Group dividers: blank row indices (0-based) after which to draw a separator
_GROUP_DIVIDERS = [2, 5, 8, 11]  # after X, after F-X-MARX, after X-MAF, after X-Level


def _heatmap(
    pivot: pd.DataFrame,
    title: str,
    centre: float,
    vmin: float,
    vmax: float,
    cmap: str,
    annot_fmt: str,
    out_path: Path,
    subtitle: str = "",
) -> None:
    """Draw a annotated heatmap and save to out_path."""
    with plt.rc_context(_HEATMAP_RC):
        n_rows, n_cols = pivot.shape
        fig, ax = plt.subplots(figsize=(8, 0.55 * n_rows + 1.6))

        import matplotlib.colors as mcolors
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=centre, vmax=vmax)
        im = ax.imshow(
            pivot.values,
            cmap=cmap,
            norm=norm,
            aspect="auto",
        )

        # Annotate cells
        for i in range(n_rows):
            for j in range(n_cols):
                val = pivot.values[i, j]
                text_color = "white" if abs(val - centre) > 0.07 else "black"
                ax.text(
                    j, i,
                    annot_fmt.format(val),
                    ha="center", va="center",
                    fontsize=8.5, color=text_color, fontweight="bold",
                )

        # Axes ticks
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f"h={h}" for h in pivot.columns], fontweight="bold")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(pivot.index.tolist())
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")

        # Group divider lines
        for div in _GROUP_DIVIDERS:
            if 0 < div < n_rows:
                ax.axhline(div + 0.5, color="white", linewidth=1.8, linestyle="-")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

        # Title
        ax.set_title(title, fontsize=12, fontweight="bold", pad=14)
        if subtitle:
            ax.text(
                0.5, 1.01, subtitle,
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=8, color="#555555",
            )

        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved %s", out_path)


def fig1_rmsfe_vs_ar(pivot: pd.DataFrame) -> None:
    _heatmap(
        pivot=pivot,
        title="Relative RMSFE vs AR(p)-BIC benchmark",
        subtitle="RF Random Forest  |  15 information sets  |  11 targets  |  OOS 1980–2017  |  values < 1.00 beat AR",
        centre=1.0,
        vmin=max(0.65, pivot.values.min() - 0.02),
        vmax=min(1.20, pivot.values.max() + 0.02),
        cmap="RdYlGn_r",
        annot_fmt="{:.3f}",
        out_path=FIGURES_DIR / "fig1_rmsfe_vs_ar.png",
    )


def fig2_rmsfe_vs_f(pivot: pd.DataFrame) -> None:
    _heatmap(
        pivot=pivot,
        title="Relative RMSFE vs RF-F baseline (transformation gains)",
        subtitle="RF Random Forest  |  15 information sets  |  11 targets  |  OOS 1980–2017  |  values < 1.00 beat F",
        centre=1.0,
        vmin=max(0.88, pivot.values.min() - 0.01),
        vmax=min(1.16, pivot.values.max() + 0.01),
        cmap="RdYlGn_r",
        annot_fmt="{:.4f}",
        out_path=FIGURES_DIR / "fig2_rmsfe_vs_f.png",
    )


# ---------------------------------------------------------------------------
# Figure 3: Best spec improvement per horizon (bar chart)
# ---------------------------------------------------------------------------


def fig3_best_spec_by_horizon(
    pivot_vs_ar: pd.DataFrame,
    pivot_vs_f: pd.DataFrame,
) -> None:
    """Grouped bar chart: best-spec relative RMSFE vs AR and vs F per horizon."""
    horizons = list(pivot_vs_ar.columns)
    best_ar = [pivot_vs_ar[h].min() for h in horizons]
    best_f  = [pivot_vs_f[h].min() for h in horizons]
    best_ar_spec = [pivot_vs_ar[h].idxmin() for h in horizons]
    best_f_spec  = [pivot_vs_f[h].idxmin() for h in horizons]

    x = np.arange(len(horizons))
    w = 0.35

    with plt.rc_context(_HEATMAP_RC):
        fig, ax = plt.subplots(figsize=(9, 4.5))

        bars1 = ax.bar(x - w / 2, best_ar, w, label="Best spec vs AR",
                       color=PALETTE["blue"], alpha=0.85, zorder=3)
        bars2 = ax.bar(x + w / 2, best_f,  w, label="Best spec vs RF-F",
                       color=PALETTE["orange"], alpha=0.85, zorder=3)

        # Reference lines
        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=2)

        # Spec name annotations
        for bar, spec in zip(bars1, best_ar_spec):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.005,
                spec, ha="center", va="top",
                fontsize=6.5, color="white", fontweight="bold",
                rotation=90,
            )
        for bar, spec in zip(bars2, best_f_spec):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() - 0.005,
                spec, ha="center", va="top",
                fontsize=6.5, color="white", fontweight="bold",
                rotation=90,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_ylabel("Relative RMSFE (lower = better)")
        ax.set_title(
            "Best information set per horizon — RF (11 targets, OOS 1980–2017)",
            fontweight="bold",
        )
        ax.set_ylim(0.65, 1.15)
        ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.05))
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.legend(framealpha=0.9)

        plt.tight_layout()
        out = FIGURES_DIR / "fig3_best_spec_by_horizon.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Figure 4 & 5: Per-target RMSFE at h=1 and h=12
# ---------------------------------------------------------------------------

# Key specs to highlight (5 representative sets)
_KEY_SPECS = ["F", "F-X-MARX", "MAF", "F-X-MAF", "F-X-MARX-Level"]
_KEY_COLORS = [PALETTE["gray"], PALETTE["blue"], PALETTE["green"],
               PALETTE["orange"], PALETTE["red"]]


def _per_target_bar(
    rf: pd.DataFrame,
    ar: pd.DataFrame,
    horizon: int,
    out_path: Path,
) -> None:
    """Grouped bar chart: relative RMSFE vs AR per target at given horizon."""
    rf_h = rf[rf["horizon"] == horizon].copy()
    rf_h["sq_err"] = (rf_h["y_hat"] - rf_h["y_true"]) ** 2
    rf_msfe = (
        rf_h[rf_h["feature_set"].isin(_KEY_SPECS)]
        .groupby(["feature_set", "target"])["sq_err"]
        .mean()
        .reset_index(name="msfe_rf")
    )

    ar_h = ar[ar["horizon"] == horizon].copy()
    ar_h["sq_err"] = (ar_h["y_hat"] - ar_h["y_true"]) ** 2
    ar_msfe = ar_h.groupby("target")["sq_err"].mean().reset_index(name="msfe_ar")

    merged = rf_msfe.merge(ar_msfe, on="target")
    merged["rel_rmsfe"] = np.sqrt(merged["msfe_rf"] / merged["msfe_ar"])

    targets = sorted(merged["target"].unique())
    x = np.arange(len(targets))
    n_specs = len(_KEY_SPECS)
    w = 0.7 / n_specs

    with plt.rc_context(_HEATMAP_RC):
        fig, ax = plt.subplots(figsize=(12, 4.5))

        for k, (spec, color) in enumerate(zip(_KEY_SPECS, _KEY_COLORS)):
            vals = []
            for tgt in targets:
                sub = merged[(merged["feature_set"] == spec) & (merged["target"] == tgt)]
                vals.append(sub["rel_rmsfe"].values[0] if len(sub) else np.nan)
            offset = (k - n_specs / 2 + 0.5) * w
            ax.bar(x + offset, vals, w, label=spec, color=color, alpha=0.85, zorder=3)

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(targets, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Relative RMSFE vs AR")
        ax.set_title(
            f"Relative RMSFE vs AR benchmark by target  (h={horizon})",
            fontweight="bold",
        )
        ax.grid(axis="y", alpha=0.3, zorder=1)
        ax.legend(framealpha=0.9, fontsize=8, ncol=len(_KEY_SPECS))

        plt.tight_layout()
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved %s", out_path)


# ---------------------------------------------------------------------------
# Figure 6: Transformation gains across horizons (line plot)
# ---------------------------------------------------------------------------


def fig6_transformation_gains(pivot_vs_f: pd.DataFrame) -> None:
    """Line plot: relative RMSFE vs RF-F for key specs across all horizons."""
    key_specs = ["F-X", "F-MARX", "F-X-MARX", "MAF", "F-X-MAF",
                 "F-Level", "F-X-MARX-Level"]
    styles = [
        ("--", "o", PALETTE["gray"]),
        ("-",  "s", PALETTE["blue"]),
        ("-",  "D", PALETTE["blue"]),
        ("-",  "^", PALETTE["green"]),
        ("-",  "v", PALETTE["orange"]),
        (":",  "x", PALETTE["red"]),
        (":",  "P", PALETTE["red"]),
    ]
    horizons = list(pivot_vs_f.columns)

    with plt.rc_context(_HEATMAP_RC):
        fig, ax = plt.subplots(figsize=(9, 4.5))

        for spec, (ls, marker, color) in zip(key_specs, styles):
            if spec not in pivot_vs_f.index:
                continue
            vals = pivot_vs_f.loc[spec, horizons].values
            ax.plot(
                range(len(horizons)), vals,
                linestyle=ls, marker=marker, color=color,
                linewidth=1.6, markersize=6, label=spec, zorder=3,
            )

        ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", zorder=2,
                   label="RF-F baseline")
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"h={h}" for h in horizons])
        ax.set_ylabel("Relative RMSFE vs RF-F")
        ax.set_title(
            "Transformation gains relative to RF-F — RF (11 targets, OOS 1980–2017)",
            fontweight="bold",
        )
        ax.set_ylim(0.93, 1.14)
        ax.grid(alpha=0.3, zorder=1)
        ax.legend(framealpha=0.9, fontsize=8, ncol=2, loc="upper right")

        plt.tight_layout()
        out = FIGURES_DIR / "fig6_transformation_gains.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("=== CLSS 2021 figure reproduction ===")
    log.info("Results dir : %s", RESULTS_DIR)
    log.info("Figures dir : %s", FIGURES_DIR)

    rf = load_rf(RESULTS_DIR)
    ar = load_ar(RESULTS_DIR)

    log.info("Computing relative RMSFE vs AR ...")
    pivot_ar = compute_rmsfe_vs_ar(rf, ar)
    log.info("\n%s", pivot_ar.round(4).to_string())

    log.info("Computing relative RMSFE vs RF-F ...")
    pivot_f = compute_rmsfe_vs_f(rf)
    log.info("\n%s", pivot_f.round(4).to_string())

    log.info("Generating figures ...")
    fig1_rmsfe_vs_ar(pivot_ar)
    fig2_rmsfe_vs_f(pivot_f)
    fig3_best_spec_by_horizon(pivot_ar, pivot_f)
    _per_target_bar(rf, ar, horizon=1,
                    out_path=FIGURES_DIR / "fig4_rmsfe_by_target_h1.png")
    _per_target_bar(rf, ar, horizon=12,
                    out_path=FIGURES_DIR / "fig5_rmsfe_by_target_h12.png")
    fig6_transformation_gains(pivot_f)

    figs = sorted(FIGURES_DIR.glob("*.png"))
    log.info("=== Done — %d figures written ===", len(figs))
    for f in figs:
        log.info("  %s", f.name)


if __name__ == "__main__":
    main()
