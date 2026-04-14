"""CLSS 2021 paper-faithful figure reproduction (v2).

Reproduces Figs. 1, 3, 4 from Coulombe, Leroux, Stevanovic, Surprenant (2021),
"Macroeconomic Data Transformations Matter", IJF 37(4):1338-1354.

Methodology (Eqs. 11-12 from the paper):
  - Pseudo-OOS R² at time t: R²_{t,h,v,m} = 1 - ê²_{t,v,m} / Var_OOS(y_{t+h,v})
  - Marginal effect of feature f: regress pairwise ΔR²_t on time FE, get HAC α_f
  - 95% confidence bands: Newey-West (sqrt(T) lags)

Figures produced (saved to ~/.macrocast/results/clss2021_paper/figures/):
  fig_marx_effects_rf.png  — Fig. 1 style: MARX marginal effects, RF, 6 horizons × 10 targets
  fig_maf_effects_rf.png   — Fig. 4 style: MAF marginal effects, RF, 6 horizons × 10 targets
  fig_vi_stacked.png       — Fig. 3 style: VI stacked bars, F-X-MARX RF, by horizon, 4 targets

Usage: uv run python scripts/clss2021_paper_figures_v2.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
# Constants matching paper
# ---------------------------------------------------------------------------

HORIZONS: list[int] = [1, 3, 6, 9, 12, 24]

# Paper uses short labels for targets; we map our column names to paper names.
# Paper targets: INDPRO, EMP, UNRATE, INCOME, CONS, RETAIL, HOUST, M2, CPI, PPI (10 targets)
# We have 11 targets (plus S&P 500, GS10, TB3MS). Map closest equivalents.
TARGET_LABELS: dict[str, str] = {
    "INDPRO": "INDPRO",
    "PAYEMS": "EMP",
    "UNRATE": "UNRATE",
    "DPCERA3M086SBEA": "INCOME",
    "PCEPI": "CONS",       # PCE deflator ~ CONS inflation
    "CPIAUCSL": "CPI",
    "WPSFD49207": "PPI",
    "M2REAL": "M2",
    "GS10": "GS10",
    "TB3MS": "TB3MS",
    "S&P 500": "SP500",
}

# Colors matching paper (approximate; paper uses color wheel for 10 targets)
TARGET_COLORS: dict[str, str] = {
    "INDPRO":           "#1f77b4",   # blue
    "PAYEMS":           "#2ca02c",   # green
    "UNRATE":           "#d62728",   # red
    "DPCERA3M086SBEA":  "#9467bd",   # purple
    "PCEPI":            "#8c564b",   # brown
    "CPIAUCSL":         "#e377c2",   # pink
    "WPSFD49207":       "#7f7f7f",   # gray
    "M2REAL":           "#bcbd22",   # yellow-green
    "GS10":             "#17becf",   # cyan
    "TB3MS":            "#ff7f0e",   # orange
    "S&P 500":          "#aec7e8",   # light blue
}

# MARX matched pairs: (with_MARX_spec, base_spec)
MARX_PAIRS: list[tuple[str, str]] = [
    ("F-MARX", "F"),
    ("F-X-MARX", "F-X"),
    ("X-MARX", "X"),
]

# MAF matched pairs: (with_MAF_spec, base_spec)
MAF_PAIRS: list[tuple[str, str]] = [
    ("MAF", "F"),
    ("F-X-MAF", "F-X"),
    ("X-MAF", "X"),
]

# VI figure: 4 targets shown in paper (Income, Employment, Inflation, M2)
VI_TARGETS: list[tuple[str, str]] = [
    ("DPCERA3M086SBEA", "Income"),
    ("PAYEMS",          "Employment"),
    ("CPIAUCSL",        "Inflation"),
    ("M2REAL",          "M2 money Stock"),
]

# VI group colors (matching paper: AR=teal, AR-MARX=orange, Factors=black, MARX=red, X=blue)
VI_GROUPS: dict[str, str] = {
    "AR":      "#2ca02c",   # green (AR lags)
    "Factors": "#000000",   # black
    "MARX":    "#d62728",   # red
    "X":       "#1f77b4",   # blue
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load combined results and AR benchmark."""
    log.info("Loading combined_results.parquet ...")
    df = pd.read_parquet(RESULTS_DIR / "combined_results.parquet")
    log.info("  Loaded %d rows, %d feature_sets", len(df), df["feature_set"].nunique())

    log.info("Loading ar_benchmark_combined.parquet ...")
    ar = pd.read_parquet(RESULTS_DIR / "ar_benchmark_combined.parquet")
    log.info("  Loaded %d AR rows", len(ar))

    return df, ar


# ---------------------------------------------------------------------------
# Pseudo-OOS R² computation (Eq. 11)
# ---------------------------------------------------------------------------


def compute_pseudo_r2(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute pseudo-OOS R² per (target, horizon, feature_set, forecast_date).

    R²_{t,h,v,m} = 1 - ê²_{t,v,m} / Var_OOS(y_{t+h,v})

    Var_OOS is the variance of y_true over the full OOS period for (target, horizon).
    """
    df = df.copy()
    df["sq_err"] = (df["y_hat"] - df["y_true"]) ** 2

    # Compute OOS variance per (target, horizon)
    var_oos = (
        df.groupby(["target", "horizon"])["y_true"]
        .var(ddof=0)
        .rename("var_oos")
        .reset_index()
    )
    df = df.merge(var_oos, on=["target", "horizon"], how="left")

    # Pseudo-R²: bounded to avoid extreme outliers
    df["pseudo_r2"] = 1.0 - df["sq_err"] / df["var_oos"]
    df["pseudo_r2"] = df["pseudo_r2"].clip(-10, 1)   # clip extreme negatives

    return df


# ---------------------------------------------------------------------------
# Marginal effects computation (Eq. 12)
# ---------------------------------------------------------------------------


def compute_marginal_effects(
    r2_df: pd.DataFrame,
    pairs: list[tuple[str, str]],
    feature_name: str,
) -> pd.DataFrame:
    """Compute marginal OOS-R² contribution of a feature (MARX or MAF).

    For each (horizon h, target v):
      - Form pairwise differences ΔR²_t = R²_{with_feature,t} - R²_{without_feature,t}
        for each matched pair (e.g. F-MARX vs F)
      - Run OLS: ΔR²_t = α_f + ε_t with Newey-West HAC SE (lags = floor(sqrt(T)))
      - Record α_f and 95% CI for each (h, v)

    Returns DataFrame with columns: horizon, target, alpha, ci_lo, ci_hi
    """
    records = []

    for h in HORIZONS:
        for tgt in r2_df["target"].unique():
            sub = r2_df[(r2_df["horizon"] == h) & (r2_df["target"] == tgt)]
            if sub.empty:
                continue

            # Collect pairwise delta-R² time series
            deltas: list[pd.Series] = []

            for feat_spec, base_spec in pairs:
                feat_data = (
                    sub[sub["feature_set"] == feat_spec]
                    .set_index("forecast_date")["pseudo_r2"]
                )
                base_data = (
                    sub[sub["feature_set"] == base_spec]
                    .set_index("forecast_date")["pseudo_r2"]
                )
                # Align on common dates
                common = feat_data.index.intersection(base_data.index)
                if len(common) < 20:
                    continue
                delta = feat_data.loc[common] - base_data.loc[common]
                deltas.append(delta.sort_index())

            if not deltas:
                continue

            # Pool all pairwise deltas
            pooled = pd.concat(deltas).sort_index()

            # OLS with intercept only (equivalent to mean after controlling for pair FE
            # since pairs are symmetric in time). HAC SE via Newey-West.
            T = len(pooled)
            nw_lags = max(1, int(np.sqrt(T)))

            y = pooled.values
            X = np.ones((T, 1))

            try:
                ols = sm.OLS(y, X).fit(
                    cov_type="HAC",
                    cov_kwds={"maxlags": nw_lags, "use_correction": True},
                )
                alpha = ols.params[0]
                ci_lo, ci_hi = ols.conf_int(alpha=0.05).iloc[0]
            except Exception:
                alpha = float(pooled.mean())
                std = float(pooled.std()) / np.sqrt(T)
                ci_lo = alpha - 1.96 * std
                ci_hi = alpha + 1.96 * std

            records.append({
                "horizon": h,
                "target": tgt,
                "feature": feature_name,
                "alpha": alpha,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Fig 1 / Fig 4 style: marginal effects dot plot
# ---------------------------------------------------------------------------


def fig_marginal_effects(
    effects_df: pd.DataFrame,
    feature_name: str,
    out_path: Path,
    x_lim: tuple[float, float] = (-0.4, 0.4),
) -> None:
    """Reproduce Fig. 1 / Fig. 4 style: dot plot of marginal OOS-R² effects.

    Rows = horizons (H=1,3,6,9,12,24), one column (RF only).
    Each row: colored dots per target with horizontal 95% CI bars.
    """
    log.info("Building %s marginal effects figure ...", feature_name)

    targets_in_data = sorted(effects_df["target"].unique())
    n_h = len(HORIZONS)

    fig, axes = plt.subplots(
        n_h, 1,
        figsize=(6, n_h * 1.5),
        sharey=False,
        gridspec_kw={"hspace": 0.10},
    )

    # Vertical dot positions within each subplot (one per target, ordered bottom-to-top)
    # Paper shows targets stacked vertically within each horizon row
    y_pos = {tgt: i for i, tgt in enumerate(targets_in_data)}
    n_tgt = len(targets_in_data)

    for row_idx, h in enumerate(HORIZONS):
        ax = axes[row_idx]
        sub = effects_df[effects_df["horizon"] == h]

        # Draw vertical zero line (thick dashed, as in paper)
        ax.axvline(0, color="black", lw=1.2, ls="--", zorder=1)

        # Light vertical grid lines
        for xv in np.arange(-0.4, 0.5, 0.2):
            ax.axvline(xv, color="lightgray", lw=0.5, zorder=0)

        for _, row in sub.iterrows():
            tgt = row["target"]
            if tgt not in y_pos:
                continue
            yv = y_pos[tgt]
            color = TARGET_COLORS.get(tgt, "#333333")
            alpha_val = row["alpha"]
            ci_lo = row["ci_lo"]
            ci_hi = row["ci_hi"]

            # Error bar (horizontal CI)
            ax.plot(
                [ci_lo, ci_hi], [yv, yv],
                color=color, lw=1.5, solid_capstyle="butt", zorder=2,
            )
            # Dot
            ax.plot(
                alpha_val, yv,
                marker="o", ms=5, color=color, zorder=3, mec="white", mew=0.4,
            )

        # Horizon label on the left (e.g. "H=1")
        ax.set_ylabel(f"H={h}", rotation=0, labelpad=28, fontsize=9, va="center")
        ax.yaxis.set_label_position("left")

        ax.set_xlim(x_lim)
        ax.set_ylim(-0.8, n_tgt - 0.2)
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        # X-axis ticks only on last row
        if row_idx < n_h - 1:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel(r"$\alpha^{(h,v)}$", fontsize=9)
            ax.tick_params(axis="x", labelsize=8)

    # Column header label
    axes[0].set_title("Random Forest", fontsize=10, pad=4)

    # Legend
    legend_handles = [
        mpatches.Patch(color=TARGET_COLORS.get(tgt, "#333333"),
                       label=TARGET_LABELS.get(tgt, tgt))
        for tgt in targets_in_data
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=min(len(targets_in_data), 6),
        fontsize=7,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        columnspacing=0.8,
        handlelength=0.8,
    )

    fig.suptitle(
        f"Distribution of {feature_name} Marginal Effects (Average Targets)\n"
        r"$\alpha^{(h,v)}$ from Eq. (12), done by $(h, v)$ subsets. "
        "SEs are HAC (Newey-West). 95% confidence bands.",
        fontsize=8, y=1.01,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Fig 3 style: Variable Importance stacked bar chart
# ---------------------------------------------------------------------------


def classify_fi_key(key: str) -> str:
    """Map a feature importance key to a VI group."""
    if key.startswith("y_lag"):
        return "AR"
    if key.startswith("factor"):
        return "Factors"
    if key.startswith("MARX"):
        return "MARX"
    if key.startswith("X_"):
        return "X"
    if key.startswith("level"):
        return "X"   # level features treated as X
    return "X"


def compute_vi_by_group(
    df: pd.DataFrame,
    target: str,
    feature_spec: str = "F-X-MARX",
) -> pd.DataFrame:
    """Compute mean normalized VI by group for each horizon.

    Returns DataFrame: index=horizon, columns=VI groups, values=normalized shares.
    """
    sub = df[(df["target"] == target) & (df["feature_set"] == feature_spec)].copy()
    sub = sub[sub["feature_importances"].notna()]

    if sub.empty:
        return pd.DataFrame()

    rows = []
    for h in HORIZONS:
        h_sub = sub[sub["horizon"] == h]
        if h_sub.empty:
            continue

        # Parse and average feature importances across OOS periods
        fi_list: list[dict] = [json.loads(s) for s in h_sub["feature_importances"]]
        # Aggregate: mean importance per key across time
        all_keys: set[str] = set().union(*[d.keys() for d in fi_list])
        mean_fi: dict[str, float] = {}
        for k in all_keys:
            vals = [d.get(k, 0.0) for d in fi_list]
            mean_fi[k] = float(np.mean(vals))

        # Group and normalize
        group_sums: dict[str, float] = {g: 0.0 for g in VI_GROUPS}
        for k, v in mean_fi.items():
            g = classify_fi_key(k)
            if g in group_sums:
                group_sums[g] += v

        total = sum(group_sums.values())
        if total == 0:
            continue

        normalized = {g: v / total for g, v in group_sums.items()}
        normalized["horizon"] = h
        rows.append(normalized)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).set_index("horizon")
    result = result.reindex(index=[h for h in HORIZONS if h in result.index])
    return result


def fig_vi_stacked(
    df: pd.DataFrame,
    out_path: Path,
    feature_spec: str = "F-X-MARX",
) -> None:
    """Reproduce Fig. 3 style: stacked VI bars, one subplot per target.

    Paper shows F-X-MARX RF, H=12 (path-avg horizons 1-12 + direct).
    We show F-X-MARX RF, direct method, for each OOS horizon.
    """
    log.info("Building VI stacked bar figure (%s) ...", feature_spec)

    n_panels = len(VI_TARGETS)
    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 8),
        sharey=True,
        gridspec_kw={"hspace": 0.35, "wspace": 0.15},
    )
    axes_flat = axes.flatten()

    group_order = ["AR", "Factors", "MARX", "X"]

    for panel_idx, (tgt_col, tgt_label) in enumerate(VI_TARGETS):
        ax = axes_flat[panel_idx]

        vi = compute_vi_by_group(df, tgt_col, feature_spec=feature_spec)

        if vi.empty:
            ax.set_title(tgt_label, fontsize=10)
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")
            continue

        horizons_present = vi.index.tolist()
        x = np.arange(len(horizons_present))
        width = 0.7

        bottoms = np.zeros(len(horizons_present))
        for grp in group_order:
            if grp not in vi.columns:
                continue
            heights = vi[grp].values
            ax.bar(
                x, heights, width,
                bottom=bottoms,
                color=VI_GROUPS[grp],
                label=grp,
                edgecolor="white",
                linewidth=0.3,
            )
            bottoms += heights

        ax.set_title(tgt_label, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([str(h) for h in horizons_present], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, len(horizons_present) - 0.5)
        ax.axhline(1.0, color="black", lw=0.5, ls="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2f}"))
        if panel_idx in (0, 2):
            ax.set_ylabel("Normalized VI", fontsize=9)
        ax.set_xlabel("Horizon h", fontsize=9)

    # Shared legend
    legend_handles = [
        mpatches.Patch(color=VI_GROUPS[g], label=g) for g in group_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(group_order),
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
    )

    fig.suptitle(
        f"Variable Importance — RF {feature_spec} (direct)\n"
        "Group VI normalized to 1 per horizon. Groups: AR (y lags), "
        "Factors (PCA), MARX (MARX cols), X (raw predictors).",
        fontsize=9, y=1.01,
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Combined summary figure (Table 3 equivalent): relative RMSFE heatmap
# ---------------------------------------------------------------------------


def compute_rmsfe_vs_ar(df: pd.DataFrame, ar: pd.DataFrame) -> pd.DataFrame:
    """Relative RMSFE vs AR benchmark per (feature_set, horizon), averaged over targets."""
    df = df.copy()
    ar = ar.copy()

    # Merge RF forecasts with AR forecasts on (target, horizon, forecast_date)
    df["sq_err"] = (df["y_hat"] - df["y_true"]) ** 2
    ar["sq_err_ar"] = (ar["y_hat"] - ar["y_true"]) ** 2

    ar_msfe = (
        ar.groupby(["target", "horizon"])["sq_err_ar"]
        .mean()
        .reset_index()
        .rename(columns={"sq_err_ar": "msfe_ar"})
    )

    rf_msfe = (
        df.groupby(["feature_set", "target", "horizon"])["sq_err"]
        .mean()
        .reset_index()
        .rename(columns={"sq_err": "msfe"})
    )

    merged = rf_msfe.merge(ar_msfe, on=["target", "horizon"], how="left")
    merged["rel_rmsfe"] = (merged["msfe"] / merged["msfe_ar"]) ** 0.5

    summary = (
        merged.groupby(["feature_set", "horizon"])["rel_rmsfe"]
        .mean()
        .reset_index()
    )
    pivot = summary.pivot(index="feature_set", columns="horizon", values="rel_rmsfe")

    # Order rows to match paper Table 1
    spec_order = [
        "F", "F-X", "X",
        "X-MARX", "F-MARX", "F-X-MARX",
        "MAF", "F-X-MAF", "X-MAF",
        "F-Level", "F-X-Level", "X-Level",
        "F-MARX-Level", "F-X-MARX-Level", "X-MARX-Level",
    ]
    pivot = pivot.reindex([s for s in spec_order if s in pivot.index])
    return pivot


def fig_rmsfe_heatmap(
    rmsfe: pd.DataFrame,
    title: str,
    out_path: Path,
    vmin: float = 0.85,
    vmax: float = 1.15,
    center: float = 1.0,
) -> None:
    """Heatmap: feature_sets × horizons, color = relative RMSFE."""
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(9, 6))

    data = rmsfe.values.astype(float)
    n_rows, n_cols = data.shape

    # Diverging colormap: green < 1 (beats benchmark), red > 1
    cmap = plt.cm.RdYlGn_r

    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            val = data[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if abs(val - center) > 0.10 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=7, color=text_color, fontweight="bold")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([str(h) for h in rmsfe.columns], fontsize=9)
    ax.set_xlabel("Horizon h", fontsize=10)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(rmsfe.index, fontsize=8)

    plt.colorbar(im, ax=ax, shrink=0.7, label="Relative RMSFE")
    ax.set_title(title, fontsize=11, pad=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    log.info("=== CLSS 2021 paper-faithful figures (v2) ===")

    df, ar = load_data()

    # ------------------------------------------------------------------
    # Step 1: Pseudo-OOS R² (Eq. 11)
    # ------------------------------------------------------------------
    log.info("Computing pseudo-OOS R² ...")
    r2_df = compute_pseudo_r2(df)
    # Add forecast_date to r2_df for time-series regression
    r2_df["forecast_date"] = df["forecast_date"]

    # ------------------------------------------------------------------
    # Step 2a: MARX marginal effects (Fig. 1 equivalent)
    # ------------------------------------------------------------------
    log.info("Computing MARX marginal effects (Fig. 1) ...")
    marx_effects = compute_marginal_effects(r2_df, MARX_PAIRS, "MARX")
    marx_effects.to_csv(RESULTS_DIR / "marx_marginal_effects.csv", index=False)
    log.info("  MARX effects: %d rows", len(marx_effects))

    fig_marginal_effects(
        marx_effects,
        feature_name="MARX",
        out_path=FIGURES_DIR / "fig_marx_effects_rf.png",
        x_lim=(-0.4, 0.4),
    )

    # ------------------------------------------------------------------
    # Step 2b: MAF marginal effects (Fig. 4 equivalent)
    # ------------------------------------------------------------------
    log.info("Computing MAF marginal effects (Fig. 4) ...")
    maf_effects = compute_marginal_effects(r2_df, MAF_PAIRS, "MAF")
    maf_effects.to_csv(RESULTS_DIR / "maf_marginal_effects.csv", index=False)
    log.info("  MAF effects: %d rows", len(maf_effects))

    fig_marginal_effects(
        maf_effects,
        feature_name="MAF",
        out_path=FIGURES_DIR / "fig_maf_effects_rf.png",
        x_lim=(-0.4, 0.4),
    )

    # ------------------------------------------------------------------
    # Step 3: Variable Importance stacked bars (Fig. 3 equivalent)
    # ------------------------------------------------------------------
    log.info("Building VI stacked bar figure (Fig. 3) ...")
    fig_vi_stacked(df, out_path=FIGURES_DIR / "fig_vi_stacked_fxmarx.png",
                   feature_spec="F-X-MARX")

    # Also produce F-X-MAF VI figure (for reference)
    # fig_vi_stacked(df, out_path=FIGURES_DIR / "fig_vi_stacked_fxmaf.png",
    #                feature_spec="F-X-MAF")

    # ------------------------------------------------------------------
    # Step 4: RMSFE heatmap vs AR (Table 3 equivalent)
    # ------------------------------------------------------------------
    log.info("Computing RMSFE vs AR for heatmap ...")
    rmsfe_ar = compute_rmsfe_vs_ar(df, ar)
    rmsfe_ar.to_csv(RESULTS_DIR / "rmsfe_vs_ar.csv")

    fig_rmsfe_heatmap(
        rmsfe_ar,
        title="Relative RMSFE vs AR benchmark — RF, averaged over targets",
        out_path=FIGURES_DIR / "fig_rmsfe_vs_ar.png",
        vmin=0.85, vmax=1.15,
    )

    # ------------------------------------------------------------------
    # Summary printout of marginal effects
    # ------------------------------------------------------------------
    log.info("\n=== MARX Marginal Effects (α, 95%% CI) ===")
    for h in HORIZONS:
        sub = marx_effects[marx_effects["horizon"] == h]
        if sub.empty:
            continue
        log.info("H=%d:", h)
        for _, r in sub.iterrows():
            sign = "+" if r["alpha"] > 0 else ""
            log.info(
                "  %-25s α=%s%.4f  [%.4f, %.4f]  %s",
                TARGET_LABELS.get(r["target"], r["target"]),
                sign, r["alpha"], r["ci_lo"], r["ci_hi"],
                "*" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else "",
            )

    log.info("\n=== MAF Marginal Effects (α, 95%% CI) ===")
    for h in HORIZONS:
        sub = maf_effects[maf_effects["horizon"] == h]
        if sub.empty:
            continue
        log.info("H=%d:", h)
        for _, r in sub.iterrows():
            sign = "+" if r["alpha"] > 0 else ""
            log.info(
                "  %-25s α=%s%.4f  [%.4f, %.4f]  %s",
                TARGET_LABELS.get(r["target"], r["target"]),
                sign, r["alpha"], r["ci_lo"], r["ci_hi"],
                "*" if r["ci_lo"] > 0 or r["ci_hi"] < 0 else "",
            )

    log.info("=== Done. Figures saved to %s ===", FIGURES_DIR)


if __name__ == "__main__":
    main()
