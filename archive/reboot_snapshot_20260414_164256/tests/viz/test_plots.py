"""Tests for evaluation/plots.py — Fig 1/2/3/6 visualization functions.

Uses the non-interactive Agg backend so tests run in headless CI environments.
All figures are explicitly closed after each test to avoid memory leaks.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must be set before importing pyplot

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from macrocast.viz.plots import (
    cumulative_squared_error_plot,
    marginal_effect_plot,
    plot_horizon_lines,
    plot_mcs_membership,
    plot_rmsfe_heatmap,
    variable_importance_plot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mc_df(with_target: bool = False) -> pd.DataFrame:
    """Minimal marginal contribution DataFrame (2 models x 2 horizons)."""
    rows = [
        {"feature": "MARX", "model": "RF", "horizon": 1,
         "alpha": 0.05, "ci_low": 0.01, "ci_high": 0.09},
        {"feature": "MARX", "model": "RF", "horizon": 3,
         "alpha": 0.08, "ci_low": 0.03, "ci_high": 0.13},
        {"feature": "MARX", "model": "EN", "horizon": 1,
         "alpha": 0.02, "ci_low": -0.01, "ci_high": 0.05},
        {"feature": "MARX", "model": "EN", "horizon": 3,
         "alpha": 0.03, "ci_low": 0.00, "ci_high": 0.06},
    ]
    df = pd.DataFrame(rows)
    if with_target:
        df["target"] = ["INDPRO", "INDPRO", "PAYEMS", "PAYEMS"]
    return df


def _make_vi_avg_df(with_target: bool = True) -> pd.DataFrame:
    """Minimal average VI DataFrame (2 horizons x 3 groups)."""
    groups = ["AR", "MARX", "X"]
    rows = []
    for h in [1, 3]:
        shares = [0.4, 0.35, 0.25]
        for g, s in zip(groups, shares):
            row = {
                "model_id": "RF",
                "feature_set": "F-MARX",
                "horizon": h,
                "group": g,
                "importance_share": s,
            }
            if with_target:
                row["target"] = "INDPRO"
            rows.append(row)
    return pd.DataFrame(rows)


def _make_result_df() -> pd.DataFrame:
    """Synthetic OOS forecast result table for cumulative SE test."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-01", periods=24, freq="QS")
    n = len(dates)
    rows = []
    for model_id, feature_set in [("AR", "AR"), ("RF", "F-MARX")]:
        y_true = rng.standard_normal(n)
        y_hat = y_true + rng.standard_normal(n) * 0.5
        for i, d in enumerate(dates):
            rows.append({
                "model_id": model_id,
                "feature_set": feature_set,
                "horizon": 1,
                "date": d,
                "y_hat": y_hat[i],
                "y_true": y_true[i],
                "target": "INDPRO",
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: marginal_effect_plot
# ---------------------------------------------------------------------------


class TestMarginalEffectPlot:
    def test_returns_figure_no_target(self) -> None:
        """Basic call without target column must return a Figure."""
        mc_df = _make_mc_df(with_target=False)
        fig = marginal_effect_plot(mc_df, feature="MARX")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_with_target(self) -> None:
        """Call with target column must return a Figure with colored dots."""
        mc_df = _make_mc_df(with_target=True)
        fig = marginal_effect_plot(mc_df, feature="MARX")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_explicit_models_horizons(self) -> None:
        """Explicit models/horizons subsets should produce correct grid shape."""
        mc_df = _make_mc_df(with_target=False)
        fig = marginal_effect_plot(
            mc_df,
            feature="MARX",
            models=["RF"],
            horizons=[1],
        )
        assert isinstance(fig, plt.Figure)
        # 1x1 grid -> one Axes
        axes = fig.get_axes()
        assert len(axes) == 1
        plt.close(fig)

    def test_wrong_feature_raises(self) -> None:
        """Filtering to a non-existent feature should raise ValueError."""
        mc_df = _make_mc_df()
        with pytest.raises(ValueError, match="No rows for feature"):
            marginal_effect_plot(mc_df, feature="NONEXISTENT")

    def test_custom_palette_and_title(self) -> None:
        """Custom palette and title must be accepted without error."""
        mc_df = _make_mc_df(with_target=True)
        fig = marginal_effect_plot(
            mc_df,
            feature="MARX",
            target_palette={"INDPRO": "#000000"},
            title="Test Title",
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_zero_line_disabled(self) -> None:
        """zero_line=False should still return a valid Figure."""
        mc_df = _make_mc_df()
        fig = marginal_effect_plot(mc_df, feature="MARX", zero_line=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_grid_dimensions_match_horizons_models(self) -> None:
        """Grid should have n_horizons rows * n_models cols of Axes."""
        mc_df = _make_mc_df(with_target=False)
        models = ["RF", "EN"]
        horizons = [1, 3]
        fig = marginal_effect_plot(
            mc_df, feature="MARX", models=models, horizons=horizons
        )
        # 2x2 grid
        axes = fig.get_axes()
        # legend handle Axes may be present; at minimum 4 data Axes
        assert len(axes) >= 4
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: variable_importance_plot
# ---------------------------------------------------------------------------


class TestVariableImportancePlot:
    def test_returns_figure_single_target(self) -> None:
        """Single-target call should return a Figure."""
        vi_avg = _make_vi_avg_df(with_target=True)
        fig = variable_importance_plot(vi_avg, targets=["INDPRO"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_returns_figure_no_target_column(self) -> None:
        """When target column is absent, targets must have one element."""
        vi_avg = _make_vi_avg_df(with_target=False)
        fig = variable_importance_plot(vi_avg, targets=["INDPRO"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_target_col_multi_targets_raises(self) -> None:
        """No target column + multiple targets should raise ValueError."""
        vi_avg = _make_vi_avg_df(with_target=False)
        with pytest.raises(ValueError, match="no 'target' column"):
            variable_importance_plot(vi_avg, targets=["INDPRO", "PAYEMS"])

    def test_with_direct_vi_avg_df(self) -> None:
        """Providing direct_vi_avg_df should add an AGR bar without error."""
        vi_avg = _make_vi_avg_df(with_target=True)
        direct_vi = _make_vi_avg_df(with_target=True)
        fig = variable_importance_plot(
            vi_avg,
            targets=["INDPRO"],
            direct_vi_avg_df=direct_vi,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_group_palette(self) -> None:
        """Custom group palette must be accepted."""
        vi_avg = _make_vi_avg_df(with_target=True)
        fig = variable_importance_plot(
            vi_avg,
            targets=["INDPRO"],
            group_palette={"AR": "#ff0000"},
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_multiple_target_panels(self) -> None:
        """Multiple targets should produce one panel per target."""
        vi_avg = _make_vi_avg_df(with_target=True)
        # Add a second target
        extra = vi_avg.copy()
        extra["target"] = "PAYEMS"
        combined = pd.concat([vi_avg, extra], ignore_index=True)
        fig = variable_importance_plot(combined, targets=["INDPRO", "PAYEMS"])
        assert isinstance(fig, plt.Figure)
        # Two panels -> at least 2 Axes
        assert len(fig.get_axes()) >= 2
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: cumulative_squared_error_plot
# ---------------------------------------------------------------------------


class TestCumulativeSquaredErrorPlot:
    def test_returns_figure(self) -> None:
        """Basic call should return a Figure."""
        result_df = _make_result_df()
        combos = [("AR", "AR"), ("RF", "F-MARX")]
        fig = cumulative_squared_error_plot(
            result_df,
            model_feature_combos=combos,
            target="INDPRO",
            horizon=1,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_labels_and_colors(self) -> None:
        """Custom labels and colors should be accepted."""
        result_df = _make_result_df()
        combos = [("AR", "AR")]
        fig = cumulative_squared_error_plot(
            result_df,
            model_feature_combos=combos,
            target="INDPRO",
            horizon=1,
            combo_labels=["Benchmark"],
            combo_colors=["#ff0000"],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_recession_shading(self) -> None:
        """Recession shading bands should be drawn without error."""
        result_df = _make_result_df()
        combos = [("AR", "AR")]
        fig = cumulative_squared_error_plot(
            result_df,
            model_feature_combos=combos,
            target="INDPRO",
            horizon=1,
            recession_shading=[("2011-01-01", "2011-06-30")],
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_wrong_horizon_raises(self) -> None:
        """Non-existent horizon should raise ValueError."""
        result_df = _make_result_df()
        with pytest.raises(ValueError, match="No rows for horizon"):
            cumulative_squared_error_plot(
                result_df,
                model_feature_combos=[("AR", "AR")],
                target="INDPRO",
                horizon=99,
            )

    def test_forecast_date_column_accepted(self) -> None:
        """result_df with 'forecast_date' instead of 'date' should work."""
        result_df = _make_result_df().rename(columns={"date": "forecast_date"})
        combos = [("AR", "AR")]
        fig = cumulative_squared_error_plot(
            result_df,
            model_feature_combos=combos,
            target="INDPRO",
            horizon=1,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_missing_combo_skipped_gracefully(self) -> None:
        """Combos not present in result_df should be silently skipped."""
        result_df = _make_result_df()
        combos = [("AR", "AR"), ("NONEXISTENT", "NONEXISTENT")]
        fig = cumulative_squared_error_plot(
            result_df,
            model_feature_combos=combos,
            target="INDPRO",
            horizon=1,
        )
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers for horse race plot functions
# ---------------------------------------------------------------------------


def _make_rmsfe_table(n_models: int = 4, horizons=None) -> "pd.DataFrame":
    """Synthetic relative MSFE table (rows=models, cols=horizons)."""
    if horizons is None:
        horizons = [1, 3, 6, 12]
    rng = np.random.default_rng(42)
    data = rng.uniform(0.7, 1.3, size=(n_models, len(horizons)))
    index = [f"model_{i}" for i in range(n_models)]
    return pd.DataFrame(data, index=index, columns=horizons)


def _make_mcs_table(n_models: int = 4, horizons=None) -> "pd.DataFrame":
    """Synthetic MCS membership table (bool values)."""
    if horizons is None:
        horizons = [1, 3, 6, 12]
    rng = np.random.default_rng(42)
    data = rng.choice([True, False], size=(n_models, len(horizons)))
    index = [f"model_{i}" for i in range(n_models)]
    return pd.DataFrame(data, index=index, columns=horizons)


# ---------------------------------------------------------------------------
# Tests: plot_rmsfe_heatmap
# ---------------------------------------------------------------------------


class TestPlotRmsfeHeatmap:
    def test_returns_figure(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_rmsfe_heatmap(tbl)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_axes_exist(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_rmsfe_heatmap(tbl)
        assert len(fig.get_axes()) >= 1
        plt.close(fig)

    def test_custom_title(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_rmsfe_heatmap(tbl, title="My heatmap")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_rmsfe_heatmap(tbl, figsize=(6, 3))
        assert fig.get_size_inches()[0] == pytest.approx(6.0)
        plt.close(fig)

    def test_no_annotations(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_rmsfe_heatmap(tbl, annot=False)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_vmin_vmax(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_rmsfe_heatmap(tbl, vmin=0.5, vmax=1.5)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_table_raises(self) -> None:
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            plot_rmsfe_heatmap(empty)

    def test_single_horizon(self) -> None:
        tbl = _make_rmsfe_table(horizons=[1])
        fig = plot_rmsfe_heatmap(tbl)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_model(self) -> None:
        tbl = _make_rmsfe_table(n_models=1)
        fig = plot_rmsfe_heatmap(tbl)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: plot_horizon_lines
# ---------------------------------------------------------------------------


class TestPlotHorizonLines:
    def test_returns_figure(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_horizon_lines(tbl)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_axes_exist(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_horizon_lines(tbl)
        assert len(fig.get_axes()) >= 1
        plt.close(fig)

    def test_custom_title(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_horizon_lines(tbl, title="Horizon comparison")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_highlight_ids(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_horizon_lines(tbl, highlight_ids=["model_0", "model_1"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_no_benchmark_line(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_horizon_lines(tbl, benchmark_line=None)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_benchmark_line(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_horizon_lines(tbl, benchmark_line=0.9)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_palette(self) -> None:
        tbl = _make_rmsfe_table(n_models=2)
        palette = {"model_0": "#ff0000", "model_1": "#0000ff"}
        fig = plot_horizon_lines(tbl, palette=palette)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_ylabel(self) -> None:
        tbl = _make_rmsfe_table()
        fig = plot_horizon_lines(tbl, ylabel="Relative RMSFE")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_empty_table_raises(self) -> None:
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            plot_horizon_lines(empty)

    def test_single_model(self) -> None:
        tbl = _make_rmsfe_table(n_models=1)
        fig = plot_horizon_lines(tbl)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Tests: plot_mcs_membership
# ---------------------------------------------------------------------------


class TestPlotMcsMembership:
    def test_returns_figure(self) -> None:
        tbl = _make_mcs_table()
        fig = plot_mcs_membership(tbl)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_axes_exist(self) -> None:
        tbl = _make_mcs_table()
        fig = plot_mcs_membership(tbl)
        assert len(fig.get_axes()) >= 1
        plt.close(fig)

    def test_custom_title(self) -> None:
        tbl = _make_mcs_table()
        fig = plot_mcs_membership(tbl, title="MCS at alpha=0.10")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_colors(self) -> None:
        tbl = _make_mcs_table()
        fig = plot_mcs_membership(tbl, color_in="#1f77b4", color_out="#aec7e8")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self) -> None:
        tbl = _make_mcs_table()
        fig = plot_mcs_membership(tbl, figsize=(7, 4))
        assert fig.get_size_inches()[0] == pytest.approx(7.0)
        plt.close(fig)

    def test_empty_table_raises(self) -> None:
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            plot_mcs_membership(empty)

    def test_all_in_mcs(self) -> None:
        tbl = pd.DataFrame(True, index=["m1", "m2"], columns=[1, 3, 6])
        fig = plot_mcs_membership(tbl)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_out_mcs(self) -> None:
        tbl = pd.DataFrame(False, index=["m1", "m2"], columns=[1, 3, 6])
        fig = plot_mcs_membership(tbl)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_integer_01_values_accepted(self) -> None:
        tbl = pd.DataFrame([[1, 0, 1], [0, 1, 0]], index=["m1", "m2"], columns=[1, 3, 6])
        fig = plot_mcs_membership(tbl)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
