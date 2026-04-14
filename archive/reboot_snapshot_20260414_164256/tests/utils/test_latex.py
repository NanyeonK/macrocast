"""Tests for macrocast/evaluation/latex.py — LaTeX table export."""

from __future__ import annotations

import pandas as pd
import pytest

from macrocast.utils.latex import _escape_latex, regime_to_latex, rmsfe_to_latex


def _make_rmsfe_table() -> pd.DataFrame:
    """Simple 4-model × 4-horizon relative MSFE table."""
    data = {
        1: [1.000, 0.952, 0.971, 0.988],
        3: [1.000, 0.963, 0.945, 0.979],
        6: [1.000, 0.981, 0.958, 0.966],
        12: [1.000, 0.974, 0.950, 0.961],
    }
    return pd.DataFrame(data, index=["AR", "LASSO", "RF", "EN"])


def _make_mcs_table() -> pd.DataFrame:
    data = {
        1: [False, True, False, False],
        3: [False, True, True, False],
        6: [False, False, True, True],
        12: [False, True, True, False],
    }
    return pd.DataFrame(data, index=["AR", "LASSO", "RF", "EN"])


class TestRmsfeToLatex:
    def test_returns_string(self) -> None:
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df)
        assert isinstance(result, str)

    def test_contains_table_environment(self) -> None:
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df)
        assert r"\begin{table}" in result
        assert r"\end{table}" in result

    def test_contains_tabular_environment(self) -> None:
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df)
        assert r"\begin{tabular}" in result
        assert r"\end{tabular}" in result

    def test_booktabs_rules_present(self) -> None:
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df)
        assert r"\toprule" in result
        assert r"\midrule" in result
        assert r"\bottomrule" in result

    def test_caption_included(self) -> None:
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df, caption="Relative MSFE for INDPRO")
        assert "Relative MSFE for INDPRO" in result
        assert r"\caption" in result

    def test_label_included(self) -> None:
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df, label="tab:indpro_rmsfe")
        assert r"\label{tab:indpro_rmsfe}" in result

    def test_all_model_names_present(self) -> None:
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df)
        for name in ["AR", "LASSO", "RF", "EN"]:
            assert name in result

    def test_all_horizon_values_present(self) -> None:
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df)
        for h in [1, 3, 6, 12]:
            assert str(h) in result

    def test_bold_min_applied(self) -> None:
        """Minimum per column should be wrapped in \\textbf{}."""
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df, bold_min=True, benchmark_row="AR")
        assert r"\textbf{" in result

    def test_bold_min_not_applied_to_benchmark(self) -> None:
        """Benchmark row values should not be bolded."""
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df, bold_min=True, benchmark_row="AR")
        # AR row has 1.000 everywhere but should NOT be bolded
        lines = result.split("\n")
        ar_lines = [l for l in lines if l.startswith("AR")]
        for line in ar_lines:
            assert r"\textbf" not in line

    def test_bold_min_false_no_textbf(self) -> None:
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df, bold_min=False)
        assert r"\textbf{" not in result

    def test_mcs_asterisk_present(self) -> None:
        df = _make_rmsfe_table()
        mcs = _make_mcs_table()
        result = rmsfe_to_latex(df, mcs_table=mcs)
        assert r"$^{*}$" in result

    def test_mcs_asterisk_not_in_non_member(self) -> None:
        """Only MCS members get asterisks; check a known non-member cell."""
        df = _make_rmsfe_table()
        mcs = _make_mcs_table()
        result = rmsfe_to_latex(df, mcs_table=mcs)
        # AR row: all False in mcs → no asterisk on AR line
        lines = result.split("\n")
        ar_lines = [l for l in lines if l.lstrip().startswith("AR")]
        for line in ar_lines:
            assert r"$^{*}$" not in line

    def test_benchmark_midrule_inserted(self) -> None:
        """A \\midrule should appear before the benchmark row."""
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df, benchmark_row="AR")
        assert r"\midrule" in result

    def test_no_caption_when_empty(self) -> None:
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df, caption="")
        assert r"\caption" not in result

    def test_nan_value_rendered_as_dashes(self) -> None:
        df = _make_rmsfe_table().copy()
        df.loc["LASSO", 1] = float("nan")
        result = rmsfe_to_latex(df)
        assert "---" in result

    def test_decimals_respected(self) -> None:
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df, decimals=2)
        assert "0.95" in result
        assert "0.952" not in result

    def test_column_count_in_tabular_spec(self) -> None:
        """tabular spec should have 1 left-align + n_cols right-align columns."""
        df = _make_rmsfe_table()
        result = rmsfe_to_latex(df)
        # 4 horizons → {lrrrr}
        assert r"\begin{tabular}{lrrrr}" in result


class TestRegimeToLatex:
    def _make_regime_table_flat(self) -> pd.DataFrame:
        """Flat-column regime table."""
        data = {
            "exp": [1.000, 0.960, 0.975],
            "rec": [1.000, 0.910, 0.930],
        }
        return pd.DataFrame(data, index=["AR", "LASSO", "RF"])

    def _make_regime_table_multiindex(self) -> pd.DataFrame:
        """MultiIndex-column regime table: (horizon, regime)."""
        cols = pd.MultiIndex.from_tuples(
            [(1, "exp"), (1, "rec"), (3, "exp"), (3, "rec")]
        )
        data = [
            [1.000, 1.000, 1.000, 1.000],
            [0.960, 0.910, 0.955, 0.920],
            [0.975, 0.930, 0.970, 0.940],
        ]
        return pd.DataFrame(data, columns=cols, index=["AR", "LASSO", "RF"])

    def test_returns_string_flat(self) -> None:
        df = self._make_regime_table_flat()
        result = regime_to_latex(df)
        assert isinstance(result, str)

    def test_returns_string_multiindex(self) -> None:
        df = self._make_regime_table_multiindex()
        result = regime_to_latex(df)
        assert isinstance(result, str)

    def test_table_environment_present(self) -> None:
        df = self._make_regime_table_flat()
        result = regime_to_latex(df)
        assert r"\begin{table}" in result
        assert r"\end{table}" in result

    def test_multiindex_horizon_header(self) -> None:
        """MultiIndex table should include \\multicolumn for horizon groups."""
        df = self._make_regime_table_multiindex()
        result = regime_to_latex(df)
        assert r"\multicolumn" in result
        assert "$h=1$" in result
        assert "$h=3$" in result

    def test_regime_labels_present_multiindex(self) -> None:
        df = self._make_regime_table_multiindex()
        result = regime_to_latex(df)
        assert "exp" in result
        assert "rec" in result

    def test_bold_min_multiindex(self) -> None:
        df = self._make_regime_table_multiindex()
        result = regime_to_latex(df, bold_min=True, benchmark_row="AR")
        assert r"\textbf{" in result

    def test_caption_and_label(self) -> None:
        df = self._make_regime_table_flat()
        result = regime_to_latex(df, caption="Regime RMSFE", label="tab:regime_test")
        assert "Regime RMSFE" in result
        assert r"\label{tab:regime_test}" in result


class TestEscapeLatex:
    def test_ampersand_escaped(self) -> None:
        assert _escape_latex("A & B") == r"A \& B"

    def test_percent_escaped(self) -> None:
        assert _escape_latex("50%") == r"50\%"

    def test_underscore_escaped(self) -> None:
        assert _escape_latex("model_id") == r"model\_id"

    def test_dollar_escaped(self) -> None:
        assert _escape_latex("$100") == r"\$100"

    def test_plain_text_unchanged(self) -> None:
        assert _escape_latex("LASSO") == "LASSO"

    def test_empty_string(self) -> None:
        assert _escape_latex("") == ""

    def test_model_id_with_underscores(self) -> None:
        result = _escape_latex("linear__none__bic__l2")
        assert "\\_\\_" in result


class TestLatexExport:
    def test_importable_from_evaluation(self) -> None:
        from macrocast.utils.latex import regime_to_latex, rmsfe_to_latex  # noqa: F401

    def test_importable_from_package(self) -> None:
        from macrocast.utils import regime_to_latex, rmsfe_to_latex  # noqa: F401
