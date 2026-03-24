"""Publication-ready LaTeX table export for forecast evaluation results.

Produces booktabs-style LaTeX tables suitable for journal submission.
Requires the ``booktabs`` and ``multirow`` packages in the LaTeX preamble.

Typical usage::

    from macrocast.evaluation import rmsfe_to_latex, regime_to_latex

    # RMSFE table (models × horizons)
    print(rmsfe_to_latex(rmsfe_df, caption="Relative MSFE — INDPRO",
                         label="tab:rmsfe_indpro"))

    # Regime-conditional table
    print(regime_to_latex(regime_df, caption="Regime-Conditional RMSFE",
                          label="tab:regime"))
"""

from __future__ import annotations

import re

import pandas as pd


def rmsfe_to_latex(
    rmsfe_table: pd.DataFrame,
    caption: str = "",
    label: str = "tab:rmsfe",
    bold_min: bool = True,
    mcs_table: pd.DataFrame | None = None,
    benchmark_row: str | None = None,
    decimals: int = 3,
    horizon_header: str = "Horizon $h$",
) -> str:
    """Format a relative MSFE table as a LaTeX table string.

    Parameters
    ----------
    rmsfe_table : pd.DataFrame
        Rows are models (index = model name strings).
        Columns are forecast horizons (int or str, e.g. 1, 3, 6, 12).
        Values are relative MSFE (benchmark = 1.000).
    caption : str
        LaTeX table caption.
    label : str
        LaTeX \\label identifier.
    bold_min : bool
        If True, bold the minimum value in each column (excluding the
        benchmark row if it equals 1.000 everywhere).
    mcs_table : pd.DataFrame or None
        Boolean DataFrame with the same index and columns as *rmsfe_table*.
        True entries receive a ``$^*$`` superscript (MCS membership).
    benchmark_row : str or None
        Index label of the benchmark model.  The benchmark row is rendered
        with a leading \\midrule separator.
    decimals : int
        Number of decimal places.  Default 3.
    horizon_header : str
        Column group header label for horizon columns.

    Returns
    -------
    str
        Complete LaTeX ``table`` environment string.

    Notes
    -----
    Requires ``booktabs`` in the LaTeX preamble::

        \\usepackage{booktabs}
    """
    df = rmsfe_table.copy()
    n_rows, n_cols = df.shape
    cols = list(df.columns)
    index = list(df.index)

    # MCS membership matrix (bool), same shape as df
    if mcs_table is not None:
        mcs = mcs_table.reindex(index=index, columns=cols).fillna(False)
    else:
        mcs = pd.DataFrame(False, index=index, columns=cols)

    # Column alignment: left for row labels, right-aligned for numeric cols
    col_spec = "l" + "r" * n_cols
    col_header = " & ".join(str(c) for c in cols)

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    if caption:
        lines.append(f"\\caption{{{_escape_latex(caption)}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Column group header spanning all horizon columns
    lines.append(
        f" & \\multicolumn{{{n_cols}}}{{c}}{{{horizon_header}}} \\\\"
    )
    lines.append(f"\\cmidrule(lr){{2-{n_cols + 1}}}")
    lines.append(f" & {col_header} \\\\")
    lines.append(r"\midrule")

    # Identify minimum per column (for bold_min)
    col_min: dict = {}
    if bold_min:
        for c in cols:
            # Exclude benchmark row from min search if it's exactly 1.0 everywhere
            candidate = df[c].copy()
            if benchmark_row is not None and benchmark_row in index:
                candidate = candidate.drop(benchmark_row, errors="ignore")
            col_min[c] = candidate.min()

    benchmark_printed = False
    for row_label in index:
        # Insert midrule before benchmark row (or after non-benchmark block)
        if benchmark_row is not None and row_label == benchmark_row and not benchmark_printed:
            if lines[-1] != r"\midrule":
                lines.append(r"\midrule")
            benchmark_printed = True

        cells: list[str] = [_escape_latex(str(row_label))]
        for c in cols:
            val = df.loc[row_label, c]
            if pd.isna(val):
                cell_str = "---"
            else:
                cell_str = f"{val:.{decimals}f}"
                # Superscript for MCS membership
                if mcs.loc[row_label, c]:
                    cell_str += r"$^{*}$"
                # Bold for column minimum (skip benchmark row)
                if (
                    bold_min
                    and c in col_min
                    and not pd.isna(col_min[c])
                    and not pd.isna(val)
                    and abs(val - col_min[c]) < 1e-10
                    and row_label != benchmark_row
                ):
                    cell_str = f"\\textbf{{{cell_str}}}"
            cells.append(cell_str)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def regime_to_latex(
    regime_table: pd.DataFrame,
    caption: str = "",
    label: str = "tab:regime",
    bold_min: bool = True,
    benchmark_row: str | None = None,
    decimals: int = 3,
) -> str:
    """Format a regime-conditional RMSFE table as a LaTeX table string.

    Parameters
    ----------
    regime_table : pd.DataFrame
        Rows are models (index = model name strings).
        Columns are a :class:`~pandas.MultiIndex` of ``(horizon, regime)``
        or a flat index of ``(horizon_regime)`` strings.
        Values are relative MSFE within each regime.
    caption : str
        LaTeX table caption.
    label : str
        LaTeX \\label identifier.
    bold_min : bool
        Bold the minimum value within each (horizon, regime) column group.
    benchmark_row : str or None
        Index label of the benchmark model.  Rendered with a \\midrule.
    decimals : int
        Number of decimal places.  Default 3.

    Returns
    -------
    str
        Complete LaTeX ``table`` environment string.
    """
    df = regime_table.copy()
    index = list(df.index)

    # Detect MultiIndex columns vs flat columns
    has_multiindex = isinstance(df.columns, pd.MultiIndex)
    if has_multiindex:
        horizons = df.columns.get_level_values(0).unique().tolist()
        all_cols = list(df.columns)
    else:
        # Flat columns: treat each as a separate column
        all_cols = list(df.columns)
        horizons = all_cols

    n_total_cols = len(all_cols)
    col_spec = "l" + "r" * n_total_cols

    lines: list[str] = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    if caption:
        lines.append(f"\\caption{{{_escape_latex(caption)}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header rows
    if has_multiindex:
        # Row 1: horizon groups
        header_parts = [" "]
        col_idx = 2  # 1-based, first data col is 2
        for h in horizons:
            group_cols = [c for c in all_cols if c[0] == h]
            header_parts.append(
                f"\\multicolumn{{{len(group_cols)}}}{{c}}{{$h={h}$}}"
            )
            col_idx += len(group_cols)
        lines.append(" & ".join(header_parts) + r" \\")
        # cmidrule for each horizon block
        col_idx = 2
        cmidrules = []
        for h in horizons:
            group_cols = [c for c in all_cols if c[0] == h]
            cmidrules.append(
                f"\\cmidrule(lr){{{col_idx}-{col_idx + len(group_cols) - 1}}}"
            )
            col_idx += len(group_cols)
        lines.append(" ".join(cmidrules))
        # Row 2: regime names
        regime_headers = [" "] + [_escape_latex(str(r)) for _, r in all_cols]
        lines.append(" & ".join(regime_headers) + r" \\")
    else:
        col_header = " & ".join(_escape_latex(str(c)) for c in all_cols)
        lines.append(f" & {col_header} \\\\")

    lines.append(r"\midrule")

    # Identify minimum per column for bold_min
    col_min: dict = {}
    if bold_min:
        for c in all_cols:
            candidate = df[c].copy()
            if benchmark_row is not None and benchmark_row in index:
                candidate = candidate.drop(benchmark_row, errors="ignore")
            col_min[c] = candidate.min()

    benchmark_printed = False
    for row_label in index:
        if benchmark_row is not None and row_label == benchmark_row and not benchmark_printed:
            if lines[-1] != r"\midrule":
                lines.append(r"\midrule")
            benchmark_printed = True

        cells = [_escape_latex(str(row_label))]
        for c in all_cols:
            val = df.loc[row_label, c]
            if pd.isna(val):
                cell_str = "---"
            else:
                cell_str = f"{val:.{decimals}f}"
                if (
                    bold_min
                    and c in col_min
                    and not pd.isna(col_min[c])
                    and not pd.isna(val)
                    and abs(val - col_min[c]) < 1e-10
                    and row_label != benchmark_row
                ):
                    cell_str = f"\\textbf{{{cell_str}}}"
            cells.append(cell_str)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in a plain text string."""
    return re.sub(
        r"[&%$#_{}~^\\]",
        lambda m: _LATEX_SPECIAL.get(m.group(), m.group()),
        text,
    )
