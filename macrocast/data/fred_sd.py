"""FRED-SD state-level macroeconomic data loader.

FRED-SD is an Excel workbook where each sheet (tab) corresponds to a
macroeconomic variable and columns correspond to U.S. states / DC. This
loader parses the workbook into a wide-format DataFrame with column names
``{variable}_{state}`` and a DatetimeIndex.

Vintage files follow the same ``YYYY-MM.xlsx`` naming convention as
FRED-MD/QD. Vintages are available from 2005-01 onward.

Reference:
    McCracken, M.W. and Owyang, M.T. (2021). "The St. Louis Fed's
    Financial Stress Index, Version 2." Federal Reserve Bank of St. Louis
    Working Paper 2021-016.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from macrocast.data.schema import (
    MacroFrame,
    MacroFrameMetadata,
    VariableMetadata,
    _load_spec,
)
from macrocast.utils.cache import (
    download_file,
    file_download_date,
    get_cached_path,
    is_cached,
)

_CURRENT_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-sd/FRED_SD.xlsx"
)
_VINTAGE_URL_TEMPLATE = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-sd/{vintage}.xlsx"
)
_DATASET = "fred_sd"
_MAX_AGE = 30  # days before re-downloading the current release


def load_fred_sd(
    vintage: str | None = None,
    states: list[str] | None = None,
    variables: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> MacroFrame:
    """Load the FRED-SD state-level dataset.

    Parameters
    ----------
    vintage : str or None
        Vintage identifier in ``"YYYY-MM"`` format (e.g. ``"2020-01"``).
        When None, the current (latest) release is fetched.
    states : list of str, optional
        Two-letter state codes to include (e.g. ``["CA", "TX"]``).
        When None, all 51 state/DC columns are retained.
    variables : list of str, optional
        Variable names (sheet names) to include (e.g. ``["UR", "EMPL"]``).
        When None, all sheets are loaded.
    start : str or None
        Sample start date.
    end : str or None
        Sample end date (inclusive).
    cache_dir : str or Path, optional
        Override default cache directory.
    force_download : bool
        Force a fresh download even if a cached copy is current.

    Returns
    -------
    MacroFrame
        Wide-format panel with columns named ``{variable}_{state}`` and
        a DatetimeIndex.

    Examples
    --------
    >>> sd = load_fred_sd(states=["CA", "TX"])
    >>> sd_2020 = load_fred_sd(vintage="2020-01", variables=["UR"])
    """
    if vintage is None:
        url = _CURRENT_URL
        filename = "FRED_SD.xlsx"
        max_age: float | None = _MAX_AGE
    else:
        url = _VINTAGE_URL_TEMPLATE.format(vintage=vintage)
        filename = f"{vintage}.xlsx"
        max_age = None  # vintage files never expire

    cache_path = get_cached_path(_DATASET, filename, cache_dir)

    if force_download or not is_cached(_DATASET, filename, cache_dir, max_age):
        download_file(url, cache_path)

    spec = _load_spec(_DATASET)
    spec_vars = spec.get("variables", {})
    all_states: list[str] = spec.get("states", [])

    # Parse all sheets from the workbook
    sheets: dict[str, pd.DataFrame] = pd.read_excel(
        cache_path, sheet_name=None, index_col=0, engine="openpyxl"
    )

    # Filter sheets if variables requested
    if variables is not None:
        sheets = {k: v for k, v in sheets.items() if k in variables}

    if not sheets:
        raise ValueError(
            "No matching sheets found in FRED_SD workbook. "
            f"Requested variables: {variables}"
        )

    # Filter states
    target_states = states if states is not None else all_states

    wide_frames: list[pd.DataFrame] = []
    variable_meta: dict[str, VariableMetadata] = {}
    effective_tcodes: dict[str, int] = {}

    for var_name, sheet_df in sheets.items():
        spec_entry = spec_vars.get(var_name, {})
        tcode = int(spec_entry.get("tcode", 1))
        freq = spec_entry.get("frequency", "monthly")

        # Ensure DatetimeIndex
        if not isinstance(sheet_df.index, pd.DatetimeIndex):
            sheet_df.index = pd.to_datetime(sheet_df.index, errors="coerce")
            sheet_df = sheet_df[sheet_df.index.notna()]

        # Filter to requested states
        available_states = [s for s in target_states if s in sheet_df.columns]
        sub = sheet_df[available_states].copy()
        sub = sub.apply(pd.to_numeric, errors="coerce")

        # Rename columns to {variable}_{state}
        sub.columns = [f"{var_name}_{s}" for s in available_states]

        for col in sub.columns:
            state = col.split("_")[-1]
            variable_meta[col] = VariableMetadata(
                name=col,
                description=f"{spec_entry.get('description', var_name)} ({state})",
                group=spec_entry.get("group", "other"),
                tcode=tcode,
                frequency=freq,
            )
            effective_tcodes[col] = tcode

        wide_frames.append(sub)

    # Concatenate all variables along columns, aligning on date index
    df = pd.concat(wide_frames, axis=1)
    df.index.name = "date"
    df.sort_index(inplace=True)

    # Apply date trimming
    if start is not None:
        df = df.loc[start:]
    if end is not None:
        df = df.loc[:end]

    data_through = df.index[-1].strftime("%Y-%m") if len(df) > 0 else None
    metadata = MacroFrameMetadata(
        dataset="FRED-SD",
        vintage=vintage,
        frequency="state_monthly",
        variables=variable_meta,
        groups=spec.get("groups", {}),
        is_transformed=False,
        download_date=file_download_date(cache_path) if vintage is None else None,
        data_through=data_through,
    )

    return MacroFrame(df, metadata, effective_tcodes)
