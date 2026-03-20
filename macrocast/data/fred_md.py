"""FRED-MD monthly macroeconomic data loader.

Provides ``load_fred_md``, the primary entry point for downloading and
preparing the McCracken & Ng (2016) FRED-MD dataset. Data are cached
locally under ``~/.macrocast/cache/fred_md/`` by default.

Reference:
    McCracken, M.W. and Ng, S. (2016). "FRED-MD: A Monthly Database for
    Macroeconomic Research." *Journal of Business & Economic Statistics*,
    34(4), 574-589.
"""

from __future__ import annotations

from pathlib import Path

from macrocast.data._base import (
    _build_vintage_url,
    _download_fred_csv,
    _parse_fred_csv,
)
from macrocast.data.schema import (
    MacroFrame,
    MacroFrameMetadata,
    VariableMetadata,
    _load_spec,
)
from macrocast.utils.cache import file_download_date, get_cached_path, is_cached

_CURRENT_URL = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-md/monthly/current.csv"
)
_VINTAGE_URL_TEMPLATE = (
    "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/"
    "fred-md/monthly/{vintage}.csv"
)
_DATASET = "fred_md"
_MAX_AGE_CURRENT = 30  # days before re-downloading the current release


def load_fred_md(
    vintage: str | None = None,
    start: str | None = None,
    end: str | None = None,
    transform: bool = False,
    tcode_override: dict[str, int] | None = None,
    cache_dir: str | Path | None = None,
    force_download: bool = False,
) -> MacroFrame:
    """Load the FRED-MD monthly macroeconomic dataset.

    Parameters
    ----------
    vintage : str or None
        Vintage identifier in ``"YYYY-MM"`` format (e.g. ``"2020-01"``).
        When None, the current (latest) release is fetched.
    start : str or None
        Sample start date, e.g. ``"1970-01"``. Applied after loading.
    end : str or None
        Sample end date (inclusive), e.g. ``"2023-12"``.
    transform : bool
        If True, apply the default McCracken-Ng stationarity
        transformations (tcode 1-7) immediately.
    tcode_override : dict[str, int], optional
        Override transformation codes for specific variables. Only used
        when ``transform=True``.
    cache_dir : str or Path, optional
        Override the default cache directory (``~/.macrocast/cache/``).
    force_download : bool
        Force a fresh download even if a valid cached file exists.

    Returns
    -------
    MacroFrame
        Loaded dataset. Transformations and trimming are applied if
        requested.

    Examples
    --------
    >>> md = load_fred_md(vintage="2024-01")
    >>> md_t = load_fred_md(transform=True, start="1970-01", end="2023-12")
    """
    # Determine URL and cache filename
    if vintage is None:
        url = _CURRENT_URL
        filename = "current.csv"
        max_age = _MAX_AGE_CURRENT
    else:
        url = _build_vintage_url(_DATASET, vintage)
        filename = f"{vintage}.csv"
        max_age = None  # vintage files never expire

    cache_path = get_cached_path(_DATASET, filename, cache_dir)

    # Download if needed
    if force_download or not is_cached(_DATASET, filename, cache_dir, max_age):
        _download_fred_csv(url, cache_path, force_download=True)

    # Parse CSV
    df, tcodes_from_file = _parse_fred_csv(cache_path)

    # Merge file tcodes with any spec tcodes (spec takes precedence for
    # described variables; file values fill in the rest)
    spec = _load_spec(_DATASET)
    spec_vars = spec.get("variables", {})
    spec_tcodes = {k: int(v["tcode"]) for k, v in spec_vars.items() if "tcode" in v}

    effective_tcodes = {**tcodes_from_file, **spec_tcodes}
    if tcode_override:
        effective_tcodes.update(tcode_override)

    # Build metadata
    variable_meta: dict[str, VariableMetadata] = {}
    for col in df.columns:
        spec_entry = spec_vars.get(col, {})
        variable_meta[col] = VariableMetadata(
            name=col,
            description=spec_entry.get("description", col),
            group=spec_entry.get("group", "other"),
            tcode=effective_tcodes.get(col, 1),
            frequency="monthly",
        )

    data_through = df.index[-1].strftime("%Y-%m") if len(df) > 0 else None
    metadata = MacroFrameMetadata(
        dataset="FRED-MD",
        vintage=vintage,
        frequency="monthly",
        variables=variable_meta,
        groups=spec.get("groups", {}),
        is_transformed=False,
        download_date=file_download_date(cache_path) if vintage is None else None,
        data_through=data_through,
    )

    mf = MacroFrame(df, metadata, effective_tcodes)

    # Apply optional trimming and transformation
    if start is not None or end is not None:
        mf = mf.trim(start=start, end=end)
    if transform:
        mf = mf.transform(override=tcode_override)

    return mf
