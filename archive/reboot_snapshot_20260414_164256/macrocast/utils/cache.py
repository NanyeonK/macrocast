"""Cache management for macrocast data downloads.

Handles local caching of FRED-MD, FRED-QD, and FRED-SD files under
~/.macrocast/cache/ by default. Current vintages expire after 30 days;
historical vintage files never expire.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests
from tqdm import tqdm

_DEFAULT_CACHE_ROOT = Path.home() / ".macrocast" / "cache"


def get_cache_dir(override: str | Path | None = None) -> Path:
    """Return the root cache directory, creating it if absent.

    Parameters
    ----------
    override : str or Path, optional
        Alternative cache root. Uses ``~/.macrocast/cache/`` when None.

    Returns
    -------
    Path
        Absolute path to the cache root.
    """
    root = Path(override) if override is not None else _DEFAULT_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_cached_path(
    dataset: str,
    filename: str,
    cache_dir: str | Path | None = None,
) -> Path:
    """Return the expected cache path for a given dataset file.

    The directory ``{cache_dir}/{dataset}/`` is created if absent.

    Parameters
    ----------
    dataset : str
        One of ``"fred_md"``, ``"fred_qd"``, ``"fred_sd"``.
    filename : str
        Filename within the dataset subdirectory (e.g. ``"current.csv"``).
    cache_dir : str or Path, optional
        Override for cache root directory.

    Returns
    -------
    Path
        Full path where the file would be cached.
    """
    root = get_cache_dir(cache_dir)
    dest_dir = root / dataset
    dest_dir.mkdir(parents=True, exist_ok=True)
    return dest_dir / filename


def is_cached(
    dataset: str,
    filename: str,
    cache_dir: str | Path | None = None,
    max_age_days: float | None = 30.0,
) -> bool:
    """Check whether a cached file exists and is within its age limit.

    Parameters
    ----------
    dataset : str
        Dataset subdirectory name.
    filename : str
        Filename to check.
    cache_dir : str or Path, optional
        Override for cache root.
    max_age_days : float or None, optional
        Maximum age in days before the cache is considered stale.
        Pass ``None`` to disable expiry (used for vintage files).

    Returns
    -------
    bool
        True if the file exists and is not stale.
    """
    path = get_cached_path(dataset, filename, cache_dir)
    if not path.exists():
        return False
    if max_age_days is None:
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86_400
    return age_days <= max_age_days


def download_file(
    url: str,
    dest: Path,
    timeout: int = 60,
) -> Path:
    """Download a file from *url* to *dest*, showing a progress bar.

    Parameters
    ----------
    url : str
        Remote URL to fetch.
    dest : Path
        Local destination path. Parent directories must already exist.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    Path
        The destination path after successful download.

    Raises
    ------
    requests.HTTPError
        If the server returns an error status code.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; macrocast/0.1; "
            "+https://github.com/macrocast/macrocast)"
        )
    }
    response = requests.get(url, stream=True, timeout=timeout, headers=headers)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0)) or None
    filename = dest.name

    with (
        open(dest, "wb") as fh,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            desc=filename,
            leave=False,
        ) as bar,
    ):
        for chunk in response.iter_content(chunk_size=8_192):
            fh.write(chunk)
            bar.update(len(chunk))

    return dest


def file_download_date(path: Path) -> str:
    """Return the ISO date on which *path* was last written to disk.

    Parameters
    ----------
    path : Path
        Path to a cached file.

    Returns
    -------
    str
        Date string in ``"YYYY-MM-DD"`` format derived from the file's
        modification time.
    """
    import datetime

    mtime = path.stat().st_mtime
    return datetime.date.fromtimestamp(mtime).isoformat()


def clear_cache(
    dataset: str | None = None,
    cache_dir: str | Path | None = None,
) -> None:
    """Delete cached files for a dataset (or the entire cache).

    Parameters
    ----------
    dataset : str, optional
        Dataset subdirectory to clear (e.g. ``"fred_md"``). Clears
        the full cache root when None.
    cache_dir : str or Path, optional
        Override for cache root.
    """
    import shutil

    root = get_cache_dir(cache_dir)
    if dataset is None:
        shutil.rmtree(root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
    else:
        target = root / dataset
        if target.exists():
            shutil.rmtree(target)
