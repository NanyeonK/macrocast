from __future__ import annotations

from pathlib import Path

from .types import RawVersionRequest


def get_raw_cache_root(cache_root: str | Path | None = None) -> Path:
    root = Path(cache_root).expanduser() if cache_root is not None else Path("~/.macrocast/raw").expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def get_manifest_path(cache_root: str | Path | None = None) -> Path:
    root = get_raw_cache_root(cache_root)
    path = root / "manifest" / "raw_artifacts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_raw_file_path(
    request: RawVersionRequest,
    cache_root: str | Path | None = None,
    *,
    suffix: str,
) -> Path:
    root = get_raw_cache_root(cache_root)
    if request.mode == "current":
        path = root / request.dataset / "current" / f"raw.{suffix}"
    else:
        if request.vintage is None:
            raise ValueError("vintage mode requires vintage string")
        path = root / request.dataset / "vintages" / f"{request.vintage}.{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
