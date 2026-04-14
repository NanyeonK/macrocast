# Raw data API Reference

## Import surface

```python
from macrocast.raw import (
    RawVersionRequest,
    RawDatasetMetadata,
    RawArtifactRecord,
    RawLoadResult,
    normalize_version_request,
    list_vintages,
    get_raw_cache_root,
    get_manifest_path,
    get_raw_file_path,
    build_raw_artifact_record,
    append_raw_manifest_entry,
    read_raw_manifest,
    parse_fred_csv,
    load_fred_md,
    load_fred_qd,
    load_fred_sd,
)
```

## Shared parser

### `parse_fred_csv()`

Reads the common CSV layout used by FRED-MD and FRED-QD.

Returns:
- parsed pandas `DataFrame`
- transformation-code mapping `dict[str, int]`

Accepted layouts:
- official FRED ordering: header row then transform row
- legacy / fixture ordering: transform row then header row

## Dataset loaders

### `load_fred_md()`

```python
def load_fred_md(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
    local_source: str | Path | None = None,
    local_zip_source: str | Path | None = None,
) -> RawLoadResult:
    ...
```

### `load_fred_qd()`

```python
def load_fred_qd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
    local_source: str | Path | None = None,
) -> RawLoadResult:
    ...
```

### `load_fred_sd()`

```python
def load_fred_sd(
    vintage: str | None = None,
    *,
    force: bool = False,
    cache_root: str | Path | None = None,
    local_source: str | Path | None = None,
    states: list[str] | None = None,
    variables: list[str] | None = None,
) -> RawLoadResult:
    ...
```

Behavior:
- normalizes current versus vintage request
- chooses deterministic cache path with `.xlsx` suffix
- fetches from local source or remote workbook URL
- parses workbook sheets with `openpyxl`
- optionally filters sheets and state columns
- emits a wide DataFrame with `{variable}_{state}` columns
- appends a raw manifest entry
- returns `RawLoadResult`

## Result and metadata objects

### `RawVersionRequest`
- normalized dataset/version identity

### `RawDatasetMetadata`
- semantic dataset metadata

### `RawArtifactRecord`
- raw provenance metadata for the concrete cached artifact

### `RawLoadResult`
- canonical loader return object containing data, dataset metadata, and artifact metadata

## Error behavior

Current meaningful failure classes include:
- invalid vintage format -> `RawVersionFormatError`
- file acquisition failure -> `RawDownloadError`
- FRED-MD parse failure after acquisition -> `RawParseError`
- FRED-SD workbook parse failure -> `RawParseError`

## Example

```python
from macrocast.raw import load_fred_sd

result = load_fred_sd(
    local_source="tests/fixtures/fred_sd_sample.xlsx",
    variables=["UR"],
    states=["CA", "TX"],
)

df = result.data
```

This returns a wide DataFrame with columns like:
- `UR_CA`
- `UR_TX`

## Status note

`load_fred_sd()` is intentionally provisional in the current package state.
It exists as a documented package surface, but its live-source behavior should be treated as pending broader verification.