# macrocast.utils

`macrocast.utils` contains cache helpers, registry utilities, and LaTeX-format export helpers.

## Import

```python
from macrocast.utils import get_cache_dir, ExperimentRegistry, rmsfe_to_latex
```

## Cache helpers

Cache exports:
- `get_cache_dir`
- `get_cached_path`
- `is_cached`
- `download_file`
- `file_download_date`
- `clear_cache`

Purpose:
- manage local file caching and cached downloads

## Registry utility

Registry export:
- `ExperimentRegistry`

Purpose:
- track experiment identities and related registry behavior

## LaTeX helpers

LaTeX exports:
- `rmsfe_to_latex`
- `regime_to_latex`

Purpose:
- convert result objects into manuscript-friendly table fragments

## Related pages

- `Examples > Recipes & Runs`
- `API Reference > macrocast.evaluation`
