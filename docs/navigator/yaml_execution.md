# YAML And Execution

YAML is the bridge between the docs navigator and runtime execution. A selected tree path should be serializable, inspectable, editable, and executable.

## Generate YAML

```bash
macrocast-navigate replications synthetic-replication-roundtrip \
  --write-yaml /tmp/synthetic-replication.yaml
```

## Resolve YAML

```bash
macrocast-navigate resolve /tmp/synthetic-replication.yaml
```

Run only when `execution_status` is `executable`.

## Run YAML

```bash
macrocast-navigate run /tmp/synthetic-replication.yaml \
  --local-raw-source tests/fixtures/fred_md_ar_sample.csv \
  --output-root /tmp/macrocast-synthetic
```

The command returns:

```json
{
  "execution_status": "executed",
  "artifact_dir": "/tmp/macrocast-synthetic/runs/...",
  "run_id": "..."
}
```

## Notebook Pattern

```python
from pathlib import Path
import yaml

from macrocast.navigator import replication_recipe_yaml
from macrocast import compile_recipe_dict, run_compiled_recipe

recipe = yaml.safe_load(replication_recipe_yaml("synthetic-replication-roundtrip"))
compiled = compile_recipe_dict(recipe)

result = run_compiled_recipe(
    compiled.compiled,
    output_root="/tmp/macrocast-synthetic",
    local_raw_source=Path("tests/fixtures/fred_md_ar_sample.csv"),
)
```

## Expected Outputs

The exact set depends on the selected path, but ordinary executable runs should write:

- `manifest.json`;
- `predictions.csv`;
- `metrics.json`;
- `comparison_summary.json`;
- optional payload files such as `forecast_payloads.jsonl`;
- optional test artifacts such as `stat_test_dm.json`.

## When To Use API Docs

Use API docs after YAML execution works and you need function signatures, extension hooks, or custom plugin internals. For choosing a path, use the navigator pages first.
