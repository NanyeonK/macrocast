# Replication

The `macrocast.replication` module provides named presets for reproducing results
from published papers using the macrocast pipeline.

## Available Replication Studies

| Module | Paper | Status |
|--------|-------|--------|
| `clss2021` | Coulombe, Leroux, Stevanovic, Surprenant (2021), IJF | Implemented |

## Design

Each study class exposes:
- `info_sets(**params)` — dict of `FeatureSpec` objects matching the paper's information sets
- Model spec constructors for the paper's model grid

This allows exact reproduction of the experimental design without hard-coding parameters
in scripts. See the [CLSS 2021 tutorial](../tutorials/clss2021-replication.md) for a
full worked example.
