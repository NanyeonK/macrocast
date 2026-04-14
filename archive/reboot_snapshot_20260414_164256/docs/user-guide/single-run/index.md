# Single Run

`macrocast_single_run()` is the main public entry point for building one path family.

## Purpose

Use `macrocast_single_run()` when you want to:
- create one recipe YAML
- choose one path step by step
- inspect route classification early
- preview compile and manifest behavior for executable single-run paths

## Entry function

```python
from macrocast import macrocast_single_run

out = macrocast_single_run()
```

## Current package boundary

Executable now:
- single-target single-model path
- recipe-native compile preview flow
- tree-context preview
- runs / manifest preview

Planned but not yet fully expanded downstream:
- single-target model grid
- single-target full sweep

Outside current single-run execution boundary:
- multi-target families
- benchmark suites
- ablation bundles
- replication bundles

## Read next

- `Stage Map`
- `Stage 0`
