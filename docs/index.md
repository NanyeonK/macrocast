# macrocast

`macrocast` is moving toward a generic tree-path forecasting package.

Target package structure:
- `taxonomy/`: selectable forecasting choice universe
- `registries/`: backing defaults/adapters/contracts
- `recipes/`: named studies/baselines/benchmarks/ablations
- `runs/`: realized outputs keyed by resolved path or recipe

The package should be generic first.
Paper studies such as CLSS 2021 should be expressed as one recipe/path through the package, not as package-defining core logic.

## Current migration state

Completed first-pass migration layers:
- taxonomy bundle
- registries layer skeleton
- recipes layer skeleton
- recipe-aware compile path
- benchmark family/options redesign
- runs layer skeleton

## Start here

- package direction: `docs/planning/treepath-package-overhaul.md`
- migration buckets: `docs/planning/treepath-migration-map.md`
- baseline recipe example: `recipes/baselines/minimal_fred_md.yaml`
- paper recipe example: `recipes/papers/clss2021.yaml`
