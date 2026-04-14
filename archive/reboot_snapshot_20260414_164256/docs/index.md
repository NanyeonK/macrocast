# Introduction

`macrocast` is a recipe-first forecasting package for building, inspecting, and later executing forecasting paths through a staged design.

The package is organized around a simple contract:
- one recipe describes one path family
- `meta` stores route-level choices
- `taxonomy_path` stores path selections
- numeric and output settings stay explicit
- `macrocast_single_run()` is the main guided entry point for one-path work

## What macrocast is for

`macrocast` is designed for users who need to:
- define one forecasting path clearly
- keep fixed axes and sweep axes conceptually separate
- compile a recipe into a structured experiment spec
- preview run identity, tree context, and manifest structure before scaling execution

## Introduction example

```python
from macrocast import macrocast_single_run

out = macrocast_single_run()
```

This starts the guided single-run flow for one path family.

## Documentation sections

The public docs are organized into five sections:
- Install & Quickstart
- User Guide
- Examples
- API Reference
- Search

## Index

Recommended reading order:
1. `Install & Quickstart`
2. `User Guide`
3. `User Guide > Stage Map`
4. `User Guide > Stage 0`
5. the relevant later stage page for the current design question

If you want one canonical example after understanding the structure:
- `Examples > CLSS 2021`
