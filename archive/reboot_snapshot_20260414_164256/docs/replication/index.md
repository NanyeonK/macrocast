# Recipes & Runs

Recipes are the public-facing way to define one forecasting path.

Current package examples:
- baseline recipe: `recipes/baselines/minimal_fred_md.yaml`
- paper recipe: `recipes/papers/clss2021.yaml`

## Current rule

- package usage should be recipe-first
- paper studies are examples of the generic package
- package architecture should not be organized around one paper helper path

## Recommended flow

1. choose one path with `macrocast_single_run()`
2. write one YAML recipe
3. compile the recipe
4. inspect tree context and manifest preview
5. run/store outputs under `runs/`
