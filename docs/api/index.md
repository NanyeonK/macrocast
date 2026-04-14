# API Reference

## Available package surfaces

The rebuilt macrocast package currently exposes three documented code surfaces:

- [`macrocast.stage0`](stage0.md)
- [`macrocast.raw`](raw.md)
- [`macrocast.recipes`](recipes.md)

## Current focus

The first rebuilt package layers are:
- Stage 0, which fixes study grammar
- Raw data, which fixes version semantics, cache layout, and provenance contracts
- Recipes/execution contract, which binds Stage 0 and raw data into a minimal study/run declaration

Use the detail pages to inspect:
- dataclasses
- main helper functions
- routing and completeness checks
- raw cache and manifest helpers
- minimal recipe/run objects
