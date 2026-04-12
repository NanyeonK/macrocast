# CLSS 2021 Replication

This tutorial is now a migration note, not the package's organizing center.

## Current package position

CLSS 2021 should be represented as one paper recipe/path:
- `recipes/papers/clss2021.yaml`

Package-specific CLSS helpers remain available only as migration scaffolding:
- `macrocast.replication.clss2021`
- `macrocast.replication.clss2021_runner`

## What to treat as canonical

Canonical for architecture:
- tree-path package plan
- recipe schema
- paper recipe file

Non-canonical for long-run package design:
- helper modules whose names are specific to one paper

## Migration-safe usage

If you need a CLSS-oriented reduced audit run during migration, use the helper path intentionally and label it as scaffolding.
If you are documenting target package structure, reference the recipe path instead.
