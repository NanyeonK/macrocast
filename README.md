# macrocast

Purpose

> Given a standardized macro dataset adapter and a fixed forecasting recipe, compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol.

This repository is being rebuilt from first principles.

Current structure
- `plans/` — internal planning and architecture notes
- `docs/` — public/distribution docs only
- `archive/` — preserved pre-reboot implementation snapshots

Current priority
- lock Stage 0 grammar and raw-data contracts before rebuilding registries, execution logic, and public docs

Development rule
- every code surface that is added to the package must be documented in `docs/` with detailed public-facing explanation once that code becomes part of the rebuilt package surface
