# macrocast

Purpose

> Given a standardized macro dataset adapter and a fixed forecasting recipe, compare forecasting tools under identical information set, sample split, benchmark, and evaluation protocol.

This repository is being rebuilt from first principles.

Current structure
- `plans/` — internal planning and architecture notes
- `docs/` — public/distribution docs only
- `archive/` — preserved pre-reboot implementation snapshots

Current priority
- keep the rebuilt compiler/runtime surface explicit while extending public route-inspection UX through `macrocast_single_run(yaml_path=...)` before any larger wizard restoration

Development rule
- every code surface that is added to the package must be documented in `docs/` with detailed public-facing explanation once that code becomes part of the rebuilt package surface
