# Stage 5 — Output / Provenance

Stage 5 defines what artifacts are saved and how run identity is preserved.

## What Stage 5 owns

- saved objects
- provenance fields
- export format
- artifact granularity
- run-manifest structure

## Main question

What should be saved so the run is reproducible, inspectable, and comparable later?

## Why Stage 5 matters

A forecasting package without strong provenance turns later evaluation and debugging into guesswork.
This stage is where the package records the identity of the run, not only its outputs.

## Current status

The package already has manifest and runs preview paths.
The stage documentation will continue to move from transitional notes into a full user guide surface.
