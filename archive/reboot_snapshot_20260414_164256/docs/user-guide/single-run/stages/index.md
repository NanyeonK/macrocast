# Stage Map

The package workflow is stage-based.

## Stage order

- Stage 0 — Meta / routing
- Stage 1 — Data / task definition
- Stage 2 — Preprocessing
- Stage 3 — Forecasting / training
- Stage 4 — Evaluation
- Stage 5 — Output / provenance
- Stage 6 — Statistical testing
- Stage 7 — Variable importance / interpretability

## Why the stage map matters

The package is not a flat option menu.
Earlier stages determine the meaning, visibility, and allowed behavior of later stages.

## Interpretation

- Stage 0 decides route ownership and whether the request stays inside `macrocast_single_run()`.
- Stages 1 to 3 define the forecasting design and training regime.
- Stages 4 to 7 define how results are evaluated, stored, tested, and interpreted.

## Recommended next pages

- `Stage 0`
- `Stage 1`
- `Stage 3`
