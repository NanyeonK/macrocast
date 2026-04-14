# Stage 2 — Preprocessing

Stage 2 defines how targets and predictors are transformed before training.

## What Stage 2 owns

- target preprocessing
- predictor preprocessing
- missing handling
- outlier handling
- scaling and normalization
- preprocessing order
- train-only fit scope and leakage prevention

## Main question

How should the raw forecasting panel be transformed into a trainable design without leaking future information?

## Why Stage 2 is separate

Data loading and task definition belong to Stage 1.
Model and tuning logic belong to Stage 3.
Stage 2 is the layer that turns raw inputs into a valid modeling surface.

## Current status

The package has partial operational coverage for Stage 2 through the current single-run flow.
A fuller stage-native breakdown still needs to be documented and surfaced in the wizard.
