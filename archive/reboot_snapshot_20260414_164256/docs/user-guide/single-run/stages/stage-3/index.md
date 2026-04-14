# Stage 3 — Forecasting / Training

Stage 3 defines how forecasts are trained, refit, validated, and tuned.

## What Stage 3 owns

- outer window family
- refit policy
- validation design
- split family
- model family
- hyperparameter search
- feature construction choices needed for training

## Main question

Given a defined task and a transformed panel, how should the forecast engine be trained and compared fairly?

## Why Stage 3 matters

This stage determines the executable forecast loop.
It is where many package comparisons become non-comparable if split fairness or benchmark logic drifts.

## Current status

Single-path operational coverage exists for the current wizard subset.
Branch-specific training behavior for model-grid and full-sweep routes still needs downstream spec work.
