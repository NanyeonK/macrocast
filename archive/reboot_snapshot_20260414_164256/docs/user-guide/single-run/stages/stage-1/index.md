# Stage 1 — Data / Task Definition

Stage 1 defines the forecasting problem before any preprocessing or model fitting starts.

## What Stage 1 owns

- data domain and dataset source
- frequency and information set
- sample and out-of-sample period
- forecast task definition
- target and predictor mapping

## Main question

What exact forecasting task is this recipe describing?

## Typical choices

- domain
- data
- frequency
- info_set
- sample
- oos
- task
- target
- x_map

## Why Stage 1 comes before preprocessing

Stage 2 should transform a clearly defined task.
It should not decide the task itself.

## Current status

The package has partial operational coverage for Stage 1 through the current single-run flow.
A fuller hierarchical Stage 1 spec still needs to be documented choice-by-choice.
