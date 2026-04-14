# User Guide

The `macrocast` user guide explains the package in the language of applied forecasting research rather than internal implementation notes.

This guide treats `macrocast` as a general forecasting package for designing, comparing, and interpreting forecasting exercises. The structure follows the sequence commonly used in forecasting papers:
- define the forecasting problem
- define the data and information set
- define preprocessing and estimation design
- define evaluation and statistical comparison
- interpret the resulting forecasts

## Terminology

The guide uses terminology consistent with forecasting practice in outlets such as the International Journal of Forecasting (IJF), Journal of Forecasting (JoF), and related macroeconomic forecasting work.

Key terms used throughout the guide:
- target variable
- predictor set
- information set
- forecast horizon
- benchmark model
- pseudo-out-of-sample evaluation
- equal predictive ability test
- model confidence set
- variable importance / interpretability

## Guide structure

The user guide is organized by stage because the package itself is organized around a staged forecasting design.

- `Stage Map` introduces the overall workflow
- `Stage 0` defines the experiment unit and route logic
- `Stage 1` defines the data and task
- `Stage 2` defines preprocessing
- `Stage 3` defines forecasting and training
- `Stage 4` defines forecast evaluation
- `Stage 5` defines output and provenance
- `Stage 6` defines formal statistical testing
- `Stage 7` defines interpretation

## Background and terminology

Before specifying models or preprocessing choices, it helps to understand how `macrocast` maps standard forecasting concepts into package structure.

Core interpretation:
- `Stage 0` decides the experiment unit and route ownership
- `Stage 1` to `Stage 3` define the forecasting design and estimation setup
- `Stage 4` to `Stage 7` define evaluation, testing, storage, and interpretation

Recommended starting pages:
- [Stage Map](single-run/stages/index.md)
- [Stage 0](single-run/stages/stage-0/index.md)

## Forecast design and routing

This part of the guide explains what one forecasting design means inside the package.

Main pages:
- [Stage Map](single-run/stages/index.md) — workflow overview
- [Stage 0](single-run/stages/stage-0/index.md) — experiment unit, routing, and fixed-vs-sweep implications

## Data, information set, and preprocessing

These pages correspond to the part of a forecasting study where the target, predictors, information timing, sample split, and transformations are defined.

Pages:
- [Stage 1](single-run/stages/stage-1/index.md) — data / task definition
- [Stage 2](single-run/stages/stage-2/index.md) — preprocessing

## Model specification and forecast generation

These pages correspond to estimation design, validation design, benchmark construction, and forecast generation.

Pages:
- [Stage 3](single-run/stages/stage-3/index.md) — forecasting / training
- [Stage 5](single-run/stages/stage-5/index.md) — output / provenance

## Forecast evaluation and statistical comparison

These pages correspond to the empirical-comparison part of a forecasting paper.

Pages:
- [Stage 4](single-run/stages/stage-4/index.md) — evaluation metrics and benchmark-relative performance
- [Stage 6](single-run/stages/stage-6/index.md) — formal statistical testing

## Interpretation

This part of the guide corresponds to post-evaluation interpretation: once forecast quality and statistical relevance are established, what is driving the result?

Pages:
- [Stage 7](single-run/stages/stage-7/index.md) — variable importance / interpretability

## Suggested reading order

For most readers:
1. [Stage Map](single-run/stages/index.md)
2. [Stage 0](single-run/stages/stage-0/index.md)
3. [Stage 1](single-run/stages/stage-1/index.md)
4. [Stage 2](single-run/stages/stage-2/index.md) or [Stage 3](single-run/stages/stage-3/index.md), depending on the current design question
5. [Stage 4](single-run/stages/stage-4/index.md)
6. [Stage 6](single-run/stages/stage-6/index.md)
7. [Stage 7](single-run/stages/stage-7/index.md)

## Scope of the current guide

The user guide is now the public explanatory front door.
Some stage sections are still overview pages and will later expand into richer topic pages, but the guide structure is now package-facing rather than session-facing.
