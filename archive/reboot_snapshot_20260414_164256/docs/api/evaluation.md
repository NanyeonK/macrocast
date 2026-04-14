# macrocast.evaluation

`macrocast.evaluation` contains the public API for forecast metrics, formal comparison tests, decomposition, forecast combination, and evaluation registries.

## Import

```python
from macrocast.evaluation import relative_msfe, dm_test, cw_test
```

## Metrics { #macrocastevaluationmetrics }

Metric exports:
- `msfe`
- `mae`
- `relative_msfe`
- `csfe`
- `oos_r2`

Purpose:
- compute point forecast quality and benchmark-relative forecast performance

## Formal comparison tests

### macrocast.evaluation.dm { #macrocastevaluationdm }

Test exports:
- `dm_test`
- `DMResult`

Purpose:
- compare forecast accuracy under the Diebold-Mariano framework

### macrocast.evaluation.cw { #macrocastevaluationcw }

Test exports:
- `cw_test`
- `CWResult`

Purpose:
- compare nested models under the Clark-West adjustment

### macrocast.evaluation.gw { #macrocastevaluationgw }

Test exports:
- `gw_test`

Purpose:
- evaluate conditional predictive ability under the Giacomini-White framework

### macrocast.evaluation.mcs { #macrocastevaluationmcs }

Test exports:
- `mcs`
- `MCSResult`

Purpose:
- identify a model confidence set among competing forecasting candidates

## Regime and decomposition tools

### macrocast.evaluation.regime { #macrocastevaluationregime }

Exports:
- `regime_conditional_msfe`

Purpose:
- analyze forecast performance across regimes

### macrocast.evaluation.decomposition { #macrocastevaluationdecomposition }

Exports:
- `decompose_treatment_effects`

Purpose:
- decompose performance gains across treatment dimensions

### macrocast.evaluation.horserace { #macrocastevaluationhorserace }

Exports:
- `horserace_summary`

Purpose:
- summarize forecast horse-race comparisons across candidates

### macrocast.evaluation.combination { #macrocastevaluationcombination }

Exports:
- `combine_forecasts`

Purpose:
- combine forecast series into aggregate predictions

## Registry helpers

Evaluation-registry exports:
- `load_evaluation_registry`
- `load_test_registry`
- `validate_evaluation_registry`
- `validate_test_registry`

Purpose:
- expose canonical metric/test registry configuration and validation logic

## Related pages

- `User Guide > Stage 4`
- `User Guide > Stage 6`
