# Interpretation Layer

The interpretation layer (`macrocast.interpretation`) provides tools for understanding
why a forecast model produces its predictions. It implements four complementary approaches
from the macroeconomic forecasting literature.

## Modules

| Module | Purpose |
|--------|---------|
| `dual` | Dual observation weights: decompose any forecast as a weighted average of training targets |
| `marginal` | Marginal contribution: OOS-R² gain from each information set component |
| `pbsv` | PBSV and oShapley-VI: Shapley-value decomposition by predictor group |
| `variable_importance` | Extract and aggregate variable importance from tree-based models |

## References

- Dual weights: Coulombe, Goulet-Coulombe, and Kichian (2024)
- Marginal contribution: Coulombe, Leroux, Stevanovic, Surprenant (2022)
- PBSV: Coulombe, Boldea, Renneson, Spierdijk (2022)
