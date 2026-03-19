# macrocast

Decomposing ML Forecast Gains in Macroeconomic Forecasting.

An open-source Python (+ R) framework for systematic evaluation of machine learning methods in macroeconomic forecasting, with built-in support for the FRED-MD, FRED-QD, and FRED-SD database ecosystem.

## Installation

```bash
pip install macrocast
# or with all extras
pip install macrocast[all]
```

## Quick Start

```python
import macrocast as mc

# Load and transform FRED-MD
md = mc.load_fred_md()
md_t = md.transform()

# Run decomposition experiment
from macrocast.pipeline import ForecastExperiment
exp = ForecastExperiment(data=md_t, target="INDPRO", horizons=[1, 6, 12])
results = exp.run()

# Analyze
from macrocast.evaluation import decompose
decompose(results).summary()
```

## Citation

If you use macrocast in your research, please cite:

```bibtex
@article{macrocast2026,
  title={macrocast: An Open-Source Framework for Decomposing Machine Learning Gains in Macroeconomic Forecasting},
  author={...},
  journal={International Journal of Forecasting},
  year={2026}
}
```

## License

MIT
