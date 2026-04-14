from pathlib import Path

import numpy as np
import pandas as pd

from macrocast.pipeline.components import CVScheme, LossFunction, Regularization, Window
from macrocast.pipeline.experiment import FeatureSpec, ForecastExperiment, ModelSpec
from macrocast.pipeline.models import RFModel


def test_recipe_metadata_drives_default_runs_layout(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    dates = pd.date_range('2005-01', periods=120, freq='MS')
    X = rng.standard_normal((120, 10))
    y = X[:, 0] + rng.standard_normal(120) * 0.2
    panel = pd.DataFrame(X, index=dates, columns=[f'x{i}' for i in range(10)])
    target = pd.Series(y, index=dates, name='target')
    spec = ModelSpec(
        model_cls=RFModel,
        regularization=Regularization.NONE,
        cv_scheme=CVScheme.KFOLD(k=2),
        loss_function=LossFunction.L2,
        model_kwargs={'n_estimators': 5, 'min_samples_leaf_grid': [5], 'cv_folds': 2},
    )
    exp = ForecastExperiment(
        panel=panel,
        target=target,
        horizons=[1],
        model_specs=[spec],
        feature_spec=FeatureSpec(n_factors=2, n_lags=2, factor_type='X'),
        window=Window.EXPANDING,
        oos_start='2014-01-01',
        oos_end='2014-01-01',
        n_jobs=1,
        output_dir=tmp_path,
        experiment_id='treepath-runtime-test',
        recipe_id='minimal_fred_md',
        taxonomy_path={'data': 'fred_md', 'model': 'random_forest'},
    )
    exp.run()
    assert (tmp_path / 'runs' / 'recipes' / 'minimal_fred_md' / 'treepath-runtime-test').exists()
