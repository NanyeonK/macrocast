from __future__ import annotations

import numpy as np

from macrocast.tuning import HPDistribution, TuningBudget, run_tuning
from macrocast.tuning.types import TuningSpec
from macrocast.tuning.validation.splitter import LastBlockSplitter, RollingBlocksSplitter, ExpandingValidationSplitter, BlockedKFoldSplitter
from macrocast.tuning.validation.scorer import get_scorer


class DummyModel:
    def __init__(self, hp=None):
        self.bias = float((hp or {}).get("bias", 0.0))
    def fit(self, X, y):
        self.mean_ = float(np.mean(y)) + self.bias
        return self
    def predict(self, X):
        return np.full(len(X), self.mean_)


def test_hp_distribution_sampling() -> None:
    rng = np.random.RandomState(42)
    assert 0.0 <= HPDistribution("float", 0.0, 1.0).sample(rng) <= 1.0
    assert 1 <= HPDistribution("int", 1, 3).sample(rng) <= 3
    assert HPDistribution("categorical", choices=("a", "b")).sample(rng) in {"a", "b"}


def test_temporal_splitters_produce_nonempty_splits() -> None:
    n = 20
    splitters = [LastBlockSplitter(4), RollingBlocksSplitter(3, 4), ExpandingValidationSplitter(5), BlockedKFoldSplitter(4)]
    for splitter in splitters:
        splits = list(splitter.split(n))
        assert splits
        for train_idx, val_idx in splits:
            assert len(train_idx) > 0 and len(val_idx) > 0


def test_get_scorer_validation_mse() -> None:
    scorer = get_scorer("validation_mse")
    assert scorer(np.array([1.0, 2.0]), np.array([1.0, 3.0])) == 0.5


def test_run_tuning_grid_random_bayes_genetic() -> None:
    X = np.arange(40, dtype=float).reshape(20, 2)
    y = np.linspace(0.0, 1.0, 20)
    hp_space = {"bias": HPDistribution("float", -0.1, 0.1)}
    for algo in ["grid_search", "random_search", "bayesian_optimization", "genetic_algorithm"]:
        spec = TuningSpec(
            search_algorithm=algo,
            tuning_objective="validation_mse",
            tuning_budget={"max_trials": 4, "max_time_seconds": 5.0, "early_stop_trials": 2},
            hp_space=hp_space,
            validation_size_rule="ratio",
            validation_size_config={"ratio": 0.2, "n": 3, "years": 1, "obs_per_year": 12},
            validation_location="last_block",
            embargo_gap="none",
            embargo_gap_size=0,
            seed=42,
        )
        result = run_tuning("ridge", lambda hp: DummyModel(hp), X, y, spec)
        assert result.total_trials >= 1
        assert isinstance(result.best_hp, dict)
