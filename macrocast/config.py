"""YAML experiment configuration loader.

Compatibility layer note
------------------------
This module remains available for legacy nested/flat experiment YAMLs during tree-path migration.
It is a compatibility layer, not the target long-run canonical package grammar.

Loads a YAML config file and returns a fully resolved ExperimentConfig object
that ForecastExperiment (single feature spec) or HorseRaceGrid (multiple
feature specs) can consume directly.

Two YAML formats are supported:

**Nested format** (explicit sections)::

    experiment:
      id: clss2021_rf
      output_dir: ~/.macrocast/results
      horizons: [1, 3, 6, 12]
      window: expanding
      oos_start: "1980-01-01"
      n_jobs: -1

    data:
      dataset: fred_md
      vintage: "2018-02"
      sample_start: "1960-01"
      targets: [INDPRO, PAYEMS, UNRATE]

    features:
      preset: clss2021            # loads all CLSS2021 info sets
      # OR specify individually:
      factor_type: "X"
      n_factors: 8
      n_lags: 4

    models:
      - name: elastic_net
      - name: rf

**Flat format** (concise, top-level keys)::

    experiment_id: clss2021_rf
    dataset: fred_md
    vintage: "2018-02"
    sample_start: "1960-01"
    oos_start: "1980-01"
    oos_end: "2017-12"
    targets: [INDPRO, PAYEMS, UNRATE]
    horizons: [1, 3, 6, 12]
    preset: clss2021
    models: [RF, EN, AL, AR, BVAR]
    window: expanding
    n_jobs: -1

Short model aliases (case-insensitive): AR, ARDI, Ridge, Lasso, AL (Adaptive
Lasso), EN (Elastic Net), GL (Group Lasso), TVP, Booging, BVAR, KRR, RF, GB,
XGBoost, NN, LSTM.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from macrocast.pipeline.components import Window
from macrocast.pipeline.experiment import FeatureSpec, ModelSpec
from macrocast.construction import normalise_model_list, parse_feature_specs, parse_model_spec

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class DataConfig:
    dataset: str = "fred_md"
    vintage: str | None = None
    target: str = "INDPRO"
    targets: list[str] = field(default_factory=list)
    sample_start: str | None = None
    cache_dir: str | None = None

    def all_targets(self) -> list[str]:
        """Return the deduplicated list of target variables.

        If ``targets`` is non-empty, it takes precedence.  Otherwise falls
        back to the singular ``target``.
        """
        if self.targets:
            return list(dict.fromkeys(self.targets))
        return [self.target]


@dataclass
class ExperimentConfig:
    """Fully resolved experiment configuration.

    Attributes
    ----------
    experiment_id : str
    output_dir : Path
    data : DataConfig
    feature_spec : FeatureSpec
        Primary (or only) feature spec.  Equal to ``feature_specs[0]``.
    feature_specs : list of FeatureSpec
        All feature specs.  Length > 1 when a preset is used (horse race
        across all CLSS 2021 information sets).
    model_specs : list of ModelSpec
    horizons : list of int
    window : Window
    rolling_size : int or None
    oos_start : str or None
    oos_end : str or None
    n_jobs : int
    """

    experiment_id: str
    output_dir: Path
    data: DataConfig
    feature_spec: FeatureSpec
    model_specs: list[ModelSpec]
    horizons: list[int]
    window: Window
    rolling_size: int | None
    oos_start: str | None
    oos_end: str | None
    n_jobs: int
    feature_specs: list[FeatureSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep feature_specs in sync with the primary feature_spec
        if not self.feature_specs:
            self.feature_specs = [self.feature_spec]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and parse a YAML experiment config file.

    When the config specifies multiple ``targets``, returns the first target's
    config.  Use :func:`load_configs` to get one config per target.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    ExperimentConfig
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    return _parse_config(raw)


def load_configs(path: str | Path) -> list[ExperimentConfig]:
    """Load config and return one ExperimentConfig per target variable.

    When ``targets`` lists multiple variables, a separate ExperimentConfig is
    returned for each — each with the same settings but a different
    ``data.target``.  Experiment IDs are suffixed with ``_{target}``.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    list of ExperimentConfig
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    return _parse_configs(raw)


def load_config_from_dict(raw: dict) -> ExperimentConfig:
    """Parse an already-loaded config dictionary (useful for testing)."""
    return _parse_config(raw)


def load_configs_from_dict(raw: dict) -> list[ExperimentConfig]:
    """Parse a config dict and return one config per target (useful for testing)."""
    return _parse_configs(raw)


def _normalise_raw(raw: dict) -> dict:
    """Normalise a flat config dict to nested-section format.

    Flat keys (``experiment_id``, ``dataset``, ``targets``, ``preset``,
    ``models: [RF, EN]``, etc.) are lifted into the appropriate sections
    so that ``_parse_config`` only needs to handle the nested form.
    """
    # Already nested if any of the section keys are present
    _SECTION_KEYS = {"experiment", "data", "features", "models"}
    if _SECTION_KEYS & raw.keys():
        # Possibly mixed: respect existing sections but backfill with flat keys
        out = {k: v for k, v in raw.items()}
        exp = dict(out.get("experiment", {}))
        data = dict(out.get("data", {}))
        feat = dict(out.get("features", {}))

        def _lift(flat_key: str, section: dict, section_key: str) -> None:
            if flat_key in raw and section_key not in section:
                section[section_key] = raw[flat_key]

        _lift("experiment_id", exp, "id")
        _lift("dataset",       data, "dataset")
        _lift("vintage",       data, "vintage")
        _lift("sample_start",  data, "sample_start")
        _lift("target",        data, "target")
        _lift("targets",       data, "targets")
        _lift("preset",        feat, "preset")

        out["experiment"] = exp
        out["data"]       = data
        out["features"]   = feat
        return out

    # Fully flat → lift everything
    exp = {
        "id":           raw.get("experiment_id"),
        "output_dir":   raw.get("output_dir", "~/.macrocast/results"),
        "horizons":     raw.get("horizons", [1]),
        "window":       raw.get("window", "expanding"),
        "rolling_size": raw.get("rolling_size"),
        "oos_start":    raw.get("oos_start"),
        "oos_end":      raw.get("oos_end"),
        "n_jobs":       raw.get("n_jobs", 1),
    }
    data = {
        "dataset":      raw.get("dataset", "fred_md"),
        "vintage":      raw.get("vintage"),
        "target":       raw.get("target", "INDPRO"),
        "targets":      raw.get("targets", []),
        "sample_start": raw.get("sample_start"),
        "cache_dir":    raw.get("cache_dir"),
    }
    feat = {
        "preset":          raw.get("preset"),
        "factor_type":     raw.get("factor_type", "X"),
        "n_factors":       raw.get("n_factors", 8),
        "n_lags":          raw.get("n_lags", 4),
        "p_marx":          raw.get("p_marx", 12),
        "append_x_factors":raw.get("append_x_factors", False),
        "append_marx":     raw.get("append_marx", False),
        "append_raw_x":    raw.get("append_raw_x", False),
        "append_levels":   raw.get("append_levels", False),
        "standardize_X":   raw.get("standardize_X", True),
        "standardize_Z":   raw.get("standardize_Z", False),
        "lookback":        raw.get("lookback", 12),
    }
    models = raw.get("models", [])
    return {"experiment": exp, "data": data, "features": feat, "models": models}


def _parse_configs(raw: dict) -> list[ExperimentConfig]:
    """Return one ExperimentConfig per target variable."""
    raw = _normalise_raw(raw)
    data_sec = raw.get("data", {})
    targets_raw = data_sec.get("targets") or []
    singular = data_sec.get("target", "INDPRO")
    targets = list(dict.fromkeys(targets_raw)) if targets_raw else [singular]

    base_id = (raw.get("experiment", {}).get("id") or "").rstrip("_")
    configs = []
    for tgt in targets:
        raw_copy = {
            **raw,
            "data": {**data_sec, "target": tgt, "targets": []},
            "experiment": {
                **raw.get("experiment", {}),
                "id": f"{base_id}_{tgt}" if base_id else None,
            },
        }
        configs.append(_parse_config(raw_copy))
    return configs


def _parse_config(raw: dict) -> ExperimentConfig:
    raw = _normalise_raw(raw)
    exp_sec  = raw.get("experiment", {})
    data_sec = raw.get("data", {})
    feat_sec = raw.get("features", {})
    mod_list = raw.get("models", [])

    # Experiment identity
    experiment_id = exp_sec.get("id") or str(uuid.uuid4())
    output_dir = Path(exp_sec.get("output_dir", "~/.macrocast/results")).expanduser()

    # Data
    targets_raw = data_sec.get("targets") or []
    data_cfg = DataConfig(
        dataset=data_sec.get("dataset", "fred_md"),
        vintage=data_sec.get("vintage"),
        target=data_sec.get("target", "INDPRO"),
        targets=list(dict.fromkeys(targets_raw)) if targets_raw else [],
        sample_start=data_sec.get("sample_start"),
        cache_dir=data_sec.get("cache_dir"),
    )

    # Feature specs (one or many when preset is used)
    feature_specs = parse_feature_specs(feat_sec)
    feature_spec  = feature_specs[0]

    # Outer loop config
    horizons    = [int(h) for h in exp_sec.get("horizons", [1])]
    window_str  = (exp_sec.get("window", "expanding") or "expanding").lower()
    window      = Window.EXPANDING if window_str == "expanding" else Window.ROLLING
    rolling_size = exp_sec.get("rolling_size")
    oos_start   = exp_sec.get("oos_start")
    oos_end     = exp_sec.get("oos_end")
    n_jobs      = int(exp_sec.get("n_jobs", 1))

    # Model specs — accept list-of-dicts or list-of-strings (short names)
    model_specs = [
        parse_model_spec(m) for m in normalise_model_list(mod_list)
    ]

    return ExperimentConfig(
        experiment_id=experiment_id,
        output_dir=output_dir,
        data=data_cfg,
        feature_spec=feature_spec,
        feature_specs=feature_specs,
        model_specs=model_specs,
        horizons=horizons,
        window=window,
        rolling_size=rolling_size,
        oos_start=str(oos_start) if oos_start else None,
        oos_end=str(oos_end) if oos_end else None,
        n_jobs=n_jobs,
    )


def build_experiment_config_from_components(
    *,
    experiment_id,
    output_dir,
    dataset,
    target,
    vintage=None,
    sample_start=None,
    feature_section=None,
    model_section=None,
    horizons=None,
    window='expanding',
    rolling_size=None,
    oos_start=None,
    oos_end=None,
    n_jobs=1,
):
    """Construct ExperimentConfig directly from already-resolved components."""
    data_cfg = DataConfig(
        dataset=dataset,
        vintage=vintage,
        target=target,
        targets=[],
        sample_start=sample_start,
        cache_dir=None,
    )
    feat_sec = feature_section or {}
    feature_specs = parse_feature_specs(feat_sec)
    feature_spec = feature_specs[0]
    model_specs = [parse_model_spec(m) for m in normalise_model_list(model_section or [])]
    window_str = (window or 'expanding').lower()
    window_enum = Window.EXPANDING if window_str == 'expanding' else Window.ROLLING
    return ExperimentConfig(
        experiment_id=experiment_id,
        output_dir=Path(output_dir).expanduser(),
        data=data_cfg,
        feature_spec=feature_spec,
        feature_specs=feature_specs,
        model_specs=model_specs,
        horizons=[int(h) for h in (horizons or [1])],
        window=window_enum,
        rolling_size=rolling_size,
        oos_start=str(oos_start) if oos_start else None,
        oos_end=str(oos_end) if oos_end else None,
        n_jobs=int(n_jobs),
    )

# ---------------------------------------------------------------------------
# Default config template
# ---------------------------------------------------------------------------


DEFAULT_CONFIG_YAML = """\
experiment:
  id: null                        # auto-generated if null
  output_dir: ~/.macrocast/results
  horizons: [1, 3, 6, 12]
  window: expanding               # expanding | rolling
  rolling_size: null
  oos_start: null                 # null = 80th percentile of sample
  oos_end: null
  n_jobs: -1

data:
  dataset: fred_md                # fred_md | fred_qd | fred_sd
  vintage: null                   # null = current release; YYYY-MM for vintage
  sample_start: null              # null = full sample
  target: INDPRO                  # singular target
  targets: []                     # multi-target: [INDPRO, PAYEMS, UNRATE]
  cache_dir: null                 # null = ~/.macrocast/cache

features:
  preset: null                    # clss2021 → loads all 16 info sets for horse race
  factor_type: "X"
  n_factors: 8
  n_lags: 4
  p_marx: 12
  append_x_factors: false
  append_marx: false
  append_raw_x: false
  append_levels: false
  standardize_X: true
  standardize_Z: false
  lookback: 12

# Models can be a list of detailed dicts or short name strings.
# Short names: AR, ARDI, Ridge, Lasso, AL, EN, GL, TVP, Booging, BVAR,
#              KRR, RF, GB, XGBoost, NN, LSTM
models:
  - name: krr
    regularization: factors
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      alpha_grid: [0.001, 0.01, 0.1, 1.0, 10.0]
      gamma_grid: [0.01, 0.1, 1.0]
      cv_folds: 5

  - name: rf
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      n_estimators: 500

  - name: xgboost
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2

  - name: nn
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      hidden_dims: [64, 128]
      max_epochs: 200
      patience: 20

  - name: lstm
    regularization: none
    cv_scheme: kfold
    kfold_k: 5
    loss_function: l2
    kwargs:
      hidden_dims: [32, 64]
      max_epochs: 200
"""
