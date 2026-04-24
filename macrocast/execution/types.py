from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..preprocessing import PreprocessContract
from ..raw import RawLoadResult
from ..recipes import RecipeSpec, RunSpec


@dataclass(frozen=True)
class ExecutionSpec:
    recipe: RecipeSpec
    run: RunSpec
    preprocess: PreprocessContract


@dataclass(frozen=True)
class ExecutionResult:
    spec: ExecutionSpec
    run: RunSpec
    raw_result: RawLoadResult
    artifact_dir: str


FORECAST_PAYLOAD_CONTRACT_VERSION = "forecast_payload_v1"


@dataclass(frozen=True)
class ForecastPayload:
    y_pred: float
    selected_lag: int
    selected_bic: float
    tuning_payload: dict[str, Any] = field(default_factory=dict)
    contract_version: str = FORECAST_PAYLOAD_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        tuning_payload = dict(self.tuning_payload)
        tuning_payload.setdefault("forecast_payload_contract", self.contract_version)
        return {
            "y_pred": self.y_pred,
            "selected_lag": self.selected_lag,
            "selected_bic": self.selected_bic,
            "tuning_payload": tuning_payload,
            "contract_version": self.contract_version,
        }


@dataclass(frozen=True)
class Layer2Representation:
    Z_train: np.ndarray
    y_train: np.ndarray
    Z_pred: np.ndarray
    feature_names: tuple[str, ...]
    block_order: tuple[str, ...] = ()
    block_roles: dict[str, str] = field(default_factory=dict)
    fit_state: tuple[dict[str, Any], ...] = ()
    alignment: dict[str, Any] = field(default_factory=dict)
    leakage_contract: str = "forecast_origin_only"
    feature_builder: str = ""
    feature_runtime_builder: str = ""
    legacy_feature_builder: str = ""
    feature_dispatch_source: str = "layer2_feature_blocks"

    @property
    def latest_fit_state(self) -> dict[str, Any] | None:
        if not self.fit_state:
            return None
        return self.fit_state[-1]

    def runtime_context(self, *, mode: str) -> dict[str, Any]:
        return {
            "feature_builder": self.feature_builder,
            "feature_runtime_builder": self.feature_runtime_builder,
            "legacy_feature_builder": self.legacy_feature_builder,
            "feature_dispatch_source": self.feature_dispatch_source,
            "mode": mode,
            "feature_names": list(self.feature_names),
            "block_order": list(self.block_order),
            "block_roles": dict(self.block_roles),
            "alignment": dict(self.alignment),
            "leakage_contract": self.leakage_contract,
        }
