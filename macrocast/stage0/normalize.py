from __future__ import annotations

from .errors import Stage0NormalizationError
from .types import ComparisonContract, FixedDesign, ReplicationInput, StudyMode, VaryingDesign

_ALLOWED_STUDY_MODES: tuple[StudyMode, ...] = (
    "single_path_benchmark_study",
    "controlled_variation_study",
    "orchestrated_bundle_study",
    "replication_override_study",
)


def _tupleize(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        if all(isinstance(item, str) for item in value):
            return value
        raise Stage0NormalizationError("tuple values must contain only strings")
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return tuple(value)
        raise Stage0NormalizationError("list values must contain only strings")
    raise Stage0NormalizationError(f"expected tuple/list of strings, got {type(value).__name__}")


def normalize_study_mode(value: str) -> str:
    if value not in _ALLOWED_STUDY_MODES:
        raise Stage0NormalizationError(
            f"unknown study_mode={value!r}; expected one of {_ALLOWED_STUDY_MODES}"
        )
    return value


def normalize_fixed_design(value: FixedDesign | dict) -> FixedDesign:
    if isinstance(value, FixedDesign):
        return value
    if not isinstance(value, dict):
        raise Stage0NormalizationError("fixed_design must be FixedDesign or dict")
    return FixedDesign(**value)


def normalize_varying_design(value: VaryingDesign | dict | None) -> VaryingDesign:
    if value is None:
        return VaryingDesign()
    if isinstance(value, VaryingDesign):
        return value
    if not isinstance(value, dict):
        raise Stage0NormalizationError("varying_design must be VaryingDesign, dict, or None")
    payload = dict(value)
    for key in (
        "model_families",
        "feature_recipes",
        "preprocess_variants",
        "tuning_variants",
        "horizons",
    ):
        payload[key] = _tupleize(payload.get(key))
    return VaryingDesign(**payload)


def normalize_comparison_contract(value: ComparisonContract | dict) -> ComparisonContract:
    if isinstance(value, ComparisonContract):
        return value
    if not isinstance(value, dict):
        raise Stage0NormalizationError("comparison_contract must be ComparisonContract or dict")
    return ComparisonContract(**value)


def normalize_replication_input(value: ReplicationInput | dict | None) -> ReplicationInput | None:
    if value is None:
        return None
    if isinstance(value, ReplicationInput):
        return value
    if not isinstance(value, dict):
        raise Stage0NormalizationError("replication_input must be ReplicationInput, dict, or None")
    payload = dict(value)
    payload["locked_constraints"] = _tupleize(payload.get("locked_constraints"))
    return ReplicationInput(**payload)
