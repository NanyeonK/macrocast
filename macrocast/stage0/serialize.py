from __future__ import annotations

from dataclasses import asdict

from .normalize import (
    normalize_comparison_contract,
    normalize_fixed_design,
    normalize_replication_input,
    normalize_study_mode,
    normalize_varying_design,
)
from .types import Stage0Frame
from .build import build_stage0_frame


def stage0_to_dict(stage0: Stage0Frame) -> dict:
    return asdict(stage0)


def stage0_from_dict(payload: dict) -> Stage0Frame:
    return build_stage0_frame(
        study_mode=normalize_study_mode(payload["study_mode"]),
        fixed_design=normalize_fixed_design(payload["fixed_design"]),
        comparison_contract=normalize_comparison_contract(payload["comparison_contract"]),
        varying_design=normalize_varying_design(payload.get("varying_design")),
        replication_input=normalize_replication_input(payload.get("replication_input")),
    )
