from __future__ import annotations

from dataclasses import dataclass

from ..stage0 import Stage0Frame


@dataclass(frozen=True)
class RecipeSpec:
    recipe_id: str
    stage0: Stage0Frame
    target: str
    horizons: tuple[int, ...]
    raw_dataset: str


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    recipe_id: str
    route_owner: str
    artifact_subdir: str
