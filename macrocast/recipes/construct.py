from __future__ import annotations

from typing import Any

from .types import RecipeSpec
from ..stage0 import Stage0Frame


def build_recipe_spec(
    *,
    recipe_id: str,
    stage0: Stage0Frame,
    target: str,
    horizons: tuple[int, ...],
    raw_dataset: str,
    benchmark_config: dict[str, Any] | None = None,
) -> RecipeSpec:
    return RecipeSpec(
        recipe_id=recipe_id,
        stage0=stage0,
        target=target,
        horizons=tuple(horizons),
        raw_dataset=raw_dataset,
        benchmark_config=dict(benchmark_config or {}),
    )
