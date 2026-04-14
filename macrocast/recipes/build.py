from __future__ import annotations

from .types import RecipeSpec, RunSpec
from ..stage0 import resolve_route_owner


def build_run_spec(recipe: RecipeSpec) -> RunSpec:
    route_owner = resolve_route_owner(recipe.stage0)
    horizon_token = "-".join(str(h) for h in recipe.horizons)
    run_id = f"{recipe.recipe_id}__{recipe.target}__h{horizon_token}"
    return RunSpec(
        run_id=run_id,
        recipe_id=recipe.recipe_id,
        route_owner=route_owner,
        artifact_subdir=f"runs/{run_id}",
    )


def recipe_summary(recipe: RecipeSpec) -> str:
    route_owner = resolve_route_owner(recipe.stage0)
    horizons = ", ".join(str(h) for h in recipe.horizons)
    return (
        f"recipe_id={recipe.recipe_id}; target={recipe.target}; raw_dataset={recipe.raw_dataset}; "
        f"route={route_owner}; horizons=[{horizons}]"
    )
