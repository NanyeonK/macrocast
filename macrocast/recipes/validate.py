from __future__ import annotations

from .errors import RecipeValidationError
from .types import RecipeSpec


def check_recipe_completeness(recipe: RecipeSpec) -> None:
    if not recipe.recipe_id.strip():
        raise RecipeValidationError("recipe_id must be non-empty")
    if recipe.targets:
        if len(recipe.targets) < 2:
            raise RecipeValidationError("multi-target recipes require at least two targets")
        if any(not target.strip() for target in recipe.targets):
            raise RecipeValidationError("all targets must be non-empty")
    else:
        if not recipe.target.strip():
            raise RecipeValidationError("target must be non-empty")
    if not recipe.raw_dataset.strip():
        raise RecipeValidationError("raw_dataset must be non-empty")
    if not recipe.horizons:
        raise RecipeValidationError("recipe must include at least one horizon")
