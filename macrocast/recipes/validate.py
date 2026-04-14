from __future__ import annotations

from .errors import RecipeValidationError
from .types import RecipeSpec


def check_recipe_completeness(recipe: RecipeSpec) -> None:
    if not recipe.recipe_id.strip():
        raise RecipeValidationError("recipe_id must be non-empty")
    if not recipe.target.strip():
        raise RecipeValidationError("target must be non-empty")
    if not recipe.raw_dataset.strip():
        raise RecipeValidationError("raw_dataset must be non-empty")
    if not recipe.horizons:
        raise RecipeValidationError("recipe must include at least one horizon")
