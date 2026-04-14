from __future__ import annotations


class RecipeExecutionError(Exception):
    """Base exception for recipe/execution contract operations."""


class RecipeValidationError(RecipeExecutionError):
    """Raised when a recipe spec is structurally invalid."""
