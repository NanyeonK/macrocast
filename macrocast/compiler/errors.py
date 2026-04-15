from __future__ import annotations


class CompileError(Exception):
    """Base exception for recipe compilation."""


class CompileValidationError(CompileError):
    """Raised when a recipe violates registry or schema rules."""
