from __future__ import annotations


class Stage0Error(Exception):
    """Base exception for stage0 operations."""


class Stage0NormalizationError(Stage0Error):
    """Raised when stage0 inputs cannot be normalized."""


class Stage0ValidationError(Stage0Error):
    """Raised when stage0 inputs violate required structure."""


class Stage0CompletenessError(Stage0Error):
    """Raised when a stage0 frame is structurally incomplete for execution."""


class Stage0RoutingError(Stage0Error):
    """Raised when no valid route owner can be derived from a stage0 frame."""
