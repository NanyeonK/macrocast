"""Lazy imports for optional deep-learning dependencies.

Deep-model executors (LSTM, GRU, TCN) call these helpers before touching
torch / pytorch-lightning so that a user who installed macrocast without
the [deep] extra gets a clear ExecutionError instead of a generic
ModuleNotFoundError at sweep time.
"""
from __future__ import annotations

from types import ModuleType

from ...errors import ExecutionError


def require_torch(model_family: str) -> ModuleType:
    try:
        import torch
    except ImportError as exc:
        raise ExecutionError(
            f"model_family {model_family!r} requires the [deep] extra. "
            "Install with: pip install macrocast[deep]"
        ) from exc
    return torch


def require_lightning(model_family: str) -> ModuleType:
    try:
        import pytorch_lightning as pl
    except ImportError as exc:
        raise ExecutionError(
            f"model_family {model_family!r} requires the [deep] extra. "
            "Install with: pip install macrocast[deep]"
        ) from exc
    return pl


__all__ = ["require_torch", "require_lightning"]
