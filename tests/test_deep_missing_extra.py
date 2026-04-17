"""Verify require_torch raises ExecutionError with install hint when torch is absent.

This test runs in every CI matrix row (not marked deep) — it simulates the
core-only install by stubbing torch out of ``sys.modules`` before calling
``require_torch``.
"""
from __future__ import annotations

import builtins
import sys
from unittest.mock import patch

import pytest

from macrocast.execution.errors import ExecutionError


def test_require_torch_raises_when_missing():
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated missing torch")
        return real_import(name, *args, **kwargs)

    with patch.dict(sys.modules, {"torch": None}):
        with patch("builtins.__import__", side_effect=blocked_import):
            from macrocast.execution.models.deep._import_guard import require_torch

            with pytest.raises(ExecutionError) as excinfo:
                require_torch("lstm")
    msg = str(excinfo.value)
    assert "lstm" in msg
    assert "[deep]" in msg
    assert "pip install macrocast[deep]" in msg


def test_require_lightning_raises_when_missing():
    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "pytorch_lightning":
            raise ImportError("simulated missing pytorch_lightning")
        return real_import(name, *args, **kwargs)

    with patch.dict(sys.modules, {"pytorch_lightning": None}):
        with patch("builtins.__import__", side_effect=blocked_import):
            from macrocast.execution.models.deep._import_guard import require_lightning

            with pytest.raises(ExecutionError) as excinfo:
                require_lightning("lstm")
    assert "[deep]" in str(excinfo.value)
