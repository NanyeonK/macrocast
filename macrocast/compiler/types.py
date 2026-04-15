from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..preprocessing import PreprocessContract
from ..recipes import RecipeSpec, RunSpec
from ..registry import AxisSelection
from ..stage0 import Stage0Frame


@dataclass(frozen=True)
class CompiledRecipeSpec:
    recipe_id: str
    layer_order: tuple[str, ...]
    axis_selections: tuple[AxisSelection, ...]
    leaf_config: dict[str, Any]
    preprocess_contract: PreprocessContract
    stage0: Stage0Frame
    recipe_spec: RecipeSpec
    run_spec: RunSpec
    execution_status: str
    warnings: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()
    wrapper_handoff: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CompileResult:
    compiled: CompiledRecipeSpec
    manifest: dict[str, Any] = field(default_factory=dict)
