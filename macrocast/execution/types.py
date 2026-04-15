from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..preprocessing import PreprocessContract
from ..raw import RawLoadResult
from ..recipes import RecipeSpec, RunSpec


@dataclass(frozen=True)
class ExecutionSpec:
    recipe: RecipeSpec
    run: RunSpec
    preprocess: PreprocessContract


@dataclass(frozen=True)
class ExecutionResult:
    spec: ExecutionSpec
    run: RunSpec
    raw_result: RawLoadResult
    artifact_dir: str
