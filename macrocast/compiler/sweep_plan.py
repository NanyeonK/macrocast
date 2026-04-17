"""Horse-race sweep plan compiler.

Expands a recipe dict's ``sweep_axes`` entries (across layers) into a
Cartesian product of concrete single-path variant recipe dicts. Each
variant is fully-specified and compilable through ``compile_recipe_dict``.

Part of Phase 1 (horse-race sweep executor - IDENTITY UNLOCK).
See plans/phases/phase_01_sweep_executor.md section 4.1.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from typing import Any

DEFAULT_MAX_VARIANTS = 1000


class SweepPlanError(ValueError):
    """Raised when a recipe dict cannot be expanded into a valid sweep plan."""


@dataclass(frozen=True)
class SweepVariant:
    """One fully-specified variant recipe derived from a parent sweep recipe.

    Attributes:
        variant_id: Stable identifier ``v-<8-hex>`` derived from the axis
            values (same values -> same id across runs and machines).
        axis_values: Layer-qualified axis values fixed for this variant,
            e.g. ``{"3_training.model_family": "ridge"}``.
        parent_recipe_id: The sweep parent recipe's ``recipe_id``.
        variant_recipe_dict: The variant's standalone recipe dict with
            ``sweep_axes`` merged into ``fixed_axes``; compilable by
            ``compile_recipe_dict`` as a single-path recipe.
    """

    variant_id: str
    axis_values: dict[str, str]
    parent_recipe_id: str
    variant_recipe_dict: dict[str, Any]


@dataclass(frozen=True)
class SweepPlan:
    """A compiled sweep plan: parent + Cartesian-expanded variants."""

    study_id: str
    parent_recipe_id: str
    parent_recipe_dict: dict[str, Any]
    axes_swept: tuple[str, ...]
    variants: tuple[SweepVariant, ...]

    @property
    def size(self) -> int:
        return len(self.variants)


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _variant_id(axis_values: dict[str, str]) -> str:
    digest = hashlib.sha256(_canonical_json(axis_values).encode("utf-8")).hexdigest()
    return f"v-{digest[:8]}"


def _study_id(
    parent_recipe_id: str,
    axes_swept: tuple[str, ...],
    variant_axis_values: list[dict[str, str]],
) -> str:
    payload = {
        "parent_recipe_id": parent_recipe_id,
        "axes_swept": list(axes_swept),
        "variants": variant_axis_values,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return f"sha256-{digest[:16]}"


def _layer_keys(recipe_dict: dict[str, Any]) -> list[str]:
    path = recipe_dict.get("path")
    if not isinstance(path, dict):
        raise SweepPlanError("recipe dict missing 'path' object")
    return sorted(path.keys())


def _collect_sweep_axes(
    recipe_dict: dict[str, Any],
) -> list[tuple[str, str, list[Any]]]:
    triples: list[tuple[str, str, list[Any]]] = []
    for layer in _layer_keys(recipe_dict):
        layer_block = recipe_dict["path"][layer]
        if not isinstance(layer_block, dict):
            continue
        sweep_axes = layer_block.get("sweep_axes") or {}
        if not isinstance(sweep_axes, dict):
            raise SweepPlanError(
                f"layer '{layer}': sweep_axes must be a mapping, got "
                f"{type(sweep_axes).__name__}"
            )
        fixed_axes = layer_block.get("fixed_axes") or {}
        if not isinstance(fixed_axes, dict):
            raise SweepPlanError(
                f"layer '{layer}': fixed_axes must be a mapping, got "
                f"{type(fixed_axes).__name__}"
            )
        for axis_name, values in sorted(sweep_axes.items()):
            if axis_name in fixed_axes:
                raise SweepPlanError(
                    f"layer '{layer}': axis '{axis_name}' appears in both "
                    f"fixed_axes and sweep_axes - pick one"
                )
            if not isinstance(values, (list, tuple)) or len(values) == 0:
                raise SweepPlanError(
                    f"layer '{layer}': sweep_axes['{axis_name}'] must be a "
                    f"non-empty list of values"
                )
            triples.append((layer, axis_name, list(values)))
    return triples


def _materialise_variant(
    parent_recipe_dict: dict[str, Any],
    picks: dict[tuple[str, str], Any],
    variant_id: str,
) -> dict[str, Any]:
    variant = copy.deepcopy(parent_recipe_dict)
    parent_id = variant.get("recipe_id", "recipe")
    variant["recipe_id"] = f"{parent_id}#{variant_id}"

    for (layer, axis_name), value in picks.items():
        layer_block = variant["path"][layer]
        sweep_axes = layer_block.get("sweep_axes") or {}
        if axis_name in sweep_axes:
            new_sweep_axes = {k: v for k, v in sweep_axes.items() if k != axis_name}
            if new_sweep_axes:
                layer_block["sweep_axes"] = new_sweep_axes
            else:
                layer_block.pop("sweep_axes", None)
        fixed_axes = dict(layer_block.get("fixed_axes") or {})
        fixed_axes[axis_name] = value
        layer_block["fixed_axes"] = fixed_axes

    return variant


def compile_sweep_plan(
    recipe_dict: dict[str, Any],
    *,
    max_variants: int | None = DEFAULT_MAX_VARIANTS,
) -> SweepPlan:
    """Expand ``sweep_axes`` across layers into a Cartesian SweepPlan.

    Args:
        recipe_dict: Parent recipe dict containing one or more
            ``path.<layer>.sweep_axes`` entries.
        max_variants: Upper bound on generated variants, default 1000. Pass
            ``None`` to disable (not recommended for user-supplied recipes).

    Returns:
        A :class:`SweepPlan` whose ``variants`` can each be handed to
        :func:`macrocast.compile_recipe_dict` as a standalone single-path
        recipe.

    Raises:
        SweepPlanError: if the recipe dict has no sweep_axes, if a sweep
            axis duplicates a fixed axis on the same layer, or if the
            Cartesian size exceeds ``max_variants``.
    """

    if not isinstance(recipe_dict, dict):
        raise SweepPlanError("recipe_dict must be a mapping")
    parent_recipe_id = recipe_dict.get("recipe_id", "recipe")

    sweep_triples = _collect_sweep_axes(recipe_dict)
    if not sweep_triples:
        raise SweepPlanError(
            "recipe dict has no sweep_axes - use compile_recipe_dict for "
            "single-path recipes"
        )

    axes_swept = tuple(f"{layer}.{axis}" for layer, axis, _ in sweep_triples)

    total = 1
    for _, _, values in sweep_triples:
        total *= len(values)
    if max_variants is not None and total > max_variants:
        raise SweepPlanError(
            f"sweep would produce {total} variants, exceeds max_variants="
            f"{max_variants}. Narrow sweep_axes or raise max_variants."
        )

    variants: list[SweepVariant] = []
    variant_axis_values: list[dict[str, str]] = []
    seen_variant_ids: set[str] = set()

    for combo in itertools.product(*[values for _, _, values in sweep_triples]):
        picks: dict[tuple[str, str], Any] = {}
        axis_values: dict[str, str] = {}
        for (layer, axis_name, _values), value in zip(sweep_triples, combo):
            picks[(layer, axis_name)] = value
            axis_values[f"{layer}.{axis_name}"] = value

        vid = _variant_id(axis_values)
        if vid in seen_variant_ids:
            raise SweepPlanError(
                f"variant_id collision for axis_values={axis_values}; this "
                "should be impossible for distinct sweep combinations"
            )
        seen_variant_ids.add(vid)

        variant_dict = _materialise_variant(recipe_dict, picks, vid)
        variants.append(
            SweepVariant(
                variant_id=vid,
                axis_values=dict(axis_values),
                parent_recipe_id=parent_recipe_id,
                variant_recipe_dict=variant_dict,
            )
        )
        variant_axis_values.append(dict(axis_values))

    study_id = _study_id(parent_recipe_id, axes_swept, variant_axis_values)

    return SweepPlan(
        study_id=study_id,
        parent_recipe_id=parent_recipe_id,
        parent_recipe_dict=copy.deepcopy(recipe_dict),
        axes_swept=axes_swept,
        variants=tuple(variants),
    )


__all__ = [
    "SweepPlan",
    "SweepVariant",
    "SweepPlanError",
    "compile_sweep_plan",
    "DEFAULT_MAX_VARIANTS",
]
