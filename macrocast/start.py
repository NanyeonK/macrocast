from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .compiler import compile_recipe_yaml

_AVAILABLE_STAGES = (
    "route_preview",
    "compile_preview",
    "tree_context",
    "runs_preview",
    "manifest_preview",
)


def _normalize_stages(stages: str | Iterable[str] | None) -> list[str]:
    if stages is None:
        return list(_AVAILABLE_STAGES)
    if isinstance(stages, str):
        stages = [stages]
    normalized: list[str] = []
    for stage in stages:
        if stage not in _AVAILABLE_STAGES:
            raise ValueError(f"unknown stage {stage!r}; valid stages: {_AVAILABLE_STAGES}")
        normalized.append(stage)
    return normalized


def _tree_context_summary(tree_context: dict[str, Any]) -> str:
    fixed_names = ",".join(sorted(tree_context.get("fixed_axes", {}))) or "none"
    sweep_names = ",".join(sorted(tree_context.get("sweep_axes", {}))) or "none"
    conditional_names = ",".join(sorted(tree_context.get("conditional_axes", {}))) or "none"
    return (
        f"route_owner={tree_context.get('route_owner', 'unknown')}; "
        f"execution_posture={tree_context.get('execution_posture', 'unknown')}; "
        f"fixed_axes=[{fixed_names}]; "
        f"sweep_axes=[{sweep_names}]; "
        f"conditional_axes=[{conditional_names}]"
    )


def _route_preview(compile_manifest: dict[str, Any]) -> dict[str, Any]:
    tree_context = dict(compile_manifest.get("tree_context", {}))
    route_owner = tree_context.get("route_owner", compile_manifest.get("run_spec", {}).get("route_owner", "unknown"))
    execution_status = compile_manifest.get("execution_status", "unknown")
    warnings = list(compile_manifest.get("warnings", []))
    blocked_reasons = list(compile_manifest.get("blocked_reasons", []))

    if route_owner == "wrapper":
        wizard_status = "wrapper_required"
        continue_in_single_run = False
        message = "Route is wrapper-owned. Inspect compiler provenance and wrapper_handoff, but do not treat it as a runnable single-path preview."
    elif execution_status == "executable":
        wizard_status = "implemented"
        continue_in_single_run = True
        message = "Route remains inside the current executable single-run surface."
    elif tree_context.get("execution_posture") == "single_run_with_internal_sweep":
        wizard_status = "planned_single_run_extension"
        continue_in_single_run = False
        message = "Route still belongs to the single-run family, but downstream internal sweep branching is not implemented yet."
    else:
        wizard_status = "blocked_or_nonexecutable"
        continue_in_single_run = False
        message = "; ".join(blocked_reasons or warnings) or "Route is not executable in the current single-run surface."

    return {
        "route_owner": route_owner,
        "execution_status": execution_status,
        "wizard_status": wizard_status,
        "continue_in_single_run": continue_in_single_run,
        "message": message,
        "warnings": warnings,
        "blocked_reasons": blocked_reasons,
        "tree_context_summary": _tree_context_summary(tree_context) if tree_context else "",
        "wrapper_handoff": dict(compile_manifest.get("wrapper_handoff", {})),
    }


def _runs_preview(compile_manifest: dict[str, Any], *, output_root: str | Path) -> dict[str, Any]:
    run_spec = dict(compile_manifest["run_spec"])
    artifact_dir = Path(output_root) / run_spec["artifact_subdir"]
    return {
        "output_root": str(Path(output_root)),
        "artifact_subdir": run_spec["artifact_subdir"],
        "artifact_dir_preview": str(artifact_dir),
        "run_id": run_spec["run_id"],
        "route_owner": run_spec["route_owner"],
    }


def _manifest_preview(compile_manifest: dict[str, Any], *, output_root: str | Path) -> dict[str, Any]:
    tree_context = dict(compile_manifest.get("tree_context", {}))
    leaf_config = dict(tree_context.get("leaf_config", compile_manifest.get("leaf_config", {})))
    stat_test = dict(compile_manifest.get("stat_test_spec", {})).get("stat_test", "none")
    importance_method = dict(compile_manifest.get("importance_spec", {})).get("importance_method", "none")
    expected_artifacts = [
        "manifest.json",
        "summary.txt",
        "data_preview.csv",
        "predictions.csv",
        "metrics.json",
        "comparison_summary.json",
    ]
    if stat_test == "dm":
        expected_artifacts.append("stat_test_dm.json")
    if stat_test == "cw":
        expected_artifacts.append("stat_test_cw.json")
    if importance_method == "minimal_importance":
        expected_artifacts.append("importance_minimal.json")
    return {
        "recipe_id": compile_manifest["recipe_id"],
        "run_id": compile_manifest["run_spec"]["run_id"],
        "artifact_dir_preview": str(Path(output_root) / compile_manifest["run_spec"]["artifact_subdir"]),
        "route_owner": compile_manifest["run_spec"]["route_owner"],
        "target": leaf_config.get("target", ""),
        "targets": list(leaf_config.get("targets", [])),
        "horizons": list(leaf_config.get("horizons", [])),
        "benchmark_spec": dict(compile_manifest.get("benchmark_spec", {})),
        "model_spec": dict(compile_manifest.get("model_spec", {})),
        "preprocess_contract": dict(compile_manifest.get("preprocess_contract", {})),
        "tree_context": tree_context,
        "expected_artifacts": expected_artifacts,
    }


def macrocast_single_run(
    *,
    yaml_path: str,
    stages: str | Iterable[str] | None = None,
    output_root: str = "/tmp/macrocast_single_run_preview",
) -> dict[str, Any]:
    selected = _normalize_stages(stages)
    compile_result = compile_recipe_yaml(yaml_path)
    compile_manifest = compile_result.manifest
    route_preview = _route_preview(compile_manifest)

    out: dict[str, Any] = {
        "selected_stages": selected,
        "input_yaml_path": str(Path(yaml_path)),
        "route_preview": route_preview,
    }

    if "compile_preview" in selected:
        out["compile_preview"] = compile_manifest
    if "tree_context" in selected:
        out["tree_context"] = dict(compile_manifest.get("tree_context", {}))

    blocked_preview_stages: list[str] = []
    if compile_result.compiled.execution_status != "executable":
        blocked_preview_stages = [stage for stage in selected if stage in {"runs_preview", "manifest_preview"}]
        if blocked_preview_stages:
            out["blocked_preview_stages"] = blocked_preview_stages
            out["blocked_preview_reason"] = route_preview["message"]
        return out

    if "runs_preview" in selected:
        out["runs_preview"] = _runs_preview(compile_manifest, output_root=output_root)
    if "manifest_preview" in selected:
        out["manifest_preview"] = _manifest_preview(compile_manifest, output_root=output_root)
    return out
