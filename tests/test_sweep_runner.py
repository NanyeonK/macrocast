"""Integration tests for the horse-race sweep runner (Phase 1 sub-task 01.7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from macrocast.compiler.sweep_plan import compile_sweep_plan
from macrocast.execution.sweep_runner import execute_sweep


FIXTURE_RAW = Path("tests/fixtures/fred_md_ar_sample.csv")


def _horse_race_recipe() -> dict:
    return {
        "recipe_id": "sweep-rt-model",
        "path": {
            "0_meta": {"fixed_axes": {"study_mode": "controlled_variation_study"}},
            "1_data_task": {
                "fixed_axes": {
                    "dataset": "fred_md",
                    "info_set": "revised",
                    "task": "single_target_point_forecast",
                },
                "leaf_config": {"target": "INDPRO", "horizons": [1, 3]},
            },
            "2_preprocessing": {
                "fixed_axes": {
                    "target_transform_policy": "raw_level",
                    "x_transform_policy": "raw_level",
                    "tcode_policy": "raw_only",
                    "target_missing_policy": "none",
                    "x_missing_policy": "none",
                    "target_outlier_policy": "none",
                    "x_outlier_policy": "none",
                    "scaling_policy": "none",
                    "dimensionality_reduction_policy": "none",
                    "feature_selection_policy": "none",
                    "preprocess_order": "none",
                    "preprocess_fit_scope": "not_applicable",
                    "inverse_transform_policy": "none",
                    "evaluation_scale": "raw_level",
                }
            },
            "3_training": {
                "fixed_axes": {
                    "framework": "rolling",
                    "benchmark_family": "zero_change",
                    "feature_builder": "raw_feature_panel",
                },
                "sweep_axes": {"model_family": ["ridge", "lasso"]},
            },
            "4_evaluation": {"fixed_axes": {"primary_metric": "msfe"}},
            "5_output_provenance": {
                "leaf_config": {
                    "manifest_mode": "full",
                    "benchmark_config": {
                        "minimum_train_size": 5,
                        "rolling_window_size": 5,
                    },
                }
            },
            "6_stat_tests": {"fixed_axes": {"stat_test": "none"}},
            "7_importance": {"fixed_axes": {"importance_method": "minimal_importance"}},
        },
    }


def test_two_variant_sweep_end_to_end(tmp_path: Path) -> None:
    plan = compile_sweep_plan(_horse_race_recipe())
    assert plan.size == 2

    result = execute_sweep(
        plan=plan,
        output_root=tmp_path,
        local_raw_source=FIXTURE_RAW,
    )

    assert result.successful_count == 2
    assert result.failed_count == 0
    assert Path(result.manifest_path).exists()

    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["schema_version"] == "1.0"
    assert manifest["study_id"] == plan.study_id
    assert manifest["study_mode"] == "controlled_variation_study"
    assert len(manifest["sweep_plan"]["variants"]) == 2
    assert all(v["status"] == "success" for v in manifest["sweep_plan"]["variants"])

    for variant in plan.variants:
        variant_dir = tmp_path / "variants" / variant.variant_id
        assert variant_dir.exists()


def test_fail_fast_raises_on_first_failure(tmp_path: Path) -> None:
    recipe = _horse_race_recipe()
    recipe["path"]["3_training"]["sweep_axes"] = {
        "model_family": ["ridge", "not_a_real_model"],
    }
    plan = compile_sweep_plan(recipe)

    with pytest.raises(Exception):
        execute_sweep(
            plan=plan,
            output_root=tmp_path,
            local_raw_source=FIXTURE_RAW,
            fail_fast=True,
        )


def test_fail_slow_records_failure_and_continues(tmp_path: Path) -> None:
    recipe = _horse_race_recipe()
    recipe["path"]["3_training"]["sweep_axes"] = {
        "model_family": ["ridge", "not_a_real_model"],
    }
    plan = compile_sweep_plan(recipe)

    result = execute_sweep(
        plan=plan,
        output_root=tmp_path,
        local_raw_source=FIXTURE_RAW,
        fail_fast=False,
    )

    assert result.successful_count == 1
    assert result.failed_count == 1

    manifest = json.loads(Path(result.manifest_path).read_text())
    statuses = {v["status"] for v in manifest["sweep_plan"]["variants"]}
    assert statuses == {"success", "failed"}
    assert manifest["summary"]["successful"] == 1
    assert manifest["summary"]["failed"] == 1


def test_sweep_reproducibility_study_id_stable(tmp_path: Path) -> None:
    plan_a = compile_sweep_plan(_horse_race_recipe())
    plan_b = compile_sweep_plan(_horse_race_recipe())

    ra = execute_sweep(plan=plan_a, output_root=tmp_path / "a", local_raw_source=FIXTURE_RAW)
    rb = execute_sweep(plan=plan_b, output_root=tmp_path / "b", local_raw_source=FIXTURE_RAW)

    assert ra.study_id == rb.study_id

    variants_a = {v.variant_id for v in plan_a.variants}
    variants_b = {v.variant_id for v in plan_b.variants}
    assert variants_a == variants_b
