"""Tests for macrocast/utils/registry.py — ExperimentRegistry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macrocast.pipeline.components import (
    CVScheme,
    LossFunction,
    Nonlinearity,
    Regularization,
    Window,
)
from macrocast.pipeline.results import ForecastRecord, ResultSet
from macrocast.utils.registry import ExperimentRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    experiment_id: str = "test-exp",
    model_id: str = "ar__none__bic__l2",
    horizon: int = 1,
    y_hat: float = 1.0,
    y_true: float = 1.5,
) -> ForecastRecord:
    return ForecastRecord(
        experiment_id=experiment_id,
        model_id=model_id,
        nonlinearity=Nonlinearity.LINEAR,
        regularization=Regularization.NONE,
        cv_scheme=CVScheme.BIC,
        loss_function=LossFunction.L2,
        window=Window.EXPANDING,
        horizon=horizon,
        train_end=pd.Timestamp("2010-01-01"),
        forecast_date=pd.Timestamp("2010-02-01"),
        y_hat=y_hat,
        y_true=y_true,
        n_train=100,
        n_factors=None,
        n_lags=4,
    )


def _make_result_set(
    experiment_id: str = "test-exp",
    n_records: int = 6,
    horizons: list[int] | None = None,
) -> ResultSet:
    """Create a minimal ResultSet with n_records records."""
    if horizons is None:
        horizons = [1, 3]
    rs = ResultSet(experiment_id=experiment_id)
    dates = pd.date_range("2010-01", periods=n_records // len(horizons), freq="MS")
    for date in dates:
        for h in horizons:
            rs.add(ForecastRecord(
                experiment_id=experiment_id,
                model_id="ar__none__bic__l2",
                nonlinearity=Nonlinearity.LINEAR,
                regularization=Regularization.NONE,
                cv_scheme=CVScheme.BIC,
                loss_function=LossFunction.L2,
                window=Window.EXPANDING,
                horizon=h,
                train_end=date,
                forecast_date=date + pd.offsets.MonthBegin(1),
                y_hat=float(np.random.randn()),
                y_true=float(np.random.randn()),
                n_train=100,
                n_factors=None,
                n_lags=4,
            ))
    return rs


# ---------------------------------------------------------------------------
# ExperimentRegistry — init and basic attributes
# ---------------------------------------------------------------------------


class TestExperimentRegistryInit:
    def test_default_root_dir(self, tmp_path: Path) -> None:
        # Just verify a custom root_dir is stored correctly
        reg = ExperimentRegistry(root_dir=tmp_path)
        assert reg.root_dir == tmp_path.resolve()

    def test_default_root_dir_is_homedir(self) -> None:
        reg = ExperimentRegistry()
        assert str(reg.root_dir).endswith(
            str(Path(".macrocast") / "results")
        )

    def test_root_dir_expands_user(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path / "test")
        assert reg.root_dir.is_absolute()


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------


class TestExperimentRegistrySave:
    def test_save_creates_directory(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("my-exp")
        reg.save(rs)
        assert (tmp_path / "my-exp").is_dir()

    def test_save_creates_parquet(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("my-exp")
        reg.save(rs)
        assert (tmp_path / "my-exp" / "results.parquet").exists()

    def test_save_creates_meta_json(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("my-exp")
        reg.save(rs)
        assert (tmp_path / "my-exp" / "meta.json").exists()

    def test_save_returns_path(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("my-exp")
        path = reg.save(rs)
        assert path == tmp_path / "my-exp"

    def test_save_override_experiment_id(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("original-id")
        reg.save(rs, experiment_id="override-id")
        assert (tmp_path / "override-id").is_dir()
        assert not (tmp_path / "original-id").exists()

    def test_save_meta_has_n_records(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("my-exp", n_records=6)
        reg.save(rs)
        meta = json.loads((tmp_path / "my-exp" / "meta.json").read_text())
        assert meta["n_records"] == 6

    def test_save_meta_has_horizons(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("my-exp", horizons=[1, 3])
        reg.save(rs)
        meta = json.loads((tmp_path / "my-exp" / "meta.json").read_text())
        assert meta["horizons"] == [1, 3]

    def test_save_meta_has_model_ids(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("my-exp")
        reg.save(rs)
        meta = json.loads((tmp_path / "my-exp" / "meta.json").read_text())
        assert len(meta["model_ids"]) >= 1

    def test_save_empty_result_set_raises(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = ResultSet(experiment_id="empty-exp")
        with pytest.raises(ValueError, match="empty"):
            reg.save(rs)

    def test_save_no_experiment_id_raises(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = ResultSet(experiment_id="")  # empty string experiment_id
        rs.add(_make_record())
        with pytest.raises(ValueError, match="experiment_id"):
            reg.save(rs)

    def test_save_custom_metadata_merged(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("my-exp")
        reg.save(rs, metadata={"target": "INDPRO", "vintage": "2018-02"})
        meta = json.loads((tmp_path / "my-exp" / "meta.json").read_text())
        assert meta["target"] == "INDPRO"
        assert meta["vintage"] == "2018-02"


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class TestExperimentRegistryLoad:
    def test_load_returns_result_set(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs_orig = _make_result_set("my-exp")
        reg.save(rs_orig)
        rs = reg.load("my-exp")
        assert isinstance(rs, ResultSet)

    def test_load_preserves_records_count(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs_orig = _make_result_set("my-exp", n_records=6)
        reg.save(rs_orig)
        rs = reg.load("my-exp")
        df = rs.to_dataframe_cached()
        assert len(df) == 6

    def test_load_preserves_horizons(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs_orig = _make_result_set("my-exp", horizons=[1, 3])
        reg.save(rs_orig)
        rs = reg.load("my-exp")
        df = rs.to_dataframe_cached()
        assert set(df["horizon"].unique()) == {1, 3}

    def test_load_preserves_experiment_id(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs_orig = _make_result_set("roundtrip-exp")
        reg.save(rs_orig)
        rs = reg.load("roundtrip-exp")
        assert rs.experiment_id == "roundtrip-exp"

    def test_load_populates_metadata(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs_orig = _make_result_set("my-exp")
        reg.save(rs_orig, metadata={"target": "INDPRO"})
        rs = reg.load("my-exp")
        assert rs.metadata.get("target") == "INDPRO"

    def test_load_not_found_raises(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            reg.load("nonexistent")


# ---------------------------------------------------------------------------
# exists
# ---------------------------------------------------------------------------


class TestExperimentRegistryExists:
    def test_exists_true_after_save(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("my-exp")
        reg.save(rs)
        assert reg.exists("my-exp") is True

    def test_exists_false_before_save(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        assert reg.exists("nonexistent") is False

    def test_exists_false_after_delete(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("my-exp")
        reg.save(rs)
        reg.delete("my-exp")
        assert reg.exists("my-exp") is False


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestExperimentRegistryDelete:
    def test_delete_removes_directory(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        rs = _make_result_set("my-exp")
        reg.save(rs)
        reg.delete("my-exp")
        assert not (tmp_path / "my-exp").exists()

    def test_delete_not_found_raises(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            reg.delete("nonexistent")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestExperimentRegistryList:
    def test_list_empty_registry(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        df = reg.list()
        assert df.empty

    def test_list_returns_dataframe(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        reg.save(_make_result_set("exp-a"))
        df = reg.list()
        assert isinstance(df, pd.DataFrame)

    def test_list_counts_experiments(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        reg.save(_make_result_set("exp-a"))
        reg.save(_make_result_set("exp-b"))
        df = reg.list()
        assert len(df) == 2

    def test_list_has_experiment_id_column(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        reg.save(_make_result_set("my-exp"))
        df = reg.list()
        assert "experiment_id" in df.columns

    def test_list_has_n_records_column(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        reg.save(_make_result_set("my-exp"))
        df = reg.list()
        assert "n_records" in df.columns

    def test_list_n_records_correct(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        reg.save(_make_result_set("my-exp", n_records=6))
        df = reg.list()
        assert df.loc[df["experiment_id"] == "my-exp", "n_records"].iloc[0] == 6

    def test_list_has_created_at_column(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(root_dir=tmp_path)
        reg.save(_make_result_set("my-exp"))
        df = reg.list()
        assert "created_at" in df.columns

    def test_list_empty_root_dir(self, tmp_path: Path) -> None:
        # root_dir does not exist yet
        reg = ExperimentRegistry(root_dir=tmp_path / "nonexistent")
        df = reg.list()
        assert df.empty


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


class TestExperimentRegistryCompare:
    def _make_two_exps(self, tmp_path: Path) -> ExperimentRegistry:
        """Save two experiments with known MSFE for AR model, horizon 1."""
        reg = ExperimentRegistry(root_dir=tmp_path)

        # exp-a: perfect forecast (se=0)
        rs_a = ResultSet(experiment_id="exp-a")
        for _ in range(5):
            rs_a.add(_make_record("exp-a", horizon=1, y_hat=1.0, y_true=1.0))
        reg.save(rs_a)

        # exp-b: large error (se=4 each)
        rs_b = ResultSet(experiment_id="exp-b")
        for _ in range(5):
            rs_b.add(_make_record("exp-b", horizon=1, y_hat=1.0, y_true=3.0))
        reg.save(rs_b)

        return reg

    def test_compare_returns_dataframe(self, tmp_path: Path) -> None:
        reg = self._make_two_exps(tmp_path)
        df = reg.compare(["exp-a", "exp-b"])
        assert isinstance(df, pd.DataFrame)

    def test_compare_rows_are_experiment_ids(self, tmp_path: Path) -> None:
        reg = self._make_two_exps(tmp_path)
        df = reg.compare(["exp-a", "exp-b"])
        assert "exp-a" in df.index
        assert "exp-b" in df.index

    def test_compare_columns_are_horizons(self, tmp_path: Path) -> None:
        reg = self._make_two_exps(tmp_path)
        df = reg.compare(["exp-a", "exp-b"])
        assert 1 in df.columns

    def test_compare_msfe_values(self, tmp_path: Path) -> None:
        reg = self._make_two_exps(tmp_path)
        df = reg.compare(["exp-a", "exp-b"])
        assert df.loc["exp-a", 1] == pytest.approx(0.0)
        assert df.loc["exp-b", 1] == pytest.approx(4.0)

    def test_compare_relative_benchmark(self, tmp_path: Path) -> None:
        reg = self._make_two_exps(tmp_path)
        # relative to exp-b (msfe=4) → exp-a should be 0/4=0, exp-b=4/4=1
        df = reg.compare(["exp-a", "exp-b"], benchmark_id="exp-b")
        assert df.loc["exp-b", 1] == pytest.approx(1.0)
        assert df.loc["exp-a", 1] == pytest.approx(0.0)

    def test_compare_filter_horizons(self, tmp_path: Path) -> None:
        reg = self._make_two_exps(tmp_path)
        df = reg.compare(["exp-a", "exp-b"], horizons=[1])
        assert list(df.columns) == [1]
