"""Tests for macrocast.data.schema (MacroFrame)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.data.schema import MacroFrame, MacroFrameMetadata, VariableMetadata


@pytest.fixture
def sample_metadata() -> MacroFrameMetadata:
    """Minimal metadata with two variables."""
    return MacroFrameMetadata(
        dataset="TEST",
        vintage="2024-01",
        frequency="monthly",
        variables={
            "X": VariableMetadata("X", "Variable X", "group_a", tcode=5),
            "Y": VariableMetadata("Y", "Variable Y", "group_b", tcode=2),
        },
        groups={"group_a": "Group A", "group_b": "Group B"},
        is_transformed=False,
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    idx = pd.date_range("2000-01", periods=12, freq="MS")
    return pd.DataFrame(
        {
            "X": np.exp(np.linspace(0, 1, 12)) * 100,
            "Y": np.linspace(1.0, 12.0, 12),
        },
        index=idx,
    )


@pytest.fixture
def mf(sample_df: pd.DataFrame, sample_metadata: MacroFrameMetadata) -> MacroFrame:
    return MacroFrame(sample_df, sample_metadata, tcodes={"X": 5, "Y": 2})


class TestMacroFrameBasics:
    def test_data_returns_dataframe(self, mf: MacroFrame) -> None:
        assert isinstance(mf.data, pd.DataFrame)

    def test_len(self, mf: MacroFrame) -> None:
        assert len(mf) == 12

    def test_vintage(self, mf: MacroFrame) -> None:
        assert mf.vintage == "2024-01"

    def test_tcodes(self, mf: MacroFrame) -> None:
        assert mf.tcodes["X"] == 5
        assert mf.tcodes["Y"] == 2

    def test_getitem_column(self, mf: MacroFrame, sample_df: pd.DataFrame) -> None:
        pd.testing.assert_series_equal(mf["X"], sample_df["X"])

    def test_getitem_list(self, mf: MacroFrame) -> None:
        result = mf[["X", "Y"]]
        assert list(result.columns) == ["X", "Y"]

    def test_repr_contains_key_info(self, mf: MacroFrame) -> None:
        r = repr(mf)
        assert "FRED" in r or "TEST" in r
        assert "2024-01" in r
        assert "T=12" in r

    def test_repr_shows_data_through_and_download_date_when_no_vintage(
        self, sample_df: pd.DataFrame
    ) -> None:
        meta = MacroFrameMetadata(
            dataset="TEST",
            vintage=None,
            frequency="monthly",
            download_date="2026-03-19",
            data_through="2000-12",
        )
        mf_current = MacroFrame(sample_df, meta)
        r = repr(mf_current)
        assert "data_through='2000-12'" in r
        assert "download_date='2026-03-19'" in r
        assert "vintage" not in r

    def test_repr_shows_vintage_when_set(self, mf: MacroFrame) -> None:
        r = repr(mf)
        assert "vintage='2024-01'" in r

    def test_data_through_set_from_data(self, sample_df: pd.DataFrame) -> None:
        meta = MacroFrameMetadata(
            dataset="TEST",
            vintage=None,
            frequency="monthly",
            data_through="2000-12",
        )
        mf_c = MacroFrame(sample_df, meta)
        assert mf_c.metadata.data_through == "2000-12"

    def test_data_through_recomputed_after_trim(self, mf: MacroFrame) -> None:
        trimmed = mf.trim(end="2000-06")
        assert trimmed.metadata.data_through == "2000-06"

    def test_data_through_preserved_after_transform(self, mf: MacroFrame) -> None:
        meta_with_through = MacroFrameMetadata(
            dataset=mf.metadata.dataset,
            vintage=mf.metadata.vintage,
            frequency=mf.metadata.frequency,
            variables=mf.metadata.variables,
            groups=mf.metadata.groups,
            data_through="2000-12",
        )
        mf2 = MacroFrame(mf.data, meta_with_through, mf.tcodes)
        transformed = mf2.transform()
        assert transformed.metadata.data_through == "2000-12"

    def test_to_numpy_shape(self, mf: MacroFrame) -> None:
        arr = mf.to_numpy()
        assert arr.shape == (12, 2)
        assert arr.dtype == float

    def test_data_is_copy(
        self, mf: MacroFrame, sample_df: pd.DataFrame
    ) -> None:
        # Modifying the underlying df should not change mf.data
        original_val = mf.data.iloc[0, 0]
        sample_df.iloc[0, 0] = 999.0
        assert mf.data.iloc[0, 0] == original_val


class TestMacroFrameTransform:
    def test_transform_returns_new_instance(self, mf: MacroFrame) -> None:
        result = mf.transform()
        assert result is not mf

    def test_transform_sets_is_transformed(self, mf: MacroFrame) -> None:
        result = mf.transform()
        assert result.metadata.is_transformed

    def test_original_unchanged(self, mf: MacroFrame) -> None:
        mf.transform()
        assert not mf.metadata.is_transformed

    def test_transform_with_override(self, mf: MacroFrame) -> None:
        result = mf.transform(override={"X": 1, "Y": 1})
        # tcode 1 = level, so data should be identical
        pd.testing.assert_frame_equal(result.data, mf.data)


class TestMacroFrameTrim:
    def test_trim_start(self, mf: MacroFrame) -> None:
        result = mf.trim(start="2000-03")
        assert result.data.index[0] >= pd.Timestamp("2000-03-01")

    def test_trim_end(self, mf: MacroFrame) -> None:
        result = mf.trim(end="2000-06")
        assert result.data.index[-1] <= pd.Timestamp("2000-06-30")

    def test_trim_min_obs_pct_drops_sparse(
        self, sample_df: pd.DataFrame, sample_metadata: MacroFrameMetadata
    ) -> None:
        import warnings

        df = sample_df.copy()
        df["Z"] = np.nan  # 100% missing
        # Add Z to metadata
        meta = MacroFrameMetadata(
            dataset="TEST",
            vintage=None,
            frequency="monthly",
            variables={
                **sample_metadata.variables,
                "Z": VariableMetadata("Z", "Sparse", "group_a", tcode=1),
            },
            groups=sample_metadata.groups,
        )
        mf2 = MacroFrame(df, meta, tcodes={"X": 5, "Y": 2, "Z": 1})
        with warnings.catch_warnings(record=True):
            result = mf2.trim(min_obs_pct=0.5)
        assert "Z" not in result.data.columns


class TestMacroFrameGroup:
    def test_group_filters_columns(self, mf: MacroFrame) -> None:
        result = mf.group("group_a")
        assert list(result.data.columns) == ["X"]

    def test_group_unknown_raises_key_error(self, mf: MacroFrame) -> None:
        with pytest.raises(KeyError, match="(?i)no variables found"):
            mf.group("nonexistent_group")


class TestMacroFrameHandleMissing:
    def test_handle_missing_returns_new_instance(self, mf: MacroFrame) -> None:
        result = mf.handle_missing("forward_fill")
        assert result is not mf

    def test_missing_report_returns_dataframe(self, mf: MacroFrame) -> None:
        report = mf.missing_report()
        assert isinstance(report, pd.DataFrame)
        assert set(mf.data.columns).issubset(set(report.index))


class TestMacroFrameOutlierFlag:
    def test_returns_boolean_dataframe(self, mf: MacroFrame) -> None:
        flags = mf.outlier_flag()
        assert flags.dtypes.eq(bool).all()
        assert flags.shape == mf.data.shape

    def test_unsupported_method_raises(self, mf: MacroFrame) -> None:
        with pytest.raises(ValueError, match="Unsupported outlier method"):
            mf.outlier_flag(method="zscore")
