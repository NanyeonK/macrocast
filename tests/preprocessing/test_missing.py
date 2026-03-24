"""Tests for macrocast.data.missing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.preprocessing.missing import (
    classify_missing,
    detect_missing_type,
    handle_missing,
)


@pytest.fixture
def panel() -> pd.DataFrame:
    """Panel with leading, trailing, and intermittent NaN patterns."""
    idx = pd.date_range("2000-01", periods=10, freq="MS")
    return pd.DataFrame(
        {
            "A": [np.nan, np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],  # 2 leading
            "B": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, np.nan, np.nan],  # 2 trailing
            "C": [1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0, 9.0, 10.0],  # 2 intermittent
            "D": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # no missing
        },
        index=idx,
    )


class TestDetectMissingType:
    def test_leading_nan(self, panel: pd.DataFrame) -> None:
        result = detect_missing_type(panel["A"])
        assert result["n_leading"] == 2
        assert result["n_trailing"] == 0
        assert result["n_intermittent"] == 0

    def test_trailing_nan(self, panel: pd.DataFrame) -> None:
        result = detect_missing_type(panel["B"])
        assert result["n_leading"] == 0
        assert result["n_trailing"] == 2
        assert result["n_intermittent"] == 0

    def test_intermittent_nan(self, panel: pd.DataFrame) -> None:
        result = detect_missing_type(panel["C"])
        assert result["n_leading"] == 0
        assert result["n_trailing"] == 0
        assert result["n_intermittent"] == 2

    def test_no_missing(self, panel: pd.DataFrame) -> None:
        result = detect_missing_type(panel["D"])
        assert result["n_leading"] == 0
        assert result["n_trailing"] == 0
        assert result["n_intermittent"] == 0
        assert result["pct_missing"] == 0.0

    def test_all_missing(self) -> None:
        s = pd.Series([np.nan] * 5)
        result = detect_missing_type(s)
        assert result["n_leading"] == 5
        assert result["pct_missing"] == 1.0
        assert result["first_valid_idx"] is None


class TestClassifyMissing:
    def test_returns_dataframe_with_variable_index(self, panel: pd.DataFrame) -> None:
        report = classify_missing(panel)
        assert isinstance(report, pd.DataFrame)
        assert set(panel.columns) == set(report.index)

    def test_report_columns(self, panel: pd.DataFrame) -> None:
        report = classify_missing(panel)
        for col in ["n_leading", "n_trailing", "n_intermittent", "pct_missing"]:
            assert col in report.columns


class TestHandleMissing:
    def test_trim_start_removes_leading(self, panel: pd.DataFrame) -> None:
        result = handle_missing(panel, "trim_start")
        # Series A has 2 leading NaNs; after trim_start, they are gone
        assert not result.isna().any().any() or result["A"].notna().all()
        # Row count should be reduced by 2
        assert len(result) == len(panel) - 2

    def test_drop_vars_removes_high_missing(self, panel: pd.DataFrame) -> None:
        # Add a variable with 70% missing
        df = panel.copy()
        df["sparse"] = [np.nan] * 7 + [1.0, 2.0, 3.0]
        with pytest.warns(UserWarning, match="drop_vars"):
            result = handle_missing(df, "drop_vars", max_missing_pct=0.5)
        assert "sparse" not in result.columns
        assert "D" in result.columns

    def test_interpolate_fills_only_interior(self, panel: pd.DataFrame) -> None:
        result = handle_missing(panel, "interpolate")
        # Interior gaps in C should be filled
        assert not result["C"].isna().any()
        # Leading NaN in A should remain
        assert result["A"].isna().sum() == 2
        # Trailing NaN in B should remain
        assert result["B"].isna().sum() == 2

    def test_forward_fill(self, panel: pd.DataFrame) -> None:
        result = handle_missing(panel, "forward_fill")
        # B trailing NaNs should be forward-filled
        assert not result["B"].isna().any()

    def test_em_raises_not_implemented(self, panel: pd.DataFrame) -> None:
        with pytest.raises(NotImplementedError):
            handle_missing(panel, "em")

    def test_unknown_method_raises_value_error(self, panel: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown missing-value method"):
            handle_missing(panel, "magic_fill")
