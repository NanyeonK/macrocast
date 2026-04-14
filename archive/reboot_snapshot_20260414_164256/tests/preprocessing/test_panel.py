"""Tests for macrocast/data/preprocessing.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from macrocast.preprocessing.panel import (
    BaseTransform,
    CustomTransform,
    DemeanTransform,
    DropTransform,
    HPFilterTransform,
    PanelTransformer,
    StandardizeTransform,
    WinsorizeTransform,
    _resolve_scope,
)
from macrocast.data.schema import MacroFrameMetadata, VariableMetadata


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_df(n: int = 60, cols: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    cols = cols or ["a", "b", "c"]
    dates = pd.date_range("2000-01", periods=n, freq="MS")
    return pd.DataFrame(rng.standard_normal((n, len(cols))), index=dates, columns=cols)


def _make_metadata(groups: dict[str, str] | None = None) -> MacroFrameMetadata:
    """Create minimal MacroFrameMetadata with two variables."""
    groups = groups or {}
    variables = {
        "a": VariableMetadata(name="a", description="A", group="output", tcode=5, frequency="monthly"),
        "b": VariableMetadata(name="b", description="B", group="prices", tcode=2, frequency="monthly"),
        "c": VariableMetadata(name="c", description="C", group="output", tcode=5, frequency="monthly"),
    }
    return MacroFrameMetadata(
        dataset="test",
        vintage=None,
        frequency="monthly",
        variables=variables,
        groups={"output": "Output", "prices": "Prices"},
    )


# ---------------------------------------------------------------------------
# _resolve_scope
# ---------------------------------------------------------------------------


class TestResolveScope:
    def test_all(self) -> None:
        df = _make_df()
        assert _resolve_scope("all", df, None) == ["a", "b", "c"]

    def test_explicit_list(self) -> None:
        df = _make_df()
        assert _resolve_scope(["a", "c"], df, None) == ["a", "c"]

    def test_explicit_list_filters_missing(self) -> None:
        df = _make_df()
        assert _resolve_scope(["a", "z"], df, None) == ["a"]

    def test_group_scope(self) -> None:
        df = _make_df()
        meta = _make_metadata()
        result = _resolve_scope("group:output", df, meta)
        assert set(result) == {"a", "c"}

    def test_group_scope_requires_metadata(self) -> None:
        df = _make_df()
        with pytest.raises(ValueError, match="metadata"):
            _resolve_scope("group:output", df, None)

    def test_tcode_scope(self) -> None:
        df = _make_df()
        meta = _make_metadata()
        result = _resolve_scope("tcode:5", df, meta)
        assert set(result) == {"a", "c"}

    def test_tcode_scope_requires_metadata(self) -> None:
        df = _make_df()
        with pytest.raises(ValueError, match="metadata"):
            _resolve_scope("tcode:5", df, None)

    def test_regex_scope(self) -> None:
        df = _make_df(cols=["x1", "x2", "y1"])
        result = _resolve_scope("re:^x", df, None)
        assert set(result) == {"x1", "x2"}

    def test_unknown_scope_raises(self) -> None:
        df = _make_df()
        with pytest.raises(ValueError, match="Unrecognised"):
            _resolve_scope("bad_scope", df, None)


# ---------------------------------------------------------------------------
# WinsorizeTransform
# ---------------------------------------------------------------------------


class TestWinsorizeTransform:
    def test_clips_outliers(self) -> None:
        df = _make_df()
        # Inject extreme outlier
        df.iloc[0, 0] = 1000.0
        t = WinsorizeTransform(lower_pct=0.01, upper_pct=0.99, scope="all")
        out = t.fit_transform(df)
        assert out.iloc[0, 0] < 1000.0

    def test_fit_before_transform_required(self) -> None:
        t = WinsorizeTransform()
        df = _make_df()
        with pytest.raises(RuntimeError, match="fit"):
            t.transform(df)

    def test_oos_uses_training_thresholds(self) -> None:
        train = _make_df(n=50)
        oos = _make_df(n=10)
        oos.iloc[0, 0] = 999.0
        t = WinsorizeTransform(lower_pct=0.01, upper_pct=0.99)
        t.fit(train)
        out = t.transform(oos)
        assert out.iloc[0, 0] < 999.0

    def test_scope_subset(self) -> None:
        df = _make_df()
        df.iloc[0, 0] = 1000.0
        df.iloc[0, 1] = 1000.0
        t = WinsorizeTransform(scope=["a"])
        out = t.fit_transform(df)
        # column a winsorized, column b unchanged
        assert out["a"].iloc[0] < 1000.0
        assert out["b"].iloc[0] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# DemeanTransform
# ---------------------------------------------------------------------------


class TestDemeanTransform:
    def test_zero_mean_after_fit_transform(self) -> None:
        df = _make_df()
        t = DemeanTransform()
        out = t.fit_transform(df)
        assert out.mean().abs().max() < 1e-10

    def test_fit_before_transform_required(self) -> None:
        t = DemeanTransform()
        with pytest.raises(RuntimeError, match="fit"):
            t.transform(_make_df())

    def test_oos_uses_training_mean(self) -> None:
        train = _make_df(n=100)
        train_mean = train.mean()
        oos = _make_df(n=10)
        t = DemeanTransform()
        t.fit(train)
        out = t.transform(oos)
        expected = oos - train_mean
        pd.testing.assert_frame_equal(out, expected)


# ---------------------------------------------------------------------------
# StandardizeTransform
# ---------------------------------------------------------------------------


class TestStandardizeTransform:
    def test_unit_variance_after_fit_transform(self) -> None:
        df = _make_df()
        t = StandardizeTransform()
        out = t.fit_transform(df)
        assert out.std().sub(1.0).abs().max() < 1e-6

    def test_zero_std_columns_unchanged(self) -> None:
        df = _make_df()
        df["const"] = 5.0
        t = StandardizeTransform()
        out = t.fit_transform(df)
        # constant column should not blow up
        assert out["const"].isna().sum() == 0

    def test_fit_before_transform_required(self) -> None:
        t = StandardizeTransform()
        with pytest.raises(RuntimeError, match="fit"):
            t.transform(_make_df())


# ---------------------------------------------------------------------------
# HPFilterTransform
# ---------------------------------------------------------------------------


class TestHPFilterTransform:
    def test_cycle_has_lower_variance(self) -> None:
        df = _make_df(n=120)
        t = HPFilterTransform(lambda_=1600, component="cycle")
        out = t.fit_transform(df)
        assert (out.var() < df.var()).all()

    def test_trend_monotone_ish(self) -> None:
        """Trend component should have substantially lower std than raw."""
        df = _make_df(n=120)
        t = HPFilterTransform(lambda_=1600, component="trend")
        out = t.fit_transform(df)
        assert (out.std() < df.std()).all()

    def test_invalid_component_raises(self) -> None:
        with pytest.raises(ValueError, match="component"):
            HPFilterTransform(component="bad")

    def test_short_series_skipped(self) -> None:
        """Columns with < 4 obs should be left unchanged."""
        df = _make_df(n=3)
        t = HPFilterTransform()
        out = t.fit_transform(df)
        pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# CustomTransform
# ---------------------------------------------------------------------------


class TestCustomTransform:
    def test_applies_function(self) -> None:
        df = _make_df()
        df = df.abs() + 1.0  # ensure positive
        t = CustomTransform(fn=np.log, scope="all")
        out = t.fit_transform(df)
        pd.testing.assert_frame_equal(out, np.log(df))

    def test_scope_subset(self) -> None:
        df = _make_df()
        df_abs = df.abs() + 1.0
        t = CustomTransform(fn=np.log, scope=["a"])
        out = t.fit_transform(df_abs)
        assert out["a"].equals(np.log(df_abs["a"]))
        assert out["b"].equals(df_abs["b"])


# ---------------------------------------------------------------------------
# DropTransform
# ---------------------------------------------------------------------------


class TestDropTransform:
    def test_drops_columns(self) -> None:
        df = _make_df()
        t = DropTransform(scope=["a", "c"])
        out = t.fit_transform(df)
        assert list(out.columns) == ["b"]

    def test_regex_scope(self) -> None:
        df = _make_df(cols=["x1", "x2", "y1"])
        t = DropTransform(scope="re:^x")
        out = t.fit_transform(df)
        assert list(out.columns) == ["y1"]

    def test_missing_columns_ignored(self) -> None:
        df = _make_df()
        t = DropTransform(scope=["a", "z_nonexistent"])
        out = t.fit_transform(df)
        assert "a" not in out.columns
        assert "b" in out.columns


# ---------------------------------------------------------------------------
# PanelTransformer
# ---------------------------------------------------------------------------


class TestPanelTransformer:
    def test_chaining(self) -> None:
        df = _make_df()
        pt = PanelTransformer([
            WinsorizeTransform(lower_pct=0.05, upper_pct=0.95),
            DemeanTransform(),
        ])
        out = pt.fit_transform(df)
        assert out.shape == df.shape
        assert out.mean().abs().max() < 1e-10

    def test_transform_before_fit_raises(self) -> None:
        pt = PanelTransformer([DemeanTransform()])
        with pytest.raises(RuntimeError, match="fit"):
            pt.transform(_make_df())

    def test_oos_uses_fit_params(self) -> None:
        train = _make_df(n=100)
        oos = _make_df(n=10)
        pt = PanelTransformer([DemeanTransform()])
        pt.fit(train)
        out = pt.transform(oos)
        # Should be shifted by training mean
        expected = oos - train.mean()
        pd.testing.assert_frame_equal(out, expected)

    def test_metadata_forwarded(self) -> None:
        df = _make_df()
        meta = _make_metadata()
        pt = PanelTransformer([DemeanTransform(scope="group:output")], metadata=meta)
        out = pt.fit_transform(df)
        # columns b (prices group) unchanged
        pd.testing.assert_series_equal(out["b"], df["b"])
        # columns a, c (output group) demeaned
        assert abs(out["a"].mean()) < 1e-10

    def test_repr(self) -> None:
        pt = PanelTransformer([DemeanTransform(), WinsorizeTransform()])
        assert "DemeanTransform" in repr(pt)
        assert "WinsorizeTransform" in repr(pt)

    def test_empty_steps(self) -> None:
        df = _make_df()
        pt = PanelTransformer([])
        out = pt.fit_transform(df)
        pd.testing.assert_frame_equal(out, df)
