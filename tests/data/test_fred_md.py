"""Tests for macrocast.data.fred_md."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from macrocast.data._base import _parse_fred_csv
from macrocast.data.fred_md import load_fred_md
from macrocast.data.schema import MacroFrame

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseFredCsv:
    """Low-level CSV parsing tests using fixture files."""

    def test_parses_sample_csv(self) -> None:
        df, tcodes = _parse_fred_csv(FIXTURES / "fred_md_sample.csv")
        assert isinstance(df, pd.DataFrame)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "INDPRO" in df.columns
        assert "CPIAUCSL" in df.columns

    def test_correct_tcodes(self) -> None:
        _, tcodes = _parse_fred_csv(FIXTURES / "fred_md_sample.csv")
        # From fixture: INDPRO=5, RPI=5, UNRATE=2, CPIAUCSL=6
        assert tcodes["INDPRO"] == 5
        assert tcodes["UNRATE"] == 2
        assert tcodes["CPIAUCSL"] == 6

    def test_date_index_is_datetime(self) -> None:
        df, _ = _parse_fred_csv(FIXTURES / "fred_md_sample.csv")
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_row_count(self) -> None:
        df, _ = _parse_fred_csv(FIXTURES / "fred_md_sample.csv")
        assert len(df) == 6  # fixture has 6 data rows

    def test_numeric_values(self) -> None:
        df, _ = _parse_fred_csv(FIXTURES / "fred_md_sample.csv")
        assert pd.api.types.is_float_dtype(df["INDPRO"])


class TestLoadFredMd:
    """Integration-style tests using monkeypatched file I/O."""

    def _mock_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> MacroFrame:
        """Set up mocks and call load_fred_md against the fixture file."""
        fixture = FIXTURES / "fred_md_sample.csv"

        # Stub is_cached to return False -> triggers download path
        # Stub _download_fred_csv to just copy fixture to cache_path
        import shutil

        from macrocast.data import fred_md as fred_md_module

        def fake_download(url: str, cache_path: Path, force_download: bool = False, timeout: int = 60) -> Path:
            shutil.copy(fixture, cache_path)
            return cache_path

        monkeypatch.setattr(fred_md_module, "_download_fred_csv", fake_download)
        monkeypatch.setattr(fred_md_module, "is_cached", lambda *a, **kw: False)

        return load_fred_md(cache_dir=tmp_path)

    def test_returns_macroframe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch)
        assert isinstance(mf, MacroFrame)

    def test_dataset_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch)
        assert mf.metadata.dataset == "FRED-MD"

    def test_frequency_monthly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch)
        assert mf.metadata.frequency == "monthly"

    def test_transform_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fixture = FIXTURES / "fred_md_sample.csv"
        import shutil

        from macrocast.data import fred_md as fred_md_module

        def fake_download(url: str, cache_path: Path, force_download: bool = False, timeout: int = 60) -> Path:
            shutil.copy(fixture, cache_path)
            return cache_path

        monkeypatch.setattr(fred_md_module, "_download_fred_csv", fake_download)
        monkeypatch.setattr(fred_md_module, "is_cached", lambda *a, **kw: False)

        mf = load_fred_md(transform=True, cache_dir=tmp_path)
        assert mf.metadata.is_transformed


@pytest.mark.network
class TestLoadFredMdNetwork:
    """Network integration tests - skipped by default."""

    def test_download_current(self, tmp_path: Path) -> None:
        mf = load_fred_md(cache_dir=tmp_path)
        assert isinstance(mf, MacroFrame)
        assert len(mf) > 100
        assert "INDPRO" in mf.data.columns
