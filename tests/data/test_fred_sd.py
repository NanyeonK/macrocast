"""Tests for macrocast.data.fred_sd."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from macrocast.data.fred_sd import load_fred_sd
from macrocast.data.schema import MacroFrame

FIXTURES = Path(__file__).parent / "fixtures"


class TestLoadFredSd:
    def _mock_load(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        **kwargs: object,
    ) -> MacroFrame:
        fixture = FIXTURES / "fred_sd_sample.xlsx"
        import shutil

        from macrocast.data import fred_sd as fred_sd_module

        def fake_download(url: str, dest: Path, **kw: object) -> Path:
            shutil.copy(fixture, dest)
            return dest

        monkeypatch.setattr(fred_sd_module, "download_file", fake_download)
        monkeypatch.setattr(fred_sd_module, "is_cached", lambda *a, **kw: False)

        return load_fred_sd(cache_dir=tmp_path, **kwargs)

    def test_returns_macroframe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch)
        assert isinstance(mf, MacroFrame)

    def test_column_names_have_state_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch)
        assert any("_" in c for c in mf.data.columns)

    def test_state_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch, states=["CA"])
        assert all(c.endswith("_CA") for c in mf.data.columns)

    def test_variable_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch, variables=["UR"])
        assert all(c.startswith("UR_") for c in mf.data.columns)

    def test_dataset_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch)
        assert mf.metadata.dataset == "FRED-SD"

    def test_date_trimming(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch, start="2000-02", end="2000-03")
        assert len(mf) == 2

    def test_current_vintage_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch)
        assert mf.vintage is None

    def test_vintage_sets_metadata(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch, vintage="2020-01")
        assert mf.vintage == "2020-01"

    def test_vintage_filename_in_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Vintage load should write to YYYY-MM.xlsx, not FRED_SD.xlsx."""
        fixture = FIXTURES / "fred_sd_sample.xlsx"
        import shutil

        from macrocast.data import fred_sd as fred_sd_module

        downloaded: list[str] = []

        def fake_download(url: str, dest: Path, **kw: object) -> Path:
            downloaded.append(dest.name)
            shutil.copy(fixture, dest)
            return dest

        monkeypatch.setattr(fred_sd_module, "download_file", fake_download)
        monkeypatch.setattr(fred_sd_module, "is_cached", lambda *a, **kw: False)

        load_fred_sd(vintage="2020-01", cache_dir=tmp_path)
        assert downloaded == ["2020-01.xlsx"]

    def test_current_filename_in_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Current load should write to FRED_SD.xlsx."""
        fixture = FIXTURES / "fred_sd_sample.xlsx"
        import shutil

        from macrocast.data import fred_sd as fred_sd_module

        downloaded: list[str] = []

        def fake_download(url: str, dest: Path, **kw: object) -> Path:
            downloaded.append(dest.name)
            shutil.copy(fixture, dest)
            return dest

        monkeypatch.setattr(fred_sd_module, "download_file", fake_download)
        monkeypatch.setattr(fred_sd_module, "is_cached", lambda *a, **kw: False)

        load_fred_sd(cache_dir=tmp_path)
        assert downloaded == ["FRED_SD.xlsx"]


@pytest.mark.network
@pytest.mark.skip(reason="FRED-SD does not have a stable public direct-download URL.")
class TestLoadFredSdNetwork:
    def test_download_current(self, tmp_path: Path) -> None:
        mf = load_fred_sd(states=["CA", "TX"], cache_dir=tmp_path)
        assert isinstance(mf, MacroFrame)

    def test_download_vintage(self, tmp_path: Path) -> None:
        mf = load_fred_sd(vintage="2020-01", states=["CA"], cache_dir=tmp_path)
        assert isinstance(mf, MacroFrame)
        assert mf.vintage == "2020-01"
