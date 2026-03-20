"""Tests for macrocast.data.fred_qd."""

from __future__ import annotations

from pathlib import Path

import pytest

from macrocast.data._base import _parse_fred_csv
from macrocast.data.fred_qd import load_fred_qd
from macrocast.data.schema import MacroFrame

FIXTURES = Path(__file__).parent / "fixtures"


class TestParseFredQdCsv:
    def test_parses_sample_csv(self) -> None:
        df, tcodes = _parse_fred_csv(FIXTURES / "fred_qd_sample.csv")
        assert "GDPC1" in df.columns
        assert "FEDFUNDS" in df.columns

    def test_correct_tcodes(self) -> None:
        _, tcodes = _parse_fred_csv(FIXTURES / "fred_qd_sample.csv")
        # From fixture: GDPC1=5, CPIAUCSL=6, FEDFUNDS=2
        assert tcodes["GDPC1"] == 5
        assert tcodes["CPIAUCSL"] == 6
        assert tcodes["FEDFUNDS"] == 2

    def test_row_count(self) -> None:
        df, _ = _parse_fred_csv(FIXTURES / "fred_qd_sample.csv")
        assert len(df) == 6


class TestLoadFredQd:
    def _mock_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> MacroFrame:
        fixture = FIXTURES / "fred_qd_sample.csv"
        import shutil

        from macrocast.data import fred_qd as fred_qd_module

        def fake_download(url: str, cache_path: Path, force_download: bool = False, timeout: int = 60) -> Path:
            shutil.copy(fixture, cache_path)
            return cache_path

        monkeypatch.setattr(fred_qd_module, "_download_fred_csv", fake_download)
        monkeypatch.setattr(fred_qd_module, "is_cached", lambda *a, **kw: False)

        return load_fred_qd(cache_dir=tmp_path)

    def test_returns_macroframe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch)
        assert isinstance(mf, MacroFrame)

    def test_dataset_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch)
        assert mf.metadata.dataset == "FRED-QD"

    def test_frequency_quarterly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        mf = self._mock_load(tmp_path, monkeypatch)
        assert mf.metadata.frequency == "quarterly"


@pytest.mark.network
class TestLoadFredQdNetwork:
    def test_download_current(self, tmp_path: Path) -> None:
        mf = load_fred_qd(cache_dir=tmp_path)
        assert isinstance(mf, MacroFrame)
        assert len(mf) > 50
