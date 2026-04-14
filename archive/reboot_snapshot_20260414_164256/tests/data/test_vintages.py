"""Tests for macrocast.data.vintages."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from macrocast.data.vintages import (
    RealTimePanel,
    _parse_vintage,
    list_available_vintages,
    load_vintage_panel,
)


class TestListAvailableVintages:
    def test_fred_md_default_start(self) -> None:
        vintages = list_available_vintages("fred_md", end="1999-03")
        assert vintages[0] == "1999-01"

    def test_fred_qd_default_start(self) -> None:
        vintages = list_available_vintages("fred_qd", end="2005-03")
        assert vintages[0] == "2005-01"

    def test_fred_sd_default_start(self) -> None:
        vintages = list_available_vintages("fred_sd", end="2005-03")
        assert vintages[0] == "2005-01"

    def test_custom_range(self) -> None:
        vintages = list_available_vintages("fred_md", start="2020-01", end="2020-04")
        assert vintages == ["2020-01", "2020-02", "2020-03", "2020-04"]

    def test_year_boundary(self) -> None:
        vintages = list_available_vintages("fred_md", start="2019-11", end="2020-02")
        assert vintages == ["2019-11", "2019-12", "2020-01", "2020-02"]

    def test_single_month(self) -> None:
        vintages = list_available_vintages("fred_md", start="2020-06", end="2020-06")
        assert vintages == ["2020-06"]

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            list_available_vintages("fred_xx")

    def test_all_three_datasets_accepted(self) -> None:
        for ds in ("fred_md", "fred_qd", "fred_sd"):
            vintages = list_available_vintages(ds, start="2010-01", end="2010-02")
            assert len(vintages) == 2


class TestLoadVintagePanelDispatch:
    """Test that load_vintage_panel dispatches to the correct loader."""

    def _mock_panel(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset: str,
        module_path: str,
        loader_name: str,
    ) -> dict:
        fake_mf = MagicMock()
        fake_loader = MagicMock(return_value=fake_mf)

        import importlib
        mod = importlib.import_module(module_path)
        monkeypatch.setattr(mod, loader_name, fake_loader)

        panel = load_vintage_panel(dataset, vintages=["2020-01", "2020-02"])
        return {"panel": panel, "loader": fake_loader, "mf": fake_mf}

    def test_fred_md_dispatches_correctly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_mf = MagicMock()
        monkeypatch.setattr(
            "macrocast.data.fred_md.load_fred_md", MagicMock(return_value=fake_mf)
        )
        panel = load_vintage_panel("fred_md", vintages=["2020-01"])
        assert "2020-01" in panel

    def test_fred_qd_dispatches_correctly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_mf = MagicMock()
        monkeypatch.setattr(
            "macrocast.data.fred_qd.load_fred_qd", MagicMock(return_value=fake_mf)
        )
        panel = load_vintage_panel("fred_qd", vintages=["2020-01"])
        assert "2020-01" in panel

    def test_fred_sd_dispatches_correctly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_mf = MagicMock()
        monkeypatch.setattr(
            "macrocast.data.fred_sd.load_fred_sd", MagicMock(return_value=fake_mf)
        )
        panel = load_vintage_panel("fred_sd", vintages=["2020-01"])
        assert "2020-01" in panel

    def test_unknown_dataset_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown dataset"):
            load_vintage_panel("fred_xx", vintages=["2020-01"])

    def test_returns_all_vintages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_mf = MagicMock()
        monkeypatch.setattr(
            "macrocast.data.fred_md.load_fred_md", MagicMock(return_value=fake_mf)
        )
        panel = load_vintage_panel("fred_md", vintages=["2019-01", "2020-01", "2021-01"])
        assert set(panel.keys()) == {"2019-01", "2020-01", "2021-01"}


class TestRealTimePanel:
    def _make_panel(self) -> RealTimePanel:
        mf_a = MagicMock()
        mf_b = MagicMock()
        return RealTimePanel({"2019-01": mf_a, "2020-01": mf_b})

    def test_vintages_sorted(self) -> None:
        rt = self._make_panel()
        assert rt.vintages == ["2019-01", "2020-01"]

    def test_getitem(self) -> None:
        mf = MagicMock()
        rt = RealTimePanel({"2020-01": mf})
        assert rt["2020-01"] is mf

    def test_len(self) -> None:
        rt = self._make_panel()
        assert len(rt) == 2

    def test_repr(self) -> None:
        rt = self._make_panel()
        r = repr(rt)
        assert "n_vintages=2" in r
        assert "2019-01" in r
        assert "2020-01" in r


class TestParseVintage:
    def test_valid_vintage(self) -> None:
        dt = _parse_vintage("2020-01")
        assert dt.year == 2020
        assert dt.month == 1

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid vintage format"):
            _parse_vintage("2020/01")
