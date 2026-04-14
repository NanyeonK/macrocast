"""Tests for macrocast/cli.py — command-line interface."""

import tempfile
from pathlib import Path

import pytest
import yaml

from macrocast.cli import main
from macrocast.config import DEFAULT_CONFIG_YAML


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


class TestCLIInit:
    def test_init_creates_file(self, tmp_path):
        out = tmp_path / "experiment.yaml"
        ret = main(["init", "--output", str(out)])
        assert ret == 0
        assert out.exists()

    def test_init_file_is_valid_yaml(self, tmp_path):
        out = tmp_path / "experiment.yaml"
        main(["init", "--output", str(out)])
        raw = yaml.safe_load(out.read_text())
        assert "experiment" in raw
        assert "models" in raw

    def test_init_refuses_existing_without_force(self, tmp_path):
        out = tmp_path / "experiment.yaml"
        out.write_text("existing content")
        ret = main(["init", "--output", str(out)])
        assert ret != 0
        # Content should be unchanged
        assert out.read_text() == "existing content"

    def test_init_force_overwrites(self, tmp_path):
        out = tmp_path / "experiment.yaml"
        out.write_text("old content")
        ret = main(["init", "--output", str(out), "--force"])
        assert ret == 0
        assert "experiment" in out.read_text()


# ---------------------------------------------------------------------------
# info command
# ---------------------------------------------------------------------------


class TestCLIInfo:
    def _write_config(self, tmp_path: Path) -> Path:
        raw = yaml.safe_load(DEFAULT_CONFIG_YAML)
        # Trim to one model for speed
        raw["models"] = raw["models"][:1]
        raw["experiment"]["horizons"] = [1]
        cfg_path = tmp_path / "test.yaml"
        cfg_path.write_text(yaml.dump(raw))
        return cfg_path

    def test_info_returns_zero(self, tmp_path):
        cfg_path = self._write_config(tmp_path)
        ret = main(["info", str(cfg_path)])
        assert ret == 0

    def test_info_missing_file_returns_nonzero(self, tmp_path):
        ret = main(["info", str(tmp_path / "nonexistent.yaml")])
        assert ret != 0


# ---------------------------------------------------------------------------
# Parser smoke test
# ---------------------------------------------------------------------------


class TestCLIParser:
    def test_no_command_returns_nonzero(self):
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code != 0

    def test_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
