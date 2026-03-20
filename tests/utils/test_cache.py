"""Tests for macrocast.utils.cache."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from macrocast.utils.cache import (
    clear_cache,
    download_file,
    file_download_date,
    get_cache_dir,
    get_cached_path,
    is_cached,
)


class TestGetCacheDir:
    def test_default_returns_home_macrocast(self, tmp_path: Path) -> None:
        result = get_cache_dir(override=tmp_path / "cache")
        assert result == tmp_path / "cache"
        assert result.exists()

    def test_creates_directory(self, tmp_path: Path) -> None:
        target = tmp_path / "new" / "nested" / "dir"
        get_cache_dir(override=target)
        assert target.exists()

    def test_none_uses_default(self) -> None:
        result = get_cache_dir()
        assert result.name == "cache"
        assert result.parent.name == ".macrocast"


class TestGetCachedPath:
    def test_returns_correct_path(self, tmp_path: Path) -> None:
        p = get_cached_path("fred_md", "current.csv", cache_dir=tmp_path)
        assert p == tmp_path / "fred_md" / "current.csv"

    def test_creates_dataset_subdirectory(self, tmp_path: Path) -> None:
        get_cached_path("fred_md", "current.csv", cache_dir=tmp_path)
        assert (tmp_path / "fred_md").exists()


class TestIsCached:
    def test_missing_file_returns_false(self, tmp_path: Path) -> None:
        assert not is_cached("fred_md", "nonexistent.csv", cache_dir=tmp_path)

    def test_existing_file_within_age_returns_true(self, tmp_path: Path) -> None:
        p = get_cached_path("fred_md", "test.csv", tmp_path)
        p.write_text("data")
        assert is_cached("fred_md", "test.csv", cache_dir=tmp_path, max_age_days=30)

    def test_old_file_returns_false(self, tmp_path: Path) -> None:
        p = get_cached_path("fred_md", "old.csv", tmp_path)
        p.write_text("data")
        # Simulate file older than 0 days
        assert not is_cached("fred_md", "old.csv", cache_dir=tmp_path, max_age_days=0)

    def test_none_max_age_never_expires(self, tmp_path: Path) -> None:
        p = get_cached_path("fred_qd", "vintage.csv", tmp_path)
        p.write_text("data")
        # Even with max_age_days=0, None means no expiry check
        assert is_cached("fred_qd", "vintage.csv", cache_dir=tmp_path, max_age_days=None)


class TestDownloadFile:
    def test_successful_download(self, tmp_path: Path) -> None:
        dest = tmp_path / "test.csv"
        mock_response = MagicMock()
        mock_response.headers = {"content-length": "4"}
        mock_response.iter_content.return_value = [b"data"]
        mock_response.raise_for_status.return_value = None

        with patch("macrocast.utils.cache.requests.get", return_value=mock_response):
            result = download_file("http://example.com/test.csv", dest)

        assert result == dest
        assert dest.read_bytes() == b"data"

    def test_http_error_propagates(self, tmp_path: Path) -> None:
        import requests as req

        dest = tmp_path / "fail.csv"
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req.HTTPError("404")

        with patch("macrocast.utils.cache.requests.get", return_value=mock_response):
            with pytest.raises(req.HTTPError):
                download_file("http://example.com/fail.csv", dest)


class TestFileDownloadDate:
    def test_returns_iso_date_string(self, tmp_path: Path) -> None:
        p = tmp_path / "test.csv"
        p.write_text("data")
        result = file_download_date(p)
        # Should be YYYY-MM-DD format
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}", result)

    def test_date_matches_file_mtime(self, tmp_path: Path) -> None:
        import datetime
        p = tmp_path / "test.csv"
        p.write_text("data")
        result = file_download_date(p)
        expected = datetime.date.fromtimestamp(p.stat().st_mtime).isoformat()
        assert result == expected


class TestClearCache:
    def test_clears_dataset_subdir(self, tmp_path: Path) -> None:
        p = get_cached_path("fred_md", "current.csv", tmp_path)
        p.write_text("data")
        clear_cache(dataset="fred_md", cache_dir=tmp_path)
        assert not (tmp_path / "fred_md").exists()

    def test_clears_all_cache(self, tmp_path: Path) -> None:
        p1 = get_cached_path("fred_md", "a.csv", tmp_path)
        p2 = get_cached_path("fred_qd", "b.csv", tmp_path)
        p1.write_text("a")
        p2.write_text("b")
        clear_cache(dataset=None, cache_dir=tmp_path)
        # Root recreated but subdirs gone
        assert tmp_path.exists()
        assert not (tmp_path / "fred_md").exists()
        assert not (tmp_path / "fred_qd").exists()
