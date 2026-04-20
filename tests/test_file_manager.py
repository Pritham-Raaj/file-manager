"""
Tests for FileManager: directory listing, file reading, and sandbox enforcement.

The sandbox is the most critical safety property of the system — every path
operation must be confined to ROOT_DIR. This module stress-tests that guarantee.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import config

from conftest import SHORT_TEXT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def populate_root(root: Path):
    """Create a small directory tree inside root for listing/reading tests."""
    (root / "reports").mkdir()
    (root / "reports" / "q1.txt").write_text("Q1 revenue: 1.2M", encoding="utf-8")
    (root / "reports" / "q2.txt").write_text("Q2 revenue: 1.5M", encoding="utf-8")
    (root / "notes.txt").write_text(SHORT_TEXT, encoding="utf-8")
    (root / ".hidden").write_text("ignore me", encoding="utf-8")


# ---------------------------------------------------------------------------
# Directory listing
# ---------------------------------------------------------------------------

class TestListFiles:

    def test_list_root_returns_entries(self, file_manager):
        populate_root(config.ROOT_DIR)
        entries = file_manager.list_files("")
        names = [e["name"] for e in entries]
        assert "reports" in names
        assert "notes.txt" in names

    def test_list_subdirectory(self, file_manager):
        populate_root(config.ROOT_DIR)
        entries = file_manager.list_files("reports")
        names = [e["name"] for e in entries]
        assert "q1.txt" in names
        assert "q2.txt" in names

    def test_entry_has_required_fields(self, file_manager):
        populate_root(config.ROOT_DIR)
        entries = file_manager.list_files("")
        for e in entries:
            assert "name"   in e
            assert "size"   in e
            assert "is_dir" in e

    def test_empty_directory_returns_empty_list(self, file_manager):
        (config.ROOT_DIR / "empty_dir").mkdir()
        entries = file_manager.list_files("empty_dir")
        assert entries == []

    def test_nonexistent_directory_raises(self, file_manager):
        with pytest.raises(Exception):
            file_manager.list_files("does_not_exist")


# ---------------------------------------------------------------------------
# File reading
# ---------------------------------------------------------------------------

class TestReadFile:

    def test_read_existing_file(self, file_manager):
        populate_root(config.ROOT_DIR)
        content = file_manager.read_file("notes.txt")
        assert "hypertension" in content.lower()

    def test_read_nested_file(self, file_manager):
        populate_root(config.ROOT_DIR)
        content = file_manager.read_file("reports/q1.txt")
        assert "Q1" in content

    def test_read_missing_file_raises(self, file_manager):
        with pytest.raises(Exception):
            file_manager.read_file("nonexistent.txt")

    def test_read_returns_string(self, file_manager):
        (config.ROOT_DIR / "test.txt").write_text("hello", encoding="utf-8")
        result = file_manager.read_file("test.txt")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Sandbox enforcement — critical safety tests
# ---------------------------------------------------------------------------

class TestSandboxEnforcement:

    def test_path_traversal_blocked(self, file_manager):
        """../../etc/passwd must never escape the sandbox."""
        with pytest.raises(Exception):
            file_manager.read_file("../../etc/passwd")

    def test_absolute_path_outside_root_blocked(self, file_manager):
        """Absolute paths outside ROOT_DIR must be rejected."""
        with pytest.raises(Exception):
            file_manager.read_file("/etc/hosts")

    def test_list_dir_traversal_blocked(self, file_manager):
        with pytest.raises(Exception):
            file_manager.list_files("../../")

    def test_symlink_escape_blocked(self, file_manager, tmp_path):
        """A symlink pointing outside ROOT_DIR must not be followed."""
        evil_target = tmp_path / "evil.txt"
        evil_target.write_text("secret", encoding="utf-8")

        link = config.ROOT_DIR / "escape_link.txt"
        try:
            link.symlink_to(evil_target)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")

        with pytest.raises(Exception):
            file_manager.read_file("escape_link.txt")
