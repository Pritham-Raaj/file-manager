"""
Tests for Writer: create, append, overwrite, and patch operations.

All operations must stay within ROOT_DIR (sandbox). Patch correctness is
verified by exact string matching — critical for AI-assisted edit workflows.
"""

from __future__ import annotations

import pytest
import config


class TestCreateFile:

    def test_create_new_file(self, writer):
        writer.create("new_doc.txt", "Hello world")
        path = config.ROOT_DIR / "new_doc.txt"
        assert path.exists()
        assert path.read_text(encoding="utf-8") == "Hello world"

    def test_create_in_subdirectory(self, writer):
        """Writer should create parent directories as needed."""
        writer.create("sub/dir/file.txt", "nested content")
        path = config.ROOT_DIR / "sub" / "dir" / "file.txt"
        assert path.exists()
        assert "nested" in path.read_text(encoding="utf-8")

    def test_create_empty_file(self, writer):
        writer.create("empty.txt", "")
        assert (config.ROOT_DIR / "empty.txt").exists()

    def test_create_outside_sandbox_blocked(self, writer):
        with pytest.raises(Exception):
            writer.create("../../outside.txt", "bad")


class TestAppendFile:

    def test_append_to_existing_file(self, writer):
        writer.create("log.txt", "line1\n")
        writer.append("log.txt", "line2\n")
        content = (config.ROOT_DIR / "log.txt").read_text(encoding="utf-8")
        assert "line1" in content
        assert "line2" in content

    def test_append_creates_file_if_missing(self, writer):
        """Append to a non-existent file should create it."""
        writer.append("auto_created.txt", "first line")
        assert (config.ROOT_DIR / "auto_created.txt").exists()

    def test_append_outside_sandbox_blocked(self, writer):
        with pytest.raises(Exception):
            writer.append("../../evil.txt", "bad")


class TestPatchFile:

    def test_patch_replaces_text(self, writer):
        writer.create("report.txt", "The diagnosis is Normal. The prognosis is good.")
        count = writer.patch("report.txt", "Normal", "Hypertension")
        content = (config.ROOT_DIR / "report.txt").read_text(encoding="utf-8")
        assert "Hypertension" in content
        assert "Normal" not in content
        assert count == 1

    def test_patch_multiple_occurrences(self, writer):
        writer.create("multi.txt", "cat cat cat")
        count = writer.patch("multi.txt", "cat", "dog")
        content = (config.ROOT_DIR / "multi.txt").read_text(encoding="utf-8")
        assert content == "dog dog dog"
        assert count == 3

    def test_patch_no_match_returns_zero(self, writer):
        writer.create("nomatch.txt", "hello world")
        count = writer.patch("nomatch.txt", "xyz", "abc")
        assert count == 0

    def test_patch_preserves_surrounding_content(self, writer):
        writer.create("preserve.txt", "Header\nTarget line\nFooter")
        writer.patch("preserve.txt", "Target line", "Replacement line")
        content = (config.ROOT_DIR / "preserve.txt").read_text(encoding="utf-8")
        assert "Header" in content
        assert "Footer" in content
        assert "Replacement line" in content

    def test_patch_missing_file_raises(self, writer):
        with pytest.raises(Exception):
            writer.patch("does_not_exist.txt", "old", "new")

    def test_patch_outside_sandbox_blocked(self, writer):
        with pytest.raises(Exception):
            writer.patch("../../etc/hosts", "localhost", "evil")
