"""
Tests for Indexer: accuracy, incremental change detection, search, edge cases.

Coverage matrix
---------------
Accuracy       - indexed content is retrievable via semantic search
Change detect  - unchanged file skipped (0 chunks returned), modified re-indexed
Search         - top-k results contain expected text; file_filter narrows results
Edge cases     - unsupported extension, empty file, directory traversal
Removal        - removed file no longer appears in search results
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from conftest import SHORT_TEXT, CSV_TEXT


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

class TestIndexAccuracy:

    def test_index_txt_returns_chunks(self, indexer, fixture_dir):
        """Indexing a non-empty .txt file should produce at least one chunk."""
        path = fixture_dir / "short.txt"
        count = indexer.index_file(path)
        assert count >= 1, "Expected at least 1 chunk for a non-empty text file"

    def test_indexed_content_is_searchable(self, indexer, fixture_dir):
        """Content from an indexed file should appear in search results."""
        path = fixture_dir / "short.txt"
        indexer.index_file(path)

        hits = indexer.search("hypertension blood pressure", n_results=3)
        assert len(hits) >= 1, "Search returned no hits after indexing"
        combined = " ".join(h["text"] for h in hits).lower()
        assert "hypertension" in combined or "blood pressure" in combined

    def test_index_csv_produces_chunks(self, indexer, fixture_dir):
        """CSV files are text-based and should be indexed."""
        count = indexer.index_file(fixture_dir / "data.csv")
        assert count >= 1

    def test_index_python_code(self, indexer, fixture_dir):
        """Python source files should be indexed as code."""
        count = indexer.index_file(fixture_dir / "code.py")
        assert count >= 1

    def test_index_markdown(self, indexer, fixture_dir):
        count = indexer.index_file(fixture_dir / "notes.md")
        assert count >= 1

    def test_search_score_between_0_and_1(self, indexer, fixture_dir):
        """Cosine similarity scores should be in [0, 1]."""
        indexer.index_file(fixture_dir / "short.txt")
        hits = indexer.search("blood pressure", n_results=3)
        for h in hits:
            assert 0.0 <= h["score"] <= 1.0, f"Score out of range: {h['score']}"

    def test_search_metadata_fields_present(self, indexer, fixture_dir):
        """Each hit must carry file_path, file_name, page_num, chunk_idx."""
        indexer.index_file(fixture_dir / "short.txt")
        hits = indexer.search("patient diagnosis", n_results=1)
        assert hits, "No hits returned"
        meta = hits[0]["metadata"]
        for field in ("file_path", "file_name", "page_num", "chunk_idx"):
            assert field in meta, f"Missing metadata field: {field}"


# ---------------------------------------------------------------------------
# Incremental change detection
# ---------------------------------------------------------------------------

class TestChangeDetection:

    def test_unchanged_file_is_skipped(self, indexer, fixture_dir):
        """Second index of an unchanged file must return 0 (skip)."""
        path = fixture_dir / "short.txt"
        first  = indexer.index_file(path)
        second = indexer.index_file(path)
        assert first >= 1,  "First index should produce chunks"
        assert second == 0, "Unchanged file should be skipped on re-index"

    def test_modified_file_is_reindexed(self, indexer, fixture_dir):
        """After content changes the file must be re-indexed (chunk count > 0)."""
        path = fixture_dir / "short.txt"
        indexer.index_file(path)

        # Modify content and touch mtime
        path.write_text(SHORT_TEXT + "\nNew finding: elevated LDL cholesterol.", encoding="utf-8")
        time.sleep(0.01)   # ensure mtime differs

        count = indexer.index_file(path)
        assert count >= 1, "Modified file should be re-indexed"

    def test_manifest_persisted_after_index(self, indexer, fixture_dir):
        """Manifest file should exist and contain the indexed path."""
        import json, config
        path = fixture_dir / "short.txt"
        indexer.index_file(path)
        assert config.INDEX_MANIFEST.exists(), "Manifest not created"
        manifest = json.loads(config.INDEX_MANIFEST.read_text())
        assert str(path) in manifest


# ---------------------------------------------------------------------------
# Search filtering
# ---------------------------------------------------------------------------

class TestSearch:

    def test_file_filter_restricts_results(self, indexer, fixture_dir):
        """file_filter should exclude results from non-matching paths."""
        indexer.index_file(fixture_dir / "short.txt")
        indexer.index_file(fixture_dir / "data.csv")

        hits = indexer.search("patient", n_results=5, file_filter="short.txt")
        for h in hits:
            assert "short.txt" in h["metadata"]["file_path"]

    def test_search_empty_index_returns_empty(self, indexer):
        """Searching an empty index should not raise and must return []."""
        hits = indexer.search("anything")
        assert hits == []

    def test_search_no_match_file_filter(self, indexer, fixture_dir):
        """A file_filter that matches nothing should return []."""
        indexer.index_file(fixture_dir / "short.txt")
        hits = indexer.search("hypertension", n_results=5, file_filter="nonexistent.txt")
        assert hits == []

    def test_n_results_respected(self, indexer, fixture_dir):
        """Result count must not exceed n_results."""
        for f in ("short.txt", "code.py", "notes.md", "data.csv"):
            indexer.index_file(fixture_dir / f)
        hits = indexer.search("health", n_results=2)
        assert len(hits) <= 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_unsupported_extension_returns_zero(self, indexer, fixture_dir):
        """Files with unsupported extensions must return 0 chunks silently."""
        count = indexer.index_file(fixture_dir / "binary.xyz")
        assert count == 0

    def test_empty_file_returns_zero(self, indexer, tmp_path):
        """An empty .txt file should return 0 chunks without raising."""
        empty = tmp_path / "empty.txt"
        empty.write_text("", encoding="utf-8")
        count = indexer.index_file(empty)
        assert count == 0

    def test_missing_file_returns_zero(self, indexer, tmp_path):
        """A path that does not exist should return 0 without raising."""
        count = indexer.index_file(tmp_path / "ghost.txt")
        assert count == 0

    def test_index_directory(self, indexer, fixture_dir):
        """index_directory should return a dict covering all supported files."""
        results = indexer.index_directory(fixture_dir)
        # At minimum: short.txt, long.txt, code.py, data.csv, notes.md
        assert len(results) >= 5, f"Expected >= 5 files indexed, got {results}"
        # binary.xyz and hidden files must be excluded
        for name in results:
            assert not name.endswith(".xyz")


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

class TestRemoval:

    def test_remove_clears_search_results(self, indexer, fixture_dir):
        """After removal, search should no longer return chunks from that file."""
        path = fixture_dir / "short.txt"
        indexer.index_file(path)

        # Verify it's findable
        assert indexer.search("hypertension") != []

        indexer.remove_file(path)

        hits = indexer.search("hypertension", n_results=5)
        for h in hits:
            assert "short.txt" not in h["metadata"].get("file_path", "")
