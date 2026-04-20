"""
Tests for Orchestrator: end-to-end routing of all user-facing operations.

The orchestrator is the top-level integration point. Tests here verify that
each public method delegates correctly, handles missing/empty state gracefully,
and surfaces meaningful messages to the UI layer.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import config

from conftest import SHORT_TEXT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_orchestrator(tmp_root, mock_ollama_client):
    """Build an Orchestrator with mocked Ollama and ephemeral ChromaDB."""
    import chromadb
    from orchestrator import Orchestrator

    orch = Orchestrator(root=tmp_root)

    # Replace the internal LLM client's underlying Ollama client
    orch._llm._client = mock_ollama_client
    orch._indexer._llm = orch._llm

    # Swap ChromaDB to ephemeral
    ephemeral = chromadb.EphemeralClient()
    col = ephemeral.get_or_create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    orch._indexer._col = col

    return orch


@pytest.fixture()
def orch(tmp_path, mock_ollama_client, isolate_config):
    return make_orchestrator(config.ROOT_DIR, mock_ollama_client)


# ---------------------------------------------------------------------------
# Q&A (ask)
# ---------------------------------------------------------------------------

class TestAsk:

    def test_ask_with_no_indexed_content(self, orch):
        """ask() on an empty index must yield a 'not found' message, not raise."""
        tokens = list(orch.ask("What is the patient's diagnosis?"))
        response = "".join(tokens)
        assert len(response) > 0
        assert "not found" in response.lower() or "no relevant" in response.lower()

    def test_ask_after_indexing(self, orch, tmp_path):
        """ask() should call the LLM after retrieving context from the index."""
        doc = config.ROOT_DIR / "patient.txt"
        doc.write_text(SHORT_TEXT, encoding="utf-8")
        orch.index(str(doc))

        tokens = list(orch.ask("What is the blood pressure?"))
        assert len(tokens) > 0   # LLM responded (mocked)

    def test_ask_with_file_filter(self, orch):
        """file_filter kwarg should be passed through to the indexer search."""
        # Just verify it doesn't raise when filter matches nothing
        tokens = list(orch.ask("anything", file_filter="nonexistent.txt"))
        assert isinstance(tokens, list)


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

class TestSummarize:

    def test_summarize_file(self, orch):
        doc = config.ROOT_DIR / "note.txt"
        doc.write_text(SHORT_TEXT, encoding="utf-8")
        tokens = list(orch.summarize_file(str(doc)))
        assert len(tokens) > 0

    def test_summarize_missing_file_raises(self, orch):
        with pytest.raises(Exception):
            list(orch.summarize_file("ghost.txt"))

    def test_summarize_page_not_indexed(self, orch):
        """summarize_page on an unindexed file should yield an error message."""
        tokens = list(orch.summarize_page("missing.txt", 1))
        response = "".join(tokens)
        assert "not found" in response.lower()


# ---------------------------------------------------------------------------
# Index routing
# ---------------------------------------------------------------------------

class TestIndex:

    def test_index_single_file(self, orch):
        doc = config.ROOT_DIR / "data.txt"
        doc.write_text(SHORT_TEXT, encoding="utf-8")
        result = orch.index(str(doc))
        assert "chunk" in result.lower()

    def test_index_directory(self, orch):
        d = config.ROOT_DIR / "docs"
        d.mkdir()
        (d / "a.txt").write_text("Alpha content", encoding="utf-8")
        (d / "b.txt").write_text("Beta content",  encoding="utf-8")
        result = orch.index(str(d))
        assert "file" in result.lower()

    def test_index_nonexistent_path_raises(self, orch):
        with pytest.raises(Exception):
            orch.index("/nonexistent/path/to/nothing.txt")


# ---------------------------------------------------------------------------
# File operations
# ---------------------------------------------------------------------------

class TestFileOps:

    def test_list_files(self, orch):
        (config.ROOT_DIR / "readme.txt").write_text("hello", encoding="utf-8")
        entries = orch.list_files("")
        assert any(e["name"] == "readme.txt" for e in entries)

    def test_read_file(self, orch):
        (config.ROOT_DIR / "read_me.txt").write_text("read this", encoding="utf-8")
        content = orch.read_file("read_me.txt")
        assert "read this" in content

    def test_create_file(self, orch):
        result = orch.create_file("created.txt", "new content")
        assert (config.ROOT_DIR / "created.txt").exists()
        assert isinstance(result, str)

    def test_patch_file(self, orch):
        orch.create_file("patch_me.txt", "old value here")
        result = orch.patch_file("patch_me.txt", "old value", "new value")
        assert "1" in result   # "1 replacement(s)"
        content = (config.ROOT_DIR / "patch_me.txt").read_text(encoding="utf-8")
        assert "new value" in content


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

class TestIndexManagement:

    def test_clear_empty_index_does_not_raise(self, orch):
        """clear_index() on a never-used index should return a sensible message."""
        result = orch.clear_index()
        assert isinstance(result, str)
        # Either "already empty" or "cleared — 0 chunk(s)"
        assert "empty" in result.lower() or "cleared" in result.lower()

    def test_clear_after_indexing_removes_all_chunks(self, orch):
        """After clear_index(), search should return nothing."""
        doc = config.ROOT_DIR / "patient.txt"
        doc.write_text(SHORT_TEXT, encoding="utf-8")
        orch.index(str(doc))

        result = orch.clear_index()
        assert "cleared" in result.lower()

        hits = list(orch.ask("hypertension"))
        response = "".join(hits)
        assert "not found" in response.lower() or "no relevant" in response.lower()

    def test_clear_resets_manifest(self, orch):
        """clear_index() must also wipe the manifest so files are re-indexed next time."""
        doc = config.ROOT_DIR / "patient.txt"
        doc.write_text(SHORT_TEXT, encoding="utf-8")
        orch.index(str(doc))
        assert len(orch._indexer._manifest) > 0

        orch.clear_index()
        assert len(orch._indexer._manifest) == 0


# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

class TestModelManagement:

    def test_list_models(self, orch):
        models = orch.list_models()
        assert isinstance(models, list)
        assert "phi3:mini" in models

    def test_switch_model(self, orch):
        result = orch.switch_model("llama3:8b")
        assert "llama3:8b" in result
        assert orch._llm.model == "llama3:8b"
