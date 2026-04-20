from __future__ import annotations

from pathlib import Path
from typing import Generator, Optional

from file_manager import FileManager
from indexer import Indexer
from llm_client import LLMClient
from logger import get_logger
from summarizer import Summarizer
from sync.cloud_sync import CloudSync, GDriveConnector, S3Connector
from sync.email_sync import EmailSync
from writer import Writer

log = get_logger(__name__)

_SYSTEM = (
    "You are a local file assistant. Answer using only the context provided. "
    "Be specific and concise. Say 'Not found in documents' if the answer is absent."
)

_QA_PROMPT = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


class Orchestrator:
    """Routes user queries to the correct tool. All state and processing stays local."""

    def __init__(self, root: Optional[Path] = None):
        self._llm        = LLMClient()
        self._fm         = FileManager(root)
        self._indexer    = Indexer(self._llm)
        self._summarizer = Summarizer(self._llm)
        self._writer     = Writer(root)
        self._cloud_sync = CloudSync(self._fm.root)

    # ── Q&A — grounded in indexed file content via RAG ────────────────────────

    def ask(self, question: str, file_filter: Optional[str] = None) -> Generator[str, None, None]:
        hits = self._indexer.search(question, n_results=5, file_filter=file_filter)
        if not hits:
            yield "No relevant content found in indexed files."
            return
        context = "\n\n---\n\n".join(h["text"] for h in hits)
        log.info("ask | %d chunks | question_len=%d", len(hits), len(question))
        yield from self._llm.generate(_QA_PROMPT.format(context=context, question=question), system=_SYSTEM)

    # ── Summarization ─────────────────────────────────────────────────────────

    def summarize_file(self, path: str) -> Generator[str, None, None]:
        text = self.read_file(path)   # routes PDFs/DOCX/PPTX/XLSX through TextExtractor
        log.info("summarize_file | %s | len=%d", path, len(text))
        yield from self._summarizer.summarize(text)

    def summarize_page(self, path: str, page_num: int) -> Generator[str, None, None]:
        text = self._indexer.get_page(path, page_num)
        if not text:
            yield "Page {} not found in index for '{}'.".format(page_num, path)
            return
        yield from self._summarizer.summarize(text)

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index(self, path: str) -> str:
        target = Path(path)
        if target.is_dir():
            results = self._indexer.index_directory(target)
            return "Indexed {} file(s), {} total chunk(s).".format(len(results), sum(results.values()))
        count = self._indexer.index_file(target)
        return "Indexed '{}' — {} chunk(s).".format(target.name, count)

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, file_filter: Optional[str] = None) -> list[dict]:
        return self._indexer.search(query, file_filter=file_filter)

    # ── File operations ───────────────────────────────────────────────────────

    def list_files(self, directory: str = "") -> list[dict]:
        return self._fm.list_files(directory)

    def read_file(self, path: str) -> str:
        # _safe() validates the path is inside the sandbox and returns a resolved Path
        target = self._fm._safe(path)
        # Binary document formats need text extraction, not raw byte decoding
        if target.suffix.lower() in {".pdf", ".docx", ".pptx", ".xlsx"}:
            return self._indexer._extractor.extract(target)
        return self._fm.read_file(str(target))

    def create_file(self, path: str, content: str = "") -> str:
        return self._writer.create(path, content)

    def append_to_file(self, path: str, content: str) -> str:
        return self._writer.append(path, content)

    def patch_file(self, path: str, old: str, new: str) -> str:
        count = self._writer.patch(path, old, new)
        return "{} replacement(s) in \'{}\'.".format(count, path)

    # -- Sync: cloud (pull-only) ------------------------------------------

    def register_s3(self, name: str, bucket: str, prefix: str = "") -> str:
        self._cloud_sync.register(name, S3Connector(bucket, prefix))
        return "S3 connector \'{}\' registered.".format(name)

    def register_gdrive(self, name: str, credentials_path: str) -> str:
        self._cloud_sync.register(name, GDriveConnector(credentials_path))
        return "Google Drive connector \'{}\' registered.".format(name)

    def sync_cloud(self, source=None) -> str:
        pulled = self._cloud_sync.pull_all(source)
        if not pulled:
            return "No files pulled."
        indexed = sum(self._indexer.index_file(p) for p in pulled)
        log.info("sync_cloud | pulled=%d indexed_chunks=%d", len(pulled), indexed)
        return "Pulled {} file(s), {} chunk(s) indexed.".format(len(pulled), indexed)

    # -- Sync: email (attachments only, read-only) -------------------------

    def sync_email(self, host: str, user: str, password: str,
                   folder: str = "INBOX", subject_filter=None,
                   max_emails: int = 50) -> str:
        syncer = EmailSync(host, user, password)
        try:
            pulled = syncer.pull_attachments(
                self._fm.root, folder=folder,
                subject_filter=subject_filter or None,
                max_emails=max_emails,
            )
            if not pulled:
                return "No attachments found."
            indexed = sum(self._indexer.index_file(p) for p in pulled)
            log.info("sync_email | pulled=%d indexed_chunks=%d", len(pulled), indexed)
            return "Pulled {} attachment(s), {} chunk(s) indexed.".format(len(pulled), indexed)
        finally:
            syncer.disconnect()

    # -- Index management -------------------------------------------------

    def list_sources(self) -> str:
        """Return a formatted summary of all indexed locations with stats."""
        sources = self._indexer.list_sources()
        if not sources:
            return "No content indexed yet."
        separator = "\n" + "-" * 60 + "\n"
        blocks = []
        for i, s in enumerate(sources, 1):
            blocks.append(
                "Location {i}\n"
                "  Directory    : {directory}\n"
                "  Files indexed: {file_count}\n"
                "  Total chunks : {chunk_count}\n"
                "  First indexed: {first_indexed}\n"
                "  Last updated : {last_updated}".format(i=i, **s)
            )
        header = "  {} location(s) in index\n".format(len(sources)) + "=" * 60
        return header + separator + separator.join(blocks)

    def remove_path(self, path: str) -> str:
        """Remove all indexed content for a file or directory."""
        count = self._indexer.remove_path(path.strip())
        if count == 0:
            return "No indexed content found for '{}'.".format(path)
        return "Removed {} chunk(s) for '{}'.".format(count, path)

    def clear_index(self) -> str:
        """Wipe the entire index and manifest. Actual files are never touched."""
        # Short-circuit if there's nothing to clear — avoids creating ChromaDB
        # files just to immediately find an empty collection.
        if not self._indexer._manifest and self._indexer._col is None:
            return "Index is already empty — nothing to clear."
        col     = self._indexer._collection()
        all_ids = col.get()["ids"]
        if all_ids:
            col.delete(ids=all_ids)
        removed = len(all_ids)
        self._indexer._manifest.clear()
        self._indexer._save_manifest()
        log.info("Index cleared | chunks_removed=%d", removed)
        return "Index cleared — {} chunk(s) removed.".format(removed)

    # -- Model management -------------------------------------------------

    def list_models(self) -> list[str]:
        return self._llm.list_models()

    def switch_model(self, model: str) -> str:
        self._llm.switch_model(model)
        return "Model switched to \'{}\'.".format(model)
