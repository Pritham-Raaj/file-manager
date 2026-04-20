from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import config
from logger import get_logger
from llm_client import LLMClient
from text_extractor import TextExtractor

log = get_logger(__name__)


class Indexer:

    def __init__(self, llm: Optional[LLMClient] = None):
        self._llm       = llm or LLMClient()
        self._extractor = TextExtractor()
        self._col       = None          # ChromaDB collection, lazy
        self._manifest  = self._load_manifest()

    # -- ChromaDB lazy init ---------------------------------------------------

    def _collection(self):
        if self._col is None:
            import chromadb
            config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
            client    = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
            self._col = client.get_or_create_collection(
                name=config.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            log.debug("ChromaDB ready | docs=%d", self._col.count())
        return self._col

    # -- Manifest: tracks file hashes to skip unchanged files -----------------

    def _load_manifest(self) -> dict:
        if config.INDEX_MANIFEST.exists():
            return json.loads(config.INDEX_MANIFEST.read_text())
        return {}

    def _save_manifest(self) -> None:
        config.INDEX_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        config.INDEX_MANIFEST.write_text(json.dumps(self._manifest, indent=2))

    # SHA-256 over first 4 KB only -- fast change detection without full-file hash
    def _signature(self, path: Path) -> tuple[float, str]:
        digest = hashlib.sha256(path.read_bytes()[:4096]).hexdigest()
        return path.stat().st_mtime, digest

    def _has_changed(self, path: Path) -> bool:
        key = str(path)
        if key not in self._manifest:
            return True
        mtime, digest = self._signature(path)
        c = self._manifest[key]
        return c["mtime"] != mtime or c["hash"] != digest

    # -- Public API -----------------------------------------------------------

    def index_file(self, path: Path) -> int:
        """Index a single file. Returns chunks added, or 0 if skipped/unsupported."""
        path = path.resolve()
        if not path.is_file():
            log.debug("Not a file or does not exist, skipping: %s", path)
            return 0
        if not self._extractor.supported(path):
            log.debug("Unsupported, skipping: %s", path.name)
            return 0
        if not self._has_changed(path):
            log.debug("Unchanged, skipping: %s", path.name)
            return 0

        log.info("Indexing: %s", path.name)
        self.remove_file(path)      # clear stale chunks first

        col, count = self._collection(), 0
        for page_num, text in self._extractor.extract_pages(path):
            for chunk_text, chunk_idx, offset in self._chunk(text):
                col.add(
                    ids=[self._chunk_id(path, page_num, chunk_idx)],
                    documents=[chunk_text],
                    embeddings=[self._llm.embed(chunk_text)],
                    metadatas=[{
                        "file_path": str(path),
                        "file_name": path.name,
                        "page_num":  page_num,
                        "chunk_idx": chunk_idx,
                        "offset":    offset,
                        "file_type": path.suffix.lower(),
                    }],
                )
                count += 1

        if count == 0:
            log.warning("No chunks produced for '%s' — skipping manifest update.", path.name)
            return 0

        mtime, digest = self._signature(path)
        self._manifest[str(path)] = {
            "mtime": mtime, "hash": digest,
            "indexed_at": datetime.now().isoformat(),
        }
        self._save_manifest()
        log.info("Indexed %d chunks | %s", count, path.name)
        return count

    def index_directory(self, directory: Path) -> dict[str, int]:
        """Index all supported files in a directory tree. Returns {name: chunk_count}."""
        results = {}
        for path in directory.rglob("*"):
            if path.is_file() and not path.name.startswith("."):
                results[path.name] = self.index_file(path)
        return results

    def search(self, query: str, n_results: int = 5,
               file_filter: Optional[str] = None) -> list[dict]:
        """Semantic search across indexed files. Filters by file path substring if given."""
        col   = self._collection()
        total = col.count()
        if total == 0:
            return []

        fetch = min(total, n_results * 3 if file_filter else n_results)
        rows  = col.query(
            query_embeddings=[self._llm.embed(query)],
            n_results=fetch,
            include=["documents", "metadatas", "distances"],
        )

        hits = [
            {"text": doc, "metadata": meta, "score": round(1 - dist, 4)}
            for doc, meta, dist in zip(
                rows["documents"][0], rows["metadatas"][0], rows["distances"][0]
            )
        ]

        if file_filter:
            hits = [h for h in hits if file_filter in h["metadata"].get("file_path", "")]

        log.debug("search | hits=%d | filter=%s", len(hits), file_filter)
        return hits[:n_results]

    def get_page(self, file_path: str, page_num: int) -> str:
        """Retrieve all indexed text for a specific page, reassembled in chunk order."""
        rows = self._collection().get(
            where={"$and": [
                {"file_path": {"$eq": file_path}},
                {"page_num":  {"$eq": page_num}},
            ]},
            include=["documents", "metadatas"],
        )
        if not rows["documents"]:
            return ""
        ordered = sorted(zip(rows["metadatas"], rows["documents"]),
                         key=lambda x: x[0]["chunk_idx"])
        return "\n".join(doc for _, doc in ordered)

    def remove_file(self, path: Path) -> None:
        existing = self._collection().get(where={"file_path": {"$eq": str(path)}})
        if existing["ids"]:
            self._collection().delete(ids=existing["ids"])
            log.debug("Removed %d chunks | %s", len(existing["ids"]), path.name)

    def remove_path(self, path: str) -> int:
        """Remove all indexed chunks for a file or every file under a directory.
        Also cleans the manifest. Returns the number of chunks deleted."""
        import os
        target = str(Path(path).resolve())
        col    = self._collection()
        rows   = col.get(include=["metadatas"])

        ids_to_delete  = []
        files_to_purge = set()
        for doc_id, meta in zip(rows["ids"], rows["metadatas"]):
            fp = meta.get("file_path", "")
            # match exact file OR any file whose path starts with the directory
            if fp == target or fp.startswith(target + os.sep) or fp.startswith(target + "/"):
                ids_to_delete.append(doc_id)
                files_to_purge.add(fp)

        if ids_to_delete:
            col.delete(ids=ids_to_delete)

        for fp in files_to_purge:
            self._manifest.pop(fp, None)
        if files_to_purge:
            self._save_manifest()

        log.info("remove_path | target=%s | chunks=%d | files=%d",
                 target, len(ids_to_delete), len(files_to_purge))
        return len(ids_to_delete)

    def list_sources(self) -> list[dict]:
        """Return indexed content grouped by parent directory with stats."""
        from collections import defaultdict
        col = self._collection()
        if col.count() == 0:
            return []

        rows  = col.get(include=["metadatas"])
        dirs  = defaultdict(lambda: {"files": set(), "chunks": 0})
        for meta in rows["metadatas"]:
            fp     = meta.get("file_path", "unknown")
            parent = str(Path(fp).parent)
            dirs[parent]["files"].add(fp)
            dirs[parent]["chunks"] += 1

        result = []
        for directory, stats in sorted(dirs.items()):
            dates = [
                self._manifest[fp]["indexed_at"]
                for fp in stats["files"]
                if fp in self._manifest and "indexed_at" in self._manifest[fp]
            ]
            result.append({
                "directory":    directory,
                "file_count":   len(stats["files"]),
                "chunk_count":  stats["chunks"],
                "first_indexed": min(dates)[:19] if dates else "unknown",
                "last_updated":  max(dates)[:19] if dates else "unknown",
            })
        return result

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _chunk_id(path: Path, page_num: int, chunk_idx: int) -> str:
        file_hash = hashlib.md5(str(path).encode()).hexdigest()[:8]
        return "{}:p{}:c{}".format(file_hash, page_num, chunk_idx)

    @staticmethod
    def _chunk(text: str):
        """Yield (chunk_text, chunk_index, char_offset) -- generator, never loads all at once."""
        step = config.CHUNK_SIZE - config.CHUNK_OVERLAP
        if step <= 0:
            step = config.CHUNK_SIZE    # guard: bad config → no overlap rather than infinite loop
        start, idx = 0, 0
        while start < len(text):
            yield text[start : start + config.CHUNK_SIZE], idx, start
            start += step
            idx   += 1
