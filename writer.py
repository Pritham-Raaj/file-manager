from __future__ import annotations

from pathlib import Path
from typing import Optional

from file_manager import FileManager
from logger import get_logger

log = get_logger(__name__)


class Writer:

    def __init__(self, root: Optional[Path] = None):
        self._fm = FileManager(root)

    def create(self, path: str, content: str = "") -> str:
        resolved = self._fm._safe(path)
        if resolved.exists():
            raise FileExistsError("Already exists: {}".format(path))
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        log.info("Created: %s", path)
        return "Created '{}'".format(path)

    def append(self, path: str, content: str) -> str:
        resolved = self._fm._safe(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open("a", encoding="utf-8") as fh:
            fh.write(content)
        log.info("Appended %d chars to: %s", len(content), path)
        return "Appended to '{}'".format(path)

    def overwrite(self, path: str, content: str) -> str:
        resolved = self._fm._safe(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        log.info("Overwritten: %s", path)
        return "Overwritten '{}'".format(path)

    # Replace every occurrence of old_text with new_text; returns replacement count
    def patch(self, path: str, old_text: str, new_text: str) -> int:
        resolved = self._fm._safe(path)
        if not resolved.is_file():
            raise ValueError("Not a file: {}".format(path))
        original = resolved.read_text(encoding="utf-8")
        if old_text not in original:
            log.warning("patch | target not found in %s", path)
            return 0
        count = original.count(old_text)
        resolved.write_text(original.replace(old_text, new_text), encoding="utf-8")
        log.info("patch | %d replacement(s) in %s", count, path)
        return count
