#File manager - Sandbox and other functions of the manager like listing, searching, reading, moving, copying, deleting files and directories. 
#Most importantly, implements the security feature


import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import chardet

import config


class FileManager:

    def __init__(self, root_dir: Optional[Path] = None):
        self.root = (root_dir or config.ROOT_DIR).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    #Security (For now, this method ensures that the file reading and operation are confined in the specified sandbox directory. If not, it raises a ValueError.)

    def _safe(self, path_str: str) -> Path:
        path = Path(path_str)
        if not path.is_absolute():
            path = self.root / path
        resolved = path.resolve()

        if not resolved.is_relative_to(self.root):
            raise ValueError(
                "Access denied: path '{}' resolves outside the sandbox".format(path_str)
            )
        return resolved

    #Listing & searching 

    def list_files(
        self,
        directory: str = "",
        extensions: Optional[set[str]] = None,
    ) -> list[dict]:
        target = self._safe(directory) if directory else self.root

        if not target.is_dir():
            raise ValueError("Not a directory: {}".format(target.name))

        entries: list[dict] = []
        for item in target.iterdir():
            if item.name.startswith("."):
                continue  # skip hidden files/dirs

            if item.is_file() and extensions and item.suffix.lower() not in extensions:
                continue

            entries.append(self._build_entry(item))
        
        entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
        return entries

    def search_files(
        self,
        pattern: str,
        directory: str = "",
        max_results: int = 100,
    ) -> list[dict]:
        target = self._safe(directory) if directory else self.root
        results: list[dict] = []

        for match in target.rglob(pattern):
            if match.name.startswith("."):
                continue
            results.append(self._build_entry(match))
            if len(results) >= max_results:
                break

        results.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
        return results

    #File Reading 

    def read_file(self, path: str) -> str:
        resolved = self._safe(path)

        if not resolved.is_file():
            raise ValueError("Not a file: {}".format(resolved.name))

        raw = resolved.read_bytes()

        # Detect encoding from first 32 KB
        sample = raw[:32_768]
        detection = chardet.detect(sample)
        encoding = detection.get("encoding") or "utf-8"

        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            return raw.decode("utf-8", errors="replace")

    #File operations
    def move_file(self, src: str, dst: str) -> str:
        src_path = self._safe(src)
        dst_path = self._safe(dst)

        if not src_path.exists():
            raise ValueError("Source not found: {}".format(src))

        shutil.move(str(src_path), str(dst_path))
        return "Moved '{}' -> '{}'".format(src, dst)

    def copy_file(self, src: str, dst: str) -> str:
        src_path = self._safe(src)
        dst_path = self._safe(dst)

        if not src_path.exists():
            raise ValueError("Source not found: {}".format(src))

        if src_path.is_dir():
            shutil.copytree(str(src_path), str(dst_path))
        else:
            shutil.copy2(str(src_path), str(dst_path))
        return "Copied '{}' -> '{}'".format(src, dst)

    def delete_file(self, path: str) -> str:
        resolved = self._safe(path)

        if not resolved.exists():
            raise ValueError("Not found: {}".format(path))

        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink()
        return "Deleted '{}'".format(path)

    def make_directory(self, path: str) -> str:
        resolved = self._safe(path)
        resolved.mkdir(parents=True, exist_ok=True)
        return "Created directory '{}'".format(path)

    #Metadata 

    def get_file_info(self, path: str) -> dict:
        """Return detailed metadata for a single file or directory."""
        resolved = self._safe(path)

        if not resolved.exists():
            raise ValueError("Not found: {}".format(path))

        stat = resolved.stat()
        return {
            "name": resolved.name,
            "path": str(resolved.relative_to(self.root)),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "extension": resolved.suffix.lower(),
            "is_dir": resolved.is_dir(),
        }

    #Auxiliary

    def _build_entry(self, item: Path) -> dict:
        stat = item.stat()
        return {
            "name": item.name,
            "path": str(item.relative_to(self.root)),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extension": item.suffix.lower() if item.is_file() else "",
            "is_dir": item.is_dir(),
        }