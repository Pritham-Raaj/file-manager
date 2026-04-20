import os
import logging
from pathlib import Path

# -- Directories
ROOT_DIR = Path(
    os.environ.get("FILE_ASSISTANT_ROOT", os.path.expanduser("~/assistant-files"))
).resolve()

CHROMA_DIR = Path(
    os.environ.get("FILE_ASSISTANT_DB", os.path.expanduser("~/.file-assistant-db"))
).resolve()

# -- LLM
OLLAMA_HOST      = "http://localhost:11434"
GENERATION_MODEL = "phi3:mini"
EMBEDDING_MODEL  = "nomic-embed-text"

# -- Chunking
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# -- Supported extensions
SUPPORTED_EXTENSIONS = {
    "pdf":  [".pdf"],
    "docx": [".docx"],
    "pptx": [".pptx"],
    "xlsx": [".xlsx"],
    "text": [".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log"],
    "code": [
        ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".cs",
        ".go", ".rs", ".rb", ".php", ".html", ".css", ".sql", ".sh", ".bat",
    ],
}

ALL_EXTENSIONS: set[str] = set()
for _exts in SUPPORTED_EXTENSIONS.values():
    ALL_EXTENSIONS.update(_exts)

# -- Gradio
GRADIO_HOST = "127.0.0.1"
GRADIO_PORT = 7860

# -- ChromaDB
COLLECTION_NAME = "file_assistant"

# -- Logging
LOG_LEVEL = logging.DEBUG
LOG_FILE  = ROOT_DIR / "assistant.log"

# -- Indexing
INDEX_MANIFEST = CHROMA_DIR / "manifest.json"

# -- Resource limits
MAX_CHUNK_WORKERS = 2       # parallel extraction threads
STREAM_RESPONSES  = True
LINES_PER_PAGE    = 100     # synthetic page size for plain-text / code files
