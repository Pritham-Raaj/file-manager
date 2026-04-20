# Local AI File Assistant — User Guide

## Getting Started

### Prerequisites
Before launching the app, make sure the following are installed and running:

- **Ollama** — the local AI engine. Download from [ollama.com](https://ollama.com)
- **Required models** — pull these once from a terminal:
  ```
  ollama pull phi3:mini
  ollama pull nomic-embed-text
  ```

### Starting the App
```bash
cd file-manager
pip install -r requirements.txt   # first time only
python app.py
```

Then open your browser and go to: **http://127.0.0.1:7860**

To stop the app, press **Ctrl + C** in the terminal.

---

## The Interface

The app has eight tabs. Here's what each one does.

---

### Chat
Ask questions about your indexed files in plain English, or summarise any file.

**Asking a question:**
1. Optionally type a file path or name in "Limit to file" to restrict answers to one document
2. Type your question in the text box and press Enter
3. The assistant searches your indexed files and streams an answer back

**Summarising a file:**
- Type `summarize <full file path>` in the chat box and press Enter
- Example: `summarize C:\Users\prith\assistant-files\report.pdf`
- Works on any file type — long documents are handled automatically

**Tips:**
- Questions work best after you've indexed your files (see the Index tab)
- The assistant only answers from indexed content — it won't guess or hallucinate beyond what's in your files
- Use the file filter to get more precise answers when the same topic appears across multiple documents

---

### Search
Find specific content across all indexed files using semantic search (meaning-based, not keyword-based).

1. Type a search query — e.g. `elevated blood pressure treatment`
2. Optionally add a file filter to narrow results to a specific file or folder
3. Click **Search**

Results show a relevance score, the file name, page number, and a 200-character excerpt of the matching text.

---

### Index
Tell the assistant which files or folders to learn from.

1. Type the full path to a file or directory — e.g. `C:\Users\prith\Documents\reports`
2. Click **Index**

**What happens:**
- Each supported file is read, chunked, embedded, and stored locally in a vector database
- Files that haven't changed since the last index are automatically skipped (fast re-runs)
- Results show how many files were indexed and how many chunks were created

**Supported file types:**
PDF, Word (.docx), PowerPoint (.pptx), Excel (.xlsx), plain text, Markdown, CSV, JSON, XML, YAML, and most code files (.py, .js, .ts, .java, .go, .sql, and more)

**Re-indexing:**
You can re-run Index on the same path at any time. Only changed files will be reprocessed.

---

### Browse
Explore the contents of any folder on your machine.

1. Type a directory path, or leave blank to see the assistant's root folder
2. Click **List**

Each entry shows whether it's a file or folder, its name, and its size in bytes.

---

### Read
View the raw contents of any file.

1. Type the full path to the file
2. Click **Read**

Useful for quickly inspecting a file before indexing or summarising it.

---

### Write
Create new files or make targeted edits to existing ones.

**Create file:**
1. Enter the full destination path (parent directories are created automatically)
2. Type the content
3. Click **Create**

**Patch file (find and replace):**
1. Enter the file path
2. Paste the exact text you want to replace in "Text to replace"
3. Type the replacement text
4. Click **Patch**

The result shows how many replacements were made. All operations are sandboxed — paths outside the assistant's root folder are blocked.

---

### Sync
Pull files from external sources into your local file store, then automatically index them.

**S3 (Amazon cloud storage):**
1. Enter a connector name (any label you choose, e.g. `work-reports`)
2. Enter the S3 bucket name
3. Optionally enter a key prefix to limit which files are pulled (e.g. `2024/reports/`)
4. Click **Register** to save the connection, then **Pull** to download files

**Email (IMAP / Gmail / Outlook):**
1. Enter your IMAP server — e.g. `imap.gmail.com` for Gmail, `outlook.office365.com` for Outlook
2. Enter your email address and password (or app password if using 2FA)
3. Set the folder — default is `INBOX`
4. Optionally enter a subject keyword to filter which emails are processed
5. Click **Pull attachments**

Sync is **pull-only** — the app never sends, uploads, or modifies anything on the remote source.

> **Gmail users:** you'll need to generate an App Password at myaccount.google.com → Security → App Passwords

---

### Settings
Switch between locally available Ollama models.

1. Click **List available models** to see what's installed
2. Type a model name and click **Switch model** to change the active generation model

The embedding model (`nomic-embed-text`) is fixed and cannot be changed here — it's set in `config.py`.

---

## Configuration

All defaults are in `file-manager/config.py`:

| Setting | Default | What it controls |
|---|---|---|
| `ROOT_DIR` | `~/assistant-files` | Where files are stored and sandboxed to |
| `CHROMA_DIR` | `~/.file-assistant-db` | Where the vector index is stored |
| `GENERATION_MODEL` | `phi3:mini` | LLM used for Q&A and summarisation |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model used to embed text for search |
| `CHUNK_SIZE` | `500` | Characters per indexed chunk |
| `GRADIO_PORT` | `7860` | Port the web UI runs on |

You can override `ROOT_DIR` and `CHROMA_DIR` without editing the file by setting environment variables:
```bash
set FILE_ASSISTANT_ROOT=D:\MyDocuments
set FILE_ASSISTANT_DB=D:\MyIndex
python app.py
```

---

## Typical Workflow

1. **Start Ollama** — make sure it's running in the background
2. **Launch the app** — `python app.py`
3. **Index your files** — go to the Index tab, enter a folder path, click Index
4. **Ask questions** — go to the Chat tab and start asking
5. **Search for specifics** — use the Search tab for excerpt-level results with scores
6. **Sync new files** — use the Sync tab to pull from S3 or email, they're indexed automatically

---

## Troubleshooting

**"No models found. Is Ollama running?"**
Ollama isn't running. Start it with `ollama serve` in a separate terminal, or launch the Ollama desktop app.

**"No relevant content found in indexed files"**
The files haven't been indexed yet, or the query doesn't match anything in the index. Go to the Index tab and index the relevant folder first.

**Chat responses are slow**
Normal for local models — `phi3:mini` is the fastest option. Larger models (e.g. `llama3:8b`) give better quality but are slower. Switch models in the Settings tab.

**KeyError or connection error on startup**
Make sure Ollama is running before launching `python app.py`.

**"Error: path not allowed"**
You tried to read, write, or browse a path outside the `ROOT_DIR` sandbox. Move the file into the assistant's root folder or change `ROOT_DIR` in `config.py`.
