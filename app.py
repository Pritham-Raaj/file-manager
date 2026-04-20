from __future__ import annotations

import gradio as gr

import config
from logger import get_logger
from orchestrator import Orchestrator

log  = get_logger(__name__)
orch = Orchestrator()


# ── Chat handler ──────────────────────────────────────────────────────────────

def chat(message: str, history: list, file_filter: str):
    """Stream a response for Q&A or file summarization."""
    history = history or []
    msg     = message.strip()

    if msg.lower().startswith("summarize "):
        path = msg[len("summarize "):].strip()
        gen  = orch.summarize_file(path)
    else:
        gen = orch.ask(msg, file_filter=file_filter or None)

    history = list(history) + [{"role": "user", "content": message}]
    partial = ""
    for token in gen:
        partial += token
        yield history + [{"role": "assistant", "content": partial}]

    log.info("chat | response_len=%d", len(partial))


# ── Utility handlers ──────────────────────────────────────────────────────────

def index_path(path: str) -> str:
    try:
        return orch.index((path or "").strip())
    except Exception as exc:
        log.error("index_path failed: %s", exc, exc_info=True)
        return "Error: {}".format(exc)


def list_dir(directory: str) -> str:
    try:
        entries = orch.list_files(directory.strip())
    except Exception as exc:
        return "Error: {}".format(exc)

    if not entries:
        return "Empty."
    lines = [
        "{} {}  ({} bytes)".format("📁" if e["is_dir"] else "📄", e["name"], e["size"])
        for e in entries
    ]
    return "\n".join(lines)


def read_file(path: str) -> str:
    try:
        return orch.read_file(path.strip())
    except Exception as exc:
        return "Error: {}".format(exc)


def search_files(query: str, file_filter: str) -> str:
    hits = orch.search(query.strip(), file_filter=(file_filter or "").strip() or None)
    if not hits:
        return "No results found."
    blocks = []
    for i, h in enumerate(hits, 1):
        meta     = h["metadata"]
        score    = int(h["score"] * 100)
        name     = meta.get("file_name", "Unknown file")
        path     = meta.get("file_path", "")
        page     = meta.get("page_num", "?")
        ftype    = meta.get("file_type", "").lstrip(".")
        excerpt  = h["text"].strip().replace("\n", " ")
        excerpt  = excerpt[:300] + ("..." if len(h["text"].strip()) > 300 else "")

        block = (
            "Result {i}  |  Relevance: {score}%\n"
            "  File   : {name}{ftype}\n"
            "  Path   : {path}\n"
            "  Page   : {page}\n"
            "  Excerpt: {excerpt}"
        ).format(
            i=i, score=score, name=name,
            ftype=" ({})".format(ftype) if ftype else "",
            path=path, page=page, excerpt=excerpt,
        )
        blocks.append(block)
    separator = "\n" + "-" * 60 + "\n"
    return separator.join(blocks)


def create_file(path: str, content: str) -> str:
    try:
        return orch.create_file(path.strip(), content)
    except Exception as exc:
        log.error("create_file failed: %s", exc, exc_info=True)
        return "Error: {}".format(exc)


def patch_file(path: str, old_text: str, new_text: str) -> str:
    try:
        return orch.patch_file(path.strip(), old_text, new_text)
    except Exception as exc:
        log.error("patch_file failed: %s", exc, exc_info=True)
        return "Error: {}".format(exc)


def register_s3(name: str, bucket: str, prefix: str) -> str:
    try:
        return orch.register_s3(name.strip(), bucket.strip(), prefix.strip())
    except Exception as exc:
        log.error("register_s3 failed: %s", exc, exc_info=True)
        return "Error: {}".format(exc)


def pull_s3(name: str) -> str:
    try:
        return orch.sync_cloud(source=name.strip() or None)
    except Exception as exc:
        log.error("pull_s3 failed: %s", exc, exc_info=True)
        return "Error: {}".format(exc)


def pull_email(host: str, user: str, password: str, folder: str, subject: str) -> str:
    try:
        return orch.sync_email(
            host.strip(), user.strip(), password,
            folder=folder.strip() or "INBOX",
            subject_filter=subject.strip() or None,
        )
    except Exception as exc:
        log.error("pull_email failed: %s", exc, exc_info=True)
        return "Error: {}".format(exc)


def list_sources() -> str:
    try:
        return orch.list_sources()
    except Exception as exc:
        log.error("list_sources failed: %s", exc, exc_info=True)
        return "Error: {}".format(exc)


def remove_path(path: str) -> str:
    try:
        return orch.remove_path(path.strip())
    except Exception as exc:
        log.error("remove_path failed: %s", exc, exc_info=True)
        return "Error: {}".format(exc)


def switch_model(model: str) -> str:
    try:
        return orch.switch_model(model.strip())
    except Exception as exc:
        return "Error: {}".format(exc)


def list_models() -> str:
    try:
        models = orch.list_models()
        return "\n".join(models) if models else "No models found. Is Ollama running?"
    except Exception as exc:
        return "Error: {}".format(exc)


def clear_index() -> str:
    try:
        return orch.clear_index()
    except Exception as exc:
        log.error("clear_index failed: %s", exc, exc_info=True)
        return "Error: {}".format(exc)


# -- UI -----------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Local File Assistant") as ui:
        gr.Markdown("## Local AI File Assistant")
        gr.Markdown(
            "A fully local AI assistant for your files. All processing happens on your machine — "
            "nothing is sent to the cloud. Index your files once, then ask questions, search, "
            "read, and summarize them through the tabs below."
        )

        with gr.Tab("💬 Chat"):
            gr.Markdown(
                "**Ask questions about your indexed files** using natural language. "
                "The assistant searches your index for relevant content and answers based on what it finds.\n\n"
                "- **Ask anything:** e.g. *What does the Q3 report say about revenue?*\n"
                "- **Summarize a file:** type `summarize <full file path>` to get a concise summary of any supported file.\n"
                "- **Limit to a specific file:** fill in the filter box with part of the file's path or name "
                "to restrict answers to that file only (e.g. `report.pdf`).\n\n"
                "> **Note:** Files must be indexed first (use the **Index** tab) before the assistant can answer questions about them."
            )
            file_filter = gr.Textbox(label="Limit to file (optional path substring)")
            chatbot     = gr.Chatbot(height=450)
            msg         = gr.Textbox(label="Ask a question or type: summarize <path>")
            msg.submit(chat, [msg, chatbot, file_filter], chatbot)
            msg.submit(lambda: "", None, msg)

        with gr.Tab("🔍 Search"):
            gr.Markdown(
                "**Semantic search** across all indexed files. Unlike a keyword search, this understands "
                "meaning — so searching *budget cuts* will also surface chunks about *cost reductions* or "
                "*spending limits*.\n\n"
                "- Enter a plain-English query and click **Search**.\n"
                "- Use the **File filter** to narrow results to a specific file or folder "
                "(enter any substring of the path, e.g. `reports/` or `invoice.pdf`).\n"
                "- Results are ranked by relevance and show the source file, page, and a text excerpt."
            )
            s_query  = gr.Textbox(label="Search query")
            s_filter = gr.Textbox(label="File filter (optional path substring)")
            s_btn    = gr.Button("Search")
            s_out    = gr.Textbox(label="Results", lines=20, interactive=False)
            s_btn.click(search_files, [s_query, s_filter], s_out)

        with gr.Tab("📥 Index"):
            gr.Markdown(
                "**Index a file or folder** so the assistant can answer questions about it. "
                "Indexing reads the file, splits it into chunks, and stores vector embeddings in a local database.\n\n"
                "- Enter the full path to a file (e.g. `/home/user/docs/report.pdf`) or a directory "
                "(e.g. `/home/user/docs/`) and click **Index**.\n"
                "- Supported formats: PDF, DOCX, PPTX, XLSX, plain text, Markdown, CSV, JSON, and most code files.\n"
                "- Already-indexed files are skipped automatically unless they have changed.\n"
                "- To remove something from the index later, use the **Manage Index** tab."
            )
            i_path = gr.Textbox(label="File or directory path")
            i_btn  = gr.Button("Index")
            i_out  = gr.Textbox(label="Result", interactive=False)
            i_btn.click(index_path, i_path, i_out)

        with gr.Tab("📂 Browse"):
            gr.Markdown(
                "**Browse the contents of your sandbox** — the local directory the assistant is allowed to access.\n\n"
                "- Leave the directory box blank to list the root of the sandbox.\n"
                "- Enter a subdirectory path to navigate deeper.\n"
                "- Folders are shown with 📁 and files with 📄, along with their sizes."
            )
            b_dir = gr.Textbox(label="Directory path (blank = sandbox root)")
            b_btn = gr.Button("List")
            b_out = gr.Textbox(label="Contents", lines=20, interactive=False)
            b_btn.click(list_dir, b_dir, b_out)

        with gr.Tab("📄 Read"):
            gr.Markdown(
                "**Read the full contents of a file** directly. "
                "For documents (PDF, DOCX, PPTX, XLSX), the text is extracted and displayed cleanly. "
                "For plain text and code files, the raw content is shown.\n\n"
                "- Enter the full path to the file and click **Read**.\n"
                "- Useful for previewing a file before indexing, or verifying its contents.\n"
                "- Very large files are truncated at 500,000 characters."
            )
            r_path = gr.Textbox(label="File path")
            r_btn  = gr.Button("Read")
            r_out  = gr.Textbox(label="Contents", lines=20, interactive=False)
            r_btn.click(read_file, r_path, r_out)

        with gr.Tab("✏️ Write"):
            gr.Markdown(
                "**Create or edit plain-text files** in the sandbox.\n\n"
                "- **Create file:** write a new file at the given path with the content you provide. "
                "The file must not already exist.\n"
                "- **Patch file:** find and replace a specific block of text inside an existing file. "
                "Enter the exact text to find and the text to replace it with. "
                "All occurrences are replaced."
            )
            with gr.Accordion("Create file", open=True):
                w_path    = gr.Textbox(label="File path (e.g. notes/todo.txt)")
                w_content = gr.Textbox(label="Content", lines=10)
                w_btn     = gr.Button("Create")
                w_out     = gr.Textbox(label="Result", interactive=False)
                w_btn.click(create_file, [w_path, w_content], w_out)
            with gr.Accordion("Patch file", open=False):
                p_path = gr.Textbox(label="File path")
                p_old  = gr.Textbox(label="Text to replace (must match exactly)", lines=4)
                p_new  = gr.Textbox(label="Replace with", lines=4)
                p_btn  = gr.Button("Patch")
                p_out  = gr.Textbox(label="Result", interactive=False)
                p_btn.click(patch_file, [p_path, p_old, p_new], p_out)

        with gr.Tab("☁️ Sync"):
            gr.Markdown(
                "**Pull files from cloud sources into your local sandbox.** "
                "This is pull-only — files are downloaded to your machine. Nothing is ever uploaded.\n\n"
                "Once pulled, files are automatically indexed so you can query them straight away."
            )
            with gr.Accordion("S3", open=True):
                gr.Markdown(
                    "Download files from an AWS S3 bucket. Requires AWS credentials to be set in your "
                    "environment (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).\n"
                    "1. Enter a **connector name** (any label you choose), the **bucket name**, and an optional "
                    "**key prefix** to limit which files are pulled (e.g. `reports/2024/`).\n"
                    "2. Click **Register** to save the connector, then **Pull** to download the files."
                )
                sy_name   = gr.Textbox(label="Connector name (your label for this source)")
                sy_bucket = gr.Textbox(label="S3 bucket name")
                sy_prefix = gr.Textbox(label="Key prefix (optional, e.g. reports/2024/)")
                with gr.Row():
                    sy_reg  = gr.Button("Register")
                    sy_pull = gr.Button("Pull")
                sy_out = gr.Textbox(label="Result", interactive=False)
                sy_reg.click(register_s3, [sy_name, sy_bucket, sy_prefix], sy_out)
                sy_pull.click(pull_s3, sy_name, sy_out)
            with gr.Accordion("Email (IMAP)", open=False):
                gr.Markdown(
                    "Download **email attachments** from any IMAP mailbox (Gmail, Outlook, etc.).\n"
                    "- For Gmail, use `imap.gmail.com` and an **App Password** (not your regular password).\n"
                    "- Use the **Subject filter** to only pull attachments from emails matching a keyword.\n"
                    "- At most 50 most recent matching emails are checked per pull."
                )
                em_host    = gr.Textbox(label="IMAP host (e.g. imap.gmail.com)")
                em_user    = gr.Textbox(label="Email address")
                em_pass    = gr.Textbox(label="Password / App Password", type="password")
                em_folder  = gr.Textbox(label="Folder", value="INBOX")
                em_subject = gr.Textbox(label="Subject filter (optional keyword)")
                em_btn     = gr.Button("Pull attachments")
                em_out     = gr.Textbox(label="Result", interactive=False)
                em_btn.click(pull_email, [em_host, em_user, em_pass, em_folder, em_subject], em_out)

        with gr.Tab("🗂️ Manage Index"):
            gr.Markdown(
                "**View and clean up your index.** The index is what powers search and chat — "
                "it holds the text chunks and embeddings extracted from your files.\n\n"
                "- Click **Show indexed locations** to see every directory and file currently in the index, "
                "along with chunk counts and when they were last indexed.\n"
                "- To free up space or remove stale content, enter a file or directory path and click "
                "**Remove from index**. This removes the embeddings only — your actual files are untouched."
            )
            mi_list_btn = gr.Button("Show indexed locations")
            mi_out      = gr.Textbox(label="Indexed sources", lines=20, interactive=False)
            mi_list_btn.click(list_sources, None, mi_out)
            gr.Markdown("---")
            mi_path = gr.Textbox(label="Path to remove (file or directory)")
            mi_rm   = gr.Button("Remove from index", variant="stop")
            mi_res  = gr.Textbox(label="Result", interactive=False)
            mi_rm.click(remove_path, mi_path, mi_res)

        with gr.Tab("⚙️ Settings"):
            gr.Markdown(
                "**Switch the active Ollama model.** The assistant uses two models:\n\n"
                "- **Generation model** — used for chat, Q&A, and summarization (default: `phi3:mini`).\n"
                "- **Embedding model** — used for indexing and search (default: `nomic-embed-text`). "
                "Changing this after files are already indexed will make existing embeddings incompatible — "
                "you'd need to re-index everything.\n\n"
                "Use **List available models** to see what's currently pulled in Ollama, "
                "then enter a model name and click **Switch model** to change the generation model."
            )
            sm_list = gr.Button("List available models")
            sm_out  = gr.Textbox(label="Available models", lines=6, interactive=False)
            sm_list.click(list_models, None, sm_out)
            sm_name = gr.Textbox(label="Model name to switch to")
            sm_btn  = gr.Button("Switch model")
            sm_res  = gr.Textbox(label="Result", interactive=False)
            sm_btn.click(switch_model, sm_name, sm_res)

        # ── End-of-session panel ──────────────────────────────────────────────
        # Sits below all tabs and is always visible. When the user tries to close
        # the tab, JavaScript shows the browser's "Leave site?" dialog — if they
        # click Stay, this panel is scrolled into view and highlighted so they
        # can make an explicit Keep / Clear choice before leaving.
        gr.HTML("""
        <hr style="margin: 24px 0 16px;">
        <div id="end-session-panel"
             style="border: 1.5px solid #e5e7eb; border-radius: 10px;
                    padding: 18px 22px; background: #f9fafb;
                    transition: box-shadow 0.4s, border-color 0.4s;">
            <p style="margin: 0 0 6px; font-weight: 600; font-size: 1.05em;">
                🔚 End of session
            </p>
            <p style="margin: 0 0 14px; color: #4b5563; font-size: 0.95em;">
                Use these buttons before closing the tab.
                <strong>Keep</strong> leaves your index intact for next time.
                <strong>Clear</strong> wipes all embeddings — your actual files are never touched.
            </p>
        </div>
        <script>
        (function () {
            // Guard against duplicate registration: Gradio's React renderer can
            // unmount/remount components, which would re-run this script and stack
            // up multiple beforeunload listeners without the flag below.
            if (window._endSessionListenerAdded) return;
            window._endSessionListenerAdded = true;

            // Warn the user when they try to close / navigate away.
            // Modern browsers show a generic "Leave site?" dialog — we can't
            // customise the text, but it gives them a chance to click "Stay"
            // and then use the Keep / Clear buttons below.
            window.addEventListener('beforeunload', function (e) {
                e.preventDefault();
                e.returnValue = '';     // required for Chrome to show the dialog

                // Scroll to and highlight the panel so it's obvious on return.
                var panel = document.getElementById('end-session-panel');
                if (panel) {
                    panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    panel.style.borderColor  = '#f59e0b';
                    panel.style.boxShadow    = '0 0 0 3px rgba(245,158,11,0.3)';
                    panel.style.background   = '#fffbeb';
                    // Fade back to normal after 4 s in case the user stays.
                    setTimeout(function () {
                        panel.style.borderColor = '#e5e7eb';
                        panel.style.boxShadow   = 'none';
                        panel.style.background  = '#f9fafb';
                    }, 4000);
                }
            });
        })();
        </script>
        """)

        with gr.Row():
            es_keep  = gr.Button("✅ Keep index", variant="primary", scale=1)
            es_clear = gr.Button("🗑️ Clear index", variant="stop", scale=1)
            es_out   = gr.Textbox(
                value="Use the buttons to keep or clear your index before closing.",
                label="", show_label=False, interactive=False, scale=3,
            )

        es_keep.click(
            fn=lambda: "Index kept — your files and embeddings are preserved for next session.",
            outputs=es_out,
        )
        es_clear.click(fn=clear_index, outputs=es_out)

    return ui


if __name__ == "__main__":
    log.info("Starting | root=%s | port=%d", config.ROOT_DIR, config.GRADIO_PORT)
    build_ui().launch(server_name=config.GRADIO_HOST, server_port=config.GRADIO_PORT,
                      theme=gr.themes.Soft())
