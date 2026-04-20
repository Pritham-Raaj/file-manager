from __future__ import annotations

from typing import Generator, Optional

import config
from logger import get_logger

log = get_logger(__name__)


class LLMClient:

    def __init__(self, model: Optional[str] = None):
        self.model   = model or config.GENERATION_MODEL
        self._client = None     # lazy -- only connects on first use

    def _get(self):
        if self._client is None:
            import ollama
            self._client = ollama.Client(host=config.OLLAMA_HOST)
            log.debug("Ollama client initialised | host=%s", config.OLLAMA_HOST)
        return self._client

    # Entry point for all text generation -- streams tokens to avoid buffering
    def generate(self, prompt: str, system: str = "") -> Generator[str, None, None]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        log.debug("generate | model=%s | prompt_len=%d", self.model, len(prompt))
        try:
            for chunk in self._get().chat(model=self.model, messages=messages, stream=True):
                # ollama >= 0.4 returns Pydantic objects; fall back to dict access for older builds
                msg = chunk.message if hasattr(chunk, "message") else chunk["message"]
                yield msg.content if hasattr(msg, "content") else msg["content"]
        except Exception as exc:
            log.error("generate failed: %s", exc, exc_info=True)
            raise

    def embed(self, text: str) -> list[float]:
        log.debug("embed | text_len=%d", len(text))
        try:
            client = self._get()
            if hasattr(client, "embed"):
                resp = client.embed(model=config.EMBEDDING_MODEL, input=text)
                raw  = resp.embeddings if hasattr(resp, "embeddings") else resp["embeddings"]
                log.debug("embed raw type=%s len=%s", type(raw), len(raw) if raw else 0)
                # ollama 0.6+ returns List[List[float]]; unwrap the outer list
                if isinstance(raw, list) and raw and isinstance(raw[0], list):
                    vec = raw[0]
                else:
                    vec = raw
            else:
                resp = client.embeddings(model=config.EMBEDDING_MODEL, prompt=text)
                vec  = resp["embedding"]

            if not vec:
                raise ValueError(
                    "embed() returned an empty vector for model '{}'. "
                    "Check that the model is pulled and supports embeddings.".format(
                        config.EMBEDDING_MODEL
                    )
                )
            return vec
        except Exception as exc:
            log.error("embed failed: %s", exc, exc_info=True)
            raise

    def list_models(self) -> list[str]:
        try:
            response = self._get().list()
            # ollama >= 0.4 returns a ListResponse Pydantic object
            models = response.models if hasattr(response, "models") else response.get("models", [])
            names  = []
            for m in models:
                # Model objects use .model attribute; older builds used m["name"]
                if hasattr(m, "model"):
                    names.append(m.model)
                elif hasattr(m, "name"):
                    names.append(m.name)
                else:
                    names.append(m["name"])
            return names
        except Exception as exc:
            log.error("list_models failed: %s", exc, exc_info=True)
            raise

    def switch_model(self, model: str) -> None:
        log.info("Model switch: %s -> %s", self.model, model)
        self.model = model
