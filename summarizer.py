from __future__ import annotations

from typing import Generator, Optional

import config
from logger import get_logger
from llm_client import LLMClient

log = get_logger(__name__)

_SYSTEM = "You are a precise summarizer. Be concise and factual. Omit filler."

_CHUNK_PROMPT   = "Summarize in 2-3 sentences, key facts only:\n\n{text}"
_COMBINE_PROMPT = "Combine these partial summaries into one coherent summary (4-6 sentences):\n\n{text}"


class Summarizer:

    def __init__(self, llm: Optional[LLMClient] = None):
        self._llm = llm or LLMClient()

    # Threshold: files under 4× CHUNK_SIZE go direct; larger files use map-reduce
    def summarize(self, text: str) -> Generator[str, None, None]:
        if len(text) <= config.CHUNK_SIZE * 4:
            yield from self._direct(text)
        else:
            yield from self._map_reduce(text)

    def _direct(self, text: str) -> Generator[str, None, None]:
        log.debug("summarize | direct | len=%d", len(text))
        yield from self._llm.generate(_CHUNK_PROMPT.format(text=text), system=_SYSTEM)

    # Map-reduce for large files: chunk → summarize each → combine into final
    def _map_reduce(self, text: str) -> Generator[str, None, None]:
        log.info("summarize | map-reduce | len=%d", len(text))
        window   = config.CHUNK_SIZE * 4
        partials = []
        pos      = 0

        while pos < len(text):
            chunk   = text[pos : pos + window]
            summary = "".join(self._llm.generate(_CHUNK_PROMPT.format(text=chunk), system=_SYSTEM))
            partials.append(summary)
            pos += window
            log.debug("map-reduce | partial %d done", len(partials))

        yield from self._llm.generate(
            _COMBINE_PROMPT.format(text="\n\n".join(partials)),
            system=_SYSTEM,
        )
