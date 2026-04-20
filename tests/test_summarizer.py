"""
Tests for Summarizer: direct path (short text) and map-reduce path (long text).

The mock LLM always yields "Mock answer." so we verify the control flow
(which code path is taken, how many generate() calls are made) rather than
the textual quality of the output — that belongs in integration tests.
"""

from __future__ import annotations

import pytest
import config
from conftest import SHORT_TEXT, LONG_TEXT


class TestDirectSummarize:
    """Text shorter than CHUNK_SIZE * 4 should use the direct (single-call) path."""

    def test_short_text_produces_output(self, summarizer):
        tokens = list(summarizer.summarize(SHORT_TEXT))
        assert len(tokens) > 0, "Summarizer produced no tokens"

    def test_output_is_strings(self, summarizer):
        tokens = list(summarizer.summarize(SHORT_TEXT))
        for t in tokens:
            assert isinstance(t, str)

    def test_short_text_single_llm_call(self, summarizer, mock_llm):
        """Direct path should call generate() exactly once."""
        list(summarizer.summarize(SHORT_TEXT))
        assert mock_llm._client.chat.call_count == 1

    def test_empty_text_does_not_raise(self, summarizer):
        """Empty input should return something (even a placeholder), not raise."""
        try:
            tokens = list(summarizer.summarize(""))
            assert isinstance(tokens, list)
        except Exception as exc:
            pytest.fail(f"summarize('') raised unexpectedly: {exc}")


class TestMapReduceSummarize:
    """Text exceeding CHUNK_SIZE * 4 must use the chunked map-reduce path."""

    def test_long_text_produces_output(self, summarizer):
        tokens = list(summarizer.summarize(LONG_TEXT))
        assert len(tokens) > 0

    def test_long_text_multiple_llm_calls(self, summarizer, mock_llm):
        """Map-reduce must call generate() more than once (once per chunk + combine)."""
        list(summarizer.summarize(LONG_TEXT))
        assert mock_llm._client.chat.call_count > 1, (
            "Long text should trigger multiple LLM calls (map-reduce)"
        )

    def test_threshold_boundary(self, summarizer, mock_llm):
        """Text exactly at the boundary should use the direct path."""
        boundary = "x" * (config.CHUNK_SIZE * 4)
        list(summarizer.summarize(boundary))
        # Boundary text is <= threshold, so exactly 1 call
        assert mock_llm._client.chat.call_count == 1
