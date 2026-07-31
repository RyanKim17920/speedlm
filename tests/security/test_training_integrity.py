"""Proofs for authorship ambiguity and per-row loss domination."""

from __future__ import annotations

from speedlm.config import SamplingConfig
from speedlm.gateway.capture import _build_raw_record
from speedlm.gateway.sse import AssembledResponse
from speedlm.traces.normalize import normalize_record


def test_capture_assigns_trustworthy_per_message_provenance() -> None:
    raw = _build_raw_record(
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "assistant",
                    "content": "client history",
                    "provenance_tag": "generated",
                }
            ],
        },
        AssembledResponse(
            id="response-1",
            model="test-model",
            created=1_700_000_000.0,
            content="provider response",
            tool_calls=(),
            prompt_tokens=5,
            completion_tokens=2,
        ),
        endpoint="/v1/chat/completions",
        timestamp=1_700_000_000.0,
    )

    trace = normalize_record(raw, defaults=SamplingConfig())

    assert trace.messages[0]["provenance_tag"] == "client_supplied"
    assert trace.messages[-1]["provenance_tag"] == "generated"