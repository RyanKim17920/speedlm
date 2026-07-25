"""Proofs that encoded or fragmented credentials survive trace persistence."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.parse import quote

import pytest

from speedlm.traces.store import TraceRecord, TraceStore


def _record(content: object) -> TraceRecord:
    return TraceRecord(
        id="security-redaction",
        timestamp=1_700_000_000.0,
        model="test-model",
        messages=(
            {"role": "user", "content": content},
            {"role": "assistant", "content": "ok"},
        ),
        tool_calls=(),
        temperature=0.0,
        top_p=1.0,
        seed=0,
        prompt_tokens=10,
        completion_tokens=1,
    )


def _persist(tmp_path: Path, content: object) -> str:
    store = TraceStore(tmp_path / "traces.jsonl")
    report = store.append(_record(content))
    assert report is not None
    return store.path.read_text(encoding="utf-8")


@pytest.mark.xfail(
    strict=True,
    reason="redaction does not decode short base64-encoded assignments",
)
def test_base64_encoded_secret_must_not_reach_disk(tmp_path: Path) -> None:
    encoded = base64.b64encode(b"api_key=hunter2").decode("ascii")

    stored = _persist(tmp_path, encoded)

    assert encoded not in stored


@pytest.mark.xfail(
    strict=True,
    reason="redaction does not decode short hex-encoded assignments",
)
def test_hex_encoded_secret_must_not_reach_disk(tmp_path: Path) -> None:
    encoded = b"api_key=hunter2".hex()

    stored = _persist(tmp_path, encoded)

    assert encoded not in stored


@pytest.mark.xfail(
    strict=True,
    reason="redaction does not percent-decode before matching assignments",
)
def test_percent_encoded_secret_must_not_reach_disk(tmp_path: Path) -> None:
    encoded = quote("api_key=hunter2", safe="")

    stored = _persist(tmp_path, encoded)

    assert encoded not in stored


@pytest.mark.xfail(
    strict=True,
    reason="redaction does not normalize compatibility homoglyphs",
)
def test_unicode_homoglyph_secret_must_not_reach_disk(tmp_path: Path) -> None:
    obfuscated = "api＿key＝hunter2"  # full-width underscore and equals

    stored = _persist(tmp_path, obfuscated)

    assert "hunter2" not in stored


@pytest.mark.xfail(
    strict=True,
    reason="zero-width characters split sensitive assignment names",
)
def test_zero_width_obfuscated_secret_must_not_reach_disk(tmp_path: Path) -> None:
    obfuscated = "api_\u200bkey=hunter2"

    stored = _persist(tmp_path, obfuscated)

    assert "hunter2" not in stored


@pytest.mark.xfail(
    strict=True,
    reason="redaction state does not join secrets split across message boundaries",
)
def test_secret_split_across_messages_must_not_reach_disk(tmp_path: Path) -> None:
    secret = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
    first, second = secret[:15], secret[15:]
    store = TraceStore(tmp_path / "traces.jsonl")
    record = TraceRecord(
        id="split-secret",
        timestamp=1_700_000_000.0,
        model="test-model",
        messages=(
            {"role": "user", "content": first},
            {"role": "user", "content": second},
            {"role": "assistant", "content": "ok"},
        ),
        tool_calls=(),
        temperature=0.0,
        top_p=1.0,
        seed=0,
        prompt_tokens=10,
        completion_tokens=1,
    )

    assert store.append(record) is not None
    stored = store.path.read_text(encoding="utf-8")

    assert first not in stored
    assert second not in stored


@pytest.mark.xfail(
    strict=True,
    reason="mapping keys are copied without redaction",
)
def test_secret_embedded_in_field_name_must_not_reach_disk(tmp_path: Path) -> None:
    sensitive_key = "password=hunter2"
    content = [{"type": "metadata", sensitive_key: "ordinary"}]

    stored = _persist(tmp_path, content)

    assert sensitive_key not in stored


@pytest.mark.xfail(
    strict=True,
    reason="obfuscated sensitive keys inside tool arguments are not normalized",
)
def test_obfuscated_tool_argument_secret_must_not_reach_disk(tmp_path: Path) -> None:
    arguments = json.dumps({"api_\u200bkey": "hunter2"})
    record = TraceRecord(
        id="tool-secret",
        timestamp=1_700_000_000.0,
        model="test-model",
        messages=(
            {"role": "user", "content": "call the tool"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "publish", "arguments": arguments},
                    }
                ],
            },
        ),
        tool_calls=(),
        temperature=0.0,
        top_p=1.0,
        seed=0,
        prompt_tokens=10,
        completion_tokens=1,
    )
    store = TraceStore(tmp_path / "traces.jsonl")

    assert store.append(record) is not None
    stored = store.path.read_text(encoding="utf-8")

    assert "hunter2" not in stored


@pytest.mark.xfail(
    strict=True,
    reason="PEM matching requires literal header whitespace and dash placement",
)
def test_whitespace_obfuscated_pem_must_not_reach_disk(tmp_path: Path) -> None:
    pem = (
        "-----BEGIN\tPRIVATE KEY-----\n"
        "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=\n"
        "-----END\tPRIVATE KEY-----"
    )

    stored = _persist(tmp_path, pem)

    assert "PRIVATE KEY" not in stored
