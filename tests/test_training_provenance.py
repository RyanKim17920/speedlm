"""Tests for training/provenance.py - self-play attestation helpers.

The provenance module provides the building blocks that decide whether captured
trace records are safe to train on with ALL_ASSISTANT_TURNS masking.
`self_play_attestation` is tested in test_self_play_trust.py (integration with
eagle3.py's renderer). Here we test the low-level primitives it depends on:
`assistant_fingerprint` and `iter_jsonl`, which are the atoms of the check.
"""

from __future__ import annotations

import pytest

from speedlm.training.provenance import (
    SelfPlayAttestation,
    assistant_fingerprint,
    iter_jsonl,
)

# ---------------------------------------------------------------------------
# assistant_fingerprint
# ---------------------------------------------------------------------------


def test_assistant_fingerprint_content_identity() -> None:
    """Two messages with the same content produce the same fingerprint."""
    msg1 = {"role": "assistant", "content": "hello", "provenance_tag": "generated"}
    msg2 = {"role": "assistant", "content": "hello", "provenance_tag": "client_supplied"}

    fp1 = assistant_fingerprint(msg1)
    fp2 = assistant_fingerprint(msg2)

    assert fp1 == fp2


def test_assistant_fingerprint_different_content() -> None:
    """Messages with different content produce different fingerprints."""
    msg1 = {"role": "assistant", "content": "hello"}
    msg2 = {"role": "assistant", "content": "world"}

    assert assistant_fingerprint(msg1) != assistant_fingerprint(msg2)


def test_assistant_fingerprint_ignores_provenance_tag() -> None:
    """provenance_tag is excluded because it differs across the round trip."""
    msg1 = {"role": "assistant", "content": "x", "provenance_tag": "generated"}
    msg2 = {"role": "assistant", "content": "x", "provenance_tag": "client_supplied"}
    msg3 = {"role": "assistant", "content": "x"}

    assert assistant_fingerprint(msg1) == assistant_fingerprint(msg2)
    assert assistant_fingerprint(msg2) == assistant_fingerprint(msg3)


def test_assistant_fingerprint_includes_tool_calls() -> None:
    """Tool call semantics affect the fingerprint."""
    tc_read = [{"function": {"name": "read", "arguments": "{}"}}]
    tc_write = [{"function": {"name": "write", "arguments": "{}"}}]
    msg1 = {"role": "assistant", "content": "", "tool_calls": tc_read}
    msg2 = {"role": "assistant", "content": "", "tool_calls": tc_write}

    assert assistant_fingerprint(msg1) != assistant_fingerprint(msg2)


def test_assistant_fingerprint_tool_call_ids_ignored() -> None:
    """Server-assigned tool call ids are excluded from the fingerprint."""
    tc1 = [{"id": "c1", "function": {"name": "read", "arguments": "{}"}}]
    tc2 = [{"id": "different", "function": {"name": "read", "arguments": "{}"}}]
    msg1 = {"role": "assistant", "content": "", "tool_calls": tc1}
    msg2 = {"role": "assistant", "content": "", "tool_calls": tc2}

    assert assistant_fingerprint(msg1) == assistant_fingerprint(msg2)


def test_assistant_fingerprint_includes_reasoning_content() -> None:
    """The reasoning channel is part of semantic identity."""
    msg1 = {"role": "assistant", "content": "x", "reasoning_content": "step 1"}
    msg2 = {"role": "assistant", "content": "x", "reasoning_content": "step 2"}

    assert assistant_fingerprint(msg1) != assistant_fingerprint(msg2)


def test_assistant_fingerprint_empty_message() -> None:
    """An empty assistant message still produces a deterministic fingerprint."""
    msg1 = {"role": "assistant", "content": None}
    msg2 = {"role": "assistant"}

    fp1 = assistant_fingerprint(msg1)
    fp2 = assistant_fingerprint(msg2)

    assert fp1 == fp2
    assert isinstance(fp1, str)
    assert len(fp1) == 64  # SHA-256 hex digest


def test_assistant_fingerprint_deterministic() -> None:
    """The same message always produces the same fingerprint."""
    msg = {"role": "assistant", "content": "deterministic test", "tool_calls": []}

    fp1 = assistant_fingerprint(msg)
    fp2 = assistant_fingerprint(msg)

    assert fp1 == fp2


# ---------------------------------------------------------------------------
# iter_jsonl
# ---------------------------------------------------------------------------


def test_iter_jsonl_parses_valid_lines() -> None:
    lines = [
        '{"id": "row-0", "messages": []}',
        '{"id": "row-1", "messages": [{"role": "user", "content": "hi"}]}',
    ]
    result = iter_jsonl(lines)
    assert len(result) == 2
    assert result[0]["id"] == "row-0"
    assert result[1]["id"] == "row-1"


def test_iter_jsonl_skips_blank_lines() -> None:
    lines = [
        '{"id": "a"}',
        "",
        "   ",
        '{"id": "b"}',
    ]
    result = iter_jsonl(lines)
    assert len(result) == 2
    assert result[0]["id"] == "a"
    assert result[1]["id"] == "b"


def test_iter_jsonl_empty_input() -> None:
    result = iter_jsonl([])
    assert result == []


def test_iter_jsonl_invalid_json_raises_with_line_number() -> None:
    lines = ['{"id": "ok"}', "not json at all", '{"id": "ok2"}']
    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        iter_jsonl(lines)


def test_iter_jsonl_non_object_raises() -> None:
    lines = ["[1, 2, 3]"]
    with pytest.raises(ValueError, match="line 1 is not a JSON object"):
        iter_jsonl(lines)


def test_iter_jsonl_string_value_raises() -> None:
    lines = ['"just a string"']
    with pytest.raises(ValueError, match="line 1 is not a JSON object"):
        iter_jsonl(lines)


# ---------------------------------------------------------------------------
# SelfPlayAttestation.detail
# ---------------------------------------------------------------------------


def test_attestation_detail_empty_rows() -> None:
    att = SelfPlayAttestation(
        attested=False,
        rows_examined=0,
        generated_turns=0,
        prefix_assistant_turns=0,
        matched_prefix_turns=0,
        unmatched=(),
    )
    assert "no rows were examined" in att.detail
    assert "nothing was attested" in att.detail


def test_attestation_detail_success() -> None:
    att = SelfPlayAttestation(
        attested=True,
        rows_examined=10,
        generated_turns=20,
        prefix_assistant_turns=5,
        matched_prefix_turns=5,
        unmatched=(),
    )
    assert "10 rows" in att.detail
    assert "20 generated turns" in att.detail
    assert "5/5" in att.detail


def test_attestation_detail_failure_names_unmatched() -> None:
    att = SelfPlayAttestation(
        attested=False,
        rows_examined=10,
        generated_turns=20,
        prefix_assistant_turns=5,
        matched_prefix_turns=3,
        unmatched=("row-a[0] tag='x'", "row-b[1] tag='y'"),
    )
    assert "row-a[0]" in att.detail
    assert "row-b[1]" in att.detail


def test_attestation_detail_elides_many_unmatched() -> None:
    att = SelfPlayAttestation(
        attested=False,
        rows_examined=10,
        generated_turns=20,
        prefix_assistant_turns=10,
        matched_prefix_turns=3,
        unmatched=tuple(f"unmatched-{i}" for i in range(10)),
    )
    assert "and 5 more" in att.detail
    # First 5 are listed
    assert "unmatched-0" in att.detail
    assert "unmatched-4" in att.detail
    assert "unmatched-5" not in att.detail


def test_attestation_to_dict() -> None:
    att = SelfPlayAttestation(
        attested=True,
        rows_examined=5,
        generated_turns=10,
        prefix_assistant_turns=2,
        matched_prefix_turns=2,
        unmatched=(),
    )
    d = att.to_dict()
    assert d["attested"] is True
    assert d["rows_examined"] == 5
    assert d["generated_turns"] == 10
    assert d["prefix_assistant_turns"] == 2
    assert d["matched_prefix_turns"] == 2
    assert d["unmatched"] == []
    assert "5 rows" in d["detail"]
