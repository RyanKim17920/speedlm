"""Tests for the idle-tuning seed corpus: what is loaded and what is sent.

CI-safe on purpose -- no ``e2e`` marker, no GPU, no SLURM, no network.  The
helpers under test are module-level pure functions in
``tests/e2e/test_live_idle_tuning.py`` (importing that module has no side
effects; it only defines constants and functions), so they can be exercised
directly on this host even though the test they serve cannot run here.

Two defects are pinned here, both of which shipped green:

* the loader asserted the FIRST message was a user turn and then kept only
  ``first["content"]`` -- a bare string.  Every corpus was therefore forced to
  be single-turn, tool schemas were dropped on the floor, and any record whose
  messages begin with a system prompt (i.e. every agentic record) hard-failed;
* ``--workload`` was a launcher option with a preflight that validated it and
  no consumer at all for this flavor, so an operator could ask for agentic
  traffic, be told the launch was fine, and get generic chat.

Both halves are asserted throughout: the new shape loads AND the historical
generic-chat shape still produces the byte-identical payload it produced
before, because three archived runs and every number in
``docs/benchmark-evidence.md`` were measured on that path.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.e2e import test_live_idle_tuning as idle
from tests.e2e.harness import workloads as W

AGENTIC_WORKLOAD = "agentic-mixed-outcome"

#: A tool schema shaped like the ones in the agentic workloads.
TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def _config() -> SimpleNamespace:
    """The only parts of ``SpeedLMConfig`` that ``_post_chat`` reads."""
    return SimpleNamespace(
        alias="speedlm-alias",
        sampling=SimpleNamespace(temperature=0.0, top_p=1.0, seed=7),
    )


def _write_corpus(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def _sent_payload(
    monkeypatch: pytest.MonkeyPatch, request: idle.SeedRequest
) -> dict[str, Any]:
    """Post one request through ``_post_chat`` and return the JSON body sent."""
    captured: dict[str, Any] = {}

    class _Response:
        status_code = 200
        text = "ok"

        @staticmethod
        def json() -> dict[str, Any]:
            return {
                "id": "cmpl-1",
                "choices": [{"finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 3},
            }

    def fake_post(url: str, **kwargs: Any) -> _Response:
        captured.update(kwargs["json"])
        return _Response()

    monkeypatch.setattr(idle.httpx, "post", fake_post)
    idle._post_chat("http://127.0.0.1:1", _config(), request, timeout=1.0)
    return captured


# ---------------------------------------------------------------------------
# Loading: the agentic shape
# ---------------------------------------------------------------------------
def test_leading_system_message_and_tools_round_trip_into_the_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record the old loader could not even read, end to end.

    Its first message is a system prompt (the old assert died here) and it
    declares tools (the old loader had nowhere to put them).  Both must survive
    all the way onto the wire, verbatim.
    """
    record = {
        "id": "agentic-0",
        "messages": [
            {"role": "system", "content": "You are a careful engineer."},
            {"role": "user", "content": [{"type": "text", "text": "fix the test"}]},
        ],
        "tools": [TOOL_SCHEMA],
    }
    corpus = _write_corpus(tmp_path / "agentic.jsonl", [record])
    monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
    monkeypatch.delenv("SPEEDLM_E2E_WORKLOAD", raising=False)

    loaded = idle._load_seed_corpus()
    assert loaded is not None and len(loaded) == 1
    assert list(loaded[0].messages) == record["messages"]
    assert list(loaded[0].tools) == [TOOL_SCHEMA]

    payload = _sent_payload(monkeypatch, loaded[0])
    assert payload["messages"] == record["messages"], (
        "the system turn or the content-part list did not reach the wire"
    )
    assert payload["tools"] == [TOOL_SCHEMA], "tool schemas were dropped"


def test_assistant_tool_call_turn_without_content_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tool-call turn carries ``tool_calls`` and no ``content``; that is valid."""
    record = {
        "messages": [
            {"role": "user", "content": "run the tests"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "bash"}}
                ],
            },
            {"role": "tool", "content": "1 failed", "name": "bash"},
        ]
    }
    corpus = _write_corpus(tmp_path / "toolloop.jsonl", [record])
    monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
    loaded = idle._load_prompt_corpus()
    assert loaded is not None
    assert [m["role"] for m in loaded[0].messages] == ["user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# Loading: the historical generic-chat shape must not move
# ---------------------------------------------------------------------------
def test_ultrachat_shaped_record_produces_the_identical_single_user_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression that matters: the archived runs' request, byte for byte.

    ``/data/ryan.kim/speedlm-corpora/ultrachat-prompts.jsonl`` lines carry
    exactly one key, ``messages``, holding one user message with string content.
    """
    prompt = "Rice tolerance to suboptimal low temperatures ..."
    corpus = _write_corpus(
        tmp_path / "ultrachat.jsonl",
        [{"messages": [{"role": "user", "content": prompt}]}],
    )
    monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
    loaded = idle._load_seed_corpus()
    assert loaded is not None

    payload = _sent_payload(monkeypatch, loaded[0])
    assert payload == {
        "model": "speedlm-alias",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 7,
        "max_tokens": 512,
    }
    assert "tools" not in payload, "a tools key appeared on a corpus that has none"


def test_selection_picks_the_same_records_the_string_version_picked() -> None:
    """``random.sample`` draws positions, so carrying whole records cannot move it."""
    prompts = [f"prompt {index}" for index in range(500)]
    corpus = [idle._user_request(text) for text in prompts]
    chosen = idle._select_requests(corpus, seed_count=32)
    assert [request.messages[0]["content"] for request in chosen] == random.Random(
        42
    ).sample(prompts, 32)


def test_synthetic_fallback_still_produces_the_original_template() -> None:
    requests = idle._select_requests(None, seed_count=3)
    assert [request.messages for request in requests] == [
        (
            {
                "role": "user",
                "content": (
                    f"This is idle-tuning seed request {index}/3. "
                    "Reply with one short sentence."
                ),
            },
        )
        for index in (1, 2, 3)
    ]
    assert all(request.tools == () for request in requests)


def test_a_corpus_smaller_than_the_seed_count_still_raises() -> None:
    with pytest.raises(AssertionError, match="prompt corpus has 2 prompts"):
        idle._select_requests([idle._user_request("a"), idle._user_request("b")], seed_count=3)


# ---------------------------------------------------------------------------
# Validation stays strict -- about the right thing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("record", "expected"),
    [
        pytest.param([], "not a JSON object", id="not-an-object"),
        pytest.param({}, "no non-empty 'messages'", id="no-messages"),
        pytest.param({"messages": []}, "no non-empty 'messages'", id="empty-messages"),
        pytest.param(
            {"messages": "hello"}, "no non-empty 'messages'", id="messages-not-a-list"
        ),
        pytest.param({"messages": ["hi"]}, "is not an object", id="message-not-object"),
        pytest.param(
            {"messages": [{"content": "hi"}]}, "has role None", id="missing-role"
        ),
        pytest.param(
            {"messages": [{"role": 3, "content": "hi"}]}, "has role 3", id="role-not-str"
        ),
        pytest.param(
            {"messages": [{"role": "moderator", "content": "hi"}]},
            "has role 'moderator'",
            id="role-the-server-rejects",
        ),
        pytest.param(
            {"messages": [{"role": "user"}]},
            "carries neither 'content' nor 'tool_calls'",
            id="nothing-to-send",
        ),
        pytest.param(
            {"messages": [{"role": "system", "content": "be nice"}]},
            "no user message with content",
            id="no-user-turn",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": "   "}]},
            "no user message with content",
            id="blank-user-turn",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": "hi"}], "tools": []},
            "not a non-empty list",
            id="empty-tools",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": "hi"}], "tools": {"a": 1}},
            "not a non-empty list",
            id="tools-not-a-list",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": "hi"}], "tools": ["bash"]},
            "tools[0] is not an object",
            id="tool-not-an-object",
        ),
    ],
)
def test_malformed_records_still_fail_loudly(record: Any, expected: str) -> None:
    with pytest.raises(AssertionError) as excinfo:
        idle._build_seed_request(record, context="corpus:1")
    assert expected in str(excinfo.value)
    assert "corpus:1" in str(excinfo.value), "the failure does not say which record"


def test_a_malformed_line_fails_the_whole_corpus_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _write_corpus(
        tmp_path / "mixed.jsonl",
        [
            {"messages": [{"role": "user", "content": "fine"}]},
            {"messages": [{"role": "narrator", "content": "not fine"}]},
        ],
    )
    monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
    with pytest.raises(AssertionError, match="has role 'narrator'"):
        idle._load_prompt_corpus()


# ---------------------------------------------------------------------------
# --workload actually selects traffic, or refuses
# ---------------------------------------------------------------------------
def test_the_default_workload_leaves_the_legacy_corpus_path_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``generic-chat`` is the launcher default, i.e. "nothing was selected"."""
    corpus = _write_corpus(
        tmp_path / "ultrachat.jsonl", [{"messages": [{"role": "user", "content": "hi"}]}]
    )
    monkeypatch.setenv("SPEEDLM_E2E_WORKLOAD", idle.DEFAULT_WORKLOAD)
    monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
    loaded = idle._load_seed_corpus()
    assert loaded is not None
    assert list(loaded[0].messages) == [{"role": "user", "content": "hi"}]


def test_no_workload_and_no_corpus_is_still_the_synthetic_fallback(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SPEEDLM_E2E_WORKLOAD", raising=False)
    monkeypatch.delenv("SPEEDLM_E2E_PROMPT_CORPUS", raising=False)
    assert idle._load_seed_corpus() is None


def test_an_unknown_workload_name_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPEEDLM_E2E_WORKLOAD", "no-such-workload")
    monkeypatch.delenv("SPEEDLM_E2E_PROMPT_CORPUS", raising=False)
    with pytest.raises(AssertionError, match="is not a declared workload"):
        idle._load_seed_corpus()


def test_a_workload_and_a_prompt_corpus_together_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _write_corpus(
        tmp_path / "ultrachat.jsonl", [{"messages": [{"role": "user", "content": "hi"}]}]
    )
    monkeypatch.setenv("SPEEDLM_E2E_WORKLOAD", AGENTIC_WORKLOAD)
    monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
    with pytest.raises(AssertionError, match="are both set"):
        idle._load_seed_corpus()


def test_a_workload_larger_than_the_engine_window_is_refused(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The silent-truncation guard: 23552 tokens of workload, 4096 of engine.

    Enforced through ``workloads.preflight_refusals`` -- the same function the
    launch preflight uses -- so the runtime check and the gate cannot drift.
    """
    monkeypatch.setenv("SPEEDLM_E2E_WORKLOAD", AGENTIC_WORKLOAD)
    monkeypatch.delenv("SPEEDLM_E2E_PROMPT_CORPUS", raising=False)
    monkeypatch.setenv(
        "SPEEDLM_E2E_VLLM_ARGS", json.dumps(["--max-model-len", "4096"])
    )
    with pytest.raises(AssertionError) as excinfo:
        idle._load_seed_corpus()
    assert "23552" in str(excinfo.value)


def test_an_unbounded_engine_cannot_carry_a_workload(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``--max-model-len`` means the requirement cannot be checked at all."""
    monkeypatch.setenv("SPEEDLM_E2E_WORKLOAD", AGENTIC_WORKLOAD)
    monkeypatch.delenv("SPEEDLM_E2E_PROMPT_CORPUS", raising=False)
    monkeypatch.setenv("SPEEDLM_E2E_VLLM_ARGS", json.dumps(["--enforce-eager"]))
    with pytest.raises(AssertionError, match="declares no --max-model-len"):
        idle._load_seed_corpus()


def _agentic_corpus_present() -> bool:
    try:
        return W.load_spec(AGENTIC_WORKLOAD).source_path.is_file()
    except W.WorkloadError:  # pragma: no cover - manifest missing on this host
        return False


@pytest.mark.skipif(
    not _agentic_corpus_present(),
    reason="the agentic-mixed-outcome records file is not on this host",
)
def test_the_real_agentic_workload_loads_with_its_tools(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end the whole change exists for: real agentic traffic, selected by name."""
    monkeypatch.setenv("SPEEDLM_E2E_WORKLOAD", AGENTIC_WORKLOAD)
    monkeypatch.delenv("SPEEDLM_E2E_PROMPT_CORPUS", raising=False)
    monkeypatch.setenv(
        "SPEEDLM_E2E_VLLM_ARGS", json.dumps(["--max-model-len", "24576"])
    )
    loaded = idle._load_seed_corpus()
    assert loaded is not None and len(loaded) > 100
    assert loaded[0].messages[0]["role"] == "system", (
        "record 1 of this corpus starts with a system prompt -- the shape the old "
        "loader asserted could not happen"
    )
    assert all(request.tools for request in loaded[:50]), "tool schemas were dropped"

    payload = _sent_payload(monkeypatch, loaded[0])
    assert payload["messages"] == [dict(m) for m in loaded[0].messages]
    assert payload["tools"] == [dict(t) for t in loaded[0].tools]
