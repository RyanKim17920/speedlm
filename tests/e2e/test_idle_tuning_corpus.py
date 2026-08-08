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


#: Every terminal state a live server may hand back on a 200 that ``_post_chat``
#: must go on from.  Imported, not spelled out, so a spelling added to the gate's
#: vocabulary is exercised here the same day it is added.
ACCEPTED_FINISH_REASONS: list[str] = sorted(
    idle.NATURAL_STOP_FINISH_REASONS | idle.TRUNCATED_FINISH_REASONS
)

#: A tool-dispatch choice exactly as an OpenAI/vLLM server returns one:
#: ``content: null``, a populated ``tool_calls`` array, and the finish reason
#: that goes with it.  This is the response the harness now provokes itself.
TOOL_DISPATCH_CHOICE: dict[str, Any] = {
    "index": 0,
    "message": {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "chatcmpl-tool-0001",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command": "pytest -q"}'},
            }
        ],
    },
    "finish_reason": "tool_calls",
}


def _sent_payload(
    monkeypatch: pytest.MonkeyPatch,
    request: idle.SeedRequest,
    *,
    choice: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Post one request through ``_post_chat`` and return the JSON body sent.

    ``choice`` is what the server answers with.  It used to be hardcoded to
    ``{"finish_reason": "stop"}``, which made this stub structurally unable to
    catch the defect the very same commit introduced: the payload above began
    shipping the record's tool schemas, so a live server could answer
    ``tool_calls``, and ``_post_chat`` asserted ``finish in ("stop", "length")``
    and killed the GPU run on the first seed that picked a tool.  Every test
    here stubbed that away.  A stub that can only produce the passing case is
    not a test of the assertion it flows through.
    """
    captured: dict[str, Any] = {}
    body_choice = {"finish_reason": "stop"} if choice is None else choice

    class _Response:
        status_code = 200
        text = "ok"

        @staticmethod
        def json() -> dict[str, Any]:
            return {
                "id": "cmpl-1",
                "choices": [body_choice],
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
            {
                "role": "tool",
                "content": "1 failed",
                "name": "bash",
                # The id pairing this result to the call above.  A server
                # rejects the turn without it; the loader used to not care.
                "tool_call_id": "c1",
            },
        ]
    }
    corpus = _write_corpus(tmp_path / "toolloop.jsonl", [record])
    monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
    loaded = idle._load_prompt_corpus()
    assert loaded is not None
    assert [m["role"] for m in loaded[0].messages] == ["user", "assistant", "tool"]


# ---------------------------------------------------------------------------
# What the server is allowed to answer with
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("finish_reason", ACCEPTED_FINISH_REASONS)
def test_post_chat_accepts_every_terminal_state_a_server_may_return(
    monkeypatch: pytest.MonkeyPatch, finish_reason: str
) -> None:
    """The whole accepted vocabulary, through the real ``_post_chat``.

    ``tool_calls`` is the entry that was actually killing runs, but the point of
    parametrising over the imported sets rather than listing spellings is that
    the next one cannot be missed either.  ``repetition`` is here for the same
    reason: the pinned vLLM emits it on a 200.
    """
    idle_request = idle._user_request("hello")

    payload = _sent_payload(
        monkeypatch,
        idle_request,
        choice={"message": {"content": "hi"}, "finish_reason": finish_reason},
    )

    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_post_chat_accepts_a_real_tool_dispatch_response(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exact round trip that aborted the run, end to end.

    An agentic record goes out *with its tool schemas* -- which is what makes
    the server able to answer this way -- and the server answers with a genuine
    dispatch: ``content: null``, a populated ``tool_calls`` array,
    ``finish_reason: "tool_calls"``.  That is a complete generation, and the
    first seed to produce one used to raise AssertionError and take the whole
    GPU run with it.
    """
    record = {
        "messages": [
            {"role": "system", "content": "You are a careful engineer."},
            {"role": "user", "content": "run the tests"},
        ],
        "tools": [TOOL_SCHEMA],
    }
    corpus = _write_corpus(tmp_path / "agentic.jsonl", [record])
    monkeypatch.setenv("SPEEDLM_E2E_PROMPT_CORPUS", str(corpus))
    loaded = idle._load_prompt_corpus()
    assert loaded is not None

    payload = _sent_payload(monkeypatch, loaded[0], choice=TOOL_DISPATCH_CHOICE)

    # The provocation and the response both have to be real for this to mean
    # anything: the schemas went out, so the dispatch coming back is earned.
    assert payload["tools"] == [TOOL_SCHEMA]


@pytest.mark.parametrize(
    "finish_reason",
    ["error", "abort", "", None, "banana"],
    ids=["error", "abort", "blank", "absent", "unknown"],
)
def test_post_chat_still_rejects_a_genuinely_bad_terminal_state(
    monkeypatch: pytest.MonkeyPatch, finish_reason: object
) -> None:
    """Widening the vocabulary must not turn the check into "accept anything".

    ``error`` is vLLM's internal failure (normally a 500, so seeing it on a 200
    means something is badly wrong), ``abort`` means the request was cancelled,
    and an unrecognised or missing value is a server this harness has never been
    validated against.  All four are reasons to stop the run and say so, which
    is the half of the old assertion worth keeping.
    """
    with pytest.raises(AssertionError, match="unexpected finish_reason"):
        _sent_payload(
            monkeypatch,
            idle._user_request("hello"),
            choice={"message": {"content": "hi"}, "finish_reason": finish_reason},
        )


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
            "carries neither 'content' nor a tool call",
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
        # ── everything below reached the wire verbatim under the old check ──
        # ``content`` was tested for key PRESENCE only, so any JSON value at all
        # satisfied "this message has something to send".
        pytest.param(
            {"messages": [{"role": "user", "content": 42}]},
            "has 'content' of type int",
            id="content-a-number",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": {"text": "hi"}}]},
            "has 'content' of type dict",
            id="content-a-bare-object",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": []}]},
            "has 'content' of type list",
            id="content-an-empty-list",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": ["hi"]}]},
            "is not a content part object",
            id="content-part-not-an-object",
        ),
        pytest.param(
            {"messages": [{"role": "user", "content": [{"text": "hi"}]}]},
            "is not a content part object",
            id="content-part-without-a-type",
        ),
        # ``tool_calls`` only had to be truthy, so a bare string claimed to be a
        # tool dispatch and the message passed with no content at all.
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "tool_calls": "yes"},
                ]
            },
            "has 'tool_calls' of type str, not a list",
            id="tool-calls-a-truthy-scalar",
        ),
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "tool_calls": [{"id": "c1"}]},
                ]
            },
            "no 'function' object naming a function",
            id="tool-call-dispatching-nothing",
        ),
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {
                        "role": "assistant",
                        "tool_calls": [{"function": {"name": "bash"}}],
                    },
                ]
            },
            "no non-empty string 'id'",
            id="tool-call-nothing-can-answer",
        ),
        # A ``tool`` turn needed no ``tool_call_id``; the server cannot pair it.
        pytest.param(
            {
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "tool", "content": "1 failed"},
                ]
            },
            "no non-empty string 'tool_call_id'",
            id="tool-result-answering-nothing",
        ),
        # A tool schema needed only to be a dict, so ``{}`` was a tool.
        pytest.param(
            {"messages": [{"role": "user", "content": "hi"}], "tools": [{}]},
            "is not a {'type': 'function', 'function': {...}} schema",
            id="tool-schema-empty",
        ),
        pytest.param(
            {
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {}}],
            },
            "has no non-empty string 'name'",
            id="tool-schema-nameless",
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


#: The generic-chat corpus the three archived runs and every number in
#: ``docs/benchmark-evidence.md`` were measured against.
ULTRACHAT_CORPUS = Path("/data/ryan.kim/speedlm-corpora/ultrachat-prompts.jsonl")


@pytest.mark.skipif(
    not _agentic_corpus_present() or not ULTRACHAT_CORPUS.is_file(),
    reason="one of the real corpora is not on this host",
)
def test_both_real_corpora_load_without_a_single_rejection() -> None:
    """The other half of tightening validation: it must not reject real traffic.

    Every case added to ``test_malformed_records_still_fail_loudly`` narrows
    what ``_build_seed_request`` accepts, and a validator can always be made
    stricter by rejecting things that are actually fine.  These two files are
    the ground truth for what "fine" means -- the agentic workload this loader
    was widened to carry, and the generic-chat corpus the archived benchmark
    numbers were measured on.  Every record in both has to survive, or the
    tightening has broken the thing it was protecting.
    """
    agentic = W.load_spec(AGENTIC_WORKLOAD).source_path
    for path in (agentic, ULTRACHAT_CORPUS):
        loaded = 0
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                idle._build_seed_request(
                    json.loads(line), context=f"{path}:{number}"
                )
                loaded += 1
        assert loaded > 100, f"{path} contributed almost nothing to this check"
