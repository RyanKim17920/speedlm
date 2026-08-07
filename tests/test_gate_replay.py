"""CPU-only tests for held-out suite replay, including its concurrency."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from speedlm.config import SamplingConfig
from speedlm.gate.replay import (
    TRUNCATED_FINISH_REASONS,
    ReplayError,
    ReplayResult,
    RunResults,
    is_truncated_finish_reason,
    replay_suite,
)
from speedlm.gate.suite import BenchmarkSuite, FrozenContext


def _suite(count: int) -> BenchmarkSuite:
    contexts = tuple(
        FrozenContext(
            context_hash=f"hash-{index:03d}",
            messages=({"role": "user", "content": f"prompt {index}"},),
            seed=0,
            temperature=0.0,
            top_p=1.0,
        )
        for index in range(count)
    )
    return BenchmarkSuite(suite_hash="suite-hash", contexts=contexts)


@dataclass
class _FakeResponse:
    body: dict[str, Any]
    status_code: int = 200

    @property
    def text(self) -> str:
        return ""

    def json(self) -> dict[str, Any]:
        return self.body


@dataclass
class _RecordingClient:
    """Async client that records concurrency and the order requests arrive in.

    Each request yields to the loop once before completing, which is what makes
    overlap observable: without a suspension point every coroutine would run to
    completion the moment it was scheduled and ``peak_in_flight`` would read 1
    even under a wide semaphore.
    """

    limits: Any = None
    in_flight: int = 0
    peak_in_flight: int = 0
    arrivals: list[str] = field(default_factory=list)
    #: Extra loop turns each request waits, keyed by prompt suffix, so a
    #: deliberately slow request can be made to finish out of arrival order.
    delays: dict[str, int] = field(default_factory=dict)
    #: Every request body sent, in arrival order.
    payloads: list[dict[str, Any]] = field(default_factory=list)
    #: Per-token logprob entries to return, or ``None`` to return none at all.
    logprob_tokens: list[str] | None = None

    async def __aenter__(self) -> _RecordingClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any]) -> _FakeResponse:
        prompt = json["messages"][0]["content"]
        self.arrivals.append(prompt)
        self.payloads.append(json)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        for _ in range(1 + self.delays.get(prompt, 0)):
            await asyncio.sleep(0)
        self.in_flight -= 1
        choice: dict[str, Any] = {
            "message": {"content": f"reply to {prompt}"},
            "finish_reason": "stop",
        }
        if self.logprob_tokens is not None:
            choice["logprobs"] = {
                "content": [{"token": t, "logprob": 0.0} for t in self.logprob_tokens]
            }
        return _FakeResponse(
            {
                "choices": [choice],
                "usage": {"prompt_tokens": 3, "completion_tokens": 7},
            }
        )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> _RecordingClient:
    recorder = _RecordingClient()

    def factory(**kwargs: Any) -> _RecordingClient:
        recorder.limits = kwargs.get("limits")
        return recorder

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return recorder


def _replay(suite: BenchmarkSuite, **kwargs: Any) -> Any:
    return asyncio.run(
        replay_suite(suite, "http://endpoint.test/", SamplingConfig(), **kwargs)
    )


def test_default_replay_is_serial(client: _RecordingClient) -> None:
    _replay(_suite(6))

    assert client.peak_in_flight == 1


def test_concurrency_puts_that_many_requests_in_flight(
    client: _RecordingClient,
) -> None:
    _replay(_suite(24), concurrency=8)

    assert client.peak_in_flight == 8


def test_concurrency_never_exceeds_the_configured_degree(
    client: _RecordingClient,
) -> None:
    _replay(_suite(24), concurrency=3)

    assert client.peak_in_flight == 3


def test_pool_limits_match_the_configured_degree(client: _RecordingClient) -> None:
    _replay(_suite(4), concurrency=5)

    assert client.limits is not None
    assert client.limits.max_connections == 5
    assert client.limits.max_keepalive_connections == 5


def test_concurrency_replays_exactly_the_same_contexts(
    client: _RecordingClient,
) -> None:
    """Concurrency changes speed, never which contexts are replayed."""
    suite = _suite(12)

    serial = _replay(suite, concurrency=1)
    serial_hashes = [r.context_hash for r in serial.run_results[0].results]

    client.arrivals.clear()
    concurrent = _replay(suite, concurrency=6)
    concurrent_hashes = [r.context_hash for r in concurrent.run_results[0].results]

    assert concurrent_hashes == serial_hashes
    assert sorted(concurrent_hashes) == sorted(c.context_hash for c in suite.contexts)


def test_results_stay_in_suite_order_when_completion_order_differs(
    client: _RecordingClient,
) -> None:
    """The per-request array is evidence; it must stay aligned with the suite.

    The first context is made much slower than the rest, so it completes last.
    Recording results in completion order would put it at the end.
    """
    suite = _suite(5)
    client.delays["prompt 0"] = 50

    result = _replay(suite, concurrency=5)

    assert client.arrivals[0] == "prompt 0"
    assert [r.context_hash for r in result.run_results[0].results] == [
        context.context_hash for context in suite.contexts
    ]
    assert result.run_results[0].results[0].response_text == "reply to prompt 0"


def test_aggregates_are_identical_across_concurrency(
    client: _RecordingClient,
) -> None:
    suite = _suite(9)

    serial = _replay(suite, concurrency=1).run_results[0]
    concurrent = _replay(suite, concurrency=4).run_results[0]

    assert concurrent.total_completion_tokens == serial.total_completion_tokens
    assert concurrent.total_prompt_tokens == serial.total_prompt_tokens
    assert concurrent.valid_count == serial.valid_count
    assert concurrent.invalid_count == serial.invalid_count
    assert concurrent.invalid_rate == serial.invalid_rate


def test_repeats_stay_sequential(client: _RecordingClient) -> None:
    """Repeats are separate samples and must not overlap each other."""
    suite = _suite(4)

    result = _replay(suite, repeats=3, concurrency=4)

    assert result.num_runs == 3
    assert client.peak_in_flight == 4
    assert len(client.arrivals) == 12


@pytest.mark.parametrize("concurrency", [0, -1, True, 1.5])
def test_invalid_concurrency_is_rejected(
    client: _RecordingClient,
    concurrency: object,
) -> None:
    with pytest.raises(ReplayError, match="concurrency"):
        _replay(_suite(2), concurrency=concurrency)


# ---------------------------------------------------------------------------
# Bounded output and token capture, for the correctness pass
# ---------------------------------------------------------------------------


def test_throughput_replay_sends_no_output_cap_and_asks_for_no_logprobs(
    client: _RecordingClient,
) -> None:
    """The default shape is unchanged: capping output would change tok/s."""
    _replay(_suite(2))

    assert all("max_tokens" not in body for body in client.payloads)
    assert all("logprobs" not in body for body in client.payloads)


def test_max_tokens_bounds_every_request(client: _RecordingClient) -> None:
    _replay(_suite(3), max_tokens=128)

    assert [body["max_tokens"] for body in client.payloads] == [128, 128, 128]


def test_capture_tokens_asks_for_logprobs_but_not_alternatives(
    client: _RecordingClient,
) -> None:
    _replay(_suite(2), capture_tokens=True)

    assert all(body["logprobs"] is True for body in client.payloads)
    assert all("top_logprobs" not in body for body in client.payloads)


def test_captured_tokens_reach_the_result(client: _RecordingClient) -> None:
    client.logprob_tokens = ["Hel", "lo", " world"]

    result = _replay(_suite(1), capture_tokens=True)

    request = result.run_results[0].results[0]
    assert request.output_tokens == ("Hel", "lo", " world")
    assert request.finish_reason == "stop"
    assert request.to_dict()["output_tokens"] == ["Hel", "lo", " world"]


def test_missing_logprobs_leave_the_token_sequence_uncaptured(
    client: _RecordingClient,
) -> None:
    """Empty means "not captured", never "the model emitted nothing"."""
    result = _replay(_suite(1), capture_tokens=True)

    assert result.run_results[0].results[0].output_tokens == ()
    assert result.run_results[0].results[0].response_text != ""


@pytest.mark.parametrize("max_tokens", [0, -1, True, 1.5])
def test_invalid_max_tokens_is_rejected(
    client: _RecordingClient,
    max_tokens: object,
) -> None:
    with pytest.raises(ReplayError, match="max_tokens"):
        _replay(_suite(2), max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# What "valid" means: reasoning models truncated at the cap
# ---------------------------------------------------------------------------


@dataclass
class _ScriptedClient:
    """Async client that returns one caller-supplied choice/usage per request.

    Unlike :class:`_RecordingClient` it does not synthesise a reply, so a test
    can hand it the exact body a served engine produced -- including the shapes
    a reasoning model emits when its ``<think>`` block runs into the cap.
    """

    choice: dict[str, Any]
    usage: dict[str, Any]
    #: Requests answered with ``choice``/``usage``; the rest get a plain reply.
    scripted: int = 0
    #: Requests failed *before* the scripted ones begin.  A failed request is
    #: precisely what ``finish_reason_count`` exists to keep out of the
    #: truncation denominator, so a test has to be able to put failures and
    #: reported responses inside one run.
    failures: int = 0
    #: Which shape of failure to answer those with -- the three ways
    #: ``_send_request`` can come back without ever having heard from the model.
    #: ``"http"``, ``"transport"``, ``"empty_choices"`` -- the three ways the
    #: model is never heard from -- or ``"empty_stop"``, the fourth shape: a
    #: 200 OK the endpoint *did* answer, carrying a finish reason but nothing
    #: to show for it, which ``_validity_error`` scores invalid.  That one is
    #: the only failure that arrives with a finish reason attached, so it is
    #: the only one that can contaminate the truncation counts.
    #:
    #: ``"blank_reason"`` is the fifth shape and the only one here that is not
    #: a failure at all: a healthy 200 OK with real content and real tokens,
    #: whose ``finish_reason`` field is present but says nothing.  It is filed
    #: alongside the failures because it occupies the same slot in a run -- the
    #: one response that is not part of the scripted, wholly-capped majority.
    failure_mode: str = "http"
    #: The blank ``finish_reason`` the ``"blank_reason"`` shape reports.  A
    #: server has several ways to say nothing, and truthiness only distinguishes
    #: one of them, so the exact spelling has to be a test parameter.
    blank_finish_reason: str = " "
    seen: int = 0
    limits: Any = None

    async def __aenter__(self) -> _ScriptedClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def post(self, url: str, *, json: dict[str, Any]) -> _FakeResponse:
        index = self.seen
        self.seen += 1
        await asyncio.sleep(0)
        if index < self.failures:
            if self.failure_mode == "transport":
                raise httpx.ConnectError("connection reset by peer")
            if self.failure_mode == "empty_choices":
                return _FakeResponse({"choices": [], "usage": {}})
            if self.failure_mode == "blank_reason":
                return _FakeResponse(
                    {
                        "choices": [
                            {
                                "message": {"content": "a real answer"},
                                "finish_reason": self.blank_finish_reason,
                            }
                        ],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 9},
                    }
                )
            if self.failure_mode == "empty_stop":
                return _FakeResponse(
                    {
                        "choices": [
                            {"message": {"content": ""}, "finish_reason": "stop"}
                        ],
                        "usage": {"prompt_tokens": 12, "completion_tokens": 9},
                    }
                )
            return _FakeResponse({"error": "internal"}, status_code=500)
        if index < self.failures + self.scripted:
            return _FakeResponse({"choices": [dict(self.choice)], "usage": self.usage})
        return _FakeResponse(
            {
                "choices": [
                    {"message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 7},
            }
        )


def _scripted(
    monkeypatch: pytest.MonkeyPatch,
    choice: dict[str, Any],
    usage: dict[str, Any],
    *,
    scripted: int,
    failures: int = 0,
    failure_mode: str = "http",
    blank_finish_reason: str = " ",
) -> _ScriptedClient:
    recorder = _ScriptedClient(
        choice=choice,
        usage=usage,
        scripted=scripted,
        failures=failures,
        failure_mode=failure_mode,
        blank_finish_reason=blank_finish_reason,
    )

    def factory(**kwargs: Any) -> _ScriptedClient:
        recorder.limits = kwargs.get("limits")
        return recorder

    monkeypatch.setattr(httpx, "AsyncClient", factory)
    return recorder


#: The shape SLURM 369147 produced on 76 of its 103 held-out contexts: Qwen3-8B
#: spent the whole ``benchmark_max_tokens`` budget inside ``<think>``, so vLLM
#: returned the generated text under ``message.reasoning`` and left ``content``
#: null, with ``finish_reason`` "length" and ``completion_tokens`` at the cap.
_TRUNCATED_REASONING_CHOICE: dict[str, Any] = {
    "message": {"content": None, "reasoning": "Okay, let's think about this. " * 40},
    "finish_reason": "length",
}
_TRUNCATED_REASONING_USAGE: dict[str, Any] = {
    "prompt_tokens": 209,
    "completion_tokens": 512,
}


def test_truncated_reasoning_is_a_valid_throughput_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for SLURM 369147's ``high_invalid_rate`` rejection.

    76 of 103 contexts hit ``benchmark_max_tokens`` mid-``<think>``, which made
    ``content`` empty and the old predicate marked every one of them invalid --
    ``invalid_rate: 0.7379`` against a 0.1 threshold, on a run whose correctness
    pass reported zero divergences.  The engine generated 512 tokens for each of
    them; that is a good throughput and acceptance sample.
    """
    _scripted(
        monkeypatch,
        _TRUNCATED_REASONING_CHOICE,
        _TRUNCATED_REASONING_USAGE,
        scripted=76,
    )

    run = _replay(_suite(103), concurrency=8, max_tokens=512).run_results[0]

    assert run.invalid_rate == 0.0
    assert run.invalid_count == 0
    assert run.total_completion_tokens == 76 * 512 + 27 * 7


def test_truncated_reasoning_keeps_the_generated_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tokens are evidence; dropping them makes the sample unfalsifiable."""
    _scripted(
        monkeypatch,
        _TRUNCATED_REASONING_CHOICE,
        _TRUNCATED_REASONING_USAGE,
        scripted=1,
    )

    request = _replay(_suite(1), max_tokens=512).run_results[0].results[0]

    assert request.valid
    assert request.finish_reason == "length"
    assert request.reasoning_text.startswith("Okay, let's think about this.")
    assert request.response_text == ""
    assert request.generated_text == request.reasoning_text


def test_reasoning_content_alias_is_recognised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some OpenAI-compatible servers name the field ``reasoning_content``."""
    _scripted(
        monkeypatch,
        {
            "message": {"content": "", "reasoning_content": "thinking hard"},
            "finish_reason": "length",
        },
        {"prompt_tokens": 4, "completion_tokens": 512},
        scripted=1,
    )

    request = _replay(_suite(1), max_tokens=512).run_results[0].results[0]

    assert request.valid
    assert request.reasoning_text == "thinking hard"


def test_a_cap_hit_that_surfaced_nothing_is_still_a_throughput_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tokens were generated and counted; where the server filed them is its business."""
    _scripted(
        monkeypatch,
        {"message": {"content": None}, "finish_reason": "length"},
        {"prompt_tokens": 4, "completion_tokens": 512},
        scripted=1,
    )

    request = _replay(_suite(1), max_tokens=512).run_results[0].results[0]

    assert request.valid
    assert request.error == ""


def test_an_engine_that_generates_nothing_is_still_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The protection ``high_invalid_rate`` exists for must survive the fix."""
    _scripted(
        monkeypatch,
        {"message": {"content": ""}, "finish_reason": "stop"},
        {"prompt_tokens": 4, "completion_tokens": 0},
        scripted=103,
    )

    run = _replay(_suite(103), concurrency=8, max_tokens=512).run_results[0]

    assert run.invalid_rate == 1.0
    assert run.results[0].error == "No generated tokens"


def test_an_empty_response_that_stopped_on_its_own_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``finish_reason`` "stop" with nothing to show is a broken engine, not a cap."""
    _scripted(
        monkeypatch,
        {"message": {"content": ""}, "finish_reason": "stop"},
        {"prompt_tokens": 4, "completion_tokens": 9},
        scripted=1,
    )

    request = _replay(_suite(1), max_tokens=512).run_results[0].results[0]

    assert not request.valid
    assert request.error == "Empty response text"


def test_missing_usage_falls_back_to_the_surfaced_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A server that omits ``usage`` still produced a usable sample."""
    _scripted(
        monkeypatch,
        {"message": {"content": "a real answer"}, "finish_reason": "stop"},
        {},
        scripted=1,
    )

    request = _replay(_suite(1)).run_results[0].results[0]

    assert request.valid
    assert request.completion_tokens == 0


def test_each_invocation_gets_its_own_session_id(client: _RecordingClient) -> None:
    """One id per invocation, shared by its repeats, never reused.

    The gate's divergence criterion reads two results carrying the same
    non-empty id as "collected back-to-back against one live engine, no restart
    in between" -- which is what decides whether its control is a valid null at
    all.  So the stamp has to be minted per call and has to actually differ
    between calls; a constant would make every comparison look same-engine and
    an empty string would make every comparison look unknowable.
    """
    first = _replay(_suite(3), repeats=2)
    second = _replay(_suite(3), repeats=2)

    assert first.session_id
    assert second.session_id
    assert first.session_id != second.session_id
    # Repeats inside one invocation really did share an engine, so they share
    # the id: it is a property of the call, not of the run.
    assert first.num_runs == 2
    assert first.to_dict()["session_id"] == first.session_id


# ---------------------------------------------------------------------------
# Tool schemas reach the endpoint
# ---------------------------------------------------------------------------

_WEATHER_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Look up the weather.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}


def _tool_suite(count: int) -> BenchmarkSuite:
    contexts = tuple(
        FrozenContext(
            context_hash=f"tool-hash-{index:03d}",
            messages=({"role": "user", "content": f"prompt {index}"},),
            seed=0,
            temperature=0.0,
            top_p=1.0,
            tools=(_WEATHER_TOOL,),
        )
        for index in range(count)
    )
    return BenchmarkSuite(suite_hash="tool-suite-hash", contexts=contexts)


def test_tool_schemas_are_sent_with_every_request(
    client: _RecordingClient,
) -> None:
    """Agentic contexts replay with the schemas production offered.

    Without this the replayed prompt renders no tool block, so the model cannot
    dispatch and the gate scores a request that was never served.
    """
    _replay(_tool_suite(3))

    assert [body["tools"] for body in client.payloads] == [[_WEATHER_TOOL]] * 3


def test_chat_requests_carry_no_tools_key(client: _RecordingClient) -> None:
    """Tool-free traffic keeps a byte-identical payload to before."""
    _replay(_suite(3))

    assert all("tools" not in body for body in client.payloads)


# ---------------------------------------------------------------------------
# The reasoning-field list must not drift again
# ---------------------------------------------------------------------------


def test_reasoning_fields_agree_across_the_three_copies() -> None:
    """``_REASONING_FIELDS`` is duplicated; the copies must stay identical.

    This list lives in ``gateway.sse``, ``gateway.capture`` and ``gate.replay``.
    It had already drifted: the gate's copy was missing ``thinking``, so a
    server using that spelling was captured as a reasoning response by the
    gateway and read as an empty one by the gate -- inflating ``invalid_rate``
    and failing a healthy candidate.  See the module comment in ``gate.replay``
    for the shared-constant requirement that would delete this test.
    """
    from speedlm.gate.replay import _REASONING_FIELDS as replay_fields
    from speedlm.gateway.capture import _REASONING_FIELDS as capture_fields
    from speedlm.gateway.sse import _REASONING_FIELDS as sse_fields

    assert set(replay_fields) == set(sse_fields) == set(capture_fields)
    # Order decides which channel wins when a server sends more than one, so
    # the gate must consult them in the same order the capture path did.
    assert tuple(replay_fields) == tuple(sse_fields)


def test_thinking_channel_is_recognised(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server that files the block under ``thinking`` is not a blank reply.

    This is the concrete consequence of the drift: before the fix this response
    surfaced nothing to the gate, so ``finish_reason: "stop"`` with an empty
    ``content`` was scored invalid even though the engine generated 128 tokens.
    """
    _scripted(
        monkeypatch,
        {
            "message": {"content": None, "thinking": "step one, step two"},
            "finish_reason": "stop",
        },
        {"prompt_tokens": 11, "completion_tokens": 128},
        scripted=1,
    )

    request = _replay(_suite(1)).run_results[0].results[0]

    assert request.reasoning_text == "step one, step two"
    assert request.valid
    assert request.error == ""


# ---------------------------------------------------------------------------
# Who chose the generation length: the truncation counting layer
# ---------------------------------------------------------------------------


def _finish_reason_choice(reason: str) -> dict[str, Any]:
    """A healthy, non-empty response that ended for the given reason."""
    return {"message": {"content": "a real answer"}, "finish_reason": reason}


#: Usage for a response that spent the whole budget, i.e. one the harness ended.
_CAPPED_USAGE: dict[str, Any] = {"prompt_tokens": 12, "completion_tokens": 512}


def test_a_length_finish_reason_is_truncated_and_a_stop_is_reported_but_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both shapes are *reported*; only one of them was ended by the harness.

    ``finish_reason_count`` and ``truncated_count`` answer different questions
    and a run has to be able to distinguish them, because the denominator is the
    field that separates "nothing was truncated" from "the endpoint never said".
    """
    _scripted(monkeypatch, _finish_reason_choice("length"), _CAPPED_USAGE, scripted=3)

    run = _replay(_suite(10), max_tokens=512).run_results[0]

    # The seven unscripted requests come back with ``finish_reason: "stop"``.
    assert run.finish_reason_count == 10
    assert run.truncated_count == 3
    assert run.natural_stop_count == 7
    assert run.truncation_rate == pytest.approx(0.3)


def test_a_tool_call_finish_reason_counts_as_a_natural_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool dispatch is the model choosing where to stop, not the harness.

    The whole point of the count is which of the two ended the generation, so
    every non-``length`` reason belongs on the natural-stop side.  Filing
    ``tool_calls`` as a truncation would make an agentic suite -- the traffic
    this gate exists to measure -- read as saturated by construction.
    """
    _scripted(
        monkeypatch,
        {
            "message": {
                "content": "Let me look that up.",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        },
        {"prompt_tokens": 31, "completion_tokens": 24},
        scripted=4,
    )

    run = _replay(_suite(4)).run_results[0]

    assert run.finish_reason_count == 4
    assert run.truncated_count == 0
    assert run.natural_stop_count == 4
    assert run.truncation_rate == 0.0


@pytest.mark.parametrize("failure_mode", ["http", "transport", "empty_choices"])
def test_failed_requests_stay_out_of_the_truncation_denominator(
    monkeypatch: pytest.MonkeyPatch,
    failure_mode: str,
) -> None:
    """A request the model never answered is not evidence about where it stops.

    Ninety of a hundred requests fail and the ten that survive were all ended by
    the cap.  Folding the failures into the denominator would report ``0.10`` --
    a comfortably bounded-looking run -- and would make the reported number fall
    the *more* of the run broke.  The honest answer is that every generation
    this run actually observed was ended by the harness.
    """
    _scripted(
        monkeypatch,
        _finish_reason_choice("length"),
        _CAPPED_USAGE,
        scripted=10,
        failures=90,
        failure_mode=failure_mode,
    )

    run = _replay(_suite(100), max_tokens=512).run_results[0]

    assert len(run.results) == 100
    assert run.invalid_count == 90
    assert run.finish_reason_count == 10
    assert run.truncated_count == 10
    assert run.truncation_rate == 1.0
    # And emphatically not the misleadingly low rate ``len(results)`` gives.
    assert run.truncation_rate != pytest.approx(10 / len(run.results))


def test_a_run_that_reported_no_finish_reason_is_unmeasured_not_untruncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``truncation_rate`` reads 0.0, and the denominator is what says why.

    Both assertions matter together: the rate alone cannot tell a run in which
    nothing was truncated from a run in which nothing was reported, and reading
    the second as the first is exactly what
    :attr:`speedlm.gate.decide.TruncationRegime.UNTESTABLE` exists to stop.
    """
    _scripted(
        monkeypatch,
        {"message": {"content": "a real answer"}},
        {"prompt_tokens": 12, "completion_tokens": 9},
        scripted=5,
    )

    run = _replay(_suite(5)).run_results[0]

    assert run.finish_reason_count == 0
    assert run.truncated_count == 0
    assert run.natural_stop_count == 0
    assert run.truncation_rate == 0.0
    # A measured zero looks identical in the rate, so the count is persisted.
    assert run.to_dict()["finish_reason_count"] == 0
    assert run.to_dict()["truncation_rate"] == 0.0


@pytest.mark.parametrize(
    "raw_finish_reason",
    [7, True, 0.5, ["length"], {"reason": "length"}, None],
    ids=["int", "bool", "float", "list", "dict", "null"],
)
def test_a_non_string_finish_reason_is_recorded_as_absent_not_crashed_on(
    monkeypatch: pytest.MonkeyPatch,
    raw_finish_reason: object,
) -> None:
    """A malformed reason costs one response's evidence, not the whole replay.

    ``choice.get("finish_reason") or ""`` accepted any truthy JSON value, so a
    server answering ``finish_reason: 7`` built a ``RequestResult`` whose
    ``str``-annotated field held an int.  Nothing failed there; the counting
    block in ``_run_single`` then called ``.strip()`` on it and raised
    ``AttributeError`` *outside* ``_send_request``'s ``except``, killing the run.

    The response itself is still a perfectly good throughput sample -- it has
    content and tokens -- so it stays valid.  What it no longer supplies is a
    finish reason, which is the truth: a value the schema cannot express is a
    reason not given, and ``finish_reason_count`` says so.
    """
    _scripted(
        monkeypatch,
        {
            "message": {"content": "a real answer"},
            "finish_reason": raw_finish_reason,
        },
        {"prompt_tokens": 12, "completion_tokens": 9},
        scripted=5,
    )

    run = _replay(_suite(5)).run_results[0]

    assert len(run.results) == 5
    assert run.valid_count == 5
    assert run.invalid_count == 0
    # The annotation is true of every result, whatever the server sent.
    assert all(isinstance(r.finish_reason, str) for r in run.results)
    assert all(r.finish_reason == "" for r in run.results)
    # And an absent reason stays out of the truncation denominator.
    assert run.finish_reason_count == 0
    assert run.truncated_count == 0


def _counted_run(*, reported: int, truncated: int) -> RunResults:
    """A run carrying nothing but the finish-reason counts under test."""
    return RunResults(
        results=(),
        total_latency_s=0.0,
        total_prompt_tokens=0,
        total_completion_tokens=0,
        valid_count=0,
        invalid_count=0,
        invalid_rate=0.0,
        finish_reason_count=reported,
        truncated_count=truncated,
    )


def test_pooled_truncation_rate_weighs_each_repeat_by_what_it_reported() -> None:
    """Pooled over reported responses, never a mean of per-repeat rates.

    The two answers only coincide when every repeat reported the same number of
    responses, which is precisely the case a broken repeat is not.  Here a
    ten-response repeat that was barely truncated sits beside a two-response one
    that was wholly truncated; pooling says ``3/12``, while averaging the two
    rates would let the near-empty repeat carry half the answer and say ``0.55``.
    """
    result = ReplayResult(
        run_results=(
            _counted_run(reported=10, truncated=1),
            _counted_run(reported=2, truncated=2),
        ),
        num_runs=2,
        suite_hash="suite-hash",
    )

    assert result.total_finish_reason_count == 12
    assert result.total_natural_stops == 9
    assert result.avg_truncation_rate == pytest.approx(3 / 12)
    # The discriminating half: mean-of-means is a different number here.
    assert result.avg_truncation_rate != pytest.approx((0.1 + 1.0) / 2)


def test_an_invalid_response_cannot_supply_the_only_natural_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One broken response must not vouch for a wholly-capped run.

    Ninety-nine healthy generations spent their whole budget and one response
    came back empty while claiming ``finish_reason: "stop"``.  Counting every
    truthy finish reason made that single failure the run's one natural stop,
    which classifies ``MIXED`` instead of ``SATURATED`` and promotes -- while
    ``invalid_rate`` sat at 0.01, an order of magnitude under its own 0.1
    threshold, so no other guard could see it either.  A response the run
    failed to obtain describes the failure, not where this model stops.
    """
    _scripted(
        monkeypatch,
        _finish_reason_choice("length"),
        _CAPPED_USAGE,
        scripted=99,
        failures=1,
        failure_mode="empty_stop",
    )

    run = _replay(_suite(100), concurrency=8, max_tokens=512).run_results[0]

    # The contaminating response really is present, really did report "stop",
    # and really is under the invalid-rate threshold -- otherwise this test
    # would be asserting against a population that cannot occur.
    assert run.invalid_count == 1
    assert run.invalid_rate == pytest.approx(0.01)
    assert [r.finish_reason for r in run.results if not r.valid] == ["stop"]

    assert run.natural_stop_count == 0
    assert run.finish_reason_count == 99
    assert run.truncated_count == 99
    assert run.truncation_rate == 1.0


@pytest.mark.parametrize("blank", [" ", "\t", "\n", "  \t ", ""])
def test_a_blank_finish_reason_cannot_supply_the_only_natural_stop(
    monkeypatch: pytest.MonkeyPatch,
    blank: str,
) -> None:
    """A finish reason that says nothing is a reason not given.

    The sibling above keeps a *failed* response out of the denominator; this
    pins the other way the denominator could be contaminated, by a response
    that is entirely healthy and simply reported no reason.  The counting
    filter tested truthiness (``r.valid and r.finish_reason``), and a
    whitespace-only string is truthy in Python: a single response reporting
    ``" "`` entered the denominator, failed the truncation test, and became the
    one natural stop that turns ``SATURATED`` into ``MIXED`` -- promoting a run
    in which the cap had ended all ninety-nine generations it actually
    measured.  ``invalid_rate`` cannot see this either, because there is
    nothing invalid here at all.

    Parametrised over the spellings of "nothing" a server can send, because
    truthiness rejects only one of them.  ``""`` is included as the boundary
    the old filter did get right; the whitespace cases are the defect.
    """
    _scripted(
        monkeypatch,
        _finish_reason_choice("length"),
        _CAPPED_USAGE,
        scripted=99,
        failures=1,
        failure_mode="blank_reason",
        blank_finish_reason=blank,
    )

    run = _replay(_suite(100), concurrency=8, max_tokens=512).run_results[0]

    # The premise: the blank-reason response really is present, really is
    # *valid*, and really did report the blank spelling under test -- otherwise
    # this would be asserting against a population that cannot occur.
    assert run.invalid_count == 0
    assert [r.finish_reason for r in run.results if not r.finish_reason.strip()] == [
        blank
    ]
    assert not is_truncated_finish_reason(blank)

    assert run.finish_reason_count == 99
    assert run.truncated_count == 99
    assert run.natural_stop_count == 0
    assert run.truncation_rate == 1.0


def test_the_gate_truncation_vocabulary_matches_the_training_filter() -> None:
    """The three copies of the truncated-finish-reason set must stay identical.

    ``gate.replay`` restates the set rather than importing it, because the gate
    must not depend on the training backend.  Restating it is what lets it
    drift: the gate's copy once held ``length`` alone, so an endpoint speaking
    the Responses API spelling ``incomplete`` -- which this project's own
    gateway emits -- reported zero truncations on a wholly-capped run and the
    run classified ``BOUNDED``, the strongest possible "the cap was not
    binding" reading of the strongest possible saturation.  This is the gate
    side of ``test_trace_store.py::test_truncated_finish_reasons_match_the_
    training_filter``; the two together pin all three copies.
    """
    from speedlm.traces import store
    from speedlm.training.backends import eagle3

    assert TRUNCATED_FINISH_REASONS == store.TRUNCATED_FINISH_REASONS
    assert TRUNCATED_FINISH_REASONS == eagle3.TRUNCATED_FINISH_REASONS
    # Spelled out as well as cross-checked, so deleting a member from all three
    # copies at once still fails here rather than silently agreeing.
    assert frozenset({"length", "incomplete"}) == TRUNCATED_FINISH_REASONS


def test_an_incomplete_finish_reason_makes_a_run_wholly_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``incomplete`` is the Responses API spelling of the identical event.

    A run in which the cap ended every single generation must read as wholly
    truncated no matter which of the two spellings the server used.  Counting
    only ``length`` turned this exact run into zero truncations and five
    natural stops -- ``BOUNDED``, and promotable.
    """
    _scripted(
        monkeypatch,
        _finish_reason_choice("incomplete"),
        _CAPPED_USAGE,
        scripted=5,
    )

    run = _replay(_suite(5), max_tokens=512).run_results[0]

    assert run.finish_reason_count == 5
    assert run.truncated_count == 5
    assert run.natural_stop_count == 0
    assert run.truncation_rate == 1.0


#: The shape a Responses-API server returns when the cap ended a generation
#: that had surfaced nothing: tokens counted in ``usage``, no content, no
#: reasoning channel, and the Responses-API spelling of "the cap ended this".
_EMPTY_CAPPED_USAGE: dict[str, Any] = {"prompt_tokens": 12, "completion_tokens": 512}


@pytest.mark.parametrize(
    "spelling", ["length", "incomplete", "Incomplete", " incomplete\n"]
)
def test_an_empty_response_the_cap_ended_is_valid_however_it_is_spelled(
    monkeypatch: pytest.MonkeyPatch,
    spelling: str,
) -> None:
    """Validity asked ``finish_reason == "length"``, so ``incomplete`` failed it.

    Rule 3 of ``_validity_error`` -- "nothing surfaced but the cap was hit" --
    was written against the chat-completions spelling alone, while the counting
    layer above it already understood both.  The consequence is worse than a
    mislabel: an arm whose every response came back empty at ``incomplete``
    scored *invalid*, dropped out of the truncation counts entirely (only valid
    responses enter the denominator), and the run rejected as
    ``HIGH_INVALID_RATE``.  ``TRUNCATION_SATURATED`` could therefore never fire
    for the exact response shape it was written to catch.

    Normalised spellings are included because the validity check and the
    counting predicate must agree on the same vocabulary; an equality test
    against a literal agrees with neither ``Incomplete`` nor ``" length"``.
    """
    _scripted(
        monkeypatch,
        {"message": {"content": ""}, "finish_reason": spelling},
        _EMPTY_CAPPED_USAGE,
        scripted=5,
    )

    run = _replay(_suite(5), max_tokens=512).run_results[0]

    assert run.invalid_count == 0
    assert run.invalid_rate == 0.0
    assert run.results[0].error == ""
    # And having stayed valid, they reach the counts as what they are.
    assert run.finish_reason_count == 5
    assert run.truncated_count == 5
    assert run.natural_stop_count == 0
    assert run.truncation_rate == 1.0


def test_an_empty_response_is_valid_only_because_the_cap_ended_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The contrast that makes the fix a widening and not a weakening.

    Two responses identical in every respect -- empty ``content``, no reasoning
    channel, 512 generated tokens -- differing only in who ended them.  The one
    the cap ended is a usable throughput sample; the one claiming it stopped of
    its own accord with nothing to show is a broken engine, and must stay
    invalid.  Accepting ``incomplete`` by widening the test to "any finish
    reason at all" would have swallowed this second case, which is precisely
    the failure ``high_invalid_rate`` exists to catch.
    """
    _scripted(
        monkeypatch,
        {"message": {"content": ""}, "finish_reason": "incomplete"},
        _EMPTY_CAPPED_USAGE,
        scripted=1,
    )
    capped = _replay(_suite(1), max_tokens=512).run_results[0].results[0]

    _scripted(
        monkeypatch,
        {"message": {"content": ""}, "finish_reason": "stop"},
        _EMPTY_CAPPED_USAGE,
        scripted=1,
    )
    self_stopped = _replay(_suite(1), max_tokens=512).run_results[0].results[0]

    assert capped.valid
    assert capped.error == ""
    assert not self_stopped.valid
    assert self_stopped.error == "Empty response text"


@pytest.mark.parametrize("spelling", ["Length", " length ", "INCOMPLETE", "Incomplete"])
def test_finish_reason_spelling_is_normalised_before_counting(
    monkeypatch: pytest.MonkeyPatch,
    spelling: str,
) -> None:
    """Case and surrounding whitespace are the server's, not evidence.

    Normalised exactly as ``traces.store`` normalises it, so a server spelling
    it ``Length`` is not silently read as a natural stop by the gate while the
    training filter drops the same row.
    """
    _scripted(monkeypatch, _finish_reason_choice(spelling), _CAPPED_USAGE, scripted=4)

    run = _replay(_suite(4), max_tokens=512).run_results[0]

    assert run.finish_reason_count == 4
    assert run.truncated_count == 4
    assert run.natural_stop_count == 0


def test_an_unknown_finish_reason_is_not_counted_as_truncated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The predicate is a positive list, never "anything that is not ``stop``".

    A reason nobody has taught the gate about is not evidence that the cap
    ended the generation, and reading it as one would let the guard fire on any
    server whose vocabulary is merely wider than this one's -- turning a
    calibrated measurement check into a veto on unfamiliar endpoints.
    """
    _scripted(monkeypatch, _finish_reason_choice("weird"), _CAPPED_USAGE, scripted=6)

    run = _replay(_suite(6), max_tokens=512).run_results[0]

    assert run.finish_reason_count == 6
    assert run.truncated_count == 0
    assert run.natural_stop_count == 6
    assert run.truncation_rate == 0.0


@pytest.mark.parametrize(
    ("value", "truncated"),
    [
        ("length", True),
        ("incomplete", True),
        (" Length\n", True),
        ("stop", False),
        ("tool_calls", False),
        ("content_filter", False),
        ("weird", False),
        ("", False),
        (None, False),
    ],
)
def test_is_truncated_finish_reason_is_a_positive_membership_test(
    value: str | None,
    truncated: bool,
) -> None:
    """The property behind the counts, stated directly on the predicate.

    Pinning both sides keeps the guard from generalising itself: the truthy
    cases stop ``incomplete`` from being dropped again, and the falsy ones stop
    the predicate from being rewritten as ``value != "stop"``, which would file
    an agentic suite's ``tool_calls`` dispatches as harness truncations and
    make realistic traffic read as saturated by construction.
    """
    assert is_truncated_finish_reason(value) is truncated
