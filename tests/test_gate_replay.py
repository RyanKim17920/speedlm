"""CPU-only tests for held-out suite replay, including its concurrency."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from speedlm.config import SamplingConfig
from speedlm.gate.replay import ReplayError, replay_suite
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
        if index < self.scripted:
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
) -> _ScriptedClient:
    recorder = _ScriptedClient(choice=choice, usage=usage, scripted=scripted)

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
