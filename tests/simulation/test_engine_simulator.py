"""The simulator is itself under test.

A simulated engine that nobody checked is not evidence.  These tests pin the
properties every downstream scenario relies on: the exposition parses to the
numbers that were dialled in, counters are monotonic within an engine lifetime
and reset across one, and each injectable fault actually manifests over HTTP.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

import pytest

from simulation.engine import (
    DraftProfile,
    EngineFaults,
    SimulatedEngine,
    running_engine,
)
from simulation.prometheus import CounterState, render_exposition
from speedlm.gate.metrics import (
    CounterResetError,
    compute_delta,
    parse_metrics,
)


def _get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def _post(
    url: str, payload: dict[str, object], timeout: float = 10.0
) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


class TestExposition:
    def test_rendered_body_parses_to_the_counters_it_was_built_from(self) -> None:
        state = CounterState(
            generated_tokens=1234.0,
            prompt_tokens=5678.0,
            decode_time_seconds=12.5,
            drafted_tokens=1000.0,
            accepted_tokens=400.0,
            num_drafts=250.0,
            num_requests_running=2.0,
            num_requests_waiting=1.0,
            finished_requests=40.0,
        )
        snapshot = parse_metrics(render_exposition(state))

        assert snapshot.generated_tokens == 1234.0
        assert snapshot.prompt_tokens == 5678.0
        # The histogram's ``_count`` and ``_bucket`` siblings must not leak
        # into the ``_sum`` field the gate reads as decode wall-time.
        assert snapshot.decode_time_seconds == 12.5
        assert snapshot.drafted_tokens == 1000.0
        assert snapshot.accepted_tokens == 400.0
        assert snapshot.num_drafts == 250.0
        assert snapshot.has_draft_counters is True
        assert snapshot.acceptance_rate == pytest.approx(0.40)
        assert snapshot.mean_accepted_length == pytest.approx(1.0 + 400.0 / 250.0)
        assert snapshot.output_tok_per_sec == pytest.approx(1234.0 / 12.5)

    def test_waiting_by_reason_breakdown_is_not_folded_into_the_gauge(self) -> None:
        # Real vLLM publishes a labelled breakdown whose members sum to the
        # plain gauge.  A parser that aggregated every matching name would
        # double this; the gate's does not, and the simulated body is shaped so
        # the difference is observable.
        snapshot = parse_metrics(render_exposition(CounterState(num_requests_waiting=3.0)))
        assert snapshot.num_requests_waiting == 3.0

    def test_omitting_spec_counters_makes_acceptance_unavailable(self) -> None:
        body = render_exposition(
            CounterState(generated_tokens=10.0, decode_time_seconds=1.0),
            include_spec_decode=False,
        )
        snapshot = parse_metrics(body)
        assert snapshot.has_draft_counters is False
        assert "spec_decode" not in body


class TestEngineHttpSurface:
    def test_serves_every_endpoint_production_actually_calls(self) -> None:
        with running_engine() as engine:
            status, _ = _get(f"{engine.url}/health")
            assert status == 200

            status, body = _get(f"{engine.url}/metrics")
            assert status == 200
            assert b"vllm:spec_decode_num_draft_tokens_total" in body

            status, _ = _post(f"{engine.url}/sleep", {"level": 1})
            assert status == 200
            status, raw = _get(f"{engine.url}/is_sleeping")
            assert json.loads(raw)["is_sleeping"] is True

            status, _ = _post(f"{engine.url}/wake_up", {})
            assert status == 200
            status, raw = _get(f"{engine.url}/is_sleeping")
            assert json.loads(raw)["is_sleeping"] is False

            status, _ = _post(
                f"{engine.url}/collective_rpc", {"method": "reload_weights"}
            )
            assert status == 200
            assert "collective_rpc:reload_weights" in engine.journal.events

    def test_routes_ignore_an_empty_query_string(self) -> None:
        # The gateway proxy builds its upstream URL with
        # ``httpx.URL.copy_with(query=b"")``, which renders as ``/health?``.
        # A handler that compares the raw request target 404s on every proxied
        # request -- which is exactly what the shipped journey fixture did.
        with running_engine() as engine:
            assert _get(f"{engine.url}/health?")[0] == 200
            assert _get(f"{engine.url}/metrics?")[0] == 200

    def test_completion_reports_usage_and_optional_token_stream(self) -> None:
        with running_engine(
            default_profile=DraftProfile(name="stock", completion_tokens=12)
        ) as engine:
            status, body = _post(
                f"{engine.url}/v1/chat/completions",
                {"model": "sim", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert status == 200
            assert body["usage"]["completion_tokens"] == 12
            assert "logprobs" not in body["choices"][0]

            status, body = _post(
                f"{engine.url}/v1/chat/completions",
                {
                    "model": "sim",
                    "messages": [{"role": "user", "content": "hi"}],
                    "logprobs": True,
                },
            )
            tokens = [entry["token"] for entry in body["choices"][0]["logprobs"]["content"]]
            assert len(tokens) == 12
            assert "".join(tokens) == body["choices"][0]["message"]["content"]

    def test_max_tokens_caps_the_generation(self) -> None:
        with running_engine(
            default_profile=DraftProfile(name="stock", completion_tokens=64)
        ) as engine:
            _, body = _post(
                f"{engine.url}/v1/chat/completions",
                {"messages": [], "max_tokens": 8},
            )
            assert body["usage"]["completion_tokens"] == 8
            assert body["choices"][0]["finish_reason"] == "length"


class TestCounterBehaviour:
    def test_counters_are_monotonic_within_one_engine_lifetime(self) -> None:
        profile = DraftProfile(
            name="stock",
            acceptance_rate=0.5,
            drafted_tokens_per_request=40,
            completion_tokens=16,
            seconds_per_request=0.0,
        )
        with running_engine(default_profile=profile) as engine:
            before = parse_metrics(engine.scrape())
            for _ in range(5):
                _post(f"{engine.url}/v1/chat/completions", {"messages": []})
            after = parse_metrics(engine.scrape())

            delta = compute_delta(before, after)
            assert delta.reset_detected is False
            assert delta.drafted_tokens == 200.0
            assert delta.accepted_tokens == 100.0
            assert delta.acceptance_rate == pytest.approx(0.5)

    def test_activation_restarts_the_engine_and_resets_its_counters(self) -> None:
        profile = DraftProfile(name="stock", seconds_per_request=0.0)
        with running_engine(default_profile=profile) as engine:
            engine.register("draft-b", DraftProfile(name="candidate", seconds_per_request=0.0))
            for _ in range(3):
                _post(f"{engine.url}/v1/chat/completions", {"messages": []})
            before = parse_metrics(engine.scrape())
            assert before.drafted_tokens > 0

            engine.activate("draft-b")
            after = parse_metrics(engine.scrape())
            assert after.drafted_tokens == 0.0

            # This is the shape the gate's guard exists for.  It is invisible
            # to a real benchmark only because each arm's first scrape happens
            # *after* its activation.
            with pytest.raises(CounterResetError):
                compute_delta(before, after)

    def test_a_mid_flight_reset_is_visible_as_a_counter_reset(self) -> None:
        # The reset must leave *less* accumulated than the opening scrape saw,
        # or the window is merely small rather than backwards -- five requests
        # before the window opens, then a reset three requests into it.
        faults = EngineFaults(reset_counters_after_requests=6)
        profile = DraftProfile(name="stock", seconds_per_request=0.0)
        with running_engine(default_profile=profile, faults=faults) as engine:
            for _ in range(5):
                _post(f"{engine.url}/v1/chat/completions", {"messages": []})
            before = parse_metrics(engine.scrape())
            for _ in range(3):
                _post(f"{engine.url}/v1/chat/completions", {"messages": []})
            after = parse_metrics(engine.scrape())

            assert "counter-reset" in engine.journal.events
            with pytest.raises(CounterResetError):
                compute_delta(before, after)

    def test_sleeping_does_not_reset_counters(self) -> None:
        # A level-1 sleep offloads weights; it does not restart the process, so
        # a window spanning a sleep/wake pair is still a valid measurement.
        profile = DraftProfile(name="stock", seconds_per_request=0.0)
        with running_engine(default_profile=profile) as engine:
            for _ in range(3):
                _post(f"{engine.url}/v1/chat/completions", {"messages": []})
            before = parse_metrics(engine.scrape())
            engine.activate_sleep()
            engine.wake()
            after = parse_metrics(engine.scrape())
            assert compute_delta(before, after).reset_detected is False


class TestFaults:
    def test_never_ready_engine_reports_unhealthy(self) -> None:
        with running_engine(faults=EngineFaults(never_ready=True)) as engine:
            assert _get(f"{engine.url}/health")[0] == 503

    def test_crash_after_n_requests_fails_every_later_request(self) -> None:
        faults = EngineFaults(crash_after_requests=2)
        profile = DraftProfile(name="stock", seconds_per_request=0.0)
        with running_engine(default_profile=profile, faults=faults) as engine:
            assert _post(f"{engine.url}/v1/chat/completions", {"messages": []})[0] == 200
            assert _post(f"{engine.url}/v1/chat/completions", {"messages": []})[0] == 200
            assert _post(f"{engine.url}/v1/chat/completions", {"messages": []})[0] == 500
            # A crash is durable and observable on the health probe, not a
            # one-off error the next request recovers from.
            assert _post(f"{engine.url}/v1/chat/completions", {"messages": []})[0] == 500
            assert _get(f"{engine.url}/health")[0] == 503

    def test_sleeping_engine_refuses_generations(self) -> None:
        with running_engine(
            default_profile=DraftProfile(name="stock", seconds_per_request=0.0)
        ) as engine:
            engine.activate_sleep()
            assert _post(f"{engine.url}/v1/chat/completions", {"messages": []})[0] == 503
            assert _get(f"{engine.url}/health")[0] == 503

    def test_refused_sleep_and_wake_surface_as_errors(self) -> None:
        with running_engine(
            faults=EngineFaults(refuse_sleep=True, refuse_wake=True)
        ) as engine:
            assert _post(f"{engine.url}/sleep", {})[0] == 500
            assert _post(f"{engine.url}/wake_up", {})[0] == 500

    def test_invalid_every_makes_a_measurable_fraction_of_requests_fail(self) -> None:
        profile = DraftProfile(name="flaky", seconds_per_request=0.0, invalid_every=2)
        with running_engine(default_profile=profile) as engine:
            statuses = [
                _post(f"{engine.url}/v1/chat/completions", {"messages": []})[0]
                for _ in range(6)
            ]
            assert statuses.count(500) == 3

    def test_divergence_point_is_exact(self) -> None:
        baseline = DraftProfile(name="stock", seconds_per_request=0.0, completion_tokens=32)
        diverging = DraftProfile(
            name="candidate",
            seconds_per_request=0.0,
            completion_tokens=32,
            divergence_at_token=5,
        )
        with running_engine(default_profile=baseline) as engine:
            _, stock_body = _post(
                f"{engine.url}/v1/chat/completions", {"messages": [], "logprobs": True}
            )
            engine.register("cand", diverging)
            engine.activate("cand")
            _, cand_body = _post(
                f"{engine.url}/v1/chat/completions", {"messages": [], "logprobs": True}
            )

        stock_tokens = [e["token"] for e in stock_body["choices"][0]["logprobs"]["content"]]
        cand_tokens = [e["token"] for e in cand_body["choices"][0]["logprobs"]["content"]]
        first_difference = next(
            index
            for index, (a, b) in enumerate(zip(stock_tokens, cand_tokens, strict=True))
            if a != b
        )
        assert first_difference == 5


def test_engine_rejects_a_double_start() -> None:
    engine = SimulatedEngine()
    engine.start()
    try:
        with pytest.raises(Exception, match="already running"):
            engine.start()
    finally:
        engine.stop()
