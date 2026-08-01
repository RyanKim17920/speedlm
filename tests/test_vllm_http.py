from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import pytest

from speedlm.gateway.control import (
    ControlAborted,
    ControlTimeout,
    DraftSwapCorrupted,
    DraftSwapUnavailable,
)
from speedlm.gateway.vllm_http import (
    VLLMControlClient,
    VLLMControlError,
    VLLMDraftSwapClient,
)


@dataclass
class FakeClock:
    now: float = 0.0
    sleep_hook: Callable[[], None] | None = None

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds
        if self.sleep_hook is not None:
            self.sleep_hook()


def make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    clock: FakeClock | None = None,
    model: str | None = None,
) -> tuple[VLLMControlClient, httpx.Client]:
    effective_clock = clock or FakeClock()
    transport_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
        trust_env=False,
    )
    return (
        VLLMControlClient(
            "http://127.0.0.1:8123",
            model=model,
            client=transport_client,
            poll_interval_seconds=0.1,
            attempt_timeout_seconds=0.25,
            clock=effective_clock,
            sleeper=effective_clock.sleep,
        ),
        transport_client,
    )


def test_wait_sleeping_polls_authoritative_state() -> None:
    observed = iter((False, False, True))
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/is_sleeping"
        return httpx.Response(200, json={"is_sleeping": next(observed)})

    client, transport_client = make_client(handler)
    try:
        client.wait_sleeping(True, timeout_seconds=1.0)
    finally:
        transport_client.close()

    assert calls == 3


def test_wait_sleeping_is_abort_aware_between_polls() -> None:
    clock = FakeClock()
    aborted = False

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"is_sleeping": False})

    def request_abort() -> None:
        nonlocal aborted
        aborted = True

    clock.sleep_hook = request_abort
    client, transport_client = make_client(handler, clock=clock)
    try:
        with pytest.raises(ControlAborted, match="sleep-state wait aborted"):
            client.wait_sleeping(
                True,
                timeout_seconds=1.0,
                should_abort=lambda: aborted,
            )
    finally:
        transport_client.close()


def test_wait_sleeping_uses_one_hard_deadline() -> None:
    clock = FakeClock()

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"is_sleeping": False})

    client, transport_client = make_client(handler, clock=clock)
    try:
        with pytest.raises(ControlTimeout, match="sleeping=true"):
            client.wait_sleeping(True, timeout_seconds=0.2)
    finally:
        transport_client.close()

    assert clock.now == pytest.approx(0.2)


def test_missing_dev_sleep_route_fails_with_actionable_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client, transport_client = make_client(handler)
    try:
        with pytest.raises(VLLMControlError, match="VLLM_SERVER_DEV_MODE=1"):
            client.wait_sleeping(True, timeout_seconds=1.0)
    finally:
        transport_client.close()


def test_wait_ready_requires_awake_state_and_successful_inference_canary() -> None:
    clock = FakeClock()
    sleeping_states = iter((True, False, False))
    completion_calls = 0
    health_calls = 0
    model_calls = 0
    canary_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal completion_calls, health_calls, model_calls
        if request.url.path == "/health":
            health_calls += 1
            return httpx.Response(200)
        if request.url.path == "/is_sleeping":
            return httpx.Response(200, json={"is_sleeping": next(sleeping_states)})
        if request.url.path == "/v1/models":
            model_calls += 1
            return httpx.Response(200, json={"data": [{"id": "served-alias"}]})
        if request.url.path == "/v1/completions":
            completion_calls += 1
            canary_payloads.append(json.loads(request.content))
            if completion_calls == 1:
                return httpx.Response(503)
            return httpx.Response(200, json={"choices": [{"text": "!"}]})
        raise AssertionError(f"unexpected endpoint: {request.url.path}")

    client, transport_client = make_client(handler, clock=clock)
    try:
        client.wait_ready(timeout_seconds=1.0)
    finally:
        transport_client.close()

    assert health_calls == 3
    assert model_calls == 1
    assert completion_calls == 2
    assert canary_payloads == [
        {
            "model": "served-alias",
            "prompt": "SpeedLM readiness",
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        },
        {
            "model": "served-alias",
            "prompt": "SpeedLM readiness",
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        },
    ]


def test_health_200_does_not_make_sleeping_child_ready() -> None:
    clock = FakeClock()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/is_sleeping":
            return httpx.Response(200, json={"is_sleeping": True})
        raise AssertionError("sleeping child must not receive a canary")

    client, transport_client = make_client(handler, clock=clock, model="model")
    try:
        with pytest.raises(ControlTimeout, match="serving readiness"):
            client.wait_ready(timeout_seconds=0.2)
    finally:
        transport_client.close()


def test_canary_4xx_fails_fast_instead_of_retrying_forever() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/is_sleeping":
            return httpx.Response(200, json={"is_sleeping": False})
        if request.url.path == "/v1/completions":
            return httpx.Response(400)
        raise AssertionError(f"unexpected endpoint: {request.url.path}")

    client, transport_client = make_client(handler, model="model")
    try:
        with pytest.raises(VLLMControlError, match="canary with HTTP 400"):
            client.wait_ready(timeout_seconds=1.0)
    finally:
        transport_client.close()


def test_canary_must_contain_a_completion_choice() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/is_sleeping":
            return httpx.Response(200, json={"is_sleeping": False})
        if request.url.path == "/v1/completions":
            return httpx.Response(200, json={"choices": []})
        raise AssertionError(f"unexpected endpoint: {request.url.path}")

    client, transport_client = make_client(handler, model="model")
    try:
        with pytest.raises(VLLMControlError, match="no completion choice"):
            client.wait_ready(timeout_seconds=1.0)
    finally:
        transport_client.close()


def test_control_post_rejects_http_failure() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client, transport_client = make_client(handler)
    try:
        with pytest.raises(VLLMControlError, match="POST /sleep"):
            client.post(
                "/sleep",
                timeout_seconds=1.0,
                query={"level": "1", "mode": "wait"},
            )
    finally:
        transport_client.close()


def test_metrics_snapshot_uses_managed_loopback_client() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/metrics"
        return httpx.Response(200, text="vllm:num_requests_running 0\n")

    client, transport_client = make_client(handler)
    try:
        metrics = client.read_metrics(timeout_seconds=1.0)
    finally:
        transport_client.close()

    assert metrics == "vllm:num_requests_running 0\n"


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:8123",
        "http://example.com:8123",
        "http://localhost:8123",
        "http://user@127.0.0.1:8123",
        "http://127.0.0.1:8123/child",
    ],
)
def test_control_client_rejects_non_loopback_origin(url: str) -> None:
    with pytest.raises(ValueError, match="loopback|origin"):
        VLLMControlClient(url)


# --- draft hot-swap transport ---------------------------------------------


DRAFT_DIRECTORY = "/runs/candidate-0007"


@dataclass
class Recorder:
    """Record the dev-route call sequence one hot-swap attempt produces."""

    paths: list[str] = field(default_factory=list)
    bodies: list[Any] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)


def make_swap_client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> tuple[VLLMDraftSwapClient, httpx.Client]:
    control, transport_client = make_client(handler, model="acme/verifier")
    return VLLMDraftSwapClient(control), transport_client


def swap_handler(
    recorder: Recorder,
    *,
    rpc: Callable[[], httpx.Response],
    resume: Callable[[], httpx.Response] | None = None,
    pause: Callable[[], httpx.Response] | None = None,
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        recorder.paths.append(request.url.path)
        recorder.queries.append(str(request.url.query.decode()))
        if request.url.path == "/pause":
            return pause() if pause is not None else httpx.Response(200, json={})
        if request.url.path == "/resume":
            return resume() if resume is not None else httpx.Response(200, json={})
        if request.url.path == "/collective_rpc":
            recorder.bodies.append(json.loads(request.content))
            return rpc()
        raise AssertionError(f"unexpected path {request.url.path}")

    return handler


def test_the_swap_rpc_carries_the_directory_verbatim_between_pause_and_resume() -> None:
    recorder = Recorder()
    client, transport_client = make_swap_client(
        swap_handler(
            recorder,
            rpc=lambda: httpx.Response(
                200, json={"results": [{"swapped": True, "parameters_loaded": 42}]}
            ),
        )
    )
    try:
        client.hot_swap_draft(DRAFT_DIRECTORY, timeout_seconds=5.0)
    finally:
        transport_client.close()

    assert recorder.paths == ["/pause", "/collective_rpc", "/resume"]
    # Directory-to-shard resolution is the worker's job, not this transport's.
    assert recorder.bodies == [
        {"method": "hot_swap_draft", "args": [DRAFT_DIRECTORY]}
    ]
    # ``wait`` drains without offloading the weights the swap is about to touch.
    assert "mode=wait" in recorder.queries[0]
    assert "clear_cache=false" in recorder.queries[0]


@pytest.mark.parametrize("status", [500, 503, 502])
def test_a_worker_side_raise_is_a_possibly_half_applied_swap(status: int) -> None:
    recorder = Recorder()
    client, transport_client = make_swap_client(
        swap_handler(recorder, rpc=lambda: httpx.Response(status, json={"x": 1}))
    )
    try:
        # A worker raise is flattened into an opaque 5xx, so validation
        # rejection and a half-written tensor are indistinguishable here.
        with pytest.raises(DraftSwapCorrupted):
            client.hot_swap_draft(DRAFT_DIRECTORY, timeout_seconds=5.0)
    finally:
        transport_client.close()

    assert recorder.paths == ["/pause", "/collective_rpc", "/resume"]


@pytest.mark.parametrize("status", [400, 404, 405])
def test_a_route_that_never_dispatched_is_a_swap_that_never_ran(status: int) -> None:
    recorder = Recorder()
    client, transport_client = make_swap_client(
        swap_handler(recorder, rpc=lambda: httpx.Response(status, json={}))
    )
    try:
        with pytest.raises(DraftSwapUnavailable):
            client.hot_swap_draft(DRAFT_DIRECTORY, timeout_seconds=5.0)
    finally:
        transport_client.close()

    assert recorder.paths == ["/pause", "/collective_rpc", "/resume"]


@pytest.mark.parametrize(
    "body",
    [
        {"results": []},
        {"results": [{"swapped": False}]},
        {"results": [{"swapped": True}, {"error": "boom"}]},
        {"results": "not-a-list"},
        {},
    ],
)
def test_an_unconfirmed_swap_is_never_read_as_success(body: dict[str, Any]) -> None:
    recorder = Recorder()
    client, transport_client = make_swap_client(
        swap_handler(recorder, rpc=lambda: httpx.Response(200, json=body))
    )
    try:
        with pytest.raises(DraftSwapCorrupted):
            client.hot_swap_draft(DRAFT_DIRECTORY, timeout_seconds=5.0)
    finally:
        transport_client.close()

    assert recorder.paths[-1] == "/resume"


def test_a_failed_pause_never_reaches_the_engine_and_never_resumes() -> None:
    recorder = Recorder()
    client, transport_client = make_swap_client(
        swap_handler(
            recorder,
            rpc=lambda: pytest.fail("must not swap without a pause"),
            pause=lambda: httpx.Response(400, json={"error": "bad mode"}),
        )
    )
    try:
        with pytest.raises(DraftSwapUnavailable):
            client.hot_swap_draft(DRAFT_DIRECTORY, timeout_seconds=5.0)
    finally:
        transport_client.close()

    # Nothing was paused, so resuming would be a lie about the engine's state.
    assert recorder.paths == ["/pause"]


def test_an_engine_left_paused_is_reported_as_corrupted() -> None:
    recorder = Recorder()
    client, transport_client = make_swap_client(
        swap_handler(
            recorder,
            rpc=lambda: httpx.Response(200, json={"results": [{"swapped": True}]}),
            resume=lambda: httpx.Response(500, json={"error": "no"}),
        )
    )
    try:
        with pytest.raises(DraftSwapCorrupted, match="paused"):
            client.hot_swap_draft(DRAFT_DIRECTORY, timeout_seconds=5.0)
    finally:
        transport_client.close()

    assert recorder.paths == ["/pause", "/collective_rpc", "/resume"]


def test_a_failed_resume_outranks_the_swap_failure_it_followed() -> None:
    recorder = Recorder()
    client, transport_client = make_swap_client(
        swap_handler(
            recorder,
            rpc=lambda: httpx.Response(404, json={}),
            resume=lambda: httpx.Response(500, json={}),
        )
    )
    try:
        # The swap alone would be a harmless rollback; a paused engine is not.
        with pytest.raises(DraftSwapCorrupted, match="paused"):
            client.hot_swap_draft(DRAFT_DIRECTORY, timeout_seconds=5.0)
    finally:
        transport_client.close()


def test_an_undelivered_request_is_a_swap_that_never_ran() -> None:
    recorder = Recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        recorder.paths.append(request.url.path)
        if request.url.path == "/collective_rpc":
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={})

    client, transport_client = make_swap_client(handler)
    try:
        with pytest.raises(DraftSwapUnavailable):
            client.hot_swap_draft(DRAFT_DIRECTORY, timeout_seconds=5.0)
    finally:
        transport_client.close()

    assert recorder.paths == ["/pause", "/collective_rpc", "/resume"]


def test_a_read_failure_leaves_the_swap_outcome_unknown() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/collective_rpc":
            raise httpx.ReadTimeout("no answer", request=request)
        return httpx.Response(200, json={})

    client, transport_client = make_swap_client(handler)
    try:
        with pytest.raises(DraftSwapCorrupted):
            client.hot_swap_draft(DRAFT_DIRECTORY, timeout_seconds=5.0)
    finally:
        transport_client.close()


@pytest.mark.parametrize(
    "endpoint",
    ["/sleep", "/wake_up", "/pause", "/resume", "/collective_rpc"],
)
def test_the_control_allowlist_covers_every_endpoint_the_tuner_needs(
    endpoint: str,
) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        return httpx.Response(200, json={})

    client, transport_client = make_client(handler)
    try:
        client.post(endpoint, timeout_seconds=1.0)
    finally:
        transport_client.close()

    assert seen == [endpoint]


@pytest.mark.parametrize(
    "endpoint",
    ["/start_weight_update", "/update_weights", "/reset_prefix_cache", "/health"],
)
def test_the_control_allowlist_still_refuses_the_rest_of_the_dev_surface(
    endpoint: str,
) -> None:
    client, transport_client = make_client(
        lambda _request: pytest.fail("must not reach the network")
    )
    try:
        with pytest.raises(ValueError, match="unsupported vLLM control endpoint"):
            client.post(endpoint, timeout_seconds=1.0)
    finally:
        transport_client.close()


def test_the_canary_requires_a_real_completion() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/completions"
        return httpx.Response(200, json={"choices": [{"text": "ok"}]})

    client, transport_client = make_client(handler, model="acme/verifier")
    try:
        client.canary(timeout_seconds=1.0)
    finally:
        transport_client.close()


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"error": "drafter is broken"}),
        httpx.Response(200, json={"choices": []}),
        httpx.Response(200, text="not json"),
    ],
)
def test_a_canary_the_engine_cannot_answer_is_a_failure(
    response: httpx.Response,
) -> None:
    client, transport_client = make_client(
        lambda _request: response, model="acme/verifier"
    )
    try:
        with pytest.raises(VLLMControlError):
            client.canary(timeout_seconds=1.0)
    finally:
        transport_client.close()
