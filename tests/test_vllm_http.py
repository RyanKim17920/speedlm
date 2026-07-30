from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

import httpx
import pytest

from speedlm.gateway.control import ControlAborted, ControlTimeout
from speedlm.gateway.vllm_http import VLLMControlClient, VLLMControlError


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
