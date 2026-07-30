"""Synchronous, loopback-only client for vLLM lifecycle control."""

from __future__ import annotations

import ipaddress
import time
from collections.abc import Callable, Mapping
from typing import Any, Final

import httpx

from speedlm.gateway.control import (
    VLLM_IS_SLEEPING_ENDPOINT,
    VLLM_SLEEP_ENDPOINT,
    VLLM_WAKE_ENDPOINT,
    AbortCheck,
    ControlAborted,
    ControlTimeout,
)

VLLM_HEALTH_ENDPOINT: Final = "/health"
VLLM_MODELS_ENDPOINT: Final = "/v1/models"
VLLM_COMPLETIONS_ENDPOINT: Final = "/v1/completions"
VLLM_METRICS_ENDPOINT: Final = "/metrics"

DEFAULT_POLL_INTERVAL_SECONDS: Final = 0.05
DEFAULT_ATTEMPT_TIMEOUT_SECONDS: Final = 2.0
_CANARY_PROMPT: Final = "SpeedLM readiness"


def _no_abort() -> bool:
    return False


class VLLMControlError(RuntimeError):
    """Raised when vLLM rejects or violates its lifecycle HTTP contract."""


class _Deadline:
    def __init__(
        self,
        timeout_seconds: float,
        *,
        clock: Callable[[], float],
        operation: str,
    ) -> None:
        _validate_positive(timeout_seconds, "timeout")
        self._clock = clock
        self._expires_at = clock() + timeout_seconds
        self._operation = operation

    def remaining(self) -> float:
        remaining = self._expires_at - self._clock()
        if remaining <= 0:
            raise ControlTimeout(f"{self._operation} timed out")
        return remaining


class VLLMControlClient:
    """Control one vLLM server without exposing its dev routes publicly.

    The client deliberately accepts only a numeric loopback address. vLLM's
    sleep endpoints require ``VLLM_SERVER_DEV_MODE=1`` and expose other
    development operations alongside them, so this transport must never point
    at an untrusted or remotely reachable server.
    """

    def __init__(
        self,
        base_url: str,
        *,
        model: str | None = None,
        client: httpx.Client | None = None,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        attempt_timeout_seconds: float = DEFAULT_ATTEMPT_TIMEOUT_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._base_url = _validate_loopback_url(base_url)
        if model is not None and not model:
            raise ValueError("model must be non-empty when supplied")
        _validate_positive(poll_interval_seconds, "poll interval")
        _validate_positive(attempt_timeout_seconds, "attempt timeout")
        self._model = model
        self._client = client or httpx.Client(
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_client = client is None
        self._poll_interval_seconds = poll_interval_seconds
        self._attempt_timeout_seconds = attempt_timeout_seconds
        self._clock = clock
        self._sleeper = sleeper
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> VLLMControlClient:
        self._require_open()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def post(
        self,
        endpoint: str,
        *,
        timeout_seconds: float,
        query: Mapping[str, str] | None = None,
    ) -> None:
        """POST one supported lifecycle command and require a 2xx response."""
        if endpoint not in {VLLM_SLEEP_ENDPOINT, VLLM_WAKE_ENDPOINT}:
            raise ValueError(f"unsupported vLLM control endpoint: {endpoint}")
        _validate_positive(timeout_seconds, "timeout")
        self._request_accepted(
            "POST",
            endpoint,
            timeout_seconds=timeout_seconds,
            params=query,
        )

    def wait_sleeping(
        self,
        sleeping: bool,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck = _no_abort,
    ) -> None:
        """Poll vLLM's authoritative sleep state until it matches *sleeping*."""
        if not isinstance(sleeping, bool):
            raise TypeError("sleeping must be boolean")
        deadline = _Deadline(
            timeout_seconds,
            clock=self._clock,
            operation=f"waiting for vLLM sleeping={str(sleeping).lower()}",
        )
        while True:
            self._check_abort(should_abort, "vLLM sleep-state wait")
            response = self._poll_request(
                "GET",
                VLLM_IS_SLEEPING_ENDPOINT,
                deadline=deadline,
            )
            if response is not None:
                if response.status_code == 404:
                    raise VLLMControlError(
                        "vLLM sleep endpoints are unavailable; start the child with "
                        "VLLM_SERVER_DEV_MODE=1"
                    )
                if response.is_success:
                    observed = _parse_sleep_state(response)
                    if observed is sleeping:
                        return
            self._poll_pause(deadline, should_abort, "vLLM sleep-state wait")

    def wait_ready(
        self,
        *,
        timeout_seconds: float,
        should_abort: AbortCheck = _no_abort,
    ) -> None:
        """Require liveness, awake state, and a successful one-token inference."""
        deadline = _Deadline(
            timeout_seconds,
            clock=self._clock,
            operation="waiting for vLLM serving readiness",
        )
        while True:
            self._check_abort(should_abort, "vLLM readiness wait")
            health = self._poll_request(
                "GET",
                VLLM_HEALTH_ENDPOINT,
                deadline=deadline,
            )
            if health is not None and health.is_success:
                state = self._poll_request(
                    "GET",
                    VLLM_IS_SLEEPING_ENDPOINT,
                    deadline=deadline,
                )
                if state is not None:
                    if state.status_code == 404:
                        raise VLLMControlError(
                            "vLLM sleep endpoints are unavailable; start the child with "
                            "VLLM_SERVER_DEV_MODE=1"
                        )
                    if state.is_success and not _parse_sleep_state(state):
                        model = self._model or self._discover_model(deadline)
                        if model is None:
                            self._poll_pause(
                                deadline,
                                should_abort,
                                "vLLM readiness wait",
                            )
                            continue
                        self._model = model
                        canary = self._poll_request(
                            "POST",
                            VLLM_COMPLETIONS_ENDPOINT,
                            deadline=deadline,
                            json={
                                "model": model,
                                "prompt": _CANARY_PROMPT,
                                "max_tokens": 1,
                                "temperature": 0.0,
                                "stream": False,
                            },
                        )
                        if canary is not None:
                            if canary.is_success:
                                _validate_canary(canary)
                                return
                            if 400 <= canary.status_code < 500:
                                raise VLLMControlError(
                                    "vLLM rejected the serving-readiness canary "
                                    f"with HTTP {canary.status_code}"
                                )
            self._poll_pause(deadline, should_abort, "vLLM readiness wait")

    def read_metrics(self, *, timeout_seconds: float) -> str:
        """Fetch one Prometheus metrics snapshot from the managed child."""
        _validate_positive(timeout_seconds, "timeout")
        response = self._request_accepted(
            "GET",
            VLLM_METRICS_ENDPOINT,
            timeout_seconds=timeout_seconds,
        )
        return response.text

    def _discover_model(self, deadline: _Deadline) -> str | None:
        response = self._poll_request(
            "GET",
            VLLM_MODELS_ENDPOINT,
            deadline=deadline,
        )
        if response is None or response.is_server_error:
            return None
        if not response.is_success:
            raise VLLMControlError(
                "vLLM did not expose a model for readiness probing "
                f"(HTTP {response.status_code})"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise VLLMControlError("vLLM returned invalid JSON from /v1/models") from exc
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise VLLMControlError("vLLM /v1/models response has no data list")
        for item in data:
            if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]:
                return item["id"]
        raise VLLMControlError("vLLM /v1/models response has no usable model id")

    def _poll_request(
        self,
        method: str,
        endpoint: str,
        *,
        deadline: _Deadline,
        json: Mapping[str, Any] | None = None,
    ) -> httpx.Response | None:
        timeout = min(deadline.remaining(), self._attempt_timeout_seconds)
        try:
            return self._client.request(
                method,
                self._url(endpoint),
                json=json,
                timeout=timeout,
            )
        except httpx.HTTPError:
            return None

    def _request_accepted(
        self,
        method: str,
        endpoint: str,
        *,
        timeout_seconds: float,
        params: Mapping[str, str] | None = None,
    ) -> httpx.Response:
        self._require_open()
        try:
            response = self._client.request(
                method,
                self._url(endpoint),
                params=params,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise VLLMControlError(
                f"vLLM {method} {endpoint} control request failed: {exc}"
            ) from exc
        return response

    def _poll_pause(
        self,
        deadline: _Deadline,
        should_abort: AbortCheck,
        operation: str,
    ) -> None:
        self._check_abort(should_abort, operation)
        self._sleeper(min(self._poll_interval_seconds, deadline.remaining()))

    @staticmethod
    def _check_abort(should_abort: AbortCheck, operation: str) -> None:
        if should_abort():
            raise ControlAborted(f"{operation} aborted")

    def _url(self, endpoint: str) -> str:
        self._require_open()
        return str(self._base_url.copy_with(path=endpoint))

    def _require_open(self) -> None:
        if self._closed:
            raise VLLMControlError("vLLM control client is closed")


def _parse_sleep_state(response: httpx.Response) -> bool:
    try:
        payload = response.json()
    except ValueError as exc:
        raise VLLMControlError("vLLM returned invalid JSON from /is_sleeping") from exc
    sleeping = payload.get("is_sleeping") if isinstance(payload, dict) else None
    if not isinstance(sleeping, bool):
        raise VLLMControlError("vLLM /is_sleeping response has no boolean is_sleeping")
    return sleeping


def _validate_canary(response: httpx.Response) -> None:
    try:
        payload = response.json()
    except ValueError as exc:
        raise VLLMControlError("vLLM readiness canary returned invalid JSON") from exc
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
        or not isinstance(choices[0].get("text"), str)
    ):
        raise VLLMControlError("vLLM readiness canary returned no completion choice")


def _validate_loopback_url(base_url: str) -> httpx.URL:
    try:
        url = httpx.URL(base_url)
    except httpx.InvalidURL as exc:
        raise ValueError(f"invalid vLLM base URL: {exc}") from exc
    if (
        url.scheme != "http"
        or url.host is None
        or bool(url.username)
        or bool(url.password)
        or url.query
        or url.fragment
        or url.path not in {"", "/"}
    ):
        raise ValueError("vLLM base URL must be an uncredentialed HTTP loopback origin")
    try:
        address = ipaddress.ip_address(url.host)
    except ValueError as exc:
        raise ValueError("vLLM base URL host must be a numeric loopback address") from exc
    if not address.is_loopback:
        raise ValueError("vLLM base URL host must be loopback")
    return url.copy_with(path="")


def _validate_positive(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
