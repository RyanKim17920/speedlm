"""Synchronous, loopback-only client for vLLM lifecycle control."""

from __future__ import annotations

import ipaddress
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx

from speedlm.gateway.control import (
    DRAFT_SWAP_RESUME_TIMEOUT_SECONDS,
    DRAFT_SWAP_RPC_METHOD,
    VLLM_COLLECTIVE_RPC_ENDPOINT,
    VLLM_IS_SLEEPING_ENDPOINT,
    VLLM_PAUSE_ENDPOINT,
    VLLM_PAUSE_MODE,
    VLLM_RESUME_ENDPOINT,
    VLLM_SLEEP_ENDPOINT,
    VLLM_WAKE_ENDPOINT,
    AbortCheck,
    ControlAborted,
    ControlTimeout,
    DraftSwapCorrupted,
    DraftSwapUnavailable,
)

VLLM_HEALTH_ENDPOINT: Final = "/health"
VLLM_MODELS_ENDPOINT: Final = "/v1/models"
VLLM_COMPLETIONS_ENDPOINT: Final = "/v1/completions"
VLLM_METRICS_ENDPOINT: Final = "/metrics"

# Every dev-mode command this transport is allowed to issue. The set is an
# allowlist rather than a denylist because ``VLLM_SERVER_DEV_MODE=1`` mounts a
# whole management surface (weight transfer, cache resets, profiling) that this
# gateway has no business reaching.
_CONTROL_ENDPOINTS: Final = frozenset(
    {
        VLLM_SLEEP_ENDPOINT,
        VLLM_WAKE_ENDPOINT,
        VLLM_PAUSE_ENDPOINT,
        VLLM_RESUME_ENDPOINT,
        VLLM_COLLECTIVE_RPC_ENDPOINT,
    }
)

DEFAULT_POLL_INTERVAL_SECONDS: Final = 0.05
DEFAULT_ATTEMPT_TIMEOUT_SECONDS: Final = 2.0
_CANARY_PROMPT: Final = "SpeedLM readiness"

# HTTP statuses vLLM's routers produce *before* dispatching anything to the
# engine: a malformed body, an absent route, or a wrong method. Every other
# non-2xx (notably the 500 a worker-side raise is flattened into) says nothing
# about whether the call took effect.
_UNDISPATCHED_STATUSES: Final = frozenset({400, 404, 405, 501})


def _no_abort() -> bool:
    return False


class VLLMControlError(RuntimeError):
    """Raised when vLLM rejects or violates its lifecycle HTTP contract."""


class VLLMRequestNotDelivered(VLLMControlError):
    """Raised when a request provably never reached the engine.

    Either the transport failed before the bytes left this process, or a
    router rejected the request before dispatching it.
    """


class VLLMRequestIndeterminate(VLLMControlError):
    """Raised when a request reached the engine but its effect is unknown.

    A worker-side raise arrives as an opaque HTTP 500 (the original exception
    type is flattened on the way out of the engine core), so a validation
    failure and a half-applied mutation are indistinguishable from here. The
    caller must assume the worse of the two.
    """


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

    @property
    def clock(self) -> Callable[[], float]:
        """The monotonic source collaborators must share to stay test-hermetic."""
        return self._clock

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
        if endpoint not in _CONTROL_ENDPOINTS:
            raise ValueError(f"unsupported vLLM control endpoint: {endpoint}")
        _validate_positive(timeout_seconds, "timeout")
        self._request_accepted(
            "POST",
            endpoint,
            timeout_seconds=timeout_seconds,
            params=query,
        )

    def collective_rpc(
        self,
        method: str,
        args: Sequence[str],
        *,
        timeout_seconds: float,
    ) -> list[Any]:
        """Invoke one worker-extension method and return its per-worker results.

        vLLM's ``/collective_rpc`` route accepts only a string ``method`` and a
        list of *string* ``args`` -- it forwards them verbatim and leaves any
        deserialization to the worker -- so this signature deliberately cannot
        express anything richer.

        Raises:
            VLLMRequestNotDelivered: The call provably never dispatched.
            VLLMRequestIndeterminate: The call may or may not have taken effect.
            VLLMControlError: The response violated the route's contract.
        """
        if not method:
            raise ValueError("collective RPC method must be non-empty")
        payload = [str(argument) for argument in args]
        if any(not argument for argument in payload):
            raise ValueError("collective RPC arguments must be non-empty strings")
        _validate_positive(timeout_seconds, "timeout")
        self._require_open()
        try:
            response = self._client.post(
                self._url(VLLM_COLLECTIVE_RPC_ENDPOINT),
                json={"method": method, "args": payload},
                timeout=timeout_seconds,
            )
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.UnsupportedProtocol) as exc:
            raise VLLMRequestNotDelivered(
                f"vLLM collective RPC {method!r} was never delivered: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VLLMRequestIndeterminate(
                f"vLLM collective RPC {method!r} left an unknown outcome: {exc}"
            ) from exc
        if response.status_code in _UNDISPATCHED_STATUSES:
            raise VLLMRequestNotDelivered(
                f"vLLM rejected collective RPC {method!r} before dispatch "
                f"with HTTP {response.status_code}"
            )
        if not response.is_success:
            raise VLLMRequestIndeterminate(
                f"vLLM collective RPC {method!r} failed with HTTP "
                f"{response.status_code}; its effect on the engine is unknown"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise VLLMControlError(
                f"vLLM returned invalid JSON from {VLLM_COLLECTIVE_RPC_ENDPOINT}"
            ) from exc
        results = body.get("results") if isinstance(body, dict) else None
        if not isinstance(results, list):
            raise VLLMControlError(
                f"vLLM collective RPC {method!r} response has no results list"
            )
        return results

    def canary(self, *, timeout_seconds: float) -> None:
        """Require one bounded completion to succeed against the live engine.

        ``wait_ready`` proves the child is up and awake; this proves the model
        it is currently holding can still produce a token. They are separate
        questions after a weight mutation.
        """
        _validate_positive(timeout_seconds, "timeout")
        deadline = _Deadline(
            timeout_seconds,
            clock=self._clock,
            operation="vLLM serving canary",
        )
        model = self._model or self._discover_model(deadline)
        if model is None:
            raise VLLMControlError("vLLM exposed no model for the serving canary")
        self._model = model
        response = self._request_accepted(
            "POST",
            VLLM_COMPLETIONS_ENDPOINT,
            timeout_seconds=deadline.remaining(),
            json={
                "model": model,
                "prompt": _CANARY_PROMPT,
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            },
        )
        _validate_canary(response)

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
        json: Mapping[str, Any] | None = None,
    ) -> httpx.Response:
        self._require_open()
        try:
            response = self._client.request(
                method,
                self._url(endpoint),
                params=params,
                json=json,
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


@dataclass(frozen=True, slots=True)
class VLLMDraftSwapClient:
    """Pause, hot-swap the drafter, and resume one live vLLM engine.

    Deliberately built on top of an existing :class:`VLLMControlClient` rather
    than opening its own transport: that reuses the loopback-only validation
    and the connection pool the gateway already owns, and keeps the dev-route
    allowlist in one place.
    """

    control: VLLMControlClient
    #: ``"wait"`` drains in-flight requests without discarding resident weights.
    #: ``/sleep`` would also quiesce, but at level 1 it offloads the weights the
    #: swap is about to mutate.
    pause_mode: str = VLLM_PAUSE_MODE
    resume_timeout_seconds: float = DRAFT_SWAP_RESUME_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        if not self.pause_mode:
            raise ValueError("pause mode must be non-empty")
        _validate_positive(self.resume_timeout_seconds, "resume timeout")

    def hot_swap_draft(self, weights_path: str, *, timeout_seconds: float) -> None:
        """Swap the drafter to the *directory* ``weights_path`` in place.

        The directory is passed through verbatim: resolving it to shards is
        worker-side work, and the route can only carry strings anyway.
        """
        if not weights_path:
            raise ValueError("draft directory must be non-empty")
        _validate_positive(timeout_seconds, "timeout")
        deadline = _Deadline(
            timeout_seconds,
            clock=self.control.clock,
            operation="draft hot-swap",
        )
        try:
            self.control.post(
                VLLM_PAUSE_ENDPOINT,
                timeout_seconds=deadline.remaining(),
                query={"mode": self.pause_mode, "clear_cache": "false"},
            )
        except (VLLMControlError, ControlTimeout) as exc:
            raise DraftSwapUnavailable(
                f"vLLM did not pause for a draft hot-swap: {exc}"
            ) from exc
        resume_error: Exception | None = None
        try:
            try:
                self._swap(weights_path, deadline)
            finally:
                resume_error = self._resume()
        except Exception:
            if resume_error is not None:
                raise DraftSwapCorrupted(
                    "vLLM was not resumed after a failed draft hot-swap and may "
                    f"still be paused: {resume_error}"
                ) from resume_error
            raise
        if resume_error is not None:
            raise DraftSwapCorrupted(
                "the drafter was swapped but vLLM was not resumed and may still "
                f"be paused: {resume_error}"
            ) from resume_error

    def _swap(self, weights_path: str, deadline: _Deadline) -> None:
        try:
            # Read the remaining budget outside the classifying ``try``: running
            # out of time before the call is dispatched leaves the engine
            # untouched, which is the benign half of the split below.
            remaining = deadline.remaining()
        except ControlTimeout as exc:
            raise DraftSwapUnavailable(
                f"the draft hot-swap ran out of time before dispatch: {exc}"
            ) from exc
        try:
            results = self.control.collective_rpc(
                DRAFT_SWAP_RPC_METHOD,
                (weights_path,),
                timeout_seconds=remaining,
            )
        except VLLMRequestNotDelivered as exc:
            raise DraftSwapUnavailable(
                f"vLLM did not dispatch the draft hot-swap: {exc}"
            ) from exc
        except (VLLMControlError, ControlTimeout) as exc:
            # Includes the flattened HTTP 500 a worker-side raise becomes, so
            # a shape-mismatch rejection and a half-written parameter tensor
            # land here together. Only the pessimistic reading is safe.
            raise DraftSwapCorrupted(
                f"the draft hot-swap may have half-applied: {exc}"
            ) from exc
        if not results or not all(_is_swapped(result) for result in results):
            raise DraftSwapCorrupted(
                "the draft hot-swap RPC did not confirm every worker swapped: "
                f"{results!r}"
            )

    def _resume(self) -> Exception | None:
        try:
            self.control.post(
                VLLM_RESUME_ENDPOINT,
                timeout_seconds=self.resume_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return exc
        return None


def _is_swapped(result: Any) -> bool:
    return isinstance(result, Mapping) and result.get("swapped") is True


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
