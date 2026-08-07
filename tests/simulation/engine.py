"""A simulated vLLM engine that speaks real HTTP.

This is a *state machine*, not a stub.  It serves the endpoints the production
code actually calls -- ``/health``, ``/v1/chat/completions``, ``/v1/models``,
``/metrics``, ``/sleep``, ``/wake_up``, ``/is_sleeping``, ``/collective_rpc`` --
and it can fail in the ways a real engine fails: refuse to become ready, crash
partway through a benchmark, stall past a deadline, drop its speculative
counters, or restart underneath the gate so its counters reset.

Fidelity boundaries are documented on :class:`SimulatedEngine`.  The short
version: this models the engine's *observable HTTP contract and its counter
arithmetic*, not its numerics.  There is no tokenizer, no KV cache, no batching
scheduler, and no real speculative decoding -- acceptance is a dialled-in
parameter, and generation is deterministic text.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from simulation.prometheus import CounterState, render_exposition

#: How often the server loop checks for a shutdown request.  See
#: :meth:`SimulatedEngine.start` for why this is not left at the default.
_POLL_SECONDS = 0.01


class EngineFault(Exception):
    """Raised by the simulator's own control surface, never over HTTP."""


@dataclass(frozen=True, slots=True)
class DraftProfile:
    """How the engine behaves while one particular draft is loaded.

    Acceptance and latency are dialled in rather than emergent: the gate's job
    is to *measure* them and decide, and a simulation that had to actually
    speculate in order to produce a 10 pp acceptance lift would be testing the
    simulator instead of the gate.

    Attributes:
        acceptance_rate: Fraction of drafted tokens the verifier accepts.
            Drives the ``spec_decode`` counter deltas the gate reads.
        seconds_per_request: Wall-clock the engine spends on one generation.
            This is what the replay client actually times, so it is what the
            gating throughput statistic (``replay_per_repeat_mean``) sees.
        completion_tokens: Tokens reported per generation.
        drafted_tokens_per_request: Draft tokens proposed per generation.
        divergence_at_token: When set, the generated text and token stream
            differ from the baseline starting at this index.  ``None`` means
            this profile reproduces the baseline output exactly.
        invalid_every: When set, every Nth request returns HTTP 500, which the
            replay records as an invalid result.
        reasoning_model: Emit like a thinking model whose ``<think>`` block is
            still open when the generation ends.  A generation that stopped on
            ``max_tokens`` files its text under ``message.reasoning`` and
            leaves ``content`` null, exactly as vLLM 0.25.1 did for Qwen3-8B
            and gpt-oss-20b under ``benchmark_max_tokens``.  A generation that
            reached its own stop token closed the block, so it reports
            normally.
        natural_stop_every: When set, every Nth generation ends on the model's
            own terms -- ``natural_stop_tokens`` tokens and ``finish_reason:
            "stop"`` -- while the rest run to whatever cap the request carries.
            Shaped like ``invalid_every`` because it is the same kind of knob:
            a deterministic minority of an otherwise uniform run.  Without it
            a profile can only produce an all-truncated or an all-natural run,
            and the regime the archived live runs actually sit in --
            heavily truncated *with* natural stops present, SLURM 8b72d9a's
            1889 of 2049 and 1747 of 2049 -- is inexpressible.  Must be >= 2:
            every generation stopping naturally is not a mixture.
        natural_stop_tokens: Tokens a naturally-stopped generation emits.
            Must be < ``completion_tokens``, so a natural stop is always
            strictly shorter than the profile's full length.  A natural stop
            is honoured only when it also fits *inside the request's own cap*;
            a request whose cap is at or below this length was ended by the
            harness, and the response says ``length`` accordingly rather than
            claiming a stop that never happened.
    """

    name: str
    acceptance_rate: float = 0.40
    seconds_per_request: float = 0.004
    completion_tokens: int = 64
    prompt_tokens: int = 96
    drafted_tokens_per_request: int = 40
    divergence_at_token: int | None = None
    invalid_every: int | None = None
    reasoning_model: bool = False
    natural_stop_every: int | None = None
    natural_stop_tokens: int = 16

    def __post_init__(self) -> None:
        if not 0.0 <= self.acceptance_rate <= 1.0:
            raise ValueError("acceptance_rate must be in [0, 1]")
        if self.seconds_per_request < 0:
            raise ValueError("seconds_per_request must be >= 0")
        for name, value in (
            ("completion_tokens", self.completion_tokens),
            ("prompt_tokens", self.prompt_tokens),
            ("drafted_tokens_per_request", self.drafted_tokens_per_request),
            ("natural_stop_tokens", self.natural_stop_tokens),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if self.natural_stop_every is not None:
            if (
                isinstance(self.natural_stop_every, bool)
                or not isinstance(self.natural_stop_every, int)
                or self.natural_stop_every < 2
            ):
                raise ValueError("natural_stop_every must be an integer >= 2")
            if self.natural_stop_tokens >= self.completion_tokens:
                raise ValueError(
                    "natural_stop_tokens must be < completion_tokens: a "
                    "generation that ran to the profile's full length did not "
                    "stop on its own terms"
                )


@dataclass
class EngineFaults:
    """Injectable failure modes, all off by default.

    Every one of these is a thing a real engine has actually done to this
    system, which is why they are modelled as first-class state rather than
    as monkeypatches at the call site.
    """

    #: ``/health`` never returns 200, so an activation waiting on readiness
    #: times out instead of proceeding.
    never_ready: bool = False
    #: After this many served generations, every further request fails with
    #: HTTP 500 and ``/health`` starts reporting 503 -- an engine that died
    #: mid-benchmark rather than one that was never up.
    crash_after_requests: int | None = None
    #: Extra wall-clock added to every generation, on top of the profile's own
    #: ``seconds_per_request``.  Used to push a benchmark past its deadline.
    stall_seconds: float = 0.0
    #: After this many served generations, the counters reset to zero as if the
    #: engine had restarted underneath the gate.  This is the *only* honest way
    #: to prove :class:`speedlm.gate.metrics.CounterResetError` fires: a reset
    #: at activation time is invisible, because the arm's first scrape happens
    #: after activation.
    reset_counters_after_requests: int | None = None
    #: Drop the three ``spec_decode`` series, as a non-speculative engine does.
    omit_spec_counters: bool = False
    #: ``/sleep`` returns HTTP 500.
    refuse_sleep: bool = False
    #: ``/wake_up`` returns HTTP 500.
    refuse_wake: bool = False
    #: ``/health`` reports 503 until the next :meth:`SimulatedEngine.activate`.
    #:
    #: Distinct from :attr:`never_ready`, which is permanent.  This models the
    #: engine a *restart* fixes, which is the only way to express "the cheap
    #: recovery path could not confirm serving, but the expensive one can":
    #: with ``never_ready`` the fallback restart would fail too, so the test
    #: could not tell a refused fast path from a dead deployment.
    unhealthy_until_restart: bool = False
    #: Successful ``/v1/completions`` responses allowed before the route starts
    #: returning HTTP 500.  Cleared by :meth:`SimulatedEngine.activate`, like
    #: :attr:`unhealthy_until_restart` and for the same reason.
    #:
    #: ``VLLMControlClient.wait_ready`` and ``VLLMControlClient.canary`` both
    #: land on this route, and they ask different questions -- "is the child
    #: up" versus "can the model it is holding still emit a token".  Budgeting
    #: the route is the only way to make the second fail while the first
    #: passes, which is exactly the case the restore fast path must refuse.
    completions_before_failure: int | None = None


@dataclass
class EngineJournal:
    """Everything the engine was asked to do, in order.

    Recorded so a test can assert on *sequence* -- was the engine woken before
    the candidate was started, was it restored to the right draft -- without
    reaching into the runtime controller's own bookkeeping.
    """

    events: list[str] = field(default_factory=list)
    activations: list[str] = field(default_factory=list)
    generations: int = 0
    scrapes: int = 0
    restarts: int = 0

    def record(self, event: str) -> None:
        self.events.append(event)


class SimulatedEngine:
    """An in-process vLLM stand-in serving real HTTP on loopback.

    Fidelity boundaries -- what this deliberately does NOT model:

    * **No tokenizer.**  ``completion_tokens`` is declared by the profile, and
      the ``logprobs.content`` token stream is synthetic.  Divergence indices
      are therefore exact by construction, which is what makes the
      ``min_divergence_token_index`` boundary testable at all, but it means
      this cannot catch a real tokenisation mismatch.
    * **No batching scheduler.**  Requests are served concurrently by a thread
      pool and each sleeps its profile's latency independently, so throughput
      scales linearly with concurrency.  A real engine's per-step cost grows
      with batch size; nothing here reproduces that.
    * **No real speculation.**  Acceptance is dialled in.  The counter
      *arithmetic* is faithful (monotonic within a lifetime, reset on restart,
      aggregated across engine labels), the physics is not.
    * **No KV cache, no memory pressure, no preemption-by-swap.**  The
      ``num_requests_swapped`` gauge is always whatever it was set to.
    * **Sleep is bookkeeping.**  ``/sleep`` flips a flag and stops serving
      generations; it does not free anything, so it cannot exercise the
      GPU-memory wait that the production runtime does after sleeping.
    * **The control plane is free.**  ``/v1/completions`` (the readiness and
      canary probe), ``/pause`` and ``/resume`` cost nothing and move no
      counters, so nothing here can reproduce a restore whose fast path is
      slower than the restart it replaced.  Only the *decisions* are modelled.
    """

    def __init__(
        self,
        *,
        profiles: Mapping[str, DraftProfile] | None = None,
        default_profile: DraftProfile | None = None,
        fallback_profile: DraftProfile | None = None,
        faults: EngineFaults | None = None,
        journal: EngineJournal | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._profiles: dict[str, DraftProfile] = dict(profiles or {})
        self._active = default_profile or DraftProfile(name="stock")
        # A candidate's reference is an artifact path that does not exist until
        # the registry mints it, so an end-to-end cycle cannot pre-register it.
        # The fallback is how "whatever the freshly trained head turns out to
        # be" gets a behaviour.
        self._fallback = fallback_profile
        self._faults = faults or EngineFaults()
        self.journal = journal or EngineJournal()
        self._counters = CounterState()
        self._sleeping = False
        self._crashed = False
        self._served = 0
        self._completions_served = 0
        self._wedged = self._faults.unhealthy_until_restart
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def url(self) -> str:
        if self._server is None:
            raise EngineFault("engine is not running")
        # ``server_address`` is typed loosely enough to admit bytes, so the
        # host is decoded rather than interpolated.
        host, port = self._server.server_address[:2]
        hostname = host.decode() if isinstance(host, bytes) else str(host)
        return f"http://{hostname}:{int(port)}"

    def start(self) -> None:
        if self._server is not None:
            raise EngineFault("engine is already running")
        engine = self
        # Bind on an ephemeral loopback port.  Verified to work in this
        # environment; a PermissionError here means the sandbox forbids
        # loopback sockets and the caller should skip rather than hang.
        server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(engine))
        server.daemon_threads = True
        # ``shutdown()`` blocks for up to one poll interval, and the default is
        # 0.5s.  With an engine per test that is the single largest cost in the
        # suite -- 0.48s of teardown against ~0.02s of work -- so the interval
        # is tightened rather than the engine being shared between tests.
        thread = threading.Thread(
            target=lambda: server.serve_forever(poll_interval=_POLL_SECONDS),
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread
        self.journal.record("start")

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._server = None
        self._thread = None
        self.journal.record("stop")

    # -- control surface (used by the simulated runtime/endpoint) ----------

    @property
    def faults(self) -> EngineFaults:
        return self._faults

    @property
    def counters(self) -> CounterState:
        with self._lock:
            return self._counters

    @property
    def sleeping(self) -> bool:
        with self._lock:
            return self._sleeping

    @property
    def served_requests(self) -> int:
        with self._lock:
            return self._served

    @property
    def loaded_reference(self) -> str | None:
        """The draft reference the engine was last launched with.

        ``None`` until something activates one.  This is the engine's *own*
        answer to "what is serving", independent of any controller's
        bookkeeping -- which is the whole point of asserting on it: a fast path
        that wrongly trusted its bookkeeping is only visible from here.
        """
        with self._lock:
            return self.journal.activations[-1] if self.journal.activations else None

    def register(self, reference: str, profile: DraftProfile) -> None:
        """Bind a draft reference (a path or a model name) to a behaviour."""
        with self._lock:
            self._profiles[reference] = profile

    def wedge(self) -> None:
        """Make ``/health`` report 503 until the next :meth:`activate`.

        Set here rather than only at construction so a test can wedge an engine
        that is already mid-run, which is the shape the fault actually takes.
        """
        with self._lock:
            self._wedged = True

    def set_completion_budget(self, successes: int | None) -> None:
        """Allow *successes* more ``/v1/completions`` before the route fails.

        The counter is reset as well as the budget, so this means "from now
        on", not "counting everything this engine has ever answered".
        """
        with self._lock:
            self._faults.completions_before_failure = successes
            self._completions_served = 0

    def set_fallback(self, profile: DraftProfile | None) -> None:
        """Set the behaviour any *unregistered* reference resolves to.

        This is how a cycle's freshly trained candidate gets a behaviour: its
        reference is an artifact directory the registry does not mint until
        mid-cycle, so it cannot be registered in advance.
        """
        with self._lock:
            self._fallback = profile

    def activate(self, draft: str | Path) -> None:
        """Load *draft* and restart, exactly as a real activation does.

        A real activation restarts ``vllm serve``, so the counters start again
        from zero.  The gate takes each arm's first scrape *after* activation,
        which is why that reset is invisible to it -- and why proving the
        reset guard fires requires
        :attr:`EngineFaults.reset_counters_after_requests` instead.
        """
        reference = str(draft)
        with self._lock:
            profile = self._profiles.get(reference) or self._fallback
            if profile is None:
                profile = DraftProfile(name=f"unregistered:{reference}")
            self._active = profile
            self._counters = CounterState()
            self._sleeping = False
            self._crashed = False
            self._served = 0
            # A fresh process cannot inherit the previous one's wedged health
            # or its exhausted control-plane budget.  Both are cleared outright
            # rather than merely reset, because both model "the thing a restart
            # fixes" -- and a fault that survived the restart meant to repair it
            # would make every recovery path in the system untestable.
            self._completions_served = 0
            self._faults.completions_before_failure = None
            self._wedged = False
            self.journal.activations.append(reference)
            self.journal.restarts += 1
            self.journal.record(f"activate:{profile.name}")

    def activate_sleep(self) -> None:
        """Put the engine to sleep, as ``POST /sleep`` does.

        Sleeping stops generations but does *not* touch the counters: a level-1
        sleep offloads weights, it does not restart the process, so a scrape
        across a sleep/wake pair must still be a valid monotonic window.
        """
        with self._lock:
            if self._faults.refuse_sleep:
                raise EngineFault("engine refused to sleep")
            self._sleeping = True
            self.journal.record("sleep")

    def wake(self) -> None:
        """Wake the engine, as ``POST /wake_up`` does."""
        with self._lock:
            if self._faults.refuse_wake:
                raise EngineFault("engine refused to wake")
            self._sleeping = False
            self.journal.record("wake_up")

    def scrape(self) -> str:
        """The ``/metrics`` body, as the endpoint would return it."""
        with self._lock:
            self.journal.scrapes += 1
            return render_exposition(
                self._counters,
                include_spec_decode=not self._faults.omit_spec_counters,
            )

    # -- request handling --------------------------------------------------

    def _generation(self, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        """Serve one chat completion, advancing counters and clock."""
        with self._lock:
            if self._sleeping:
                return 503, {"error": "engine is sleeping"}
            if self._crashed:
                return 500, {"error": "engine has crashed"}
            profile = self._active
            faults = self._faults
            self._served += 1
            served = self._served
            if (
                faults.crash_after_requests is not None
                and served > faults.crash_after_requests
            ):
                self._crashed = True
                self.journal.record("crash")
                return 500, {"error": "engine has crashed"}
            if (
                faults.reset_counters_after_requests is not None
                and served == faults.reset_counters_after_requests
            ):
                # An engine restart underneath the gate.  Counters restart from
                # zero mid-window, which is precisely the shape that must be
                # reported as unmeasurable rather than as a negative rate.
                self._counters = CounterState()
                self.journal.record("counter-reset")
            invalid = (
                profile.invalid_every is not None
                and served % profile.invalid_every == 0
            )
            stops_naturally = (
                profile.natural_stop_every is not None
                and served % profile.natural_stop_every == 0
            )

        latency = profile.seconds_per_request + faults.stall_seconds
        if latency > 0:
            time.sleep(latency)

        if invalid:
            return 500, {"error": "simulated upstream failure"}

        requested = request.get("max_tokens")
        budget = profile.completion_tokens
        if isinstance(requested, int) and not isinstance(requested, bool):
            budget = min(budget, max(requested, 1))
        # A natural stop is a claim that the model chose the length, so it is
        # honoured only when it would have landed strictly inside the budget.
        # Under a cap at or below ``natural_stop_tokens`` the harness is still
        # the thing that ended the generation, and saying "stop" there would
        # let a test manufacture a natural stop the run never observed.
        stopped_naturally = stops_naturally and profile.natural_stop_tokens < budget
        emitted = profile.natural_stop_tokens if stopped_naturally else budget
        drafted = profile.drafted_tokens_per_request
        accepted = int(round(drafted * profile.acceptance_rate))
        # One draft per generation keeps ``mean_accepted_length`` well defined
        # without inventing a step schedule the gate never inspects.
        with self._lock:
            self._counters = self._counters.advanced(
                prompt_tokens=profile.prompt_tokens,
                generated_tokens=emitted,
                decode_seconds=latency,
                drafted=drafted,
                accepted=accepted,
                drafts=1,
            )
            self.journal.generations += 1

        tokens = _token_stream(
            emitted,
            divergence_at=profile.divergence_at_token,
            marker=profile.name,
        )
        text = "".join(tokens)
        # ``usage.completion_tokens``, the emitted token stream and the finish
        # reason all read off ``emitted``, so both branches stay consistent:
        # a naturally-stopped response is short *and* says ``stop``.
        truncated = not stopped_naturally and emitted < profile.completion_tokens
        # A thinking model that ran out of budget never closed ``<think>``, so
        # the server has nothing to put in ``content``.  The tokens are all
        # still there, and ``usage`` still counts them.
        thinking = profile.reasoning_model and truncated
        message: dict[str, Any] = {
            "role": "assistant",
            "content": None if thinking else text,
        }
        if thinking:
            message["reasoning"] = text
        body: dict[str, Any] = {
            "id": "chatcmpl-sim",
            "object": "chat.completion",
            "created": 1700000000,
            "model": request.get("model", "sim-model"),
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "length" if truncated else "stop",
                }
            ],
            "usage": {
                "prompt_tokens": profile.prompt_tokens,
                "completion_tokens": emitted,
                "total_tokens": profile.prompt_tokens + emitted,
            },
        }
        if request.get("logprobs"):
            body["choices"][0]["logprobs"] = {
                "content": [
                    {"token": token, "logprob": -0.1, "bytes": None, "top_logprobs": []}
                    for token in tokens
                ]
            }
        return 200, body

    def _completion(self, request: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
        """Serve one legacy ``/v1/completions``, the control plane's canary.

        Deliberately does *not* advance the speculative counters.  This route
        exists only because ``VLLMControlClient.wait_ready`` and ``canary`` use
        it, and those run at activation and restore boundaries -- inside the
        gate's scrape windows in the general case.  Letting a readiness probe
        move the acceptance counters would make the gate's arithmetic depend on
        how many times the controller happened to check whether the engine was
        up, which is a property of the *control* plane, not of the drafter.
        """
        with self._lock:
            if self._sleeping:
                return 503, {"error": "engine is sleeping"}
            if self._crashed or self._wedged:
                return 500, {"error": "engine cannot serve"}
            budget = self._faults.completions_before_failure
            self._completions_served += 1
            if budget is not None and self._completions_served > budget:
                return 500, {"error": "engine cannot complete"}
            model = self._active.name
        return 200, {
            "id": "cmpl-sim",
            "object": "text_completion",
            "created": 1700000000,
            "model": request.get("model", model),
            "choices": [
                {
                    "index": 0,
                    "text": " ok",
                    "finish_reason": "length",
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
        }

    def _healthy(self) -> bool:
        with self._lock:
            return not (
                self._faults.never_ready
                or self._wedged
                or self._crashed
                or self._sleeping
            )


def _token_stream(
    count: int,
    *,
    divergence_at: int | None,
    marker: str,
) -> list[str]:
    """A deterministic token sequence, optionally diverging at an index.

    Baseline tokens are shared by every profile, so two profiles produce
    byte-identical output unless one declares a divergence point.  From that
    index on the stream carries the profile's own marker, which is what makes
    :func:`speedlm.gate.decide.first_divergence` report exactly the requested
    offset.
    """
    tokens: list[str] = []
    for index in range(count):
        if divergence_at is not None and index >= divergence_at:
            tokens.append(f" alt{marker}{index}")
        else:
            tokens.append(f" tok{index}")
    return tokens


def _make_handler(engine: SimulatedEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        # The request target carries a query string, and a gateway that
        # forwards an empty one sends ``/health?``.  Routing on the raw target
        # is the bug that made the shipped journey fixture 404 on every
        # request; splitting it is what a real ASGI server does.
        @property
        def route(self) -> str:
            return urlsplit(self.path).path

        def _send(self, status: int, payload: object) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_text(self, status: int, text: str) -> None:
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "text/plain; version=0.0.4")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            route = self.route
            if route == "/health":
                if engine._healthy():
                    self._send(200, {"status": "ok"})
                else:
                    self._send(503, {"error": "not ready"})
            elif route == "/metrics":
                self._send_text(200, engine.scrape())
            elif route == "/is_sleeping":
                self._send(200, {"is_sleeping": engine.sleeping})
            elif route == "/v1/models":
                self._send(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": "sim-model", "object": "model"}],
                    },
                )
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            route = self.route
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                request = json.loads(raw or b"{}")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send(400, {"error": "invalid json"})
                return
            if not isinstance(request, dict):
                request = {}

            if route == "/v1/chat/completions":
                status, payload = engine._generation(request)
                self._send(status, payload)
            elif route == "/v1/completions":
                status, payload = engine._completion(request)
                self._send(status, payload)
            elif route in {"/pause", "/resume"}:
                # vLLM's RLHF dev router.  Bookkeeping only: the simulator has
                # no scheduler to actually stop, so this records the call and
                # succeeds.  A hot-swap that must fail does so at the
                # ``/collective_rpc`` step, which is where a real one does.
                engine.journal.record(route.lstrip("/"))
                self._send(200, {"status": "ok"})
            elif route == "/sleep":
                if engine.faults.refuse_sleep:
                    self._send(500, {"error": "cannot sleep"})
                    return
                with engine._lock:
                    engine._sleeping = True
                    engine.journal.record("sleep")
                self._send(200, {"status": "ok"})
            elif route == "/wake_up":
                if engine.faults.refuse_wake:
                    self._send(500, {"error": "cannot wake"})
                    return
                with engine._lock:
                    engine._sleeping = False
                    engine.journal.record("wake_up")
                self._send(200, {"status": "ok"})
            elif route == "/collective_rpc":
                method = request.get("method")
                engine.journal.record(f"collective_rpc:{method}")
                self._send(200, {"results": [None]})
            elif route == "/is_sleeping":
                self._send(200, {"is_sleeping": engine.sleeping})
            else:
                self._send(404, {"error": "not found"})

    return Handler


@contextmanager
def running_engine(
    *,
    profiles: Mapping[str, DraftProfile] | None = None,
    default_profile: DraftProfile | None = None,
    fallback_profile: DraftProfile | None = None,
    faults: EngineFaults | None = None,
    journal: EngineJournal | None = None,
) -> Iterator[SimulatedEngine]:
    """Start a :class:`SimulatedEngine` and guarantee it is torn down."""
    engine = SimulatedEngine(
        profiles=profiles,
        default_profile=default_profile,
        fallback_profile=fallback_profile,
        faults=faults,
        journal=journal,
    )
    engine.start()
    try:
        yield engine
    finally:
        engine.stop()
