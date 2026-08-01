"""Replay a frozen benchmark suite against an OpenAI-compatible endpoint."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Final

from speedlm.config import SamplingConfig
from speedlm.gate.suite import BenchmarkSuite, FrozenContext

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ReplayError(RuntimeError):
    """Raised when a replay run encounters a critical error."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class RequestResult:
    """Result of a single suite-request replay."""

    context_hash: str
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_text: str
    valid: bool
    error: str = ""
    #: The generated sequence as the endpoint tokenised it, in order.
    #:
    #: Populated only when the request asked for logprobs -- see
    #: :func:`_send_request`.  Empty means "not captured", which is different
    #: from "the model emitted nothing"; a caller comparing two generations
    #: must fall back to characters rather than treat an empty tuple as a
    #: zero-length match.
    output_tokens: tuple[str, ...] = ()
    #: vLLM's ``finish_reason`` verbatim (``"stop"``, ``"length"``, ...).
    #: Recorded so a divergence at the very end of a bounded generation can be
    #: told apart from one that ended naturally.
    finish_reason: str = ""
    #: The reasoning channel a thinking model fills instead of ``content``.
    #:
    #: Reasoning models spend most of a bounded budget inside ``<think>``, and
    #: an OpenAI-compatible server that parses that block files it under
    #: ``message.reasoning`` (vLLM 0.25.x) or ``message.reasoning_content``,
    #: leaving ``content`` null until the block closes.  Captured because it is
    #: the *only* direct evidence that a response whose ``content`` is empty was
    #: nonetheless a complete, healthy generation.
    reasoning_text: str = ""

    @property
    def invalid(self) -> bool:
        return not self.valid

    @property
    def generated_text(self) -> str:
        """Everything the model emitted, reasoning channel included.

        ``response_text`` alone is not the generation for a thinking model: at
        the caps this gate replays under, it is routinely empty while hundreds
        of tokens sit in ``reasoning_text``.  A character-basis comparison over
        ``response_text`` would therefore compare ``""`` against ``""`` and
        report agreement it never checked.
        """
        return self.reasoning_text + self.response_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_hash": self.context_hash,
            "latency_s": self.latency_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "response_text": self.response_text,
            "valid": self.valid,
            "error": self.error,
            "output_tokens": list(self.output_tokens),
            "finish_reason": self.finish_reason,
            "reasoning_text": self.reasoning_text,
        }


@dataclass(frozen=True, slots=True)
class RunResults:
    """Results from a single replay run (one pass over all contexts)."""

    results: tuple[RequestResult, ...]
    total_latency_s: float
    total_prompt_tokens: int
    total_completion_tokens: int
    valid_count: int
    invalid_count: int
    invalid_rate: float

    @property
    def avg_latency_s(self) -> float:
        if not self.results:
            return 0.0
        return self.total_latency_s / len(self.results)

    @property
    def output_tok_per_sec(self) -> float:
        if self.total_latency_s == 0:
            return 0.0
        return self.total_completion_tokens / self.total_latency_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "total_latency_s": self.total_latency_s,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "valid_count": self.valid_count,
            "invalid_count": self.invalid_count,
            "invalid_rate": self.invalid_rate,
        }


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Aggregated results from N replay runs."""

    run_results: tuple[RunResults, ...]
    num_runs: int
    suite_hash: str

    @property
    def avg_invalid_rate(self) -> float:
        if not self.run_results:
            return 0.0
        return sum(r.invalid_rate for r in self.run_results) / len(self.run_results)

    @property
    def avg_output_tok_per_sec(self) -> float:
        if not self.run_results:
            return 0.0
        return sum(r.output_tok_per_sec for r in self.run_results) / len(self.run_results)

    @property
    def any_high_invalid_rate(self, threshold: float = 0.1) -> bool:
        return any(r.invalid_rate > threshold for r in self.run_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_results": [r.to_dict() for r in self.run_results],
            "num_runs": self.num_runs,
            "suite_hash": self.suite_hash,
        }


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

#: ``finish_reason`` an OpenAI-compatible server reports when a generation
#: stopped because it exhausted the request's ``max_tokens``, as opposed to
#: because the model emitted its own stop token.  This is the one signal that
#: separates a deliberate truncation from a broken generation.
FINISH_REASON_LENGTH: Final = "length"

#: Field names a server may file a thinking model's ``<think>`` block under,
#: in the order they are consulted.  vLLM 0.25.x uses ``reasoning``; the
#: ``reasoning_content`` spelling is what several other OpenAI-compatible
#: servers (and older vLLM reasoning parsers) emit.
_REASONING_FIELDS: Final = ("reasoning", "reasoning_content")

#: Recorded on a response the engine answered without generating anything at
#: all.  This is the failure ``high_invalid_rate`` exists to catch: zero
#: generated tokens means the throughput and acceptance numbers measured over
#: that sample are meaningless.
_ERROR_NO_TOKENS: Final = "No generated tokens"

#: Recorded on a response that generated tokens and then surfaced none of them
#: while claiming to have stopped of its own accord.  Also a broken engine --
#: a healthy one that stops on its own has something to show for it.
_ERROR_EMPTY_TEXT: Final = "Empty response text"


def _extract_reasoning(message: dict[str, Any]) -> str:
    """Recover the reasoning channel, whichever name the server used for it."""
    for field in _REASONING_FIELDS:
        value = message.get(field)
        if isinstance(value, str) and value:
            return value
    return ""


def _validity_error(
    *,
    completion_tokens: int,
    text: str,
    reasoning: str,
    output_tokens: tuple[str, ...],
    finish_reason: str,
) -> str:
    """Classify one 200-OK response: ``""`` when usable, else why it is not.

    "Valid" used to mean "``choices[0].message.content`` is a non-empty
    string", which asks a question neither replay pass needs answered.  The
    throughput/acceptance pass never reads the text -- it needs tokens/second
    and it needs the engine's Prometheus acceptance counters to move.  The
    correctness pass compares token streams recovered from logprobs, which do
    not care what the text renders as.  Only the predicate cared, and on a
    reasoning model it was wrong most of the time: SLURM 369147 replayed
    Qwen3-8B at ``benchmark_max_tokens=512``, 76 of 103 held-out contexts spent
    the whole budget inside ``<think>``, and every one came back with
    ``content: null``, ``finish_reason: "length"`` and
    ``completion_tokens: 512``.  The gate rejected on ``invalid_rate 0.7379``
    while its own correctness pass reported zero divergences.  gpt-oss-20b
    (SLURM 369148) produces the identical shape, so this is a property of
    reasoning models, not of one tokenizer.

    What replaces it is "the engine did real work", which is what the threshold
    is actually protecting.  Ordered so the genuinely broken cases still fail:

    1. Nothing generated and nothing surfaced -> invalid.  ``usage`` is the
       cleanest signal available and it was non-zero for every one of the 512
       captured responses in that run, so a zero here is a real anomaly.
       Surfaced output is accepted as a substitute because a server may omit
       ``usage`` entirely.
    2. Anything surfaced -- content, reasoning, or captured tokens -> valid.
    3. Nothing surfaced but the cap was hit -> valid.  The tokens exist and
       were counted; which channel the server filed them under does not change
       what a throughput sample measures.
    4. Nothing surfaced and the model claims it stopped on its own -> invalid.
       That is an engine returning empty responses, which is exactly the
       measurement-is-worthless case the threshold guards.
    """
    surfaced = bool(text) or bool(reasoning) or bool(output_tokens)
    if completion_tokens <= 0 and not surfaced:
        return _ERROR_NO_TOKENS
    if surfaced:
        return ""
    if finish_reason == FINISH_REASON_LENGTH:
        return ""
    return _ERROR_EMPTY_TEXT


def _extract_output_tokens(choice: dict[str, Any]) -> tuple[str, ...]:
    """Recover the generated token sequence from an OpenAI-shaped choice.

    The OpenAI-compatible surface vLLM serves has no way to return raw token
    *ids* per request: ``--return-tokens-as-token-ids`` is a server-launch flag,
    and the gate replays against an engine it did not launch with that flag.
    What the endpoint does expose per request is ``logprobs.content``, one entry
    per generated token carrying that token's decoded piece.  That is a genuine
    token-level alignment -- the segmentation is the model's own, not a
    re-tokenisation of the response string -- which is what a first-divergence
    index needs.  (If the server *was* started with
    ``--return-tokens-as-token-ids`` these pieces are literally ``token_id:N``
    strings, and the comparison becomes an exact id comparison for free.)

    Returns an empty tuple when logprobs were not requested or not returned, so
    the caller can tell "not captured" from "captured, and empty".
    """
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return ()
    content = logprobs.get("content")
    if not isinstance(content, list):
        return ()
    tokens: list[str] = []
    for entry in content:
        if not isinstance(entry, dict):
            return ()
        token = entry.get("token")
        if not isinstance(token, str):
            return ()
        tokens.append(token)
    return tuple(tokens)


async def _send_request(
    client: Any,
    ctx: FrozenContext,
    sampling: SamplingConfig,
    model: str,
    *,
    max_tokens: int | None = None,
    capture_tokens: bool = False,
) -> RequestResult:
    """Send a single context to the OpenAI-compatible endpoint.

    Args:
        max_tokens: Hard cap on generated tokens, or ``None`` to let the
            generation run to the served model's length limit.  Both of the
            gate's passes set it, at different values: the correctness pass at
            ``correctness_max_tokens`` and the throughput/acceptance pass at
            ``benchmark_max_tokens``.  An earlier revision of this docstring
            said the throughput pass never sets it, on the theory that bounding
            output would change the statistic the throughput threshold is
            calibrated against.  That was wrong twice over -- ``None`` is not
            "unbounded" but "bounded by ``max_model_len`` minus the prompt",
            which is model-dependent, and the throughput statistic is an
            arm-to-arm ratio under a shared cap, so a common cap cancels.  See
            :attr:`speedlm.config.IdleTuningConfig.benchmark_max_tokens`.
        capture_tokens: Ask for per-token logprobs so the response carries the
            model's own tokenisation of its output.
    """
    import httpx
    payload: dict[str, Any] = {
        "model": model,
        "messages": [dict(m) for m in ctx.messages],
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "seed": sampling.seed,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if capture_tokens:
        # ``top_logprobs`` is deliberately omitted: the gate needs the chosen
        # token per position, not the distribution around it, and asking for
        # alternatives multiplies the response size by the requested width.
        payload["logprobs"] = True
    start = time.monotonic()
    try:
        resp = await client.post("v1/chat/completions", json=payload)
        latency = time.monotonic() - start

        if resp.status_code >= 400:
            return RequestResult(
                context_hash=ctx.context_hash,
                latency_s=latency,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                response_text="",
                valid=False,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        body = resp.json()
        choices = body.get("choices", [])
        if not choices:
            return RequestResult(
                context_hash=ctx.context_hash,
                latency_s=latency,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                response_text="",
                valid=False,
                error="Empty choices array",
            )

        choice = choices[0]
        message = choice.get("message", {})
        raw_text = message.get("content")
        text = raw_text if isinstance(raw_text, str) else ""
        reasoning = _extract_reasoning(message)
        output_tokens = _extract_output_tokens(choice)
        finish_reason = choice.get("finish_reason") or ""
        usage = body.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)

        error = _validity_error(
            completion_tokens=ct,
            text=text,
            reasoning=reasoning,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
        )

        return RequestResult(
            context_hash=ctx.context_hash,
            latency_s=latency,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            response_text=text,
            valid=not error,
            error=error,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            reasoning_text=reasoning,
        )

    except httpx.HTTPError as exc:
        latency = time.monotonic() - start
        return RequestResult(
            context_hash=ctx.context_hash,
            latency_s=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response_text="",
            valid=False,
            error=f"HTTPError: {exc}",
        )
    except Exception as exc:
        latency = time.monotonic() - start
        return RequestResult(
            context_hash=ctx.context_hash,
            latency_s=latency,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            response_text="",
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _run_single(
    client: Any,
    suite: BenchmarkSuite,
    sampling: SamplingConfig,
    model: str,
    concurrency: int,
    max_tokens: int | None = None,
    capture_tokens: bool = False,
) -> RunResults:
    """Execute one full pass over the suite, up to *concurrency* in flight.

    Concurrency changes only how fast the pass runs, never what it contains.
    Every context in ``suite.contexts`` is sent exactly once, and the recorded
    ``results`` stay in suite order regardless of completion order:
    :func:`asyncio.gather` returns results positionally, so the per-request
    array remains alignable with the suite that produced it.  The aggregates
    below are order-independent sums, so they are unaffected either way.

    At ``concurrency == 1`` this is byte-for-byte the sequential pass it
    replaced -- a semaphore of one admits one waiter at a time, in the order
    the tasks were created.
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded(ctx: FrozenContext) -> RequestResult:
        async with semaphore:
            return await _send_request(
                client,
                ctx,
                sampling,
                model,
                max_tokens=max_tokens,
                capture_tokens=capture_tokens,
            )

    results: list[RequestResult] = list(
        await asyncio.gather(*(bounded(ctx) for ctx in suite.contexts))
    )

    total_latency = sum(r.latency_s for r in results)
    total_pt = sum(r.prompt_tokens for r in results)
    total_ct = sum(r.completion_tokens for r in results)
    valid = sum(1 for r in results if r.valid)
    invalid = len(results) - valid

    return RunResults(
        results=tuple(results),
        total_latency_s=total_latency,
        total_prompt_tokens=total_pt,
        total_completion_tokens=total_ct,
        valid_count=valid,
        invalid_count=invalid,
        invalid_rate=invalid / len(results) if results else 0.0,
    )


async def replay_suite(
    suite: BenchmarkSuite,
    endpoint_url: str,
    sampling: SamplingConfig,
    *,
    repeats: int = 1,
    timeout: float = 120.0,
    model: str = "auto",
    concurrency: int = 1,
    max_tokens: int | None = None,
    capture_tokens: bool = False,
) -> ReplayResult:
    """Replay suite against an OpenAI-compatible endpoint N times.

    Args:
        suite: The frozen benchmark suite.
        endpoint_url: Base URL of the endpoint.
        sampling: Sampling parameters (temperature, top_p, seed).
        repeats: Number of full passes over the suite.
        timeout: Per-request timeout in seconds.
        model: Served model name to request.
        concurrency: Requests kept in flight within one pass.  Repeats stay
            strictly sequential -- a repeat is the unit the gate's per-repeat
            statistic is computed over, so overlapping two of them would fold
            two samples into one measurement.
        max_tokens: Per-request output cap, or ``None`` for the served model's
            own limit.  Bounding output changes what throughput means, so the
            gate sets this only on its correctness pass.
        capture_tokens: Request per-token logprobs so each result carries the
            model's own tokenisation, which is what
            :func:`speedlm.gate.decide.first_divergence` aligns on.

    Returns:
        Aggregated :class:`ReplayResult` with per-run data.

    Raises:
        ReplayError: If repeats < 1, concurrency < 1, max_tokens < 1, or the
            suite is empty.
    """
    import httpx

    if repeats < 1:
        raise ReplayError(f"repeats must be >= 1, got {repeats}")
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ReplayError(f"concurrency must be an integer >= 1, got {concurrency!r}")
    if max_tokens is not None and (
        isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1
    ):
        raise ReplayError(f"max_tokens must be None or an integer >= 1, got {max_tokens!r}")
    if not suite.contexts:
        raise ReplayError("Cannot replay empty suite")

    runs: list[RunResults] = []

    async with httpx.AsyncClient(
        base_url=endpoint_url,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
        # The pool must not become the narrower limit than the semaphore, or
        # the configured degree would silently not be the degree achieved.
        limits=httpx.Limits(
            max_connections=concurrency,
            max_keepalive_connections=concurrency,
        ),
    ) as client:
        for _ in range(repeats):
            run = await _run_single(
                client,
                suite,
                sampling,
                model,
                concurrency,
                max_tokens=max_tokens,
                capture_tokens=capture_tokens,
            )
            runs.append(run)

    return ReplayResult(
        run_results=tuple(runs),
        num_runs=len(runs),
        suite_hash=suite.suite_hash,
    )
