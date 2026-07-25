"""Replay a frozen benchmark suite against an OpenAI-compatible endpoint."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

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

    @property
    def invalid(self) -> bool:
        return not self.valid

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

async def _send_request(
    client: Any,
    ctx: FrozenContext,
    sampling: SamplingConfig,
) -> RequestResult:
    """Send a single context to the OpenAI-compatible endpoint."""
    import httpx
    payload: dict[str, Any] = {
        "model": "auto",
        "messages": [dict(m) for m in ctx.messages],
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "seed": sampling.seed,
    }
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

        message = choices[0].get("message", {})
        text = message.get("content", "")
        usage = body.get("usage", {})
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)

        if not text:
            return RequestResult(
                context_hash=ctx.context_hash,
                latency_s=latency,
                prompt_tokens=pt,
                completion_tokens=ct,
                total_tokens=pt + ct,
                response_text="",
                valid=False,
                error="Empty response text",
            )

        return RequestResult(
            context_hash=ctx.context_hash,
            latency_s=latency,
            prompt_tokens=pt,
            completion_tokens=ct,
            total_tokens=pt + ct,
            response_text=text,
            valid=True,
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
) -> RunResults:
    """Execute one full pass over the suite."""
    results: list[RequestResult] = []
    for ctx in suite.contexts:
        result = await _send_request(client, ctx, sampling)
        results.append(result)

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
) -> ReplayResult:
    """Replay suite against an OpenAI-compatible endpoint N times.

    Args:
        suite: The frozen benchmark suite.
        endpoint_url: Base URL of the endpoint.
        sampling: Sampling parameters (temperature, top_p, seed).
        repeats: Number of full passes over the suite.
        timeout: Per-request timeout in seconds.

    Returns:
        Aggregated :class:`ReplayResult` with per-run data.

    Raises:
        ReplayError: If repeats < 1 or suite is empty.
    """
    import httpx

    if repeats < 1:
        raise ReplayError(f"repeats must be >= 1, got {repeats}")
    if not suite.contexts:
        raise ReplayError("Cannot replay empty suite")

    runs: list[RunResults] = []

    async with httpx.AsyncClient(
        base_url=endpoint_url,
        timeout=timeout,
        headers={"Content-Type": "application/json"},
    ) as client:
        for _ in range(repeats):
            run = await _run_single(client, suite, sampling)
            runs.append(run)

    return ReplayResult(
        run_results=tuple(runs),
        num_runs=len(runs),
        suite_hash=suite.suite_hash,
    )
