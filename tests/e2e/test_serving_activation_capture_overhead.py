"""Measure what serving-time activation capture costs: TTFT and tokens/second.

The repo owner's standing requirement is to *minimise* the overhead of logging
the embeddings.  Until this file existed, that overhead had never been measured
end to end.  What existed instead was:

* ``test_serving_activation_capture.py`` — arms the real hook on a live engine
  but contains no timer at all, sends non-streaming requests (so TTFT is not
  even observable) and caps ``max_tokens`` at 16 (too few decode steps for a
  throughput number).  Both of its tests launch *with* the capture worker
  extension, so there is no capture-OFF arm to compare against.
* ``test_proxy_overhead.py`` — has the TTFT/inter-token/throughput
  instrumentation, but its two arms are direct-to-engine versus
  through-the-gateway, and the thing it toggles is HTTP *trace* capture, a
  different subsystem.  Activation-capture cost is constant across those arms
  and cancels exactly.

This file measures the missing quantity.

Design
------

**One engine, no restart.**  ``ActivationCaptureExtension.deactivate_capture``
genuinely tears down: it restores the original method on the *live* runner class
(``hook.py:_deactivate_impl``) and restores the aux-layer list
(``hook.py:_restore_aux_layers``).  The monkeypatch is installed only inside
``activate_capture``, so a loaded-but-unarmed extension costs nothing on the
forward path.  That makes a valid A/B possible on a single engine, which removes
the engine-lifecycle confound entirely — the confound that biased an earlier
measurement in this project by ~2 percentage points.

**ABBA interleaving.**  Each cycle runs four blocks in the order OFF, ON, ON,
OFF.  The first half pairs (OFF, ON) and the second half pairs (ON, OFF), so the
two halves carry opposite orderings and first-order machine drift cancels
instead of aliasing onto arm identity.  Running one condition to completion and
then the other — which is how this project got burned before — is exactly what
this structure refuses to do.

**Paired differencing.**  The same fixed prompts and the same seeds run in both
arms, so every ON sample has an OFF partner measured under near-identical
conditions.  Differencing within a pair strips out prompt-to-prompt variance for
free.  Reported as a mean paired delta **and a standard error** — a mean without
an error bar is precisely what made the earlier throughput numbers in this repo
untrustworthy.

**Warmup.**  One shared warmup pass before the first measured repetition, plus a
per-condition warmup pass after *every* toggle: installing and restoring the
monkeypatch perturbs the first pass through it, and CUDA graph replay has its
own first-touch cost.

Engine flags that are load-bearing here
---------------------------------------

* ``--enforce-eager`` **is** passed, and not by preference.  An earlier revision
  of this file deliberately omitted it, on the theory that CUDA graphs are on in
  production so measuring without them would not describe production.  That
  configuration cannot run at all: ``activate_capture`` extends the model's
  ``aux_hidden_state_layers`` at runtime, but the forward that reads that tuple
  (``EagleModelMixin._maybe_add_hidden_state``) is traced and CUDA-graph captured
  once at startup and never re-traced, so the replayed graph keeps emitting the
  three default eagle3 aux states while the attribute claims four.  The hook's
  own labelling guard then kills EngineCore on the first captured forward
  (SLURM 370798).  **Consequence for reading these numbers:** activation capture
  is today usable only on an eager engine, so what this file measures is the
  cost of capture in eager mode.  It is not, and cannot yet be, the cost of
  capture in a graph-capturing engine.
* ``--no-enable-prefix-caching`` **is** passed.  The same prompts are replayed
  ~40 times each; with prefix caching on, every repetition after the first would
  skip the prefill and TTFT would measure a cache lookup rather than the work
  capture actually taxes.
* ``max_tokens`` is 128, not 16, so decode dominates prefill and the tok/s
  figure is a decode measurement rather than a prefill measurement.

What is asserted vs. what is only recorded
------------------------------------------

Asserted: that capture was really armed in every ON block and really disarmed in
every OFF block (an unarmed "ON" arm would make this whole file measure zero and
pass), that the two arms did comparable token work, and the paired-mean bounds
below.  Recorded only: the per-sample series, the standard errors and the
percentage figures.  Artifacts are written *before* any assertion runs, so a
failing bound cannot discard an H100 measurement.
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

pytestmark = pytest.mark.e2e

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_VENV = Path("/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm")
VLLM_PYTHON = Path(
    os.environ.get("SPEEDLM_E2E_VLLM_PYTHON", str(VLLM_VENV / "bin" / "python"))
)

#: Own gate, following the existing per-flavor convention
#: (``SPEEDLM_E2E``, ``SPEEDLM_E2E_ACTIVATION_CAPTURE``,
#: ``SPEEDLM_E2E_IDLE_TUNING``, ``SPEEDLM_E2E_DRAFT_HOT_SWAP``).  Without it this
#: file must never run by accident: it costs an H100 for several minutes.
GATE_ENV: Final = "SPEEDLM_E2E_CAPTURE_OVERHEAD"

#: Enough decode steps that throughput is a decode measurement.  At 16 (what
#: ``test_serving_activation_capture.py`` uses) prefill dominates and the
#: per-decode-step cost of the device-to-host capture copy — the thing being
#: measured — is buried.
MAX_TOKENS: Final = 128

#: ABBA cycles.  Each contributes two ON repetitions and two OFF repetitions, so
#: this many cycles gives ``2 * CYCLES`` repetitions per condition.
CYCLES: Final = 10

#: Repetitions per condition implied by :data:`CYCLES`; ~20 as designed.
REPS_PER_CONDITION: Final = 2 * CYCLES

#: Prompt-set passes before the first measured repetition.
SHARED_WARMUP_PASSES: Final = 2

#: Prompt-set passes after every toggle, discarded.
PER_BLOCK_WARMUP_PASSES: Final = 1

REQUEST_TIMEOUT_SECONDS: Final = 300.0

#: Deterministic, fixed, and identical across arms — that is what makes the
#: pairing free.  Three lengths so a length-dependent capture cost is visible
#: rather than averaged away.
PROMPTS: Final[tuple[tuple[str, str], ...]] = (
    ("short", "Name three primary colours."),
    (
        "medium",
        "Explain how speculative decoding accepts or rejects draft tokens. "
        "Answer in numbered points.",
    ),
    (
        "long",
        "Describe, step by step and in detail, how an inference server "
        "schedules a batch of requests: admission, prefill, KV cache "
        "allocation, decode steps, and eviction. Use numbered sections and "
        "give a concrete worked example throughout.",
    ),
)

#: Per-prompt seed, identical in both arms.
_PROMPT_SEEDS: Final[dict[str, int]] = {
    label: index for index, (label, _) in enumerate(PROMPTS)
}

TOKEN_DELTA_FIELDS: Final = ("content", "reasoning_content", "reasoning", "thinking")

# ---------------------------------------------------------------------------
# Regression bounds
# ---------------------------------------------------------------------------
#
# These are regression detectors, not service-level objectives, and they are the
# reason this test can fail.  A latency test that cannot detect a regression is
# worthless, so every bound below is checked against the *paired mean*, which is
# the statistic this design exists to produce.
#
# TTFT is bounded in absolute seconds, not as a percentage, for the same reason
# test_proxy_overhead.py bounds it that way: the percentage is dominated by how
# small the baseline TTFT is, so a faster engine would inflate the percentage
# without capture getting any worse.
#
# Capture's per-forward cost is one fused device-to-host copy of
# (num_scheduled_tokens, H) x num_aux_layers (hook.py:_to_host).  On a prefill
# that is one copy; on TTFT it is one prefill copy plus the first decode step.
# 100 ms of paired TTFT overhead is far more than that transfer can plausibly
# cost on an H100 and is therefore a real ceiling, not a rubber stamp.
MAX_TTFT_ABSOLUTE_OVERHEAD_SECONDS: Final = 0.100

#: Decode-side cost, where capture pays one copy per step. A percentage bound is
#: the right shape here because both quantities scale with generation length.
MAX_THROUGHPUT_PERCENTAGE_LOSS: Final = 25.0
MAX_INTER_TOKEN_PERCENTAGE_OVERHEAD: Final = 25.0

#: The pairing only means anything if the two arms did comparable work.  Capture
#: copies activations, it does not change numerics, so the arms should generate
#: essentially the same number of tokens; a large asymmetry would mean the delta
#: is measuring a token-count difference rather than capture cost.
MAX_COMPLETION_TOKEN_ASYMMETRY_PERCENT: Final = 5.0

#: Fault-injection knob, in milliseconds, applied to the ON arm only.  This is
#: how the paired-difference machinery is proven able to detect a regression on
#: real hardware: set it above MAX_TTFT_ABSOLUTE_OVERHEAD_SECONDS and the test
#: must fail.  Unset (the default) it does nothing.
INJECT_ENV: Final = "SPEEDLM_E2E_CAPTURE_OVERHEAD_INJECT_MS"


# ---------------------------------------------------------------------------
# Environment / scaffolding (mirrors test_serving_activation_capture.py)
# ---------------------------------------------------------------------------


def _require_environment() -> tuple[str, str, Path]:
    """Check required environment and return (verifier, drafter, artifact_root)."""
    if os.environ.get(GATE_ENV) != "1":
        pytest.skip(f"set {GATE_ENV}=1 in an allocated GPU job")
    assert os.environ.get("SLURM_JOB_ID"), (
        "capture-overhead E2E must run inside a SLURM allocation"
    )
    assert os.environ.get("CUDA_VISIBLE_DEVICES"), (
        "SLURM allocation did not expose a GPU"
    )

    verifier = os.environ.get("SPEEDLM_E2E_VERIFIER_MODEL")
    if not verifier:
        raise AssertionError("SPEEDLM_E2E_VERIFIER_MODEL is required")
    drafter = os.environ.get("SPEEDLM_E2E_DRAFTER_MODEL")
    if not drafter:
        raise AssertionError("SPEEDLM_E2E_DRAFTER_MODEL is required")

    artifact_root = os.environ.get("SPEEDLM_E2E_ARTIFACT_DIR")
    if not artifact_root:
        raise AssertionError("SPEEDLM_E2E_ARTIFACT_DIR is required")
    artifact_path = Path(artifact_root)
    assert artifact_path.exists(), f"artifact dir does not exist: {artifact_path}"

    return verifier, drafter, artifact_path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _ready_timeout() -> float:
    raw = os.environ.get("SPEEDLM_E2E_READY_TIMEOUT", "1800")
    try:
        return float(raw)
    except ValueError as exc:
        raise AssertionError("SPEEDLM_E2E_READY_TIMEOUT must be a number") from exc


def _inject_delay_seconds() -> float:
    """Return the ON-arm fault-injection delay in seconds (0.0 when unset)."""
    raw = os.environ.get(INJECT_ENV)
    if not raw:
        return 0.0
    try:
        milliseconds = float(raw)
    except ValueError as exc:
        raise AssertionError(f"{INJECT_ENV} must be a number of milliseconds") from exc
    assert milliseconds >= 0.0, f"{INJECT_ENV} must not be negative"
    return milliseconds / 1000.0


def _vllm_env() -> dict[str, str]:
    """Return os.environ plus VLLM_SERVER_DEV_MODE, which /collective_rpc needs."""
    env = os.environ.copy()
    env["VLLM_SERVER_DEV_MODE"] = "1"
    return env


def _create_artifact_dir(root: Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = root / f"capture-overhead-{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_log_tail(log_path: Path, lines: int = 100) -> str:
    try:
        raw = log_path.read_text(encoding="utf-8", errors="replace")
        return "\n".join(raw.splitlines()[-lines:])
    except FileNotFoundError:
        return "(log file not found)"


def _wait_for_ready(
    url: str,
    process: subprocess.Popen[bytes],
    timeout: float,
    *,
    log_path: Path,
) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            returncode = process.poll()
            if returncode is not None:
                raise AssertionError(
                    f"vLLM exited before readiness with code {returncode}\n"
                    f"--- vLLM log (last 100 lines) ---\n"
                    f"{_read_log_tail(log_path)}\n"
                    f"--- end of vLLM log ---"
                )
            try:
                response = client.get(f"{url}/health")
                if 200 <= response.status_code < 300:
                    return
                last_error = f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                last_error = repr(exc)
            time.sleep(0.5)
    raise AssertionError(
        f"vLLM did not become ready within {timeout}s; {last_error}\n"
        f"--- vLLM log (last 100 lines) ---\n"
        f"{_read_log_tail(log_path)}\n"
        f"--- end of vLLM log ---"
    )


def _get_served_model_id(url: str) -> str:
    """Return the id vLLM actually serves (the resolved snapshot path)."""
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        response = client.get(f"{url}/v1/models")
        response.raise_for_status()
        model_ids = [entry["id"] for entry in response.json().get("data", [])]
    if not model_ids:
        raise AssertionError(
            "/v1/models returned no served models — the engine may not have "
            "finished loading"
        )
    return model_ids[0]


def _collective_rpc_results(port: int, method: str, *args: object) -> list[Any]:
    """Issue a collective_rpc call and return the per-worker return values.

    Same contract as the helper in ``test_serving_activation_capture.py``: vLLM's
    dev router answers ``{"results": [...]}``, one entry per worker, and a
    worker-side exception arrives either as a non-200 or as an ``error`` key.
    """
    url = f"http://127.0.0.1:{port}"
    with httpx.Client(timeout=60.0, trust_env=False) as client:
        response = client.post(
            f"{url}/collective_rpc",
            json={"method": method, "args": [str(arg) for arg in args]},
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"collective_rpc {method} failed: "
                f"{response.status_code} {response.text}"
            )
        if not response.content:
            return []
        body = response.json()
    results = body.get("results")
    if not isinstance(results, list):
        return []
    for index, result in enumerate(results):
        if isinstance(result, dict) and result.get("error"):
            raise RuntimeError(
                f"collective_rpc {method} worker {index} error: {result['error']}"
            )
    return results


def _collective_rpc(port: int, method: str, *args: object) -> None:
    _collective_rpc_results(port, method, *args)


# ---------------------------------------------------------------------------
# Measurement (streaming; reuses test_proxy_overhead.py's approach)
# ---------------------------------------------------------------------------


def _payload(served_model_id: str, label: str, prompt: str) -> dict[str, Any]:
    return {
        "model": served_model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "seed": _PROMPT_SEEDS[label],
        "max_tokens": MAX_TOKENS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take the median of an empty sample")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _measure_streaming(
    client: httpx.Client,
    url: str,
    payload: dict[str, Any],
    *,
    inject_delay_seconds: float = 0.0,
) -> dict[str, Any]:
    """Time one streaming completion: TTFT, inter-token gaps, e2e, throughput.

    Structurally the same measurement ``test_proxy_overhead._measure_streaming``
    makes (a token counts whether it lands in ``content`` or in a reasoning
    field; completion-token counts come from the authoritative usage block), but
    returning one sample rather than a list so the caller can label it with its
    arm and pair.

    ``inject_delay_seconds`` is the fault-injection hook: it delays consumption
    of the stream, so it inflates the observed TTFT and end-to-end time of this
    request exactly as a real regression on this path would.
    """
    started = time.perf_counter()
    first_token_at: float | None = None
    previous_token_at: float | None = None
    inter_token_seconds: list[float] = []
    token_event_count = 0
    completion_tokens: int | None = None
    saw_done = False

    with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        if inject_delay_seconds > 0.0:
            time.sleep(inject_delay_seconds)
        for line in response.iter_lines():
            received_at = time.perf_counter()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            if data == "[DONE]":
                saw_done = True
                continue

            event = json.loads(data)
            usage = event.get("usage")
            if isinstance(usage, dict) and isinstance(
                usage.get("completion_tokens"), int
            ):
                completion_tokens = int(usage["completion_tokens"])

            token_bearing = False
            for choice in event.get("choices", []):
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                token_bearing = token_bearing or any(
                    isinstance(delta.get(field), str) and bool(delta[field])
                    for field in TOKEN_DELTA_FIELDS
                )
            if not token_bearing:
                continue

            token_event_count += 1
            if first_token_at is None:
                first_token_at = received_at
            if previous_token_at is not None:
                inter_token_seconds.append(received_at - previous_token_at)
            previous_token_at = received_at

    elapsed = time.perf_counter() - started
    if not saw_done:
        raise AssertionError("stream ended without [DONE]")
    if first_token_at is None:
        raise AssertionError("stream contained no token-bearing SSE event")
    if completion_tokens is None or completion_tokens <= 0:
        raise AssertionError("stream did not report positive completion-token usage")
    if not inter_token_seconds:
        raise AssertionError("stream contained fewer than two token-bearing SSE events")

    return {
        "ttft_seconds": first_token_at - started,
        "e2e_seconds": elapsed,
        "completion_tokens": completion_tokens,
        "token_event_count": token_event_count,
        "median_inter_token_seconds": _median(inter_token_seconds),
        "throughput_tok_per_sec": completion_tokens / elapsed,
    }


# ---------------------------------------------------------------------------
# Paired statistics
#
# These are pure functions of the sample lists on purpose: they are the part of
# this file that decides pass or fail, and keeping them free of HTTP and of the
# engine is what lets the differencing machinery be exercised against a known
# injected delay without an H100.
# ---------------------------------------------------------------------------

#: Metrics carried through the pairing.  ``worse_when`` says which direction is a
#: regression, so the reported delta is always "positive means worse".
_PAIRED_METRICS: Final[tuple[tuple[str, str], ...]] = (
    ("ttft_seconds", "higher"),
    ("e2e_seconds", "higher"),
    ("median_inter_token_seconds", "higher"),
    ("throughput_tok_per_sec", "lower"),
)


def _pair_key(sample: dict[str, Any]) -> tuple[int, int, str]:
    return (int(sample["cycle"]), int(sample["half"]), str(sample["prompt"]))


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot take the mean of an empty sample")
    return math.fsum(values) / len(values)


def _stdev(values: list[float]) -> float:
    """Sample standard deviation (n-1). Zero for a single observation."""
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    variance = math.fsum((value - average) ** 2 for value in values) / (
        len(values) - 1
    )
    return math.sqrt(variance)


def _paired_statistic(deltas: list[float], baselines: list[float]) -> dict[str, float]:
    """Mean paired delta with its standard error, plus the percentage form.

    The standard error is ``stdev(deltas) / sqrt(n)`` over the *paired
    differences*, not over the raw values: that is the whole point of pairing,
    and it is the error bar whose absence made the earlier throughput numbers in
    this repo uninterpretable.
    """
    count = len(deltas)
    assert count == len(baselines), "delta and baseline series must be the same length"
    mean_delta = _mean(deltas)
    stdev = _stdev(deltas)
    stderr = stdev / math.sqrt(count) if count else 0.0
    mean_baseline = _mean(baselines)
    return {
        "n": float(count),
        "mean_delta": mean_delta,
        "stdev_delta": stdev,
        "stderr_delta": stderr,
        "mean_baseline": mean_baseline,
        "percentage_delta": (
            mean_delta / mean_baseline * 100.0 if mean_baseline > 0 else 0.0
        ),
        #: Delta in units of its own standard error.  Recorded, not asserted: it
        #: says whether an observed delta is distinguishable from noise at all,
        #: which is a different question from whether it is within budget.
        "delta_over_stderr": mean_delta / stderr if stderr > 0 else 0.0,
    }


def _pair_samples(
    off_samples: list[dict[str, Any]],
    on_samples: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Match every ON sample to the OFF sample with the same (cycle, half, prompt).

    An unmatched sample is an error rather than a silent drop: silently pairing
    fewer observations than were measured would shrink the error bar without
    anyone noticing.
    """
    off_by_key = {_pair_key(sample): sample for sample in off_samples}
    assert len(off_by_key) == len(off_samples), "duplicate OFF pair keys"
    on_by_key = {_pair_key(sample): sample for sample in on_samples}
    assert len(on_by_key) == len(on_samples), "duplicate ON pair keys"
    assert set(off_by_key) == set(on_by_key), (
        "the two arms did not cover the same (cycle, half, prompt) set: "
        f"OFF-only={sorted(set(off_by_key) - set(on_by_key))} "
        f"ON-only={sorted(set(on_by_key) - set(off_by_key))}"
    )
    return [(off_by_key[key], on_by_key[key]) for key in sorted(off_by_key)]


def _summarize(
    off_samples: list[dict[str, Any]],
    on_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the paired summary that the bounds below are checked against."""
    pairs = _pair_samples(off_samples, on_samples)
    metrics: dict[str, dict[str, float]] = {}
    for name, worse_when in _PAIRED_METRICS:
        deltas: list[float] = []
        baselines: list[float] = []
        for off_sample, on_sample in pairs:
            off_value = float(off_sample[name])
            on_value = float(on_sample[name])
            if worse_when == "higher":
                deltas.append(on_value - off_value)
            else:
                deltas.append(off_value - on_value)
            baselines.append(off_value)
        metrics[name] = _paired_statistic(deltas, baselines)

    off_tokens = _mean([float(sample["completion_tokens"]) for sample in off_samples])
    on_tokens = _mean([float(sample["completion_tokens"]) for sample in on_samples])
    return {
        "pairs": len(pairs),
        "metrics": metrics,
        "completion_tokens": {
            "off_mean": off_tokens,
            "on_mean": on_tokens,
            "asymmetry_percent": (
                abs(on_tokens - off_tokens) / off_tokens * 100.0
                if off_tokens > 0
                else 0.0
            ),
        },
    }


def _overhead_failures(summary: dict[str, Any]) -> list[str]:
    """Return one message per violated bound; empty means the run is within budget.

    Split out from the test body so the decision procedure can be driven with
    known inputs — including a known injected delay — without a GPU.
    """
    failures: list[str] = []
    metrics = summary["metrics"]

    ttft = metrics["ttft_seconds"]
    if ttft["mean_delta"] > MAX_TTFT_ABSOLUTE_OVERHEAD_SECONDS:
        failures.append(
            f"TTFT: paired mean overhead {ttft['mean_delta'] * 1000:.1f} ms "
            f"(+/- {ttft['stderr_delta'] * 1000:.1f} ms SE, n={int(ttft['n'])}) "
            f"exceeds {MAX_TTFT_ABSOLUTE_OVERHEAD_SECONDS * 1000:.0f} ms"
        )

    throughput = metrics["throughput_tok_per_sec"]
    if throughput["percentage_delta"] > MAX_THROUGHPUT_PERCENTAGE_LOSS:
        failures.append(
            f"throughput: paired mean loss {throughput['percentage_delta']:.2f}% "
            f"({throughput['mean_delta']:.2f} +/- {throughput['stderr_delta']:.2f} "
            f"tok/s SE, n={int(throughput['n'])}) exceeds "
            f"{MAX_THROUGHPUT_PERCENTAGE_LOSS:.0f}%"
        )

    inter_token = metrics["median_inter_token_seconds"]
    if inter_token["percentage_delta"] > MAX_INTER_TOKEN_PERCENTAGE_OVERHEAD:
        failures.append(
            f"inter-token: paired mean overhead "
            f"{inter_token['percentage_delta']:.2f}% "
            f"({inter_token['mean_delta'] * 1000:.3f} +/- "
            f"{inter_token['stderr_delta'] * 1000:.3f} ms SE, "
            f"n={int(inter_token['n'])}) exceeds "
            f"{MAX_INTER_TOKEN_PERCENTAGE_OVERHEAD:.0f}%"
        )

    asymmetry = summary["completion_tokens"]["asymmetry_percent"]
    if asymmetry > MAX_COMPLETION_TOKEN_ASYMMETRY_PERCENT:
        failures.append(
            f"the two arms did not do comparable work: mean completion tokens "
            f"OFF={summary['completion_tokens']['off_mean']:.1f} "
            f"ON={summary['completion_tokens']['on_mean']:.1f} "
            f"({asymmetry:.2f}% apart, limit "
            f"{MAX_COMPLETION_TOKEN_ASYMMETRY_PERCENT:.0f}%), so the paired "
            f"delta is not attributable to capture"
        )

    return failures


def _summary_text(summary: dict[str, Any]) -> str:
    lines = [
        "",
        "Activation-capture serving overhead (paired, ABBA-interleaved, "
        "one engine)",
        f"  pairs: {summary['pairs']}  "
        f"(={REPS_PER_CONDITION} repetitions x {len(PROMPTS)} prompts "
        f"per condition)",
        f"  completion tokens: OFF "
        f"{summary['completion_tokens']['off_mean']:.1f} / ON "
        f"{summary['completion_tokens']['on_mean']:.1f}",
        "",
        f"  {'metric':<30}{'OFF mean':>12}{'delta':>14}{'SE':>12}{'delta%':>10}",
    ]
    units = {
        "ttft_seconds": ("ms", 1000.0),
        "e2e_seconds": ("s", 1.0),
        "median_inter_token_seconds": ("ms", 1000.0),
        "throughput_tok_per_sec": ("tok/s", 1.0),
    }
    for name, _ in _PAIRED_METRICS:
        statistic = summary["metrics"][name]
        unit, scale = units[name]
        lines.append(
            f"  {name + ' (' + unit + ')':<30}"
            f"{statistic['mean_baseline'] * scale:>12.3f}"
            f"{statistic['mean_delta'] * scale:>14.3f}"
            f"{statistic['stderr_delta'] * scale:>12.3f}"
            f"{statistic['percentage_delta']:>10.2f}"
        )
    lines.append("  (delta is positive when capture is WORSE)")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The measurement run
# ---------------------------------------------------------------------------


def _run_pass(
    client: httpx.Client,
    url: str,
    served_model_id: str,
    *,
    inject_delay_seconds: float,
) -> list[dict[str, Any]]:
    """Send one request per prompt, in prompt order, and return the samples."""
    samples: list[dict[str, Any]] = []
    for label, prompt in PROMPTS:
        sample = _measure_streaming(
            client,
            f"{url}/v1/chat/completions",
            _payload(served_model_id, label, prompt),
            inject_delay_seconds=inject_delay_seconds,
        )
        sample["prompt"] = label
        samples.append(sample)
    return samples


def _assert_capture_disarmed(port: int, block: str) -> None:
    """Prove the OFF arm really was unarmed.

    ``flush_capture`` raises ``RuntimeError("capture is not active")``
    (``hook.py:196-197``) when the extension is not armed, so a successful flush
    here would mean the "OFF" arm was silently still capturing and the whole
    measurement collapses to zero.  Checked outside every timed region.
    """
    try:
        _collective_rpc(port, "flush_capture")
    except RuntimeError:
        return
    raise AssertionError(
        f"{block}: flush_capture succeeded while capture was supposed to be "
        f"deactivated, so the OFF arm was still capturing and every reported "
        f"delta is a difference between two capturing arms"
    )


def _assert_capture_armed(capture_dir: Path, block: str) -> None:
    """Prove the ON arm really captured something."""
    written = capture_dir / "captured.safetensors"
    assert written.is_file(), (
        f"{block}: capture was armed but wrote no {written.name}; the ON arm "
        f"did not actually capture, so this run measures nothing"
    )
    assert written.stat().st_size > 0, (
        f"{block}: {written} is empty, so the ON arm captured no rows"
    )


@pytest.mark.e2e
def test_activation_capture_serving_overhead() -> None:
    """Measure TTFT and tok/s with activation capture ON versus OFF.

    One engine, launched with the capture worker extension exactly as in
    production, toggled with ``activate_capture`` / ``deactivate_capture`` over
    ``collective_rpc``.  ABBA-interleaved, warmed up after every toggle, paired
    per prompt across ``REPS_PER_CONDITION`` repetitions per condition, and
    reported with a standard error.
    """
    verifier, drafter, artifact_root = _require_environment()
    artifact_dir = _create_artifact_dir(artifact_root)
    port = int(os.environ.get("SPEEDLM_E2E_PORT", _free_port()))
    timeout = _ready_timeout()
    inject_delay = _inject_delay_seconds()
    if inject_delay > 0.0:
        logger.warning(
            "FAULT INJECTION ACTIVE: %s=%s adds %.1f ms to every ON-arm request",
            INJECT_ENV, os.environ[INJECT_ENV], inject_delay * 1000,
        )

    speculative_config = {
        "method": "eagle3",
        "num_speculative_tokens": 5,
        "model": drafter,
    }

    capture_root = artifact_dir / "captured"
    capture_root.mkdir(exist_ok=True)
    vllm_log = artifact_dir / "vllm.log"
    log_handle = vllm_log.open("wb")

    vllm_proc = subprocess.Popen(
        [
            str(VLLM_PYTHON),
            "-m", "vllm.entrypoints.cli.main",
            "serve",
            verifier,
            "--speculative_config",
            json.dumps(speculative_config),
            "--worker-extension-cls",
            "speedlm.activation_capture.hook.ActivationCaptureExtension",
            "--port",
            str(port),
            #: The long prompt plus MAX_TOKENS must fit; 512 (what the
            #: correctness test uses) does not leave room for 128 output tokens.
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "8",
            "--gpu-memory-utilization",
            "0.75",
            #: The same prompts are replayed ~40 times each.  With prefix
            #: caching on, every repetition after the first would skip the
            #: prefill entirely and TTFT would measure a cache hit rather than
            #: the forward pass that capture actually taxes.
            "--no-enable-prefix-caching",
            #: --enforce-eager is REQUIRED, and this is a limitation of the hook
            #: rather than a preference of this test.  ``activate_capture``
            #: works by calling ``set_aux_hidden_state_layers`` on the live
            #: model (hook.py:_extend_aux_layers), and the model consults that
            #: tuple inside its own forward -- ``EagleModelMixin.
            #: _maybe_add_hidden_state`` does a plain ``if layer_idx in
            #: self.aux_hidden_state_layers`` (vLLM
            #: model_executor/models/interfaces.py:1336).  Without this flag
            #: vLLM compiles (CompilationMode.VLLM_COMPILE) and CUDA-graph
            #: captures that forward ONCE at startup, with the three default
            #: eagle3 aux layers baked in, and never re-traces.  Arming capture
            #: afterwards then moves the *attribute* to four layers while the
            #: replayed graph keeps emitting three, and the very first forward
            #: dies in ``_buffer_aux``:
            #:
            #:   RuntimeError: the model reported 4 aux layers (2, 18, 33, 36)
            #:   but the forward produced 3 aux hidden states; the layer
            #:   labelling cannot be trusted
            #:
            #: killing EngineCore (observed: SLURM 370798).  So a graph-capturing
            #: engine cannot be capture-toggled at all today, and the overhead
            #: measured here is therefore eager-mode overhead.  Read it as the
            #: cost of capture in the only configuration capture currently runs
            #: in -- NOT as the cost it would have in a graphed engine.
            "--enforce-eager",
        ],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=str(REPO_ROOT),
        env=_vllm_env(),
    )

    url = f"http://127.0.0.1:{port}"
    off_samples: list[dict[str, Any]] = []
    on_samples: list[dict[str, Any]] = []
    armed_blocks = 0
    disarmed_blocks = 0

    try:
        _wait_for_ready(url, vllm_proc, timeout, log_path=vllm_log)
        served_model_id = _get_served_model_id(url)

        #: Start from a known-disarmed engine.  ``deactivate_capture`` is safe
        #: on a never-armed extension: ``_deactivate_impl`` no-ops when nothing
        #: was patched and ``_restore_aux_layers`` returns early while
        #: ``_final_layer_idx`` is None (hook.py:510-511).
        _collective_rpc(port, "deactivate_capture")

        with httpx.Client(
            timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS), trust_env=False
        ) as client:
            # Shared warmup, before the first measured repetition of either arm:
            # weights are resident but CUDA graphs, the sampler and the drafter
            # all pay a first-touch cost that belongs to neither condition.
            for _ in range(SHARED_WARMUP_PASSES):
                _run_pass(
                    client, url, served_model_id, inject_delay_seconds=0.0
                )

            #: ABBA.  Half 0 runs (OFF, ON); half 1 runs (ON, OFF).  Opposite
            #: orderings in the two halves is what makes linear drift cancel
            #: rather than land on one arm.
            block_plan: tuple[tuple[int, tuple[str, ...]], ...] = (
                (0, ("off", "on")),
                (1, ("on", "off")),
            )

            for cycle in range(CYCLES):
                for half, arms in block_plan:
                    for arm in arms:
                        block = f"cycle{cycle}-half{half}-{arm}"
                        if arm == "on":
                            capture_dir = capture_root / block
                            capture_dir.mkdir(exist_ok=True)
                            _collective_rpc(
                                port, "activate_capture", str(capture_dir)
                            )
                        else:
                            _collective_rpc(port, "deactivate_capture")

                        # Per-condition warmup AFTER the toggle: installing and
                        # restoring the monkeypatch perturbs the first pass
                        # through it, and that perturbation belongs to neither
                        # measured condition.
                        for _ in range(PER_BLOCK_WARMUP_PASSES):
                            _run_pass(
                                client,
                                url,
                                served_model_id,
                                inject_delay_seconds=(
                                    inject_delay if arm == "on" else 0.0
                                ),
                            )

                        samples = _run_pass(
                            client,
                            url,
                            served_model_id,
                            inject_delay_seconds=(
                                inject_delay if arm == "on" else 0.0
                            ),
                        )
                        for sample in samples:
                            sample["cycle"] = cycle
                            sample["half"] = half
                            sample["arm"] = arm
                        if arm == "on":
                            on_samples.extend(samples)
                        else:
                            off_samples.extend(samples)

                        # Outside every timed region: prove the arm was what it
                        # claimed to be, and (for ON) drain the host-side buffer
                        # so it cannot grow across the whole run.
                        if arm == "on":
                            _collective_rpc(port, "flush_capture")
                            _assert_capture_armed(capture_dir, block)
                            armed_blocks += 1
                        else:
                            _assert_capture_disarmed(port, block)
                            disarmed_blocks += 1
    finally:
        try:
            _collective_rpc(port, "deactivate_capture")
        except Exception:  # noqa: BLE001 - teardown must not mask a real failure
            logger.warning("final deactivate_capture failed", exc_info=True)
        vllm_proc.terminate()
        try:
            vllm_proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            vllm_proc.kill()
            vllm_proc.wait()
        log_handle.close()

    # ---------------------------------------------------------------------
    # Summarize and WRITE BEFORE ASSERTING.  A failing bound must never be able
    # to discard an H100 measurement.
    # ---------------------------------------------------------------------
    summary = _summarize(off_samples, on_samples)
    results = {
        "verifier": verifier,
        "drafter": drafter,
        "max_tokens": MAX_TOKENS,
        "cycles": CYCLES,
        "reps_per_condition": REPS_PER_CONDITION,
        "prompts": [label for label, _ in PROMPTS],
        "shared_warmup_passes": SHARED_WARMUP_PASSES,
        "per_block_warmup_passes": PER_BLOCK_WARMUP_PASSES,
        "enforce_eager": True,
        "prefix_caching": False,
        "inject_delay_ms": inject_delay * 1000.0,
        "armed_blocks": armed_blocks,
        "disarmed_blocks": disarmed_blocks,
        "bounds": {
            "max_ttft_absolute_overhead_seconds": (
                MAX_TTFT_ABSOLUTE_OVERHEAD_SECONDS
            ),
            "max_throughput_percentage_loss": MAX_THROUGHPUT_PERCENTAGE_LOSS,
            "max_inter_token_percentage_overhead": (
                MAX_INTER_TOKEN_PERCENTAGE_OVERHEAD
            ),
            "max_completion_token_asymmetry_percent": (
                MAX_COMPLETION_TOKEN_ASYMMETRY_PERCENT
            ),
        },
        "summary": summary,
        "samples": {"off": off_samples, "on": on_samples},
    }
    results_path = artifact_dir / "capture_overhead.json"
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    text = _summary_text(summary)
    (artifact_dir / "capture_overhead.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {results_path}")

    expected_blocks = CYCLES * 2
    assert armed_blocks == expected_blocks, (
        f"only {armed_blocks} of {expected_blocks} ON blocks were verified as "
        f"armed"
    )
    assert disarmed_blocks == expected_blocks, (
        f"only {disarmed_blocks} of {expected_blocks} OFF blocks were verified "
        f"as disarmed"
    )
    assert summary["pairs"] == REPS_PER_CONDITION * len(PROMPTS), (
        f"expected {REPS_PER_CONDITION * len(PROMPTS)} pairs, got "
        f"{summary['pairs']}"
    )

    failures = _overhead_failures(summary)
    assert not failures, (
        "activation capture exceeded its serving-overhead budget:\n  - "
        + "\n  - ".join(failures)
        + f"\nfull measurement: {results_path}"
    )
