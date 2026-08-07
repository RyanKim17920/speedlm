"""Live matrix benchmark for speculative decoding inference configurations.

This is deliberately a matrix of *inference regimes*, not another measurement
of the one eager/concurrency-8/short-context regime used by the promotion gate.
Each of the twelve cells combines:

* eager or CUDA-graph execution;
* request concurrency 1, 8, or 32; and
* a short or long prompt band sampled from a declared workload.

Within every cell the reference and candidate drafts use the production gate
runner's mirrored two-block schedule (ABBA).  Every block starts a new vLLM
process, warms that fresh process once, and only then opens its Prometheus
measurement window.  No arm inherits a process or warm state from another.

The live test is opt-in because the full matrix starts 48 engines.  A pure
synthetic test exercises the same regression decision without a GPU, proving
that the harness has a real failing path instead of merely recording numbers.

WHERE THE PROMPTS COME FROM
---------------------------
Bands are percentile windows over a workload's own prompt-length distribution
(``tests/e2e/harness/workloads.py`` plus the manifests beside it), selected with
``SPEEDLM_E2E_WORKLOAD`` and defaulting to ``generic-chat``.  This replaces an
earlier scheme that took the eight shortest prompts truncated to 480 characters
as "short" and the eight longest self-tiled to exactly 6000 characters as
"long".  Both were extremes rather than samples, and the tiled long band was
repetition-padded text, which is maximally predictable and therefore flatters
any drafter.  Nothing here truncates or pads: a band prompt is verbatim corpus
text or the cell fails.

CI COVERAGE
-----------
Only the live matrix carries the ``e2e`` marker.  The band-construction, band
validation, preflight-refusal and regression-decision logic is covered by
marker-free tests at the bottom of this module, so ``-m "not e2e"`` -- what CI
runs -- selects them.  A module-level ``pytestmark`` used to deselect all of it,
including the synthetic regression control.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import httpx
import pytest

from speedlm.gate import runner as gate_runner
from speedlm.gate.metrics import compute_delta, parse_metrics
from speedlm.gateway.process import build_vllm_argv
from speedlm.profiles import ModelProfile, resolve_profile, resolve_speculative_tokens
from speedlm.tuner.composition import declared_draft_depth

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.e2e.harness import workloads  # noqa: E402

# NOTE: there is deliberately no module-level ``pytestmark``.  One used to live
# here, and it deselected the GPU-free tests below from every CI run -- the
# synthetic regression control included.  Only the live matrix is marked e2e.

VLLM_VENV = Path("/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm")
VLLM = VLLM_VENV / "bin" / "vllm"

EXECUTION_MODES: Final = ("eager", "cuda_graphs")
CONCURRENCIES: Final = (1, 8, 32)
CONTEXT_LENGTHS: Final = ("short", "long")
REPEATS_PER_DRAFT: Final = 4
BLOCKS_PER_DRAFT: Final = 2
MAX_MODEL_LEN: Final = 4096
MAX_TOKENS: Final = 128
WARMUP_REPEATS: Final = 1

DEFAULT_MAX_SLOWDOWN_PERCENT: Final = 5.0
DEFAULT_MAX_ACCEPTED_LENGTH_LOSS: Final = 0.25
CONFIDENCE_MULTIPLIER: Final = 1.96

#: Workload used when the launcher does not name one.  ``generic-chat`` is the
#: comparability anchor every previously published matrix number was taken on.
DEFAULT_WORKLOAD: Final = "generic-chat"

#: How far the *served* model's tokenizer may disagree with the tokenizer the
#: manifest's percentiles were computed under before the band is called
#: mislabelled.  The manifests declare their basis in
#: ``provenance.token_basis`` (currently openai/gpt-oss-20b); a different
#: verifier tokenizes the same text differently, and that difference is a
#: legitimate, bounded offset.  A ratio surprise larger than this is not an
#: offset, it is a band that is not the band it claims to be, so the cell fails
#: rather than publishing a mislabelled number.
DEFAULT_BAND_TOKEN_TOLERANCE: Final = 0.35

#: A prompt whose longest proper border is at least this fraction of its length
#: is treated as repetition-padded.  Natural prose borders are a handful of
#: characters; ``(prompt * copies)[:6000]`` -- the padding this module used to
#: do -- has a border of thousands.  See :func:`_border_fraction`.
REPETITION_BORDER_LIMIT: Final = 0.25

ExecutionMode = Literal["eager", "cuda_graphs"]
ContextLength = Literal["short", "long"]
Arm = Literal["stock", "candidate"]
JsonObject = dict[str, Any]


@dataclass(frozen=True, slots=True)
class MatrixCell:
    execution_mode: ExecutionMode
    concurrency: int
    context_length: ContextLength

    @property
    def name(self) -> str:
        return f"{self.execution_mode}-c{self.concurrency}-{self.context_length}"


@dataclass(frozen=True, slots=True)
class LiveConfiguration:
    model: str
    reference_draft: str
    candidate_draft: str
    workload: str
    artifact_dir: Path
    startup_timeout: float
    request_timeout: float
    inject_slowdown_percent: float
    max_slowdown_percent: float
    max_accepted_length_loss: float
    max_model_len: int
    band_token_tolerance: float
    #: Only recorded.  The matrix no longer reads a raw corpus file; the
    #: workload manifest owns the corpus, its digest and its distribution.
    #: An un-updated launcher still sets ``SPEEDLM_E2E_PROMPT_CORPUS``, so the
    #: value is kept in the artifact rather than silently discarded.
    legacy_prompt_corpus: Path | None = None


MATRIX: Final = tuple(
    MatrixCell(mode, concurrency, context)
    for mode in EXECUTION_MODES
    for concurrency in CONCURRENCIES
    for context in CONTEXT_LENGTHS
)


def _number_from_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise AssertionError(f"{name} must be a number, got {raw!r}") from exc
    assert math.isfinite(value) and value >= 0.0, f"{name} must be finite and >= 0"
    return value


def _require_live_configuration() -> LiveConfiguration:
    if os.environ.get("SPEEDLM_E2E_CONFIG_MATRIX") != "1":
        pytest.skip("set SPEEDLM_E2E_CONFIG_MATRIX=1 inside an allocated GPU job")
    assert os.environ.get("SLURM_JOB_ID"), "configuration matrix must run under Slurm"
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible_devices and visible_devices != "-1", "Slurm exposed no GPU"
    assert VLLM.is_file(), f"missing vLLM executable: {VLLM}"

    def required(name: str) -> str:
        value = os.environ.get(name)
        assert value, f"{name} is required"
        return value

    raw_corpus = os.environ.get("SPEEDLM_E2E_PROMPT_CORPUS")
    legacy_prompt_corpus = (
        Path(raw_corpus).expanduser().resolve() if raw_corpus else None
    )
    workload = os.environ.get("SPEEDLM_E2E_WORKLOAD") or DEFAULT_WORKLOAD
    available = workloads.available_workloads()
    assert workload in available, (
        f"SPEEDLM_E2E_WORKLOAD={workload!r} is not a declared workload; "
        f"available: {', '.join(available)}"
    )
    artifact_dir = Path(required("SPEEDLM_CONFIG_MATRIX_ARTIFACT_DIR")).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return LiveConfiguration(
        model=required("SPEEDLM_CONFIG_MATRIX_MODEL"),
        reference_draft=required("SPEEDLM_CONFIG_MATRIX_REFERENCE_DRAFT"),
        candidate_draft=required("SPEEDLM_CONFIG_MATRIX_CANDIDATE_DRAFT"),
        workload=workload,
        legacy_prompt_corpus=legacy_prompt_corpus,
        artifact_dir=artifact_dir,
        max_model_len=int(
            _number_from_env("SPEEDLM_CONFIG_MATRIX_MAX_MODEL_LEN", MAX_MODEL_LEN)
        ),
        band_token_tolerance=_number_from_env(
            "SPEEDLM_CONFIG_MATRIX_BAND_TOKEN_TOLERANCE",
            DEFAULT_BAND_TOKEN_TOLERANCE,
        ),
        startup_timeout=_number_from_env("SPEEDLM_CONFIG_MATRIX_STARTUP_TIMEOUT", 1800),
        request_timeout=_number_from_env("SPEEDLM_CONFIG_MATRIX_REQUEST_TIMEOUT", 600),
        inject_slowdown_percent=_number_from_env(
            "SPEEDLM_CONFIG_MATRIX_INJECT_SLOWDOWN_PERCENT", 0.0
        ),
        max_slowdown_percent=_number_from_env(
            "SPEEDLM_CONFIG_MATRIX_MAX_SLOWDOWN_PERCENT",
            DEFAULT_MAX_SLOWDOWN_PERCENT,
        ),
        max_accepted_length_loss=_number_from_env(
            "SPEEDLM_CONFIG_MATRIX_MAX_ACCEPTED_LENGTH_LOSS",
            DEFAULT_MAX_ACCEPTED_LENGTH_LOSS,
        ),
    )


def _resolve_matrix_depth(
    configuration: LiveConfiguration,
) -> tuple[ModelProfile, int]:
    """Resolve one profile-owned draft depth shared by both matrix arms."""
    profile = resolve_profile(served_model=configuration.model)
    drafts: dict[Arm, str] = {
        "stock": configuration.reference_draft,
        "candidate": configuration.candidate_draft,
    }
    resolved_by_arm = {
        arm: resolve_speculative_tokens(
            explicit=profile.num_speculative_tokens,
            drafter_declared=declared_draft_depth(draft),
        )
        for arm, draft in drafts.items()
    }
    reference_depth = resolved_by_arm["stock"]
    candidate_depth = resolved_by_arm["candidate"]
    assert reference_depth == candidate_depth, (
        "reference and candidate must be served at the same speculative depth; "
        f"resolved reference={reference_depth}, candidate={candidate_depth} "
        f"for profile {profile.name!r}"
    )
    return profile, reference_depth


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _distinct_prompts_needed(concurrency: int) -> int:
    """How many distinct prompts a cell must draw.

    ``_request_batch`` indexes ``prompts[(repeat * concurrency + index) % len]``,
    so a cell consumes ``concurrency`` prompts per repeat and walks
    ``REPEATS_PER_DRAFT`` scored repeats.  Anything smaller and a later repeat
    replays an earlier one's inputs, which turns independent repeats into
    correlated ones and understates the standard error.  The old code drew a
    fixed eight, so concurrency 8 replayed the identical eight every repeat and
    concurrency 32 replayed them four times *within* a single repeat.
    """
    return concurrency * REPEATS_PER_DRAFT


def _border_fraction(text: str) -> float:
    """Longest proper border of ``text``, as a fraction of its length.

    A border is a proper prefix that is also a suffix.  Text produced by tiling
    a shorter string -- ``((prompt + "\\n") * copies)[:6000]``, which is exactly
    what this module used to do to manufacture its "long" band -- has a border
    of at least ``len(text) - len(prompt + "\\n")``, so a large fraction.  Real
    prose borders are a few characters.  Computed with the KMP failure
    function, linear in the length of the text.
    """
    length = len(text)
    if length < 2:
        return 0.0
    failure = [0] * length
    candidate = 0
    for index in range(1, length):
        while candidate and text[index] != text[candidate]:
            candidate = failure[candidate - 1]
        if text[index] == text[candidate]:
            candidate += 1
        failure[index] = candidate
    return failure[-1] / length


def _band_prompt_defects(
    prompts: Sequence[str],
    sources: Sequence[str],
    *,
    requested: int,
) -> list[str]:
    """Every way this band's prompts fail to be verbatim, distinct corpus text.

    Pure and GPU-free on purpose: this is the check the whole fix rests on, so
    it is exercised directly by the tests at the bottom of this module rather
    than only through a live run nobody can reproduce on a laptop.
    """
    defects: list[str] = []
    if len(prompts) != requested:
        defects.append(
            f"band produced {len(prompts)} prompts but the cell requested {requested}"
        )
    distinct = len(set(prompts))
    if distinct != len(prompts):
        defects.append(
            f"band produced {len(prompts)} prompts of which only {distinct} are "
            f"distinct; repeats would replay identical inputs"
        )
    for index, (prompt, source) in enumerate(zip(prompts, sources, strict=True)):
        if prompt != source:
            defects.append(
                f"prompt {index} is not verbatim corpus text: sent {len(prompt)} "
                f"characters, the record holds {len(source)} "
                f"({'truncated' if len(prompt) < len(source) else 'altered or padded'})"
            )
            continue
        border = _border_fraction(prompt)
        if border >= REPETITION_BORDER_LIMIT:
            defects.append(
                f"prompt {index} repeats a shorter substring of itself: longest "
                f"proper border is {border:.0%} of its {len(prompt)} characters, at "
                f"or above the {REPETITION_BORDER_LIMIT:.0%} limit; repetition-padded "
                f"text is trivially predictable and flatters any drafter"
            )
    return defects


def _percentile_key(fraction: float) -> str:
    """The manifest percentile key for a band bound, or a loud failure.

    Band bounds and declared percentiles must line up exactly.  A band at
    ``[0.07, 0.25]`` has no declared ``p7``, and quietly rounding it to ``p5``
    would validate the measurement against a window it was not drawn from.
    """
    scaled = fraction * 100.0
    rounded = round(scaled)
    key = f"p{rounded}"
    assert abs(scaled - rounded) < 1e-9 and key in workloads.PERCENTILE_KEYS, (
        f"band bound {fraction} is not one of the manifest's declared percentiles "
        f"({', '.join(workloads.PERCENTILE_KEYS)}); the declared window cannot be "
        "looked up, so the measurement could not be validated against it"
    )
    return key


def _declared_token_window(
    spec: workloads.WorkloadSpec, band_name: str
) -> tuple[str, float, str, float]:
    """The band's declared prompt-token window, straight from the manifest."""
    band = spec.band(band_name)
    block = spec.characteristics["prompt_tokens"]
    assert isinstance(block, dict), (
        f"workload {spec.name!r} declares no prompt_tokens distribution; there is "
        "nothing to validate the served token counts against"
    )
    lower_key = _percentile_key(band.lower)
    upper_key = _percentile_key(band.upper)
    return lower_key, float(block[lower_key]), upper_key, float(block[upper_key])


def _preflight_refusals(
    spec: workloads.WorkloadSpec, *, max_model_len: int
) -> tuple[str, ...]:
    """Reasons this workload must not be served at this window.

    The matrix sends a single user turn with no tool schemas, so ``tool_support``
    is honestly False.  Three of the four shipped workloads need a window far
    above the matrix default of 4096 -- 18432, 23552 and 360960 -- and the
    correct response is to refuse, not to trim the workload until it fits.  A
    truncated corpus still produces numbers; they are just not numbers about
    that workload.
    """
    return workloads.preflight_refusals(
        spec, max_model_len=max_model_len, tool_support=False
    )


def _band_plan(
    spec: workloads.WorkloadSpec,
    cell: MatrixCell,
    chosen: Sequence[workloads.WorkloadRecord],
    prompts: Sequence[str],
    *,
    token_tolerance: float,
    max_model_len: int,
) -> JsonObject:
    """Everything about this cell's measurement input, for the artifact.

    A measurement whose input is not recorded is not reproducible, so the
    workload name, its manifest version, the corpus digest, the percentile
    window and the exact record ids all land in the cell's ``result.json``.
    """
    band = spec.band(cell.context_length)
    lower_key, lower, upper_key, upper = _declared_token_window(spec, cell.context_length)
    corpus_tokens = [record.prompt_tokens for record in chosen]
    corpus_chars = [record.prompt_chars for record in chosen]
    return {
        "workload": spec.name,
        "workload_version": spec.version,
        "workload_domain": spec.domain,
        "manifest_path": str(spec.manifest_path),
        "manifest_schema_version": int(spec.manifest["schema_version"]),
        "corpus_path": str(spec.source_path),
        "corpus_sha256": spec.source_sha256,
        "corpus_size_bytes": spec.source_size_bytes,
        "corpus_verified_at_cell_start": True,
        "band": cell.context_length,
        "band_metric": spec.band_metric,
        "percentile_window": [band.lower, band.upper],
        "declared_prompt_tokens": {
            "lower_percentile": lower_key,
            "lower": lower,
            "upper_percentile": upper_key,
            "upper": upper,
        },
        "token_tolerance": token_tolerance,
        "token_basis": spec.provenance.get("token_basis"),
        "max_model_len": max_model_len,
        "requested_prompts": _distinct_prompts_needed(cell.concurrency),
        "distinct_prompts": len(set(prompts)),
        "sampling_seed": spec.seed,
        "record_ids": [record.id for record in chosen],
        "corpus_prompt_tokens": {
            "n": len(corpus_tokens),
            "mean": math.fsum(corpus_tokens) / len(corpus_tokens),
            "min": min(corpus_tokens),
            "max": max(corpus_tokens),
        },
        "corpus_prompt_chars": {
            "n": len(corpus_chars),
            "mean": math.fsum(corpus_chars) / len(corpus_chars),
            "min": min(corpus_chars),
            "max": max(corpus_chars),
        },
    }


def _cell_band(
    spec: workloads.WorkloadSpec,
    cell: MatrixCell,
    *,
    token_tolerance: float,
    max_model_len: int,
) -> tuple[tuple[str, ...], JsonObject]:
    """Verify the corpus and draw this cell's band.  Called at cell start.

    ``verify_workload`` re-derives the manifest from the file on disk -- digest,
    per-record character counts, every declared percentile and fraction -- so a
    corpus that was swapped, regenerated or truncated between cells fails the
    cell instead of quietly changing what was measured.
    """
    refusals = _preflight_refusals(spec, max_model_len=max_model_len)
    assert not refusals, f"{cell.name}: " + "; ".join(refusals)

    records = workloads.verify_workload(spec)
    requested = _distinct_prompts_needed(cell.concurrency)
    try:
        chosen = workloads.band_records(
            spec, cell.context_length, requested, records=records
        )
    except workloads.BandExhaustedError as error:
        raise AssertionError(f"{cell.name}: {error}") from error

    by_id = {record.id: record for record in records}
    prompts = tuple(record.final_user_text for record in chosen)
    sources = tuple(by_id[record.id].final_user_text for record in chosen)
    defects = _band_prompt_defects(prompts, sources, requested=requested)
    assert not defects, f"{cell.name} band {cell.context_length!r}: " + "; ".join(defects)

    plan = _band_plan(
        spec,
        cell,
        chosen,
        prompts,
        token_tolerance=token_tolerance,
        max_model_len=max_model_len,
    )
    return prompts, plan


def _option_value(argv: Sequence[str], option: str) -> str | None:
    for index, argument in enumerate(argv):
        if argument == option:
            assert index + 1 < len(argv), f"{option} has no value in argv"
            return argv[index + 1]
        if argument.startswith(f"{option}="):
            return argument.split("=", 1)[1]
    return None


def _engine_regime(argv: Sequence[str]) -> JsonObject:
    """Describe the engine from the argv actually passed to Popen."""
    speculative_raw = _option_value(argv, "--speculative-config")
    assert speculative_raw is not None, "engine argv omitted --speculative-config"
    speculative: object = json.loads(speculative_raw)
    assert isinstance(speculative, dict), "--speculative-config is not a JSON object"
    max_model_len = _option_value(argv, "--max-model-len")
    assert max_model_len is not None, "engine argv omitted --max-model-len"
    return {
        "argv": list(argv),
        "argv_shell": shlex.join(argv),
        "execution_mode": "eager" if "--enforce-eager" in argv else "cuda_graphs",
        "max_model_len": int(max_model_len),
        "gpu_memory_utilization": float(
            _option_value(argv, "--gpu-memory-utilization") or "nan"
        ),
        "prefix_caching": "--no-enable-prefix-caching" not in argv,
        "model": argv[2] if len(argv) > 2 else None,
        "draft": speculative.get("model"),
        "speculative_method": speculative.get("method"),
        "num_speculative_tokens": speculative.get("num_speculative_tokens"),
    }


def _engine_argv(
    model: str,
    draft: str,
    cell: MatrixCell,
    *,
    port: int,
    num_speculative_tokens: int,
    profile: ModelProfile,
    max_model_len: int = MAX_MODEL_LEN,
) -> list[str]:
    speculative = json.dumps(
        {
            "method": "eagle3",
            "model": draft,
            "num_speculative_tokens": num_speculative_tokens,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    passthrough = [
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        "0.75",
        "--no-enable-prefix-caching",
        "--speculative-config",
        speculative,
    ]
    if cell.execution_mode == "eager":
        passthrough.append("--enforce-eager")
    argv = build_vllm_argv(
        model,
        passthrough,
        host="127.0.0.1",
        port=port,
        executable=str(VLLM),
    )
    regime = _engine_regime(argv)
    assert regime["execution_mode"] == cell.execution_mode
    assert regime["max_model_len"] == max_model_len
    assert regime["num_speculative_tokens"] == profile.num_speculative_tokens, (
        f"engine argv would serve {regime['num_speculative_tokens']} speculative "
        f"tokens, but model profile {profile.name!r} declares "
        f"{profile.num_speculative_tokens}"
    )
    return argv


def _wait_ready(url: str, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "not attempted"
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            returncode = process.poll()
            assert returncode is None, f"vLLM exited before readiness with code {returncode}"
            try:
                response = client.get(f"{url}/health")
                if 200 <= response.status_code < 300:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            except httpx.HTTPError as exc:
                last_error = repr(exc)
            time.sleep(0.25)
    raise AssertionError(f"vLLM was not ready within {timeout:g}s: {last_error}")


@contextmanager
def _fresh_engine(
    argv: Sequence[str],
    log_path: Path,
    *,
    startup_timeout: float,
) -> Iterator[str]:
    url = f"http://127.0.0.1:{_option_value(argv, '--port')}"
    env = os.environ.copy()
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            argv,
            cwd=REPO_ROOT,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            _wait_ready(url, process, startup_timeout)
            yield url
        except BaseException:
            log.flush()
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            if tail:
                print("\n".join(tail))
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)


async def _request_batch(
    url: str,
    model: str,
    prompts: Sequence[str],
    *,
    concurrency: int,
    repeat: int,
    timeout: float,
) -> JsonObject:
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(timeout),
        trust_env=False,
        limits=httpx.Limits(max_connections=concurrency),
    ) as client:

        async def request(index: int) -> JsonObject:
            prompt = prompts[(repeat * concurrency + index) % len(prompts)]
            response = await client.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "seed": 1000 + index,
                    "max_tokens": MAX_TOKENS,
                },
            )
            response.raise_for_status()
            body: object = response.json()
            assert isinstance(body, dict), "chat response is not an object"
            usage = body.get("usage")
            assert isinstance(usage, dict), f"chat response has no usage: {body}"
            prompt_tokens = usage.get("prompt_tokens")
            completion_tokens = usage.get("completion_tokens")
            assert isinstance(prompt_tokens, int) and prompt_tokens > 0
            assert isinstance(completion_tokens, int) and completion_tokens > 0
            return {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }

        started = time.perf_counter()
        responses = await asyncio.gather(*(request(index) for index in range(concurrency)))
        elapsed = time.perf_counter() - started
    completion_tokens = sum(int(response["completion_tokens"]) for response in responses)
    return {
        "elapsed_seconds": elapsed,
        "batch_tokens_per_second": completion_tokens / elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": [int(response["prompt_tokens"]) for response in responses],
    }


def _scrape(url: str, timeout: float) -> str:
    with httpx.Client(timeout=timeout, trust_env=False) as client:
        response = client.get(f"{url}/metrics")
    response.raise_for_status()
    return response.text


def _mean_standard_error(values: Sequence[float]) -> JsonObject:
    assert values, "cannot summarize an empty sample"
    count = len(values)
    mean = math.fsum(values) / count
    if count < 2:
        standard_error = 0.0
    else:
        variance = math.fsum((value - mean) ** 2 for value in values) / (count - 1)
        standard_error = math.sqrt(variance) / math.sqrt(count)
    return {"n": count, "mean": mean, "standard_error": standard_error}


def _summarize_cell(
    samples: dict[Arm, list[JsonObject]],
    *,
    inject_slowdown_percent: float = 0.0,
) -> JsonObject:
    reference = samples["stock"]
    candidate = samples["candidate"]
    assert len(reference) == len(candidate) == REPEATS_PER_DRAFT
    scale = 1.0 - inject_slowdown_percent / 100.0
    assert scale >= 0.0, "injected slowdown cannot exceed 100%"

    arms: JsonObject = {}
    for arm, arm_samples in samples.items():
        arms[arm] = {
            "tokens_per_second": _mean_standard_error(
                [float(sample["tokens_per_second"]) for sample in arm_samples]
            ),
            "mean_accepted_length": _mean_standard_error(
                [float(sample["mean_accepted_length"]) for sample in arm_samples]
            ),
            "batch_tokens_per_second": _mean_standard_error(
                [float(sample["batch_tokens_per_second"]) for sample in arm_samples]
            ),
            "prompt_tokens": _mean_standard_error(
                [
                    float(prompt_tokens)
                    for sample in arm_samples
                    for prompt_tokens in sample["prompt_tokens"]
                ]
            ),
        }

    throughput_losses = []
    accepted_length_losses = []
    for reference_sample, candidate_sample in zip(reference, candidate, strict=True):
        reference_tps = float(reference_sample["tokens_per_second"])
        candidate_tps = float(candidate_sample["tokens_per_second"]) * scale
        throughput_losses.append((reference_tps - candidate_tps) / reference_tps * 100.0)
        accepted_length_losses.append(
            float(reference_sample["mean_accepted_length"])
            - float(candidate_sample["mean_accepted_length"])
        )
    return {
        "arms": arms,
        "paired_candidate_loss": {
            "tokens_per_second_percent": _mean_standard_error(throughput_losses),
            "mean_accepted_length": _mean_standard_error(accepted_length_losses),
        },
        "fault_injection": {
            "candidate_slowdown_percent": inject_slowdown_percent,
            "applied_to": "decision input only; raw arm measurements are unchanged",
        },
    }


def _regression_failures(
    summary: JsonObject,
    *,
    max_slowdown_percent: float,
    max_accepted_length_loss: float,
) -> list[str]:
    paired = summary["paired_candidate_loss"]
    assert isinstance(paired, dict)
    limits = (
        ("tokens_per_second_percent", max_slowdown_percent, "%"),
        ("mean_accepted_length", max_accepted_length_loss, " tokens"),
    )
    failures: list[str] = []
    for metric, limit, unit in limits:
        statistic = paired[metric]
        assert isinstance(statistic, dict)
        mean = float(statistic["mean"])
        standard_error = float(statistic["standard_error"])
        lower_confidence_bound = mean - CONFIDENCE_MULTIPLIER * standard_error
        if lower_confidence_bound > limit:
            failures.append(
                f"candidate {metric} loss {mean:.3f}{unit} +/- "
                f"{standard_error:.3f}{unit} SE has 95% lower bound "
                f"{lower_confidence_bound:.3f}{unit}, above limit {limit:.3f}{unit}"
            )
    return failures


def _band_token_bounds(plan: JsonObject) -> tuple[float, float]:
    """The declared window widened by the tokenizer tolerance."""
    declared = plan["declared_prompt_tokens"]
    assert isinstance(declared, dict)
    tolerance = float(plan["token_tolerance"])
    return (
        float(declared["lower"]) / (1.0 + tolerance),
        float(declared["upper"]) * (1.0 + tolerance),
    )


def _validate_context_band(cell: MatrixCell, summary: JsonObject, plan: JsonObject) -> None:
    """The served band must be the band the manifest says it is.

    The bounds are the workload's OWN declared percentile window for this band,
    not constants.  The constants that used to sit here (short at or below 256
    tokens, long between 768 and ``MAX_MODEL_LEN - MAX_TOKENS``) encoded the
    faked bands: they were satisfied by the eight shortest prompts truncated to
    480 characters and by text tiled to 6000 characters, and they would have
    been satisfied by almost anything else too.

    What is kept from the old check is the virtue that made it worth having: it
    compares the SERVER-reported ``usage.prompt_tokens`` against the expectation,
    so a verifier whose tokenizer disagrees with the manifest's basis by more
    than the stated tolerance fails the cell instead of publishing a number
    under a band label that does not describe it.
    """
    declared = plan["declared_prompt_tokens"]
    assert isinstance(declared, dict)
    lower, upper = _band_token_bounds(plan)
    arms = summary["arms"]
    assert isinstance(arms, dict)
    for arm in ("stock", "candidate"):
        arm_summary = arms[arm]
        assert isinstance(arm_summary, dict)
        prompt_statistic = arm_summary["prompt_tokens"]
        assert isinstance(prompt_statistic, dict)
        mean = float(prompt_statistic["mean"])
        assert lower <= mean <= upper, (
            f"{cell.name}/{arm}: the server reported a mean of {mean:.1f} prompt "
            f"tokens, outside workload {plan['workload']!r} v{plan['workload_version']} "
            f"band {plan['band']!r}, whose declared "
            f"{declared['lower_percentile']}..{declared['upper_percentile']} window is "
            f"[{declared['lower']:.0f}, {declared['upper']:.0f}] tokens "
            f"(basis {plan['token_basis']!r}) and widens to [{lower:.1f}, {upper:.1f}] "
            f"at the {float(plan['token_tolerance']):.0%} tokenizer tolerance. "
            f"The corpus-side mean for the same draw is "
            f"{float(plan['corpus_prompt_tokens']['mean']):.1f} tokens. Either the "
            f"served tokenizer is not comparable to the manifest's, or the prompts "
            f"that reached the server are not the ones this band drew."
        )


def _measure_cell(
    configuration: LiveConfiguration,
    profile: ModelProfile,
    num_speculative_tokens: int,
    cell: MatrixCell,
    spec: workloads.WorkloadSpec,
    cell_dir: Path,
) -> JsonObject:
    # Cell start: re-verify the corpus against its manifest and draw this cell's
    # band.  Done per cell, not once per run, so a corpus that changes underneath
    # a multi-hour matrix fails the next cell rather than silently shifting what
    # the remaining cells measured.
    prompts, band_plan = _cell_band(
        spec,
        cell,
        token_tolerance=configuration.band_token_tolerance,
        max_model_len=configuration.max_model_len,
    )
    _write_json(cell_dir / "band.json", band_plan)

    # This private helper is intentionally reused: it is the production gate's
    # canonical mirrored-round schedule and is the design this harness promises.
    schedule = gate_runner._block_schedule(  # noqa: SLF001
        "stock", blocks=BLOCKS_PER_DRAFT, repeats=REPEATS_PER_DRAFT
    )
    samples: dict[Arm, list[JsonObject]] = {"stock": [], "candidate": []}
    blocks: list[JsonObject] = []
    drafts: dict[Arm, str] = {
        "stock": configuration.reference_draft,
        "candidate": configuration.candidate_draft,
    }
    logical_repeat: dict[Arm, int] = {"stock": 0, "candidate": 0}

    for block_index, (raw_arm, block_repeats) in enumerate(schedule):
        if raw_arm == "stock":
            arm: Arm = "stock"
        else:
            assert raw_arm == "candidate"
            arm = "candidate"
        port = _free_port()
        argv = _engine_argv(
            configuration.model,
            drafts[arm],
            cell,
            port=port,
            num_speculative_tokens=num_speculative_tokens,
            profile=profile,
            max_model_len=configuration.max_model_len,
        )
        regime = _engine_regime(argv)
        block_dir = cell_dir / f"block-{block_index:02d}-{arm}"
        block_dir.mkdir(parents=True)
        _write_json(block_dir / "engine-regime.json", regime)
        block_record: JsonObject = {
            "block_index": block_index,
            "arm": arm,
            "scored_repeats": block_repeats,
            "fresh_process": True,
            "warmup_repeats": WARMUP_REPEATS,
            "engine_regime": regime,
        }
        blocks.append(block_record)
        with _fresh_engine(
            argv,
            block_dir / "vllm.log",
            startup_timeout=configuration.startup_timeout,
        ) as url:
            for warmup_index in range(WARMUP_REPEATS):
                asyncio.run(
                    _request_batch(
                        url,
                        configuration.model,
                        prompts,
                        concurrency=cell.concurrency,
                        repeat=warmup_index,
                        timeout=configuration.request_timeout,
                    )
                )
            before = _scrape(url, configuration.request_timeout)
            (block_dir / "metrics-before.prom").write_text(before, encoding="utf-8")
            for _ in range(block_repeats):
                repeat = logical_repeat[arm]
                batch = asyncio.run(
                    _request_batch(
                        url,
                        configuration.model,
                        prompts,
                        concurrency=cell.concurrency,
                        repeat=repeat,
                        timeout=configuration.request_timeout,
                    )
                )
                after = _scrape(url, configuration.request_timeout)
                (block_dir / f"metrics-after-repeat-{repeat}.prom").write_text(
                    after, encoding="utf-8"
                )
                delta = compute_delta(parse_metrics(before), parse_metrics(after))
                assert delta.output_tok_per_sec > 0.0, "metrics reported zero throughput"
                assert delta.accepted_length_available, (
                    "speculative counters or mean accepted length were unavailable"
                )
                sample: JsonObject = {
                    "repeat": repeat,
                    "tokens_per_second": delta.output_tok_per_sec,
                    "mean_accepted_length": delta.mean_accepted_length,
                    **batch,
                }
                samples[arm].append(sample)
                before = after
                logical_repeat[arm] += 1

    summary = _summarize_cell(
        samples,
        inject_slowdown_percent=configuration.inject_slowdown_percent,
    )
    report: JsonObject = {
        "cell": {
            "name": cell.name,
            "execution_mode": cell.execution_mode,
            "request_concurrency": cell.concurrency,
            "context_length": cell.context_length,
            "resolved_num_speculative_tokens": num_speculative_tokens,
        },
        "band": band_plan,
        "schedule": {
            "design": "mirrored rounds from speedlm.gate.runner._block_schedule",
            "arm_blocks": BLOCKS_PER_DRAFT,
            "repeats_per_arm": REPEATS_PER_DRAFT,
            "blocks": blocks,
        },
        "samples": samples,
        "summary": summary,
    }
    _write_json(cell_dir / "result.json", report)
    _validate_context_band(cell, summary, band_plan)
    failures = _regression_failures(
        summary,
        max_slowdown_percent=configuration.max_slowdown_percent,
        max_accepted_length_loss=configuration.max_accepted_length_loss,
    )
    report["regression_failures"] = failures
    report["passed"] = not failures
    _write_json(cell_dir / "result.json", report)
    return report


def _synthetic_samples(
    reference_tps: Sequence[float],
    candidate_tps: Sequence[float],
) -> dict[Arm, list[JsonObject]]:
    assert len(reference_tps) == len(candidate_tps) == REPEATS_PER_DRAFT

    def arm(values: Sequence[float]) -> list[JsonObject]:
        return [
            {
                "repeat": index,
                "tokens_per_second": value,
                "mean_accepted_length": 3.0,
                "batch_tokens_per_second": value,
                "prompt_tokens": [128],
            }
            for index, value in enumerate(values)
        ]

    return {"stock": arm(reference_tps), "candidate": arm(candidate_tps)}


def test_synthetic_throughput_regression_is_detected() -> None:
    """A known 20% candidate slowdown must exercise the failing decision path."""
    summary = _summarize_cell(
        _synthetic_samples(
            [100.0, 101.0, 99.0, 100.0],
            [100.0, 101.0, 99.0, 100.0],
        ),
        inject_slowdown_percent=20.0,
    )
    failures = _regression_failures(
        summary,
        max_slowdown_percent=DEFAULT_MAX_SLOWDOWN_PERCENT,
        max_accepted_length_loss=DEFAULT_MAX_ACCEPTED_LENGTH_LOSS,
    )
    assert len(failures) == 1
    assert "tokens_per_second_percent" in failures[0]
    with pytest.raises(AssertionError, match="tokens_per_second_percent"):
        assert not failures, "; ".join(failures)


@pytest.mark.e2e
def test_speculative_inference_configuration_matrix() -> None:
    configuration = _require_live_configuration()
    profile, num_speculative_tokens = _resolve_matrix_depth(configuration)
    spec = workloads.load_spec(configuration.workload)

    # Refuse before spending an allocation, and name both numbers.  A workload
    # that does not fit the serving window is not shrunk to fit it.
    refusals = _preflight_refusals(spec, max_model_len=configuration.max_model_len)
    assert not refusals, (
        "the configuration matrix refuses this workload:\n  "
        + "\n  ".join(refusals)
        + "\nRaise SPEEDLM_CONFIG_MATRIX_MAX_MODEL_LEN to at least "
        f"{spec.requirements.get('min_max_model_len')} (and allocate the memory for "
        "it), or select a workload that fits the window with SPEEDLM_E2E_WORKLOAD."
    )

    manifest: JsonObject = {
        "model": configuration.model,
        "profile": profile.name,
        "reference_draft": configuration.reference_draft,
        "candidate_draft": configuration.candidate_draft,
        "workload": {
            "name": spec.name,
            "version": spec.version,
            "domain": spec.domain,
            "manifest_path": str(spec.manifest_path),
            "manifest_schema_version": int(spec.manifest["schema_version"]),
            "corpus_path": str(spec.source_path),
            "corpus_sha256": spec.source_sha256,
            "corpus_size_bytes": spec.source_size_bytes,
            "record_count": spec.characteristics.get("record_count"),
            "token_basis": spec.provenance.get("token_basis"),
            "band_metric": spec.band_metric,
            "sampling_seed": spec.seed,
            "percentile_windows": {
                name: [
                    spec.band(name).lower,
                    spec.band(name).upper,
                ]
                for name in CONTEXT_LENGTHS
            },
            "declared_prompt_tokens_windows": {
                name: dict(
                    zip(
                        ("lower_percentile", "lower", "upper_percentile", "upper"),
                        _declared_token_window(spec, name),
                        strict=True,
                    )
                )
                for name in CONTEXT_LENGTHS
            },
            "band_token_tolerance": configuration.band_token_tolerance,
            "distinct_prompts_per_cell": {
                str(concurrency): _distinct_prompts_needed(concurrency)
                for concurrency in CONCURRENCIES
            },
            "requirements": dict(spec.requirements),
            "preflight_refusals": list(refusals),
            "honest_scope": (
                "Bands are percentile windows over this workload's own prompt-length "
                "distribution: neither truncated nor repetition-padded, and never the "
                "corpus extremes. On generic-chat the p75..p95 'long' band is roughly "
                "150-600 prompt tokens, so a generic-chat cell does NOT exercise a "
                "long-context regime -- it exercises the long tail of short chat. "
                "Genuinely long context needs agentic-tool-loop (18432), "
                "agentic-mixed-outcome (23552) or long-context-sessions (360960), all "
                "of which this launch would refuse at "
                f"--max-model-len {configuration.max_model_len}."
            ),
        },
        "legacy_prompt_corpus": (
            str(configuration.legacy_prompt_corpus)
            if configuration.legacy_prompt_corpus
            else None
        ),
        "matrix": [cell.name for cell in MATRIX],
        "matrix_dimensions": {
            "execution_modes": list(EXECUTION_MODES),
            "request_concurrency": list(CONCURRENCIES),
            "context_lengths": list(CONTEXT_LENGTHS),
        },
        "measurement": {
            "repeats_per_draft": REPEATS_PER_DRAFT,
            "blocks_per_draft": BLOCKS_PER_DRAFT,
            "warmup_repeats_per_block": WARMUP_REPEATS,
            "max_tokens": MAX_TOKENS,
            "resolved_num_speculative_tokens": num_speculative_tokens,
            "throughput_statistic": "Prometheus generation tokens / decode seconds",
            "accepted_length_statistic": "1 + accepted tokens / draft steps",
            "max_model_len": configuration.max_model_len,
        },
        "fault_injection": {
            "candidate_slowdown_percent": configuration.inject_slowdown_percent,
        },
    }
    _write_json(configuration.artifact_dir / "manifest.json", manifest)

    reports: list[JsonObject] = []
    failures: list[str] = []
    for cell in MATRIX:
        cell_dir = configuration.artifact_dir / cell.name
        cell_dir.mkdir(parents=True, exist_ok=True)
        report = _measure_cell(
            configuration,
            profile,
            num_speculative_tokens,
            cell,
            spec,
            cell_dir,
        )
        reports.append(report)
        cell_failures = report["regression_failures"]
        assert isinstance(cell_failures, list)
        failures.extend(f"{cell.name}: {failure}" for failure in cell_failures)
        _write_json(
            configuration.artifact_dir / "matrix-result.json",
            {"manifest": manifest, "cells": reports, "failures": failures},
        )

    assert not failures, "\n".join(failures)


# ---------------------------------------------------------------------------
# GPU-free coverage of the band machinery
#
# These carry no marker, so `-m "not e2e"` -- what CI runs -- selects them.
# Every one of them has been driven RED by mutating the code it covers; the
# mutation and the resulting message are recorded in each docstring.  A test
# whose failure was never observed is not evidence.
# ---------------------------------------------------------------------------


_FIXTURE_WORDS: Final = (
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
)


def _fixture_body(index: int) -> str:
    """Non-repeating prose whose length increases strictly with ``index``.

    Every word carries its own position, so the text has no internal period and
    a border fraction of essentially zero.  A fixture built from ``"word " * n``
    would itself be repetition-padded and could not distinguish the padding
    detector working from it being broken.
    """
    words = [
        f"{_FIXTURE_WORDS[(index * 7 + position) % len(_FIXTURE_WORDS)]}{position}"
        for position in range(index + 1)
    ]
    return f"question {index:05d}: " + " ".join(words)


def _build_fixture_workload(
    tmp_path: Path, *, count: int, name: str = "fixture-matrix"
) -> workloads.WorkloadSpec:
    """A real workload built by the real builder, small enough to live in tmp_path."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import build_workload as builder  # noqa: PLC0415
    finally:
        sys.path.pop(0)

    tokenizer = builder.Tokenizer("fixture", chars_per_token=4.0)
    records = [
        builder.make_record(
            f"fixture-{index:05d}",
            [{"role": "user", "content": _fixture_body(index)}],
            tokenizer,
        )
        for index in range(count)
    ]
    records_path = tmp_path / "corpora" / name / "records.jsonl"
    builder.write_records(records_path, records)
    spec_dir = tmp_path / "specs"
    builder.emit_manifest(
        name=name,
        version="1.0.0",
        description="synthetic fixture workload for the configuration matrix",
        domain="fixture",
        records_path=records_path,
        records=records,
        provenance={"upstream": [], "method": "synthesized in a test"},
        tokenizer=tokenizer,
        output_reserve=MAX_TOKENS,
        needs_tool_support=False,
        ground_truth=None,
        spec_dir=spec_dir,
    )
    return workloads.load_spec(name, directory=spec_dir)


@pytest.fixture
def fixture_spec(tmp_path: Path) -> workloads.WorkloadSpec:
    # 800 records leaves 160 in each 20%-wide band, enough for the largest cell
    # (concurrency 32 x 4 repeats = 128 distinct prompts).
    return _build_fixture_workload(tmp_path, count=800)


def _cell(concurrency: int = 8, context: ContextLength = "short") -> MatrixCell:
    return MatrixCell("eager", concurrency, context)


def _summary_with_prompt_token_mean(mean: float) -> JsonObject:
    return {
        "arms": {
            arm: {"prompt_tokens": {"n": 8, "mean": mean, "standard_error": 0.0}}
            for arm in ("stock", "candidate")
        }
    }


def test_distinct_prompt_count_is_sized_from_concurrency_and_repeats() -> None:
    """A cell must draw enough prompts that no scored repeat replays another.

    MUTATION: ``return concurrency * REPEATS_PER_DRAFT`` -> ``return 8`` in
    ``_distinct_prompts_needed``.
    """
    for concurrency in CONCURRENCIES:
        needed = _distinct_prompts_needed(concurrency)
        assert needed == concurrency * REPEATS_PER_DRAFT, (
            f"concurrency {concurrency} would draw {needed} prompts, but "
            f"{REPEATS_PER_DRAFT} repeats of {concurrency} concurrent requests "
            f"consume {concurrency * REPEATS_PER_DRAFT}; the old fixed-8 draw "
            "replayed inputs"
        )
        # The index _request_batch actually reaches on the last scored repeat.
        highest_index = (REPEATS_PER_DRAFT - 1) * concurrency + (concurrency - 1)
        assert highest_index < needed, (
            f"concurrency {concurrency}: _request_batch reaches index "
            f"{highest_index} but only {needed} prompts were drawn, so it wraps "
            "and replays"
        )


def test_repetition_padding_is_rejected_and_real_prose_is_not() -> None:
    """The exact padding this module used to do must be caught, prose must not.

    MUTATION: ``REPETITION_BORDER_LIMIT = 0.25`` -> ``= 1.1`` (unreachable).
    """
    prose = _fixture_body(300)
    assert len(prose) < 6000, "fixture prose must be shorter than the old pad width"
    # Verbatim reproduction of the band this module used to manufacture:
    #     copies = math.ceil(6000 / len(prompt))
    #     ((prompt + "\n") * copies)[:6000]
    copies = math.ceil(6000 / len(prose))
    padded = ((prose + "\n") * copies)[:6000]

    padded_border = _border_fraction(padded)
    assert padded_border >= REPETITION_BORDER_LIMIT, (
        f"the 6000-char self-tiled prompt was accepted: border "
        f"{padded_border:.0%} is below the {REPETITION_BORDER_LIMIT:.0%} limit"
    )
    assert _border_fraction(prose) < REPETITION_BORDER_LIMIT, (
        "natural prose was misflagged as repetition-padded; the detector would "
        "reject honest corpora"
    )

    defects = _band_prompt_defects((padded,), (padded,), requested=1)
    assert any("repeats a shorter substring" in defect for defect in defects), (
        f"_band_prompt_defects did not report the padding: {defects}"
    )
    assert not _band_prompt_defects((prose,), (prose,), requested=1)


def test_band_prompt_defects_catch_truncation_and_recycling() -> None:
    """Truncation and duplicate prompts are defects, not silent adjustments.

    MUTATION: in ``_band_prompt_defects`` change ``if prompt != source:`` to
    ``if False:``.
    """
    source = _fixture_body(400)
    truncated = source[:480]  # the old "short" band recipe
    defects = _band_prompt_defects((truncated,), (source,), requested=1)
    assert any("not verbatim" in defect and "truncated" in defect for defect in defects), (
        f"truncation was not reported: {defects}"
    )

    duplicated = (source, source)
    defects = _band_prompt_defects(duplicated, duplicated, requested=2)
    assert any("distinct" in defect for defect in defects), (
        f"a recycled prompt was not reported: {defects}"
    )

    short_draw = (source,)
    defects = _band_prompt_defects(short_draw, short_draw, requested=4)
    assert any("requested 4" in defect for defect in defects), (
        f"an undersized band was not reported: {defects}"
    )


def test_band_is_verbatim_corpus_text(fixture_spec: workloads.WorkloadSpec) -> None:
    """Every prompt a cell sends is byte-identical to the record it came from.

    MUTATION: in ``_cell_band`` change
    ``prompts = tuple(record.final_user_text for record in chosen)`` to
    ``prompts = tuple(record.final_user_text[:480] for record in chosen)``.
    """
    records = workloads.load_records(fixture_spec)
    by_text = {record.final_user_text for record in records}
    for context in CONTEXT_LENGTHS:
        prompts, plan = _cell_band(
            fixture_spec,
            _cell(8, context),
            token_tolerance=DEFAULT_BAND_TOKEN_TOLERANCE,
            max_model_len=MAX_MODEL_LEN,
        )
        assert len(prompts) == 32
        assert len(set(prompts)) == 32
        for prompt in prompts:
            assert prompt in by_text, (
                "a band prompt is not any record's text; it was synthesized, "
                "truncated or padded somewhere between the corpus and the wire"
            )
            assert _border_fraction(prompt) < REPETITION_BORDER_LIMIT
        assert plan["corpus_sha256"] == fixture_spec.source_sha256
        assert plan["percentile_window"] == [
            fixture_spec.band(context).lower,
            fixture_spec.band(context).upper,
        ]


def test_short_band_is_not_the_shortest_records(
    fixture_spec: workloads.WorkloadSpec,
) -> None:
    """THE fix, stated as one assertion: a band is a window, never an extreme.

    The old code took the eight shortest prompts in the corpus.  A percentile
    window at p5..p25 must not reach the corpus floor -- or its ceiling.

    MUTATION: in ``_cell_band`` replace the ``workloads.band_records(...)`` call
    with ``chosen = tuple(sorted(records, key=lambda r: r.prompt_chars)[:requested])``
    -- i.e. restore the old extreme.
    """
    records = workloads.load_records(fixture_spec)
    corpus_min = min(record.prompt_chars for record in records)
    corpus_max = max(record.prompt_chars for record in records)

    short_prompts, short_plan = _cell_band(
        fixture_spec,
        _cell(8, "short"),
        token_tolerance=DEFAULT_BAND_TOKEN_TOLERANCE,
        max_model_len=MAX_MODEL_LEN,
    )
    long_prompts, long_plan = _cell_band(
        fixture_spec,
        _cell(8, "long"),
        token_tolerance=DEFAULT_BAND_TOKEN_TOLERANCE,
        max_model_len=MAX_MODEL_LEN,
    )

    short_min = int(short_plan["corpus_prompt_chars"]["min"])
    long_max = int(long_plan["corpus_prompt_chars"]["max"])
    assert short_min > corpus_min, (
        f"short band reached the corpus floor: its shortest prompt is "
        f"{short_min} chars and the corpus minimum is {corpus_min}"
    )
    assert long_max < corpus_max, (
        f"long band reached the corpus ceiling: its longest prompt is "
        f"{long_max} chars and the corpus maximum is {corpus_max}"
    )
    assert short_plan["corpus_prompt_chars"]["max"] < long_plan["corpus_prompt_chars"]["min"], (
        "the short and long bands overlap; they are not separated windows"
    )
    assert set(short_prompts).isdisjoint(long_prompts)


def test_band_token_mean_is_validated_against_the_declared_window(
    fixture_spec: workloads.WorkloadSpec,
) -> None:
    """The served token mean must agree with the manifest, and disagreement fails.

    MUTATION: in ``_validate_context_band`` change
    ``assert lower <= mean <= upper`` to ``assert True``.
    """
    cell = _cell(8, "short")
    _, plan = _cell_band(
        fixture_spec,
        cell,
        token_tolerance=DEFAULT_BAND_TOKEN_TOLERANCE,
        max_model_len=MAX_MODEL_LEN,
    )
    declared = plan["declared_prompt_tokens"]
    corpus_mean = float(plan["corpus_prompt_tokens"]["mean"])

    # The band the workload actually produced passes.
    _validate_context_band(cell, _summary_with_prompt_token_mean(corpus_mean), plan)

    # A mean far outside the declared window must fail, and say both numbers.
    outside = float(declared["upper"]) * 20.0
    with pytest.raises(AssertionError) as raised:
        _validate_context_band(cell, _summary_with_prompt_token_mean(outside), plan)
    message = str(raised.value)
    assert f"{outside:.1f}" in message and str(int(declared["upper"])) in message, (
        f"the failure message did not name both numbers: {message}"
    )

    with pytest.raises(AssertionError):
        _validate_context_band(
            cell, _summary_with_prompt_token_mean(float(declared["lower"]) / 100.0), plan
        )

    lower_bound, upper_bound = _band_token_bounds(plan)
    assert lower_bound < float(declared["lower"]) and upper_bound > float(declared["upper"]), (
        "the tolerance must widen the declared window, not narrow it"
    )
    assert lower_bound <= corpus_mean <= upper_bound, (
        f"a mean of {corpus_mean:.1f} tokens, drawn from the band itself, does not "
        f"survive its own declared window [{lower_bound:.1f}, {upper_bound:.1f}]; "
        "the guard would fail every honest run"
    )


def test_band_too_small_fails_loudly_instead_of_recycling(tmp_path: Path) -> None:
    """An exhausted band names the counts; it never tops itself up with duplicates.

    MUTATION: in ``_cell_band`` replace the ``except workloads.BandExhaustedError``
    body with a recycling draw (``chosen = (window * 64)[:requested]``).
    """
    tiny = _build_fixture_workload(tmp_path, count=60, name="fixture-tiny")
    window = workloads.band_window(tiny, "short", workloads.load_records(tiny))
    assert len(window) < _distinct_prompts_needed(32), "fixture is not small enough"

    with pytest.raises(AssertionError) as raised:
        _cell_band(
            tiny,
            _cell(32, "short"),
            token_tolerance=DEFAULT_BAND_TOKEN_TOLERANCE,
            max_model_len=MAX_MODEL_LEN,
        )
    message = str(raised.value)
    assert "eager-c32-short" in message
    assert "128" in message and str(len(window)) in message, (
        f"the refusal did not name the requested and available counts: {message}"
    )
    # Specifically the exhaustion refusal, not the distinctness backstop firing
    # after a recycled draw: the band must never be topped up in the first place.
    assert "percentile window holds only" in message, (
        f"the band was topped up and only caught downstream: {message}"
    )


def test_workload_too_large_for_the_window_is_refused_and_a_fitting_one_is_not() -> None:
    """Both halves: refuse what does not fit, admit what does.

    A workload is never shrunk to fit the serving window.  A truncated corpus
    still produces numbers; they are just not numbers about that workload.

    MUTATION: make ``_preflight_refusals`` ``return ()``.
    """
    fits = workloads.load_spec("generic-chat")
    assert int(fits.requirements["min_max_model_len"]) <= MAX_MODEL_LEN
    assert _preflight_refusals(fits, max_model_len=MAX_MODEL_LEN) == (), (
        "generic-chat fits the matrix window and must not be refused"
    )

    for name in ("agentic-tool-loop", "agentic-mixed-outcome", "long-context-sessions"):
        spec = workloads.load_spec(name)
        required = int(spec.requirements["min_max_model_len"])
        assert required > MAX_MODEL_LEN, f"{name} no longer exceeds the matrix window"
        refusals = _preflight_refusals(spec, max_model_len=MAX_MODEL_LEN)
        assert refusals, (
            f"{name} was admitted at --max-model-len {MAX_MODEL_LEN} although it "
            f"declares min_max_model_len {required}"
        )
        joined = " ".join(refusals)
        assert str(required) in joined and str(MAX_MODEL_LEN) in joined, (
            f"the refusal for {name} must name both numbers: {joined}"
        )
        # ...and raising the window to what the workload declares clears the
        # length refusal.  What may remain is the tool-support requirement,
        # which this matrix genuinely does not satisfy: it sends one user turn
        # with no tool schemas.
        remaining = _preflight_refusals(spec, max_model_len=required)
        assert all("tool" in reason for reason in remaining), (
            f"{name} still refuses on length at its own declared window: {remaining}"
        )


def test_declared_percentile_lookup_refuses_an_undeclared_bound() -> None:
    """A band bound with no matching declared percentile cannot be validated.

    MUTATION: in ``_percentile_key`` drop the
    ``and key in workloads.PERCENTILE_KEYS`` clause.
    """
    assert _percentile_key(0.05) == "p5"
    assert _percentile_key(0.95) == "p95"
    for undeclared in (0.07, 0.3, 0.125):
        with pytest.raises(AssertionError, match="declared percentiles"):
            _percentile_key(undeclared)


_GENERIC_CHAT_CORPUS = workloads.load_spec(DEFAULT_WORKLOAD).source_path


@pytest.mark.skipif(
    not _GENERIC_CHAT_CORPUS.is_file(),
    reason=f"generic-chat corpus is not on this machine: {_GENERIC_CHAT_CORPUS}",
)
def test_generic_chat_bands_are_honest_on_the_real_corpus() -> None:
    """The shipped default workload, end to end, on the real 22,362-record file.

    MUTATION: in ``_cell_band`` replace the draw with the shortest records
    (the old behaviour).
    """
    spec = workloads.load_spec(DEFAULT_WORKLOAD)
    records = workloads.verify_workload(spec)
    corpus_min = min(record.prompt_chars for record in records)
    corpus_max = max(record.prompt_chars for record in records)

    for context in CONTEXT_LENGTHS:
        cell = _cell(8, context)
        prompts, plan = _cell_band(
            spec,
            cell,
            token_tolerance=DEFAULT_BAND_TOKEN_TOLERANCE,
            max_model_len=MAX_MODEL_LEN,
        )
        assert len(set(prompts)) == 32
        assert int(plan["corpus_prompt_chars"]["min"]) > corpus_min, (
            f"generic-chat {context} band reached the corpus floor: shortest "
            f"drawn prompt is {plan['corpus_prompt_chars']['min']} chars, corpus "
            f"minimum is {corpus_min}"
        )
        assert int(plan["corpus_prompt_chars"]["max"]) < corpus_max
        for prompt in prompts:
            assert _border_fraction(prompt) < REPETITION_BORDER_LIMIT, (
                "a real UltraChat prompt looks repetition-padded; either the "
                "corpus is polluted or the detector is too strict"
            )
        # The corpus-side mean must survive the same guard the served mean faces.
        _validate_context_band(
            cell,
            _summary_with_prompt_token_mean(float(plan["corpus_prompt_tokens"]["mean"])),
            plan,
        )
