"""Live matrix benchmark for speculative decoding inference configurations.

This is deliberately a matrix of *inference regimes*, not another measurement
of the one eager/concurrency-8/short-context regime used by the promotion gate.
Each of the twelve cells combines:

* eager or CUDA-graph execution;
* request concurrency 1, 8, or 32; and
* short or long prompts drawn from the configured corpus.

Within every cell the reference and candidate drafts use the production gate
runner's mirrored two-block schedule (ABBA).  Every block starts a new vLLM
process, warms that fresh process once, and only then opens its Prometheus
measurement window.  No arm inherits a process or warm state from another.

The live test is opt-in because the full matrix starts 48 engines.  A pure
synthetic test exercises the same regression decision without a GPU, proving
that the harness has a real failing path instead of merely recording numbers.
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

pytestmark = pytest.mark.e2e

REPO_ROOT = Path(__file__).resolve().parents[2]
VLLM_VENV = Path("/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm")
VLLM = VLLM_VENV / "bin" / "vllm"

EXECUTION_MODES: Final = ("eager", "cuda_graphs")
CONCURRENCIES: Final = (1, 8, 32)
CONTEXT_LENGTHS: Final = ("short", "long")
REPEATS_PER_DRAFT: Final = 4
BLOCKS_PER_DRAFT: Final = 2
MAX_MODEL_LEN: Final = 4096
MAX_TOKENS: Final = 128
NUM_SPECULATIVE_TOKENS: Final = 5
WARMUP_REPEATS: Final = 1

DEFAULT_MAX_SLOWDOWN_PERCENT: Final = 5.0
DEFAULT_MAX_ACCEPTED_LENGTH_LOSS: Final = 0.25
CONFIDENCE_MULTIPLIER: Final = 1.96

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
    corpus: Path
    artifact_dir: Path
    startup_timeout: float
    request_timeout: float
    inject_slowdown_percent: float
    max_slowdown_percent: float
    max_accepted_length_loss: float


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

    corpus = Path(required("SPEEDLM_E2E_PROMPT_CORPUS")).expanduser().resolve()
    assert corpus.is_file(), f"prompt corpus is not a file: {corpus}"
    artifact_dir = Path(required("SPEEDLM_CONFIG_MATRIX_ARTIFACT_DIR")).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return LiveConfiguration(
        model=required("SPEEDLM_CONFIG_MATRIX_MODEL"),
        reference_draft=required("SPEEDLM_CONFIG_MATRIX_REFERENCE_DRAFT"),
        candidate_draft=required("SPEEDLM_CONFIG_MATRIX_CANDIDATE_DRAFT"),
        corpus=corpus,
        artifact_dir=artifact_dir,
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


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_corpus(path: Path, *, limit: int = 2048) -> list[str]:
    prompts: list[str] = []
    with path.open(encoding="utf-8") as stream:
        for raw_line in stream:
            if len(prompts) >= limit:
                break
            stripped = raw_line.strip()
            if not stripped:
                continue
            value: object = json.loads(stripped)
            if not isinstance(value, dict):
                continue
            messages = value.get("messages")
            if not isinstance(messages, list) or not messages:
                continue
            first = messages[0]
            if not isinstance(first, dict) or first.get("role") != "user":
                continue
            content = first.get("content")
            if isinstance(content, str) and content.strip():
                prompts.append(content.strip())
    assert prompts, f"corpus {path} yielded no user prompts"
    return prompts


def _prompts_by_context(corpus: Sequence[str]) -> dict[ContextLength, tuple[str, ...]]:
    """Build deterministic, materially separated prompt-length bands.

    Character bounds keep this helper tokenizer-free.  The live measurement
    records and validates the server-reported token counts, so a tokenizer whose
    ratio is surprising fails the cell instead of silently mislabelling it.
    """
    ordered = sorted(corpus, key=lambda prompt: (len(prompt), prompt))
    short = tuple(prompt[:480] for prompt in ordered[:8])
    long_source = ordered[-8:]
    long: list[str] = []
    for prompt in long_source:
        copies = math.ceil(6000 / len(prompt))
        long.append(((prompt + "\n") * copies)[:6000])
    assert len(short) == 8 and len(long) == 8, "corpus needs at least eight prompts"
    return {"short": short, "long": tuple(long)}


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
) -> list[str]:
    speculative = json.dumps(
        {
            "method": "eagle3",
            "model": draft,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    passthrough = [
        "--max-model-len",
        str(MAX_MODEL_LEN),
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
    assert regime["max_model_len"] == MAX_MODEL_LEN
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


def _validate_context_band(cell: MatrixCell, summary: JsonObject) -> None:
    arms = summary["arms"]
    assert isinstance(arms, dict)
    for arm in ("stock", "candidate"):
        arm_summary = arms[arm]
        assert isinstance(arm_summary, dict)
        prompt_statistic = arm_summary["prompt_tokens"]
        assert isinstance(prompt_statistic, dict)
        mean = float(prompt_statistic["mean"])
        if cell.context_length == "short":
            assert mean <= 256, f"{cell.name}/{arm} short prompt averaged {mean:.1f} tokens"
        else:
            assert 768 <= mean <= MAX_MODEL_LEN - MAX_TOKENS, (
                f"{cell.name}/{arm} long prompt averaged {mean:.1f} tokens"
            )


def _measure_cell(
    configuration: LiveConfiguration,
    cell: MatrixCell,
    prompts: Sequence[str],
    cell_dir: Path,
) -> JsonObject:
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
        argv = _engine_argv(configuration.model, drafts[arm], cell, port=port)
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
        },
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
    _validate_context_band(cell, summary)
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


def test_speculative_inference_configuration_matrix() -> None:
    configuration = _require_live_configuration()
    prompts = _prompts_by_context(_load_corpus(configuration.corpus))
    manifest: JsonObject = {
        "model": configuration.model,
        "reference_draft": configuration.reference_draft,
        "candidate_draft": configuration.candidate_draft,
        "corpus": str(configuration.corpus),
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
            "throughput_statistic": "Prometheus generation tokens / decode seconds",
            "accepted_length_statistic": "1 + accepted tokens / draft steps",
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
        report = _measure_cell(configuration, cell, prompts[cell.context_length], cell_dir)
        reports.append(report)
        cell_failures = report["regression_failures"]
        assert isinstance(cell_failures, list)
        failures.extend(f"{cell.name}: {failure}" for failure in cell_failures)
        _write_json(
            configuration.artifact_dir / "matrix-result.json",
            {"manifest": manifest, "cells": reports, "failures": failures},
        )

    assert not failures, "\n".join(failures)
