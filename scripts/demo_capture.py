#!/usr/bin/env python3
"""Record a stock-vs-tuned drafting comparison, token by token, for the demo video.

The gate already answers "is the tuned head faster" as a number (+0.2989 accepted
length / +9.94% tok/s on the session-disjoint suite, job 378546).  This script
answers the same question as a *timeline*: it replays the same held-out contexts
through both draft heads and writes down when every single token arrived, so the
renderer can play the two arms side by side and let the wall clock make the
argument.

What it does NOT do is re-measure the gate.  The gate pools over 5 repeats x 100
contexts x 2 blocks per arm precisely because a single sequential pass is too
noisy to promote on; this pass is one repeat over a hand-picked subset at
concurrency 1, chosen for legibility rather than statistical power.  Treat the
numbers it emits as an illustration of the gated result, never as a replacement
for it -- ``docs/agentic-selfplay-result.md`` holds the number that counts.

Three things are held identical across the arms, because each of them would
otherwise show up as a speed difference that has nothing to do with drafting:

  * the engine argv, apart from the one ``--speculative-config`` model field;
  * the sampling parameters, taken from the gate config (greedy: temperature 0,
    top_p 1, seed 0), so the two arms should emit the SAME text and any
    divergence is a finding rather than noise;
  * the contexts and their order, resolved once up front and reused verbatim.

The arms run sequentially in one process on one GPU.  That is slower than racing
two engines but it is the only way the comparison means anything: two engines on
two GPUs differ by silicon and thermals, and two engines on one GPU contend.

Usage (inside a SLURM allocation with a GPU -- see scripts/demo_capture.sbatch):

    python scripts/demo_capture.py \
        --suite-dir  /data/.../regate-unseen-run1/unseen-suite \
        --selection  /data/.../regate-unseen-run1/unseen-suite/selection.json \
        --stock-draft     RedHatAI/Qwen3-8B-speculator.eagle3 \
        --candidate-draft /data/.../runs/<id>/draft-model \
        --out-dir    /data/.../demo-video-run1
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from speedlm.gate.metrics import (  # noqa: E402
    CounterResetError,
    MetricsSnapshot,
    compute_delta,
    parse_metrics,
)
from speedlm.gate.suite import FrozenContext, load_suite  # noqa: E402

# The gate's own sampling block (configs/agentenv-qwen8b.json).  Hard-coded
# rather than re-read so that a config edit cannot silently make this recording
# non-comparable with the decision it illustrates; --config overrides it and
# records the override in the manifest.
GATE_SAMPLING = {"temperature": 0.0, "top_p": 1.0, "seed": 0}
GATE_MAX_TOKENS = 512

ARMS = ("stock", "candidate")


class CaptureError(RuntimeError):
    """Raised when the recording cannot be made comparable across arms."""


# ---------------------------------------------------------------------------
# Recording types
# ---------------------------------------------------------------------------


@dataclass
class TokenEvent:
    """One streamed chunk and the moment the client saw it.

    ``t`` is seconds since the request was issued, not since the arm started, so
    the renderer can lay requests out however it likes.  Chunks are recorded as
    vLLM emitted them; a chunk is usually one token but the renderer must not
    assume that, which is why ``index`` counts chunks and the token totals come
    from the usage block instead.
    """

    index: int
    t: float
    text: str
    reasoning: bool = False


@dataclass
class RequestRecord:
    """Everything observed for one context on one arm."""

    context_hash: str
    family: str
    order: int
    prompt_chars: int
    turn_depth: int

    # Client-side wall clock.  This is what the video plays back and what
    # "how long did it take" means to a user watching the terminal.
    t_start: float = 0.0
    ttft_s: float = 0.0
    wall_s: float = 0.0

    # Server-side accounting from the streamed usage block.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str = ""

    # Engine-side speculative counters, differenced across this request alone.
    # Sound only because the replay is strictly sequential: with concurrency 1
    # no other request contributes to the window.
    accepted_length: float = 0.0
    acceptance_rate: float = 0.0
    engine_tok_per_sec: float = 0.0
    metrics_available: bool = False

    text: str = ""
    tokens: list[TokenEvent] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Engine lifecycle
# ---------------------------------------------------------------------------


def build_argv(template: list[str], draft: str, port: int) -> list[str]:
    """Rewrite the recorded vLLM argv for one arm.

    Exactly two things change: the model inside ``--speculative-config`` and the
    port.  Everything else -- notably ``--enforce-eager``, which is what makes
    the recorded throughput comparable with the gate's eager-mode decision -- is
    copied through untouched.
    """
    argv = list(template)

    # argv.index would quietly take the FIRST occurrence, so a template that
    # carried a flag twice would have its first copy rewritten and its second
    # copy left alone -- vLLM honours the last one, so the engine would bind a
    # port the client never polls and the run would die 1200s later in
    # wait_ready with a readiness timeout that says nothing about the real
    # cause.  Refusing a duplicated flag turns that into an immediate,
    # self-explaining failure.
    def value_index(flag: str) -> int:
        occurrences = [i for i, item in enumerate(argv) if item == flag]
        if len(occurrences) != 1:
            raise CaptureError(
                f"argv template must contain {flag} exactly once, found {len(occurrences)}"
            )
        at = occurrences[0] + 1
        if at >= len(argv):
            raise CaptureError(f"argv template ends with {flag}, which has no value after it")
        return at

    spec_at = value_index("--speculative-config")
    port_at = value_index("--port")

    spec = json.loads(argv[spec_at])
    spec["model"] = draft
    argv[spec_at] = json.dumps(spec, separators=(",", ":"))
    argv[port_at] = str(port)

    # --enable-sleep-mode belongs to the tuner's engine-reuse path, which has no
    # counterpart here: each arm gets a cold process.  Leaving it on would be
    # harmless but it is one more difference from a plain `vllm serve`.
    if "--enable-sleep-mode" in argv:
        argv.remove("--enable-sleep-mode")
    return argv


@dataclass
class Engine:
    """A running vLLM process and the log file it is writing into.

    The log handle is held here rather than dropped on the floor: the child
    inherits the file descriptor, but the parent's copy still has to be closed or
    the engine's final lines can be lost to an unflushed buffer when the process
    is killed.
    """

    process: subprocess.Popen[bytes]
    log: IO[bytes]


def start_engine(argv: list[str], log_path: Path, env: dict[str, str]) -> Engine:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("wb")
    try:
        # start_new_session so a stuck engine can be killed as a process group;
        # vLLM spawns worker children that outlive a bare terminate otherwise.
        process = subprocess.Popen(
            argv, stdout=handle, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
    except BaseException:
        handle.close()
        raise
    return Engine(process=process, log=handle)


def wait_ready(base_url: str, proc: subprocess.Popen[bytes], timeout: float) -> None:
    """Poll /health until the engine serves, or the engine dies trying."""
    deadline = time.monotonic() + timeout
    with httpx.Client(timeout=5.0) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise CaptureError(
                    f"engine exited with code {proc.returncode} before becoming ready"
                )
            try:
                if client.get(f"{base_url}/health").status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
    raise CaptureError(f"engine was not ready within {timeout:.0f}s")


def _signal_group(proc: subprocess.Popen[bytes], sig: int) -> None:
    """Signal the engine's process group, tolerating it having already exited.

    stop_engine runs from a ``finally``, so a ProcessLookupError raised here --
    which is just a race with the engine exiting on its own -- would replace the
    real exception on the way out and hide why the capture actually failed.
    """
    with contextlib.suppress(ProcessLookupError):
        os.killpg(os.getpgid(proc.pid), sig)


def stop_engine(engine: Engine) -> None:
    proc = engine.process
    try:
        if proc.poll() is None:
            _signal_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                _signal_group(proc, signal.SIGKILL)
                proc.wait(timeout=60)
    finally:
        engine.log.close()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def scrape(client: httpx.Client, base_url: str) -> MetricsSnapshot:
    response = client.get(f"{base_url}/metrics")
    response.raise_for_status()
    return parse_metrics(response.text)


def scrape_settled(
    client: httpx.Client, base_url: str, before: MetricsSnapshot, generated: int
) -> tuple[MetricsSnapshot, bool]:
    """Scrape until the request we just finished is visible in the counters.

    vLLM publishes counters on an interval, so a scrape taken the instant the
    stream ends can still miss the tail of the request that just completed.  A
    short poll for the generation counter to advance by the number of tokens we
    actually received turns a silently-truncated window into a correct one.

    The second element of the returned pair says whether that actually happened.
    It is False when the deadline passed with the counter still short, and also
    when ``generated`` is not positive -- with zero tokens the ``>= generated``
    test is trivially true, so it would report "settled" on a window that
    contains none of the request.  An unsettled snapshot must not be differenced
    against ``before``: the resulting window does not contain the request, so
    every per-request number computed from it is wrong rather than merely
    imprecise, and the caller has to record "unknown" instead.
    """
    deadline = time.monotonic() + 15.0
    after = scrape(client, base_url)
    if generated <= 0:
        return after, False
    while time.monotonic() < deadline:
        if after.generated_tokens - before.generated_tokens >= generated:
            return after, True
        time.sleep(0.5)
        after = scrape(client, base_url)
    return after, after.generated_tokens - before.generated_tokens >= generated


# ---------------------------------------------------------------------------
# Streaming replay
# ---------------------------------------------------------------------------


def sse_chunks(response: httpx.Response) -> Iterator[dict[str, Any]]:
    for line in response.iter_lines():
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        # A bare ``data:`` line is a legal SSE keep-alive and a frame that parses
        # to something other than an object is not a chat completion chunk;
        # either one would blow up as a JSONDecodeError or an AttributeError deep
        # inside the chunk loop, killing a capture over a frame that carries no
        # information anyway.  Skipping them keeps the replay running.
        if not payload:
            continue
        chunk = json.loads(payload)
        if not isinstance(chunk, dict):
            continue
        yield chunk


def replay_context(
    client: httpx.Client,
    base_url: str,
    model: str,
    context: FrozenContext,
    *,
    order: int,
    family: str,
    sampling: dict[str, Any],
    max_tokens: int,
) -> RequestRecord:
    """Stream one context and record when every chunk landed."""
    messages = [dict(message) for message in context.messages]
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
        # Without this vLLM omits usage from a streaming response entirely, and
        # the completion-token count is what makes the tok/s overlay honest --
        # counting SSE chunks would count chunks, not tokens.
        "stream_options": {"include_usage": True},
        **sampling,
    }
    if context.tools:
        payload["tools"] = [dict(tool) for tool in context.tools]

    record = RequestRecord(
        context_hash=context.context_hash,
        family=family,
        order=order,
        prompt_chars=sum(len(str(m.get("content") or "")) for m in messages),
        turn_depth=len(messages),
    )

    before = scrape(client, base_url)
    pieces: list[str] = []
    started = time.perf_counter()
    record.t_start = started

    with client.stream("POST", f"{base_url}/v1/chat/completions", json=payload) as response:
        response.raise_for_status()
        for chunk in sse_chunks(response):
            now = time.perf_counter() - started
            usage = chunk.get("usage")
            if usage:
                record.prompt_tokens = int(usage.get("prompt_tokens") or 0)
                record.completion_tokens = int(usage.get("completion_tokens") or 0)
            for choice in chunk.get("choices") or ():
                delta = choice.get("delta") or {}
                # Qwen3 emits its chain-of-thought through reasoning_content, and
                # ~89% of this corpus's drafted text is exactly that.  Recording
                # which stream a chunk came from lets the renderer dim the
                # monologue instead of pretending it is answer text.
                for key, is_reasoning in (("reasoning_content", True), ("content", False)):
                    text = delta.get(key)
                    if not text:
                        continue
                    record.tokens.append(
                        TokenEvent(
                            index=len(record.tokens),
                            t=now,
                            text=text,
                            reasoning=is_reasoning,
                        )
                    )
                    pieces.append(text)
                if choice.get("finish_reason"):
                    record.finish_reason = str(choice["finish_reason"])

    record.wall_s = time.perf_counter() - started
    record.ttft_s = record.tokens[0].t if record.tokens else record.wall_s
    record.text = "".join(pieces)

    # The two arms must replay the identical workload for the side-by-side
    # comparison to mean anything.  A request that streamed nothing would be
    # written down as 0 tokens in ~0 seconds, which drags one arm's totals
    # towards a number no engine produced and looks exactly like a fast run --
    # this codebase's recurring defect is precisely a green measurement of
    # nothing, so an empty response is a hard failure rather than a data point.
    if record.completion_tokens <= 0 or not record.tokens:
        raise CaptureError(
            f"context {record.context_hash} (family {record.family}, order {record.order}) "
            f"streamed no output: {record.completion_tokens} completion tokens, "
            f"{len(record.tokens)} chunks, finish_reason {record.finish_reason!r}"
        )

    after, settled = scrape_settled(client, base_url, before, record.completion_tokens)
    try:
        delta = compute_delta(before, after)
    except CounterResetError as exc:
        raise CaptureError(f"engine counters reset mid-capture: {exc}") from exc
    # Only a settled window actually contains this request, so an unsettled
    # scrape records "unknown" and leaves the speed fields at zero: a wrong
    # speed number is worse than a missing one here, because a missing one is
    # visibly dropped from the arm's mean while a wrong one is averaged in and
    # silently moves the comparison the whole recording exists to make.
    if settled and delta.accepted_length_available:
        record.metrics_available = True
        record.accepted_length = delta.mean_accepted_length
        record.acceptance_rate = delta.acceptance_rate
        record.engine_tok_per_sec = delta.output_tok_per_sec
    return record


# ---------------------------------------------------------------------------
# Context selection
# ---------------------------------------------------------------------------


def family_hash(messages: Sequence[Mapping[str, Any]]) -> str:
    """The builder's task-family key: sha256 of the first user message.

    Kept byte-compatible with ``_family`` in build_unseen_session_suite.py, which
    is what wrote the ``task_family`` values in selection.json.  Every seed of a
    family gets a byte-identical prompt, so this collapses the corpus to exactly
    six values -- useless as a session key, exactly right as a family label.
    """
    for message in messages:
        if message.get("role") == "user":
            content = message.get("content")
            text = content if isinstance(content, str) else json.dumps(content)
            return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return "no-user"


def load_family_names(trajectories_dir: Path | None) -> dict[str, str]:
    """Recover readable family names by joining the hash back to the corpus.

    selection.json stores the family as a hash, because the gate artifacts never
    knew the corpus.  The trajectory files do carry a ``family`` string, so
    re-deriving the hash from each trajectory's first user message reconstructs
    the mapping instead of hard-coding six literals that would rot.
    """
    if trajectories_dir is None or not trajectories_dir.is_dir():
        return {}
    names: dict[str, str] = {}
    for path in sorted(trajectories_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        name = payload.get("family")
        messages = payload.get("messages") or ()
        if not name or not messages:
            continue
        names.setdefault(family_hash(messages), str(name))
    return names


def load_families(
    selection_path: Path | None, trajectories_dir: Path | None
) -> dict[str, str]:
    """Map context hash -> readable task family, for the on-screen label.

    The suite manifest deliberately carries no family field (it is a gate
    artifact, and family is a property of the corpus), so the label comes from
    the builder's selection.json, resolved to a name through the corpus when the
    trajectories are available and left as the hash when they are not.
    """
    if selection_path is None or not selection_path.exists():
        return {}
    names = load_family_names(trajectories_dir)
    payload = json.loads(selection_path.read_text())
    return {
        entry["context_hash"]: names.get(
            str(entry.get("task_family") or ""), str(entry.get("task_family") or "")
        )
        for entry in payload.get("contexts") or ()
    }


def choose_contexts(
    contexts: tuple[FrozenContext, ...],
    families: dict[str, str],
    *,
    count: int,
    min_expected_chars: int,
) -> list[FrozenContext]:
    """Pick the contexts worth watching, deterministically.

    Half of this suite's held-out turns are a bare tool call -- an empty think
    block and ~4 tokens of JSON.  Those are perfectly good gate material and
    completely unwatchable, so contexts whose reference continuation is that
    short are dropped and the rest are ranked longest-prompt-first.  Selection
    is a pure function of the suite, so both arms and any re-run agree.
    """
    watchable = [
        context
        for context in contexts
        if len(context.expected_response) >= min_expected_chars
    ]
    if len(watchable) < count:
        raise CaptureError(
            f"only {len(watchable)} of {len(contexts)} contexts have a reference "
            f"continuation of >= {min_expected_chars} chars; asked for {count}"
        )
    ranked = sorted(
        watchable,
        key=lambda c: (-sum(len(str(m.get("content") or "")) for m in c.messages), c.context_hash),
    )
    chosen = ranked[:count]

    # Deal the chosen contexts round-robin across families, so the video opens
    # with six different tasks rather than nine consecutive bugfix-localize runs
    # -- a viewer reading "the same prompt again" as "the same work again" is a
    # fair reading of a grouped ordering.
    by_family: dict[str, list[FrozenContext]] = {}
    for context in chosen:
        by_family.setdefault(families.get(context.context_hash, "unknown"), []).append(context)
    queues = [by_family[name] for name in sorted(by_family)]
    interleaved: list[FrozenContext] = []
    while queues:
        for queue in list(queues):
            interleaved.append(queue.pop(0))
            if not queue:
                queues.remove(queue)
    return interleaved


# ---------------------------------------------------------------------------
# Arm driver
# ---------------------------------------------------------------------------


def run_arm(
    arm: str,
    draft: str,
    *,
    argv_template: list[str],
    port: int,
    model: str,
    contexts: list[FrozenContext],
    families: dict[str, str],
    sampling: dict[str, Any],
    max_tokens: int,
    out_dir: Path,
    env: dict[str, str],
    startup_timeout: float,
) -> dict[str, Any]:
    base_url = f"http://127.0.0.1:{port}"
    argv = build_argv(argv_template, draft, port)
    log_path = out_dir / f"engine-{arm}.log"

    print(f"[{arm}] launching engine with draft {draft}", flush=True)
    print(f"[{arm}] argv: {' '.join(argv)}", flush=True)
    engine = start_engine(argv, log_path, env)
    records: list[RequestRecord] = []
    try:
        wait_ready(base_url, engine.process, startup_timeout)
        print(f"[{arm}] engine ready", flush=True)

        with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            # One throwaway pass over the first context, discarded.  The first
            # request into a cold engine pays for CUDA graph capture and cache
            # warmup, and charging that to the stock arm alone (it runs first)
            # would hand the tuned arm a free win.
            print(f"[{arm}] warmup", flush=True)
            replay_context(
                client, base_url, model, contexts[0],
                order=-1, family="warmup", sampling=sampling, max_tokens=64,
            )

            arm_started = time.perf_counter()
            for order, context in enumerate(contexts):
                record = replay_context(
                    client, base_url, model, context,
                    order=order,
                    family=families.get(context.context_hash, "unknown"),
                    sampling=sampling,
                    max_tokens=max_tokens,
                )
                records.append(record)
                print(
                    f"[{arm}] {order + 1}/{len(contexts)} {record.family} "
                    f"{record.completion_tokens} tok in {record.wall_s:.2f}s "
                    f"({record.completion_tokens / max(record.wall_s, 1e-9):.1f} tok/s, "
                    f"accepted_length {record.accepted_length:.3f})",
                    flush=True,
                )
            arm_wall = time.perf_counter() - arm_started
    finally:
        stop_engine(engine)

    timeline_path = out_dir / f"timeline-{arm}.jsonl"
    with timeline_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")

    total_tokens = sum(r.completion_tokens for r in records)
    weighted_accepted = [r for r in records if r.metrics_available]
    summary = {
        "arm": arm,
        "draft": draft,
        "contexts": len(records),
        "total_completion_tokens": total_tokens,
        "total_wall_seconds": arm_wall,
        "wall_tok_per_sec": total_tokens / arm_wall if arm_wall > 0 else 0.0,
        "mean_ttft_seconds": sum(r.ttft_s for r in records) / max(len(records), 1),
        "mean_accepted_length": (
            sum(r.accepted_length for r in weighted_accepted) / len(weighted_accepted)
            if weighted_accepted
            else 0.0
        ),
        "mean_engine_tok_per_sec": (
            sum(r.engine_tok_per_sec for r in weighted_accepted) / len(weighted_accepted)
            if weighted_accepted
            else 0.0
        ),
        "truncated": sum(1 for r in records if r.finish_reason == "length"),
        "timeline": str(timeline_path),
    }
    print(f"[{arm}] {json.dumps(summary, indent=2)}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv_in: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument(
        "--trajectories", type=Path, default=None,
        help="run5 traffic/trajectories dir, to name families instead of hashing them",
    )
    parser.add_argument("--wiring", type=Path, required=True,
                        help="regate_wiring.json whose vllm_argv is the argv template")
    parser.add_argument("--stock-draft", required=True)
    parser.add_argument("--candidate-draft", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--num-contexts", type=int, default=49)
    parser.add_argument("--min-expected-chars", type=int, default=400)
    parser.add_argument("--max-tokens", type=int, default=GATE_MAX_TOKENS)
    parser.add_argument("--startup-timeout", type=float, default=1200.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve everything except the GPU: suite, selection, argv")
    args = parser.parse_args(argv_in)

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = load_suite(args.suite_dir)
    families = load_families(args.selection, args.trajectories)
    contexts = choose_contexts(
        suite.contexts,
        families,
        count=args.num_contexts,
        min_expected_chars=args.min_expected_chars,
    )
    argv_template = json.loads(args.wiring.read_text())["vllm_argv"]

    candidate = Path(args.candidate_draft)
    if not candidate.exists():
        raise CaptureError(f"candidate draft does not exist: {candidate}")

    manifest = {
        "suite_dir": str(args.suite_dir),
        "suite_hash": suite.suite_hash,
        "suite_contexts": len(suite.contexts),
        "selected_contexts": [
            {
                "order": i,
                "context_hash": c.context_hash,
                "family": families.get(c.context_hash, "unknown"),
                "turn_depth": len(c.messages),
                "prompt_chars": sum(len(str(m.get("content") or "")) for m in c.messages),
                "expected_response_chars": len(c.expected_response),
            }
            for i, c in enumerate(contexts)
        ],
        "stock_draft": args.stock_draft,
        "candidate_draft": str(candidate),
        "sampling": GATE_SAMPLING,
        "max_tokens": args.max_tokens,
        "model": args.model,
        "arms_run_sequentially_on_one_gpu": True,
        "stock_argv": build_argv(argv_template, args.stock_draft, args.port),
        "candidate_argv": build_argv(argv_template, str(candidate), args.port),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": os.uname().nodename,
    }
    (out_dir / "capture_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_contexts"}, indent=2))
    print(f"selected {len(contexts)} contexts:", flush=True)
    for entry in manifest["selected_contexts"]:
        print(f"  {entry['order']:2d} {entry['family']:20s} "
              f"depth={entry['turn_depth']:3d} prompt={entry['prompt_chars']:6d} "
              f"expected={entry['expected_response_chars']:5d} {entry['context_hash'][:12]}",
              flush=True)

    if args.dry_run:
        print("dry run: engine not launched", flush=True)
        return 0

    env = dict(os.environ)
    env.setdefault("VLLM_SERVER_DEV_MODE", "1")

    summaries = {}
    for arm, draft in (("stock", args.stock_draft), ("candidate", str(candidate))):
        summaries[arm] = run_arm(
            arm,
            draft,
            argv_template=argv_template,
            port=args.port,
            model=args.model,
            contexts=contexts,
            families=families,
            sampling=GATE_SAMPLING,
            max_tokens=args.max_tokens,
            out_dir=out_dir,
            env=env,
            startup_timeout=args.startup_timeout,
        )

    stock, candidate_summary = summaries["stock"], summaries["candidate"]
    comparison = {
        "arms": summaries,
        "wall_seconds_stock": stock["total_wall_seconds"],
        "wall_seconds_candidate": candidate_summary["total_wall_seconds"],
        "wall_speedup_pct": (
            (stock["total_wall_seconds"] - candidate_summary["total_wall_seconds"])
            / stock["total_wall_seconds"] * 100.0
            if stock["total_wall_seconds"] > 0
            else 0.0
        ),
        "accepted_length_delta": (
            candidate_summary["mean_accepted_length"] - stock["mean_accepted_length"]
        ),
        "engine_tok_per_sec_delta_pct": (
            (candidate_summary["mean_engine_tok_per_sec"] - stock["mean_engine_tok_per_sec"])
            / stock["mean_engine_tok_per_sec"] * 100.0
            if stock["mean_engine_tok_per_sec"] > 0
            else 0.0
        ),
        # A single sequential pass, NOT the gated measurement.  See the module
        # docstring: docs/agentic-selfplay-result.md holds the number to cite.
        "gated_result_reference": {
            "accepted_length_delta": 0.2989,
            "throughput_delta_pct": 9.94,
            "source": "docs/agentic-selfplay-result.md (job 378546, unseen suite)",
        },
    }
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
