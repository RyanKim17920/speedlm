#!/usr/bin/env python
"""Gate an already-trained draft head against a supplied benchmark suite.

There is no production entry point for this.  The idle-tuning cycle owns the
whole sequence -- lease traces, split, train, gate -- and the gate is reachable
only from inside it.  This driver is the smallest honest way to run *just* the
gate: same :class:`~speedlm.gate.runner.BenchmarkGateRunner`, same block
schedule, same thresholds, same ``decision.json`` writer, on a draft that
already exists and a suite that was chosen in advance.

It exists to answer one question about run5's +0.65 accepted length: was that
generalisation, or recall of sessions the head had already trained on?  Nothing
here re-trains anything, and nothing here relaxes a gate check -- in particular
the engine-restart invariant (runner.py:848, every scored block must open from a
real ``activate``) is left exactly as production runs it, so both arms are
measured under identical engine lifecycles.

Wiring
------
Replicates the gate block of ``speedlm.tuner.composition.create_production_tuner``
(composition.py:928-970) with the collaborators
``speedlm.cli._run_tuned_vllm_gateway`` builds (cli.py:500-649), minus
everything the gate does not use: no HTTP gateway, no capture, no admission
gate, no tuner service.  The gate replays straight against the vLLM child's own
OpenAI endpoint, so the proxy would be dead weight.

``_DraftEndpoint`` and ``_MetricsSource`` are imported from ``composition``
rather than reimplemented.  They are private, but they are also the objects that
carry the restart semantics the gate's invariant is defined against; a local
copy would be a second implementation of the one thing this experiment must not
get wrong.

What ``--dry-run`` proves
-------------------------
Everything short of the GPU: config loads, the profile resolves, the launch plan
builds real vLLM argv, the suite loads through the same ``load_suite`` the runner
uses, the runner constructs, the stock draft resolves to the same string the
original ``decision.json`` recorded, and the gate's own ``_check_suite_leakage``
runs against the real training hashes and passes.  It stops before
``supervisor.start``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from speedlm.cli import (  # noqa: E402
    _configured_model_alias,
    _profiled_vllm_passthrough,
)
from speedlm.config import SpeedLMConfig, load_config  # noqa: E402
from speedlm.gate.runner import (  # noqa: E402
    INTERLEAVED_ARM_BLOCKS,
    BenchmarkGateRunner,
)
from speedlm.gate.suite import BenchmarkSuite, load_suite  # noqa: E402
from speedlm.gateway.process import (  # noqa: E402
    LOOPBACK_HOST,
    VLLMProcess,
    reserve_loopback_port,
)
from speedlm.gateway.supervisor import (  # noqa: E402
    ThreadsafeProcessControl,
    VLLMSupervisor,
)
from speedlm.gateway.vllm_http import VLLMControlClient  # noqa: E402
from speedlm.profiles import resolve_profile  # noqa: E402
from speedlm.storage import ensure_layout  # noqa: E402
from speedlm.traces.store import TraceRecord  # noqa: E402
from speedlm.tuner.composition import (  # noqa: E402
    TuningLaunchPlan,
    _DraftEndpoint,
    _MetricsSource,
    build_tuning_launch_plan,
)
from speedlm.tuner.orchestrator import write_decision  # noqa: E402

logger = logging.getLogger("regate")


class RegateError(RuntimeError):
    """Raised when the re-gate cannot be run as an apples-to-apples comparison."""


class _RefusingTraceSource:
    """A trace source that must never be consulted.

    ``BenchmarkGateRunner`` takes a trace source so it can *build* a suite when
    none is persisted (runner.py:1207).  This driver always supplies a persisted
    suite, and building one here would silently benchmark contexts nobody chose
    -- the exact failure this experiment exists to rule out.  Failing loudly is
    the only correct behaviour.
    """

    def iter_records(self) -> Iterator[TraceRecord]:
        raise RegateError(
            "the gate tried to build a suite from traces; it must load the "
            "supplied suite_manifest.json instead"
        )


def _read_stock_draft(decision_path: Path) -> str:
    """The stock draft the original run measured against, from its decision.

    Not defaulted and not inferred from the profile.  The whole experiment is a
    comparison against run5's numbers, and a different baseline makes the two
    deltas incomparable while still producing a plausible-looking number.
    """
    try:
        payload = json.loads(decision_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RegateError(
            f"cannot read original decision {decision_path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RegateError(f"original decision is not valid JSON: {exc}") from exc
    stock = payload.get("stock_draft")
    if not isinstance(stock, str) or not stock:
        raise RegateError(
            f"original decision {decision_path} has no usable 'stock_draft'; "
            "refusing to guess the baseline"
        )
    return stock


def _read_training_hashes(path: Path) -> frozenset[str]:
    """The real leased training hashes, so the leakage check can actually fail.

    An empty set satisfies ``_check_suite_leakage`` trivially -- that is what
    the simulation harness passes, and it is exactly the "test that cannot
    fail" shape.  Refusing an empty file here is what keeps the gate's
    mandatory proof a proof.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    hashes = payload.get("training_context_hashes")
    if not isinstance(hashes, list) or not hashes:
        raise RegateError(
            f"{path} carries no training context hashes; an empty set makes "
            "the gate's leakage check unfalsifiable"
        )
    if not all(isinstance(value, str) and value for value in hashes):
        raise RegateError(f"{path} contains a non-string context hash")
    return frozenset(hashes)


@dataclass(frozen=True)
class ResolvedGate:
    """Everything the gate needs that can be settled without a GPU."""

    config: SpeedLMConfig
    launch: TuningLaunchPlan
    child_url: str
    suite: BenchmarkSuite
    suite_dir: Path
    training_hashes: frozenset[str]
    stock_draft: str
    repeats: int
    facts: dict[str, Any]


def resolve_gate(
    *,
    config_path: Path,
    suite_dir: Path,
    training_hashes_path: Path,
    original_decision: Path,
    home: Path,
    passthrough: Sequence[str],
    repeats: int,
) -> ResolvedGate:
    """Validate every input and build the vLLM launch plan. No process started."""
    config = load_config(config_path)
    manifest = suite_dir / "suite_manifest.json"
    if not manifest.exists():
        raise RegateError(
            f"{manifest} does not exist; the runner would fall back to building "
            "a suite from traces"
        )
    suite = load_suite(suite_dir)
    training_hashes = _read_training_hashes(training_hashes_path)
    overlaps = suite.check_leakage(set(training_hashes))
    if overlaps:
        raise RegateError(
            f"supplied suite overlaps the training set on {len(overlaps)} "
            "context hash(es); the gate would refuse it"
        )
    stock_draft = _read_stock_draft(original_decision)

    layout = ensure_layout(home)
    profile = resolve_profile(
        {"model": config.model, "profile": config.profile},
        served_model=config.model,
        home=layout.root,
    )
    effective = _profiled_vllm_passthrough(
        config.model,
        _configured_model_alias(config, passthrough),
        profile=profile,
        home=layout.root,
    )
    child_port = reserve_loopback_port()
    child_url = f"http://{LOOPBACK_HOST}:{child_port}"
    launch = build_tuning_launch_plan(
        config,
        passthrough=effective,
        child_port=child_port,
        home=layout.root,
    )
    if str(launch.active_draft) != stock_draft:
        # The launch plan resolves the *currently active* artifact, which in a
        # fresh home is the profile's warm-start draft.  If that is not the
        # string the original decision recorded, this run would be measuring a
        # different baseline, and its delta would not be comparable to run5's.
        raise RegateError(
            f"launch plan resolves the stock arm to {str(launch.active_draft)!r}, "
            f"but the original decision measured against {stock_draft!r}; "
            "point --home at a directory with no promoted artifact"
        )
    facts: dict[str, Any] = {
        "config": str(config_path),
        "model": config.model,
        "alias": config.alias,
        "profile": profile.name,
        "home": str(layout.root),
        "stock_draft": stock_draft,
        "suite_dir": str(suite_dir),
        "suite_hash": suite.suite_hash,
        "suite_contexts": len(suite.contexts),
        "training_context_hashes": len(training_hashes),
        "leakage_overlaps": len(overlaps),
        "repeats": repeats,
        "warmup_repeats": config.tuning.warmup_repeats,
        "arm_blocks": INTERLEAVED_ARM_BLOCKS,
        "replay_concurrency": config.tuning.benchmark_concurrency,
        "benchmark_max_tokens": config.tuning.benchmark_max_tokens,
        "correctness_max_tokens": config.tuning.correctness_max_tokens,
        "held_out_fraction": config.tuning.held_out_fraction,
        "candidate_arm_first": config.tuning.benchmark_candidate_arm_first,
        "engine_execution_mode": launch.engine_execution.execution_mode,
        "child_url": child_url,
        "vllm_argv": launch.argv_factory(launch.active_draft),
    }
    return ResolvedGate(
        config=config,
        launch=launch,
        child_url=child_url,
        suite=suite,
        suite_dir=suite_dir,
        training_hashes=training_hashes,
        stock_draft=stock_draft,
        repeats=repeats,
        facts=facts,
    )


def make_runner(
    resolved: ResolvedGate,
    *,
    endpoint: Any,
    metrics_source: Any,
) -> BenchmarkGateRunner:
    """Build the production gate runner around an endpoint.

    Every keyword below is the production value from
    ``create_production_tuner`` (composition.py:928-970) except ``repeats``,
    which is passed explicitly so the driver's flag is the single place it is
    set, and ``trace_source``, which is a refusing stub because a persisted
    suite is mandatory here.
    """
    config = resolved.config
    return BenchmarkGateRunner(
        config=config,
        trace_source=_RefusingTraceSource(),
        suite_dir=resolved.suite_dir,
        stock_draft=resolved.stock_draft,
        endpoint=endpoint,
        metrics_source=metrics_source,
        repeats=resolved.repeats,
        warmup_repeats=config.tuning.warmup_repeats,
        arm_blocks=INTERLEAVED_ARM_BLOCKS,
        replay_concurrency=config.tuning.benchmark_concurrency,
        correctness_max_tokens=config.tuning.correctness_max_tokens,
        benchmark_max_tokens=config.tuning.benchmark_max_tokens,
        held_out_fraction=config.tuning.held_out_fraction,
        training_context_hashes=resolved.training_hashes,
        candidate_arm_first=config.tuning.benchmark_candidate_arm_first,
        engine_execution=resolved.launch.engine_execution,
    )


async def _run(args: argparse.Namespace) -> int:
    resolved = resolve_gate(
        config_path=args.config,
        suite_dir=args.suite_dir,
        training_hashes_path=args.training_hashes,
        original_decision=args.original_decision,
        home=args.home,
        passthrough=args.passthrough,
        repeats=args.repeats,
    )
    print(json.dumps(resolved.facts, indent=2))

    if args.dry_run:
        # ``endpoint``/``metrics_source`` are not touched before the first
        # ``activate``, and the dry run stops well short of that, so the runner
        # built here is the same object the real run builds.
        runner = make_runner(resolved, endpoint=None, metrics_source=None)
        # The gate's own check, not a restatement of it: this is the private
        # method ``benchmark`` calls at runner.py:788, and the one that would
        # abort the real run.
        runner._check_suite_leakage(runner._load_or_build_suite())
        print("\nleakage check PASSED against the real training hashes")
        print("dry run complete; stopped before starting any engine")
        return 0

    config = resolved.config
    launch = resolved.launch

    def process_factory(argv: Sequence[str], *, health_url: str) -> VLLMProcess:
        return VLLMProcess(
            argv,
            health_url=health_url,
            startup_timeout=config.startup_timeout_seconds,
            startup_stall_timeout=config.startup_stall_seconds,
            env_overrides={"VLLM_SERVER_DEV_MODE": "1"},
        )

    supervisor = VLLMSupervisor(
        argv_factory=launch.argv_factory,
        health_url=f"{resolved.child_url}/health",
        process_factory=process_factory,
    )
    result = None
    try:
        await supervisor.start(launch.active_draft)
        await supervisor.wait_ready(timeout=config.startup_timeout_seconds)
        http = VLLMControlClient(resolved.child_url, model=config.alias)
        await asyncio.to_thread(
            http.wait_ready,
            timeout_seconds=config.startup_timeout_seconds,
            should_abort=lambda: False,
        )
        endpoint = _DraftEndpoint(
            url=resolved.child_url,
            process=ThreadsafeProcessControl(supervisor, asyncio.get_running_loop()),
            http=http,
            # No RuntimeController: this driver serves no traffic, so there is
            # nothing to consult about what is already running, and ``None``
            # restores unconditional restart.  The gate clears
            # ``allow_engine_reuse`` for every scored block anyway, so this
            # cannot weaken the restart invariant; it can only cost one extra
            # restart, which both arms pay equally.
            runtime=None,
        )
        runner = make_runner(
            resolved, endpoint=endpoint, metrics_source=_MetricsSource(http)
        )
        # The benchmark is synchronous and drives the supervisor through
        # ``ThreadsafeProcessControl``, which posts coroutines back to this
        # loop -- so it must not run *on* this loop.
        result = await asyncio.to_thread(
            runner.benchmark,
            args.candidate_draft,
            timeout_seconds=args.timeout_seconds,
            should_abort=lambda: False,
        )
    finally:
        await supervisor.shutdown()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "regate_wiring.json").write_text(
        json.dumps(resolved.facts, indent=2) + "\n", encoding="utf-8"
    )
    print(f"gate passed={result.passed} reason={result.reason}")
    if result.failure is not None:
        print(f"gate failure: {result.failure}")
    if result.decision is None:
        print("no decision produced; nothing to persist")
        return 1
    path = write_decision(args.run_dir, result.decision)
    print(f"wrote {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--training-hashes", type=Path, required=True)
    parser.add_argument(
        "--original-decision",
        type=Path,
        required=True,
        help="the run whose stock arm this must reproduce exactly",
    )
    parser.add_argument("--candidate-draft", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--home",
        type=Path,
        default=None,
        help="SPEEDLM_HOME for this gate; must hold no promoted artifact",
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "passthrough",
        nargs="*",
        default=[],
        help="vLLM flags (after --); must match the original run's engine flags",
    )
    args = parser.parse_args(argv)
    if args.home is None:
        env_home = os.environ.get("SPEEDLM_HOME")
        if not env_home:
            raise SystemExit("--home or SPEEDLM_HOME is required")
        args.home = Path(env_home)
    if not args.dry_run and not args.candidate_draft.is_dir():
        raise SystemExit(f"candidate draft {args.candidate_draft} is not a directory")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
