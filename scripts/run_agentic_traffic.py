#!/usr/bin/env python
"""Drive the executable agent environments against a running SpeedLM gateway.

This is the traffic generator for agentic idle tuning.  Point it at a gateway
that is already serving (``speedlm vllm serve ... --enable-idle-tuning``) and it
will run every selected task instance as a real tool loop, writing one
trajectory file per instance plus a summary.

    scripts/run_agentic_traffic.py \
        --base-url http://127.0.0.1:8000 \
        --model Qwen/Qwen3-8B \
        --seeds 8 \
        --out /data/ryan.kim/speedlm-runs/agentenv-qwen

WHY A SCRIPT AND NOT ONLY A TEST
--------------------------------
The traffic and the measurement are separable, and conflating them is how the
previous agentic attempts died: each one failed in the *training* stage, after
the traffic had already been generated and thrown away.  Trajectories written
here survive the process that made them, so a failed training run can be
re-attempted against the same traffic instead of paying for another allocation
to regenerate it.

WHAT IT ASSERTS BEFORE IT BELIEVES ITS OWN OUTPUT
------------------------------------------------
A run that produces zero dispatched tool calls is a run that measured
single-turn chat while reporting agentic traffic, and it is the single most
likely way this can go quietly wrong (see
:class:`~tests.e2e.agentenv.loop.ToolParserMissingError`).  The summary
therefore carries ``dispatched_tool_calls`` and ``mean_turns_per_trajectory``,
and ``--require-tool-calls`` (on by default) makes the script exit non-zero when
the whole run dispatched none.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx  # noqa: E402

from tests.e2e.agentenv.catalog import TASKS, all_instances  # noqa: E402
from tests.e2e.agentenv.loop import LoopLimits, run_agent_loop  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True, help="Gateway base URL.")
    parser.add_argument("--model", required=True, help="Model name to send in each request.")
    parser.add_argument("--out", required=True, type=Path, help="Directory for trajectories.")
    parser.add_argument(
        "--seeds",
        type=int,
        default=4,
        help="How many seeded instances of each family to run.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help=(
            "First seed; instances run over [seed-start, seed-start+seeds). "
            "Task instances are a pure function of the seed, so concurrent "
            "shards MUST take disjoint windows or they generate identical work."
        ),
    )
    parser.add_argument(
        "--families",
        default="",
        help=(
            "Comma-separated family names to run; default is all of: "
            + ", ".join(task.name for task in TASKS)
        ),
    )
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--request-timeout", type=float, default=600.0, help="Per-request timeout, seconds."
    )
    parser.add_argument(
        "--trajectory-wall-clock",
        type=float,
        default=1800.0,
        help="Budget for one trajectory, seconds.",
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=REPO_ROOT / ".venv" / "bin" / "python",
        help="Interpreter the workspaces run their own pytest with.",
    )
    parser.add_argument(
        "--require-tool-calls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero if the whole run dispatched no tool call.",
    )
    parser.add_argument(
        "--stop-after-seconds",
        type=float,
        default=0.0,
        help="Stop starting new trajectories after this much wall clock (0 = no limit).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    families = tuple(name for name in args.families.split(",") if name) or None
    instances = all_instances(
        seeds=args.seeds, families=families, seed_start=args.seed_start
    )
    out = args.out
    (out / "trajectories").mkdir(parents=True, exist_ok=True)
    (out / "workspaces").mkdir(parents=True, exist_ok=True)

    limits = LoopLimits(
        max_turns=args.max_turns,
        max_output_tokens=args.max_output_tokens,
        request_timeout_seconds=args.request_timeout,
        wall_clock_seconds=args.trajectory_wall_clock,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    summaries: list[dict[str, Any]] = []
    started = time.monotonic()
    with httpx.Client() as client:
        for position, instance in enumerate(instances):
            elapsed = time.monotonic() - started
            if args.stop_after_seconds and elapsed > args.stop_after_seconds:
                print(
                    f"stopping after {elapsed:.0f}s with {len(instances) - position} "
                    "instances not started",
                    flush=True,
                )
                break
            workspace_root = out / "workspaces" / instance.id
            sandbox = instance.materialize(workspace_root, python=args.python)
            try:
                result = run_agent_loop(
                    instance,
                    sandbox,
                    base_url=args.base_url,
                    model=args.model,
                    client=client,
                    limits=limits,
                )
            except Exception as error:  # noqa: BLE001 - one bad task must not end the run
                # A crash here is recorded rather than raised: the traffic already
                # sent still reached the capture layer, and losing the remaining
                # instances would cost the allocation for one model's bad turn.
                summaries.append(
                    {
                        "instance_id": instance.id,
                        "family": instance.family,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                print(f"[{instance.id}] ERROR {type(error).__name__}: {error}", flush=True)
                continue

            grade = instance.grade(sandbox)
            record = result.to_dict()
            record["grade"] = grade.to_dict()
            (out / "trajectories" / f"{instance.id}.json").write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            summary = {
                "instance_id": instance.id,
                "family": instance.family,
                "solved": grade.solved,
                "turns": len(result.turns),
                "tool_calls": result.tool_call_count,
                "failed_tool_calls": result.failed_tool_call_count,
                "stop_condition": result.stop_condition,
                "final_prompt_tokens": (
                    result.turns[-1].prompt_tokens if result.turns else None
                ),
                "wall_clock_seconds": round(result.wall_clock_seconds, 2),
                "grade_detail": grade.detail,
            }
            summaries.append(summary)
            print(
                f"[{instance.id}] solved={grade.solved} turns={summary['turns']} "
                f"tool_calls={summary['tool_calls']} "
                f"({summary['failed_tool_calls']} failed) "
                f"stop={summary['stop_condition']} "
                f"prompt_tokens={summary['final_prompt_tokens']}",
                flush=True,
            )

    report = _report(summaries, args)
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["totals"], indent=2, sort_keys=True))

    if args.require_tool_calls and report["totals"]["dispatched_tool_calls"] == 0:
        print(
            "REFUSING to report success: the whole run dispatched zero tool calls, so "
            "this measured single-turn chat rather than agentic traffic. Serve with "
            "--enable-auto-tool-choice and --tool-call-parser for this model family.",
            file=sys.stderr,
        )
        return 1
    return 0


def _report(summaries: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    ran = [item for item in summaries if "error" not in item]
    errored = [item for item in summaries if "error" in item]
    by_family: dict[str, dict[str, Any]] = {}
    for item in ran:
        block = by_family.setdefault(
            item["family"], {"attempted": 0, "solved": 0, "turns": 0, "tool_calls": 0}
        )
        block["attempted"] += 1
        block["solved"] += int(bool(item["solved"]))
        block["turns"] += int(item["turns"])
        block["tool_calls"] += int(item["tool_calls"])
    for block in by_family.values():
        block["solve_rate"] = (
            round(block["solved"] / block["attempted"], 4) if block["attempted"] else None
        )
        block["mean_turns"] = (
            round(block["turns"] / block["attempted"], 2) if block["attempted"] else None
        )

    prompt_tokens = [
        item["final_prompt_tokens"]
        for item in ran
        if isinstance(item.get("final_prompt_tokens"), int)
    ]
    totals = {
        "instances_attempted": len(summaries),
        "instances_completed": len(ran),
        "instances_errored": len(errored),
        "solved": sum(int(bool(item["solved"])) for item in ran),
        "solve_rate": round(sum(int(bool(item["solved"])) for item in ran) / len(ran), 4)
        if ran
        else None,
        "dispatched_tool_calls": sum(int(item["tool_calls"]) for item in ran),
        "failed_tool_calls": sum(int(item["failed_tool_calls"]) for item in ran),
        "total_turns": sum(int(item["turns"]) for item in ran),
        "mean_turns_per_trajectory": round(
            sum(int(item["turns"]) for item in ran) / len(ran), 2
        )
        if ran
        else None,
        "max_final_prompt_tokens": max(prompt_tokens) if prompt_tokens else None,
        "median_final_prompt_tokens": (
            sorted(prompt_tokens)[len(prompt_tokens) // 2] if prompt_tokens else None
        ),
        "stop_conditions": _counted(item["stop_condition"] for item in ran),
    }
    return {
        "invocation": {
            "base_url": args.base_url,
            "model": args.model,
            "seeds": args.seeds,
            "seed_start": args.seed_start,
            "families": args.families or "all",
            "max_turns": args.max_turns,
            "max_output_tokens": args.max_output_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
        "totals": totals,
        "by_family": by_family,
        "instances": summaries,
    }


def _counted(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
