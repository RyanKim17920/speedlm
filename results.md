# SpeedLM — raw material and pointers

Status 2026-08-22. This file points at source material. It does not contain
rendered output, and nothing below depends on a video I generated.

## What SpeedLM does

A wrapper around `vllm serve`. It captures real serving traffic, waits until the
server goes idle, trains an Eagle3 speculative-draft head on that captured
traffic, benchmarks the new head against the stock one on a held-out slice, and
promotes or rejects it. Same serve command as normal, plus one flag.

## Raw material that still exists

| what | where | notes |
|---|---|---|
| terminal session recording | `/data/ryan.kim/speedlm-demo/raw/session_fast.cast` | asciinema v2, 100x30, ~72s real time. Byte-level capture of a real shell. Replayable with `asciinema play`. This is the only raw recording that survived. |
| step script that drove it | `demo/session_fast.json` | 13 steps; each runs a real command against real artifacts |
| measurement record | `docs/speedup-ceiling.md` | the gate numbers, the K=3 vs K=5 result, the data-scaling fit |
| earlier measurement | `docs/agentic-selfplay-result.md` | the superseded +0.2989 result and its leakage history |
| harness description | `docs/e2e-harness.md`, `docs/benchmark-evidence.md` | |
| task generator | `tests/e2e/agentenv/` | catalog, families, phrasing |
| traffic driver | `scripts/run_agentic_traffic.py` | |

## What is gone

`/data/ryan.kim/speedlm-runs/` (142 GB) was deleted 2026-08-21 in an approved
cleanup; see `/data/ryan.kim/.cleanup_manifest_20260821.txt`. That removed the
captured corpora, all training runs, every `decision.json`, and the race capture
timelines. None of it can be recovered. Any figure below comes from the docs
listed above, not from a live artifact.

## Measured results

From `docs/speedup-ceiling.md` (gate `regate-big-run2`, 287 held-out
session-disjoint contexts, 8 scored repeats, greedy, 512-token cap):

- accepted length, stock 2.3051 -> tuned 2.6507
- delta +0.3457, standard error 0.0029, i.e. +15.0%
- acceptance +11.52 percentage points
- reproduced across three independent gates: +0.3416, +0.3402, +0.3457
- throughput was VETOED as non-stationary on a shared node. Four gates, four
  vetoes. Quote accepted length, not a throughput number.
- ceiling: with `num_speculative_tokens=3` the arithmetic bound on accepted
  length is 4.0, so 20-40% is not reachable with this draft head. K=5 raised
  accepted length but lowered candidate throughput.

## Traffic generator

The corpus that trains the head is generated agent traffic. Task families live
in `tests/e2e/agentenv/catalog.py` and `families_v2.py`; per-seed instruction
wording comes from `tests/e2e/agentenv/phrasing.py`.

Twelve families: bugfix-localize, feature-implement, log-triage,
refactor-rename, schema-migrate, call-chain-trace, flaky-test-quarantine,
dep-version-conflict, api-contract-drift, perf-hotspot, config-precedence-bug,
error-swallow-audit.

`instance(seed)` is a pure function of the seed, so any workspace and prompt
shown can be regenerated exactly. To produce sample prompts without a GPU:

    from tests.e2e.agentenv.catalog import all_instances
    for i in all_instances(seeds=3, seed_start=0):
        print(i.family, i.instruction)

Last measured corpus: 3,777 records over 707 trajectories, 707 distinct opening
prompts. The prior corpus was 3,758 records over 601 trajectories with 6
distinct prompts, so the two are size-matched and differ in diversity.

## Areas a "how it works" video needs to cover

Each item names where the real thing lives. No mock-ups.

1. The serve command, stock vs SpeedLM — one added flag. `demo/session_fast.json`.
2. Traffic accumulating in the trace store during normal serving.
3. The client going away, and the idle threshold tripping on its own.
4. The lifecycle transitions READY -> QUIESCING -> SLEEPING -> EXTRACTING ->
   TRAINING, emitted to `events.jsonl` by the running gateway.
5. Training the draft head on the captured traffic.
6. The self-benchmark: replaying a held-out, session-disjoint slice against both
   the stock and the new head.
7. The gate deciding promote or reject, including a reject. Gate logic is in
   `src/speedlm/gate/`.
8. Side-by-side decoding of the same prompts under both heads, showing identical
   output and different token counts per verifier step.

Regenerating 2 through 8 with live data requires GPU jobs: `demo/traffic-v2.sbatch`
for capture, `demo/cyclev2.sbatch` for train and gate, `demo/capture.sbatch` for
the paired decode. These are multi-hour and were preempted four times.

## Known defects

- `feature-implement` is broken. `--max-output-tokens 1024` truncates Qwen3's
  reasoning block before it emits a tool call, so trajectories end after about
  two turns with no tool calls. Needs a larger cap and a re-capture.
- About 80% of captured records are dropped before training because self-play
  attestation cannot attribute the reasoning turns. The prior corpus was worse
  at 83%, so this predates the current generator. The backend flag that would
  trust those turns sits upstream of the promotion gate, which has no rollback.

## Generated, not raw

For completeness: `/data/ryan.kim/speedlm-demo/` holds videos and clips I
composed, with my own title cards, footers and chart overlays burned in. They
are not source material and are not referenced above.
