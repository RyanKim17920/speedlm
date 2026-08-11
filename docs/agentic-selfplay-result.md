# Agentic self-play idle tuning: what the measurement actually shows

Status as of 2026-08-11. Supersedes the headline numbers in
`configs/README-agentenv-launch.md`, which record the original runs without the
leakage correction below.

## The claim, corrected

Idle tuning a Qwen3-8B Eagle3 draft head on agentic self-play traffic **does**
improve serving on traffic the head has never seen:

| | accepted length | tok/s |
|---|---|---|
| stock draft | 2.3107 | 130.7 |
| idle-tuned candidate | 2.6096 | 143.7 |
| **delta** | **+0.2989** (SE 0.0046) | **+9.94%** |

Verdict `promote`, `both_thresholds_met`, against thresholds of +0.05 accepted
length and -2.0% throughput. 100 contexts, 5 repeats, both arms `bounded` on
truncation, dispersion `measured`.

The originally reported +0.6525 / +32.3% was **inflated roughly 2.2x by
session-level leakage in the held-out split**. The gain is real; the magnitude
was not.

## Why the first number was wrong

The held-out split reserved individual *record* hashes, and its leakage guard
asserted only that no benchmark context was byte-identical to a training row.
For single-turn chat that is genuine disjointness, because records are
independent. Agentic traffic is different: record N's `messages` list contains
record N-1's verbatim, so the turns of one agent session are nested rather than
independent.

Consequence, measured on the promoted runs: ~85 of 100 held-out contexts came
from sessions that also contributed training rows; for ~75 the training corpus
held a *later* turn of the same session, whose prefix is the very continuation
the draft head is scored on predicting. Median longest common substring between
a gate context and its nearest training row was ~6,000-7,300 characters.

This also explains why the earlier generic-chat runs showed nothing
(+0.024/+0.0227/+0.0218). Those corpora are single-turn, so the same guard
enforced real disjointness. The jump to +0.65 coincided with a change in
*leakage structure*, not only a change in domain.

Fixed in the commit "hold out whole agent sessions, because multi-turn records
nest": sessions are now reserved atomically, keyed by their opening exchange.

## How the correction was measured

`scripts/regate_unseen.sbatch` (job 378546) gates the **already-trained** run-5
head — no retraining — against two suites in a single allocation:

| | suite | accepted length delta | tok/s delta |
|---|---|---|---|
| **UNSEEN** | 100 contexts from 61 sessions contributing **zero** training rows | **+0.2989** | +9.94% |
| **CONTROL** | run 5's original suite, re-gated here | +0.6557 | +28.68% |
| reference | run 5 as originally recorded | +0.6525 | +32.25% |

The control is what makes this conclusive twice over. It reproduces the original
+0.6525 to within 0.005 on the same GPU in the same job, which validates the
re-gate harness and rules out node, thermal state and engine build as
explanations. And because both suites were scored against the same head in the
same allocation, the **suite is the only thing that differs** between +0.6557 and
+0.2989.

Both suites pass the gate's own leakage assertion. That is the point: the
assertion is row-level, and the original suite satisfies it while being
session-contaminated.

## What this does and does not establish

Established: on agent sessions it never trained on, the idle-tuned head drafts
meaningfully better than stock — 6x the promotion threshold, with a standard
error two orders of magnitude below the effect.

Not established:

- **Domain transfer.** An unseen *session* of a seen *task family* is still
  familiar traffic. All six families appear in both training and the unseen
  suite. This measures session-level generalization, not transfer to unfamiliar
  agentic work.
- **Suite composition.** The unseen suite runs deeper (median input depth 16 vs
  ~10 for the original), so some of the gap between +0.2989 and +0.6557 is prompt
  length rather than leakage alone. The direction is not in doubt; the exact
  split between the two causes is unquantified.
- **Throughput precision.** Run 5's own decision carried
  `throughput_stationarity: material_shift_unresolved`. Accepted length is the
  more trustworthy of the two numbers throughout.

## Reproducing

```bash
sbatch scripts/regate_unseen.sbatch          # ~30 min, 1 GPU, both arms
```

The suite build is deterministic: `scripts/build_unseen_session_suite.py`
rebuilds to suite hash `6116537d2d3aab44e404e66f769b0a85ba8f460e3e68769eaaeb96802bec1022`
and fails closed unless it recovers exactly the 412 training rows the run-5
gateway log recorded.
