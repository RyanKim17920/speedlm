# Why we cannot show a 20-40% throughput win

Status as of 2026-08-13. Companion to `docs/agentic-selfplay-result.md`, which
establishes that the idle-tuning gain is real. This document explains why the
gain is the size it is, and what would have to change for it to be bigger.

Every number below is traceable to a `decision.json` or a line of source. Rows
marked *extrapolation* are model output, not measurement.

## The one-line answer

With `num_speculative_tokens=3` and this Eagle3 draft head, roughly **+15% is the
ceiling**. The honest, reproducible claim is **accepted length 2.3051 -> 2.6507
(+0.3457, SE 0.0029)** — 15.0% more tokens accepted per verifier step — measured
on 287 session-disjoint contexts over 8 repeats
(`/data/ryan.kim/speedlm-runs/regate-big-run2/decision.json`, job 380736).
Throughput is the same effect divided by a baseline that drifts on a shared node,
which is why it reads anywhere from +9.9% to +19.9% across four gates of the same
two heads.

## 1. The arithmetic ceiling

The draft head proposes `K` tokens per verifier step, and the verifier always
emits at least the one token it would have emitted anyway. So

```
accepted_length = 1 + accepted_draft_tokens / num_drafts     <=  K + 1
```

That is literally the metric definition —
`src/speedlm/gate/metrics.py:282-284`:

```python
mean_accepted_len = (
    1.0 + delta_accepted / delta_drafts if delta_drafts > 0 else 0.0
)
```

`K` for this pair is 3: `QWEN_3_8B_EAGLE3_PROFILE` sets
`num_speculative_tokens=3` at `src/speedlm/profiles.py:1141`, and
`ModelProfile.speculative_depth` returns it verbatim for a fixed-K profile
(`src/speedlm/profiles.py:341`). Every gate below confirms it in
`regate_wiring.json` -> `vllm_argv` -> `{"num_speculative_tokens":3}`. The hard
cap on the field is 16 (`MAX_SPECULATIVE_TOKENS`, `src/speedlm/profiles.py:63`),
so K=3 is a choice, not a limit — see section 4 for what happens when we raise it.

**Bound: 4.0.** Where the two heads sit (run2, the best-conditioned gate):

| | accepted length | as fraction of the 4.0 bound |
|---|---|---|
| bound (K+1) | 4.0000 | 100% |
| stock draft | 2.3051 | 57.6% |
| idle-tuned candidate | 2.6507 | 66.3% |
| headroom above stock | 1.6949 | — |
| **captured by tuning** | **+0.3457** | **20.4% of the remaining headroom** |

Note the gate's `acceptance_rate` is exactly `(accepted_length - 1) / K`:
(2.6507 - 1)/3 = 0.5502 = `candidate_avg_acceptance`, and (2.3051 - 1)/3 = 0.4350
= `stock_avg_acceptance`. The two reported quantities are one measurement.

### What +20 / +30 / +40% would require

Speculative decoding emits `accepted_length` tokens per verifier step, and the
cost of a step is fixed — one verifier forward over K+1 positions plus K draft
forwards — regardless of how many of those drafts survive. So to first order

```
tok/s_candidate / tok/s_stock  =  AL_candidate / AL_stock
```

This is not assumed; it is checked against the four gates below (the "conversion"
column). Two derivations, one per plausible conversion:

**(a) Ideal conversion (1.00), the optimistic floor on what is needed.**
Required AL = 2.3051 x (1 + g):

| target | required AL | delta vs stock | implied per-position acceptance p | headroom consumed |
|---|---|---|---|---|
| +20% | 2.766 | +0.461 | 0.758 | 27.2% |
| +30% | 2.997 | +0.692 | 0.810 | 40.8% |
| +40% | 3.227 | +0.922 | 0.858 | 54.4% |

**(b) Measured conversion from the cleanest gate (run5, 0.77).** Required AL =
2.3051 x (1 + g/0.769):

| target | required AL | delta vs stock | implied per-position acceptance p | headroom consumed |
|---|---|---|---|---|
| +20% | 2.905 | +0.600 | 0.790 | 35.4% |
| +30% | 3.205 | +0.900 | 0.854 | 53.1% |
| +40% | 3.505 | +1.200 | 0.912 | 70.8% |

`p` is solved from the geometric acceptance model `AL = 1 + p + p^2 + p^3`
(uniform per-position acceptance; the true chain is position-dependent, so treat
`p` as an average, *extrapolation*). For reference the same solve puts stock at
p = 0.638 and the tuned head at p = 0.730.

So: the brief's "+0.75-ish accepted length for +20%" is slightly pessimistic —
the real requirement is **+0.46 to +0.60**, with +0.75 landing between the +20%
and +30% targets. Either way it is **1.3x to 1.7x the entire gain we measured**,
and it demands that the draft head guess the verifier's *third* token correctly
about three-quarters of the time.

## 2. What we measured, four times

All four gate the same two arms (stock `RedHatAI/Qwen3-8B-speculator.eagle3` vs an
idle-tuned copy), `eager` execution, replay concurrency 8, benchmark max tokens
512, `dispersion: measured`, both arms `bounded` on truncation, zero output
mismatches.

| gate | head (rendered training rows) | contexts | warmup | repeats | accepted-length delta (SE) | throughput delta (SE) | final verdict |
|---|---|---|---|---|---|---|---|
| `regate-unseen-run1` (job 378546) | run5, 369 rows | 100 | 1 | 5 | **+0.2989** (0.0046) | +9.94% (0.64) | `promote` / `both_thresholds_met` |
| `bigcycle-run1` run `e7004c4c` | big, 652 rows | 287 | 1 | 5 | **+0.3416** (0.0040) | +14.76% (3.39) | `promote` / `both_thresholds_met` |
| `regate-big-run1` (job 380566) | big, 652 rows | 287 | 3 | 5 | **+0.3402** (0.0028) | +19.91% (5.56) | `promote`, `vetoed: false` |
| `regate-big-run2` (job 380736) | big, 652 rows | 287 | 3 | **8** | **+0.3457** (0.0029) | +16.15% (0.83) | **`reject` / `throughput_not_stationary`** |

Paths: `/data/ryan.kim/speedlm-runs/{regate-unseen-run1,regate-big-run1,regate-big-run2}/decision.json`
and `/data/ryan.kim/speedlm-runs/bigcycle-run1/speedlm_home/runs/e7004c4c0c7548fba65b05a924aa57ea/decision.json`.

> Correction to the framing this document was commissioned under: only **run2**
> carries a stationarity veto. `bigcycle-run1` has no `vetoed` field at all (it
> predates the veto), and `regate-big-run1` records `"vetoed": false` with
> `final_verdict: "promote"`. The veto fired exactly once, on the run with the
> most repeats.

**Accepted length reproduced. Throughput did not.** On the same head and the same
287-context suite, accepted length landed at +0.3416, +0.3402, +0.3457 — a spread
of 0.0055, comparable to a single run's standard error. Throughput on those same
three runs landed at +14.76%, +19.91%, +16.15% — a spread of 5.2 points.

The reason is structural: accepted length is a *ratio internal to one arm*
(accepted drafts per draft), while throughput is a *ratio across arms* whose
denominator is a wall-clock baseline on a shared node.

run2's own numbers are the proof. Per-repeat throughput deltas, in order:

```
+13.92  +13.98  +13.16  +12.94  +17.48  +20.72  +18.28  +19.16
```

The candidate arm is flat — `candidate_throughput_trend_pct_per_repeat = +0.051`
(%/repeat), `candidate_throughput_flat_from_repeat = 0`. The stock arm is not —
`stock_throughput_trend_pct_per_repeat = -0.821`, and stock only settles at
`stock_throughput_flat_from_repeat = 5`. Stock decayed from 127.05 to 121.37
tok/s across the block while candidate held 144.7-146.7. The delta grew ~6 points
without the candidate changing at all. That is exactly the veto the gate fired.

The same mechanism, larger, explains `regate-big-run1`: both arms were decaying
(`-3.55` and `-4.35` %/repeat) and its throughput SE is 5.56 points. Its +19.91%
is the number a contended node produces, not a property of the head.

Accepted length over the same eight repeats moved between +0.3365 and +0.3551.

## 3. Why more data has diminishing returns

Both heads were trained the same way; only the corpus size differs.

| head | rendered training rows | source | accepted-length delta |
|---|---|---|---|
| run5 | **369** | `agentenv-qwen8b-run5/.../stage-logs/training-row-rendering/stderr.log` ("Loaded 369 samples") | +0.2989 |
| big | **652** | `bigcycle-run1/.../e7004c4c.../stage-logs/training-row-rendering/stderr.log` ("Loaded 652 samples") | +0.3457 |

1.77x the data bought **+0.047** accepted length. (Caveat: the two are gated on
different suites — 100 contexts vs 287 — so this is not a perfectly controlled
comparison. It is the only data-scaling evidence we have.)

**Extrapolation, not measurement.** Fitting the standard log-linear scaling form
through those two points, `delta = -0.187 + 0.0822 x ln(rows)`:

| rows | predicted accepted-length delta | note |
|---|---|---|
| 652 | 0.346 | fitted |
| 1,304 (2x) | 0.403 | *extrapolation* |
| 2,608 (4x) | 0.460 | *extrapolation* |
| 10,000 (15x) | 0.570 | *extrapolation* |
| 100,000 (153x) | 0.759 | *extrapolation* |

Inverting: the +0.461 needed for +20% under the *ideal* conversion arrives at
~2,650 rows (4x). The +0.600 needed under the *measured* conversion arrives at
~14,400 rows (22x). A +0.75 delta needs ~89,000 rows (137x).

A two-point log fit is weak evidence and will overstate the tail if the curve
saturates before the 4.0 bound — which it must, since it cannot cross it. The
defensible reading is: **4x more in-domain traffic plausibly reaches the low end
of the +20% requirement and nothing more**; +30% and +40% are off the table for
data scaling alone at any corpus size we could realistically collect.

## 4. Why raising K does not help

`regate-k5-run1` (jobs 379899/379948) re-gates the **same run5 head, unchanged**,
against the **same 100-context suite** (`suite_hash 6116537d...bec1022`), with
only `num_speculative_tokens` changed to 5 via `profile-override.json` —
confirmed in `regate-k5-run1/regate_wiring.json` -> `vllm_argv`.

The ceiling rises 4.0 -> 6.0, and the accepted-length delta grows as predicted:

| | K=3 (`regate-unseen-run1`) | K=5 (`regate-k5-run1`) |
|---|---|---|
| stock accepted length | 2.3107 | 2.4821 |
| candidate accepted length | 2.6096 | 2.9469 |
| **accepted-length delta** | +0.2989 | **+0.4648** |
| stock tok/s | 130.66 | **109.79** (-16.0%) |
| **candidate tok/s** | **143.65** | **132.14** (-8.0%) |
| throughput delta | +9.94% | +20.36% |

Read the absolute column, not the delta column. Going K=3 -> K=5 cost the
candidate **8.0% of its absolute throughput** (143.65 -> 132.14 tok/s) and cost
stock 16.0% (130.66 -> 109.79). Each extra draft position is a full forward pass
through the draft model, paid on every verifier step whether or not that token is
accepted; per-position acceptance also falls (p = 0.730 at K=3 vs 0.701 at K=5 on
the candidate), so the marginal draft tokens are the least likely to survive.

The K=5 gate's headline +20.36% is therefore the most misleading number in this
whole set: it is +20% *relative to a stock arm that K=5 damaged more than it
damaged the candidate*. In absolute terms K=5 made serving slower. K=3 is
already at or near the optimum for this pair.

## 5. What would actually move it

Ordered by expected value. Nothing here is measured — all four are **speculative**
until gated.

1. **A larger or better draft architecture** (*speculative*). The gain we are
   chasing is per-position acceptance p: 0.638 -> 0.730 is what tuning bought; we
   need ~0.79 for +20%. That is an architecture and capacity question, not a
   corpus question. The Eagle3 head here is small by design; a deeper head with a
   proper tree/multi-branch proposal would raise the expected accepted prefix at
   fixed K — at some draft-forward cost that section 4 shows is not free. No
   measurement.
2. **A different verifier/draft pair** (*speculative*). Acceptance is a property
   of the pair. `src/speedlm/profiles.py` already carries three others
   (gpt-oss-20b, Llama-3.1-8B, qwen3.5-9b-mtp — the latter two at K=5). A larger,
   slower verifier makes each draft forward relatively cheaper and shifts the
   K optimum upward, which is exactly the regime where speculative decoding pays.
   Untested here.
3. **Far more in-domain traffic** (*extrapolation, section 3*). ~4x is the
   plausible route to the low end of +20%; beyond that the fit is not trustworthy.
   Cheapest thing to try, smallest guaranteed payoff.
4. **Accept a latency-for-throughput tradeoff at low concurrency**
   (*speculative*). Every gate here runs `replay_concurrency: 8` with
   `--enforce-eager`. Speculative decoding pays best when the verifier is
   latency-bound and the GPU has idle capacity; at concurrency 1-2 the same head
   should convert its accepted-length gain into a much larger wall-clock win, and
   at high concurrency into approximately none. We have never gated at another
   concurrency. This is the only item on the list that could plausibly show
   +20-40% *with the head we already have* — and it would be a different claim
   (per-request latency at low load), not a throughput claim.

## 6. What to claim

> Idle tuning the Qwen3-8B Eagle3 draft head raises mean accepted length from
> 2.3051 to 2.6507 (+0.3457, SE 0.0029) on 287 session-disjoint agentic contexts
> — 15.0% more tokens accepted per verifier step, 20.4% of the headroom to the
> K=3 arithmetic bound of 4.0. Wall-clock throughput moves with it, measured
> between +9.9% and +19.9% across four gates; the spread is baseline drift on a
> shared node, not variation in the head.

Do not claim a single throughput percentage without also stating the gate, the
repeat count, and the arm trends. Accepted length is the trustworthy half of the
measurement, as `docs/agentic-selfplay-result.md` already notes.

## Reproducing

```bash
sbatch scripts/regate_unseen.sbatch     # run5 head, 100 unseen contexts, K=3
# regate-big-run2: /data/ryan.kim/speedlm-runs/regate-big-run2/regate_big.sbatch
#   config bigcycle-qwen8b-warmup3.json (warmup_repeats 3, repeats 8)
#   suite  bigcycle-run1/.../e7004c4c.../held-out  (hash 4b8e9d19...20654c5)
# regate-k5-run1: same as regate_unseen plus profile-override.json
#   setting num_speculative_tokens=5
```

To reduce throughput noise rather than accepted-length noise, raise
`warmup_repeats` and `repeats` — run2 (warmup 3, 8 repeats) has a throughput SE
of 0.83 points against run1's 5.56 on the identical head and suite.
