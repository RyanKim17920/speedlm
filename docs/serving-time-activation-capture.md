# Serving-time activation capture

**Status:** design proposal, with Stage 0 (Section 9) prototyped. Sections 1-8
remain a proposal: no cache, no retention, no training integration exists. What
does exist is the Stage 0 de-risking prototype —
`src/speedlm/activation_capture/`,
`tests/e2e/test_serving_activation_capture.py`, and
`tests/e2e/test_capture_overhead.py`.

**Stage 0's last open exit criterion — the serving latency cost — has now been
measured (Section 7.2.1).** Two things came out of it that change how this
document should be read. First, capture was **non-functional under CUDA graphs**,
the production default, until commit `c5aa14c`; every earlier correctness result
in this document was obtained in eager mode (Section 7.2.2). Second, the first
honest graphed measurement showed capture costing **-14.49% throughput**, which
is not a rounding error on the "byproduct of serving" framing in Section 1;
deferring the transfer synchronization brought steady-state cost to
statistically zero and left **+4.40 ms of TTFT**.

**Read Section 6.2 before citing Stage 0 results.** Stage 0's original
comparison (serving capture vs. vLLM's offline extraction) returned a perfect
`0.0` on every layer, and that number was initially read as settling the
correctness question. It does not. The two paths transport a single tensor
object rather than deriving it twice, so bit-identity is the expected outcome
and says nothing about *which* quantity was captured. The independent evidence
comes from a separate float32 HuggingFace re-derivation, added afterwards. That
leg first ran successfully in job 369256 (2026-08-01), which **met** the
identity criterion — every claimed layer was the strict argmin over its
neighbours by 23-81x against a 3x margin — on 18 prompt rows of one prompt on
one model. The table in Section 9 says which question each leg answers, and the
limitations directly beneath it say what 369256 does not cover.

**Audience:** engineers on SpeedLM who have not followed the investigation that
produced it. It is written to stand alone.

**Citation convention.** Every technical claim carries a `file:line` reference.
Three source trees are cited:

| Symbol | Path | Version |
| --- | --- | --- |
| `V` | `/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/lib/python3.12/site-packages/vllm` | vLLM 0.25.1 |
| `S` | `/admin/home/ryan.kim/speedlm/.preflight/speculators` | Speculators v0.6.0 |
| (bare) | this repository, `/admin/home/ryan.kim/speedlm-fr` | — |

Line numbers are pinned to those exact checkouts and will drift on upgrade.
Anything that could **not** be confirmed against source is explicitly labelled
**[unverified]**. Please keep that discipline if you edit this document: on this
system, confident inference has repeatedly turned out to be wrong, and several
figures in the first draft of this proposal did not survive being checked.

---

## 1. Summary

Today, the EAGLE-3 idle-tuning cycle throws away the verifier activations it
needs and then pays to recompute them: it launches a **second** full vLLM engine
during the idle window purely to re-run already-served tokens through the
verifier and dump aux hidden states, then tears that engine down.

> **Proposal:** capture the verifier's intermediate activations during **normal
> serving**, write them to a cache, and train the draft head from that cache
> during the existing sleep window — deleting the separate extraction stage
> entirely.

The premise is confirmed in source: the exact tensors the extraction stage
recomputes are already materialized on every forward pass, for every token
position, in the serving engine, and are discarded a few lines later
(Section 4). The marginal cost of keeping them is a host-side copy of roughly
17-23 KiB per token, which at observed serving rates is a fraction of a percent
of PCIe bandwidth (Section 7).

This is not a free win. vLLM exposes no supported hook for this, so any
implementation is an **owned dependency on vLLM internals** (Section 5), and
there are at least five ways to capture activations that are *silently wrong*
rather than loudly broken (Section 6). This project has already shipped two
silent-wrong-data bugs to production, which is the reason Section 6 insists that
every mitigation ship with a detector.

---

## 2. Current architecture and its measured cost

### 2.1 What happens today

`speedlm vllm serve` runs a gateway in front of a managed vLLM child, journals
every admitted exchange, and normalizes completed request/response pairs into a
bounded trace store (README, **How it is structured**). When the gateway goes
quiet, the tuner closes admission, drains, sleeps the serving engine, and runs a
tuning cycle (README, **Promotion thresholds**).

Inside that cycle, the EAGLE-3 backend must produce a hidden-states file for
Speculators training. It does so by shelling out to the Speculators
`prepare_data.py` script to build the arrow dataset
(`SpeculatorsTrainingRowRenderer.render_rows` in
`src/speedlm/training/backends/eagle3.py`), validating the prepared dataset in a
second subprocess with `--require-nonzero-loss-mask`, and then running a
**separate hidden-state generation stage** that stands up its own vLLM engine
over the verifier.

That second engine needs the verifier weights resident in GPU memory at the same
time the serving engine's allocation must be released, which is why the cycle
carries a hard memory precondition before it will proceed
(`create_production_tuner` in `src/speedlm/tuner/composition.py`, default in
`SpeculatorsPipelineConfig`, enforcement in `GPUMemoryPrecondition` in
`src/speedlm/gateway/control.py`). See Section 8 for exactly what that
precondition says — the popular one-line summary of it is wrong.

### 2.2 Measured cost — read the caveats

Durations below are derived from state-transition timestamps in the 12
`events.jsonl` files under `log_artifacts/*/results/live-idle-tuning/speedlm_home/runs/`.
The event schema records state transitions only; durations are timestamp deltas
between consecutive states.

**Corpus statistics (all recorded cycles):**

| Stage | n intervals | mean (s) | min (s) | max (s) |
| --- | --- | --- | --- | --- |
| `EXTRACTING` | 22 | 69.1 | 0.04 | 240.9 |
| `TRAINING` | 6 | 181.8 | 100.1 | 220.5 |
| `CANDIDATE_STARTING` | 4 | 113.1 | 101.3 | 126.7 |
| `BENCHMARKING` | 4 | 324.9 | 224.1 | 496.5 |
| `SLEEPING` | 22 | 6.43 | 5.75 | 11.4 |
| `WAKING` | 21 | 0.067 | 0.060 | 0.081 |
| `ROLLING_BACK` | 20 | 105.2 | 90.7 | 125.0 |

**The corpus is heavily truncated and the means are not representative of a
successful cycle.** Of 21 complete cycles, only 6 reached `TRAINING`, 4 reached
`BENCHMARKING`, and 1 reached `PROMOTING`; 16 of 22 aborted straight from
`EXTRACTING` into `ROLLING_BACK`, 11 of those within 0.25 s. The low
`EXTRACTING` mean is dominated by those instant aborts, not by real extraction
work.

**A single fully-successful cycle** — the second cycle of
`log_artifacts/live-idle-thresholds-20260730T213537Z/` — is the honest reference
point for what a working cycle costs:

| Stage | Duration (s) | Share of the four work stages |
| --- | --- | --- |
| `EXTRACTING` | 199.6 | 19.5% |
| `TRAINING` | 220.5 | 21.5% |
| `CANDIDATE_STARTING` | 107.8 | 10.5% |
| `BENCHMARKING` | 496.5 | 48.5% |
| **Sum of the four** | **1024.4** | 100% |
| Full wall clock for the cycle | **1134.8** | — |

Corrections to figures that have circulated informally, so they are not repeated:

- The 220.5 s `TRAINING` and 496.5 s `BENCHMARKING` numbers are that cycle's
  values, which happen to be the **corpus maxima**, not means.
- "`BENCHMARKING` 496.5 s, n=5" is wrong. Only **4** `BENCHMARKING` intervals
  exist in the entire corpus. The `benchmark_repeats` default of 5
  (README, **Promotion thresholds**; `src/speedlm/config.py`) is a count of
  held-out suite passes *inside* one benchmark stage, not a sample size for the
  stage duration.
- "1024.4 s total cycle" is the sum of four work stages of one cycle, not its
  wall clock (1134.8 s), and not a corpus mean (295.0 s over 21 complete cycles,
  range 97.8-1134.8 s — but that mean is dominated by aborts).
- "Up to 120 s of GPU-memory-release wait" is **not a measurement**. It is the
  constant `DEFAULT_GPU_MEMORY_TIMEOUT_SECONDS = 120.0`
  (`src/speedlm/gateway/control.py`) — a timeout ceiling. No stage in the
  corpus measures a GPU-release wait, and `SLEEPING` never exceeded 11.4 s.

### 2.3 The shape of the problem

Sleeping the engine costs ~6.4 s and waking it costs ~0.067 s. Against that, the
cycle spends ~200 s extracting, and both `CANDIDATE_STARTING` (~113 s) and a
large share of `BENCHMARKING` are believed to be vLLM engine startup and weight
loading. The event data does not separate startup from replay within that stage.

So the cycle is overwhelmingly **fixed overhead — process starts and weight
loads — rather than useful work.** The frequently-quoted "~90% overhead, ~10%
useful work" split is a reasonable characterization of that single reference
cycle but is **[unverified]** as a precise decomposition: the event schema
records state transitions, not a breakdown of time *within* a stage, so we
cannot separate "engine startup" from "actual work" inside `BENCHMARKING` or
`TRAINING` from this data. Treat it as a qualitative claim.

What is not in doubt is that `EXTRACTING` recomputes tensors the serving engine
already had.

---

## 3. Proposed architecture

### 3.1 Concept

```
                    ┌──────────────────────────────────────┐
   client traffic ─►│ gateway ─► vLLM serving engine       │
                    │              │                       │
                    │              ├─► tokens out to client│
                    │              └─► aux hidden states ──┼──► activation cache
                    └──────────────────────────────────────┘        (on disk)
                                                                          │
   idle window:  sleep engine ──► train from cache ──► candidate ──► gate │◄┘
```

Activations are captured as a byproduct of serving. The idle window no longer
needs the verifier at all — it needs only the cache and the draft head.

### 3.2 Before / after

| Cycle stage | Today | Proposed |
| --- | --- | --- |
| Detect idle, drain, close admission | unchanged | unchanged |
| Snapshot traces | unchanged | unchanged |
| Prepare training rows | subprocess to `prepare_data.py` (`SpeculatorsTrainingRowRenderer.render_rows` in `src/speedlm/training/backends/eagle3.py`) | still needed for token IDs + loss mask, unless Section 6.6 is resolved |
| **Sleep serving engine** | ~6.4 s | ~6.4 s |
| **`EXTRACTING`** | **~200 s: start a second vLLM engine, reload verifier weights, re-run ~1500 already-served tokens, dump hidden states, tear down** | **deleted** |
| `TRAINING` | ~180-220 s | unchanged |
| `CANDIDATE_STARTING` | ~113 s | unchanged |
| `BENCHMARKING` | ~225-500 s | unchanged |
| Promote or roll back | unchanged | unchanged |
| Wake serving engine | ~0.067 s | ~0.067 s |

The delta is the removal of one stage and one full engine lifecycle. On the
reference cycle that is 199.6 s out of 1134.8 s wall clock — about **18%**. It
also removes the memory precondition that gates the whole cycle (Section 8),
which on the recorded corpus is the more consequential effect: 16 of 22 cycles
aborted out of `EXTRACTING`.

**[unverified]** The 16 aborts have not been individually attributed to a cause.
Do not claim serving-time capture would have prevented them until someone reads
those failure records. It is plausible and it is not established.

### 3.3 What moves into the serving path

- A capture hook on the verifier forward pass that copies aux hidden states to
  pinned host memory asynchronously.
- An attribution step mapping rows of the captured tensor back to
  `(request_id, token_position)`.
- A writer that appends to an activation cache with an explicit retention policy
  (Section 7.3).
- Assertions that the cache is complete and correctly aligned (Section 6).

---

## 4. Evidence that the premise holds

The claim to establish: **the tensors `EXTRACTING` recomputes are already
produced, for all token positions, by the serving engine, and then dropped.**

**4.1 The serving engine turns aux hidden state output on for EAGLE-3.**
`V/v1/worker/gpu_model_runner.py:625-627`:

```python
if self.speculative_config.method == "eagle3":
    self.use_aux_hidden_state_outputs = (
        self.drafter.eagle3_use_aux_hidden_state)
```

**4.2 The aux layers are installed from the drafter's config.**
`V/v1/worker/gpu_model_runner.py:5402` calls
`self.model.set_aux_hidden_state_layers(aux_layers)`; the layer ids are read from
`eagle_aux_hidden_state_layer_ids` in `_get_eagle3_aux_layers_from_config`
(defined at `:5404`, reads at `:5420` and `:5432`).

**4.3 The model collects them inside the decoder loop.** The shared helper is
`SupportsEagleBase._maybe_add_hidden_state` at
`V/model_executor/models/interfaces.py:1329-1339` (`if layer_idx in
self.aux_hidden_state_layers:` → `aux_hidden_states.append(value)`). Each model
calls it from its own decoder loop; for the model this project serves,
`V/model_executor/models/gpt_oss.py:349-355`, returning at `:360-361`
(`if len(aux_hidden_states) > 0: return x, aux_hidden_states`). `GptOssForCausalLM`
declares `SupportsEagle3` at `V/model_executor/models/gpt_oss.py:1171`.

**4.4 The runner unpacks them.** `V/v1/worker/gpu_model_runner.py:4362-4368`:
`hidden_states, aux_hidden_states = model_output` when
`self.use_aux_hidden_state_outputs`, else `aux_hidden_states = None`.

**4.5 They cover every scheduled position, not just the sampled one.** The
eagle3 path concatenates `[h[:num_scheduled_tokens] for h in aux_hidden_states]`
at `V/v1/worker/gpu_model_runner.py:5119-5121`, i.e. all scheduled tokens of the
batch — prefill and decode alike. Compare the logits path at `:4387`,
`sample_hidden_states = hidden_states[logits_indices]`, which keeps only the last
position per request. **This is the crux of the proposal:** the expensive,
full-sequence quantity is already computed; only the last-position slice survives.

**4.6 They are consumed and immediately freed.** The concatenated tensor is fed
to `combine_hidden_states`
(`V/model_executor/models/llama_eagle3.py:359-379`, ending
`return self.model.fc(hidden_states)`), after which nothing retains it.

Note on width: the concatenation is `len(aux_hidden_states) * hidden_size`, not a
hardcoded `3 * H`. Three is the conventional aux-layer count, not a constant in
the code. The drafter's `fc` input width follows the same count —
`V/model_executor/models/llama_eagle3.py:187`:
`self.fc_input_size = target_hidden_size * self.num_aux_hidden_states`, consumed
at `:208-214`.

**4.7 Two other code paths take the same variable.** There are three
concatenation sites, differing only in how they index rows:
`:5119-5121` (contiguous, all scheduled tokens), `:5136-5138` (gather by
`token_indices`), `:5157-5159` (contiguous to `total_num_tokens`). And the
offline hidden-states connector path reads the identical variable at `:5035`
(`target_hidden_states = [h[:num_scheduled_tokens] for h in aux_hidden_states]`,
guarded by a raise at `:5031` if it is absent) — it keeps the list rather than
concatenating.

That last point matters twice over. First: **the offline extraction machinery
vLLM ships already reads exactly the variable we want to capture, from exactly
the serving-engine code path.** The gap is purely one of plumbing and activation
conditions (Section 5), not of the data existing.

Second, and less comfortably: it means a capture-vs-offline comparison is a
comparison of **one tensor against itself**, since neither side recomputes
anything from `:5035` onward. That is why Stage 0's `0.0` result is not the
correctness proof it looks like, and why an independent re-derivation was
needed. See Section 6.2.

**4.8 Attribution data is live at the same point.** Row → request mapping does
not need to be reconstructed. `req_indices = np.repeat(self.arange_np[:num_reqs],
num_scheduled_tokens)` at `V/v1/worker/gpu_model_runner.py:1937`,
`query_start_loc` construction at `:2027-2033`, and the batch's request ids via
the `req_ids` property at `V/v1/worker/gpu_input_batch.py:305` (backing field
initialized at `:125`) are all in scope where `aux_hidden_states` exists.
Attribution is **not** a blocker.

---

## 5. Mechanism options

vLLM registers no forward hooks and exposes no hook registry for this
**[unverified as an exhaustive absence claim — this is "we found none", not "the
code proves none exist"]**. Every option below is therefore a custom mechanism.

### 5.1 Why the shipped connector cannot simply be turned on

`ExampleHiddenStatesConnector`
(`V/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py`)
does almost exactly what we want, and does it well: a dedicated copy stream into
pinned host memory (`:351-431`, including
`pinned_hs = torch.empty_like(hidden_states_gpu, device="cpu", pin_memory=True)`
followed by a `non_blocking=True` copy), a writer thread pool (`:210-215`,
`ThreadPoolExecutor(max_workers=... "num_writer_threads", 8)`), and event-based
polling that never blocks the decode path (`:570-575`,
`if event is None or event.query():`).

But it is bound to the offline extraction configuration:

- It requires a speculative config at all. `:186-189` asserts
  `self._vllm_config.speculative_config is not None` with the message
  *"ExampleHiddenStatesConnector only works when using 'extract_hidden_states'
  speculative method"*. **Correction to a claim in circulation:** this is a
  not-None assertion whose *message* names the method; it does not compare
  `speculative_config.method == "extract_hidden_states"` at those lines.
  Whether an equality check exists elsewhere is **[unverified]**.
- It is designed around the fake-KV-cache wrapper model
  `ExtractHiddenStatesModel` (`V/model_executor/models/extract_hidden_states.py:339`,
  layers at `:364`, `forward` at `:377`), whose caching attention layer is
  `CacheOnlyAttentionLayer` at `:237` and whose fake KV cache shape is defined at
  `:127-136` (`num_kv_heads = num_hidden_layers`, `head_size = hidden_size`).
  An EAGLE-3 serving engine never instantiates that wrapper.

The copy pattern is proven in-tree and should be reused. The activation condition
is what must change.

### 5.2 Option A — `worker_extension_cls` plus a forward hook

`--worker-extension-cls` is a real, accepted `vllm serve` flag
(`V/engine/arg_utils.py:1145`), and the injection mechanism is explicit:
`V/v1/worker/worker_base.py:261-287` mixes the class into the worker's bases
(`worker_class.__bases__ = worker_class.__bases__ + (worker_extension_cls,)` at
`:279-281`) specifically "for extended collective_rpc calls".

Extension methods run inside the worker process, so they can reach the model and
register a PyTorch forward hook on the aux layers.

- **Pro:** the entry point (`--worker-extension-cls`) is a documented, supported
  flag. It requires no patching of vLLM files on disk.
- **Con:** what the extension *does* — reaching into `self.model_runner.model`
  and the layer objects — is entirely private API. The hook must independently
  reproduce the runner's row-slicing semantics (Section 6.5) because a layer-level
  hook sees the padded activation buffer, not `num_scheduled_tokens`.
- **Con:** `collective_rpc` is the natural way to drive an extension, and vLLM
  explicitly tells you not to use it as a data plane:
  `V/v1/executor/abstract.py:181-183`, *"It is recommended to use this API to only
  pass control messages, and set up data-plane communication to pass data."*
  Requests reach `EngineCore.collective_rpc` (`V/v1/engine/core.py:846-853`) via
  the input queue processed by the same busy loop that runs engine steps
  (`:1259-1266`, `_process_input_queue()` then `_process_engine_step()`, dispatch
  at `:1372` / `:1386`), so RPCs serialize against stepping. Use it for setup and
  teardown only; the actual bytes must leave by their own path.

### 5.3 Option B — patch the model runner

Add capture directly where `aux_hidden_states` is unpacked
(`V/v1/worker/gpu_model_runner.py:4362-4368`) or where it is concatenated
(`:5119-5121`).

- **Pro:** this is the *correct* place. Every hazard in Section 6 —
  `num_scheduled_tokens` slicing, `token_indices` gather, rejected-draft row
  layout — is already handled correctly a few lines away, so the patch inherits
  the right semantics instead of reimplementing them.
- **Con:** it requires shipping a patched vLLM (a fork, a build-time patch, or an
  import-time monkeypatch of a private method). This repository currently pins
  vLLM as an out-of-band part of the GPU host image and deliberately does not
  declare it as a runtime dependency (README, **GPU runtime stack**); a patch
  changes that posture.

### 5.4 Option C — upstream a connector change

Relax the connector's activation condition so a hidden-states connector can be
attached to a normal EAGLE-3 serving engine, and upstream it.

- **Pro:** the only option with a bounded long-term maintenance cost. The copy
  and writer machinery already exists and is already correct.
- **Con:** upstream latency is unbounded and outside our control. It cannot be
  the plan for the first working version; it can be the plan for the third.

### 5.5 Recommendation and the maintenance cost we would be signing up for

Prototype with **Option A**, because it does not require patching vLLM on disk
and can be stood up behind a flag. Expect to discover during the prototype
whether a layer-level hook can reproduce the runner's row semantics safely; if it
cannot, fall back to **Option B** for correctness and pursue **Option C** in
parallel as the exit strategy.

Be explicit about what is being accepted: **every option creates an owned
dependency on vLLM private internals.** Concretely, that means at minimum
`gpu_model_runner`'s aux-state unpacking and slicing, the worker base-class
injection mechanism, and the model-side `_maybe_add_hidden_state` contract. Every
vLLM upgrade becomes a re-validation event, and the failure mode of a silent
internals change is not an import error — it is a subtly wrong cache. The
detectors in Section 6 are therefore not optional polish; they are the only thing
that makes this dependency survivable across upgrades. Budget for them in the
same breath as the feature.

---

## 6. Correctness hazards

### 6.0 Why detection is mandatory, not optional

The failure mode this design must defend against is **silently wrong data**: a
cache that is populated, well-formed, correctly shaped, and materially incorrect.
Training on it produces a draft head that is merely worse, not broken — and the
gate is a statistical comparison, so a mildly-degraded candidate reads as noise,
not as a bug.

This is not hypothetical for this project. It has already happened at least twice
in production paths:

1. **Silently empty training dataset.** Traces were written under a `messages`
   key while the loader read `conversations`, so the prepared dataset came out
   empty and training "succeeded" on nothing. Fixed in commit `46a6308`. The
   guard that now makes it loud is `EmptySpeculatorsDatasetError`, raised when
   `written == 0` in `_render_speculators_dataset` in
   `src/speedlm/training/backends/eagle3.py`, pinned by
   `test_prepare_without_convertible_records_raises_named_error` in
   `tests/test_training_speculators.py`.
2. **Silently zero-sample gate.** The gate could return a favorable decision from
   an empty replay. Fixed in commits `6779110` and `20231d3`. The guard is the
   `Reason.TOO_FEW_REPEATS` rejection in `decide` in
   `src/speedlm/gate/decide.py`, pinned by
   `test_zero_sample_benchmark_can_never_promote` in
   `tests/test_gate_decide.py` (empty replay plus favorable metrics must still
   REJECT) and asserted end-to-end by `_assert_gate_measured_something` in
   `tests/e2e/test_live_idle_tuning.py`.

A third case is often cited as "silently all-zeros batch on stale cache." **That
bug does not exist in this repository's history.** What does exist is
commit `103d8f9`, all-zero metric *snapshots* caused by reading the wrong vLLM
counter names — same silent-wrong-data character, different mechanism. Two
stale-data bugs also exist but were not all-zeros: `d01101c` / `645e91c` (a stale
`decision.json` selected by mtime) and `9afcddc` (a stale HF cache, which failed
*loudly* with `IncompleteSnapshotError`). Cite the real ones.

The pattern across all of them is identical: **the system had no assertion that
the data it was about to use was real.** Each hazard below therefore lists a
mitigation *and* the detector that proves the mitigation is working. A mitigation
without a detector is not a mitigation.

### 6.1 Prefix caching — the highest-severity hazard

Prefix caching is on by default: `enable_prefix_caching: bool = True` at
`V/config/cache.py:93`. On a cache hit, the matched tokens are **not forwarded**;
the scheduler receives block references and a computed-token count from
`get_computed_blocks` (`V/v1/core/kv_cache_manager.py:206-246`, called at
`V/v1/core/sched/scheduler.py:724-725`,
`new_computed_blocks, num_new_local_computed_tokens = ...`).

**[Qualification]** This is an *absence* claim. The cited lines show that only
block references and a token count cross that boundary; they do not contain a
statement saying "no hidden states are produced." The inference is
straightforward — no forward pass, no activations — but it is inference, and it
should be settled empirically by the prototype rather than argued.

#### 6.1.1 Status: CONFIRMED empirically (job 369256), after a withdrawn retirement and a null first run

**Current status: the hazard is real and has been measured.** Job 369256
(`prefix_cache_result.json`, 2026-08-01) sent the same ~165-token prompt twice to
an engine with prefix caching at its default and recorded:

| Counter | Before | After |
| --- | --- | --- |
| `vllm:prefix_cache_queries_total` | 165 | 330 |
| `vllm:prefix_cache_hits_total` | **0** | **144** |
| captured rows/layer (cold → warm) | 213 | 69 |

144 hit tokens, and 213 − 69 = **144 prompt rows that produced no activation
row**. The two numbers come from independent sources — one counted off the
safetensors on disk, one read off `/metrics` — and they agree exactly. The
expectation was *derived* from vLLM's block arithmetic before the run
(`((165−1)//16)−1 = 9` hittable blocks × 16 = 144) and is asserted as an
equality, not as `> 0`, so a change to the block size, to eagle's
last-matched-block drop, or to the last-token recompute cap fails the test
rather than silently moving the goalposts.

**What changed since the withdrawal.** Three things, in order:

1. The hardcoded literals were removed and every field made measured — the
   retirement had rested on `{"cache_hit": true, "rows_missing": 0}` written by a
   test with no `assert`.
2. The prompt was lengthened from 18 to ~165 tokens, because at 18 tokens zero
   hits is the *correct* engine result (the arithmetic below) and the hazard is
   not representable at all.
3. The expected hit size was derived from source and asserted exactly.

So the hazard is no longer resting on hardcoded literals *or* on inference from
an absence claim: a cache hit demonstrably drops rows, in the predicted amount.

**One correction to the artifact.** Job 369256's `prefix_cache_result.json`
reports `rows_missing: 96`, computed as `prompt_token_count −
captured_rows_per_layer` = 165 − 69. That is wrong by 48: the warm capture's 69
rows are 21 prompt rows plus 48 *decode* rows, so the subtraction credited
decode rows as present prompt rows. The true figure is 144. No verdict moved —
the assertion was `> 0`, and the exact `hit_tokens` equality carried the proof —
but **do not quote 96**. The field is now named `prompt_rows_missing`, is
measured cold-minus-warm, and is asserted equal to the engine's hit-token count.

---

*History, retained because the retirement was cited elsewhere:*

This hazard was previously recorded in this document as **retired empirically**,
on the strength of `{"cache_hit": true, "rows_missing": 0}` in
`prefix_cache_result.json`. Those were **hardcoded literals**. The test that
"produced" them contained no `assert` at all and wrote both fields as constants;
it could not have observed a cache hit and did not. The retirement rested on a
value nobody measured, and it is withdrawn. Do not cite it.

**What the first real measurement says.** With the literals removed, job 369236
measured, for two identical requests on an engine with
`enable_prefix_caching=True`, `enable_chunked_prefill=True`, eagle3 speculative
decoding and a 153,056-token GPU KV cache:

| Counter | Before | After |
| --- | --- | --- |
| `vllm:prefix_cache_queries_total` | 18 | 36 |
| `vllm:prefix_cache_hits_total` | 0 | 0 |
| captured rows/layer (cold → warm) | 48 | 48 |

Prefix caching was engaged — queries advanced by exactly the 18 prompt tokens,
and `record()` is only reached past the `enable_caching` early return
(`V/v1/core/kv_cache_manager.py:222-244`) — but **nothing matched**.

**Root cause: the prompt was too short for a hit to be representable.** This is
correct engine behaviour, not a capture defect. Three independent truncations
apply to an 18-token prompt at the default block size of 16
(`V/config/cache.py:47`, `DEFAULT_BLOCK_SIZE: ClassVar[int] = 16`; FlashAttention
keeps 16 on CUDA, `V/v1/attention/backends/flash_attn.py:81-85`):

1. **Only full blocks are hashed.** `V/v1/core/kv_cache_utils.py:709-712` —
   `if end_token_idx > num_tokens: break  # We only hash full blocks`. Tokens
   16–17 are never hashed, so an 18-token prompt yields **one** block hash.
2. **The last token is always recomputed.**
   `V/v1/core/kv_cache_manager.py:225-231` — "When all tokens hit the cache, we
   must recompute the last token to obtain logits", so
   `max_cache_hit_length = request.num_tokens - 1` = 17, and
   `V/v1/core/single_type_kv_cache_manager.py:590` scans
   `max_num_blocks = 17 // 16` = **1** block.
3. **Eagle drops the last matched block.**
   `V/v1/core/single_type_kv_cache_manager.py:602-605` —
   `if drop_eagle_block and computed_blocks[0]: computed.pop()`, reached because
   `eagle3` is in `SpeculativeConfig.use_eagle()`
   (`V/config/speculative.py:1238-1242`), and the coordinator conservatively
   flags all groups when `use_eagle` is set
   (`V/v1/core/kv_cache_coordinator.py:100-105`).

One block hashed, one block scanned, that one block popped ⇒ **zero hits is the
only possible result**, for any 18-token prompt, on any run of this engine
configuration. The stats confirm rather than contradict this: `queries` counts
*tokens*, not blocks (`V/v1/metrics/stats.py:115-142`, "`queries`: Refers to the
number of tokens that were queried"), which is why it read 36 for two 18-token
requests.

Chunked prefill is not implicated: `get_computed_blocks` runs before any
chunking and reads the full `request.block_hashes`
(`V/v1/core/sched/scheduler.py:478-479`).

**Consequence for the hazard.** Nothing about job 369236 argues against 6.1 —
the experiment simply never reached the regime where it applies. The test moved
to a ~165-token prompt (10 hashable blocks, 9 hittable after the two structural
drops) and asserts the hit size exactly against the block arithmetic above.
**That run is job 369256 and it demonstrated the hazard**; see the top of this
subsection. 6.1 is confirmed, not argued-from-source.

**Minimum prompt length, for anyone reproducing this.** With block size 16, hit
blocks = `(N-1)//16 - 1` under eagle. So `N ≥ 33` for any hit at all, and
`N ≥ 16·(k+2)+1` for `k` hit blocks. Without eagle-family speculation the second
term disappears and `N ≥ 17` suffices.

**Consequence:** for cache-hit tokens there is no activation row at all. Coverage
holes are **data-dependent** — they appear exactly on repeated prefixes, which in
a real deployment means system prompts and few-shot preambles, i.e. the tokens
most likely to be over-represented in traces. A cache built without handling this
is biased, not merely incomplete.

**Mitigations available in source:**

- `--no-enable-prefix-caching` on the serving engine. Simple; costs serving
  throughput on prefix-heavy traffic.
- Per-request opt-out: `skip_reading_prefix_cache: bool | None = None` at
  `V/sampling_params.py:343`. Lets the gateway sample a capture subset without
  penalizing all traffic.

**Detector:** for each captured request, assert
`captured_row_count == prompt_token_count + generated_token_count`, per request,
at write time. Record the shortfall as a first-class metric. A cycle whose
capture coverage is below a configured floor must fail closed the way the empty
dataset and post-filter row floors now do in `_render_speculators_dataset` in
`src/speedlm/training/backends/eagle3.py`, not proceed on partial data.

### 6.2 Pre-norm versus post-norm regression target

The offline extraction path appends the final decoder layer (layer 24 for the
reference model) as a **fourth** aux layer, and aux states are collected inside
the decoder loop — that is, **before** the final norm:
`V/model_executor/models/llama.py:426-428` collects, and `:435` applies
`hidden_states, _ = self.norm(hidden_states, residual)` after the loop ends.

Training then uses the last slice of that stack as the regression target:
`S/src/speculators/train/data.py:385-387`,
`"verifier_last_hidden_states": loaded_hs["hidden_states"][:, -1]` (with
`[:, :-1].flatten(1)` at `:381-383` forming the drafter input).

The obvious serving-side substitute is the runner's `hidden_states` from
`V/v1/worker/gpu_model_runner.py:4364` — but that is the model's **post-norm**
output. Substituting it silently changes the regression target to a different
quantity. Nothing would error; the draft head would simply be trained against the
wrong thing.

**Mitigation:** add the final decoder layer to the serving engine's aux layer
list so it is collected pre-norm, exactly as offline does. **This does NOT change
the drafter's architecture or break warm-start.** Verified in source
(this correction is derived from code reading, not measurement — Stage 0's
elementwise comparison remains the empirical check): the drafter's `fc_input_size`
is computed in the drafter's own `__init__`
(`V/model_executor/models/llama_eagle3.py:176-187`) from the drafter's
`hf_config` (`speculative_config.draft_model_config.hf_config` at `:139`), fixed
at drafter construction time. The set of aux layers collected during serving is
applied to the target model via `set_aux_hidden_state_layers`
(`V/v1/worker/gpu_model_runner.py:5395-5402`) landing on the target's
`EagleModelMixin` (`V/model_executor/models/interfaces.py:1326-1327`), consumed
by `_maybe_add_hidden_state` (`:1329-1339`). These operate on different models at
different times, even though both read the same
`eagle_aux_hidden_state_layer_ids` config key. The offline extraction path
demonstrates this separation in practice: it collects 4 entries (3 aux layers +
final decoder layer) while the drafter's `fc` remains 3-wide, because training
slices `hidden_states[:, :-1]` as drafter input and `[:, -1]` as regression
target. **What collecting the 4th layer DOES require:** the concatenation at
`V/v1/worker/gpu_model_runner.py:5119-5121` must slice the extra entry out before
feeding the drafter, or a 4H tensor hits a 3H `fc` and crashes at runtime. This
is a localized model-runner patch, consistent with Option B (Section 5.3).
Additionally, capturing the final layer's pre-norm output requires adding its
index to the aux collection list — the pre-norm value exists only inside the
decoder loop and is not stored or returned elsewhere; after the loop, the final
RMSNorm is applied and only the post-norm tensor survives.

**Detector — and a correction to what it detects.** The original detector was
"run the same short prompt through the offline extraction path and the serving
capture path and assert the produced tensors match." That test exists
(`tests/e2e/test_serving_activation_capture.py`) and job 369229 passed it with
`mean_rel_error` of exactly `0.0` and `max_abs_diff` of `0.0` on all three
layers at shape `[18, 4096]`.

**That result does not mean what the first draft of this document claimed it
meant.** The two paths are not two derivations of the aux hidden states; they
are two transports of one tensor object. Traced and confirmed against the
pinned checkout:

- Both originate at `V/model_executor/models/interfaces.py:1337`,
  `value = hidden_states + residual`, appended by
  `EagleModelMixin._maybe_add_hidden_state` (`:1329`) and unpacked by the runner
  at `V/v1/worker/gpu_model_runner.py:4364`.
- The serving capture wraps `_model_forward` and takes `result[1]` — the same
  list object — then copies it through `_to_host` and writes it from
  `flush_capture` in `src/speedlm/activation_capture/hook.py`.
- The offline path reads the *identical variable* one branch later at
  `V/v1/worker/gpu_model_runner.py:5035` and copies it: `torch.stack`
  (`V/v1/spec_decode/extract_hidden_states.py:132`) → assignment into a
  same-dtype buffer (`:136`, allocated at `:78-82` from `model_config.dtype`)
  → integer-indexed KV scatter (`V/model_executor/models/extract_hidden_states
  .py:81-90`, preceded by dtype-equality **asserts** at `:220,223` that raise
  rather than cast) → the inverse gather
  (`V/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector
  .py:35-42`) → `torch.empty_like(..., device="cpu", pin_memory=True)` plus a
  same-dtype `copy_` (`:393-396`) → `save_file` (`:320-338`). **No
  floating-point arithmetic and no dtype cast at any step.**
- The offline "draft model" computes nothing: `load_weights` returns `set()`
  (`V/model_executor/models/extract_hidden_states.py:392-394`) and
  `CacheOnlyAttentionImpl.forward` (`:229-231`) is a bare `pass`. It has no
  `nn.Parameter` at all.

So `0.0` is the **expected** outcome, not a surprise, and the 10 % relative
tolerance in `compare.py` was sized for a two-forward comparison this test does
not perform. What the test genuinely detects — and nothing else here does — is
transport failure: slot-mapping errors, layer misordering, truncation, row
misalignment, prefix-cache row loss. Those are real hazards (6.1, 6.3, 6.5) and
this detector is the right one for them. It is simply not evidence about
*which quantity* was captured.

**The identity detector.** The missing independent leg is
`src/speedlm/activation_capture/hf_reference.py`, exercised as Leg 2 of the same
E2E test: the same prompt token ids re-run through **HuggingFace transformers in
float32** — a different implementation, a different dtype, no shared code path.
Each captured aux layer must satisfy two checks:

1. **Tolerance.** `mean_rel_error` within a *derived* bf16-vs-fp32 bound
   `u * (k + 4)` where `u = 2^-8 = 3.906e-3` is bfloat16's unit roundoff (7
   stored significand bits + 1 implicit) and `k` is the aux layer id, i.e. the
   number of times the residual stream has been materialized in bf16. The
   worst-case linear accumulation is used rather than the `sqrt(k)*u`
   random-walk bound because vLLM's reduction order is not statistically
   independent of HuggingFace's. For Qwen3-8B, whose aux layers resolve to
   `[2, 18, 33]`, this is 2.34 %, 8.59 % and 14.45 %, and 15.63 % at the
   appended final layer 36 — tighter than the flat 10 % at shallow depth, and
   one to two orders of magnitude below the O(1) error a wrong quantity
   produces. (An earlier draft quoted `2/12/21` here; those are gpt-oss-20b's
   layers, misattributed. `bf16_relative_tolerance` is always called with the
   resolved layer id, never a positional index, so only the worked example was
   wrong.)
2. **Identification.** The claimed reference index must beat both neighbouring
   depths by at least 3x. This is the check a lossless round-trip cannot fake:
   an off-by-one layer or a post-norm substitution moves the argmin, and it does
   not depend on the tolerance in (1) being correctly derived.

**Of the two, (2) is the sharp one and (1) is a ceiling.** Job 369256 measured
0.511 % / 0.963 % / 1.550 % / 1.772 % against bounds of 2.34 % / 8.59 % /
14.45 % / 15.63 % — 21.8 % / 11.2 % / 10.7 % / 11.3 % of the budget, so capture
fidelity could degrade about ninefold and (1) would still pass. The
discrimination ratios from the same run were 81.46 / 23.14 / 24.03 / 50.02
against the 3.0 margin. Quote those. The budget fraction is recorded per layer
in the artifact as `tolerance_budget_used` so the looseness is visible rather
than implied; the bound is deliberately not tightened to the measurement, which
would fit it to 18 rows of a single prompt. Layer 36's ratio is one-sided — 36
is the last layer, so only neighbour 35 exists — and what covers the missing
side is `final_layer_prenorm_confirmed`, not the ratio.

The errors fit `err(k) ≈ 0.391 % + 0.0363 % · k`. The intercept is within
rounding of one bf16 unit roundoff (`2^-8 = 0.390625 %`), i.e. the
depth-independent cost of materializing the stream in bf16 at all; the slope is
~10.8× shallower than the bound's `u` per layer, because the bound assumes
perfectly correlated per-layer roundoff and real rounding partially cancels.
That is an explanation of the slack, not a proposed replacement for the bound.

The layer-index mapping is proved, not assumed. vLLM calls
`_maybe_add_hidden_state([], 0, ...)` with `residual is None` before the decoder
loop and `(..., idx + 1, ...)` after each layer
(`V/model_executor/models/qwen2.py:415-421`; `Qwen3Model` subclasses
`Qwen2Model` at `qwen3.py:260`), so **aux id `k` is the residual stream entering
decoder layer `k`**, and `self.norm` is applied only to the main return value
afterwards (`qwen2.py:429`). HuggingFace 5.10.4 builds `output_hidden_states` by
prepending the first layer's input and appending each layer's output
(`transformers/utils/output_capturing.py:99-118`), then — because
`tie_last_hidden_states` defaults to `True` — **overwrites the last entry with
the post-norm `last_hidden_state`** (`:264-266`). Therefore `hf.hidden_states[k]`
equals vLLM aux id `k` exactly for `0 <= k <= L-1`, while aux id `L` (the final
decoder layer this design appends) has *no* counterpart in the HF tuple and is
taken instead from a forward hook on `model.layers[L-1]`. Both halves are
asserted at runtime rather than trusted: the tuple length must be `L+1`, the
hook must fire, and the post-norm tensor must be measurably *further* from the
capture than the hooked pre-norm one — which is what makes
`final_layer_prenorm_confirmed` a real answer to 6.2 rather than a restatement
of the question.

Both legs must exist before any training consumes captured data. Leg 1 alone
cannot tell a correct capture from a well-transported wrong one.

**The two Leg 1 sides do not even run the same model runner.** The offline
extraction engine logs `Model Runner V2 does not yet support speculative method
'extract_hidden_states'; using the V1 model runner instead`, while the capture
engine runs V2. So Leg 1's `max_abs_diff = 0.0` is a V2-runner capture agreeing
bit-for-bit with a V1-runner extraction.

This **weakens** the "they are the same tensor, so 0.0 is uninformative"
reading, and it is worth being precise about which direction it cuts. Two
different runners scheduling and slicing independently would not normally be
expected to agree to the last bit; that they do is mild evidence that the
capture is reading the tensor at the same point in the computation the offline
path does, rather than at a shifted one. But it does **not** upgrade Leg 1 into
an identity check, for the reason 6.2 gives: both runners obtain the aux states
from the same `aux_hidden_state_layers` collection inside the same verifier
forward, so a wrong *layer* or a wrong *side of the norm* would be wrong
identically on both sides and still print 0.0. A runner difference perturbs
scheduling, batching and slicing — exactly the things Leg 1 is designed to
catch — not the choice of quantity. Leg 2 remains the only thing that answers
identity, and the conclusion "0.0 carries no information about identity" stands
unchanged.

The Leg 1 verdict is driven by the *aggregate* relative error
(`mean_rel_error = mean|cap-off| / mean|off|`, tolerance 0.10) together with the
shape check and the pre-norm check; cosine similarity is the corroborating
signal.

**Cosine had a ~2e-5 noise floor and must not be read as precise.** Job 369256's
Leg 1 reported cosines of 1.0000027857, 1.0000187356 and 1.0000228824 on tensors
for which `torch.equal` was `True` and every abs/rel error was exactly `0.0`. A
cosine above 1.0 is arithmetically impossible for real vectors: the dot product
and the two norms are three separate reductions with three different summation
orders, and a float32 accumulator over a residual stream (~5e4 elements whose
magnitudes reach 2e4) leaves an O(1e-5) residue. The metric now accumulates in
float64 and is clamped to `[-1, 1]`, so a bit-identical pair reports exactly
`1.0`; artifacts written before that carry the floor and their trailing digits
are noise. Independently of the floor, cosine is a weak statistic here — it is
dominated by the largest elements and stays above 0.99 for tensors that differ
by far more than tolerance — so it corroborates and never gates.

The 0.10 bound is retained deliberately even though 0.0 is what the
transport actually achieves: it is the correct bound if the two transports ever
stop being bit-identical (an upstream cast, a fused-MoE reduction-order change),
and tightening it to 0.0 would turn any benign vLLM change into a red test
without adding information. Note also that `check_pre_norm` on this leg is a
tolerance check on one extra layer, not a structural one — **both** sides
collect at `interfaces.py:1337`, so it can never distinguish pre-norm from
post-norm. Only Leg 2's `final_layer_prenorm_confirmed` can. The elementwise metrics `max_rel_error` and `p99_rel_error` are
diagnostics and do **not** gate PASS/FAIL. They divide by
`max(|off_i|, 1e-3 * RMS(off))`, not by `|off_i| + eps`: a residual stream is
full of near-zero elements, and a bare additive epsilon turns ordinary bf16
noise on those elements into ratios of 1e12-1e14 (which is exactly what Stage 0
artifacts written before 2026-07-31 report). Because the floor is a fraction of
the reference tensor's own RMS, every relative metric is scale-invariant, while
a genuinely diverging element of normal magnitude is still divided by its own
value and still reported at full size.

### 6.3 Rejected draft rows are counterfactual

With speculative decoding, the batch layout allocates `num_draft_tokens + 1`
sampled-token slots per request
(`V/v1/worker/gpu_model_runner.py:2217`, `num_sampled_tokens = num_draft_tokens + 1`).
Acceptance is a contiguous prefix: `rejected` is set once and never cleared in
`V/v1/sample/rejection_sampler.py:743-763`, so evaluation stops at the first
rejection.

The default padded path does not filter the rejected rows out. At
`V/v1/worker/gpu_model_runner.py:5151-5161`, with padding enabled
(`disable_padded_drafter_batch=False`) the code takes a contiguous range —
`target_hidden_states = hidden_states[:total_num_tokens]`, with the comment
noting that "when padding the batch, token_indices is just a range."

**[Precision]** The code never names those rows as junk or sanitizes them; the
retention is implicit in taking a contiguous slice. State the claim as *"the
padded path slices a range and does not filter rejected rows"* rather than
*"it keeps junk rows as padding"* — the former is what the source shows.

Those rows correspond to token positions that were never actually served. Their
activations are conditioned on a draft continuation the verifier rejected, so
they are counterfactual with respect to the served text.

**Mitigation:** compute the rejection count the way vLLM does and drop those rows:
`V/v1/spec_decode/llm_base_proposer.py:1192-1195`,
`num_rejected_tokens = [n + 1 - len(sampled_token_ids[i]) if n > 0 else 0 for i, n in enumerate(num_draft_tokens)]`.

**Complication:** under async scheduling the accept counts are optimistic and
corrected a step later. The deferral machinery is at
`V/v1/worker/gpu_model_runner.py:3701-3727` (`prev_sampled_token_ids`,
`prev_req_id_to_index`), with the optimistic placeholder at `:3726-3727`,
`sampled_ids = [-1] if req_idx not in invalid_req_indices_set else None`. A
capture path that filters using the current step's counts will filter using
provisional data. Either disable async scheduling for capture, or defer the
commit of captured rows by one step to match the correction.

**Detector:** cross-check captured row count per request against the served token
count from the trace store, per request, and assert exact equality. A row-count
mismatch is the observable signature of every variant of this bug. Refuse to
train on a cache with any mismatch.

### 6.4 Prefill/decode numerical identity is not guaranteed

`VLLM_BATCH_INVARIANT` defaults to `False` (`V/envs.py:89`). The reference model
is MoE, and fused-MoE kernel configuration is selected by the token count `M`:
`V/model_executor/layers/fused_moe/fused_moe.py:1382`,
`config = configs[min(configs.keys(), key=lambda x: abs(x - M))]`, and the block
size at `:1285`, `"BLOCK_SIZE_M": 16 if M <= 64 else 64`.

So the same token can produce bitwise-different activations depending on whether
it was seen in a large prefill batch or a small decode step.

**This hazard is not introduced by the proposal — it already applies to the
current offline extraction path,** which re-runs served tokens as prefill and
therefore already produces activations from a different kernel configuration than
serving used. Serving-time capture arguably *reduces* this mismatch by capturing
under the conditions that actually occurred. But it changes its character: a
cache accumulated over heterogeneous batch shapes is internally heterogeneous.

**Mitigation:** measure the magnitude before deciding. If it matters, set
`VLLM_BATCH_INVARIANT=1` for capture and pay the throughput cost — **[unverified]**
what that cost is on this stack; it must be measured, not assumed.

**Detector:** the same-prompt equivalence harness from 6.2, run at several batch
sizes, reporting the elementwise spread. Publish the number rather than asserting
it is small.

### 6.5 CUDA-graph padding

In full CUDA-graph mode the batch is padded, and only `slot_mapping` is
sanitized: `V/v1/worker/gpu_model_runner.py:4034`,
`slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)`, commented
"Needed for reshape_and_cache in full cuda graph mode." No sibling tensor gets
equivalent treatment. Padded activation rows therefore carry stale contents from
whatever occupied the buffer previously — plausible-looking floats, not zeros or
NaNs.

**Mitigation:** always slice to `total_num_scheduled_tokens` before copying, which
is exactly what the runner's own eagle3 path does at
`V/v1/worker/gpu_model_runner.py:5119-5121`. A hook placed at the layer level does
**not** get this for free and must obtain the count separately — this is the
strongest argument for Option B (Section 5.3) over Option A.

**Detector:** assert captured row count equals the step's
`total_num_scheduled_tokens` on every capture, in-process, not as a post-hoc
audit. Additionally, spot-check that no captured row is bitwise identical to a
row captured in a previous step at the same buffer offset — the signature of
stale buffer reuse.

**Related, and separate from padding:** CUDA graphs also broke capture outright
(the aux-layer count is baked into the traced forward, and the compile cache can
replay a graph from before a fix). That is Section 7.2.2, and it is the reason
every correctness result above was obtained in eager mode.

### 6.6 The loss mask is not in the hidden-state file

The loss mask does not travel with hidden states. It lives in the arrow dataset
and is joined by integer row index: `S/src/speculators/train/data.py:388`,
`"loss_mask": self.data[index]["loss_mask"]`, with an alignment guard at `:372-378`
that warns and returns `None` when the loaded token ids do not match
`self.data[index]["input_ids"]`.

In this repository, serving-side capture stores **no token IDs at all** —
`TraceRecord` in `src/speedlm/traces/store.py` has no token-id field. The mask
is instead re-derived by shelling out to Speculators'
`prepare_data.py` (`SpeculatorsTrainingRowRenderer.render_rows` in
`src/speedlm/training/backends/eagle3.py`), with validation via
`--require-nonzero-loss-mask` and offender localization in `_zero_row` (raising
`FinalAssistantMaskError`).

This repository does retain pieces of an alternative provenance-based path:
`training_row_from_trace` and `_generated_assistant_spans` in
`src/speedlm/training/rows.py`, plus `loss_mask_from_offsets` in
`src/speedlm/training/masking.py`. It does **not** retain a complete path that
tokenizes a row and combines those pieces into a prepared row. The former
`prepare_training_row` function was deleted in commit `110c528`, and none of
these helpers is the production renderer. Production still uses
`SpeculatorsTrainingRowRenderer.render_rows` in
`src/speedlm/training/backends/eagle3.py` and shells out to `prepare_data.py`.

**Consequence:** capturing activations does not by itself let us drop
`prepare_data.py`. Token IDs must be captured alongside activations, or the
provenance-based preparation path must be rebuilt and wired into production
before the pipeline is genuinely
simplified. If neither happens, we save the `EXTRACTING` engine but keep the
subprocess.

**Detector:** the alignment guard already exists upstream
(`S/src/speculators/train/data.py:372-378`) but only *warns* and returns `None` —
which is precisely the silent-empty-data shape of bug #1 in Section 6.0. Any
integration must convert that warning into a hard failure on our side, and must
assert `len(loss_mask) == captured_row_count` per row before training starts.

### 6.7 Prospective tokenization skew — the two legs do not share a renderer

**Status: prospective, not a current production-training cause.** The renderer
difference below is measured, but today's EAGLE-3 training flow does not join
serving-rendered activations to separately offline-rendered rows. It therefore
does not currently introduce tokenization skew into the training alignment.

Measured 2026-07-31 on the Stage 0 four-prompt matrix. The `template_divergent`
case failed on `openai/gpt-oss-20b` under **both** runner generations, identically
(`capture-gptoss-v1` / `-v2`), and passed on `Qwen/Qwen3-8B` under both:
`cannot align layer 2: offline has fewer rows (124) than the prompt (125)`.

The prompt body contained `<|endoftext|>` as ordinary user text.

* `<|endoftext|>` is a real special token of gpt-oss's `o200k_harmony` vocabulary
  (id `199999`). `<|im_start|>` / `<|im_end|>` are not, and tokenize as six
  ordinary pieces each on both models.
* The **serving** leg does not use the HF chat template for gpt-oss. vLLM renders
  Harmony models with the `openai_harmony` encoder
  (`vllm/entrypoints/openai/parser/harmony_utils.py:449`, `render_for_completion`),
  which encodes message content with no special tokens allowed. The user's
  `<|endoftext|>` stays literal text and costs seven tokens — **125** prompt
  tokens, which is what `usage.prompt_tokens` reported.
* The **offline** leg renders through Speculators' `prepare_data.py`
  (driven from `extract` in
  `src/speedlm/activation_capture/offline_extract.py`),
  which reaches `_get_input_ids_loss_mask`
  (`S/src/speculators/data_generation/preprocessing.py:515-624`) and applies the
  **HF** chat template. HF's tokenizer parses special tokens out of ordinary
  text, so the same `<|endoftext|>` collapses to the single id `199999`: 120
  tokens for the user turn, plus the four the conversation renderer appends for
  the assistant turn (`<|channel|>final<|message|><|return|>`) — **124** rows.

The branch is runtime-detectable. In the pinned vLLM 0.25.1 environment,
`V/entrypoints/openai/chat_completion/serving.py:152` passes
`is_harmony=self.model_config.hf_config.model_type == "gpt_oss"` to the parser
manager; Responses serving uses the same model-type predicate. It is not a
model-name allowlist.

Both counts are correct for the renderer that produced them. They can only
disagree when user content contains a string that *is* a control token, so this
is a template-injection input rather than a capture defect, and the assertion
that caught it (`compare.align_prompt_rows`) fired correctly. Harmony's behaviour
is the defensible one — user content must never contribute control tokens — so
the legs are not made to agree on such an input. The e2e case was narrowed to
markup-*shaped* text that is special in neither vocabulary, and the original body
was retained as `control_token_injection`, selectable with
`SPEEDLM_E2E_PROMPT_SET=injection`, with the expected failure documented on it
(`tests/e2e/test_serving_activation_capture.py`). A pre-flight guard,
`_assert_prompts_free_of_control_tokens`, now checks every standard prompt body
against the verifier's own vocabulary before an engine is started. This is a
description of the harness code path, not a claim that a definitive tuning run
observed the guard firing.

**Downstream, in production.** The eagle3 flow writes captured trace text to a
`conversations` JSONL (`_speculators_record` in
`src/speedlm/training/backends/eagle3.py`) and hands it to
`prepare_data.py`. That flow re-derives both the hidden states and the training
rows from the *same* prepared dataset, so it is self-consistent: this divergence
does **not** misalign production training rows today. It would begin to matter
the moment serving-time captured activations are joined to offline-rendered rows,
which is exactly what Section 6.6's simplification proposes.

**Security note, separate from the alignment question.** On a model whose serving
path *is* the HF chat template — Qwen3-8B here — a user who writes
`<|im_start|>system<|im_end|>` verbatim gets real ChatML control tokens
(`151644` / `151645`) into the served prompt, and into any training row rendered
from that trace. Qwen passing this case is not the injection failing to land; it
is the injection landing identically on both legs. gpt-oss is protected only
incidentally, by Harmony's `allowed_special` policy. Nothing in this repository
strips or escapes control tokens from user content on the way in
(`src/speedlm/traces/store.py` stores message text verbatim), so redaction and
trace capture inherit the same exposure. Recorded here as an observation; no
mitigation is proposed in this document.

### 6.8 Hazard summary

| # | Hazard | Severity | Mitigation | Detector |
| --- | --- | --- | --- | --- |
| 6.1 | Prefix-cache coverage holes — **confirmed empirically**, job 369256: 144 hit tokens, 144 prompt rows lost (6.1.1) | **High** (data-dependent, biased) | `--no-enable-prefix-caching` or per-request `skip_reading_prefix_cache` | Per-request row count == prompt + generated; fail closed below a floor |
| 6.2 | Pre-norm vs post-norm target | **High** (wrong quantity, invisible) | Collect the final layer as a 4th aux layer; slice it before drafter `fc` (localized model-runner patch) — does NOT change drafter architecture | Offline-vs-serving elementwise equivalence test |
| 6.3 | Rejected draft rows | Medium | Filter by `num_rejected_tokens`; handle async deferral | Captured rows == served tokens, per request, exact |
| 6.4 | Prefill/decode numerics | Medium (pre-existing) | Measure first; `VLLM_BATCH_INVARIANT=1` if warranted | Equivalence harness across batch sizes; publish the spread |
| 6.5 | CUDA-graph padding | Medium (stale, plausible values) | Slice to `total_num_scheduled_tokens` | Row count == scheduled tokens, in-process; stale-row spot check |
| 6.6 | Loss mask provenance | Medium (blocks the simplification) | Capture token IDs, or promote `rows.py` path | Hard-fail the alignment guard; `len(mask) == rows` |
| 6.7 | Prospective tokenization skew when serving-rendered activations are joined to offline-rendered rows; renderer difference **confirmed empirically**, gpt-oss-20b, both runners | None in today's self-consistent EAGLE-3 preparation; **High** if the Section 6.6 join lands | Preserve token IDs with captures, or make the join use one renderer; strip or escape control tokens if the simplification lands | Runtime predicate `hf_config.model_type == "gpt_oss"`; `_assert_prompts_free_of_control_tokens`; `align_prompt_rows` |

---

## 7. Cost and benefit

### 7.1 What is saved

- The `EXTRACTING` stage: 199.6 s on the reference cycle, ~18% of its 1134.8 s
  wall clock.
- One full vLLM engine lifecycle per cycle (process start, verifier weight load,
  teardown) and its GPU memory churn.
- The memory precondition that currently gates the cycle (Section 8) — which, on
  a corpus where 16 of 22 cycles aborted out of `EXTRACTING`, may matter more
  than the seconds. **[unverified: abort causes not individually attributed.]**

### 7.2 What it costs

**Bandwidth.** With hidden size 2880 and bf16, per token per layer is 5.625 KiB.
At 3 aux layers that is 16.875 KiB/token; at 4 (Section 6.2) it is 22.5 KiB/token.

| Serving rate | 3 layers | 4 layers |
| --- | --- | --- |
| ~77 tok/s (historical planning assumption; not a valid tuned-head benchmark) | ~1.3 MB/s | ~1.7 MB/s |
| 2000 tok/s (hypothetical high load) | ~34 MB/s | ~45 MB/s |

Against PCIe Gen4 x16 (~32 GB/s theoretical), the high-load case is roughly 0.14%
of link bandwidth. **[unverified: the actual host link generation and width on the
target machine has not been read from `lspci`; the ratio assumes Gen4 x16.]**

**Latency. Measured on GPU — see 7.2.1.** The earlier statement that "TTFT and
tokens per second have never been measured on a GPU with capture enabled" is
now stale. They have been, twice: once with the blocking copy, which cost
**-14.49% throughput** under CUDA graphs, and once after deferring the transfer
synchronization, which reduced the throughput and inter-token cost to
statistically zero. The remaining cost is on TTFT.

The in-process transfer-count instrumentation still exists and is still only a
transfer-count measurement, not a serving benchmark: the hook stacks the four
uniform aux tensors on the device and issues one device-to-host transfer per
forward pass instead of four; the non-uniform fallback still makes four
(`_to_host` in `src/speedlm/activation_capture/hook.py`;
`TestHotPathTransferCount` in `tests/test_activation_capture_hook.py`). Do not
cite it as a latency result.

The connector pattern proposed for a production implementation uses a dedicated
copy stream and `non_blocking=True` into pinned memory
(`example_hidden_states_connector.py:351-431`), with completion detected by
event polling (`:570-575`). The shipped Stage 0 hook now follows the same shape
(commit `5c492ab`), and 7.2.1 is its measurement.

### 7.2.1 Measured serving overhead of capture

**Harness.** `tests/e2e/test_capture_overhead.py`, paired and ABBA-interleaved
on **one engine with no restart between arms**, so an engine-lifecycle
difference cannot be mistaken for a capture effect — the failure mode that
voided `badba71-gptoss-idle` (see [Benchmark evidence](benchmark-evidence.md)).
Qwen3-8B verifier with the stock `RedHatAI/Qwen3-8B-speculator.eagle3` drafter
on an H100, 3 prompts x 20 repetitions per condition = 60 paired samples, 128
completion tokens per request, prefix caching off, 10 cycles, 20 armed and 20
disarmed capture blocks.

**Before — blocking copy, CUDA graphs (the production default).** Job 371003 at
commit `c5aa14c`:

`/data/ryan.kim/speedlm-runs/capture-overhead-cachekey-c5aa14c/results/capture-overhead-20260806T212332Z/capture_overhead.json`

| Metric | OFF baseline | Capture-ON delta | SE | Delta % |
| --- | ---: | ---: | ---: | ---: |
| TTFT | 19.549 ms | +10.069 ms | 0.308 | **+51.5%** |
| Throughput | 283.94 tok/s | -41.14 tok/s | 0.89 | **-14.49%** |
| Median inter-token | 8.378 ms | +1.3881 ms | 0.034 | **+16.57%** |
| End-to-end | 0.4522 s | +0.0771 s | 0.0019 | **+17.04%** |

Every delta is 32-46 standard errors from zero. This is a material serving
regression, not noise.

**After — deferred synchronization, same regime.** Job 371031 at commit
`5c492ab`:

`/data/ryan.kim/speedlm-runs/capture-overhead-deferred-sync-5c492ab/results/capture-overhead-20260806T234924Z/capture_overhead.json`

| Metric | OFF baseline | Capture-ON delta | SE | Delta / SE | Delta % |
| --- | ---: | ---: | ---: | ---: | ---: |
| TTFT | 17.717 ms | +4.401 ms | 0.275 | 16.0 | +24.84% |
| Throughput | 280.11 tok/s | +0.403 tok/s | 0.173 | 2.3 | +0.14% |
| Median inter-token | 8.390 ms | -0.00028 ms | 0.0024 | -0.11 | -0.003% |
| End-to-end | 0.4581 s | +0.000637 s | 0.00026 | 2.5 | +0.14% |

The steady-state cost is gone. Inter-token overhead is statistically
indistinguishable from zero (-0.11 SE). Throughput and end-to-end sit at
+0.14%, about 2.3-2.5 SE — small enough that the sign is not worth defending;
what matters is that the -14.49% is not there. **TTFT is the one remaining
cost: +4.40 ms at 16.0 SE, unambiguously real.** It is the prefill step, where
the first transfer is issued and there is no prior step to have hidden it
behind.

**Eager, for contrast.** Job 370816
(`/data/ryan.kim/speedlm-runs/capture-overhead-snap-20260806T031659Z/results/capture-overhead-20260806T032133Z/`)
measured the blocking copy under `--enforce-eager`: TTFT +1.106 ms on a
46.956 ms baseline (+2.36%), throughput -2.951 tok/s on 116.71 (-2.53%). The
graphed regime was ~9x worse in absolute TTFT on a *faster* baseline — the cost
grew, it was not merely a bigger fraction of a cheaper step.

**Mechanism.** A blocking device-to-host copy cannot be folded into a replayed
CUDA graph. It therefore stalls the serving thread on the compute stream every
step — a cost eager never had to pay, because eager was already synchronizing
per layer. The fix is not to copy less: it is to copy asynchronously on a
dedicated stream into a pooled pinned buffer, record an event, and **defer the
synchronization to the points that actually read captured state** (reading
pending rows, flush, reset, arm, disarm). The hot path drains without waiting.

**The speedup is not dropped rows.** This was checked before the number was
believed, because "make it fast by capturing less" is the obvious way this
result would be fake:

| Check | Before (371003) | After (371031) |
| --- | --- | --- |
| Bytes per armed block | 67,633,488 | **68,813,136** (larger) |
| Rows per layer per block | 2,064 | **2,100** (more) |
| Aux layers present | `layer_2/18/33/36`, all 20 blocks | same, all 20 blocks |
| Armed blocks non-empty | 20 / 20 | 20 / 20 |
| Disarmed flushes that raised | — | 20 / 20, `RuntimeError: capture is not active` |
| Completion-token asymmetry | 0.0% | 0.0% |

The after-run captured *more* data, in the same four layers, in every block, in
less time. The disarmed-flush count is the other half of the check: capture
being genuinely off when disarmed is what makes the OFF arm a valid baseline.

**Not measured: p50/p99 percentiles.** The harness reports paired means with
standard errors over 60 samples, which is the right statistic for a paired
A/B but is not the tail. The per-sample values are retained in
`samples.on` / `samples.off` in both JSONs, so percentiles are derivable, but
nobody has published them and no tail claim should be made.

### 7.2.2 Capture was non-functional under CUDA graphs until this work

Worth recording, because the eager-only measurements above were for a while the
*only* ones that could exist. Capture worked in eager mode and was
**dead on a graph-capturing engine — the production default** — and fixing it
took two distinct changes, the second of which only surfaced because the first
appeared to work:

1. **Declare the aux-layer set before compilation** (commit `aa02ab8`). Arming
   extended the model's aux-layer tuple from three to four on the live model,
   but vLLM compiles and graph-captures the forward once at startup and never
   re-traces. The attribute said four, the graph emitted three, the labelling
   guard refused, and EngineCore died — reproduced on GPU as job 370798, dead in
   three minutes. The fix separates *declaration* from *buffering*: the full
   aux-layer set is declared before compilation so the graph bakes in the final
   count, and arming toggles only whether states are buffered. Arming then
   changes no shape and no graph, so eager and graphed engines capture
   identically.
2. **Fold the declared layers into the vLLM compile-cache key** (commit
   `c5aa14c`). Fix 1 was defeated in job 370927 by a *cached* compiled graph
   written by job 370798 — the pre-fix run. vLLM never recompiled; it replayed a
   three-aux graph. The cache key hashes only the **config-declared** aux layer
   ids, and ours was appended imperatively at load time, so it was invisible to
   the key. Declaration now also records the declared tuple into the config's
   `additional_config`, which is folded into both the backend and AOT keys, so a
   four-aux engine gets its own cache namespace.

The secondary finding is the more instructive one: with the graph stale, the
whole of fix 1 was a **silent no-op**. Disarmed blocks and every warmup ran the
old three-aux graph without complaint, because the layer strip only truncates
when the list is longer than expected. Nothing surfaced until arming. A stale
graph is now detected on the first forward, armed or not, and the e2e harness
gained a preflight after warmup and before any measured block, so this class of
failure fails in about ninety seconds with an exact message instead of killing
the engine mid-run. This is the same defect class Section 6.0 is about: the
check that cannot fail, and the fix that cannot be observed to have applied.

**Maintenance.** An owned dependency on vLLM internals, re-validated on every
upgrade (Section 5.5).

**Correctness engineering.** Seven detectors (Section 6.8). This is real work and it
is not optional.

### 7.3 Storage and retention must be designed, not inherited

At 22.5 KiB/token, the current trace buffer default of
`max_tokens: int = 8_000_000` (`TraceBufferConfig.max_tokens` in
`src/speedlm/config.py`) would imply about
**184 GB** of activations. The current cycle trains on roughly 1500 tokens, or
~34 MB — five orders of magnitude less.

Do not inherit the trace retention policy. The activation cache needs its own:

- **Cap by bytes, not tokens.** Tokens are the wrong unit when a token costs
  22.5 KiB.
- **Size it from what training consumes**, plus headroom for the held-out split
  and a margin for growth — not from what serving produces.
- **Evict on a policy that preserves distribution**, not purely by recency. Naive
  FIFO on a bursty deployment yields a cache of whatever happened in the last
  hour. Reservoir sampling over the serving stream is the obvious alternative and
  is **[unverified as a design choice — nobody has evaluated it here]**.
- **Retention must be observable.** The repo already treats retention passes as
  events worth recording (commits `628ce83`, `ebe258a`); the activation cache
  should meet that bar from day one.
- **Store on the same volume policy as `exchanges/`.** Activations derived from
  user traffic are user data. `exchanges/` is already private-0700 because raw
  bodies can carry credentials (README, **How it is structured**). The
  activation cache inherits that sensitivity class — an activation is a lossy
  but real function of the prompt. Treat it as such by default.

**Detector:** an assertion that the cache's on-disk size stays under its
configured cap, checked on every write, with the retention pass recorded as an
event.

---

## 8. What this does and does not do for low-VRAM deployment

### 8.1 Training does not need the verifier on GPU

Verified: Speculators' training entry point touches the verifier only through
`AutoConfig.from_pretrained` (`S/scripts/train.py:139`, `:279`, `:304`, and
indirectly at `:252` via
`get_target_vocab_size(None, args.verifier_name_or_path)`), and loads only draft
weights: `:345-347`,
`draft_model = model_class.from_pretrained(args.from_pretrained, t2d=t2d, d2t=d2t)`.
There is no `AutoModel` load of the verifier.

So the verifier is needed on GPU **only** to produce hidden states. Capture
during serving removes that need entirely.

### 8.2 What that changes

Today the cycle requires that the verifier fit in GPU memory **twice,
sequentially** — the serving engine must fully release before the extraction
engine can allocate. That is enforced as a precondition:
`create_production_tuner` in `src/speedlm/tuner/composition.py` passes
`required_fraction=pipeline.gpu_memory_utilization` (default 0.80 in
`SpeculatorsPipelineConfig`), and `GPUMemoryPrecondition` in
`src/speedlm/gateway/control.py` computes the required bytes from device-total
memory and compares that value against free memory.

**Correction to the common summary.** This is *"0.80 of device **total** must be
**free**"*, not "0.80 of free memory." The phrase "0.80 × device-total free" is
ambiguous to the point of being wrong; state it the long way.

Serving-time capture removes this precondition. "Fits twice, sequentially"
becomes "fits once."

### 8.3 What it does not change

**The serving engine remains the binding constraint, and it is large.** The
reference verifier, gpt-oss-20b, is roughly 13.56 GiB of weights
**[unverified — this figure was not confirmed against a checkpoint on disk during
this review; verify before quoting]**. On a 16 GB card that leaves no usable KV
cache budget, so a 16 GB deployment remains infeasible whether or not this
proposal lands. Capture does not shrink the model.

**The real benefit is on H100-class hardware**, where it converts a
memory-precondition-gated cycle into an unconditional one and removes a ~200 s
stage.

### 8.4 An important framing correction

**This repository documents no consumer-GPU or VRAM target anywhere.** A search
across code, docs, and config returns zero hits for RTX / GeForce / 4090 / 3090 /
5090 / "consumer" / "gaming" / 24GB. The documented targeting is the opposite:
README **Project status** describes the live E2E test as an **H100** test, `docs/DEMO.md:346`
places runs on an H100 in a SLURM partition, and `docs/DEMO.md:321` explicitly rules
out laptop-local execution. README **GPU runtime stack** pins the GPU host stack
but names no GPU model or VRAM size. Every one of the 25 GPU-identifying records in
`log_artifacts/` is an NVIDIA H100 80GB HBM3.

So "this unlocks low-VRAM deployment" is not a claim this proposal should make.
There is no low-VRAM target to unlock. If one is desired, that is a separate
product decision that this work would *help* but nowhere near *achieve*.

---

## 9. Implementation plan

Each stage has an exit criterion. **A stage that does not meet its exit criterion
kills or reshapes the plan; it does not get waived.**

### Stage 0 — De-risking prototype (do this first, alone)

**The smallest experiment that would prove or kill the approach.**

Attach a capture mechanism to a running EAGLE-3 serving engine, serve one fixed
short prompt, and write the aux hidden states to a file. Separately, run the
existing offline extraction path over the same prompt. **Compare the two tensors
elementwise.** Then re-derive the same activations independently, with
HuggingFace transformers in float32, and compare against that.

Nothing else. No cache, no retention, no training, no integration.

**What Stage 0 proves, and what it does not.** This split is load-bearing and
the first version of this document got it wrong, so it is stated explicitly.

| Question | Settled by | Status |
| --- | --- | --- |
| Can a mechanism reach `aux_hidden_states` in a serving engine? | The capture running at all | **Yes** — `worker_extension_cls` + `_model_forward` monkeypatch (5.2) |
| Does the capture survive transport to disk intact — right rows, right layers, right order, no truncation? | Leg 1 (capture vs. offline) | **Yes** — job 369229, `max_abs_diff = 0.0` on all three layers. Note the two sides run *different model runners* (capture on V2, offline extraction falls back to V1 for `extract_hidden_states`), which makes the bit-identity a slightly stronger transport result than "same tensor twice" — but still says nothing about identity (see 6.2). |
| Are the captured values the **same quantity** the trainer expects (pre-norm, layer `k`, not `k±1`)? | Leg 2 (capture vs. HF fp32) | **Met — job 369256**, the first run in which the reference leg actually executed. The fp32 reference ran on `cuda`; every claimed layer was the **strict argmin** over `{k−1, k, k+1}`, beating its nearest neighbour by 81.46× / 23.14× / 24.03× / 50.02× at layers 2 / 18 / 33 / 36 against a 3.0× required margin. Per-layer `mean_rel_error` 0.511 % / 0.963 % / 1.550 % / 1.772 %, all inside the derived 2.34 % / 8.59 % / 14.45 % / 15.63 %. Note Leg 1 still says nothing here: a `0.0` there is expected and carries no information about identity. See the limitations below before quoting this. |
| Does a prefix-cache hit genuinely drop rows (6.1)? | `test_prefix_cache_coverage` | **Demonstrated — job 369256.** `vllm:prefix_cache_hits_total` 0 → 144 on the repeat, captured rows/layer 213 cold → 69 warm, difference exactly 144, matching a hit count *derived* from vLLM's block arithmetic and asserted as an equality. This no longer rests on hardcoded literals (the earlier `cache_hit: true` / `rows_missing: 0` retirement was withdrawn as fabricated) nor on the null first run (job 369236, zero hits — correct, an 18-token prompt cannot fill a hittable block under eagle3, see 6.1.1). The artifact's own `rows_missing: 96` is wrong by 48 and is superseded by `prompt_rows_missing: 144`. |
| What is the real serving latency cost of capture? | Paired capture-on vs. capture-off on one engine, both execution modes | **Measured (7.2.1).** Under CUDA graphs, with the blocking copy: -14.49% throughput, +10.07 ms TTFT (job 371003). After deferring the transfer sync: throughput and inter-token overhead statistically zero, **+4.40 ms TTFT remaining** (job 371031). Verified not to come from dropped rows. p50/p99 tails still unpublished. |
| Does capture work at all on a graph-capturing engine (the production default)? | Running an armed engine under CUDA graphs | **Yes, since `c5aa14c`** — it did not before, and needed two fixes (7.2.2). |

Do not cite "Stage 0 passed" as settling the correctness question. Cite which
leg, and what that leg can see.

**Limitations of job 369256 — read these before quoting the numbers above.**
This is **one model, one prompt, 18 prompt rows**. Specifically:

- **Sample size.** The identity leg compared 18 rows of a single prompt on a
  single verifier/drafter pair. It is a strong result on that sample and says
  nothing about prompt-length, batch-composition or model dependence.
- **Layer 36's discrimination is one-sided.** 36 is the last decoder layer, so
  the only neighbour available to beat is 35 — there is no `k+1` side. Its
  50.02× ratio is therefore a weaker structural claim than layer 18's, which was
  bracketed on both sides. What covers the missing side is a *different* check:
  `final_layer_prenorm_confirmed`, which shows the post-norm tensor differs from
  the captured pre-norm one by far more than tolerance.
- **The tolerance gate is ~9× loose.** The derived bound is a worst case that
  assumes every per-layer bf16 rounding aligns. The measurements used 21.8 % /
  11.2 % / 10.7 % / 11.3 % of their budget, so capture fidelity could degrade
  roughly ninefold and `within_tolerance` would still read `true`. It is a
  sanity floor. **The discrimination ratio is the sharp instrument** — that is
  the number to cite. The bound is deliberately *not* tightened to the measured
  values: fitting it to 18 rows of one prompt would flake on any other model,
  prompt or vLLM build. The slack is instead recorded per layer as
  `tolerance_budget_used`.
- **Cosine similarity carries a noise floor.** Job 369256's transport leg
  reported cosines of 1.0000027857 / 1.0000187356 / 1.0000228824 on tensors that
  were bitwise identical — arithmetically impossible, and a ~2e-5 float32
  accumulator artefact. The metric now accumulates in float64 and is clamped to
  `[-1, 1]`, so a bit-identical pair reports exactly `1.0`; artifacts written
  before that fix carry the floor. Cosine is corroborating only in either case:
  it is dominated by the largest elements and stays above 0.99 for tensors that
  differ by far more than tolerance.

**Error versus depth.** The four measured errors fit
`err(k) ≈ 0.391 % + 0.0363 % · k` almost exactly. Two things are worth noting,
and neither is a tuning knob:

- The **intercept, 0.391 %, is one bf16 unit roundoff** (`2^-8 = 0.390625 %`) —
  the depth-independent floor you would expect from materializing the residual
  stream in bf16 at all, before any layer-to-layer accumulation.
- The **measured slope is ~10.8× shallower** than the bound's slope
  (`u = 0.391 %` per layer). That is the expected sign and roughly the expected
  size: the bound assumes perfectly correlated per-layer roundoff (worst-case
  linear accumulation), while real rounding partially cancels. It is an
  observation about why the gate is loose, **not** a licence to replace the
  bound with the fit — a fitted slope from one sample would encode this model's
  kernel selection as if it were a property of bf16.

This single experiment settles the questions that determine whether the rest of
the plan is worth writing:

- Can a mechanism reach `aux_hidden_states` at all in a serving engine
  (Section 5)?
- Are the captured values the *same quantity* the trainer expects, or does the
  pre-norm/post-norm split (Section 6.2) make them silently different?
- Does the prefix-cache inference in Section 6.1 hold — i.e., is a row genuinely
  missing for a cache-hit token?
- What is the actual, measured cost in serving latency (Section 7.2)?

**Exit criteria:**
- Captured and offline tensors match to a stated, written-down tolerance, or the
  discrepancy is fully explained (and the explanation is not "it's probably
  fine"). **Met — but see the table above for what it establishes.** Because
  the two paths share the tensor, "match" here is bit-identity and the exit
  criterion is weaker than it reads.
- Captured tensors match an **independent** float32 re-derivation to a derived
  bf16 tolerance, *and* are the strict nearest match among neighbouring
  residual-stream depths. This criterion was absent from the original plan;
  without it the previous criterion is satisfiable by any wrong quantity that
  is transported correctly. **Met — job 369256**, the first run in which the
  reference leg executed at all. Every layer was the strict argmin over its
  neighbours by 81.46× / 23.14× / 24.03× / 50.02× against a 3.0× required
  margin, and every `mean_rel_error` was inside its derived bound. Cite the
  discrimination ratios rather than `within_tolerance`, and cite the
  limitations above alongside them: 18 rows, one prompt, one model, and layer
  36's discrimination is one-sided.
- A cache-hit prompt demonstrably yields fewer captured rows than tokens,
  confirming 6.1 empirically. **Met — job 369256.**
  `vllm:prefix_cache_hits_total` 0 → 144 on the repeat and 213 → 69 captured
  rows per layer, a shortfall of exactly the 144 hit tokens, against an
  expectation derived from vLLM's block arithmetic and asserted as an equality
  rather than as `> 0`. The shortfall is measured cold-minus-warm; the
  artifact's `rows_missing: 96` conflated 48 decode rows into a prompt-row count
  and is superseded by `prompt_rows_missing: 144` (6.1.1).
- Measured serving latency with capture on versus off, published.
  **Met in substance, with one gap.** Published in 7.2.1: paired,
  ABBA-interleaved, one engine, 60 pairs, in both execution modes, before and
  after the deferred-sync fix. The gap is that the published statistic is the
  paired **mean** with a standard error, not p50/p99 — the per-sample values are
  retained in the artifacts but the tails have not been reported, so no tail
  claim is licensed.
- **New, and it was not in the original plan:** capture must actually function
  under CUDA graphs, not only under `--enforce-eager`. It did not until
  `c5aa14c` (7.2.2). Any future criterion that can be satisfied in eager alone
  should be assumed not to hold in the production regime until it is run there.

**Kill condition:** if the tensors cannot be made to match, or if matching
requires a model-runner patch that cannot be isolated from the drafter's forward
path (e.g., the serving engine fundamentally cannot produce the pre-norm
activation the trainer expects), stop and re-scope. The correct outcome of Stage
0 may be "do not build this." Collecting the final layer as a 4th aux layer
does NOT break warm-start (see Section 6.2 — corrected).

### Stage 1 — Correct capture of one request

Extend the prototype to attribute rows to `(request_id, position)` using
`req_indices` / `query_start_loc` / `req_ids` (Section 4.8), filter rejected
draft rows (6.3), and slice CUDA-graph padding (6.5).

**Exit:** for a mixed workload of ≥100 requests including multi-turn and
speculative decoding, captured row count equals served token count **exactly**,
per request, with zero mismatches. Not "almost all" — zero.

### Stage 2 — Cache with retention

Add the on-disk activation cache, byte-capped, with an eviction policy and
retention events (Section 7.3).

**Exit:** cache size stays under cap across a sustained serving run; retention
passes are recorded; a restart resumes without loss or duplication.

### Stage 3 — Train from cache

Wire the EAGLE-3 backend to read the cache instead of invoking extraction. Resolve
the loss-mask join (6.6).

**Exit:** a training run from cached activations produces a candidate whose gate
metrics are statistically indistinguishable from a candidate trained via the
current extraction path on the same traces. Same-or-better, measured against the
existing thresholds (README, **Promotion thresholds**).

### Stage 4 — Delete the extraction stage

Remove `EXTRACTING` from the cycle and drop the memory precondition.

**Exit:** end-to-end idle-tuning cycles complete without an extraction engine;
cycle wall clock drops by approximately the measured `EXTRACTING` duration; the
live H100 E2E test passes.

### Stage 5 — Reduce the maintenance surface

Pursue Option C (Section 5.4) upstream. Independently, pin the vLLM-internals
contract with a version-check test that fails loudly on upgrade rather than
silently capturing garbage.

**Exit:** upgrading vLLM produces a *loud* failure when the internals contract
changes.

---

## 10. Open questions — what must be measured, not assumed

1. **What is the real serving latency cost of capture?** **Answered, and design
   intent was wrong before it was right.** Measured on GPU (7.2.1): under CUDA
   graphs the original blocking copy cost **-14.49% throughput and +10.07 ms
   TTFT** — nowhere near zero. Deferring the transfer synchronization removed the
   steady-state cost entirely (throughput +0.14%, inter-token -0.11 SE, both
   indistinguishable from zero) and left **+4.40 ms TTFT** at 16.0 SE. What
   remains open is narrower: the p50/p99 tails (means only were published), and
   whether the residual TTFT can be pushed lower. Note also that "near zero"
   would have been reported as true had this only ever been measured in eager
   mode, where the same blocking copy cost -2.5%.
2. **Do captured and offline activations actually match?** **Answered, and the
   question was the wrong one.** They match bit-for-bit (job 369229), because
   they are the same tensor transported two ways with no arithmetic on either
   path (6.2). The question that actually matters — *is the captured tensor the
   quantity the trainer expects* — is answered by the independent HuggingFace
   fp32 leg, not by this comparison. Do not let a `0.0` in `result.json` retire
   this line item. That leg has now run: job 369256 identified every layer by
   81.46×/23.14×/24.03×/50.02× over its neighbours (Stage 0 table, and the
   limitations directly under it).
3. **Does the prefix-cache absence inference hold?** **Answered — yes,
   measured.** Section 6.1 was inference from what crosses an API boundary; job
   369256 replaced it with a measurement: `vllm:prefix_cache_hits_total` 0 → 144
   and 213 → 69 captured rows per layer, a 144-row shortfall matching a derived
   expectation exactly (6.1.1). Two earlier states of this line item are
   superseded but worth remembering: before 2026-08-01 `test_prefix_cache_coverage`
   asserted nothing and wrote its "findings" as literals, and 6.1 was retired on
   those literals — any citation of *that* was citing a fabricated value; then
   the first honest run (job 369236) got zero hits because the prompt was too
   short to fill a hittable block under eagle3. The confirmation comes from
   neither of those, but from a run with a prompt long enough for the hazard to
   be representable and an exact assertion on the hit size.
4. **What did the 16 aborted cycles actually fail on?** The claim that capture
   would have prevented them is currently unsupported. Read the failure records.
5. **What is the throughput cost of `VLLM_BATCH_INVARIANT=1` on this stack?**
   Unmeasured (6.4).
6. **How large does the cache need to be for training quality to plateau?** The
   current cycle trains on ~1500 tokens. Nobody has measured whether more helps,
   and the retention design depends on the answer (7.3).
7. **Does adding the final decoder layer as a 4th aux layer break warm-start from
   the public checkpoint?** Code reading says no — the drafter's `fc` input width
   is fixed from the drafter's own config at construction time
   (`llama_eagle3.py:176-187`, `:139`), independent of how many aux layers the
   serving engine collects. The binding constraint is a localized model-runner
   patch that slices the 4th entry before feeding the drafter. This correction is
   derived from code reading, not measurement. The empirical check is that the
   engine serves at all with the 4th layer collected (the drafter's `fc` would
   raise a shape error otherwise) plus Leg 2's `final_layer_prenorm_confirmed`;
   Leg 1's elementwise comparison cannot check it, since both sides collect the
   4th layer at the same place.
8. **What is the actual size of the reference verifier's weights, and what is the
   host's PCIe link generation and width?** Both are quoted in Section 7 and
   Section 8 from memory rather than from the machine.
9. **Is there a supported vLLM extension point we missed?** Section 5 asserts
   "we found none," which is weaker than "none exists." Worth one careful pass
   through the upstream plugin and connector documentation before committing to a
   private-API dependency.
10. **What is the sensitivity classification of stored activations?** Section 7.3
    argues they inherit `exchanges/`'s sensitivity. That is a judgment call, not a
    verified policy, and it should be confirmed by whoever owns the data-handling
    posture.
