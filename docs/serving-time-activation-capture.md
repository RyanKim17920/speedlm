# Serving-time activation capture

**Status:** design proposal. Nothing here is implemented. No code change accompanies
this document.

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
bounded trace store (`README.md:18-27`). When the gateway goes quiet, the tuner
closes admission, drains, sleeps the serving engine, and runs a tuning cycle
(`README.md:170-176`).

Inside that cycle, the EAGLE-3 backend must produce a hidden-states file for
Speculators training. It does so by shelling out to the Speculators
`prepare_data.py` script to build the arrow dataset
(`src/speedlm/training/backends/eagle3.py:452-474`), validating the prepared
dataset in a second subprocess with `--require-nonzero-loss-mask`
(`:476-490`), and then running a **separate hidden-state generation stage** that
stands up its own vLLM engine over the verifier.

That second engine needs the verifier weights resident in GPU memory at the same
time the serving engine's allocation must be released, which is why the cycle
carries a hard memory precondition before it will proceed
(`src/speedlm/tuner/composition.py:330`, default at
`src/speedlm/training/backends/eagle3.py:188`, enforcement at
`src/speedlm/gateway/control.py:270` and `:277`). See Section 8 for exactly what
that precondition says — the popular one-line summary of it is wrong.

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
  (`README.md:165`, `src/speedlm/config.py`) is a count of held-out suite passes
  *inside* one benchmark stage, not a sample size for the stage duration.
- "1024.4 s total cycle" is the sum of four work stages of one cycle, not its
  wall clock (1134.8 s), and not a corpus mean (295.0 s over 21 complete cycles,
  range 97.8-1134.8 s — but that mean is dominated by aborts).
- "Up to 120 s of GPU-memory-release wait" is **not a measurement**. It is the
  constant `DEFAULT_GPU_MEMORY_TIMEOUT_SECONDS = 120.0`
  (`src/speedlm/gateway/control.py:44`) — a timeout ceiling. No stage in the
  corpus measures a GPU-release wait, and `SLEEPING` never exceeded 11.4 s.

### 2.3 The shape of the problem

Sleeping the engine costs ~6.4 s and waking it costs ~0.067 s. Against that, the
cycle spends ~200 s extracting, and both `CANDIDATE_STARTING` (~113 s) and a
large share of `BENCHMARKING` are vLLM engine startup and weight loading
(`README.md:167-168` attributes the benchmark phase's cost to engine restarts).

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
| Prepare training rows | subprocess to `prepare_data.py` (`training/backends/eagle3.py:452-474`) | still needed for token IDs + loss mask, unless Section 6.6 is resolved |
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

That last point matters: **the offline extraction machinery vLLM ships already
reads exactly the variable we want to capture, from exactly the serving-engine
code path.** The gap is purely one of plumbing and activation conditions
(Section 5), not of the data existing.

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
  declare it as a runtime dependency (`README.md:190-193`); a patch changes that
  posture.

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
   `written == 0` at `src/speedlm/training/backends/eagle3.py:1070`, pinned by
   `tests/test_training_speculators.py:750`.
2. **Silently zero-sample gate.** The gate could return a favorable decision from
   an empty replay. Fixed in commits `6779110` and `20231d3`. The guard is the
   `TOO_FEW_REPEATS` rejection at `src/speedlm/gate/decide.py:329-330`, pinned by
   `tests/test_gate_decide.py:450` (empty replay plus favorable metrics must still
   REJECT) and asserted end-to-end at
   `tests/e2e/test_live_idle_tuning.py:330-344`.

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
dataset now does (`training/backends/eagle3.py:1070` is the precedent), not
proceed on partial data.

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

**Detector:** a numerical equivalence test. Run the same short prompt through the
offline extraction path and the serving capture path and assert the produced
tensors match to a stated tolerance, elementwise, per layer. This test is the
single most valuable artifact of the whole project — it is what converts
"we believe capture is equivalent" into "we check it every CI run." It must exist
before any training consumes captured data.

The verdict is driven by the *aggregate* relative error
(`mean_rel_error = mean|cap-off| / mean|off|`, tolerance 0.10) together with the
shape check and the pre-norm check; cosine similarity is the corroborating
signal. The elementwise metrics `max_rel_error` and `p99_rel_error` are
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

### 6.6 The loss mask is not in the hidden-state file

The loss mask does not travel with hidden states. It lives in the arrow dataset
and is joined by integer row index: `S/src/speculators/train/data.py:388`,
`"loss_mask": self.data[index]["loss_mask"]`, with an alignment guard at `:372-378`
that warns and returns `None` when the loaded token ids do not match
`self.data[index]["input_ids"]`.

In this repository, serving-side capture stores **no token IDs at all** — the
trace store's record structure (`src/speedlm/traces/store.py:69-141`) has no
token-id field. The mask is instead re-derived by shelling out to Speculators'
`prepare_data.py` (`src/speedlm/training/backends/eagle3.py:452-474`), with
validation via `--require-nonzero-loss-mask` (`:476-490`) and offender
localization in `_zero_row` (`:503-529`, raising `FinalAssistantMaskError`).

A provenance-based path that derives the mask from capture metadata exists —
`prepare_training_row` at `src/speedlm/training/rows.py:289-340`, using
`_generated_assistant_spans` (`:306`) and `loss_mask_from_offsets`
(`:326`, defined `src/speedlm/training/masking.py:64-88`) — but it is **dead in
production**: every caller outside `training/__init__.py`'s re-export
(`:14`, `:27`) is a test
(`tests/integration/test_trace_pipeline.py`, `tests/e2e/test_agent_harness.py`,
`tests/security/test_training_integrity.py`, `tests/test_training_templates.py`).

**Consequence:** capturing activations does not by itself let us drop
`prepare_data.py`. Token IDs must be captured alongside activations, or the
provenance path must be promoted to production, before the pipeline is genuinely
simplified. If neither happens, we save the `EXTRACTING` engine but keep the
subprocess.

**Detector:** the alignment guard already exists upstream
(`S/src/speculators/train/data.py:372-378`) but only *warns* and returns `None` —
which is precisely the silent-empty-data shape of bug #1 in Section 6.0. Any
integration must convert that warning into a hard failure on our side, and must
assert `len(loss_mask) == captured_row_count` per row before training starts.

### 6.7 Hazard summary

| # | Hazard | Severity | Mitigation | Detector |
| --- | --- | --- | --- | --- |
| 6.1 | Prefix-cache coverage holes | **High** (data-dependent, biased) | `--no-enable-prefix-caching` or per-request `skip_reading_prefix_cache` | Per-request row count == prompt + generated; fail closed below a floor |
| 6.2 | Pre-norm vs post-norm target | **High** (wrong quantity, invisible) | Collect the final layer as a 4th aux layer; slice it before drafter `fc` (localized model-runner patch) — does NOT change drafter architecture | Offline-vs-serving elementwise equivalence test |
| 6.3 | Rejected draft rows | Medium | Filter by `num_rejected_tokens`; handle async deferral | Captured rows == served tokens, per request, exact |
| 6.4 | Prefill/decode numerics | Medium (pre-existing) | Measure first; `VLLM_BATCH_INVARIANT=1` if warranted | Equivalence harness across batch sizes; publish the spread |
| 6.5 | CUDA-graph padding | Medium (stale, plausible values) | Slice to `total_num_scheduled_tokens` | Row count == scheduled tokens, in-process; stale-row spot check |
| 6.6 | Loss mask provenance | Medium (blocks the simplification) | Capture token IDs, or promote `rows.py` path | Hard-fail the alignment guard; `len(mask) == rows` |

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
| ~77 tok/s (measured replay throughput, `README.md:147-148`) | ~1.3 MB/s | ~1.7 MB/s |
| 2000 tok/s (hypothetical high load) | ~34 MB/s | ~45 MB/s |

Against PCIe Gen4 x16 (~32 GB/s theoretical), the high-load case is roughly 0.14%
of link bandwidth. **[unverified: the actual host link generation and width on the
target machine has not been read from `lspci`; the ratio assumes Gen4 x16.]**

**Latency.** The copy pattern to imitate introduces no synchronization point in
the decode path — dedicated copy stream and `non_blocking=True` into pinned
memory (`example_hidden_states_connector.py:351-431`), completion detected by
event polling (`:570-575`). Steady-state serving impact should be near zero.
**[unverified: not measured. "Should be near zero" is a design intent, and
measuring it is a prototype exit criterion, not an assumption.]**

**Maintenance.** An owned dependency on vLLM internals, re-validated on every
upgrade (Section 5.5).

**Correctness engineering.** Six detectors (Section 6.7). This is real work and it
is not optional.

### 7.3 Storage and retention must be designed, not inherited

At 22.5 KiB/token, the current trace buffer default of
`max_tokens: int = 8_000_000` (`src/speedlm/config.py:192`) would imply about
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
  bodies can carry credentials (`README.md:51`). The activation cache inherits
  that sensitivity class — an activation is a lossy but real function of the
  prompt. Treat it as such by default.

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
`src/speedlm/tuner/composition.py:330` passes
`required_fraction=pipeline.gpu_memory_utilization` (default 0.80 at
`src/speedlm/training/backends/eagle3.py:188`; `composition.py:294` does not
override it), and `src/speedlm/gateway/control.py:270` computes
`int(device.total_bytes * self.required_fraction)`, compared against free memory
at `:277`.

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
`README.md:204` describes the live E2E test as an **H100** test, `DEMO.md:346`
places runs on an H100 in a SLURM partition, and `DEMO.md:321` explicitly rules
out laptop-local execution. `README.md:178-190` pins the GPU host stack but names
no GPU model or VRAM size. Every one of the 25 GPU-identifying records in
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
elementwise.**

Nothing else. No cache, no retention, no training, no integration.

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
  fine").
- A cache-hit prompt demonstrably yields fewer captured rows than tokens,
  confirming 6.1 empirically.
- Measured p50 and p99 serving latency with capture on versus off, published.

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
existing thresholds (`README.md:117-163`).

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

1. **What is the real serving latency cost of capture?** Design intent says near
   zero. Nobody has measured it. Stage 0.
2. **Do captured and offline activations actually match?** The single most
   important unknown. Stage 0.
3. **Does the prefix-cache absence inference hold?** Section 6.1 is inference from
   what crosses an API boundary, not from an explicit statement in code.
   Empirical check required.
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
   derived from code reading, not measurement; Stage 0's elementwise comparison
   remains the empirical check.
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
