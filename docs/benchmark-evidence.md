# Benchmark evidence

The definitive promotion-gate record is for commit
`8b72d9a654cdb0743ba1a0968863c151dd34f6fc`. The provenance files for both
definitive runs record that exact commit and a clean source tree:

- `/data/ryan.kim/speedlm-runs/8b72d9a-qwen-idle/snapshot-provenance.txt`
- `/data/ryan.kim/speedlm-runs/8b72d9a-gptoss-idle/snapshot-provenance.txt`

The configuration-matrix section below is from a later effort at `5c492ab` and
re-uses the same rejected candidate artifact; its provenance files are
`/data/ryan.kim/speedlm-runs/config-matrix-depth3-5c492ab/snapshot-provenance.txt`
and `/data/ryan.kim/speedlm-runs/config-matrix-eager-c8-5c492ab/snapshot-provenance.txt`.
It does not change the gate result. Read it after the gate result, not instead
of it.

## Definitive result: the tuned head does not improve throughput

These are the first trustworthy measurements. The tuned head does **not**
improve tokens per second on either model.

| Model | Mean accepted length delta | Replay throughput delta | Independent Prometheus delta | Result |
| --- | ---: | ---: | ---: | --- |
| Qwen3-8B | +0.024, below required +0.05 | +0.48% ± 1.11% (0.43 sigma) | +0.46% | rejected; throughput result is noise, not an improvement |
| gpt-oss-20b | −0.163, below required +0.05 | −5.57% ± 1.15% (4.8 sigma) | −5.69% | rejected; material regression that breaches the −2% floor |

Both candidates were rejected on the k-invariant mean-accepted-length criterion
before throughput could qualify them. `acceptance_delta_pp` remains in the
decision for compatibility, but it is not the promotion criterion.

Exact decision artifacts:

- `/data/ryan.kim/speedlm-runs/8b72d9a-qwen-idle/results/live-idle-tuning/terminal-decision.json`
- `/data/ryan.kim/speedlm-runs/8b72d9a-qwen-idle/results/live-idle-tuning/speedlm_home/runs/b172cadd93d947f1a7bf82fada02dcb5/decision.json`
- `/data/ryan.kim/speedlm-runs/8b72d9a-gptoss-idle/results/live-idle-tuning/terminal-decision.json`
- `/data/ryan.kim/speedlm-runs/8b72d9a-gptoss-idle/results/live-idle-tuning/speedlm_home/runs/a0ca6d36e7ac424cbc73978079a047dc/decision.json`
- `/data/ryan.kim/speedlm-runs/8b72d9a-qwen-idle/results/live-idle-tuning/speedlm_home/runs/events.jsonl`
- `/data/ryan.kim/speedlm-runs/8b72d9a-gptoss-idle/results/live-idle-tuning/speedlm_home/runs/events.jsonl`

The decisions record the same interleaved block schedule for both models:
candidate 3, stock 3, stock 2, candidate 2, with restarts at the block
boundaries recorded in `block_schedule`.

The Qwen gate replayed **410 UltraChat contexts** at 512 max tokens over 5
scored repeats (`num_contexts: 410`, `benchmark_max_tokens: 512`,
`num_repeats: 5`; the 410-row suite is
`.../runs/b172cadd93d947f1a7bf82fada02dcb5/held-out/suite_contexts.jsonl`, and
the corpus is `/data/ryan.kim/speedlm-corpora/ultrachat-prompts.jsonl`). Stock
107.682 tok/s against candidate 108.196 tok/s; stock accepted length 2.13195
against candidate 2.15572. Keep those figures in mind for the next section.

## The configuration matrix, and why it does not overturn the gate

The gate measures one regime. A configuration matrix was built to check whether
the "no throughput improvement" result was an artifact of that regime — in
particular whether `--enforce-eager`, which the gate uses, was hiding a win that
CUDA graphs would reveal.

### The execution-mode hypothesis is refuted

Two cells, both at the **profile-resolved draft depth 3** (the matrix previously
hardcoded depth 5, which is wrong for Qwen; depth is now resolved from the
model's profile, shared by both arms, and recorded per cell — commit `c5aa14c`).
Qwen3-8B, request concurrency 8, short context band, prefix caching off in both
arms, and the **same candidate artifact** the gate rejected
(`.../8b72d9a-qwen-idle/results/live-idle-tuning/speedlm_home/runs/artifacts/91aa5142…993a`,
recorded as `candidate_draft` in each cell's `results/manifest.json`).

| Cell | Job | Artifact | Stock tok/s | Candidate tok/s | Candidate delta | Stock accepted | Candidate accepted | Accepted delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA graphs | 371032 | `/data/ryan.kim/speedlm-runs/config-matrix-depth3-5c492ab/results/cuda_graphs-c8-short/result.json` | 274.862 ± 0.505 | 289.348 ± 1.094 | **+5.27% ± 0.36** | 2.1584 ± 0.0061 | 2.2792 ± 0.0103 | **+0.121** |
| Eager | 371035 | `/data/ryan.kim/speedlm-runs/config-matrix-eager-c8-5c492ab/results/eager-c8-short/result.json` | 103.126 ± 4.106 | 111.046 ± 1.650 | **+8.05% ± 3.23** | 2.1956 ± 0.0208 | 2.3364 ± 0.000 | **+0.141** |

Both modes show the candidate ahead. **Execution mode explains nothing.** The
hypothesis that CUDA graphs would change the sign, or that eager was suppressing
a win, is refuted by its own test.

### The workload explains it, and the workload is not representative

The matrix and the gate disagree by five to sixfold **on the same quantity, the
same artifact, the same model, the same depth**:

| | Matrix short band | Gate replay |
| --- | --- | --- |
| Inputs | **8 unique prompts**, 10-25 tokens (mean 16.0) | 410 UltraChat contexts |
| Generation | 128 tokens | 512 tokens |
| Repeats | 4 scored repeats of the same 8 prompts | 5 |
| Accepted-length delta | **+0.121 to +0.141** | **+0.024 ± 0.0011** |
| Throughput delta | +5.27% to +8.05% | +0.48% ± 1.11% |

The cause was in the harness, and it was structural. The diagnosis below is
kept in the past tense and pinned to commit `5c492ab`, because it is what
explains the archived numbers above; the code it cites no longer exists. What
replaced it is described in the next subsection, and replacing it does **not**
retroactively make the archived numbers interpretable.

`_prompts_by_context` in
`tests/e2e/test_inference_configuration_matrix.py:217-232` (at `5c492ab`) sorted the corpus by
**character length** and took the eight shortest for the short band and the
eight longest for the long band. The short band was therefore **by construction
the length floor of the corpus**. On the UltraChat corpus that floor is not
prompts at all — the eight shortest entries are things like ` ```yaml `,
`- 1 oz argan oil`, and `- 2 cloves garlic (minced)`: recipe fragments and a
code fence. At concurrency 8, request `i` of repeat `r` got prompt
`(r*8 + i) % 8 == i`, so every scored repeat replayed the identical eight inputs
(line 396).

Two refinements to how this had been stated informally, both from reading the
harness at `5c492ab`:

- "No corpus choice moves it" was too strong. `_load_corpus`
  (`:192-214`) truncated to the **first 2048 entries** before sorting, so the
  band was the floor of *that* slice. A different corpus — or merely a different
  ordering of the same 22,362-line file — yielded a different short band. The
  accurate statement is that the band was pinned to the degenerate minimum of
  whatever it was handed, so it inherited that corpus's junk rather than sampling
  its distribution.
- The **long band was not natural long prompts either.** The eight longest were
  each self-tiled by repetition to exactly 6000 characters (`:227-231`), which
  is unusually easy for a speculative drafter to predict. The long band was as
  synthetic as the short one, in the opposite direction.

The band guard (`_validate_context_band`, `:536-550`) only checked that the short
mean was under 256 tokens and the long mean landed in [768, 3968]. It did not
enforce a floor, and it was not what set the band.

### The stated cause has been removed

Every mechanism named above is gone from
`tests/e2e/test_inference_configuration_matrix.py`. (Line anchors in this
subsection are to the working tree at the time of writing; the benchmarking
harness under `tests/e2e/harness/` and `scripts/speedbench` is not yet
committed, so they will move.) Bands are no longer
constructed by the test at all; they are drawn from a declarative workload
manifest by `tests/e2e/harness/workloads.py`.

- **Bands are percentile windows, declared in the manifest, not extremes.**
  `workloads.band_window` (`tests/e2e/harness/workloads.py:788`) sorts the whole
  corpus by the manifest's `band_metric` (`prompt_chars` for all four shipped
  workloads) and slices the declared quantile interval; `band_records`
  (`:818`) draws `count` distinct records from that window with an RNG seeded on
  `(name, version, band, count, spec.seed)` (`:837-843`). The shipped windows are
  `short = [0.05, 0.25]` and `long = [0.75, 0.95]` (defaults at
  `scripts/build_workload.py:117-121`; a `medium = [0.40, 0.60]` band is declared
  but the matrix does not use it). There is no truncation to a prefix of the
  corpus: `band_window` sees all 22,362 generic-chat records.
- **Under-supply is refused, not padded.** `band_records` raises
  `BandExhaustedError` (`workloads.py:149-162`, raised `:836`) rather than
  returning fewer prompts, and the matrix converts that to an `AssertionError`
  (`test_inference_configuration_matrix.py:476-477`) instead of topping the band
  up from outside the window.
- **The band is no longer eight prompts replayed four times.**
  `_distinct_prompts_needed` (`:263-274`) asks for `concurrency *
  REPEATS_PER_DRAFT` distinct records, so a concurrency-8 cell now draws 32
  distinct prompts and a concurrency-32 cell draws 128.
- **Prompts are used verbatim and the tiling is now a rejection fixture.**
  `_cell_band` takes `record.final_user_text` unmodified (`:480`), and
  `_band_prompt_defects` (`:301`) asserts exact count, full distinctness,
  byte-identical match against the source record, and a
  repetition-border fraction below `REPETITION_BORDER_LIMIT = 0.25`
  (`:112`, computed by `_border_fraction` `:277`). The old
  `((prose + "\n") * copies)[:6000]` recipe is reconstructed at `:1280-1285`
  purely so that a test can prove the guard rejects it.
- **The band guard now checks the declared window, not a hardcoded range.**
  `_validate_context_band` still exists (`:809`) but compares the mean of the
  server-reported `usage.prompt_tokens` against the manifest's declared
  percentile window widened by `DEFAULT_BAND_TOKEN_TOLERANCE = 0.35`
  (`_band_token_bounds`, `:798`; asserted `:836`).
- **Declared statistics are machine-computed, never hand-typed.** Both
  `scripts/build_workload.py` (manifest writer, `emit_manifest` `:809-880`) and
  `workloads.verify_workload` (`:637-767`) call the same
  `recompute_characteristics` (`workloads.py:551-612`), and `verify_workload`
  re-derives every declared percentile, count and fraction from the corpus on
  disk against the declared tolerance. The matrix runs it **per cell**
  (`test_inference_configuration_matrix.py:470`), so a corpus that drifted from
  its manifest fails the run rather than quietly changing the band.

Measured against the generic-chat manifest, the bands are now interior to the
corpus rather than at its edges (128-prompt draw, i.e. a concurrency-32 cell;
corpus `prompt_chars` runs 12 to 4101):

| band | window | prompt-char range of the draw | mean prompt tokens |
| --- | --- | ---: | ---: |
| short | p5–p25 | 70 – 149 | 19.3 |
| long | p75–p95 | 742 – 2674 | 310.7 |

Neither band touches the corpus extreme in either direction, which is exactly
the property the old construction lacked.

### What may and may not be claimed

- **Every archived config-matrix result on disk remains uninterpretable as a
  production number, and re-running is required.** The +5.27% and +8.05% figures
  above were produced by the old band construction; fixing the construction does
  not retroactively re-measure them. Nothing under
  `/data/ryan.kim/speedlm-runs` was produced by the new code — none of the 65
  run directories carries a `workload` key or names any of the four workloads,
  and the newest config-matrix runs predate the manifests. Do not quote +5.27%
  or +8.05% as a throughput improvement; do not quote them as a corrected
  number either. The only way to obtain a quotable matrix number is a fresh GPU
  run under the new band construction, which has not been made.
- **The gate's 410-context measurement remains the trustworthy one.** On
  representative traffic there is still **no demonstrated throughput
  improvement**: +0.48% against a ±1.11% standard error, and the gate rejected
  this candidate on accepted length (+0.024, below the required +0.05).
- What the old matrix suggested is narrower than it looked and is now only a
  hypothesis: that on short prompts with short generations this candidate may
  outperform stock. The effect was measured on eight degenerate inputs replayed
  four times, so it is a lead to re-test under the new bands, not a result. It
  is not a headline, and it is not evidence that the candidate should be
  promoted.

### Limitations that remain after the fix

These are properties of the new system, not of the archived runs.

- **A generic-chat cell does not exercise a long-context regime, and cannot.**
  The generic-chat corpus has a `prompt_tokens` p95 of 575 and a maximum of
  1293 (`tests/e2e/harness/workload_specs/generic-chat.json`). Its "long" band
  is the p75–p95 character window, whose measured 128-prompt draw means 310.7
  prompt tokens; the corpus's own token percentiles at those ranks are 144 and
  575. A "long" generic-chat cell is a few-hundred-token cell. Any claim about
  long-context speculative decoding needs one of the agentic or long-context
  workloads.
- **The workloads that would exercise long context cannot run on the matrix
  today.** Their declared `min_max_model_len` — 18432 for `agentic-tool-loop`,
  23552 for `agentic-mixed-outcome`, 360960 for `long-context-sessions` — all
  exceed the matrix's `MAX_MODEL_LEN = 4096`
  (`tests/e2e/test_inference_configuration_matrix.py:86`), and
  `workloads.preflight_refusals` (`workloads.py:950-977`) refuses the launch
  rather than truncating. That refusal is the correct behaviour and is asserted
  before any GPU spend (`:1046-1053`), but the consequence is that
  `generic-chat` (`min_max_model_len` 3584) is the **only** workload the config
  matrix can currently run. No run on any other workload has been made.
- **The agentic workloads are narrow, and one is survivorship-biased.**
  `agentic-tool-loop` has 626 per-turn records, but its own provenance block
  records the honest denominator: 43 session instances over 16 bug templates on
  a single synthetic e-commerce application, yielding only 129 unique system
  prompts and 44 unique first user messages. It is built from a success-only
  upstream corpus, so it contains no failed sessions at all.
  `agentic-mixed-outcome` is the corrective: 617 records split 314 success /
  303 safe-failed, across 189 unique system prompts and 59 unique first user
  messages. It is the only workload carrying a `ground_truth` block — an
  `acceptance_fraction` distribution (n=617, mean 0.809, p50 0.841, p5–p95
  0.556–0.979) measured upstream against gpt-oss-20b with a 3-token eagle3
  speculator under greedy sampling, and explicitly *not* recomputed here, so it
  is a landing zone for sanity-checking a harness run, not an independent
  measurement. Neither is a sample of real production traffic shape; both are
  synthetic-agent traffic with a small template denominator, and a result on
  either should be reported with that denominator attached.
- **Nothing in the new system has been validated on a GPU.** The harness
  contract, workload loader and verifier, results index, comparator, regression
  tracker, preflight and CLI are covered by GPU-free tests only
  (`tests/e2e/test_harness_workloads.py`, `test_harness_preflight.py`,
  `test_harness_compare.py`, `test_harness_regression.py`,
  `test_harness_resultsdb.py`, `test_harness_cli.py`, plus the marker-free block
  at `test_inference_configuration_matrix.py:1161-1574`). None of them starts a
  vLLM engine. No GPU run has yet exercised the new band construction end to
  end, so the new construction is verified as code and unverified as a
  measurement.

The serving-time capture overhead measured in the same effort is recorded in
[Serving-time activation capture](serving-time-activation-capture.md), Section
7.2.1.

## Voided archive results

The runs below are historical diagnostics only. None is evidence that the tuned
head improves acceptance or throughput.

### `multicycle-20260731T190000Z`: VOID

The terminal decision says `promote`, +6.83 percentage points acceptance and
+8.06% replay throughput, but the result is void:

- The decision omits `stock_draft`, `num_contexts`, and
  `candidate_acceptance_stdev` (readers therefore see null for those fields). It
  has no accepted-length criterion, no throughput standard error, no arm
  schedule, and no persisted divergence records. It is the thinnest decision
  schema reviewed for this record.
- The held-out file has 103 rows, but 102 are numbered variants of
  `This is idle-tuning seed request N/512. Reply with one short sentence.`; the
  remaining row is a preemption prompt. Exact message hashes differ because the
  number differs, so exact-hash deduplication did not detect the semantic
  duplication. The trace snapshot used for training has 409 records and uses
  the same numbered seed-request template, so the held-out suite is not a
  distinct template distribution.
- The run persisted aggregate `output_mismatches: 0` counts, but no response
  bodies, token sequences, pair identities, or divergence records in the
  decision. The run log identifies source commit `6d996ac`; `decide.py` at that
  commit did perform an exact stock/candidate `response_text` comparison. The
  archive does not preserve enough evidence to audit that comparison or
  establish output equivalence after the fact. It is therefore incorrect to
  say that the gate omitted an output-equality check; the void finding is that
  the archive cannot independently demonstrate what that check compared.

Artifacts:

- `/data/ryan.kim/speedlm-runs/multicycle-20260731T190000Z/results/live-idle-tuning/terminal-decision.json`
- `/data/ryan.kim/speedlm-runs/multicycle-20260731T190000Z/results/live-idle-tuning/speedlm_home/runs/50d9a0702c16429a9f13388c89d635c8/decision.json`
- `/data/ryan.kim/speedlm-runs/multicycle-20260731T190000Z/results/live-idle-tuning/speedlm_home/runs/50d9a0702c16429a9f13388c89d635c8/held-out/suite_contexts.jsonl`
- `/data/ryan.kim/speedlm-runs/multicycle-20260731T190000Z/results/live-idle-tuning/speedlm_home/runs/1aa30d827f334cb097a04b338865d0a1/trace-snapshot/traces.jsonl`
- `/data/ryan.kim/speedlm-runs/multicycle-20260731T190000Z/slurm-368906.out`

### July 30 live-idle runs: VOID

These runs measured no candidate acceptance effect. In every run, all
substantive `vllm:spec_decode_*_total` values and per-position totals have the
same stock and candidate deltas. The only differences in the raw
`spec_decode` exposition are process-specific `*_created` timestamps, so it is
not literally byte-identical. The acceptance evidence is non-discriminating and
cannot support a promotion. The warmup deltas are 1,155 drafted and 549 accepted
in each arm; the threshold and validate5 deltas are 1,925 drafted and 915
accepted in each arm, including identical per-position totals. Unrelated scrape
lines differ, and the logs identify separate stock and candidate engine
lifecycles with different draft arguments; the artifacts do not establish that
both arms scraped one engine. What they establish is that the speculative-decode
acceptance counters measured no arm effect.

The warmup run promoted with a 0.0 percentage-point acceptance delta because
its run configuration explicitly set `min_acceptance_delta_pp` to 0.0. The
nominal/default bar was 1.0 percentage point; the later threshold and validate5
runs used 1.0 and rejected the same 0.0 delta. This was a permissive run
configuration, not a gate clearing a configured 1.0-point bar.

Artifacts (these July 30 artifacts are stored in the repository's
`log_artifacts`, not under `/data/ryan.kim/speedlm-runs`):

- `/admin/home/ryan.kim/speedlm-fr/log_artifacts/live-idle-warmup-20260730T203017Z/config.json`
- `/admin/home/ryan.kim/speedlm-fr/log_artifacts/live-idle-warmup-20260730T203017Z/results/live-idle-tuning/terminal-decision.json`
- `/admin/home/ryan.kim/speedlm-fr/log_artifacts/live-idle-warmup-20260730T203017Z/results/live-idle-tuning/gate-metrics/`
- `/admin/home/ryan.kim/speedlm-fr/log_artifacts/live-idle-thresholds-20260730T213537Z/config.json`
- `/admin/home/ryan.kim/speedlm-fr/log_artifacts/live-idle-thresholds-20260730T213537Z/results/live-idle-tuning/terminal-decision.json`
- `/admin/home/ryan.kim/speedlm-fr/log_artifacts/live-idle-thresholds-20260730T213537Z/results/live-idle-tuning/gate-metrics/`
- `/admin/home/ryan.kim/speedlm-fr/log_artifacts/live-idle-validate5-20260730T232443Z/config.json`
- `/admin/home/ryan.kim/speedlm-fr/log_artifacts/live-idle-validate5-20260730T232443Z/results/live-idle-tuning/terminal-decision.json`
- `/admin/home/ryan.kim/speedlm-fr/log_artifacts/live-idle-validate5-20260730T232443Z/results/live-idle-tuning/gate-metrics/`

### `badba71-gptoss-idle`: VOID

The apparent +33.60% Prometheus decode-rate win is a sequential-arm lifecycle
artifact. Under the interleaved `8b72d9a` run, candidate throughput is nearly
unchanged, 63.20 to 63.32 tok/s, while stock moves from 47.26 to 67.05 tok/s on
the same model and identical relevant vLLM arguments (`--max-model-len 4096`,
`--gpu-memory-utilization 0.75`, `--enforce-eager`). The earlier "win" came from
measuring the stock arm under a degraded engine lifecycle, not from a faster
candidate.

Artifacts:

- `/data/ryan.kim/speedlm-runs/badba71-gptoss-idle/results/live-idle-tuning/terminal-decision.json`
- `/data/ryan.kim/speedlm-runs/badba71-gptoss-idle/results/live-idle-tuning/invocation.json`
- `/data/ryan.kim/speedlm-runs/badba71-gptoss-idle/slurm-369970.out`
- `/data/ryan.kim/speedlm-runs/8b72d9a-gptoss-idle/results/live-idle-tuning/terminal-decision.json`
- `/data/ryan.kim/speedlm-runs/8b72d9a-gptoss-idle/results/live-idle-tuning/invocation.json`
- `/data/ryan.kim/speedlm-runs/8b72d9a-gptoss-idle/slurm-370593.out`

## Limits of the evidence

The definitive decisions record eager execution (`engine_enforce_eager: true`).
They do not establish that `--enforce-eager` is speed-only or that execution
mode has zero acceptance or output-equivalence effect, and divergence must still
be measured for each runtime configuration.

The configuration matrix narrows this slightly but does not close it. It shows
that execution mode does **not** flip the sign of the candidate-versus-stock
comparison — both modes favour the candidate on the same workload — while the
absolute accepted lengths do differ across modes (stock 2.1584 under CUDA graphs
against 2.1956 eager, on identical prompts). So mode is not a confound on the
*direction* of that comparison, and is not established to be neutral on the
measured quantities themselves. No output-equivalence measurement across modes
exists.

Separately, note that a result being reproducible in both execution modes is not
evidence that it generalizes: the matrix result reproduces in both modes and
still does not transfer to representative traffic, because the confound was the
workload, not the runtime.

### The truncation caveat

The gate now classifies each replay arm into a `TruncationRegime`
(`src/speedlm/gate/decide.py:47-86`) based on how much of the workload's own
stopping behaviour survived into the measurement. An arm in which **no**
generation stopped naturally (regime `SATURATED`) is rejected with
`truncation_saturated` before throughput is compared, because the output cap —
not the model — chose every generation length, and the measured throughput
number describes fixed-length decode, not the workload.

Every archived decision predates the truncation columns
(`stock_finish_reasons`, `candidate_finish_reasons`, `stock_truncated`,
`candidate_truncated` on each `per_repeat[]` row), so all archived records
classify as `untestable` rather than as low-truncation. The truncation check
has not yet run on a live GPU benchmark; its classification logic is covered by
GPU-free unit tests only.

The archived live runs' serving traffic was 85-92% truncated at a 512-token
cap: the Qwen3-8B run `8b72d9a-qwen-idle` finished 1889 of 2049 responses at
`length` (92.2%), the gpt-oss-20b run `8b72d9a-gptoss-idle` finished 1747 of
2049 (85.3%). Both sit in the `mixed` regime — heavily truncated, but with
enough natural stops (160 and 302 respectively) to bound where the model's own
stopping distribution lies. The throughput numbers from these runs are genuine
measurements of the candidate versus stock under that cap; the caveat is that
they are bounded by a 512-token ceiling and should not be read as claims about
unbounded generation throughput.

Code-level fail-closed checks must not be described as exercised merely because
they exist. Several guards have not been observed firing in a real run. In
particular, neither definitive run produced an observable **publish-time**
vocabulary-bound-check result: both candidates were rejected before the publish
path ran, and both terminal decisions have null `artifact_id`, `candidate_draft`,
and `draft_vocab_mapping` fields. The guard's own docstring says why a successful
report must be recorded: a check whose only output is an exception is
indistinguishable from a check that never ran. Here there is no such report. The
absence of output is not evidence that the guard passed or that publication was
validated. The implementation is
`Eagle3Adapter._assert_published_vocab_mapping` in
`src/speedlm/tuner/eagle3.py`.
