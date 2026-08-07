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

The cause is in the harness, and it is structural. `_prompts_by_context` in
`tests/e2e/test_inference_configuration_matrix.py:217-232` sorts the corpus by
**character length** and takes the eight shortest for the short band and the
eight longest for the long band. The short band is therefore **by construction
the length floor of the corpus**. On the UltraChat corpus that floor is not
prompts at all — the eight shortest entries are things like ` ```yaml `,
`- 1 oz argan oil`, and `- 2 cloves garlic (minced)`: recipe fragments and a
code fence. At concurrency 8, request `i` of repeat `r` gets prompt
`(r*8 + i) % 8 == i`, so every scored repeat replays the identical eight inputs
(line 396).

Two refinements to how this has been stated informally, both from reading the
harness:

- "No corpus choice moves it" is too strong. `_load_corpus`
  (`:192-214`) truncates to the **first 2048 entries** before sorting, so the
  band is the floor of *that* slice. A different corpus — or merely a different
  ordering of the same 22,362-line file — yields a different short band. The
  accurate statement is that the band is pinned to the degenerate minimum of
  whatever it is handed, so it inherits that corpus's junk rather than sampling
  its distribution.
- The **long band is not natural long prompts either.** The eight longest are
  each self-tiled by repetition to exactly 6000 characters (`:227-231`), which
  is unusually easy for a speculative drafter to predict. The long band is as
  synthetic as the short one, in the opposite direction.

The band guard (`_validate_context_band`, `:536-550`) only checks that the short
mean is under 256 tokens and the long mean lands in [768, 3968]. It does not
enforce a floor, and it is not what sets the band.

### What may and may not be claimed

- **No matrix cell result is a production number** until the bands are built by
  sampling the corpus rather than taking its extremes. Do not quote +5.27% or
  +8.05% as a throughput improvement.
- **The gate's 410-context measurement remains the trustworthy one.** On
  representative traffic there is still **no demonstrated throughput
  improvement**: +0.48% against a ±1.11% standard error, and the gate rejected
  this candidate on accepted length (+0.024, below the required +0.05).
- What the matrix does establish is narrower and real: **on short prompts with
  short generations, this candidate measurably outperforms stock** — by more
  than five times the effect the gate sees, in both execution modes, with
  non-overlapping accepted-length errors in the CUDA-graph cell. That is worth
  investigating as a workload-dependence result. It is not a headline, and it is
  not evidence that the candidate should be promoted.

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
