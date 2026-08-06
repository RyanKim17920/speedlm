# Benchmark evidence

This is the benchmark record for commit
`8b72d9a654cdb0743ba1a0968863c151dd34f6fc`. The provenance files for both
definitive runs record that exact commit and a clean source tree:

- `/data/ryan.kim/speedlm-runs/8b72d9a-qwen-idle/snapshot-provenance.txt`
- `/data/ryan.kim/speedlm-runs/8b72d9a-gptoss-idle/snapshot-provenance.txt`

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

The definitive decisions record eager execution. They do not establish that
`--enforce-eager` is speed-only or that execution mode has zero acceptance or
output-equivalence effect. No such evidence exists; divergence must be measured
for each runtime configuration.

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
