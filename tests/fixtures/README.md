# Fixtures

Vendored gate artifact fixtures for `tests/test_demo_render_numbers.py`.

**Provenance:** The original artifacts lived at
`/data/ryan.kim/speedlm-runs/regate-big-run2/decision.json` and
`/data/ryan.kim/speedlm-runs/demo-video-run2/capture_manifest.json`.
They were **deleted on 2026-08-21** as part of an approved 142 GB cleanup and
will never be restored.

These fixtures are **transcriptions** from surviving documentation, not copies
of the original files. Every value was cross-referenced against
`docs/speedup-ceiling.md` (2026-08-13).

## Files

- `regate-big-run2/decision.json` — Transcribed from `docs/speedup-ceiling.md`
  section 2 ("What we measured, four times", row `regate-big-run2`).
- `demo-video-run2/capture_manifest.json` — Minimal manifest with `slurm_job_id`
  378951, as recorded in the test comment for `test_capture_job_id`.

## Values that could not be sourced

- `per_repeat` tok/s values: Reconstructed from documented per-repeat throughput
  deltas (+13.92 through +19.16) and stock decay (127.05 -> 121.37 tok/s) in
  `docs/speedup-ceiling.md` section 2. Candidate tok/s = stock x (1 + delta/100).
  This is a reconstruction, not a direct transcription.
