# SpeedLM CLI journey report

Date: 2026-07-25  
Scope: acceptance tests in `tests/journeys/`, invoking `.venv/bin/speedlm` only as a
subprocess with an isolated `SPEEDLM_HOME`.

## Verification

| Check | Result |
| --- | --- |
| `.venv/bin/python -m pytest -q tests/journeys` | PASS: 21 passed, 9 expected failures, 3 skipped |
| `.venv/bin/ruff check tests/journeys` | PASS: `All checks passed!` |
| `.venv/bin/mypy src/speedlm` | PASS: `Success: no issues found in 45 source files` |
| `.venv/bin/python -m pytest --ignore=tests/property` | PASS: 448 passed, 7 skipped, 9 expected failures |
| `.venv/bin/python -m pytest` | BLOCKED during collection: 5 pre-existing property-test modules require `hypothesis`, which is not installed in the project venv |

The full-suite collection errors are in `tests/property/test_metrics.py`,
`test_normalize.py`, `test_redact.py`, `test_sse.py`, and `test_store.py`. The
worktree already contains an uncommitted `pyproject.toml` change adding Hypothesis, but
the venv has not been updated. I did not install or modify anything outside the
permitted journey/report paths.

J4 and J6 are skipped on this particular runner because its sandbox rejects creation
of even a loopback TCP socket with `PermissionError: [Errno 1] Operation not
permitted`. They remain real subprocess journeys and will execute on a CPU host that
allows localhost. No external network, GPU, or real vLLM is used.

## Journey outcomes

### J1 — Day one, cold install: FAIL (one UX defect)

The ordered cold-home workflow (`--version`, `status`, `traces stats`, `gain`,
`doctor`) is otherwise understandable. Every command produces output, no command
crashes, status plainly says nothing is running, gain plainly says no gate has run,
and doctor gives a useful execution-mode explanation with a sane non-zero diagnostic
exit when requirements are unavailable.

The empty trace statistics are misleading under the acceptance contract:

- Location: `src/speedlm/cli.py:423-427`
- User did: ran `speedlm traces stats` before collecting or importing any traces.
- User saw: `measured : 0` and `estimated: 0`.
- User should see: absence of data, not zero values formatted under measurement
  labels.
- Suggested message:

  ```text
  count    : 0
  tokens   : not available (no traces)
  measured : not measured
  estimated: not estimated
  ```

Covered by strict `xfail`:
`test_day_one_stats_do_not_present_zero_as_a_measurement`.

### J2 — Bootstrap from existing logs: PASS

The journey imports a four-line, mixed-shape OpenAI JSONL file containing one
measured OpenAI response, one estimated bare conversation, malformed JSON, and an
unrecoverable object. The CLI reports two accepted records, both detected shapes,
two rejected records, and a per-line reason for each rejection. It also emits a
warning, so partial success is not silent. A subsequent subprocess reports exactly
10 measured tokens and a separate non-zero estimated bucket.

UX judgment: understandable, actionable, and honest about token provenance.

### J3 — Wrong file: FAIL (two defects)

CSV, a JSON array, a directory, a nonexistent path, and a BOM-prefixed file all
return non-zero, non-empty, traceback-free errors. The messages identify malformed
JSON, a non-object line, a directory, a missing file, or the UTF-8 BOM respectively.

Two failures remain:

1. Empty and zero-byte files are unexplained.

   - Location: `src/speedlm/cli.py:377-393`
   - User did: imported an empty/zero-byte file.
   - User saw: only `imported 0 record(s) []`, followed by exit 1.
   - User should see: why nothing was imported and what file shape is required.
   - Suggested message:

     ```text
     [speedlm] error: INPUT is empty; expected UTF-8 JSONL with one JSON object per line. No traces were imported.
     ```

   Covered by two strict `xfail` cases in
   `test_empty_file_explains_that_jsonl_records_are_required`.

2. Binary input leaks an implementation traceback.

   - Locations: `src/speedlm/traces/normalize.py:648-649` and
     `src/speedlm/cli.py:396-400`
   - User did: pointed `traces import` at a binary file.
   - User saw: a full `UnicodeDecodeError` traceback containing internal paths and
     call frames.
   - User should see: a concise encoding/format error and a corrective action.
   - Suggested message:

     ```text
     [speedlm] error: INPUT is not UTF-8 text JSONL (invalid byte at offset 0). Export the log as UTF-8 JSONL and retry.
     ```

   Covered by strict `xfail`: `test_binary_file_never_exposes_a_traceback`.

### J4 — Serve and inspect: NOT RUN ON THIS SANDBOX

The test launches the actual `speedlm vllm serve` subprocess with a lightweight
fake `vllm` executable, sends a real OpenAI chat request through the gateway, then
runs `status --json` and `traces stats` in separate processes. It asserts the live
model/port and six measured tokens, terminates serve, and checks that a fresh status
process reports `stopped` without stale state.

This runner forbids loopback sockets, so the journey skips before launch. It must be
run on a localhost-capable CPU runner before J4 can be called accepted.

### J5 — Did it help?: PASS

No-decision gain output plainly says no gate has ever run and renders no throughput
or percentage. A measured promotion shows stock/candidate values, deltas, thresholds,
verdict, and reason. A measured rejection shows the failed acceptance threshold and
explains the rejection. A `counter_reset` rejection renders all affected values as
`not measured` and never prints zero throughput or `0.00%`.

UX judgment: understandable, actionable, and not misleading.

### J6 — Enabling tuning: NOT RUN; one code-confirmed UX defect

The acceptance tests start serve with `--enable-idle-tuning`, require real proxied
traffic to succeed, and separately require a visible no-GPU refusal. Both skip here
because localhost is forbidden.

The refusal message is nevertheless unreachable in the CLI's normal output:

- Location: `src/speedlm/cli.py:64-72`
- User does: runs `speedlm vllm serve MODEL --enable-idle-tuning` without a usable
  GPU.
- User sees: serving startup logs, but no statement that tuning was refused.
- User should see: refusal reason plus confirmation that serving continues.
- Cause: the refusal is logged at `INFO`, while the CLI does not configure this
  logger for user-visible output.
- Suggested message:

  ```text
  [speedlm] warning: idle tuning disabled: no usable NVIDIA GPU is available for auto-tuning. The gateway will continue serving without tuning.
  ```

Covered by strict `xfail`:
`test_no_gpu_tuning_refusal_is_explained_to_the_user`. End-to-end confirmation that
serving continues remains blocked on this sandbox.

### J7 — Discoverability: FAIL (two defects)

All nine help paths exit zero, start with a usage line, and contain no stale
“not implemented” claims. The top-level command list matches the real public
commands, import help explains JSONL bootstrapping, and documented JSON output works.

Two discoverability/behavior mismatches remain:

1. Misspelled options are silently ignored.

   - Location: `src/speedlm/cli.py:501`
   - User does: types `status --jsoon`, `gain --jsoon`, `doctor --jsoon`, or
     `traces stats --strore somewhere`.
   - User sees: the ordinary command output and often exit 0; there is no indication
     that the requested option had no effect.
   - User should see: argparse's normal exit 2 and an `unrecognized arguments`
     message.
   - Suggested message:

     ```text
     speedlm status: error: unrecognized arguments: --jsoon
     ```

   Covered by four strict `xfail` cases in
   `test_help_contract_rejects_unknown_or_misspelled_options`.

2. Serve help hides forwarded vLLM arguments.

   - Locations: help definition at `src/speedlm/cli.py:129-138`; forwarding behavior
     at `src/speedlm/cli.py:515-525`
   - User does: runs `speedlm vllm serve --help`.
   - User sees: usage ending at `model`, with no indication that extra vLLM flags
     are accepted.
   - User should see: the passthrough operand and its behavior.
   - Suggested usage/message:

     ```text
     usage: speedlm vllm serve [OPTIONS] MODEL [VLLM_ARGS...]
     Additional arguments are forwarded to `vllm serve`; SpeedLM owns --host and --port.
     ```

   Covered by strict `xfail`:
   `test_serve_help_discloses_forwarded_vllm_arguments`.

## Overall acceptance

Not accepted yet. J2 and J5 pass. J1, J3, and J7 expose reproducible CLI/UX defects
and are preserved as strict expected failures. J4 and J6 require one rerun on a host
that permits localhost; J6 also has a code-confirmed missing refusal message. No
production source was changed.
