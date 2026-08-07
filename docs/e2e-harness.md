# The e2e harness

**Status:** descriptive. Everything below was read out of the sources at commit
`1c8f5c0` and, where marked *(measured)*, executed. This documents what the
harness **is**, not what it should be.

**Audience:** anyone about to burn GPU hours on a `tests/e2e/` run, or trying to
read the artifacts one left behind.

**Citation convention.** Technical claims carry a source or artifact reference
relative to `/admin/home/ryan.kim/speedlm-fr` unless the path is absolute.
Symbol references are preferred because line anchors drift. Untouched historical
line anchors remain pinned to commit `1c8f5c0`, as stated above. Anything not
confirmed against a source or a measurement is labelled **[unverified]**.

---

## 1. Summary

`tests/e2e/` holds eleven test modules that drive a **real vLLM engine on a real
GPU** — the eleven that declare a module-level `pytestmark`. (It was nine at
`1c8f5c0`; `test_serving_activation_capture_overhead.py` and
`test_inference_configuration_matrix.py` have been added since.) Alongside them
sit six GPU-free modules — `test_harness_workloads.py`,
`test_harness_preflight.py`, `test_harness_compare.py`,
`test_harness_regression.py`, `test_harness_resultsdb.py` and
`test_harness_cli.py` — which cover the benchmarking harness described in
Section 10 and start no engine. The eleven are not selected by a marker; they are collected on every single
`pytest` run and skip themselves at runtime unless an environment variable opts
them in. Three things must line up or the run is wasted:

1. **The gate variable** for that specific test, set to exactly `"1"`.
2. **The right interpreter** — the project `.venv` has no torch, the vLLM venv
   has no pytest. Neither alone is sufficient for every flavor.
3. **A real GPU allocation** — several tests hard-`assert` on `SLURM_JOB_ID` and
   `CUDA_VISIBLE_DEVICES`, so they fail rather than skip outside a job.

Everything is gated **off** by default. A `pytest tests/` on a login node
reports these as skips with reasons, never as passes.

---

## 2. Which interpreter — the venv split

This is the single most common way to waste a run, because both failure modes
look like unrelated import errors.

| | project `.venv` | vLLM venv |
|---|---|---|
| path | `/admin/home/ryan.kim/speedlm-fr/.venv/bin/python` | `/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin/python` |
| `torch` | **absent** *(measured: `ModuleNotFoundError`)* | 2.11.0+cu129 *(measured)* |
| `pytest` | present | **absent** *(measured: `ModuleNotFoundError`)* |
| `speedlm` | editable install → live tree | **absent** — needs `PYTHONPATH` |

The vLLM venv gets pytest from `/data/ryan.kim/pylibs/pytest`, and `speedlm`
from the repo, via:

```
PYTHONPATH=<repo>/src:/data/ryan.kim/pylibs/pytest
```

That exact string is what `hotswap-smoke-TEMPLATE/job.sbatch` sets, with the
reasoning inline: the project's `pythonpath = ["src"]` (`pyproject.toml:66`)
only applies when pytest is launched from the project venv.

**Correction to common folklore:** it is *not* true that every e2e test runs
under the vLLM venv. The idle-tuning flavor runs under the **project** venv —
see `/data/ryan.kim/speedlm-runs/qwen-cross-20260801T063000Z/job.sbatch`, whose
last line is `"$repo/.venv/bin/python" -m pytest`. That works because
`test_live_idle_tuning.py` never imports torch in-process; it drives the gateway
over HTTP and spawns the engine as a subprocess (`tests/e2e/test_live_idle_tuning.py:867`).

Which flavor needs which:

| flavor | test module | interpreter | why |
|---|---|---|---|
| idle tuning | `test_live_idle_tuning.py` | project `.venv` | pure HTTP driver, no in-process torch |
| activation capture | `test_serving_activation_capture.py` | vLLM venv | imports `torch` and `safetensors.safe_open`; without them its `pytestmark` skipif (`:142-143`) turns the file into a skip |
| draft hot-swap | `test_serving_draft_hot_swap.py` | vLLM venv | parses safetensors headers by hand (`:421-429`) so it needs no in-process torch, but is kept on the vLLM venv by convention and to match the engine it drives |

---

## 3. Canonical invocation

```bash
#SBATCH --gres=gpu:1          # not optional, see below

export SPEEDLM_E2E_DRAFT_HOT_SWAP=1
export SPEEDLM_E2E_ARTIFACT_DIR=/data/ryan.kim/speedlm-runs/<run>/results
export PYTHONPATH=<repo>/src:/data/ryan.kim/pylibs/pytest

"$vllm_env/bin/python" -m pytest -o addopts='' -p no:cacheprovider -q -ra -s \
    tests/e2e/test_serving_draft_hot_swap.py
```

**`--gres=gpu:1` is genuinely required.** Not because the tests detect a GPU
gracefully, but because they *assert*:
`test_serving_activation_capture.py:456,459`, `test_capture_harness_matrix.py:70,73`,
`test_agent_harness.py:126,127`, `test_model_matrix.py:189` all hard-assert
`SLURM_JOB_ID` / `CUDA_VISIBLE_DEVICES` after the skip gate. Outside an
allocation these **fail**, they do not skip.

**`-s` matters** — these runs are long and `-s` is what lets the engine's stdout
reach the slurm log while it happens.

### `-o addopts=''` is a convention, NOT a requirement

Folklore says `-o addopts=''` is *required* to clear the repo's `-q -ra`
default. **That is false** *(measured)*. Running
`pytest -p no:cacheprovider -s tests/e2e/test_serving_draft_hot_swap.py` with
the project default intact works fine and collects the same single test.

What the override actually does is drop `-ra` — and `-ra` is the flag
`pyproject.toml:53-64` added on purpose, because plain `-q` once reported
"1109 passed" while >200 tests were silently skipped. Compare *(measured)*:

```
# without -o addopts=''
SKIPPED [1] tests/e2e/test_serving_draft_hot_swap.py:137: set SPEEDLM_E2E_DRAFT_HOT_SWAP=1 in an allocated GPU job
1 skipped in 0.44s

# with -o addopts=''
1 skipped in 0.40s
```

The second form is how a misconfigured GPU job silently reports "1 skipped" with
no reason attached. Snapshot jobs retain `-o addopts=''` but now add `-ra`
explicitly, so a gate skip always prints its reason in the Slurm log.

---

## 4. Environment variable reference

### 4.1 Gate variables

All gates compare with `!= "1"` and call `pytest.skip`. Only the literal string
`"1"` enables; `"true"`, `"yes"`, `"0"` and empty all skip. Unset ⇒ skip.

| variable | gate site | enables |
|---|---|---|
| `SPEEDLM_E2E_IDLE_TUNING` | `test_live_idle_tuning.py:223` | idle-tuning full cycle |
| `SPEEDLM_E2E_ACTIVATION_CAPTURE` | `test_serving_activation_capture.py:454` | stage-0 activation capture |
| `SPEEDLM_E2E_DRAFT_HOT_SWAP` | `test_serving_draft_hot_swap.py:136` | live drafter hot-swap |
| `SPEEDLM_E2E` | `test_live_vllm.py:60`, `test_proxy_overhead.py:130`, `test_token_fidelity.py:44`, `test_model_matrix.py:180`, `test_capture_harness_matrix.py:68`, `test_agent_harness.py:124` | the six generic live tests |

Two caveats on the generic `SPEEDLM_E2E` group:

- `test_model_matrix.py` additionally **hard-asserts** `SPEEDLM_MATRIX_CELL` is
  set and is a known cell (`:182-188`) — an assert, not a skip.
- `test_capture_harness_matrix.py` and `test_agent_harness.py` gate on
  `SPEEDLM_E2E=1` but read **no** `SPEEDLM_E2E_*` config variables at all. Their
  knobs are `SPEEDLM_CAPTURE_*` (`:79,81,92,119,409,623,672,703,765`) and
  `SPEEDLM_AGENT_*` (`:212,214,223,303,405,439,464,954,962,1026`). Exporting
  `SPEEDLM_E2E_MODEL` and friends does nothing for them.

### 4.2 Configuration variables

| variable | read at | default | controls |
|---|---|---|---|
| `SPEEDLM_E2E_ARTIFACT_DIR` | `test_token_fidelity.py:235`, `test_proxy_overhead.py:612`, `test_live_vllm.py:374`, `test_serving_activation_capture.py:470`, `test_serving_draft_hot_swap.py:152`, `test_live_idle_tuning.py:227` | **none — hard failure if unset** | root under which each stage creates its subdirectory. The shell runners default it to `$repo/log_artifacts` (`run_slurm_e2e.sh:5`), Python does not. |
| `SPEEDLM_E2E_READY_TIMEOUT` | `test_live_vllm.py:93` / `test_serving_activation_capture.py:486` / `test_serving_draft_hot_swap.py:199` / `test_live_idle_tuning.py:821` | **`360` / `360` / `900` / `900`** — four sites, three defaults | seconds to wait for engine readiness. `float()`. |
| `SPEEDLM_E2E_TUNING_TIMEOUT` | `test_live_idle_tuning.py:822` | `7200` | seconds for the tuning run itself |
| `SPEEDLM_E2E_REQUEST_TIMEOUT` | `test_live_idle_tuning.py:823` | `1200` | per-HTTP-request timeout during seeding/chat |
| `SPEEDLM_E2E_VLLM_ARGS` | `test_live_vllm.py:371`, `test_proxy_overhead.py:607`, `test_live_idle_tuning.py:274`, `test_token_fidelity.py:245` | `"[]"` everywhere **except** `test_token_fidelity.py`, which supplies the bounded defaults near `:245-255` | extra `vllm serve` args. JSON array of strings; asserted list-of-str. |
| `SPEEDLM_E2E_MODEL` | `test_live_vllm.py:369`, `test_proxy_overhead.py:605` | `"Qwen/Qwen3.5-2B"` / module `MODEL_DEFAULT` | model served |
| `SPEEDLM_E2E_STAGE` | `test_live_vllm.py:370`, `test_proxy_overhead.py:606` | `"stage1-qwen"` / `"proxy-overhead"` | artifact subdirectory name |
| `SPEEDLM_E2E_TUNING_CONFIG` | `test_live_idle_tuning.py:226` | **none — required, asserted `is_file()`** | the SpeedLM config JSON driving the run |
| `SPEEDLM_E2E_TUNING_PROFILE` | `test_live_idle_tuning.py:228` | `None` | optional profile copied into the run |
| `SPEEDLM_E2E_PROMPT_CORPUS` | `test_live_idle_tuning.py:63` | `None` → synthetic prompts | JSONL prompt corpus, see §7 |
| `SPEEDLM_E2E_SEED_REQUESTS` | `test_live_idle_tuning.py:260` | `max(config.tuning.min_trace_records, config.tuning.min_corpus_records)` — derived from the config, not a literal (`:271`) | seed chat requests fired before tuning; must be `> 0` |
| `SPEEDLM_E2E_ALLOW_UNMEASURED_GATE` | `test_live_idle_tuning.py:217` | unset → `False` | lets a gate decision that measured zero samples pass instead of failing (`:507`). **Module-level**, so `monkeypatch.setenv` cannot reach it. |
| `SPEEDLM_E2E_VERIFIER_MODEL` | `test_serving_activation_capture.py:463`, `test_serving_draft_hot_swap.py:145` | **none — required** | verifier/target model |
| `SPEEDLM_E2E_DRAFTER_MODEL` | `test_serving_activation_capture.py:466`, `test_serving_draft_hot_swap.py:148` | **none — required** | speculative drafter |
| `SPEEDLM_E2E_DRAFTER_DIR` | `test_serving_draft_hot_swap.py:169` | resolve from `HF_HOME/hub/<slug>` | pins the drafter snapshot **directory**; validated by asserting `config.json` exists in it (`:171-174`) |
| `SPEEDLM_E2E_VLLM_PYTHON` | `test_serving_activation_capture.py:157`, `test_serving_draft_hot_swap.py:97`, `src/speedlm/activation_capture/offline_extract.py:25` | `/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin/python` | interpreter for vLLM-venv subprocesses |
| `SPEEDLM_E2E_PORT` | `test_serving_activation_capture.py:1646,2145`, `test_serving_draft_hot_swap.py:640` | an auto-assigned free port | engine port. `int()` — a non-numeric value raises `ValueError`, not a clean assertion. |
| `SPEEDLM_E2E_PROMPT` | `test_serving_activation_capture.py:1635,2141`, `test_serving_draft_hot_swap.py:638` | `"The quick brown fox jumps over the lazy dog."` | the canary prompt |
| `SPEEDLM_E2E_TARGET_LAYER_IDS` | `test_serving_activation_capture.py:630` | derive from drafter declaration, model depth, and profile (`:642-671`) | aux layer ids for offline extraction, cross-checked against the engine's actual aux layers (`:1742-1748`) |
| `SPEEDLM_E2E_STRICT_VERDICT` | `test_serving_activation_capture.py:2022` | `"1"` → strict | whether a FAIL comparison verdict fails the test. **Only the literal `"0"` disables**; empty string leaves strict on. |
| `SPEEDLM_E2E_CAPTURE_METRICS` | `test_live_vllm.py:611` | unset → off | scrape acceptance metrics into `acceptance-metrics.txt` |

---

## 5. Traps

Each of these has cost real time. All six were re-verified against the sources.

### 5.1 Module-level `pytest.importorskip` makes a file collect as ZERO tests

**Confirmed, and already fixed** in `tests/e2e/` — no module there uses
`importorskip` any more; all eleven GPU modules declare `pytestmark`. The pattern is
documented in-tree at `test_serving_draft_hot_swap.py:66-69` ("a module-level
`importorskip` would make the whole file collect as zero tests on any
interpreter without torch — a silent pass, not a skip") and
`test_serving_activation_capture.py:126-129`.

The fix is a module-level `pytestmark` list combining the marker with a
`skipif`, e.g. `test_serving_activation_capture.py:140-150`, so both tests always
*collect* and are always *reported*.

`pyproject.toml:53-64` records why `-ra` exists at all: this exact failure mode.
`importorskip` **inside a function body** is fine and still used
(`tests/test_draft_swap.py:1910`, `tests/test_training_rows.py:1509,1518`,
`tests/e2e/test_harness_preflight.py:310`) —
function scope does not abort collection.

### 5.2 The served model id is the resolved snapshot path, not the repo id

vLLM registers the model under its **resolved snapshot path**. Sending the
friendly repo id gets a **404, not a 400** — so it reads like a routing bug, not
a bad argument. The id is therefore always read back from `/v1/models` rather
than assumed: `_get_served_model_id` at `test_serving_draft_hot_swap.py:272-288`
(docstring `:274-278`), threaded as a keyword-only arg through every request
(`:364,378`; call sites `:762,819,882,894`). A 404 is caught and re-raised with
the actual served ids listed (`:388-396`). Same pattern in
`test_serving_activation_capture.py:740-757,779-821`.

Related: the *drafter* must also be a resolved **directory**, not a repo id —
`hot_swap_draft` takes a directory (`_resolve_drafter_dir`, `:161-189`).

### 5.3 `--worker-extension-cls` needs a DOTTED path

`pkg.mod.Class`, never `pkg.mod:Class`. vLLM resolves it with
`resolve_obj_by_qualname`, which does `qualname.rsplit(".", 1)` then `getattr` —
so the colon form tries to import `pkg` and look up an attribute literally named
`mod:Class`. Constant and reasoning: `DRAFT_SWAP_WORKER_EXTENSION_CLS` in
`src/speedlm/tuner/composition.py`;
test-side constant `test_serving_draft_hot_swap.py:100-102`, flag constructed at
`:663-664`. Also noted at `src/speedlm/gateway/draft_swap.py:12-14` and
`src/speedlm/activation_capture/hook.py:26,94`.

### 5.4 `/collective_rpc` needs `VLLM_SERVER_DEV_MODE=1` and takes STRING args only

vLLM only mounts `/collective_rpc` when `VLLM_SERVER_DEV_MODE` is truthy;
without it **every RPC 404s** (`test_serving_draft_hot_swap.py:213-221`).
Production sets it at `src/speedlm/cli.py:522` and
`src/speedlm/gateway/vllm_http.py:110`, with an error message asserting it at
`vllm_http.py:300`.

Args are serialized to strings — enforced by `"args": [str(a) for a in args]`
at `test_serving_draft_hot_swap.py:309,357`, mirrored in `vllm_http.py:189` and
documented at `draft_swap.py:286-288`. Dev mode also mounts extra routes; the
allowlist note is at `vllm_http.py:36` and the route list is under test at
`tests/security/test_vllm_dev_routes.py:8`.

### 5.5 vLLM never calls a worker extension's `__init__`

It appends the class to `worker_class.__bases__`. There is no instance
construction, so **any state must be a class-level default plus lazy init**.

`src/speedlm/gateway/draft_swap.py:290-294`: "vLLM never calls `__init__` on an
extension — it appends the class to `worker_class.__bases__`. This class
therefore holds no instance state at all; the attributes below are bare
annotations (not assignments) so they do not appear in `dir()`" — annotations at
`:299-302`. The lazy pattern itself lives in
`src/speedlm/activation_capture/hook.py:106` with defaults at `:112-132` and
`_ensure_init` / `_get_lock` / `_get_pending` at `:134,145,151`. The composite
`CombinedWorkerExtension` (`draft_swap.py:886`) inherits it unchanged and
explains why at `:898-904`.

### 5.6 `finish_reason == "length"` is NORMAL

It means the response hit `max_tokens`, which is ordinary serving behaviour on
realistic prompts (ultrachat p95 ≈ 2751 chars) and still produces valid training
traces. `test_live_idle_tuning.py:398-405` accepts both:
`assert finish in ("stop", "length")`. No gate rejects on `"length"` —
`finish_reason` is recorded verbatim in replay results
(`src/speedlm/gate/replay.py:46-49,87,380,391,403`).

The one place a specific value is demanded is the tool-calling harness, which
requires `"tool_calls"` for tool-call traces (`test_agent_harness.py:914`); its
non-tool branch only requires a non-empty string (`:916`).

---

## 6. Reading the artifacts

Run artifacts live under `<SPEEDLM_HOME>/runs` (`src/speedlm/storage.py:54-69`),
with per-cycle directories `<runs>/<run_id>` (`src/speedlm/tuner/orchestrator.py:537`).

### `events.jsonl` — `<runs>/events.jsonl`

The state-machine transition log, appended by `TunerStateMachine`
(`src/speedlm/tuner/state.py:116`; writes at `:151-161`, `:194-204`, `:227-237`).
One record shape, six fields:

`sequence` (monotonic int from 0) · `timestamp` (float wall clock) · `from` (str,
`null` only on the seed record, `:199`) · `to` (str) · `reason` (str; recovery
records read `"restart recovery from <STATE>"`, `:224`) · `recovery` (bool, true
only for `_recovery_transition`, `:235`).

States: `READY, QUIESCING, SLEEPING, EXTRACTING, TRAINING, CANDIDATE_STARTING,
BENCHMARKING, PROMOTING, ROLLING_BACK, WAKING` (`state.py:27-36`); legal edges in
`VALID_TRANSITIONS` (`:45-62`). The sibling `state.json` holds the current
snapshot `{state, sequence, updated_at, reason}` (`:74-80`).

**Read it for:** did the cycle reach `PROMOTING` or `ROLLING_BACK`, and how long
each phase took. A `recovery: true` record means the process restarted mid-cycle.

### `scheduler.json` — `<runs>/scheduler.json`

Written by `TunerService._write_scheduler_status` (`src/speedlm/tuner/service.py:872-879`,
payload `:820-870`). Write failures are logged, not raised (`:878-879`) — so an
absent or stale file is not itself an error.

Fields: `schema_version` (=1), `enabled`, `lifecycle`, `serving_unrestored`,
`created_at`, `updated_at`, `lifecycle_changed_at`, `last_attempt_at`,
`last_result_at`, `last_error_at`, `cooldown_remaining_seconds`,
`last_watermark`, `last_result`
(`{outcome, artifact_id, error, decision_path, val_loss, serving_restored,
gate_acceptance}`), `last_error`.
Watermark sub-fields: `count, tokens, oldest, newest, unknown_token_records`
(`:138-142`). The reader treats the file as unusable if `enabled` is not a bool
or `lifecycle` is missing (`src/speedlm/report.py:549-556`). `serving_unrestored`
is read back as `bool | None`, where `None` means the record predates the field —
never `False`, which would let an old record reassure a reader about a condition
it never checked. `speedlm status` prints a `SERVING : NOT RESTORED` line when it
is true.

`gate_acceptance` is non-null only on a `promoted` cycle that produced a
decision, and carries `{candidate_rate, candidate_stdev, stock_rate,
stock_stdev, num_repeats, source: "gate_held_out_suite"}`. It is the gate's
**held-out suite** measurement recorded at the moment of promotion so that a
later comparison is possible at all — it is *not* a live figure and is not
comparable to one without a control arm; see the docstring on
`_gate_acceptance_baseline` in `src/speedlm/tuner/service.py`.

**Read it for:** why the scheduler did or did not fire — `cooldown_remaining_seconds`
and `last_watermark.count` answer "why no cycle yet".

**`serving_unrestored: true` is an incident, not a status.** It means a cycle's
rollback could not respawn the engine, so the child vLLM is loaded with a draft
the durable active pointer does not name — live traffic is being answered by an
unvalidated (or abandoned) draft head. Speculative decoding is lossless, so
answers are unaffected and throughput is not; the tuner re-attempts the restore
on every poll, paced by `tuning.serving_recovery_interval_seconds` and
deliberately *ahead* of `tuning.retry_cooldown_seconds`, and starts no new cycle
until it clears.

### `serving-unrestored.json` — `<runs>/serving-unrestored.json`

Present only while the condition above holds. Written by
`TunerOrchestrator._mark_serving_unrestored` and removed by the first restore
that succeeds. Fields: `schema_version` (=1), `detected_at`,
`expected_active_draft`, `error`. It is durable on purpose: the state machine
ends a preempted cycle at `READY` and so has nowhere to carry "the cycle is over
*and* serving is wrong", and the condition must survive a process restart.

`speedlm gain` reads this file directly (`speedlm.report.read_serving_unrestored`)
and prints a `SERVING NOT RESTORED` banner *above* every figure, because a
measured gain that is not being delivered is worse than no figure at all. It is
carried on `GainReport.serving_unrestored`, not as a `GainStatus` member: the
incident is orthogonal to whether a measurement exists and can coexist with any
of the four statuses. Presence is the signal — an unreadable payload still
banners.

### `decision.json` — `<run_dir>/decision.json`

The promotion verdict. Written by `write_decision`
(`src/speedlm/tuner/orchestrator.py:121-147`, called from `:817-833` via `:664`);
payload is exactly `Decision.to_dict()` (`src/speedlm/gate/decide.py`).
It refuses to persist when `num_repeats != len(per_repeat)` (`:135-140`).

Top-level keys include `verdict`, `reason`, `acceptance_delta_pp`,
`accepted_length_delta`, `throughput_statistic`, `throughput_delta_pct`,
`prometheus_throughput_delta_pct`, `min_acceptance_delta_pp`,
`min_accepted_length_delta`, `min_throughput_delta_pct`, `num_repeats`,
`warmup_repeats`, `per_repeat[]`, the acceptance criterion
(`acceptance_criterion`), the stock/candidate acceptance means and stdevs,
the stock/candidate accepted-length means and stdevs, and the output-divergence
summary. The gate promotes on `accepted_length_delta >= min_accepted_length_delta`
(not on `acceptance_delta_pp`; see `GATING_ACCEPTANCE_CRITERION` in
`src/speedlm/gate/decide.py`).

`per_repeat[]` rows (`RepeatResult.to_dict` in `decide.py`): `repeat_index, stock_tok_per_sec,
candidate_tok_per_sec, stock_acceptance_rate, candidate_acceptance_rate,
invalid_rate, output_mismatches, stock_accepted_length, candidate_accepted_length`.
`output_divergences[]` rows (`ContextDivergence.to_dict` in `decide.py`):
`context_hash, repeat_index, first_divergence_index, basis` (`"token"` or
`"character"`),
`stock_length, candidate_length, early`.

**Which number gates:** `throughput_delta_pct`. `prometheus_throughput_delta_pct`
is explicitly diagnostic and **never gates** (`Decision` and `decide` in
`src/speedlm/gate/decide.py`) — do
not quote it as the result.

### `gate-metrics/*.prom.gz` — `<run_dir>/gate-metrics/`

Written by `write_metrics_bodies` (`src/speedlm/tuner/orchestrator.py:150-191`,
called at `:830`). Each file is a **verbatim Prometheus text exposition body**,
gzipped with `mtime=0` for reproducibility, one file per scrape label
(`:180-185`). Labels must match `^[A-Za-z0-9][A-Za-z0-9._-]*$` or the write is
refused (`:54,176-179`).

Labels emitted by `_measure_block` in `src/speedlm/gate/runner.py` are
`<arm>-before` / `<arm>-after` when a block spans the whole arm, or
`<arm>-before-repeat-<i>` / `<arm>-after-repeat-<i>` for interleaved blocks,
where arm ∈ {stock, candidate}. The raw
exposition is kept because acceptance is a **counter delta** and the reported
rates must stay reconcilable (`_measure_block`). The metrics of interest are
vLLM's `vllm:spec_decode_*` counters plus decode throughput.

```bash
zcat <run_dir>/gate-metrics/candidate-after.prom.gz | rg 'spec_decode'
```

### `result.json` (hot-swap) — present keys tell you the phase reached

`tests/e2e/test_serving_draft_hot_swap.py` writes it in a **`finally`** block
(`:923-926`) into `<SPEEDLM_E2E_ARTIFACT_DIR>/draft-hot-swap-<UTC ts>/result.json`
(`:207-209`), so the file exists even on failure. Read it by asking which keys
are *present*:

| key | written at | means it got past |
|---|---|---|
| `verifier`, `drafter`, `drafter_dir`, `worker_extension_cls` | `:686-691` | process launch only — nothing verified |
| `served_model_id` | `:696` | engine reached `/health` and `/v1/models` answered |
| `injected_rpc_calls` | `:703` | phase 1: injection log line parsed (assertions on it are `:704-711`) |
| `draft_info_before` | `:736-741` | phase 2: real drafter reachable with non-empty shapes |
| `null_swap` | `:789` | phase 3b: null swap succeeded and parameter count matched (`:782-788`) |
| `canary_tokens` | `:830` | phase 4 and the phase-3c token-identity check |
| `post_swap_spec_metrics` | `:848-854` | phase 6: spec-decode counters advanced |
| `incompatible_rejection_body` | `:863` | phase 5: the bad swap was rejected non-200 (`:860-862`) |
| `capture_meta` | `:899` | phase 7: capture flushed and artifacts validated (`:898`) |

Diagnostic rules: no `served_model_id` ⇒ never became ready. `null_swap` present
but `canary_tokens` absent ⇒ died in the meta-device or token-identity check.
`incompatible_rejection_body` present but `capture_meta` absent ⇒ died in phase
5's post-rejection checks or in capture.

**Note the phase numbering in the code is not execution order:** 3a → 3b → 4 →
3c → 6 → 5 → 7. Use the table, not the numbers.

---

## 7. The realistic prompt corpus

`/data/ryan.kim/speedlm-corpora/ultrachat-prompts.jsonl` — **22,362 prompts**
*(measured: `wc -l` = 22362, 17.4 MB)*.

Built by `scripts/prepare_ultrachat_corpus.py` from a locally cached
ultrachat_200k **test-split parquet** (`--input`, read with `pyarrow.parquet` at
`:33,45`; no network). Output is one JSON object per line:

```json
{"messages": [{"role": "user", "content": "..."}]}
```

written at `:77-80` — the "bare-conversation" shape that
`speedlm.traces.normalize._detect_shape` recognises (`:8-9`).

Filters (`:11-19`, implemented `:54-71`): drop empty/whitespace-only first user
messages; drop first-user messages over 4096 chars (the harness caps `max_tokens`
at 512, `:13-17`); order-preserving exact-duplicate dedup.

Wired in via `SPEEDLM_E2E_PROMPT_CORPUS`, loaded by
`_load_prompt_corpus` (`test_live_idle_tuning.py:63`), which asserts `is_file()`
(`:67`) and parses JSONL objects with a `messages` list. Unset ⇒ the test falls
back to synthetic prompts.

Note the **default differs by consumer**: `tests/simulation/corpus.py:32` hard-codes
this same path as `DEFAULT_CORPUS_PATH` and uses it when the variable is unset,
returning `None` only if the file is absent (`:41`). The e2e test has no such
default.

---

## 8. Pinning a run to an immutable snapshot

Historically every `job.sbatch` did `cd /admin/home/ryan.kim/speedlm-fr` and ran
from the **live working tree**. The `git rev-parse HEAD` those scripts echo is a
*label*, not a guarantee: an edit landing mid-run changes what executes, and the
recorded commit becomes a lie. `scripts/make_snapshot_run.sh` removes the
coupling.

```bash
scripts/make_snapshot_run.sh --flavor hot-swap
scripts/make_snapshot_run.sh --flavor activation-capture
scripts/make_snapshot_run.sh --flavor idle-tuning \
    --tuning-config /data/ryan.kim/speedlm-runs/<run>/config.json
```

It `git archive`s the commit into
`/data/ryan.kim/speedlm-snapshots/<full-sha>/`, makes it read-only, creates a run
directory under `/data/ryan.kim/speedlm-runs/`, and writes a `job.sbatch` that
runs from the snapshot. It **never submits**; it prints the `sbatch` command.

### Why `PYTHONPATH`, and why a git worktree is not enough

The project venv at `/admin/home/ryan.kim/speedlm-fr/.venv` is an **editable**
install. Its site-packages holds `_editable_impl_speedlm.pth` containing the
single line `/admin/home/ryan.kim/speedlm-fr/src`. `.pth` files are processed by
`site` as it walks site-packages, so that path is **appended** to `sys.path`.
Any interpreter using that venv resolves `import speedlm` back to the live tree
**regardless of the working directory** — which is why merely checking out a
`git worktree` and `cd`-ing into it does not isolate a run.

*(measured, from a real `git archive` snapshot)*:

```
cd <snapshot> ; .venv/bin/python -c "import speedlm; print(speedlm.__file__)"
  → /admin/home/ryan.kim/speedlm-fr/src/speedlm/__init__.py     # live tree wins
```

`PYTHONPATH` fixes it because `site` inserts those entries **before** it scans
site-packages, so they precede the `.pth` entry. With
`PYTHONPATH=<snapshot>/src` under the project venv *(measured)*:

```
sys.path[1] = <snapshot>/src                                   ← wins
sys.path[5] = .../.venv/lib/python3.12/site-packages
sys.path[6] = /admin/home/ryan.kim/speedlm-fr/src              ← the editable .pth
speedlm.__file__ = <snapshot>/src/speedlm/__init__.py
```

The snapshot shadows the live tree by five positions. This was confirmed by
appending a unique marker constant to the snapshot's `src/speedlm/__init__.py`
and checking which copy the import produced — not by reasoning about
documented ordering.

The generated sbatch does not trust any of this. It runs a **provenance gate**
before touching the GPU: it imports `speedlm`, prints `speedlm.__file__`, and
exits non-zero with a full `sys.path` dump if the module does not resolve under
the snapshot. *(measured: the gate passes with `PYTHONPATH` set and correctly
fails when it is dropped.)*

Subprocesses are covered too — every engine spawn builds its environment with
`os.environ.copy()` (`test_serving_draft_hot_swap.py:219`,
`test_serving_activation_capture.py:687`, `test_live_idle_tuning.py:859`), so the
vLLM worker that loads `--worker-extension-cls speedlm...` inherits `PYTHONPATH`
and resolves the extension from the snapshot as well.

Snapshots are ~2.5 MB, named for the full sha, and reused across runs of the same
commit. **`git archive` serializes the commit only** — uncommitted local edits are
not included. The launcher records `source_tree_dirty_at_generation` both in
`job.sbatch` and the run-level `snapshot-provenance.txt`, as well as warning on
stderr, so a snapshot result cannot be mistaken for excluded working-tree work.

For idle tuning, the launcher derives `SPEEDLM_E2E_TUNING_TIMEOUT` from the
requested Slurm wall time with a 15-minute margin; `--tuning-timeout` can
override it only with a positive value below the wall time. Both deadlines are
printed by the generated job. Its preamble also clears inherited result-relaxing
switches (`SPEEDLM_E2E_ALLOW_UNMEASURED_GATE`,
`SPEEDLM_E2E_STRICT_VERDICT`, `SPEEDLM_E2E_HF_REFERENCE`, and
`VLLM_USE_V2_MODEL_RUNNER`) before applying explicit launcher options.

---

## 9. The GPU-free alternative: `tests/simulation/`

For orchestration-level work, do **not** queue a GPU job. `tests/simulation/`
runs the same orchestration chain on CPU in seconds. Nothing there imports torch
and nothing needs a GPU (`tests/simulation/__init__.py:1-10`); it has no `e2e`
marker and no env gate, so it runs as part of the ordinary suite. There is no
script entry point — it is plain pytest over `tests/simulation/`. The fullest
chain is `test_end_to_end.py`; the assembly helpers are `build_simulation` /
`real_gate` / `simulation_config` in `harness.py`.

**What it really exercises:** the real `TunerOrchestrator`, real
`TunerStateMachine`, real `ArtifactRegistry`, real `BenchmarkGateRunner`, real
`decide_promotion` — against a `SimulatedEngine` speaking real HTTP on loopback.
Only the three GPU-facing protocols (`SpeculatorBackend`, `RuntimeController`,
`BenchmarkGate`) are simulated. Suite freezing, HTTP replay, Prometheus scraping,
promotion arithmetic, artifact publication, pointer promotion and the persisted
`decision.json` are all genuinely covered.

**What it does NOT validate — read this before quoting a simulation result:**

> **It models arithmetic, not physics.** Acceptance and latency are *dialled in
> rather than emergent* (`tests/simulation/engine.py:45-48`): defaults
> `acceptance_rate=0.40`, `seconds_per_request=0.004` (`:72-79`). A simulation
> that had to actually speculate in order to produce a 10 pp acceptance lift
> would be testing the simulator instead of the gate. **It therefore never
> validates the speedup.**

The explicit boundaries (`engine.py:169-192`):

- **No tokenizer.** The token stream is synthetic and divergence indices are
  exact by construction, so it cannot catch a real tokenisation mismatch.
- **No batching scheduler.** Throughput scales linearly with concurrency, unlike
  a real engine.
- **No real speculation.** The counter *arithmetic* is faithful; the physics is
  not.
- **No KV cache, memory pressure, or preemption-by-swap.**
- **Sleep is bookkeeping** — it flips a flag and frees nothing, so it cannot
  exercise the production post-sleep GPU-memory wait.

Additionally, the gating throughput statistic is client wall-clock, which on a
loaded machine can swamp the millisecond-scale latency difference between the two
simulated arms (`test_end_to_end.py:35-40`) — those tests are about the chain,
not the number.

Prompts are real, not filler: `corpus.py:1-5` replays the real prompt-length
distribution, reusing the e2e corpus loader when importable (`:7-16`).

Use the simulation for orchestration, state-machine, gate-reason and
promotion-arithmetic changes. Use a GPU e2e run for anything that claims a
speedup, an acceptance rate, or correct behaviour of the vLLM integration.

---

## 10. `speedbench` — the operator surface

**Status of this section:** read out of sources in the working tree at the time
of writing. The code it describes — `tests/e2e/harness/`, `scripts/speedbench`,
`scripts/build_workload.py` and the rewritten config matrix — is **not yet
committed**, so unlike the rest of this document its line anchors are not pinned
to a commit and will move. Nothing in it has been exercised by a GPU run yet;
the harness modules it describes are covered by the six GPU-free
`test_harness_*.py` modules only. Treat every behavioural claim below as "this
is what the code does", not "this is what a run showed".

`scripts/speedbench` is a single CLI over the whole cycle: pick a workload,
validate the launch, submit it, index what came back, compare two runs, and
track a metric across commits. It adds the repo root to `sys.path`
(`scripts/speedbench:40-42`) and imports the harness modules directly
(`:44-49`), so it runs under the project `.venv` and needs no torch.
`DEFAULT_RUN_ROOT` is `/data/ryan.kim/speedlm-runs` (`:51`) and `LAUNCHER` is
`scripts/make_snapshot_run.sh` (`:52`).

### 10.1 The seven subcommands

Parser at `scripts/speedbench:875-933`, dispatch at `main()` `:936`. Every
subcommand's exit code is load-bearing — each one exits non-zero on the failure
it exists to detect, so they compose in a shell script without parsing stdout.

| command | handler | what it does | exits non-zero when |
|---|---|---|---|
| `workloads` | `cmd_workloads` `:195` (parser `:882`) | lists the declarative workloads and re-verifies each against the corpus on disk | any workload fails verification (`:236`) |
| `preflight` | `cmd_preflight` `:435` (`:889`) | runs the launch-validation checks against the argv you are about to submit | any ERROR finding (`:439`) |
| `launch` | `cmd_launch` `:458` (`:894`) | preflights, then invokes `make_snapshot_run.sh` | preflight refused and the override flag was absent (`:464-471`) |
| `index` | `cmd_index` `:539` (`:906`) | walks the run root and builds the results index | the index is stale, or zero runs were found (`:589-592`) |
| `compare` | `cmd_compare` `:654` (`:913`) | paired comparison of two runs, per cell / arm / metric | any metric returns `PROVEN_REGRESSION` (`:705`) |
| `regress` | `cmd_regress` `:711` (`:920`) | tracks one metric across commit-ordered history | any step is a proven regression (`:772`) |
| `show` | `cmd_show` `:778` (`:928`) | prints one run's cells, arms and metrics | the run parsed to nothing (`:856`) |

`index` deserves a note: an empty root is not an empty history, it is a wrong
`--root`, and the code says so inline (`:586-591`) — reporting "0 runs, all
fine" with exit 0 is exactly the silent green this CLI exists to refuse.

Two selectors are shared by `compare` and `regress`: `resolve_selector`
(`:616`) and `pair_runs` (`:629`). Two run ids give exactly one pair;
two commits give the latest run per flavor on each side and compare only the
flavors present on **both** — the intersection, never a best-effort union.

### 10.2 Adding a workload without touching test code

A workload is a JSON manifest in `tests/e2e/harness/workload_specs/`
(`workloads.spec_directory()` `tests/e2e/harness/workloads.py:410`), discovered
by filename (`available_workloads()` `:414`). The config matrix selects one by
name through `SPEEDLM_E2E_WORKLOAD`
(`tests/e2e/test_inference_configuration_matrix.py:189-194`, default
`generic-chat` at `:96`). Adding a workload therefore means adding a manifest
and a corpus; no test module changes.

The manifest is **not** hand-written. `scripts/build_workload.py` materializes
the corpus into `/data/ryan.kim/speedlm-workloads/<name>/records.jsonl` and
emits the manifest from the materialized records (`emit_manifest` `:809-880`).
The declared statistics come from `recompute_characteristics`
(`workloads.py:551-612`) — the *same* function the verifier uses — so a
hand-typed number cannot enter the manifest at all. `min_max_model_len` is
derived, not chosen: `round_up(longest_prompt_tokens + output_reserve, 512)`
(`build_workload.py:861`, reserve default 2048 at `:906`). The builder refuses
to write a manifest when a builder produced no records (`:948-950`).

So the loop is:

1. Add a builder function and register it in `WORKLOADS`
   (`scripts/build_workload.py:887`).
2. Run the builder. It writes `records.jsonl` plus the manifest, with source
   digest, tolerance, provenance, `builder_sha256` and `token_basis`
   (`:847-869`).
3. `speedbench workloads` — this re-runs `verify_workload`
   (`workloads.py:637-767`), which re-derives every declared count, percentile
   and fraction from the corpus, re-checks the source `sha256` and
   `size_bytes`, re-checks per-record `prompt_chars`, and re-checks that
   `min_max_model_len` still covers `max(prompt_tokens) + output_reserve`
   (`:753-763`). Failures are collected and raised together
   (`WorkloadVerificationError` `:133-146`), not one at a time.
4. `SPEEDLM_E2E_WORKLOAD=<name>` on the run.

Two refusals worth knowing before you write a builder. `load_spec` refuses a
manifest whose `name` disagrees with its filename (`:477-480`) and whose
`schema_version` is not `SCHEMA_VERSION = 1` (`:91`, checked `:432-437`). And
band bounds must land exactly on `PERCENTILE_KEYS` (`:99-111`) for the config
matrix, because `_percentile_key`
(`tests/e2e/test_inference_configuration_matrix.py:343`) needs the declared
token window at the band's own quantiles to validate what the server actually
received.

Bands themselves are declared quantile windows over `band_metric` (default
`prompt_chars`, `workloads.py:278-280`); the shipped defaults are
`short [0.05, 0.25]`, `medium [0.40, 0.60]`, `long [0.75, 0.95]`
(`scripts/build_workload.py:117-121`). `band_window` (`workloads.py:788`) slices
the length-sorted corpus and de-duplicates byte-identical prompts;
`band_records` (`:818`) draws `count` distinct records from that window with an
RNG seeded on `(name, version, band, count, spec.seed)` (`:837-843`), so the
draw is reproducible across runs but is not the corpus's extremes. If the
window cannot supply `count` distinct records it raises `BandExhaustedError`
(`:149-162`, raised `:836`) rather than returning a short band.

### 10.3 The preflight gate, and when `--skip-preflight` is legitimate

Preflight answers one question from the argv alone: *could this allocation have
produced a measurement?* The rationale is written at
`scripts/make_snapshot_run.sh:585-597` and names the allocations that motivated
it — a flavor whose vLLM args were silently dropped and OOMed four minutes in,
an in-test deadline shorter than the wall clock, and a gate variable that was
not exactly `"1"` and consumed a whole allocation while reporting success.

The checks are `_CHECKS` (`tests/e2e/harness/preflight.py:774`), run in
order by `preflight()` (`:789`). Only `Severity.ERROR` blocks (`:66`, `:74`;
`Finding.blocking` `:87`); warnings print and proceed. `render_findings`
(`:815`) prints errors first and ends with `DO NOT LAUNCH` or `launch is OK`.

| # | check | site | severity |
|---|---|---|---|
| 1 | `--vllm-args` given to a flavor that never reads it | `_check_vllm_args_consumed` `:492` | ERROR |
| 2 | a flavor's required options missing or empty | `_check_required_options` `:507` | ERROR |
| 3 | no `--max-model-len` / `--gpu-memory-utilization` reaching an unbounded-defaulting test | `_check_memory_bounds` `:522` | ERROR |
| 4 | Qwen3.5/GDN model without `--gdn-prefill-backend triton` | `_check_gdn_prefill_backend` `:555` | WARNING |
| 5 | `--max-model-len` below the workload's `min_max_model_len` | `_check_workload_compatibility` `:627` | ERROR |
| 6 | in-test timeout at or beyond the allocation, or wasting >30 min of paid wall clock | `_check_timeouts` `:680` | WARNING / ERROR |
| 7 | gate variable not exactly `"1"`, or a result-relaxing switch set | `_check_silent_green` `:711` | ERROR |
| 8 | commit does not exist; working tree dirty in a path the flavor executes | `_check_provenance` `:748` | ERROR / WARNING |

Check 1 is the silent-OOM bug made decidable. `model-matrix` declares
`consumes_vllm_args=False` (`:276`) because the test builds its own argv, so a
`--vllm-args` on that flavor produced a job.sbatch byte-identical to one without
it. `config-matrix` declares the same (`:325`) for the same reason —
`_engine_argv()` builds each cell's argv itself. The flavor table
(`FLAVORS` `:152-333`) is not a hand-maintained copy of the
launcher: `launcher_flavors()` (`:435`) parses the launcher's own
`case "$flavor" in` block, so the table cannot rot silently.

**Where the gate actually sits.** The launcher runs it, not you:
`make_snapshot_run.sh:599-615` rebuilds `preflight_argv` from the *original*
argv minus `--skip-preflight` (`:603-606`) — deliberately, so the gate sees what
the operator typed rather than a second hand-maintained copy — and on refusal
prints `nothing was submitted` and `exit 2` (`:609-611`). That exit happens
before `mkdir -p "$run_dir/results"` at `:680`, so a refused launch leaves **no
run directory** behind: there is no half-run to mistake for a result later.

`speedbench launch` therefore always appends `--skip-preflight` to the launcher
command (`scripts/speedbench:454`, reason at `:452-453`): it has already run
preflight itself, and running it twice would double every warning. That is the
one automatic use of the flag and it is not a bypass.

**When passing `--skip-preflight` yourself is legitimate:**

- You are driving the launcher from something that already ran
  `speedbench preflight` on the same argv — the `speedbench launch` case above.
- You are deliberately launching a configuration preflight cannot model, and you
  have read the two `WARNING:` lines it prints (`:600-601`) that say the launch
  was not validated and that nothing has checked it can measure anything.

**When it is not:** to get past an ERROR you do not want to fix. For that case
`speedbench launch` has a separate flag,
`--launch-anyway-despite-preflight-errors` (`scripts/speedbench:56`), named that
way on purpose (`:54-55`); it refuses without it (`:464-471`) and prints a loud
warning with it (`:473-483`). An ERROR is a prediction that the allocation
cannot produce a measurement, so overriding it usually costs the GPU hours it
was trying to save.

### 10.4 Reading a comparison

**A bare mean is not a finding here.** `format_mean`
(`scripts/speedbench:103-111`) never prints one; when a standard error does not
exist the row carries `NO_STANDARD_ERROR_MARKER = "no SE"`
(`tests/e2e/harness/compare.py:450`) instead of a number. This is the contract
at `tests/e2e/harness/model.py:167-170`: `delta_standard_error` is
`float | None` and is **`None` when the sample cannot support one — never
`0.0`**, so that a consumer dividing by it fails loudly instead of computing
infinite headroom. The zero-variance case is converted explicitly at
`compare.py:224-233` and independently at `scripts/speedbench:78-91`, and
`dispersion_basis` (`model.py:173`) records which of `measured` / `degenerate` /
`unsampled` produced the absence.

`MetricSeries` (`model.py:49-78`) stores raw observations, never a pre-reduced
mean; `__post_init__` (`:66-78`) rejects an empty series, rejects a
`pass_indices` tuple whose length disagrees with `values`, and rejects
`reduced=True` with anything other than exactly one value. A reduced series is a series that can
no longer produce an error bar, and it is marked as such rather than silently
averaged.

Pairing comes before statistics. `pair_observations` (`compare.py:107-147`)
matches observations by recovered pass index; if the two index *sets* differ it
raises `MispairedObservations` (`:134-141`) rather than intersecting them,
because pairing the overlap would shrink the error bar without anyone noticing.
If either side has no pass indices it falls back to list position, sets
`positional=True`, and records a note (`:116-130`). Cells whose comparability
keys disagree (`COMPARABILITY_KEYS` `:73-79`: execution mode, concurrency,
context band, model, speculative depth) are routed to `incomparable`
(`config_differences` `:356-368`) instead of being compared.

The verdict ladder is `_verdict` (`compare.py:310-353`), with
`DEFAULT_MATERIALITY_PCT = 2.0` (`:59`) and `DEFAULT_SIGNIFICANCE_T = 2.0`
(`:64`). Six members of `ComparisonVerdict` (`model.py:128-151`):

| verdict | meaning | decided at |
|---|---|---|
| `PROVEN_FLAT` | the delta is below the materiality threshold **and** the sample was powerful enough that a material shift would have been resolved: `material_delta >= significance_t * standard_error` | `compare.py:336-338` |
| `MATERIAL_SHIFT_UNRESOLVED` | the delta is at or above materiality but `t < 2.0` — something moved, the sample cannot say it was real | `:347-348` |
| `PROVEN_REGRESSION` | material, significant, and in the direction the metric's `Direction` calls worse | `:350-353` |
| `UNRESOLVED` | the comparison could not be made at all: no standard error, no t-statistic, a zero baseline mean, or a small delta that the sample could not have resolved a material shift against either | `:324-345` |

Two further members exist and are not findings: `PROVEN_IMPROVEMENT` (`:352`)
and `DESCRIPTIVE` (`model.py:151`), returned immediately for
`Direction.NEUTRAL` metrics (`compare.py:322-323`) — a neutral metric is
reported, never judged.

The distinction the module exists for is **`PROVEN_FLAT` versus `UNRESOLVED`**.
Both look like "no change". Only the first one means it. A small delta with a
large error bar is `UNRESOLVED` with a note saying so (`:339-345`), not a pass.
And **`UNRESOLVED` specifically covers the zero-variance case**: fewer than two
observations per arm, or every observation returning an identical value
(`model.py:145-148`). In that case there is no standard error, so none is
reported — the field is absent, not `0.0`. An arm that produced the same number
four times has not proven stability; it has failed to measure dispersion.

`RunComparison` exposes the two lists worth acting on: `regressions` (`:203-207`)
and `unresolved_shifts` (`:209-215`).

### 10.5 Reading a regression track

`speedbench regress` orders runs by **topology, not timestamp**.
`GitCommitOrdering._load` (`tests/e2e/harness/regression.py:166-174`) runs
`git rev-list --topo-order --reverse --all` once and caches the position of
every sha; `position()` (`:183-188`) resolves short shas through
`git rev-parse --verify --quiet <c>^{commit}`.

Three honesty properties are worth relying on:

- **Runs with no recorded commit are excluded and named** (`:224-237`), never
  attributed to HEAD. An unattributable run does not silently become a data
  point.
- **Non-ancestor consecutive pairs get a caveat** rather than being compared
  silently (`:259-265`, `is_ancestor` `:190-191`): two runs on diverging
  branches are adjacent in topological order without one following from the
  other. Config drift between adjacent runs gets a caveat too (`:266-274`).
- **Each step is judged by the same `compare_metric`** (`:275-285`), so history
  uses the identical verdict ladder as a two-run comparison — there is no
  second, looser standard for trends.

`summarize_trend` (`:299-325`) returns `TrendStatus.UNTESTABLE` when the slope
t-statistic does not exist (`:307-316`), `TRENDING` at or above
`FLAT_TREND_T_STATISTIC` and `FLAT` below (`:317-319`). `UNTESTABLE` is again
the "we did not measure this" answer, distinct from `FLAT`.

### 10.6 The measurement rules the harness encodes

These cost real GPU time to learn. They are listed here because reading them is
cheaper than rediscovering them, and because the code below is where they are
enforced — if you write a new e2e measurement, it has to satisfy the same list.

1. **Arms interleave; they do not run in sequence.** The config matrix reuses
   the production gate's mirrored-round schedule:
   `gate_runner._block_schedule("stock", blocks=BLOCKS_PER_DRAFT,
   repeats=REPEATS_PER_DRAFT)`
   (`tests/e2e/test_inference_configuration_matrix.py:873-875`;
   `src/speedlm/gate/runner.py:341`, docstring `:350-359`). A sequential
   stock-then-candidate schedule confounds the arm with anything that drifts
   over the allocation — this is exactly what voided
   `badba71-gptoss-idle` (see `benchmark-evidence.md`).
2. **The measured block opens on the same engine lifecycle state.** Each block
   gets a fresh engine process (`_fresh_engine` `:597-639`, one per block at
   `:890-917`, recorded as `"fresh_process": True` at `:908`) and burns
   `WARMUP_REPEATS = 1` before the measurement window opens (`:918-929`). Two
   arms compared across different amounts of accumulated engine state are not
   compared.
3. **Always report a standard error, and report its absence as absence.** See
   §10.4. `format_mean` (`scripts/speedbench:103-111`) has no code path that
   prints a bare mean.
4. **The control must be able to fail.** The GPU-free block of the matrix
   (`test_inference_configuration_matrix.py:1161-1574`) states a concrete
   MUTATION in each test's docstring (`:1254, 1277, 1307, 1333, 1369, 1411,
   1455, 1487, 1521, 1541`) and drives the guard red with it. A guard that has
   never been observed failing is a guard whose behaviour is unknown; this is
   the repo's recurring defect class.
5. **Artifacts are written before assertions, so a failing bound does not
   discard the run.** `band.json` at `:869` before any measurement;
   `engine-regime.json` at `:903`; per-repeat metrics inside the loop
   (`:930, :944`); `result.json` at `:984` *before* `_validate_context_band`
   (`:985`) and `_regression_failures` (`:986`), then rewritten with the verdict
   at `:993`; `matrix-result.json` rewritten after every cell (`:1153-1156`) so
   a crash mid-matrix still leaves the completed cells. The final
   `assert not failures` is last (`:1158`). An assertion that fires before the
   write turns an expensive allocation into no evidence at all.
6. **Anti-vacuity guards must prove the thing under test actually happened.**
   `_band_prompt_defects` (`:301`) asserts the exact prompt count, full
   distinctness, byte-identical match against the source record, and a
   repetition-border fraction under `REPETITION_BORDER_LIMIT = 0.25`
   (`:112`, `_border_fraction` `:277`). `_engine_argv` asserts the argv it built
   matches the cell's declared regime and the profile's speculative depth
   (`:568-573`). `_request_batch` asserts a `usage` block exists with both token
   counts above zero (`:672-678`). Each scored repeat asserts
   `output_tok_per_sec > 0.0` and `accepted_length_available` (`:948-951`).
   `_validate_context_band` (`:809`) proves the prompts the server actually saw
   fall in the declared band window. Without these, a green run is compatible
   with the engine having speculated nothing.
7. **Refuse before spending, not after.** `workloads.preflight_refusals`
   (`tests/e2e/harness/workloads.py:950-977`) refuses a workload whose
   `min_max_model_len` exceeds the engine's `--max-model-len`, or that needs
   tool support the cell does not have, and the matrix asserts that refusal
   before the GPU is touched (`test_inference_configuration_matrix.py:1046-1053`).
   Truncating the workload to fit would have produced a number, and the number
   would have been of a different workload.
