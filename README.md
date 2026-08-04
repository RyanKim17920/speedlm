# SpeedLM

SpeedLM is a near-drop-in wrapper around `vllm serve`. It launches vLLM behind
a streaming gateway, proxies OpenAI-compatible traffic, and captures completed
requests inline. Its adaptation pipeline is designed to use idle GPU periods to
personalize a speculative draft head from that traffic, then promote the
candidate only when a held-out gate measures improvements in both draft
acceptance and throughput.

SpeedLM is a warm-start adaptation system, not a draft-model trainer from
scratch. Training always starts from an existing public draft head—for example,
`RedHatAI/gpt-oss-20b-speculator.eagle3`—and fine-tunes it to the traffic seen
by a particular deployment. The tuner explicitly refuses a missing
`from_pretrained` base.

## How it is structured

- **Gateway** — `speedlm vllm serve` starts a loopback vLLM child and exposes a
  streaming reverse proxy. Every admitted HTTP exchange is assigned an
  `exchange_id` and journaled as exact request/response bytes before those
  bytes advance to vLLM or the client. Storage work runs outside the asyncio
  event loop with bounded, per-chunk backpressure.
- **Traces** — completed request/response pairs are normalized and appended to
  a persistent, bounded JSONL trace store. Protocol-neutral raw exchanges
  remain available when an endpoint is unknown, a response is non-2xx or
  malformed, or semantic decoding is added later. Decoded traces link back to
  their raw exchange. Existing OpenAI-style JSONL can also be imported.
- **Tuner** — the idle-cycle logic detects a quiet gateway, quiesces serving,
  snapshots traces, prepares training rows, extracts verifier hidden states,
  warm-starts Speculators training from the public draft head, and materializes
  an immutable candidate. New traffic preempts the cycle and serving is
  restored on failure.
- **Gate** — a frozen held-out suite is replayed against the stock and candidate
  drafts. The decision is fail-closed: unavailable acceptance metrics, counter
  resets, invalid responses, output mismatches, or either improvement threshold
  being missed all reject the candidate. Only a candidate with **measurably
  improved acceptance** and no throughput regression is promoted — see
  [Promotion thresholds](#promotion-thresholds).
- **Doctor** — `speedlm.doctor` implements read-only checks for Python, GPU,
  CUDA, pinned packages, disk, memory, and verifier/draft compatibility, and
  produces an execution plan.
- **Report** — the read-only status and gain reports summarize gateway state,
  traces, tuner state, active artifacts, and real persisted gate decisions.
  They do not invent a gain when no trustworthy measurement exists.

Persistent state defaults to `~/.speedlm/` and can be relocated with
`SPEEDLM_HOME` or the global `--home` option:

```text
~/.speedlm/
  exchanges/  # private 0700; exact raw bodies may contain credentials/secrets
  traces/
  profiles/
  runs/
```

## Install

SpeedLM requires Python 3.12:

```bash
python -m pip install -e .
```

For development checks:

```bash
python -m pip install -e ".[dev]"
```

## CLI: what works today

These commands are wired to working handlers:

```text
speedlm vllm serve MODEL [--host HOST] [--port PORT] [VLLM_ARGS...]
speedlm vllm serve MODEL --config CONFIG [--enable-idle-tuning|--no-enable-idle-tuning]
speedlm traces import PATH [--model MODEL] [--store PATH]
speedlm traces stats [--store PATH]
speedlm status [--json]
speedlm gain [--json]
speedlm --version
```

`speedlm vllm serve` supervises the child vLLM process, proxies traffic, and
captures traces. `status` and `gain` are read-only; `gain` reports measurements
only when a completed gate decision exists.

Idle tuning is an opt-in lifecycle of `vllm serve`; it is not a separate
command. Configuration can enable it persistently with `tuning_enabled`, while
the CLI flags are an explicit tri-state override. Enabling validates the
profile, Speculators checkout, training interpreter, durable artifact pointer,
and required scripts before launching vLLM.

At minimum, an EAGLE-3 profile needs portable training locations:

```json
{
  "model": "org/verifier-model",
  "profile": "my-eagle3-profile",
  "tuning_enabled": true,
  "tuning": {
    "speculators_repo": "/path/to/speculators",
    "training_python": "/path/to/training-venv/bin/python",
    "min_trace_records": 32,
    "held_out_fraction": 0.2,
    "benchmark_repeats": 5
  }
}
```

Custom profiles live in `$SPEEDLM_HOME/profiles`; production composition is
selected by speculative method, not by model name. EAGLE-3 is the first
production-wired training method. Native MTP remains fail-closed until its
training and artifact contract is validated.

### Promotion thresholds

```json
{
  "promotion": {
    "min_accepted_length_delta": 0.05,
    "min_acceptance_delta_pp": 1.0,
    "min_throughput_delta_pct": -2.0
  }
}
```

The knobs are not symmetric, and the defaults are derived from the measured
dispersion of live gate runs rather than picked round.

`min_accepted_length_delta` is the **promotion criterion**. It measures mean
accepted length in tokens per verifier step (`1 + accepted_tokens / num_drafts`),
which is directly proportional to throughput at fixed step cost. Unlike the
acceptance rate (`(mean_accepted_length - 1) / k`), it does not divide by the
draft depth and is comparable across different `k` values. The 0.05 default is
the old 1.0 pp bar re-expressed at `k=5`, so no archived rejection can flip to
a promotion. Acceptance comes from vLLM's `spec_decode` counters over a
deterministic greedy replay of a frozen suite, so it carries no timing
component: on job 368670 the two arms produced byte-identical counters (1155
drafted, 730 accepted, delta exactly 0.0 pp). The noise floor is therefore the
counter quantum itself, making the 0.05 bar approximately a 2% relative lift in
accepted length.

`min_acceptance_delta_pp` is **recorded, not gated**. The acceptance-rate delta
is still computed and published for backward compatibility, but it carries the
draft depth in its denominator and cannot be compared across depths, so it no
longer drives promotion decisions.

`min_throughput_delta_pct` is a **regression guard**, and is negative on
purpose: it asks only that the candidate not be visibly slower. The gate
compares the **replay** statistic (`replay_per_repeat_mean`), so every number
below is replay-derived unless labelled Prometheus; the Prometheus decode rate
is recorded alongside for diagnosis but is not what the gate reads. Throughput
is timing, and timing is noisy. Across job 368670's scored repeats the
within-arm replay standard deviation was 1.80 tok/s (stock) and 0.58 tok/s
(candidate) on a ~76.7 tok/s mean, giving a 1.43% standard error on the
arm-to-arm delta at three repeats and 1.10% at the default five. A *positive*
throughput bar below the one-sided 95% minimum detectable effect (~3.0% at three
repeats, ~2.1% at five) cannot be earned on merit — it is cleared by whichever
way the noise fell. Job 368670 is the worked example: its replay delta was
+0.96% (the Prometheus delta coincidentally agreed at +0.96%), and the same data
reads −0.78% if you drop its first scored repeat. −2.0% sits ~1.8 standard
errors below zero at five repeats, so ordinary jitter does not trip it, while
the real regression this gate has already caught — job 368648's un-warmed
candidate arm at −17.5% replay (−19.2% on the Prometheus decode rate) — is
~16 standard errors past it.

Both values are configurable. Setting them to `0.0` / `0.0` reduces the gate to
"not measurably worse", which promotes on noise indefinitely; that is the exact
failure mode this gate exists to prevent, and it is not a supported production
configuration.

`benchmark_repeats` defaults to 5 (floor 3). Each extra repeat is one more
held-out suite pass per arm — roughly 8s each on job 368670's hardware, against
a ~275s benchmark phase dominated by vLLM engine restarts — so precision here
is cheap.

The managed child is launched with vLLM sleep support on a private loopback
port. Once the gateway is idle, it atomically closes admission, drains accepted
requests and pending semantic capture, sleeps vLLM with `mode=wait`, trains,
starts and benchmarks a candidate, and promotes or restores the known-good
draft. A request arriving during the cycle advances the preemption watermark,
waits at the gateway, and proceeds after serving is restored; the caller does
not need retry-specific harness behavior.

## GPU runtime stack

The target GPU-host stack is pinned as follows:

| Component | Version |
| --- | --- |
| Python | 3.12 |
| vLLM | 0.25.1+cu129 |
| Speculators | v0.6.0, installed from source |
| torch | 2.11.0 |
| CUDA | 12.9 |

Those heavy packages are deliberately not declared as SpeedLM runtime
dependencies. CUDA, torch, vLLM, and Speculators are platform-specific parts of
the GPU host image and are installed out of band. The proxy wrapper itself only
declares `fastapi`, `httpx`, and `uvicorn`.

## Project status

The repository has unit-tested production composition for configuration and
storage, streaming proxying and exact raw capture, trace normalization, atomic
admission and preemption, vLLM sleep-state reconciliation, replaceable child
supervision, capture barriers, deterministic leakage-safe train/held-out
splits, warm-start training, artifact rollback, held-out gate decisions,
durable scheduler status, diagnostics, and honest reporting.

The live H100 E2E test launches a real vLLM process and exercises the gateway
against it. It is an infrastructure test and is intentionally excluded from
CPU-only CI.

The CPU/injected lifecycle suite is not a substitute for the final literal GPU
milestone. A real adaptation run must still demonstrate: live vLLM
sleep/wake, real Speculators extraction and training, candidate restart,
held-out gate rejection or promotion, request-driven preemption, and restart
recovery. Earlier measurements came from a prototype under confounded
conditions; they are not SpeedLM benchmark results and are intentionally not
presented here.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local checks and the GPU-test
boundary.
