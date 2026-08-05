# SpeedLM

SpeedLM is a near-drop-in wrapper around `vllm serve`. It launches vLLM behind
a streaming gateway, proxies OpenAI-compatible traffic, and captures completed
requests inline. Its adaptation pipeline is designed to use idle GPU periods to
personalize a speculative draft head from that traffic, then promote the
candidate only when a held-out gate measures the required increase in mean
accepted length and the candidate stays above the throughput regression floor.

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
  resets, invalid responses, excessive output divergence relative to the
  stock-stock control, a missed mean-accepted-length threshold, or a breached
  throughput floor rejects the candidate. Only a candidate with the required
  **mean accepted length** increase and no material throughput regression is
  promoted — see
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

The knobs are not symmetric. The accepted-length threshold is the promotion
criterion, the acceptance-rate threshold is retained only as a reported legacy
field, and the throughput threshold is a regression floor.

`min_accepted_length_delta` is the **promotion criterion**. It measures mean
accepted length in tokens per verifier step (`1 + accepted_tokens / num_drafts`),
which is directly proportional to throughput at fixed step cost. Unlike the
acceptance rate (`(mean_accepted_length - 1) / k`), it does not divide by the
draft depth and is comparable across different `k` values. The 0.05 default is
the old 1.0 pp bar re-expressed at `k=5`. Acceptance comes from vLLM's
`spec_decode` counter deltas over deterministic greedy replay; it has no timing
component, but that does not make every archived counter comparison valid.
Several early runs produced identical substantive counters for both arms and
are voided in the [benchmark evidence record](docs/benchmark-evidence.md).

`min_acceptance_delta_pp` is **recorded, not gated**. The acceptance-rate delta
is still computed and published for backward compatibility, but it carries the
draft depth in its denominator and cannot be compared across depths, so it no
longer drives promotion decisions.

`min_throughput_delta_pct` is a **regression guard**, and is negative on
purpose: it asks only that the candidate not be visibly slower. The gate
compares the **replay** statistic (`replay_per_repeat_mean`), so every number
below is replay-derived unless labelled Prometheus; the Prometheus decode rate
is recorded alongside for diagnosis but is not what the gate reads. Throughput
is timing, so the decision records a standard error and uses interleaved arm
blocks to expose lifecycle effects. The first trustworthy measurements at
commit `8b72d9a` show no throughput improvement on either supported model:
Qwen is +0.48% ± 1.11% (0.43 sigma, noise), while gpt-oss is −5.57% ± 1.15%
(4.8 sigma, a regression that breaches the −2% floor). The independent gpt-oss
Prometheus decode counter agrees at −5.69%. Both candidates had already failed
the k-invariant criterion: accepted-length deltas +0.024 and −0.163 versus the
required +0.05. See the [benchmark evidence record](docs/benchmark-evidence.md)
for exact artifact paths and the voided archive.

The two gating thresholds are configurable. Setting
`min_accepted_length_delta` / `min_throughput_delta_pct` to `0.0` / `0.0`
reduces the gate to "not measurably worse", which promotes on noise
indefinitely; that is the exact failure mode this gate exists to prevent, and it
is not a supported production configuration.

`benchmark_repeats` defaults to 5 (floor 3). Each extra repeat is one more
held-out suite pass per arm. Repeats improve precision but do not repair a
confounded arm lifecycle, a non-discriminating metric scrape, or a semantically
degenerate held-out suite.

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

The CPU/injected lifecycle suite is not a substitute for literal GPU evidence.
The definitive `8b72d9a` runs reached live sleep, extraction, training,
candidate start, benchmarking, rejection, rollback, and wake. Real-run evidence
is still required for promotion, request-driven preemption, and restart
recovery. Earlier positive measurements came from a prototype under confounded
conditions and are void. The current honest result and exact artifact paths are
recorded in [Benchmark evidence](docs/benchmark-evidence.md).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local checks and the GPU-test
boundary.
