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
  streaming reverse proxy. Successful chat and completion responses are
  observed while their bytes continue to the client.
- **Traces** — completed request/response pairs are normalized and appended to
  a persistent, bounded JSONL trace store. Capture is part of the gateway
  request path; there is no separate recording harness. Existing OpenAI-style
  JSONL can also be imported to bootstrap the store.
- **Tuner** — the idle-cycle logic detects a quiet gateway, quiesces serving,
  snapshots traces, prepares training rows, extracts verifier hidden states,
  warm-starts Speculators training from the public draft head, and materializes
  an immutable candidate. New traffic preempts the cycle and serving is
  restored on failure.
- **Gate** — a frozen held-out suite is replayed against the stock and candidate
  drafts. The decision is fail-closed: unavailable acceptance metrics, counter
  resets, invalid responses, output mismatches, or either improvement threshold
  being missed all reject the candidate. Only a candidate with improved
  acceptance and throughput is promoted.
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
speedlm traces import PATH [--model MODEL] [--store PATH]
speedlm traces stats [--store PATH]
speedlm status [--json]
speedlm gain [--json]
speedlm --version
```

`speedlm vllm serve` supervises the child vLLM process, proxies traffic, and
captures traces. `status` and `gain` are read-only; `gain` reports measurements
only when a completed gate decision exists.

The following public command is currently stubbed:

```text
speedlm doctor
```

It prints a “not yet implemented” message and returns exit code 2. This is a
CLI wiring gap: `src/speedlm/doctor.py` is fully implemented and unit tested,
but the command does not call it yet.

The tuner and gate are implemented as internal components, but there is no
public `speedlm tune` command and the current `vllm serve` handler does not
schedule an idle adaptation cycle. Do not interpret the presence of those
modules as a claim that unattended adaptation is already available through the
CLI.

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

The repository has unit-tested logic for configuration and storage, streaming
proxying and inline capture, trace normalization, idle detection and
preemption, warm-start training contracts, artifact rollback, held-out gate
decisions, diagnostics, and honest reporting.

The live H100 E2E test launches a real vLLM process and exercises the gateway
against it. It is an infrastructure test and is intentionally excluded from
CPU-only CI.

There is not yet a validated end-to-end adaptation run demonstrating an
acceptance gain from a SpeedLM-personalized draft. Earlier measurements came
from a prototype under confounded conditions; they are not SpeedLM benchmark
results and are intentionally not presented here.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the local checks and the GPU-test
boundary.
