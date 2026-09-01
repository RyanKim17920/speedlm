# SpeedLM — Demo Runbook

## What SpeedLM is, in three sentences

SpeedLM automatically speeds up local LLM serving with GPU time that would
otherwise sit idle. You keep the same OpenAI-compatible vLLM workflow and local
serving boundary; SpeedLM learns from the traffic your deployment already sees
and improves the speculative helper behind it.

Under the hood, SpeedLM launches `vllm serve` behind a streaming gateway,
captures completed requests as it forwards bytes to the client, and fine-tunes
an **existing public speculative draft head** (for example,
`RedHatAI/gpt-oss-20b-speculator.eagle3`) during confirmed idle periods. It
never trains a draft from scratch and refuses a missing `from_pretrained` base.

A frozen held-out gate then replays stock versus candidate and promotes the
candidate **only** if the configured acceptance and throughput safeguards
pass; otherwise it rolls back. The full model still verifies every proposed
token.

After the one-time Speculators and profile configuration described below, the
recurring user-facing flow is two commands:

```bash
speedlm vllm serve MODEL --enable-idle-tuning
speedlm gain
```

`speedlm traces import corpus.jsonl` is an optional bootstrap when you already
have compatible traces; live serving captures completed traffic automatically.

There is no separate harness. No manual prepare / extract-hidden-states / train / stitch
scripts. That is the point of the demo.

---

## Pre-flight checklist

Tick every box before you present.

| # | Must be true | How to check |
|---|---|---|
| 1 | `speedlm` is on PATH from the project venv | `.venv/bin/speedlm --version` -> `speedlm 0.1.0` |
| 2 | You have decided your `SPEEDLM_HOME` and it is **not** your real one | `export SPEEDLM_HOME=/tmp/speedlm-demo/home1` |
| 3 | You have a bootstrap corpus JSONL on disk | `wc -l corpus.jsonl` |
| 4 | For GPU steps: you are **inside a SLURM allocation**, not on the login node | `nvidia-smi` succeeds |
| 5 | For GPU steps: the pinned vLLM venv is on PATH | `command -v vllm` resolves to the preflight venv |
| 6 | A **real** `decision.json` from a pre-run tuning cycle exists, if you plan to show `gain` with numbers | `ls $SPEEDLM_HOME/runs/*/decision.json` |
| 7 | Terminal font is large; `speedlm` output is the deliverable | — |
| 8 | You have read "What we are NOT claiming" below and will say it out loud | — |

Two environment knobs matter for the live-demo failure mode where vLLM looks hung:

- `SPEEDLM_STARTUP_TIMEOUT_SECONDS` (default `900.0`)
- `SPEEDLM_STARTUP_STALL_SECONDS` (default `600.0`)

`SPEEDLM_HOME` (default `~/.speedlm`) can also be overridden per-invocation with the global
`--home` flag.

---

## The happy path

Everything in steps 1–5 below was **actually executed on this login node** and the output is
pasted verbatim. Steps 6–8 require a GPU and are marked ILLUSTRATIVE.

Set up a throwaway home first so nothing you demo touches real state:

```bash
export SPEEDLM_HOME=/tmp/speedlm-demo/home1
mkdir -p "$SPEEDLM_HOME"
alias speedlm=/admin/home/ryan.kim/speedlm-fr/.venv/bin/speedlm
```

### 1. `speedlm --version`

```bash
speedlm --version
```

**Real output:**

```
speedlm 0.1.0
```

### 2. `speedlm doctor` — read-only diagnosis, fail-closed

```bash
speedlm doctor
```

**Real output (from this login node, which has no GPU):**

```
SpeedLM doctor
[PASS] python: Python 3.12.11 is within >=3.12,<3.13
[FAIL] gpu: nvidia-smi is present but cannot communicate with a usable driver: NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
[SKIP] cuda: CUDA detection skipped because no usable NVIDIA driver was found
[FAIL] packages: Pinned runtime package check failed: vllm is missing; torch is missing; speculators is missing
[PASS] disk: 845.9 GiB free at /tmp/speedlm-demo/home1
[PASS] memory: 36.2 GiB available of 125.8 GiB system RAM
[PASS] model_pair: Profile 'gpt-oss-20b-eagle3' is coherent; eagle3 draft uses a separate model reference
Overall: FAIL
Execution mode: unavailable — No usable NVIDIA GPU is available for auto-tuning
```

Exit code: **1** (doctor exits 1 on overall FAIL).

> **This FAIL is expected and is a known-good demo output.** The login node genuinely has no
> GPU driver. Do not treat it as a bug and do not hide it. `doctor` also emits `--json`.

**What to say:** "Seven read-only checks, an overall verdict, and an *execution mode*. The
execution mode is what the tuner consults before it will do anything — `idle`, `colocated`,
or `unavailable`. On this box it is `unavailable`, so launching with
`--enable-idle-tuning` would fail closed before tuned serving is exposed and say why. Run
without the flag when you only want normal proxy serving on a host that cannot tune."

### 3. `speedlm status` — before any traffic

```bash
speedlm status
```

**Real output:**

```
SpeedLM status
home         : /tmp/speedlm-demo/home1
gateway      : no gateway is running (no runtime record at /tmp/speedlm-demo/home1/gateway.json)
active draft : no active draft (nothing has been promoted yet)
traces       : 0 record(s), 0 token(s)
  age        : n/a (trace buffer is empty)
tuner        : no tuner state (the tuner has not run; /tmp/speedlm-demo/home1/runs/state.json does not exist)
verifier     : openai/gpt-oss-20b
draft        : RedHatAI/gpt-oss-20b-speculator.eagle3
  source     : built-in default pair (no /tmp/speedlm-demo/home1/config.json)
```

**What to say:** "Every empty field explains *why* it is empty and *which file* it looked for.
Nothing is invented. Note the model pair: a public verifier and a public draft head. SpeedLM
warm-starts from that draft; it does not build one."

### 4. `speedlm traces import` — bootstrap the store

Traces are normally captured automatically by the gateway. `import` exists only to bootstrap
from logs you already have. Format is auto-detected across five shapes
(`request-response`, `proxy-capture`, `openai-response`, `internal`, `bare-conversation`).

```bash
speedlm traces import /tmp/speedlm-demo/corpus.jsonl --model openai/gpt-oss-20b
```

**Real output (61-row corpus):**

```
imported 61 record(s) [internal: 61]
```

Malformed lines are rejected individually and reported with line numbers; the command exits 1
only if *nothing* was accepted.

**What to say:** "One command. No schema to conform to — it sniffs the envelope. And it is
strictly optional: if you just run the gateway, capture happens on the request path."

### 5. `speedlm traces stats` and `speedlm status` again

```bash
speedlm traces stats
```

**Real output:**

```
count    : 61
tokens   : 2590
measured : 2590
estimated: 0
oldest   : 2025-07-24T23:33:20+00:00
newest   : 2025-07-25T00:10:20+00:00
store    : /tmp/speedlm-demo/home1/traces/traces.jsonl
```

```bash
speedlm status
```

**Real output (only the changed lines shown here; the rest is identical to step 3):**

```
traces       : 61 record(s), 2590 token(s)
  oldest     : 1753400000 (365.2d ago)
  newest     : 1753402220 (365.2d ago)
```

**What to say:** "`measured` vs `estimated` is a real distinction — token counts that came from
the provider's `usage` block are separated from ones SpeedLM had to estimate. The trace buffer
is bounded (default 8M tokens / 14 days), so this never grows without limit."

### 6. `speedlm vllm serve --enable-idle-tuning` — ILLUSTRATIVE (needs a GPU)

```bash
speedlm vllm serve openai/gpt-oss-20b \
  --host 127.0.0.1 --port 8100 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --enable-idle-tuning
```

Anything SpeedLM does not recognise is passed through to the child `vllm serve`. `--host` and
`--port` in the passthrough are stripped: the child is **forced onto loopback on a reserved
ephemeral port**, and only the gateway is reachable.

**ILLUSTRATIVE expected output** — the first line is the real format string; the vLLM banner
below is copied from an actual run recorded in `log_artifacts/stage1-qwen/gateway-and-vllm.log`
on node `n-4` (that run used `Qwen/Qwen3.5-2B`; ports and pids will differ):

```
[speedlm] launching vLLM on 127.0.0.1:54761; gateway listening on 127.0.0.1:8100
(APIServer pid=...) INFO ... [api_utils.py:339]
(APIServer pid=...) INFO ... [api_utils.py:339]  ▄▄ ▄█ █     █     █ ▀▄▀ █  version 0.25.1
(APIServer pid=...) INFO ... [api_utils.py:273] non-default args: {..., 'host': '127.0.0.1', 'port': 54761, ...}
(EngineCore pid=...) INFO ... [gpu_worker.py:538] Available KV cache memory: 59.08 GiB
(EngineCore pid=...) INFO ... [core.py:344] init engine (profile, create kv cache, warmup model) took 460.42 s
```

**That 460-second number is measured, not hypothetical** — and it was with `--enforce-eager`,
i.e. with `torch.compile` and CUDA graphs *disabled*. Read the failure-modes section before you
put this on a projector.

Once up, the gateway is a plain OpenAI-compatible endpoint:

```bash
curl -s http://127.0.0.1:8100/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"hi"}]}'
```

**What to say:** "The client change is the base URL. That's it. Every successful chat and
completion response is normalized and appended to the trace store while its bytes are still
streaming to the caller — capture is on the request path, not a sidecar."

### 7. The idle cycle — ILLUSTRATIVE, do not run live

With `--enable-idle-tuning`, once the gateway has been quiet for `idle_threshold_seconds`
(default **300 s**) and the store holds at least **256** records, the tuner walks this state
machine:

```
READY -> QUIESCING -> SLEEPING -> EXTRACTING -> TRAINING
      -> CANDIDATE_STARTING -> BENCHMARKING -> PROMOTING -> WAKING -> READY
```

Any failure diverts to `ROLLING_BACK -> WAKING -> READY`. New traffic preempts
stages that are safe to cancel; at non-interruptible boundaries it waits until
a verified serving state is restored. Default stage timeouts: quiesce 30 s,
sleep 120 s, candidate start 600 s, benchmark 1800 s, restore 600 s.

Watch it with `speedlm status`, which reads `$SPEEDLM_HOME/runs/state.json`.

**What to say:** "Prepare rows, extract verifier hidden states, warm-start Speculators training
from the *public* draft head, materialize an immutable candidate, restart the child vLLM on it,
benchmark both arms. You are not going to watch that happen in a demo slot — see below."

### 8. `speedlm gain` — the honest reporter

`gain` reads a real persisted `decision.json` under `$SPEEDLM_HOME/runs/`. It never invents a
number.

**Real output, no gate has ever run** (executed on this box):

```
SpeedLM gain
measurement       : no_gate_run
No gate has ever run: there is no completed benchmark in /tmp/speedlm-demo/home1/runs, so there is no measured gain to report.
```

Older revisions of this runbook contained a 61-row acceptance-rate example
from a retired decision schema. Do not present that output as the current gate.
Run `speedlm gain` against the actual `decision.json` you intend to show and
retain that artifact with the demo.

The current default gate uses accepted length as the draft-efficiency
criterion and throughput as a regression guard:

- `min_accepted_length_delta = 0.05`
- `min_throughput_delta_pct = -2.0`

Those thresholds are deliberately asymmetric: the candidate must improve
accepted length, while the noisier wall-clock channel must remain inside the
configured regression budget. Validity, repeat-count, output-integrity, and
stationarity checks still fail closed.

For the current video, keep the two real results distinct. The archived 287×8
gate measured **+15.0% accepted length** but did not promote because its timing
channel was non-stationary. A separate retained 125×8 gate measured **+13.4%**
and did promote. That distinction demonstrates the user benefit of gating:
SpeedLM does not deploy every promising training run.

---

## Demoing without localhost

The tension is real: this is a hackathon where a laptop-local service is not an acceptable
demo target, and SpeedLM is inherently a local tool that wraps a local `vllm serve`. Here are
four options, ranked, with honest tradeoffs.

### Option 1 — Terminal-first framing (recommended default)

**What the audience sees:** a shared terminal running `speedlm doctor`, `status`,
`traces import`, `traces stats`, `gain`. Screen share, or an `asciinema` recording played back
live if you want determinism.

**Setup cost:** essentially zero.
**Risk of failing live:** lowest of the four.

**Argue this, don't apologize for it.** SpeedLM's product surface *is* the CLI. There is no web
app, and adding one would be demo theater. A local inference accelerator legitimately has no
URL — the thing being accelerated is somebody else's inference server, and SpeedLM's deliverable
is the *decision* it prints. The gate verdict in step 8 is the product. That renders in a
terminal.

If a judge pushes back, the honest answer is: "the endpoint SpeedLM exposes is a standard
OpenAI-compatible URL and I can forward it in ten seconds (Option 3) — but the URL isn't the
interesting artifact, the rejection is."

### Option 2 — Real GPU, remote host (this is what will actually happen)

The demo is already remote: it cannot run on a laptop. It runs on an H100 in SLURM partition
`n`. The login node (`login-1.sophont-n.cgen`) has **no GPU** — `nvidia-smi` cannot reach a
driver, which is exactly what step 2's `doctor` FAIL shows. All GPU work must be inside an
allocation.

```bash
# From your machine
ssh login-1.sophont-n.cgen

# Grab an interactive H100. Always an explicit --time. Never pin --nodelist.
srun --partition=n --gres=gpu:1 --time=02:00:00 --cpus-per-task=16 --pty bash

# Inside the allocation
nvidia-smi                          # must succeed here
export PATH=/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin:$PATH
export SPEEDLM_HOME=/tmp/speedlm-demo/home1
/admin/home/ryan.kim/speedlm-fr/.venv/bin/speedlm doctor
```

**What the audience sees:** the same terminal, but `doctor` now goes green and reports an
execution mode other than `unavailable`.

**Setup cost:** one interactive allocation, obtained before you present.
**Risk of failing live:** moderate — the queue may make you wait. **Get the allocation before
the demo starts and keep it warm.** Do not `scancel` or otherwise disturb job `363643`
(`qwen-serve-low-pri`, node `n-4`).

### Option 3 — If a reachable URL is genuinely required

The gateway is an ordinary OpenAI-compatible HTTP server, so it *can* be exposed.

**Safest: SSH port-forward.** Nothing is published; only you can reach it.

```bash
# Forward the cluster-side gateway to your own laptop
ssh -N -L 8100:127.0.0.1:8100 login-1.sophont-n.cgen
# then hit http://127.0.0.1:8100/v1/chat/completions locally
```

If the gateway is on a compute node rather than the login node, chain the hop or use
`ssh -J`. A remote-forward (`ssh -R`) works in the other direction if you need the cluster to
reach something of yours.

**Public tunnel.** I checked this machine:

| Tool | Present? |
|---|---|
| `cloudflared` | **yes** — `/usr/local/bin/cloudflared` |
| `ngrok` | no |
| `tailscale` | no |
| `localtunnel` / `lt` | no |
| `bore`, `frp`, `socat` | no |
| `ssh` | yes — `/usr/bin/ssh` |

So `cloudflared tunnel --url http://127.0.0.1:8100` is the only public-tunnel option actually
available here. Verify it still resolves at demo time with `command -v cloudflared`.

**Warnings, state these out loud if you do this:**

- You are publishing an **unauthenticated LLM inference endpoint** on someone else's GPU
  allocation. Anyone with the URL can spend your compute.
- SpeedLM **captures traffic to disk** by design. Every request that arrives through a public
  tunnel lands in `$SPEEDLM_HOME/traces/traces.jsonl`. Do not point it at anything sensitive,
  and do not paste real data through it during a public demo.
- Tear the tunnel down the moment the demo ends.

**What the audience sees:** a clickable URL that answers a `curl`.
**Setup cost:** low for `ssh -L`, low-but-fragile for `cloudflared`.
**Risk of failing live:** SSH forward is reliable; a public tunnel adds a network dependency and
a cold-start delay right when you can least afford one.

### Option 4 — Pre-run the cycle, live-narrate the result (this is the primary plan)

Not a fallback. A full tuning cycle is two vLLM lifecycles plus training and a benchmark —
tens of minutes, with a *measured* 460 s just to init the engine once. Nobody should watch that.

1. **Before the demo**, in a SLURM allocation, run the real cycle to completion against your
   corpus. Let it write a genuine `$SPEEDLM_HOME/runs/<run-id>/decision.json`.
2. **During the demo**, run `speedlm gain` live. It reads that real file and prints the real
   verdict, with the real `source` path and `source mtime` on screen.

**What the audience sees:** live command execution producing real measured numbers — and the
`source` / `source mtime` lines prove it came from a file on disk, not from a slide.
**Setup cost:** one pre-run, done hours earlier.
**Risk of failing live:** near zero. `gain` is read-only and CPU-only.

Combine 1 + 2 + 4: shared terminal, inside an allocation, `gain` reading a real decision. Add 3
only if a judge insists on a URL.

---

## Failure modes and recovery

### vLLM is slow to start and looks like a hang

**This is the single most likely way the demo goes wrong.** In this repo's own recorded run,
`init engine (profile, create kv cache, warmup model) took 460.42 s` — 7 minutes and 40
seconds — *with `--enforce-eager`, so with `torch.compile` and CUDA graphs already disabled*.
A cold JIT/compile path without `--enforce-eager` can be substantially longer. There is no
progress bar. It looks exactly like a hang.

- **Do not Ctrl-C.** SpeedLM forwards signals to the child; interrupting mid-warmup means you
  start the 460 s over.
- **Pre-warm.** Start the gateway before you walk on stage and leave it running. The HF cache
  and compile cache make the second start much faster.
- Use `--enforce-eager` and a small `--max-model-len` (e.g. 4096) for demo purposes.
- If you must raise the ceiling: `SPEEDLM_STARTUP_TIMEOUT_SECONDS` (default 900) and
  `SPEEDLM_STARTUP_STALL_SECONDS` (default 600). The stall timer is what fires when output goes
  quiet; raise it before the total timeout.
- Confirm liveness from another terminal rather than staring at the log:
  `curl -s http://127.0.0.1:8100/health`.

### The GPU is unavailable

`doctor` will report `[FAIL] gpu` and `Execution mode: unavailable`, exit 1. If you started the
gateway with `--enable-idle-tuning`, the tuner refuses to construct itself and logs
`idle tuner refused: doctor reports execution mode unavailable — ...`; **serving continues
normally**. Tuning is strictly optional and never blocks the proxy.

- On the login node this is correct behavior, not a fault. Show it deliberately (step 2).
- To fix: get into a SLURM allocation (Option 2). `nvidia-smi` must succeed *inside* it.
- `[FAIL] packages` on the login node is likewise expected — vLLM/torch/speculators live in the
  pinned GPU-host venv, not the project venv. Put that venv on `PATH` inside the allocation.

### The gate rejects

**Expect this, and lean into it.** On the 61-row corpus the measured result was +0.047 pp
acceptance and −13.7% throughput. `gain` will print `verdict : reject`,
`reason : throughput_below_threshold`.

Do not scramble. Say: "That is the system working. The candidate was worse and it was not
shipped. `speedlm status` still shows `active draft : no active draft` — nothing was promoted,
nothing rolled back badly, serving was never degraded."

If you want a run that *could* promote, the honest levers are: a much larger and more
homogeneous corpus, and — only if you disclose it — lowering
`min_acceptance_delta_pp` / `min_throughput_delta_pct` in `config.json`. **Lowering the gate to
manufacture a promotion for a demo is the exact failure the prototype exhibited. Don't.**

### Other failure paths worth knowing

- `no_gate_run` from `gain`: no `decision.json` exists in `$SPEEDLM_HOME/runs`. Check that you
  exported the same `SPEEDLM_HOME` you pre-ran under.
- `unreadable` from `gain`: a `decision.json` exists but is malformed. It says so; it does not
  guess.
- `traces import` exits 1 with a rejection list if every line was malformed. Rejections are
  per-line with line numbers.
- Killed mid-cycle: the tuner's rollback path (`ROLLING_BACK -> WAKING -> READY`) restores
  serving. The gateway runtime record (`$SPEEDLM_HOME/gateway.json`) is removed after the child
  is reaped, so a stale `status` should not survive a clean shutdown.

---

## What we are NOT claiming

Read this section aloud. It is the credibility of the whole demo.

1. **No validated end-to-end adaptation gain has been demonstrated.** Not once. There is no run
   in which a SpeedLM-personalized draft beat the stock draft on both metrics under a clean
   gate.
2. **The one measurement we have is a regression**: +0.047 pp acceptance, **−13.7% throughput**,
   on a 61-row corpus. 61 rows is far too small to expect adaptation to work. We are not
   presenting this as a benchmark result; we are presenting it as evidence that the gate catches
   regressions.
3. **Earlier prototype numbers are not SpeedLM results.** They came from confounded conditions
   and a gate that reported `all_compatibility_gates_passed: true` for the same checkpoint the
   current gate rejects. We are deliberately not showing them.
4. **We are not claiming the idle cycle has run unattended to completion in production.** The
   stages are implemented and unit-tested; the live H100 E2E test exercises the gateway against
   a real vLLM. That is not the same as a validated autonomous adaptation loop.
5. **We do not train draft models from scratch.** Every run warm-starts from an existing public
   draft head. If the base is missing, the tuner refuses.
6. **The gate's job is to say no.** A system that only ever promotes is a system with a broken
   gate. Ours has said no exactly once, correctly.

What we *are* claiming: the ergonomics collapse from a multi-script pipeline to three commands;
capture is inline in the serving path with no separate harness; and every number the CLI prints
is traceable to a file on disk, with the path and mtime shown.

---

## Appendix A — SLURM commands for the GPU parts

Rules: partition `n`, nodes `n-[1-8]`, `--gres=gpu:1`, **always an explicit `--time`**, and
**never pin `--nodelist`** — pinning has previously queued a job behind one busy node while
others sat idle. Do not disturb job `363643` (`qwen-serve-low-pri`, `n-4`).

Check the partition:

```bash
sinfo -p n -o "%P %a %l %D %t %N"
squeue -u "$USER" -o "%.10i %.20j %.10P %.8T %.10M %R"
```

Interactive session (best for a live demo — you keep a shell):

```bash
srun --partition=n --gres=gpu:1 --time=02:00:00 --cpus-per-task=16 --pty bash
```

Batch job for the pre-run tuning cycle (Option 4):

```bash
cat > /tmp/speedlm-demo/cycle.sbatch <<'EOF'
#!/bin/bash
#SBATCH --job-name=speedlm-demo-cycle
#SBATCH --partition=n
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --output=/tmp/speedlm-demo/cycle-%j.log

set -euo pipefail
export PATH=/admin/home/ryan.kim/speedlm/.preflight/venvs/vllm/bin:$PATH
export SPEEDLM_HOME=/tmp/speedlm-demo/home1
SPEEDLM=/admin/home/ryan.kim/speedlm-fr/.venv/bin/speedlm

nvidia-smi
"$SPEEDLM" doctor || true          # informational; exits 1 on FAIL
"$SPEEDLM" traces stats

"$SPEEDLM" vllm serve openai/gpt-oss-20b \
  --host 127.0.0.1 --port 8100 \
  --max-model-len 4096 --gpu-memory-utilization 0.85 --enforce-eager \
  --enable-idle-tuning
EOF

sbatch /tmp/speedlm-demo/cycle.sbatch
```

Then, from anywhere with the same `SPEEDLM_HOME` visible:

```bash
watch -n 30 'speedlm status'       # follow the tuner state machine
speedlm gain                        # once runs/<id>/decision.json exists
```

Tear down:

```bash
squeue -u "$USER" -n speedlm-demo-cycle      # find YOUR job id
scancel <your-job-id>                        # never 363643
```

## Appendix B — Full command inventory

Every command that actually exists in the CLI (verified against `speedlm --help` and each
subcommand's `--help`):

```
speedlm [--version] [--home HOME] {vllm,traces,status,gain,doctor}

speedlm vllm serve MODEL [--host HOST] [--port PORT] [--enable-idle-tuning] [VLLM_ARGS...]
speedlm traces import PATH [--model MODEL] [--store STORE]
speedlm traces stats [--store STORE]
speedlm status [--json]
speedlm gain [--json]
speedlm doctor [--json]
```

There is deliberately **no** `speedlm tune`, `speedlm prepare`, `speedlm extract`,
`speedlm train`, `speedlm promote`, or `speedlm bench`. If someone asks where the training
command is, that is the demo landing: there isn't one, because you don't run it — the gateway
does, when the GPU is idle.

### Keep this inventory aligned with the CLI

The README and this demo guide now describe the implemented `doctor` command
and `--enable-idle-tuning` path. Before a live demo, compare both documents with
`speedlm --help` and `speedlm vllm serve --help` so future CLI changes do not
turn documentation into a promise the executable no longer keeps.
