# Reusable Devpost answers

These are neutral defaults. Event packets should add a concrete track fit,
problem beneficiary, or sponsor integration only when the submitted build and
retained evidence support it.

Counts below use whitespace-separated words and include the visible Markdown
link text only when it is part of the answer. Re-check counts after editing.

## Names, taglines, and short summaries

### Project name

SpeedLM

### Tagline — 54 characters

Automatically speeds up local LLMs with idle GPU time.

### Elevator pitch — under 150 characters

SpeedLM automatically speeds up local LLMs with idle GPU time, without changing
the OpenAI/vLLM workflow.

### Compact description — under 300 characters

SpeedLM makes local LLM serving faster with idle GPU time. It keeps data in the
operator's environment and preserves the OpenAI/vLLM workflow while it tunes a
speculative draft. A held-out gate controls promotion, and the target model
still verifies every token.

### Suggested tags

`artificial-intelligence`, `machine-learning`, `developer-tools`, `inference`,
`gpu`, `vllm`, `pytorch`, `fastapi`, `speculative-decoding`

## Word-budget summaries

### Up to 50 words

SpeedLM automatically speeds up local LLMs with otherwise idle GPU time. Teams
keep their serving data in their own environment and keep using the normal
OpenAI/vLLM workflow. SpeedLM tunes a speculative draft, gates it on held-out
traffic, and leaves the full target model authoritative for every token.

### Up to 100 words

SpeedLM helps self-hosted teams get faster LLM inference from otherwise idle
GPU time. It preserves the normal OpenAI-compatible vLLM workflow and keeps
captured serving data inside the operator-controlled environment. During a
confirmed quiet window, it trains an isolated speculative draft and tests it on
session-disjoint held-out traffic. The full target model still verifies every
token. An archived Qwen3-8B gate measured 15.0% more accepted tokens per
verifier step but rejected promotion because timing was non-stationary; a
separate retained gate promoted +13.4% accepted length.

### Up to 200 words

SpeedLM helps teams make self-hosted LLMs faster without sending their serving
data to an optimization vendor, buying separate training hardware, or changing
the normal OpenAI/vLLM client workflow. It turns quiet periods on the existing
inference GPU into training time for a workload-specific speculative draft.

After five quiet minutes by default, SpeedLM safely drains accepted work,
releases vLLM's GPU memory, and trains an isolated candidate from completed
local input/output exchanges. Whole sessions stay on one side of the
train/held-out split. A candidate is promoted only after it clears a gate
against the incumbent; otherwise SpeedLM rolls back. Requests arriving during
a preemptible stage wait while verified serving state is restored.

The full target model remains authoritative for every response: the draft only
proposes tokens for it to accept or reject. On Qwen3-8B agentic traffic, an
archived session-disjoint gate measured accepted length rising from 2.3051 to
2.6507 across 287 contexts and 8 repeats—15.0% more accepted tokens per verifier
step. It rejected promotion because wall-clock timing was non-stationary. A
separate retained 125-context, 8-repeat gate promoted a +13.4% accepted-length
gain. SpeedLM is a research preview live-tested on a pinned NVIDIA H100 stack.

### Up to 500 words

Teams self-host LLMs to keep serving traffic inside infrastructure they control,
but they give up many provider-side optimizations. Their GPUs also sit idle
between traffic bursts, while a generic speculative draft may poorly fit their
coding agents, tool calls, or other real workloads. SpeedLM closes that gap: it
automatically uses quiet GPU time to adapt local LLM inference without changing
the normal OpenAI-compatible vLLM workflow.

Clients keep sending the same requests. SpeedLM forwards them to real vLLM and
captures only completed input/output exchanges as bounded local traces. The raw
traffic stays in the operator-controlled environment rather than becoming a
submitted dataset. SpeedLM trains only the smaller speculative draft; the full
target model remains authoritative and verifies every emitted token.

After five quiet minutes by default, SpeedLM atomically closes admission and
drains requests that were already accepted. It lets vLLM release serving GPU memory,
snapshots a bounded traffic window, extracts verifier training signals, and
warm-starts an isolated EAGLE-3 candidate. Whole multi-turn sessions stay on
one side of the train/held-out boundary. The candidate cannot modify the active
draft in place: SpeedLM evaluates it against the incumbent and promotes it
atomically only after the configured acceptance and throughput safeguards pass.
Failure or preemption restores the incumbent.

Requests arriving during a preemptible stage wait at the gateway while SpeedLM
cancels work and restores a verified serving state. They are never intentionally
sent to a sleeping engine or partially trained weights. The current admission
wait is unbounded, so clients still need a timeout suitable for engine recovery.

On Qwen3-8B agentic traffic, the strongest documented session-disjoint gate
measured mean accepted length rising from 2.3051 to 2.6507 across 287 contexts
and 8 repeats: 15.0% more accepted tokens per verifier step. That means the
target accepted more useful draft work on each verification step. That archived
gate rejected promotion because its wall-clock channel was non-stationary. A
separate retained 125-context, 8-repeat gate promoted a +13.4% accepted-length
gain. Across the archived gates, wall-clock throughput varied from +9.9% to
+19.9%, so neither result is presented as a universal serving-speed claim.

The authentic demo shows real vLLM and SpeedLM server PTYs, identical content
for one recorded greedy request, a 31-minute-55-second traffic capture, verbatim
trace examples, the real idle transition, unsmoothed training metrics, and a
separate retained promotion gate. The research-preview stack has been
live-tested on NVIDIA H100 with Python 3.12, CUDA 12.9, PyTorch 2.11, vLLM
0.25.1+cu129, and Speculators v0.6.0. Broader hardware and runtime support still
needs validation.

## Standard project-story sections

### Inspiration

Teams self-host LLMs to keep serving traffic inside infrastructure they control,
but their local server does not automatically learn from its workload the way a
provider can optimize across a hosted service. The same expensive GPU often
sits idle between request bursts. We built SpeedLM so operators can turn that
otherwise unused time into faster, workload-specific local inference without
changing the OpenAI/vLLM workflow or handing responses to unverified weights.

### What it does

SpeedLM automatically turns confirmed idle GPU windows into training time for a
local speculative draft. Existing OpenAI-compatible clients keep talking to
vLLM normally, and captured serving data stays in the operator-controlled
environment. SpeedLM drains accepted work, releases serving memory, trains an
isolated candidate, and tests it on session-disjoint held-out traffic. Only a
passing candidate is promoted; failure or preemption restores the incumbent.
The target model remains authoritative and verifies every emitted token.

### How we built it

The Python package separates serving, data, training, and promotion boundaries.
FastAPI and httpx preserve streaming behavior; a supervisor owns the private
vLLM child lifecycle; exact and normalized trace stores keep paired exchanges;
isolated PyTorch and Speculators workers train EAGLE-3 candidates; and the gate
compares immutable stock and candidate artifacts on a frozen held-out suite.
Atomic admission, process-group cleanup, health checks, and durable state make
recovery part of the serving contract.

### Challenges

The difficult part was protecting serving while another process used the same
GPU. Admission and the idle watermark had to be atomic, worker groups had to
terminate cleanly, and partially trained checkpoints could never reach live
requests. Multi-turn trajectories also made row-level random splitting unsafe,
so the evaluation boundary was changed to hold out whole sessions. Finally,
shared-node throughput drift forced us to separate a stable acceptance metric
from a noisy wall-clock channel.

### Accomplishments

- Operators keep the familiar OpenAI-compatible vLLM request workflow.
- Completed serving traffic stays in the operator-controlled environment and
  becomes local training data rather than a submitted dataset.
- After one-time configuration, idle tuning is enabled with one serving flag.
- Idle GPU time becomes training time without exposing live requests to a
  sleeping engine or partially trained draft.
- The target model remains authoritative, and candidate drafts stay isolated
  until a session-disjoint held-out gate permits promotion.
- An archived gate measured 15.0% more accepted tokens per verifier step but
  rejected promotion because timing was non-stationary; a separate retained
  gate promoted a +13.4% accepted-length gain.

### What we learned

Adaptive inference is a systems-safety problem as much as a training problem.
Keeping the verifier authoritative makes every new draft a disposable,
measurable hypothesis. We also learned that accepted tokens per verifier step
and wall-clock throughput need separate evidence: a stable improvement in the
first can coexist with a noisy shared-host measurement in the second.

### What's next

- Add bounded admission queues and server-side wait deadlines.
- Validate more GPU, model, and runtime combinations.
- Publish a pinned CUDA container and tagged release.
- Exercise repeated unattended promotion cycles and drift-triggered rollback.
- Publish a sanitized, independently reproducible evidence bundle.

## Built With

### One-line field

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Hugging Face models, CUDA 12.9, NVIDIA H100, SLURM, pytest, Ruff, mypy,
Remotion, OpenAI Codex

### Chip-style field

`python` · `fastapi` · `httpx` · `uvicorn` · `vllm` · `pytorch` ·
`speculators` · `eagle-3` · `hugging-face` · `cuda` · `nvidia-h100` ·
`slurm` · `pytest` · `ruff` · `mypy` · `remotion` · `openai-codex`

## AI-use disclosure

### Compact

OpenAI Codex assisted with coding, tests, documentation, code review, and video
editing. The author reviewed the work and remains responsible for the code and
claims. Metrics come from executed GPU runs and retained artifacts, not
generated terminal output.

### Detailed

OpenAI Codex was used as a coding, testing, documentation, code-review, and
video-editing assistant. The author reviewed the generated suggestions and
remains responsible for the submitted implementation and claims. The demo's
terminal regions come from real recorded PTYs, and its charts and metrics are
derived from retained run artifacts. No synthetic terminal transcript is
presented as an executed command. If narration is later generated from an
authorized voice sample, disclose that separately in the event packet.

## Pre-existing-project disclosure

### Short

SpeedLM was not represented as starting at this event. Its earliest recorded
repository commit is July 25, 2026; only work listed in this submission's
event-specific change log is claimed as new during the event.

### Detailed

SpeedLM has repository history beginning July 25, 2026. That timestamp does not
by itself establish when all material work began, so the entrant separately
attests the actual project timeline in the event worksheet. The baseline is
disclosed as pre-existing whenever the event opened later. Event-period judging
should consider only the dated additions listed in the packet's change log.
No sponsor integration, deployment, feature, or benchmark is claimed as new
unless it was actually implemented during the eligible period and supported by
retained evidence.

### Event-specific additions block

> Pre-existing baseline: SpeedLM serving gateway, trace capture, idle
> controller, draft training, and promotion gate, as applicable on the event's
> opening date. New work submitted for this event: `TODO(owner: list exact
> commits/features, dates, and retained proof)`. If there was no qualifying new
> work, write “none” and do not enter a new-work-only track.

## Project links

- Repository: `https://github.com/RyanKim17920/speedlm`
- README/setup: `https://github.com/RyanKim17920/speedlm#readme`
- `TODO(owner): public YouTube/Vimeo demo URL accepted by the event`
- `TODO(owner): live deployment URL, only if one actually exists`

Do not submit the first two links until they work in a logged-out browser.
