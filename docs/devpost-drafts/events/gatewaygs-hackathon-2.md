# GatewayGS Hackathon 2 — SpeedLM offline draft

## Status and urgency

- **Status:** `direct-active-with-date-warning`; the live overview/schedule is
  current, while the custom rules page contains stale dates
- **Urgency:** medium-high — use the live schedule's earlier/current deadline,
  **September 16, 2026 at 5:00 PM EDT / 21:00 UTC**.
- **Online Devpost draft saved:** **no**

## Official record

- Overview: https://gatewaygs-hackathon-2.devpost.com/
- Rules: https://gatewaygs-hackathon-2.devpost.com/rules
- Schedule: https://gatewaygs-hackathon-2.devpost.com/details/dates
- Direct submission manager:
  https://devpost.com/submit-to/30842-gatewaygs-hackathon-2/manage/submissions
- Overview/schedule: submissions September 1 at 12:00 AM EDT through September
  16 at 5:00 PM EDT.
- **Conflict:** the rules page still says March 15–April 5 without a year. That
  contradicts the event header and live schedule. Use September 16 operationally
  and ask the organizer which build period governs eligibility.
- Eligibility: students age 13+, worldwide subject to standard exclusions;
  solo or teams up to four.
- Build: working AI project solving a real-world problem. Developer tools and
  APIs are explicitly acceptable; AI must be core rather than decorative.
- Required: title, short description, problem, solution, AI-use explanation,
  demo by video/live link/screenshots, and source repository public or private
  with judge access. No video duration is stated.
- Existing work: neither current overview nor rules states a pre-existing-work
  ban. Disclose the July baseline and do not imply it was built September 1–16.

## Paste-ready fields

### Project name

SpeedLM — Faster Local LLMs From Idle GPU Time

### Tagline (54 characters)

Automatically speeds up local LLMs with idle GPU time.

### Short description

SpeedLM automatically speeds up local LLMs with idle GPU time. Teams keep their
serving data inside infrastructure they control and keep using the normal
OpenAI-compatible vLLM workflow. SpeedLM learns a workload-specific speculative
draft, promotes it only after held-out evaluation, and leaves the full target
model authoritative for every response.

### Problem statement

Teams choose local vLLM so prompts, tool calls, and outputs can stay inside
their own infrastructure. But a local server does not automatically receive
provider-wide optimization. Its costly GPU sits idle between bursts, while a
fixed speculative draft may not reflect the coding, tool-use, or chat traffic
the team actually serves.

### Solution and AI-use explanation

SpeedLM turns that otherwise unused GPU time into local training time without
changing the client workflow. Its OpenAI-compatible gateway forwards real
streaming requests to vLLM and stores completed input/output pairs as bounded
traces inside the operator-controlled environment. After five quiet minutes by
default it safely drains accepted work, releases serving memory, and trains an
isolated EAGLE-3 speculative draft. A session-disjoint held-out suite compares
candidate and incumbent before promotion; failure or preemption restores
verified serving state.

AI is the core system, not an added chatbot: SpeedLM trains a model component,
uses the target model's verifier signals, and evaluates how many draft proposals
the target accepts. The target verifies every emitted token.

### Full story

SpeedLM gives local-AI builders a benefit hosted optimization normally does not:
their inference stack adapts to their own workload while their serving data
stays in their environment. They do not need a new client API or a separate
training GPU. SpeedLM automatically uses quiet periods on the existing vLLM
machine, while the full target model keeps final authority over every token.

On Qwen3-8B agentic traffic, a documented gate measured accepted length rising
from 2.3051 to 2.6507 across 287 session-disjoint contexts and eight repeats:
15.0% more accepted tokens per verifier step. That archived gate rejected
promotion because its wall-clock channel was non-stationary. A separate retained
125-context, eight-repeat gate promoted a +13.4% accepted-length gain. Neither
result establishes universal serving speed, energy savings, or cost reductions.

Under the hood, SpeedLM joins request admission, local trace capture, idle
detection, vLLM sleep/wake, isolated training, leakage-resistant evaluation,
rollback, and recovery. The authentic demo shows real vLLM and SpeedLM
commands, matching content for one recorded greedy request, live traffic,
verbatim trace examples, the idle transition, unsmoothed training metrics, and
a separate retained promotion gate. Next we plan bounded request waits, wider
hardware validation, a pinned container, and repeated unattended promotion
testing.

### Built With

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Qwen3-8B, Hugging Face models, CUDA 12.9, NVIDIA H100, SLURM, pytest, Ruff,
mypy, Remotion, OpenAI Codex

### AI-use disclosure

OpenAI Codex assisted with coding, tests, documentation, code review, and video
editing. The author reviewed the output and remains responsible for the code
and claims. Metrics and terminal scenes come from executed GPU runs and
retained artifacts, not generated console text.

### Existing-work disclosure

SpeedLM predates GatewayGS's September submission opening; its earliest
recorded repository commit is July 25, 2026. The posted pages do not prohibit
existing projects, but their March/April rules dates conflict with the live
September schedule. `TODO(owner): attest actual project timing and prior
entries`; seek organizer clarification before representing any work as built
during this event.

## Exact asset plan

- Demo: `docs/assets/speedlm-devpost-59s.mp4` — verified 59.051-second H.264/AAC
  cut with burned English captions. Public hosting remains pending; the event
  states no cap.
- Screenshots: `01-serving-proof.png`, `03-idle-tuning.png`, and
  `05-promotion-gate.png`.
- Repository: `https://github.com/RyanKim17920/speedlm` — public, MIT, and
  anonymously verified at commit `3d1848e`.
- Hosted demo URL: `TODO(owner)`.

## Final owner-only checks

- [ ] Get organizer clarification on the March/April versus September dates.
- [ ] Confirm jurisdiction, school/enrollment details, and team roster.
- [ ] Confirm actual timeline, prior entries, authorship, and asset rights.
- [ ] Verify one permitted demo path; repository access is already public.
- [ ] Review and personally accept the event attestations.
