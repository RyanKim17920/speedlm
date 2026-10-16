# Zero Origin — SpeedLM conditional offline draft

## Status and urgency

- **Status:** `excluded-from-direct-campaign`, because country and exact team
  eligibility are unverified; the project fit itself is conditional on
  registration, and Round-1 presentation requirements
- **Urgency:** use the earliest official time: **September 26, 2026 at 12:00 AM
  IST / September 25 at 18:30 UTC**.
- **Online Devpost draft saved:** **no**
- **Track:** Open Innovation within AI / Cloud Computing

## Official record and date conflict

- Overview: https://zero-origin.devpost.com/
- Rules: https://zero-origin.devpost.com/rules
- Schedule: https://zero-origin.devpost.com/details/dates
- Direct submission manager:
  https://devpost.com/submit-to/30524-zero-origin/manage/submissions
- Devpost schedule: Round 1 opens August 27 and closes September 26 at **12:00
  AM IST**.
- **Conflict:** the current overview header says September 26 at **11:00 PM
  IST**, 23 hours later. Use the schedule's midnight deadline unless the
  organizer corrects it in writing.
- Eligibility: students age 13+, exact team of 2–4, one team per participant,
  and only the countries/territories allowed by the eligibility panel.
- Additional registration: the overview requires the organizer's separate
  registration form and community updates.
- Round 1: problem, solution, PPT, and pitch video. Judges assess problem
  validation, affected users, innovation, feasibility, and potential impact.
- Round 2: shortlisted teams pay ₹50/person and build/prove a functional
  prototype.
- Existing work: rules require originality but state no explicit event-only
  build window. Existing-work treatment is unclear.

## Paste-ready Round-1 fields

### Project name

SpeedLM — Local LLM Serving That Learns From Its Workload

### Tagline (54 characters)

Automatically speeds up local LLMs with idle GPU time.

### Problem statement and affected users

Teams that self-host LLMs with vLLM keep their infrastructure and data boundary
local, but their speculative draft is usually static and generic. That draft
may poorly match coding agents, tools, or a deployment's actual traffic.
Researchers, student AI teams, and small operators need a way to adapt the
helper without sending requests across a new vendor boundary or allowing
unverified weights to control responses.

### Solution overview

SpeedLM is an OpenAI-compatible vLLM gateway that learns during confirmed idle
windows. It records completed paired exchanges, atomically closes admission,
drains accepted work, releases serving memory, and trains an isolated EAGLE-3
candidate. Whole sessions stay on one side of the train/held-out boundary. The
candidate is promoted only after comparison with the incumbent; failure or
preemption restores verified serving state. The full target model verifies
every emitted token.

### Validation and measurable improvement

The prototype has been exercised end to end on a pinned NVIDIA H100 stack. A
documented Qwen3-8B gate measured accepted length increasing from 2.3051 to
2.6507 across 287 session-disjoint contexts and eight repeats: 15.0% more
accepted tokens per verifier step. Shared-node throughput varied, so this pitch
does not claim universal speed, cost, or energy savings. The measurable result
is acceptance-side improvement plus a retained safe lifecycle demonstration.

### Innovation and feasibility

Hosted APIs cross a vendor-controlled boundary and may benefit from
provider-wide optimization under varying policies. SpeedLM fills the local
optimization gap without changing the API. Its distinctive contribution is the
full serving-safe loop: workload capture, atomic idle transition, isolated
training, session-disjoint gate, promotion, rollback, and request recovery.

### Built With

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Qwen3-8B, Hugging Face models, CUDA 12.9, NVIDIA H100, SLURM, pytest, Ruff,
mypy, Remotion, OpenAI Codex

### AI-use disclosure

OpenAI Codex assisted with coding, tests, documentation, review, and video
editing. The author reviewed the work and remains responsible. Metrics and
terminal scenes come from executed runs and retained artifacts.

### Existing-work disclosure

SpeedLM's earliest recorded repository commit is July 25, 2026, before Zero
Origin's August 27 Round-1 opening. The rules require originality but do not
explain whether pre-existing work is accepted. `TODO(owner): attest actual
material start and prior entries`; obtain organizer confirmation before final
submission.

## Exact asset/deliverable plan

- Pitch video: `docs/assets/speedlm-devpost-119s.mp4` — verified 119.061-second
  H.264/AAC cut with burned English captions; public hosting is pending.
- PPT: `REQUIRED BUILD: create the Round-1 deck; no compliant PPT exists yet`.
- Evidence images for the deck: poster plus `01-serving-proof.png`,
  `03-idle-tuning.png`, `04-training-metrics.png`, and
  `05-promotion-gate.png`.
- Repository: `https://github.com/RyanKim17920/speedlm` — include only if judge
  access is authorized and verified.
- Hosted pitch/PPT links: `TODO(owner)`.

## Final owner-only checks

- [ ] Confirm the entrant's country is in the allowed list.
- [ ] Form an exact 2–4-person eligible team; solo submission is invalid.
- [ ] Complete the separate organizer registration.
- [ ] Resolve the deadline and pre-existing-work questions with the organizer.
- [ ] Build the required PPT and verify the pitch video/link.
- [ ] Confirm ownership, timeline, prior entries, and personally accept terms.
