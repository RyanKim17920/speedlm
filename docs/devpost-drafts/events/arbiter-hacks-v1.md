# Arbiter Hacks V1 — SpeedLM offline draft

## Status and timing

- **Status:** `future-direct`; offline only until submissions open
- **Urgency:** low — submissions open October 20 at 9:00 AM PDT and close
  **December 20, 2026 at 9:00 PM PST / December 21 at 05:00 UTC**.
- **Online Devpost draft saved:** **no**
- **Track:** General Track. Do not enter the Rethinking Waste track.

## Official record

- Overview: https://arbiter-hacks-v1.devpost.com/
- Rules: https://arbiter-hacks-v1.devpost.com/rules
- Schedule: https://arbiter-hacks-v1.devpost.com/details/dates
- Direct submission manager:
  https://devpost.com/submit-to/30789-arbiter-hacks-v1/manage/submissions
- Eligibility: students age 13+, worldwide subject to standard exclusions;
  companies/professional organizations excluded.
- Team: solo or 1–4; one submission to one track.
- General Track: any working software or hardware project.
- Required: public code repository; YouTube/Vimeo demo **2–3 minutes** showing
  the project working; written description covering what it does, how it was
  built, and what comes next. Missing repository or video is ineligible.
- Existing work: the posted overview/rules do not state a pre-existing-project
  or event-only build restriction. Disclose the baseline.

## Paste-ready fields

### Project name

SpeedLM — Faster Local LLMs From Idle GPU Time

### Tagline (54 characters)

Automatically speeds up local LLMs with idle GPU time.

### Short description

SpeedLM automatically speeds up local LLMs with idle GPU time. Operators keep
their serving data in their own environment and keep the normal
OpenAI-compatible vLLM workflow. SpeedLM trains a workload-specific speculative
draft, promotes it only after held-out evaluation, and leaves the full target
model authoritative for every token.

### Written story

#### What it does

Teams self-host LLMs so prompts, tool calls, and outputs can stay inside
infrastructure they control. The tradeoff is that local vLLM does not
automatically receive provider-wide optimization. A generic speculative draft
may not fit the team's real workload, while the inference GPU sits idle between
traffic bursts. SpeedLM turns that quiet time into local optimization without
changing the client API or requiring a separate training GPU.

Operators keep the normal OpenAI-compatible vLLM workflow. SpeedLM forwards
real requests and stores completed input/output pairs as bounded traces in the
operator-controlled environment. After five quiet minutes by default, it
safely drains accepted work, releases serving GPU memory, and trains an isolated
EAGLE-3 candidate. Whole sessions stay on one side of the train/held-out
boundary. SpeedLM promotes only a passing candidate and otherwise restores the
incumbent. New traffic waits during safe preemption and recovery rather than
reaching a sleeping engine or partial weights. The target model verifies every
emitted token.

#### How we built it

The Python stack deliberately separates streaming proxying, exact and
normalized trace storage, vLLM process supervision, idle orchestration,
PyTorch/Speculators training, immutable artifact promotion, and rollback. The
authentic Remotion demo uses real server/curl PTYs and retained traffic,
training, and gate evidence.

#### Result and next steps

For users, SpeedLM turns hardware they already operate into automatic local
adaptation while preserving their API, data boundary, and authoritative target
model. On Qwen3-8B agentic traffic, an archived gate measured 15.0% more accepted
tokens per verifier step across 287 session-disjoint contexts and eight repeats,
but rejected promotion because its timing channel was non-stationary. A separate
retained 125-context, eight-repeat gate promoted a +13.4% accepted-length gain.
Neither result establishes universal speed, cost, or energy savings. Next we
will add bounded request waits, validate more hardware, publish a pinned
container, and test longer promotion chains.

### Built With

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Qwen3-8B, Hugging Face models, CUDA 12.9, NVIDIA H100, SLURM, pytest, Ruff,
mypy, Remotion, OpenAI Codex

### AI-use disclosure

OpenAI Codex assisted with coding, tests, documentation, review, and video
editing. The author reviewed the result and remains responsible. Metrics and
terminal footage come from executed runs and retained artifacts, not generated
console output.

### Existing-work disclosure

SpeedLM predates Arbiter Hacks V1's October 20 submission opening. Its earliest
recorded repository commit is July 25, 2026. Arbiter's posted rules do not ban
existing projects or define a build window; this baseline is disclosed rather
than hidden. `TODO(owner): attest actual material start and prior entries`.

## Exact asset plan

- Required demo: `docs/assets/speedlm-devpost-demo.mp4` — verified at 150.059
  seconds as H.264/AAC with burned English captions, inside the 2–3 minute
  requirement; YouTube/Vimeo hosting remains pending.
- Repository: `https://github.com/RyanKim17920/speedlm` — public, MIT, and
  anonymously verified at commit `3d1848e`.
- Gallery: poster plus `01-serving-proof.png`, `03-idle-tuning.png`, and
  `05-promotion-gate.png`.
- Hosted video: `TODO(owner): public YouTube/Vimeo URL`.

## Final owner-only checks

- [ ] Wait until registration/submissions open on October 20.
- [ ] Confirm jurisdiction, enrollment details, and 1–4-person roster.
- [ ] Confirm actual timeline, prior entries, authorship, and asset rights.
- [ ] Verify public repository and the 2–3 minute hosted demo.
- [ ] Review and personally accept the event attestations.
