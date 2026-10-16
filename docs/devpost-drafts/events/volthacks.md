# VoltHacks — SpeedLM offline draft

## Status and urgency

- **Status:** `direct-active`
- **Urgency:** high — deadline **September 5, 2026 at 5:00 PM EDT / 21:00 UTC**.
- **Online Devpost draft saved:** **no**
- **Category:** Open Innovation / AI-powered tools, or the closest general AI
  category actually offered by the form.

## Official record

- Overview: https://volthacks.devpost.com/
- Rules: https://volthacks.devpost.com/rules
- Schedule: https://volthacks.devpost.com/details/dates
- Direct submission manager:
  https://devpost.com/submit-to/29884-volthacks/manage/submissions
- Event/submission period: May 22–September 5, 2026.
- Eligibility panel: ages 13+, students only, worldwide subject to standard
  exclusions; companies/professional organizations excluded.
- Solo and team entries are allowed.
- Required: description; demo video **or** presentation; photos/screenshots;
  source/repository where applicable. No video duration is stated.
- Timing rule: projects are to be created during the event; an existing project
  needs significant new development during it.

The repository's earliest recorded commit is July 25, inside the posted event
period, but that is not a substitute for the entrant's actual-timeline
attestation.

## Paste-ready fields

### Project name

SpeedLM — Faster Local LLMs From Idle GPU Time

### Tagline (54 characters)

Automatically speeds up local LLMs with idle GPU time.

### Short description

SpeedLM automatically speeds up local LLMs with GPU time that would otherwise
sit idle. Operators keep their serving data inside their own environment and
keep using the normal OpenAI-compatible vLLM workflow. SpeedLM learns a
workload-specific speculative draft, promotes it only after held-out
evaluation, and leaves the full target model authoritative. A documented
Qwen3-8B gate measured 15.0% more accepted tokens per verifier step across 287
contexts and eight repeats, but rejected promotion because its wall-clock
channel was non-stationary. A separate retained gate promoted +13.4% accepted
length.

### Project story

#### Problem and inspiration

Teams choose local LLMs to keep prompts, tool calls, and model outputs inside
infrastructure they control. The tradeoff is that they do not automatically get
provider-wide workload optimization. Their expensive inference GPU also sits
idle between traffic bursts, while a fixed speculative draft may be a poor fit
for their actual coding, tool-use, or chat workload. We built SpeedLM to turn
that unused local capacity into better inference without changing how clients
call vLLM.

#### Solution

Operators keep the normal OpenAI-compatible vLLM workflow. SpeedLM forwards
real streaming requests, keeps completed training traces in the
operator-controlled environment, and automatically waits for a confirmed idle
window. After five quiet minutes by default it safely drains accepted work,
lets vLLM release serving GPU memory, and uses that GPU time to train an isolated
EAGLE-3 candidate. A frozen session-disjoint suite compares the candidate with
the incumbent, and SpeedLM promotes it only when the configured safeguards pass.

New traffic does not reach a sleeping engine or partially trained artifact. It
waits while a preemptible stage is cancelled and verified serving state is
restored. The target model still accepts or rejects every proposed token.

#### Innovation and impact

SpeedLM gives local-AI teams three practical benefits: their traffic stays in
their environment, their clients keep the OpenAI-compatible serving API, and idle hardware becomes
training capacity instead of waste. Safety does not depend on trusting the new
draft: the full target model remains authoritative for every token. An archived
gate measured accepted length rising from 2.3051 to 2.6507 on 287
session-disjoint contexts over eight repeats—15.0% more accepted tokens per
verifier step—but rejected promotion because its wall-clock channel was
non-stationary. A separate retained gate promoted a +13.4% accepted-length gain;
neither result is a universal speed claim.

#### What we learned and next steps

The most difficult work was lifecycle safety and leakage-resistant evaluation,
not launching a trainer. Next steps are bounded admission waits, broader
hardware validation, a pinned CUDA container, additional trainable profiles,
and long-running promotion/rollback testing.

### Built With

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Qwen3-8B, Hugging Face models, CUDA 12.9, NVIDIA H100, SLURM, pytest, Ruff,
mypy, Remotion, OpenAI Codex

### AI-use disclosure

OpenAI Codex assisted with implementation, tests, documentation, review, and
video editing. The author reviewed the work and remains responsible for it.
Metrics and terminal scenes come from executed H100/vLLM runs and retained
artifacts, not fabricated console output.

### Existing-work/build-window disclosure

The repository's earliest recorded commit is July 25, 2026, inside VoltHacks'
May 22–September 5 period. `TODO(owner): attest the actual material start date
and list the significant work completed during the event`. If material work
predates May 22 and the event-period additions are not significant, do not
finalize this entry.

## Exact asset plan

- Demo video: `docs/assets/speedlm-devpost-119s.mp4` — verified 119.061-second
  H.264/AAC cut with burned English captions. Public hosting remains pending;
  VoltHacks gives no duration cap.
- Alternate deliverable if video hosting is unavailable: a presentation built
  from the six screenshots in `docs/assets/devpost/`.
- Hero: `docs/assets/speedlm-demo-poster.jpg`.
- Repository: `https://github.com/RyanKim17920/speedlm` — public, MIT, and
  anonymously verified at commit `3d1848e`.
- Hosted demo/presentation URL: `TODO(owner)`.

## Final owner-only checks

- [ ] Confirm jurisdiction, institution/enrollment detail, and team roster.
- [ ] Attest actual project timing and significant event-period development.
- [ ] Confirm prior entries, code/media ownership, and third-party licenses.
- [ ] Verify screenshots plus video or presentation satisfy the form.
- [ ] Review and personally accept the event attestations.
