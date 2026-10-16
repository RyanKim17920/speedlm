# Code for Humanity — excluded research record

## Status and timing

- **Status:** `excluded-needs-different-project`; SpeedLM alone is not a
  sufficient humanitarian submission.
- **Urgency:** low — overview deadline **January 15, 2027 at 5:00 PM PKT /
  12:00 UTC**.
- **Online Devpost draft saved:** **no**

## Official record and conflicts

- Overview: https://code-for-humanity.devpost.com/
- Rules: https://code-for-humanity.devpost.com/rules
- Schedule: https://code-for-humanity.devpost.com/details/dates
- Direct submission manager:
  https://devpost.com/submit-to/30740-code-for-humanity/manage/submissions
- Eligibility: above legal majority, students only, worldwide subject to
  standard exclusions; solo entries allowed.
- Theme: technology improving access, opportunity, or quality of life for an
  underserved or underrepresented community.
- Required: public GitHub repository and working demo video no longer than
  three minutes.
- New-work rule: overview says all code must be written during the hackathon.
  Posted period begins July 16, 2026; the repository begins July 25, but only
  the entrant can attest whether all material code fits.
- **Conflict 1:** rules say July 16, 2026–January 15, **2026**, an impossible
  reverse interval; overview/schedule say January 15, 2027.
- **Conflict 2:** overview includes a likely copied requirement about
  “self-measurement or reducing measurement-related waste,” which conflicts
  with the rest of the humanitarian brief. Obtain organizer clarification.

## Why unchanged SpeedLM is not ready

SpeedLM is generic local-inference infrastructure. It has no retained evidence
of deployment with an underserved community, accessibility research, a named
humanitarian workflow, or measured beneficiary outcomes. Do not claim those
things. A qualifying submission needs a genuine extension and evidence, such
as a deployment that lets a resource-constrained nonprofit or under-resourced
school run a specific local AI service safely. That extension has not been
built.

## Conditional copy after a real extension exists

### Project name

SpeedLM — Adaptive Local AI Infrastructure

### Tagline (54 characters)

Automatically speeds up local LLMs with idle GPU time.

### Honest current description

SpeedLM gives self-hosted vLLM deployments a safe workload-adaptation loop. It
keeps the serving boundary local, learns a speculative draft from completed
traffic during confirmed idle windows, and promotes only after
session-disjoint held-out evaluation. This infrastructure is working, but a
Code for Humanity entry still requires a real underserved-community deployment
and beneficiary evidence; neither is claimed in the current build.

### Story foundation

Hosted APIs cross a vendor-controlled boundary and may receive provider-wide
optimization under policies that vary. Some organizations choose local serving
for infrastructure control, but local vLLM generally uses a static generic
draft. SpeedLM fills that technical gap: its OpenAI-compatible gateway captures
completed exchanges, waits for confirmed idle time, drains requests, trains an
isolated EAGLE-3 candidate, and promotes only after a held-out gate. The target
model remains authoritative for every token.

On Qwen3-8B agentic traffic, a documented gate measured 15.0% more accepted
tokens per verifier step across 287 session-disjoint contexts and eight
repeats. We do not claim universal throughput, energy, cost, accessibility, or
humanitarian outcomes from that infrastructure benchmark.

`REQUIRED BUILD BEFORE THIS BECOMES A SUBMISSION: identify an eligible
community with consent; implement a specific accessible service using SpeedLM;
retain usability/deployment evidence; document safeguards and data governance;
measure an outcome relevant to that community; and rewrite this story around
what was actually built.`

### Built With

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Qwen3-8B, Hugging Face models, CUDA 12.9, NVIDIA H100, SLURM, pytest, Ruff,
mypy, Remotion, OpenAI Codex; `REQUIRED BUILD: add only the technologies used by
the humanitarian extension`.

### AI-use disclosure

OpenAI Codex assisted with coding, tests, documentation, review, and video
editing. The author reviewed the work and remains responsible. Any future
beneficiary research, deployment, or outcome claim must come from retained
evidence rather than generated copy.

### Existing-work/build-window disclosure

SpeedLM's earliest recorded repository commit is July 25, 2026, after the
overview's July 16 start. `TODO(owner): attest the actual material start and all
prior work/submissions`. Any humanitarian extension must have its own dated
change log. Do not finalize if the “all code during the hackathon” requirement
cannot be met.

## Exact asset plan after extension

- Base technical demo: `docs/assets/speedlm-devpost-demo.mp4` — exactly 150
  seconds and within the three-minute cap; verified at 150.059 seconds as
  H.264/AAC with burned English captions.
- Current video does not demonstrate a humanitarian deployment.
  `REQUIRED BUILD: replace or add authentic beneficiary/workflow footage while
  staying under three minutes`.
- Required public repository: `https://github.com/RyanKim17920/speedlm` —
  `TODO(owner): authorize public release and verify logged out`.
- Hosted video/deployment URLs: `TODO(owner)`.

## Final owner-only checks

- [ ] Obtain organizer clarification on the date and measurement-waste text.
- [ ] Build and evidence a genuine underserved-community solution.
- [ ] Attest that all code meets the stated build-window rule.
- [ ] Confirm jurisdiction, enrollment details, team, ownership, and prior entries.
- [ ] Verify public repository and a deployment-specific ≤3-minute video.
- [ ] Review and personally accept all event attestations.
