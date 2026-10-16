# BunnieX Hackathon — SpeedLM conditional offline draft

## Status and urgency

- **Status:** `excluded-from-direct-campaign` unless the entrant can truthfully establish
  that SpeedLM was created for this hackathon and all work occurred during its
  permitted period.
- **Urgency:** deadline shown by Devpost is **September 22, 2026 at 5:00 PM IST
  / 11:30 UTC**.
- **Online Devpost draft saved:** **no**

## Official record and conflict

- Overview: https://buuniex-hackathon.devpost.com/
- Rules: https://buuniex-hackathon.devpost.com/rules
- Schedule: https://buuniex-hackathon.devpost.com/details/dates
- Direct submission manager:
  https://devpost.com/submit-to/30487-buuniex-hackathon/manage/submissions
- Devpost schedule: June 22–September 22, 2026.
- **Conflict:** rules still list registration, hacking start, deadline, judging,
  and results as TBA while the event header/schedule gives September 22.
- Eligibility panel: students only and above legal majority; prose says
  students and developers of all skill levels. Use the stricter student panel.
- Team: 1–4; each person on only one team.
- Required: meaningful AI use; original project created for this hackathon; all
  work during the hackathon unless otherwise stated; project description,
  functional GitHub repository, and presentation. Demo video recommended; no
  duration stated.

Current unchanged SpeedLM must not be submitted merely because its July 25
repository date falls after June 22. “Created for this hackathon” is a purpose
and prior-entry attestation only the owner/organizer can resolve.

## Conditional paste-ready fields

### Project name

SpeedLM — Safer Workload-Adaptive AI Serving

### Tagline (54 characters)

Automatically speeds up local LLMs with idle GPU time.

### Short description

SpeedLM is an AI developer tool that fills the optimization gap between hosted
and local model serving. It keeps vLLM's OpenAI-compatible local boundary,
learns a workload-specific speculative draft during confirmed idle windows,
and promotes only after session-disjoint held-out evaluation. The target model
remains authoritative for every token.

### Story

Hosted APIs cross a vendor-controlled boundary and may receive provider-wide
optimization under policies that vary. Local vLLM keeps the serving boundary
under the operator's control, but its speculative draft is usually static and
generic. SpeedLM uses the local GPU's quiet periods to train a better helper
from completed traffic.

The gateway records paired exchanges, waits for five quiet minutes by default,
closes admission atomically, drains accepted work, releases serving memory, and
trains an isolated EAGLE-3 candidate. Whole sessions remain on one side of the
train/held-out split. A failing or preempted candidate is discarded; a passing
candidate is promoted atomically. Requests wait for verified recovery instead
of reaching a sleeping engine or partial weights.

On Qwen3-8B agentic traffic, a documented gate measured 15.0% more accepted
tokens per verifier step across 287 contexts and eight repeats. Wall-clock
results varied, so this is not a universal 15% speed, energy, or cost claim.
Next steps are bounded waits, broader hardware validation, a pinned container,
and long-running promotion/rollback tests.

### Built With

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Qwen3-8B, Hugging Face models, CUDA 12.9, NVIDIA H100, SLURM, pytest, Ruff,
mypy, Remotion, OpenAI Codex

### AI-use disclosure

OpenAI Codex assisted with implementation, tests, documentation, review, and
video editing. The author reviewed the work and remains responsible. Metrics
and terminal scenes come from executed runs and retained evidence.

### Existing-work disclosure and eligibility stop

The earliest recorded repository commit is July 25, 2026. SpeedLM has also
been prepared for other hackathon drafts. `TODO(owner): attest the true project
purpose, material start, and all prior submissions`. Obtain organizer
confirmation before claiming it was “created for this hackathon.” If that
statement is not true, leave this packet offline and do not finalize it.

## Exact asset plan if eligibility is resolved

- Recommended demo: `docs/assets/speedlm-devpost-119s.mp4` — verified
  119.061-second H.264/AAC cut with burned English captions; public hosting is
  pending.
- Required presentation: `REQUIRED BUILD: create a concise deck from the poster
  and six verified screenshots; no presentation file exists yet`.
- Repository: `https://github.com/RyanKim17920/speedlm` —
  `TODO(owner): authorize access and verify`.
- Hosted video/presentation URL: `TODO(owner)`.

## Final owner-only checks

- [ ] Get organizer clarification on TBA rules dates and “created for” scope.
- [ ] Truthfully establish material timing/purpose and disclose prior entries.
- [ ] Confirm jurisdiction, enrollment details, and team roster.
- [ ] Build the required presentation and verify repository/demo access.
- [ ] Review and personally accept the event attestations.
