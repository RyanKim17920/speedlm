# Global Innovation Build Challenge V2 — Track 03 — SpeedLM offline draft

## Status and urgency

- **Status:** `direct-conditional`, conditional on timing and prior-entry attestations
- **Urgency:** medium — deadline **September 21, 2026 at 11:45 PM Taipei time / 15:45 UTC**.
- **Online Devpost draft saved:** **no**
- **Track:** **Track 03 — Open (General Technical Invention)**

> **Do not finalize unchanged SpeedLM here after another hackathon
> submission.** GIBC disallows a project substantially the same as a previous
> hackathon entry. Drafting HackSocial and GIBC in parallel is fine; submitting
> HackSocial first can make unchanged SpeedLM ineligible for GIBC. A later GIBC
> submission needs either no conflicting prior entry or a materially distinct,
> dated extension that the rules permit.

## Official record

- Overview: https://gibc-v2.devpost.com/
- Rules: https://gibc-v2.devpost.com/rules
- Schedule: https://gibc-v2.devpost.com/details/dates
- Direct submission manager:
  https://devpost.com/submit-to/30408-global-innovation-build-challenge-v2/manage/submissions
- Build/submission period: July 11–September 21, 2026.
- Eligibility: current students age 13+, worldwide subject to exclusions; solo
  or teams up to six; one team and one track per entrant/project.
- Required: English description; unrestricted public source repository with
  reproducible setup; complete Built With including AI assistants, models,
  hardware, tools, and data; real-name team details; at least three
  screenshots; and a public/unlisted YouTube, Vimeo, or Youku video lasting
  **2–5 minutes** with English audio or subtitles.
- Timing/prior-entry rule: work must be built during July 11–September 21 and
  cannot be commercial or substantially the same as a previous hackathon
  entry.

Track 01 is not valid: SpeedLM uses Qwen3-8B and warm-starts a pretrained draft.
Track 03 imposes no such small-model/from-scratch restriction.

## Paste-ready fields

### Project name

SpeedLM — Faster Local LLMs From Idle GPU Time

### Tagline (54 characters)

Automatically speeds up local LLMs with idle GPU time.

### Short description

SpeedLM automatically speeds up local LLMs with idle GPU time. Operators keep
serving data in their own environment and keep the normal OpenAI-compatible
vLLM workflow. SpeedLM trains a workload-specific speculative draft, gates it
on held-out sessions, and leaves the full target model authoritative. A
documented Qwen3-8B gate measured 15.0% more accepted tokens per verifier step
across 287 contexts and eight repeats, but rejected promotion because its timing
channel was non-stationary. A separate retained gate promoted +13.4% accepted
length.

### Project story

#### Inspiration

Teams self-host LLMs to keep prompts, tool calls, and outputs inside
infrastructure they control. The cost of that control is an optimization gap:
local deployments do not automatically benefit from provider-wide learning,
their speculative draft is commonly frozen and generic, and their expensive
GPU sits idle between traffic bursts. SpeedLM converts that unused capacity
into workload-specific local inference improvement without changing the API.

#### What it does

Operators keep using the normal OpenAI-compatible vLLM workflow. SpeedLM
forwards real requests, records completed paired exchanges as bounded traces in
the operator-controlled environment, and waits for a confirmed idle window.
After five quiet minutes by default it safely drains accepted work, releases
vLLM memory, and trains an isolated EAGLE-3 candidate. Whole sessions stay on
one side of the train/held-out split. The candidate is promoted only after it
passes a frozen held-out gate; failure or preemption restores the incumbent
before admission resumes.

The verifier decides every emitted token. A tuned draft changes how much
proposed work the target accepts, not which model controls the answer.

#### Technical execution and evidence

The user-facing result rests on a complete lifecycle: real streaming proxying,
exact and normalized local trace paths, atomic admission, vLLM sleep/wake
supervision, isolated PyTorch and Speculators workers, immutable candidates,
session-disjoint gating, rollback, and recovery diagnostics. The demo shows
literal server and curl PTYs, 31:55 of authentic traffic in time-lapse,
verbatim reasoning/tool traces, real idle milestones, 262 unsmoothed optimizer
steps, and a retained promotion gate.

The strongest documented gate measured accepted length 2.3051 to 2.6507 on 287
contexts over eight repeats: 15.0% more accepted tokens per verifier step. That
means the authoritative target accepted more useful draft work per step. Its
throughput channel was not stationary, so the archived gate rejected promotion.
A separate retained 125-context, eight-repeat gate promoted a +13.4%
accepted-length gain. Neither result is a universal wall-clock speed claim.

#### What we learned and what is next

Online adaptation succeeds only when recovery is part of the design. Next we
will bound gateway waits, validate more hardware/runtime combinations, publish
a pinned container and sanitized evidence bundle, and test repeated unattended
promotion cycles.

### Built With

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Qwen3-8B, Hugging Face models, CUDA 12.9, NVIDIA H100, SLURM, pytest, Ruff,
mypy, Remotion, OpenAI Codex

### AI-use disclosure

OpenAI Codex assisted with implementation, tests, documentation, code review,
and video editing. The author reviewed its suggestions and remains responsible
for the submission. Terminal regions are real PTY recordings; charts and
metrics are derived from retained run artifacts. No generated terminal
transcript is presented as executed output.

### Data sources and handling

SpeedLM's tuning corpus is made from completed input/output exchanges captured
by its local OpenAI-compatible gateway. The retained demonstration includes
coding-agent and tool-use traffic with reasoning and tool-call tokens. Before
training, exchanges are bounded, normalized, and passed through the project's
redaction path; whole sessions remain on one side of the train/held-out split
to reduce leakage. Raw serving traffic remains in the operator-controlled
environment and is not included in this submission. The demo exposes only
selected redacted examples and aggregate or retained evaluation artifacts.

No proprietary hosted-API dataset is claimed. Qwen3-8B and the warm-started
EAGLE-3 draft are identified in the repository and Built With list; the live
validation used an NVIDIA H100 with the pinned CUDA/vLLM stack documented in
the README.

### Build-window and prior-work disclosure

The earliest recorded repository commit is July 25, 2026, within GIBC's July
11–September 21 window. `TODO(owner): attest whether all material work began on
or after July 11, list any pre-repository work, and disclose every prior
hackathon entry/commercial use`. Git history alone cannot make those
attestations.

## Exact asset plan

- Required video: `docs/assets/speedlm-devpost-demo.mp4` — verified 150.059-second
  H.264/AAC cut, within the 2–5 minute rule, with burned English captions.
  Public or unlisted YouTube/Vimeo/Youku hosting remains pending.
- Upload at least three screenshots; recommended set:
  `06-local-inference-gap.png`, `01-serving-proof.png`, `03-idle-tuning.png`,
  `04-training-metrics.png`, and `05-promotion-gate.png`.
- Repository: `https://github.com/RyanKim17920/speedlm` — public, MIT, and
  anonymously verified at commit `3d1848e`.
- Hosted video: `TODO(owner): public/unlisted YouTube, Vimeo, or Youku URL`.

## Final owner-only checks

- [ ] Resolve the HackSocial/other-hackathon prior-entry conflict.
- [ ] Attest all material work began within the permitted build window.
- [ ] Confirm jurisdiction, enrollment details, real-name team roster, and one-team rule.
- [ ] Verify public repository/reproduction instructions and all model/data/hardware disclosures.
- [ ] Verify 2–5 minute hosted video plus at least three screenshots.
- [ ] Review and personally accept all GIBC attestations.
