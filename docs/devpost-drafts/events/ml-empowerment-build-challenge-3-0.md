# ML Empowerment Build Challenge 3.0 — SpeedLM offline draft

## Status and urgency

- **Status:** `direct-active`, with an official date conflict
- **Urgency:** use the earlier official deadline: **September 30, 2026 at 11:45
  PM PDT / October 1 at 06:45 UTC**.
- **Online Devpost draft saved:** **no**

## Official record

- Overview: https://ml--build-challenge-3-0.devpost.com/
- Rules: https://ml--build-challenge-3-0.devpost.com/rules
- Direct submission manager:
  https://devpost.com/submit-to/31197-ml-empowerment-build-challenge-3-0/manage/submissions
- Event start: August 30, 2026.
- **Conflict:** the overview says September 30 at 11:45 PM PDT; the rules say
  October 30 at 9:00 PM PDT. Use September 30 unless the organizer publishes a
  correction.
- Eligibility: students; overview requires legal majority while rules say
  high-school and college; worldwide; solo or teams of 1–5. The entrant has
  stated current student status and age over 18.
- Build: an AI project that solves a real-world problem or creates positive
  social impact. AI tools and productivity/developer tools are accepted.
- Required: title; detailed problem, solution, features, technologies, and
  target-user description; at least one screenshot, video, or other project
  file; and team members/roles. Repository is optional but recommended.
- Existing work: neither overview nor rules states a pre-existing-project ban
  or an explicit event-only build window. Disclose the July repository baseline.
- **Direct-fit rationale:** SpeedLM itself is the submitted AI developer and
  productivity tool. No separate application, sponsor integration, or themed
  wrapper is required by the posted requirements.
- Live check on September 1 showed 32 participants and twelve $100 Top Project
  awards ($1,200 cash total).

## Paste-ready fields

### Project name

SpeedLM — Faster Local LLMs From Idle GPU Time

### Tagline (54 characters)

Automatically speeds up local LLMs with idle GPU time.

### Short description

SpeedLM automatically speeds up local LLMs with idle GPU time. Self-hosted
teams keep serving data in their own environment, keep the normal
OpenAI-compatible vLLM workflow, and gain workload-specific optimization. Every
new draft is tested on held-out sessions before promotion, and the full target
model remains authoritative for every response.

### Detailed description

#### Problem and target users

Researchers, student teams, and organizations often self-host LLMs so prompts,
tool calls, and outputs stay inside infrastructure they control. But local
vLLM deployments do not automatically receive provider-wide workload
optimization. Their speculative draft is usually fixed and generic, even when
the actual workload is coding agents or tool use, and their expensive GPU sits
idle between traffic bursts.

#### Solution and features

SpeedLM converts that unused local capacity into training time for a
workload-specific speculative draft. Existing clients keep using the normal
OpenAI-compatible vLLM workflow, while completed paired exchanges become
bounded traces inside the operator-controlled environment. After five quiet
minutes by default, SpeedLM safely drains accepted work, lets vLLM release serving GPU
memory, and trains an isolated EAGLE-3 candidate. Whole sessions remain on one
side of the train/held-out split. The candidate is promoted only after it
clears the configured gate; failure or preemption restores the incumbent.

The verifier accepts or rejects every draft proposal and remains responsible
for each emitted token. Requests arriving during a preemptible stage wait for a
verified serving state instead of reaching a sleeping engine or partial weights.

#### Evidence and impact

For users, the benefit is local control without giving up automatic workload
adaptation: data stays in their environment, their clients do not change, and
idle GPU time becomes useful training time. The authoritative target model
still verifies every token. A Qwen3-8B session-disjoint gate measured accepted
length increasing from 2.3051 to 2.6507 across 287 contexts and eight repeats:
15.0% more accepted tokens per verifier step. That archived gate rejected
promotion because its wall-clock channel was non-stationary. A separate retained
125-context, eight-repeat gate promoted a +13.4% accepted-length gain. Neither
result establishes universal serving speed or measured cost/energy savings.

#### What is next

Next steps are bounded gateway waits, more GPU/runtime validation, a pinned
CUDA container, additional trainable profiles, and longer unattended
promotion/rollback tests.

### Built With

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Qwen3-8B, Hugging Face models, CUDA 12.9, NVIDIA H100, SLURM, pytest, Ruff,
mypy, Remotion, OpenAI Codex

### AI-use disclosure

OpenAI Codex assisted with implementation, testing, documentation, code review,
and video editing. The author reviewed the work and remains responsible for the
submission. Benchmark figures and terminal footage come from executed GPU runs
and retained evidence, not generated terminal output.

### Existing-work disclosure

SpeedLM was not represented as starting with this challenge. Its earliest
recorded repository commit is July 25, 2026, before the August 30 event start.
The posted pages do not prohibit existing work or define a build-only window,
so the baseline is disclosed transparently. `TODO(owner): provide actual
material start and prior-submission history if requested by the form`.

## Exact asset plan

- Video: **none required**.
- Required project file: upload at least
  `docs/assets/devpost/01-serving-proof.png`; recommended additions are
  `03-idle-tuning.png` and `05-promotion-gate.png`.
- Optional video if the form benefits from one:
  `docs/assets/speedlm-devpost-59s.mp4`, verified at 59.051 seconds as H.264/AAC
  with burned English captions. The raw public GitHub asset was anonymously
  verified; direct file upload may still be preferable in the Devpost form.
- Repository: `https://github.com/RyanKim17920/speedlm` — optional by rules;
  public, MIT, and anonymously verified at commit `3d1848e`.

## Final owner-only checks

- [ ] Treat September 30 as the deadline or obtain a written correction.
- [ ] Confirm jurisdiction, institution/enrollment details, and team/role list.
- [ ] Confirm actual timeline, prior entries, authorship, and asset rights.
- [ ] Upload at least one functioning-design/evidence file.
- [ ] Review and personally accept the event attestations.
