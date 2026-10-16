# Hyperbloom September — AI/ML — SpeedLM offline draft

## Status and urgency

- **Status:** `direct-active`
- **Urgency:** medium — deadline **September 14, 2026 at 5:00 PM EDT / 21:00 UTC**.
- **Online Devpost draft saved:** **no**

## Official record

- Overview: https://hyperbloom-september.devpost.com/
- Rules: https://hyperbloom-september.devpost.com/rules
- Schedule: https://hyperbloom-september.devpost.com/details/dates
- Direct submission manager:
  https://devpost.com/submit-to/31082-hyperbloom-september-ai-ml/manage/submissions
- Submissions opened August 25, 2026.
- Eligibility panel: students only and above the legal age of majority; the
  prose also mentions high-school and college students. The entrant has stated
  current student status and age over 18, but jurisdiction and school details
  remain owner-provided form data.
- Required: **200–500 word** description, repository link, every team member,
  AI-tools disclosure, and post-event survey feedback.
- No video, screenshot, or license requirement is stated.
- Existing work: the posted rules do not ban pre-existing projects and do not
  define August 25 as a build-start date. Disclose the baseline instead of
  inventing a start date.

## Paste-ready fields

### Project name

SpeedLM — Faster Local LLMs From Idle GPU Time

### Tagline (54 characters)

Automatically speeds up local LLMs with idle GPU time.

### Short description

SpeedLM automatically speeds up local LLMs with otherwise idle GPU time. Teams
keep serving data in their own environment and keep their normal
OpenAI-compatible vLLM workflow. SpeedLM trains a workload-specific speculative
draft, gates it on held-out sessions, and leaves the full target model
authoritative for every token.

### Required 200–500 word description (284 words)

Teams self-host LLMs so prompts, tool calls, and model outputs can stay inside
infrastructure they control. The tradeoff is that a local server does not
automatically receive provider-wide workload optimization. Its speculative
draft is often static and generic, even when the real workload is coding agents
or tool use, and its expensive GPU sits idle between traffic bursts.

SpeedLM closes that gap. It automatically turns confirmed idle GPU time into
training time for a workload-specific speculative draft. Operators keep the
normal OpenAI-compatible vLLM workflow: existing clients send the same requests
to a real vLLM server, while completed input/output exchanges become bounded
training traces inside the operator-controlled environment.

After five quiet minutes by default, SpeedLM safely drains accepted requests,
lets vLLM release serving GPU memory, and trains an isolated EAGLE-3 candidate. Whole
sessions remain on one side of the train/held-out boundary. The candidate is
promoted only if it beats the incumbent under the configured safeguards;
failure or safe preemption restores verified serving state. Requests never
intentionally reach a sleeping engine or partial weights.

The full target model remains authoritative and verifies every emitted token.
The tuned draft only proposes work for it to accept or reject. On Qwen3-8B
agentic traffic, the strongest documented session-disjoint gate measured
accepted length rising from 2.3051 to 2.6507 across 287 contexts and eight
repeats—15.0% more accepted tokens per verifier step. That archived gate
rejected promotion because its wall-clock channel was non-stationary. A separate
retained 125-context, eight-repeat gate promoted a +13.4% accepted-length gain;
neither result supports a universal wall-clock serving speed claim.

We built SpeedLM in Python with FastAPI, httpx, vLLM, PyTorch, Speculators,
EAGLE-3, CUDA, and an NVIDIA H100. It is a research preview; broader GPU and
runtime support still needs validation.

### Built With

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Qwen3-8B, Hugging Face models, CUDA 12.9, NVIDIA H100, SLURM, pytest, Ruff,
mypy, Remotion, OpenAI Codex

### Required AI-use disclosure

OpenAI Codex assisted with coding, tests, documentation, code review, and video
editing. The author reviewed the work and remains responsible for the submitted
code and claims. Benchmark figures come from executed GPU runs and retained
artifacts, not generated terminal output.

### Existing-work disclosure

SpeedLM predates Hyperbloom's August 25 submission opening. Its earliest
recorded repository commit is July 25, 2026. The posted rules do not prohibit
existing projects or define an event-only build window; this date is disclosed
so judges can evaluate the work transparently. `TODO(owner): provide the actual
material start date and any prior-submission history if the form asks`.

## Exact asset plan

- Required video: **none**.
- Required screenshot: **none**.
- Optional gallery if the form exposes one: poster plus
  `01-serving-proof.png`, `03-idle-tuning.png`, and `05-promotion-gate.png`.
- Repository: `https://github.com/RyanKim17920/speedlm` — public, MIT, and
  anonymously verified at commit `3d1848e`.

## Final owner-only checks

- [ ] Confirm jurisdiction, school/enrollment details, and team roster.
- [ ] Verify the final description remains within 200–500 words after edits.
- [ ] Complete the required AI disclosure and post-event survey.
- [ ] Confirm actual timing, prior entries, ownership, and right to submit.
- [ ] Review and personally accept all event attestations.
