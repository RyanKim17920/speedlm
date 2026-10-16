# HackSocial 2026 — SpeedLM offline draft

## Status and urgency

- **Status:** `expired-unconfirmed`
- **Closed:** August 31, 2026 at 11:45 PM EDT / September 1 at 03:45 UTC.
- **Online Devpost draft saved:** **no**
- **Track:** AI/ML

The authenticated and logged-out submission pages now report that the period
has ended. Keep this packet only as an archive unless the entrant can document
that an online entry was saved or finalized before the deadline.

> **Cross-submission warning:** GIBC V2 disallows a project substantially the
> same as a previous hackathon entry. Preparing both drafts is fine, but
> submitting unchanged SpeedLM to HackSocial first may make the later GIBC
> entry ineligible. Do not finalize both unchanged. Either choose HackSocial or
> make the later GIBC project materially distinct and document that work.

## Official record

- Overview: https://hacksocial2026.devpost.com/
- Rules: https://hacksocial2026.devpost.com/rules
- Schedule: https://hacksocial2026.devpost.com/details/dates
- Direct submission manager:
  https://devpost.com/submit-to/30724-hacksocial-2026/manage/submissions
- Submission window: August 1–31, 2026.
- Eligibility: ages 13+, current students, worldwide subject to Devpost's
  standard exclusions; companies and professional organizations excluded.
- Team: 1–4 individually registered eligible people; one team per entrant and
  one submitting team per project.
- Required: project/problem text plus a demo, live link, or source repository
  that lets judges evaluate it. The overview asks for repository/code,
  inspiration, stack, and lessons learned. Screenshots are encouraged, not
  mandatory.
- Existing work: the posted rules require original, authorized work but do not
  state a project-start or event-only build restriction.

The overview and schedule now agree on **11:45 PM EDT**. Older campaign copy
that says 5:00 PM EDT is stale and must not be reused.

## Eligibility note

The entrant has stated that they are a current student and over 18. Before
submission, `TODO(owner)` is limited to the entrant's eligible jurisdiction,
school/enrollment details requested by the form, actual solo/team roster,
ownership of every public asset, and any prior submission history.

## Paste-ready fields

### Project name

SpeedLM — Idle-Tuned Speculative Decoding

### Tagline (54 characters)

Automatically speeds up local LLMs with idle GPU time.

### Short description

SpeedLM is an OpenAI-compatible vLLM gateway that learns from completed serving
traffic. During confirmed idle periods it trains an isolated EAGLE-3
speculative draft, evaluates the candidate on session-disjoint held-out
traffic, and promotes only a passing artifact. The target model still verifies
every emitted token. On Qwen3-8B agentic traffic, a documented gate measured
15.0% more accepted tokens per verifier step across 287 contexts and eight
repeats; shared-node wall-clock results varied, so this is not presented as a
universal 15% speedup.

### Project story

#### Inspiration

Hosted APIs cross a vendor-controlled boundary and may receive provider-wide
optimization under policies that vary by provider. Local vLLM keeps the serving
and data boundary under the operator's control, but speculative decoding usually
uses a frozen generic draft that may poorly match coding-agent, tool-use, or
conversational traffic. SpeedLM fills that local optimization gap with a safer
adaptation loop during otherwise idle periods.

#### What it does

SpeedLM replaces `vllm serve` with an OpenAI-compatible streaming gateway that
supervises a real vLLM process. It records only completed input/output pairs as
bounded traces. After five quiet minutes by default, it closes admission
atomically, drains accepted requests, releases serving memory, and trains an
isolated speculative-draft candidate. Whole sessions remain on one side of the
train/held-out split. The candidate is promoted atomically only after it clears
the configured gate; failure or preemption restores the incumbent.

The verifier remains authoritative. The draft proposes tokens, but Qwen3-8B
accepts or rejects every proposal. A request that arrives during a preemptible
stage waits while SpeedLM restores verified serving state instead of reaching a
sleeping engine or partial weights.

#### Impact and evidence

The practical problem is infrastructure efficiency for researchers, student
teams, and smaller AI builders that already use vLLM. The measured result is
15.0% more accepted tokens per verifier step on 287 session-disjoint contexts
over eight repeats. We do not claim a measured energy, cost, or universal
throughput reduction. The authentic demo shows real launch, server, curl,
traffic, training, and promotion evidence rather than generated terminal text.

#### What we learned and what is next

Adaptive inference is a systems-safety problem as much as a model-training
problem. Admission, sleep, worker cleanup, leakage-resistant evaluation,
rollback, and recovery must form one contract. Next we will add bounded request
waits, validate additional GPU/runtime combinations, publish a pinned CUDA
container, and exercise longer unattended promotion chains.

### Built With

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Qwen3-8B, Hugging Face models, CUDA 12.9, NVIDIA H100, SLURM, pytest, Ruff,
mypy, Remotion, OpenAI Codex

### AI-use disclosure

OpenAI Codex assisted with coding, tests, documentation, code review, and video
editing. The author reviewed the work and remains responsible for the code and
claims. Metrics come from executed GPU runs and retained artifacts, and the
demo's terminal regions come from real PTY recordings rather than generated
terminal output.

### Existing-work disclosure

SpeedLM is not represented as starting at HackSocial. Its earliest recorded
repository commit is July 25, 2026. HackSocial's posted rules do not impose an
event-only build window, but only the entrant can attest the actual material
start date and prior submission history.

## Exact asset plan

- Video: `docs/assets/speedlm-devpost-59s.mp4` — verified 59.051-second
  H.264/AAC cut with burned English captions; **public hosting is pending**.
  Video is optional if the
  repository is the judge-evaluation path.
- Hero: `docs/assets/speedlm-demo-poster.jpg`
- Gallery: `00-launch-commands.png`, `01-serving-proof.png`, and
  `05-promotion-gate.png` from `docs/assets/devpost/`.
- Repository field: `https://github.com/RyanKim17920/speedlm` — public, MIT,
  and anonymously verified at commit `3d1848e`.
- Hosted video: `TODO(owner): public YouTube/Vimeo URL, if used`.

## Final owner-only checks

- [ ] Resolve the HackSocial-versus-GIBC cross-submission choice.
- [ ] Confirm jurisdiction, school/enrollment details, and actual team roster.
- [ ] Confirm actual work timeline, prior entries, authorship, and asset rights.
- [ ] Verify the repository or other evaluation path works logged out.
- [ ] Verify the 59-second file and hosted URL if attaching video.
- [ ] Review and personally accept all Devpost/HackSocial attestations.
