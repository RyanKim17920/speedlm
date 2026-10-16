# Shared truthful project facts

Use this file as the factual boundary for event drafts. An event packet may use
a narrower claim, but it must not silently strengthen one of these claims.

## Identity and concise description

- **Project name:** SpeedLM
- **Package/version:** `speedlm` 0.1.0, development status Alpha
- **Default tagline:** Automatically speeds up local LLMs with idle GPU time.
- **One-sentence description:** SpeedLM helps operators make local LLM serving
  faster without changing their OpenAI/vLLM workflow: it uses completed local
  traffic and otherwise idle GPU time to tune a speculative draft, while the
  target model remains authoritative and a held-out gate blocks regressions.
- **Public repository URL:** `https://github.com/RyanKim17920/speedlm`
- **Public-release verification:** anonymous HTTP 200 on September 1, 2026 at
  commit `e96774e`; GitHub reports `visibility=PUBLIC` and recognizes the MIT
  license.
- **Public 59-second demo:**
  `https://raw.githubusercontent.com/RyanKim17920/speedlm/main/docs/assets/speedlm-devpost-59s.mp4`
  (anonymous HTTP 200; 11,597,374 bytes; SHA-256
  `4997ab100db20b3a1bf32e9ba37ed4d872746e51a4ab673b719181d230ca12d9`).

## What the current code does

- `speedlm vllm serve` starts a streaming FastAPI/httpx gateway and supervises a
  real vLLM child process on a private loopback port.
- Arguments that SpeedLM does not consume are forwarded to vLLM.
- Completed, paired request/response exchanges can be stored in an exact ledger
  and as bounded normalized traces. Raw exchanges may contain sensitive data;
  configured redaction applies to normalized traces, not automatically to
  every raw byte.
- Idle tuning is opt-in through `--enable-idle-tuning` after one-time runtime
  and profile configuration.
- The documented default is five quiet minutes plus confirmation polls before
  the idle transition. The much shorter threshold visible in the demo was used
  only to make the recording practical.
- During a tuning cycle, admission closes atomically, accepted work drains,
  vLLM releases serving GPU memory, and isolated workers extract signals and train a
  candidate draft.
- Whole sessions stay on one side of the train/held-out boundary.
- Candidate artifacts are immutable until the promotion gate finishes. A
  failed or preempted cycle restores the incumbent before requests resume.
- A request arriving during the cycle waits at the gateway while recovery or
  safe preemption occurs. The current server-side admission wait is unbounded,
  so clients still need an appropriate timeout.
- The target model verifies every emitted token. Tuning the draft changes the
  amount of proposed work the verifier accepts; it does not give the draft
  authority over the response.

## Safe measured claims

### Primary acceptance claim

The strongest documented session-disjoint gate for the Qwen3-8B EAGLE-3 pair
measured mean accepted length increasing from **2.3051 to 2.6507**: a **15.0%**
relative increase, or **+0.3457 with standard error 0.0029**, across **287
contexts and 8 repeats**.

That gate rejected promotion because its throughput channel was not stationary.
The accepted-length effect reproduced closely across three gates on the same
head and suite. Wall-clock throughput varied from **+9.9% to +19.9%** across
four documented gates, so it is not honest to present one of those percentages
as a universal speedup.

Evidence: [`../speedup-ceiling.md`](../speedup-ceiling.md). The opening race in
the 2:30 video is visibly labeled as an archived gate because its original raw
race timelines no longer survive.

### Retained demo promotion

The longer demo also shows a separate retained gate with **125 contexts × 8
repeats**, mean accepted length **2.429 to 2.755**, and **+13.4%**, which did
promote. Do not merge its verdict or sample count with the archived 15.0% gate.

Evidence and retained-artifact hashes are described in
[`../devpost-video-script.md`](../devpost-video-script.md).

### Recorded traffic and training evidence

- Authentic traffic recording: **31 minutes 55 seconds**.
- Retained trace store summarized in the demo: **6,176 completed exchanges**,
  **26.6 million input tokens**, and **2.55 million predicted output tokens**.
- Retained training visualization: **262 optimizer steps** over **3 epochs**,
  training loss **9.46 to 1.42**, held-out position-three accuracy **75.5%**,
  and a training stage lasting about **6 minutes 24 seconds**.
- The side-by-side serving scene uses real server and curl PTYs. One recorded
  greedy request returned the same response content and the same 30-prompt / 10-
  completion-token usage in both runs.

These facts support the demonstration shown; they are not claims about every
workload, GPU, model, or request.

## Runtime and support boundary

- Python requirement: `>=3.12,<3.13`.
- Lightweight declared runtime dependencies: FastAPI, httpx, and Uvicorn.
- GPU/runtime packages are installed separately from the wrapper.
- Live-tested stack: NVIDIA H100, CUDA 12.9, Python 3.12, PyTorch 2.11.0,
  vLLM 0.25.1+cu129, and Speculators v0.6.0.
- Built-in profiles are documented for Qwen3-8B EAGLE-3,
  Llama-3.1-8B-Instruct EAGLE-3, gpt-oss-20b EAGLE-3, and Qwen3.5-9B MTP. A
  profile being listed does not mean every idle-tuning path has been validated;
  Qwen3.5-9B MTP is explicitly not training-wired.
- Plain proxy serving may launch other vLLM models. Idle tuning requires a
  compatible trainable profile and a warm-start draft.
- This is a research preview, not a claim of production readiness or support on
  every NVIDIA GPU.

## Built-with facts

### Runtime and model stack

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Hugging Face models, CUDA 12.9, NVIDIA H100

### Engineering and evidence tooling

SLURM, pytest, Ruff, mypy, Remotion, OpenAI Codex

Only select a sponsor technology or branded track when newly retained evidence
shows that the submitted version actually uses it.

## Known limitations worth disclosing

- Validation is currently limited to one pinned H100 software stack.
- CUDA, PyTorch, vLLM, weights, and Speculators are installed out of band; there
  is no one-command universal GPU installer.
- Admission waiting has no server-side queue bound or deadline.
- Long unattended chains of compounding promotions have not been validated.
- There is no general drift-triggered post-promotion rollback policy yet.
- Raw exchanges can contain credentials, user data, system prompts, and model
  output and must be governed by the deployment owner.

## Publication and entrant facts that are not settled here

Complete these in every event packet. Do not derive them from a profile page,
email address, repository metadata, or prior form.

- `TODO(owner): full legal name and Devpost display name`
- `TODO(owner): current age/date-of-birth attestation required by the event`
- `TODO(owner): current student status, institution, program, and graduation`
- `TODO(owner): country/territory of residence and any event-specific location`
- `TODO(owner): solo entry or complete team roster; confirm every member`
- `TODO(owner): actual date material project work began`
- `TODO(owner): all prior hackathon entries, prizes, publications, and
  commercial use of SpeedLM`
- `TODO(owner): ownership/right-to-submit attestation for code and assets`
- Repository license and access are settled: MIT, public, anonymously verified.
- `TODO(owner): public video-host URLs and any narration/voice-clone consent`

The earliest recorded repository commit is documented as **July 25, 2026**.
That is useful version-control evidence, but it does not by itself establish
when material work began. Only the owner can attest the actual timeline.
