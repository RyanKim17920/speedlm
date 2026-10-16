# SpeedLM — Devpost submission draft

> Historical single-event draft. Use
> [`docs/devpost-drafts/README.md`](devpost-drafts/README.md) and its event
> packets for current submission copy, deadlines, and assets.

## Submission target

**Immediate target:** [VoltHacks](https://volthacks.devpost.com/)

- Deadline: September 5, 2026 at 5:00 PM EDT
- Eligibility: age 13+, worldwide, solo or team
- Build window: May 22–September 5, 2026
- Repository history begins July 25, 2026, inside the build window
- Judging: innovation, technical complexity, real-world impact, design and
  functionality, and presentation quality

Before submitting, confirm that no material SpeedLM implementation predates May
22. The repository history supports the timing, but only the author can make
that final eligibility attestation.

## Paste-ready Devpost fields

### Project name

SpeedLM

### Elevator pitch

Automatically speed up local LLM serving with GPU time that would otherwise sit
idle. SpeedLM learns from your real traffic without changing the
OpenAI-compatible API or letting unverified weights touch a live response.

### Tagline

Automatically speeds up local LLMs with idle GPU time.

### Project story

#### Inspiration

Teams run LLMs locally to keep control of their data and infrastructure, but
they give up the workload-wide optimization available to large hosted
platforms. Their serving GPU also has quiet periods that produce no value.

SpeedLM closes that gap automatically: it uses those idle windows to make local
inference better for the traffic the deployment actually receives, while
remaining a normal vLLM endpoint whenever a request arrives. The mechanism is
speculative decoding—a small draft proposes tokens and the full model verifies
them—but users do not need to replace their model or serving API.

#### What it does

SpeedLM is a serving wrapper for `vllm serve`. It keeps the same
OpenAI-compatible API, streams requests through to a real vLLM process, and
records completed input/output exchanges as bounded training traces.

After five quiet minutes by default, SpeedLM closes admission, drains work that
was already accepted, and lets vLLM release serving GPU memory. It then:

1. snapshots a bounded window of captured traffic;
2. extracts verifier training signals;
3. warm-starts an EAGLE-3 speculative draft from the current or public head;
4. evaluates the candidate on a session-disjoint held-out suite; and
5. promotes it only if it clears the accepted-length and throughput safeguards.

If the candidate fails, SpeedLM restores the incumbent. If new traffic arrives
during a preemptible stage, the request waits at the gateway while SpeedLM
cancels the work and restores a verified serving state. Requests are never sent
to a sleeping engine or to partially trained weights.

The target model remains the verifier for every emitted token. A tuned draft can
change how much work the verifier accepts per step; it does not replace the
target model or gain authority over the response.

#### Measured result

On Qwen3-8B agentic coding traffic, a session-disjoint gate measured mean
accepted tokens per verifier step rising from **2.3051 to 2.6507**: a **15.0%**
increase across **287 held-out contexts × 8 scored repeats**.

Wall-clock throughput for those two heads ranged from +9.9% to +19.9% across
shared-node runs. Because that timing channel was non-stationary, the archived
gate **rejected promotion**. A separate retained gate across 125 held-out
contexts × 8 repeats measured a **13.4% accepted-length gain and promoted**.
SpeedLM does not claim a universal 15% throughput increase.

#### How we built it

The project is a Python 3.12 package and CLI with several deliberately separate
layers:

- a FastAPI/httpx streaming gateway that preserves the vLLM API and records an
  exchange only after request and response are paired;
- a supervisor that launches vLLM on a private loopback port and exposes a
  durable serving lifecycle;
- an append-only raw exchange ledger plus normalized, bounded, configurable
  redacted traces;
- an idle controller with atomic admission, preemption, recovery, and rollback;
- PyTorch and Speculators/EAGLE-3 training workers isolated from the live
  verifier; and
- a promotion gate that separates whole sessions between training and
  evaluation and compares the incumbent with an immutable candidate.

The authentic demo was composed in Remotion from real vLLM, curl, capture, idle,
and trainer output. Its five-second idle threshold is only a shortened recording
setting; the software default is five minutes.

#### Challenges we ran into

The difficult part was not launching a training script. It was protecting the
serving contract around it.

We had to make admission atomic with the idle watermark so a racing request is
either admitted before quiescence or held outside the engine. We also needed to
terminate worker process groups safely, recover the incumbent after an aborted
sleep or training stage, keep partial checkpoints away from live serving, and
avoid leakage from multi-turn agent trajectories into both training and held-out
evaluation.

Benchmarking was another lesson. Accepted length was stable across repeated
gates, while shared-node wall-clock throughput was not. We chose to expose that
distinction rather than turn a noisy timing run into a cleaner marketing number.

#### Accomplishments that we're proud of

- A normal OpenAI-compatible request reaches a real vLLM server; SpeedLM does
  not simulate the serving path.
- After one-time runtime configuration, idle tuning is one serving flag:
  `--enable-idle-tuning`.
- Candidate weights are immutable and gated before atomic promotion.
- Requests arriving during tuning do not see sleeping or half-trained state.
- Tool-carrying and multi-turn traffic can be declared and checked against the
  captured workload instead of silently applying tool-free assumptions.
- The benchmark keeps complete sessions on one side of the train/evaluation
  boundary and reports when a timing result is not trustworthy.

#### What we learned

Adaptive inference is as much a systems-safety problem as a model-training
problem. The strongest architecture was to keep the verifier authoritative,
make the draft disposable, and treat every promotion as a measured hypothesis.

We also learned that evidence needs multiple channels. A draft can improve the
number of accepted tokens per verifier step while a noisy GPU host obscures the
wall-clock effect. Reporting both—and rejecting unstable evidence—is more useful
than optimizing for one headline.

#### What's next

- Add bounded admission queues and explicit server-side wait deadlines.
- Validate more GPU and runtime combinations beyond the current H100 stack.
- Ship a pinned CUDA container and tagged release for reproducible installation.
- Add and validate more trainable verifier/draft profiles.
- Exercise repeated unattended promotion cycles and add drift-triggered
  post-promotion rollback.
- Port the system to Nebius AI Cloud with an NVIDIA open-source model and publish
  a new cloud-reproducible benchmark.

### Built with

Python 3.12, FastAPI, httpx, Uvicorn, vLLM, PyTorch, Speculators, EAGLE-3,
Hugging Face models, SLURM, pytest, Ruff, mypy, Remotion, OpenAI Codex

### AI assistance disclosure

OpenAI Codex was used as a coding, test, documentation, code-review, and video
editing assistant. The submitted repository retains the implementation and test
history, and the benchmark claims come from executed GPU runs rather than
generated or hand-edited terminal output.

### Try it out

- Repository: https://github.com/RyanKim17920/speedlm
- Setup and architecture: https://github.com/RyanKim17920/speedlm#readme
- Public 59-second demo:
  https://github.com/RyanKim17920/speedlm/blob/main/docs/assets/speedlm-devpost-59s.mp4?raw=1

### Suggested tags

`artificial-intelligence`, `machine-learning`, `developer-tools`, `inference`,
`gpu`, `vllm`, `pytorch`, `fastapi`, `speculative-decoding`

## Assets to upload

1. Hero image: `docs/assets/devpost/06-local-inference-gap.png`
2. Launch commands: `docs/assets/devpost/00-launch-commands.png`
3. Serving proof: `docs/assets/devpost/01-serving-proof.png`
4. Idle tuning: `docs/assets/devpost/03-idle-tuning.png`
5. Demo video: `docs/assets/speedlm-devpost-59s.mp4`

The public MP4 is 59 seconds. Local 1:59 and 2:30 variants are available for
events with longer storytelling requirements; upload the required cut to a
public video host when Devpost requires an embeddable URL.

## Final submission checks

- [ ] Confirm the author meets the age/location eligibility terms.
- [ ] Confirm no material implementation predates May 22, 2026.
- [ ] Verify the public README, MIT license, hero, and 59-second MP4 from a
      logged-out browser.
- [ ] Upload at least three readable screenshots.
- [ ] Upload the correct 0:59, 1:59, or 2:30 cut to YouTube or Vimeo if required
      by the form.
- [ ] Add the author and any real collaborators to the Devpost team.
- [ ] Select no sponsor prize unless SpeedLM genuinely implements its required
      technology.
- [ ] Preview every link in a private/logged-out window before final submission.

## Other current matches

1. **GIBC V2, Track 03 (Open):** strongest immediate AI/technical category if
   the author is currently a student. Its July 11–September 21 window includes
   SpeedLM's July 25 repository start.
2. **Nebius × NVIDIA Global AI Hackathon:** strongest strategic and prize match,
   but not eligible as-is. It requires a significant post-August-26 update, a
   real Nebius deployment, and at least one NVIDIA open-source model.
3. **BunnieX:** explicitly welcomes AI developer tools and its build window
   includes SpeedLM, but its official pages conflict on whether entrants must be
   students.
