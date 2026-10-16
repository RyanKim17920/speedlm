# SpeedLM Devpost demo — 2:30 story-first master

## Delivery target

- **Composition:** `SpeedLMSubmission`, exactly 4,500 frames at 30 fps.
- **Rendered artifact:** `docs/assets/speedlm-devpost-demo.mp4`, verified at
  150.059 seconds, 1920×1080 H.264/AAC, SHA-256
  `27a139d4b700f8b18e04c8ec2a5f5d1bc1ea29e25b6f210ea14f05a9d8f0e8d5`.
- **Status:** final user-benefit copy rendered and verified on September 1,
  2026.
- **Format:** 1920×1080, muted authentic source footage, burned English
  explanation, and an optional owner-recorded voiceover.
- **Short variants:** `SpeedLMCampaign60` is 1,770 frames (0:59) and
  `SpeedLMCampaign120` is 3,570 frames (1:59).
- **Voiceover:** use the timestamped owner reads in
  `docs/devpost-voiceover.md`. No voice audio has been generated or cloned.

## Editorial rules

- Open with the user outcome: **SpeedLM automatically speeds up supported local
  LLM profiles with idle GPU time.** Then make the three benefits concrete:
  captured data stays on the user’s infrastructure, existing clients keep the
  OpenAI-compatible serving API, and quiet GPU windows become workload-specific
  draft training.
- A terminal-looking pane may contain only pixels from a real PTY recording.
- Artifact-derived cards and charts must identify the retained source.
- Say **“+15.0% accepted length”** and explain it as more accepted draft tokens
  per verifier step—not “universally 15% faster.” That archived gate did not
  promote because its wall-clock channel was non-stationary.
- Keep the later **+13.4% accepted-length** result clearly labeled as a
  separate retained gate, not a repeat of the archived +15.0% result.
- The hosted/local comparison describes product boundaries, not every vendor:
  use “vendor-controlled / policy-dependent” and never claim that a provider
  trains on customer data.
- Do not add unsupported cost, energy, privacy, or universal-speed claims.
- Every compressed wait or traffic excerpt must remain visibly labeled as a
  time-lapse.

## Exact master timeline

Frame ranges are inclusive and assume 30 fps.

| Time | Frames | Scene | Evidence rule |
| --- | ---: | --- | --- |
| 0:00–0:11 | 0–329 | **Faster local LLMs, automatically.** Compare the outcomes that matter: where data goes, whether the system learns the user’s workload, and whether a quiet GPU window starts workload tuning. | Explanatory product comparison, clearly labeled as policy-dependent and limited to supported profiles. It is not terminal footage. |
| 0:11–0:19 | 330–569 | One matching recorded reply plus **+15.0% accepted length**: the target model accepts more draft tokens per verifier step. The unstable timing channel blocks promotion. | `short-race.mp4`; keep **ARCHIVED GATE · NOT PROMOTED** visible. |
| 0:19–0:25 | 570–749 | Show why the final answer remains trustworthy: the workload-tuned disposable draft suggests tokens; Qwen3-8B still verifies every emitted token. | Architecture grounded in `src/speedlm/gate/` and `src/speedlm/tuner/`; faint backing is real server footage. |
| 0:25–0:41 | 750–1229 | Show that the serving workflow stays familiar: full normal-vLLM and SpeedLM commands, then both literal server PTYs. | Job 394208 assets and `VERIFIED.json`; do not crop away `--enable-idle-tuning`. |
| 0:41–0:56 | 1230–1679 | Show one unchanged OpenAI-compatible chat request shape: two literal curl sessions return the matching recorded reply for the same greedy request. | Job 394208 client PTYs and response JSON; demonstrate this request, not every vLLM control route or universal byte equality across independent batching. |
| 0:56–1:07 | 1680–2009 | Show that workload learning comes from traffic the user already serves: three marked excerpts from the authentic 31:55 server capture, then real agent-harness output. | `speedlm-vllm-serve-live.mp4` (`fc1cb445…`) and `traffic-results-live.mp4` (`9ca9a9e9…`); keep **TIME-LAPSED** visible. |
| 1:07–1:27 | 2010–2609 | Let users inspect exactly what trains the draft: two verbatim completed inputs and model-predicted outputs, with reasoning first and a tool call second. | `trafficEvidence.json`, extracted from the retained 6,176-line trace store (`1d052dbe…`). This is a data display, not a recreated terminal. |
| 1:27–1:45 | 2610–3149 | Show downtime becoming training: authentic side-by-side server PTYs and the exact clock from requests stopping to SpeedLM tuning. | Job 394208 timing sidecars. Keep **demo threshold 5s · software default 5 min** visible for every frame of this scene. |
| 1:45–2:03 | 3150–3689 | Prove the idle window produces measurable learning: 262 unsmoothed optimizer points, draft loss, draft accuracy, and retained training time. | Retained `cyclev2-run1` trainer stdout (`24834950…`) and lifecycle events. |
| 2:03–2:20 | 3690–4199 | Explain the safety outcome: only measured improvements can serve. Show the separate session-disjoint 125×8 gate, **2.429 → 2.755**, **+13.4% accepted length**, and **PROMOTED**. | Retained decision (`62f5e6cc…`); this is a separate run corroborating the archived opening result. |
| 2:20–2:30 | 4200–4499 | End on the user action: install once, add one serving flag, and keep the existing OpenAI-compatible client API. | Documentation card, not a fake install recording. |

## Evidence boundaries

Safe now:

- Real launch, server, curl, traffic, and idle PTY recordings.
- A 6,176-exchange retained trace store with reasoning and tool-call examples.
- A retained 262-step, three-epoch training run and a retained promoted gate.
- The archived +15.0% replay, provided it remains explicitly labeled as not
  promoted because its throughput channel was non-stationary.

Do not imply:

- A Nebius deployment or NVIDIA-sponsored integration; neither is currently
  recorded in the assets.
- A live mid-training-arrival demonstration; implementation/tests exist, but no
  authentic GPU recording is present.
- That every hosted provider has the same policy, uses customer data for
  training, or exposes no optimization controls.
- That the accepted-length gain is a universal wall-clock, cost, or energy gain.

## Recording notes

- Ask the owner to record the chosen timestamped script as WAV, FLAC, or a
  high-quality phone voice memo.
- A real owner read is preferred. A clone requires explicit consent for this
  use and an intentionally supplied sample; disclose it where required.
- Generate word-level captions only after the final narration exists, then
  hand-correct `SpeedLM`, `vLLM`, `Qwen3-8B`, and `EAGLE-3`.
