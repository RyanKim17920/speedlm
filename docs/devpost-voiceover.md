# SpeedLM campaign voiceovers

These are owner-readable scripts for the three story-first Remotion cuts. Record
one clean take for the chosen cut; do not stretch a shorter recording to fit a
longer edit. No audio has been generated or cloned.

## 0:59 — `SpeedLMCampaign60` (1,770 frames)

| Time | Section | Read |
| --- | --- | --- |
| 0:00–0:11 | Why SpeedLM | “SpeedLM automatically speeds up supported local LLM profiles with idle GPU time. Your data stays on your server, and your app keeps its OpenAI-compatible serving API.” |
| 0:11–0:18 | Archived result | “An archived gate measured fifteen percent more accepted length, but unstable throughput blocked promotion.” |
| 0:18–0:27 | Real launch | “Keep the workflow you know: these real commands add the SpeedLM prefix and one idle-tuning flag. Unconsumed vLLM arguments pass through.” |
| 0:27–0:36 | Same API | “This OpenAI-compatible client request keeps the same shape. The same greedy curl request returned the same recorded reply from normal vLLM and SpeedLM.” |
| 0:36–0:46 | Idle lifecycle | “When traffic stops, normal vLLM stays ready without a tuner. SpeedLM uses the quiet window for local training. Demo wait: five seconds. Default: five minutes.” |
| 0:46–0:50 | Training | “A real run cut draft loss from 9.46 to 1.42.” |
| 0:50–0:55 | Promotion gate | “A separate retained gate promoted a 13.4-percent accepted-length gain.” |
| 0:55–0:59 | Quick start | “Install and set up once. Then add one flag.” |

This cut intentionally omits the traffic-corpus tour. Its hard limit is better
spent making the comparison, launch, curl response, and idle behavior legible.

## 1:59 — `SpeedLMCampaign120` (3,570 frames)

| Time | Section | Read |
| --- | --- | --- |
| 0:00–0:10 | Why SpeedLM | “SpeedLM automatically speeds up supported local LLM profiles with idle GPU time. Your data stays on your server, and your app keeps its OpenAI-compatible serving API.” |
| 0:10–0:18 | Archived result | “An archived gate measured fifteen percent more accepted length, but unstable throughput blocked promotion.” |
| 0:18–0:23 | Verifier | “The draft gets better matched to the workload; the target model still verifies every output token.” |
| 0:23–0:37 | Real launch | “Keep the vLLM workflow you already know. These real H100 recordings show the SpeedLM prefix and one idle-tuning flag; unconsumed vLLM arguments pass through.” |
| 0:37–0:49 | Same API | “Your clients keep the OpenAI-compatible API. The same greedy curl request reaches both real servers and returns the same recorded reply: thirty prompt tokens and ten completion tokens.” |
| 0:49–0:58 | Real traffic | “SpeedLM learns from traffic you already serve. This real coding-agent run retained 6,176 exchanges and 2.55 million predicted tokens.” |
| 0:58–1:11 | Verbatim traces | “Here is exactly what trains the local draft: completed inputs and model-predicted outputs, including reasoning and a later tool call, shown verbatim from the hashed store.” |
| 1:11–1:26 | Idle lifecycle | “When requests stop, normal vLLM stays ready without a tuner. SpeedLM confirms quiet at 6.9 seconds, puts serving to sleep at 16.1, and starts tuning at 3 minutes 39. Demo wait: five seconds. Default: five minutes.” |
| 1:26–1:38 | Training | “That idle window produces measurable learning. Across 262 real steps, draft loss falls from 9.46 to 1.42, and held-out accuracy reaches 75.5 percent.” |
| 1:38–1:51 | Promotion gate | “Only a measured improvement can serve. SpeedLM compares current and tuned drafts on session-disjoint traffic, then promotes or rolls back atomically. This separate gate promoted 13.4 percent accepted length.” |
| 1:51–1:59 | Quick start | “Install once, configure the trainer, run `speedlm doctor`, and add one flag. Your clients keep the OpenAI-compatible serving API.” |

## 2:30 — `SpeedLMSubmission` (4,500 frames)

| Time | Section | Read |
| --- | --- | --- |
| 0:00–0:11 | Why SpeedLM | “SpeedLM automatically speeds up supported local LLM profiles with idle GPU time. Your data stays on your server, your draft learns your workload, and your app keeps its OpenAI-compatible serving API.” |
| 0:11–0:19 | Archived result | “An archived gate measured fifteen percent more accepted length, but unstable throughput blocked promotion.” |
| 0:19–0:25 | Verifier | “The draft gets better matched to your workload. Qwen3-8B still verifies every output token.” |
| 0:25–0:41 | Real launch | “Keep the vLLM workflow you already know. These literal H100 recordings show the full commands: use the SpeedLM prefix, add `--enable-idle-tuning`, and pass unconsumed vLLM arguments through.” |
| 0:41–0:56 | Same API | “Your clients keep the same OpenAI-compatible API. These two real curl sessions send the same greedy request. Both servers return the same recorded reply: thirty prompt tokens and ten completion tokens.” |
| 0:56–1:07 | Real traffic | “SpeedLM learns from traffic you already serve. This thirty-one-minute coding-agent run retained 6,176 completed exchanges, 26.6 million prompt tokens, and 2.55 million predicted tokens.” |
| 1:07–1:27 | Verbatim traces | “Here is exactly what trains your local draft: completed inputs and model-predicted outputs, not a staged summary. The first retained example includes reasoning; the second reaches a tool call. Both come verbatim from the hashed store, and whole sessions stay together when the data splits.” |
| 1:27–1:45 | Idle lifecycle | “When requests stop, normal vLLM stays ready without a tuner. In these real logs, SpeedLM confirms quiet after 6.9 seconds, puts serving to sleep after 16.1 seconds, and starts local tuning after 3 minutes 39. Demo wait: five seconds. Software default: five minutes.” |
| 1:45–2:03 | Training | “That downtime becomes measurable learning. Across 262 real optimizer steps and three epochs, draft training loss falls from 9.46 to 1.42. Held-out position-three accuracy reaches 75.5 percent, and training lasts 6 minutes 24.” |
| 2:03–2:20 | Promotion gate | “Only a measured improvement can serve. SpeedLM compares current and tuned drafts on session-disjoint held-out traffic, then promotes or rolls back atomically. This separate 125-context, eight-repeat gate promoted plus 13.4 percent accepted length.” |
| 2:20–2:30 | Quick start | “Install once, configure the trainer, run `speedlm doctor`, and add one serving flag. Your app keeps the same OpenAI-compatible API.” |

## Recording notes

- Record in a quiet room as WAV, FLAC, or a high-quality phone voice memo.
- Read the cut once end-to-end while watching its timestamps; pause rather than
  speeding up if a sentence lands early.
- Pronounce `vLLM` as “vee-ell-ell-em,” `Qwen` as “quinn,” and `EAGLE-3` as
  “eagle three.”
- A real owner read is preferred. Only use a voice clone with explicit consent
  for these submissions and a sample intentionally supplied for that purpose;
  disclose AI-assisted narration wherever a competition asks.
