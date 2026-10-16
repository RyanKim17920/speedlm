# Devpost asset manifest

All listed assets are local. They are not automatically public, embeddable, or
accepted by a Devpost form. Use an event packet to record the eventual hosted
URL and verify it while logged out.

## Video inventory

| Local asset | Duration / frame | Size | SHA-256 | Best use | Boundary |
| --- | --- | ---: | --- | --- | --- |
| `docs/assets/speedlm-devpost-59s.mp4` | 59.051 s / 1,770 frames / 1920×1080 | 11,597,374 B | `4997ab100db20b3a1bf32e9ba37ed4d872746e51a4ab673b719181d230ca12d9` | Strict one-minute ceilings and concise optional demos | Final story-first H.264/AAC render with burned English captions; the AAC track is silent. Public raw GitHub URL is anonymously verified; use a different host when event rules require YouTube/Vimeo. |
| `docs/assets/speedlm-devpost-119s.mp4` | 119.061 s / 3,570 frames / 1920×1080 | 28,910,941 B | `e2bf71e8bc33888b13db9e81ff30e2474087ac434a5123edf64a09bd008d0eed` | Strict two-minute ceilings or uncapped technical demos | Final story-first H.264/AAC render with burned English captions; the AAC track is silent. Public hosting is still required. |
| `docs/assets/speedlm-demo.mp4` | 63.445 s / 1280×720 | 4,236,224 B | `f3698ba6db5cbecd63923312c818de2361a64ee1c0ecc2a2f1f1e9e909e4580a` | Short technical overview when the event has no minimum and permits more than 63.4 seconds | Older technical cut. Too long for a strict 60-second cap; prefer a final story-first cut for judging. |
| `docs/assets/speedlm-devpost-demo.mp4` | 150.059 s / 4,500 frames / 1920×1080 | 35,288,827 B | `27a139d4b700f8b18e04c8ec2a5f5d1bc1ea29e25b6f210ea14f05a9d8f0e8d5` | Technical demo for 2–5 minute or ≤3-minute limits | Final story-first H.264/AAC render with burned English captions; the AAC track is silent. Public hosting is still required. |

The final 2:30 cut opens with the hosted-versus-local serving problem and
SpeedLM's solution, then retains real launch/server/curl
PTYs, authentic traffic, verbatim trace examples, real idle timing, unsmoothed
training points, and gate output. The 15.0% gate must remain marked archived;
the separate retained +13.4% promoted gate must not be merged with it.

### Duration policy

- **Strict ≤60 seconds:** use the verified 59-second campaign cut.
- **No minimum, cap ≥64 seconds:** use the 63.4-second cut unless the event asks
  for more technical depth.
- **Strict ≤2 minutes:** use the verified 119-second campaign cut.
- **2–5 minutes, ≤3 minutes, or ≤5 minutes:** use the 2:30 cut.
- **Narration required:** no owner recording or intentional voice-clone source
  sample is present. The owner has offered to provide one. Prefer an owner read
  of `docs/devpost-voiceover.md`; only clone a voice from an intentionally
  supplied sample and disclose synthetic narration where required.
- **Sponsor/platform proof required:** neither existing video proves a sponsor
  deployment unless the required technology is actually visible and retained
  evidence supports it. Record a genuine event-specific addendum.

Never pad a duration with repeated footage. Any new cut should preserve the
truthfulness rules in `docs/devpost-video-script.md`.

## Gallery image inventory

| Local asset | Dimensions | SHA-256 | Caption |
| --- | ---: | --- | --- |
| `docs/assets/speedlm-demo-poster.jpg` | 1280×720 | `d7f961391c65d022798bbe711c3f5fe4416b1171d34693cd87083cc9b264059e` | Normal vLLM and SpeedLM serving proof; use as the project hero after a visual review. |
| `docs/assets/devpost/00-launch-commands.png` | 1920×1080 | `c9423ffac9fef842b3d8afff6dde3bf2f42bb78fb96ffe5e3db9776828b84bb6` | Literal normal-vLLM and SpeedLM launch commands at readable scale. |
| `docs/assets/devpost/01-serving-proof.png` | 1920×1080 | `10c06199b91c25149e186c54ae19480957c28c7d61c57ca2d7bf51625126b906` | Real curl sessions for one matching request and reply. |
| `docs/assets/devpost/02-trace-evidence.png` | 1920×1080 | `3eb5ac48ddd1ac6fed75e9897f7d4c4d5959ac36a3d6cc7addb31451edbe61b1` | Verbatim completed input and predicted output trace evidence. |
| `docs/assets/devpost/03-idle-tuning.png` | 1920×1080 | `1053319ed351b9067211589707541ee927d65f6f3d7274d81992326e9bec52df` | Real server PTYs and actual elapsed idle-tuning milestones. |
| `docs/assets/devpost/04-training-metrics.png` | 1920×1080 | `2e6232cae6cfd344a8f1906e65972194eb4a6239157ff0658759bcecce2e0698` | 262 unsmoothed optimizer points parsed from retained trainer output. |
| `docs/assets/devpost/05-promotion-gate.png` | 1920×1080 | `6f8d61821e2d5ea596e1ad115d83b84da3f370fb09531c268fd788d83691bbdc` | Retained 125-context × 8-repeat promotion gate. |
| `docs/assets/devpost/06-local-inference-gap.png` | 1920×1080 | `52cba3e39f034c93dec31de47d99049116d9d2d2df7ca2870ac16d64cdcce151` | Hosted API versus local vLLM versus SpeedLM: the local idle-time optimization gap and how SpeedLM fills it. |

The source description for the gallery set is
[`../assets/devpost/README.md`](../assets/devpost/README.md).

## Per-event upload record

Copy this block into the event packet after hosting assets:

```text
Hero image URL: TODO
Gallery image URLs: TODO
Video local source: TODO
Video hosted URL: TODO
Host visibility: TODO(public or unlisted, checked logged out)
Checked at: TODO(UTC timestamp)
Duration requirement: TODO
Audio/caption requirement: TODO
License/attribution requirement: TODO
```
