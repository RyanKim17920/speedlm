# Devpost screenshot set

All images are 1920×1080 frames extracted from the captioned 2:30 SpeedLM
submission cut. They are intended for Devpost galleries and judge-facing
presentations.

1. `00-launch-commands.png` — both literal launch commands at readable scale.
2. `01-serving-proof.png` — literal normal-vLLM and SpeedLM curl responses.
3. `02-trace-evidence.png` — verbatim completed input and predicted output.
4. `03-idle-tuning.png` — literal server PTYs and actual elapsed milestones.
5. `04-training-metrics.png` — 262 unsmoothed retained optimizer points.
6. `05-promotion-gate.png` — retained 125-context × 8-repeat decision.
7. `06-local-inference-gap.png` — the story-first hosted/local/SpeedLM
   comparison and the local idle-time optimization gap.

The parent video is `../speedlm-devpost-demo.mp4`, SHA-256
`27a139d4b700f8b18e04c8ec2a5f5d1bc1ea29e25b6f210ea14f05a9d8f0e8d5`.
Terminal-looking regions originate from real recorded PTYs; charts and trace
panes identify their retained artifacts in the video.
