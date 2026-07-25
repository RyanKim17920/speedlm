# SpeedLM

A near-drop-in replacement for `vllm serve`. SpeedLM launches and proxies a vLLM
inference server, captures every completed chat/completion request as a
structured trace, and later fine-tunes a speculative draft head during idle GPU
periods. A newly trained draft head is promoted only if both held-out draft
acceptance and throughput improve on a validation set.

## Pinned stack

The following packages define the runtime environment on the GPU host. They are
installed out-of-band -- not by `pip install speedlm` -- because they carry
heavy, platform-specific dependencies (CUDA, PyTorch, FlashAttention). The
wrapper itself only requires fastapi, httpx, and uvicorn.

| Component   | Version           |
|-------------|-------------------|
| Python      | 3.12              |
| vllm        | 0.25.1+cu129      |
| speculators | v0.6.0 (source)   |
| torch       | 2.11.0            |
| CUDA        | 12.9              |

## Install (dev)

```bash
pip install -e ".[dev]"
pytest
```

## Storage layout

All persistent data lives under `~/.speedlm/` (override with the
`SPEEDLM_HOME` environment variable).

```
~/.speedlm/
  traces/      # captured request/response traces
  profiles/    # GPU and throughput profiles
  runs/        # training run artifacts
```

## CLI

```
speedlm vllm serve <model> [vllm args...]   launch a proxied vLLM server
speedlm traces import <path>                 import OpenAI-JSONL trace files
speedlm traces stats                         show trace store statistics
speedlm status                               show system and server status
speedlm gain                                 show token savings from draft head
speedlm doctor                               diagnose host environment
```

## Status

This phase implements configuration, storage, the trace store (a rolling buffer
bounded by token count and record age), OpenAI-JSONL normalization/import, and
the CLI skeleton. The `traces import` and `traces stats` subcommands work
today. The `vllm serve`, `status`, `gain`, and `doctor` subcommands are
declared but not yet implemented; they exit non-zero when invoked. Draft-head
training and promotion are not built yet.