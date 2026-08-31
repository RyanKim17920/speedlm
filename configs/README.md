# Configurations

Start with
[<code>qwen3-8b-eagle3.example.json</code>](qwen3-8b-eagle3.example.json).
It uses the production five-minute idle threshold and leaves the three
machine-specific runtime paths explicit.

Before enabling tuning:

1. Point <code>speculators_repo</code> at a Speculators v0.6.0 checkout.
2. Point <code>training_python</code> at the interpreter that can run the
   Speculators trainer.
3. Point <code>vllm_python</code> at the interpreter that can run the vLLM
   extraction process.
4. Describe the real traffic under <code>workload</code>. Leave a field null
   only when you intentionally want SpeedLM to infer it from captured traffic.
5. Keep vLLM's <code>--max-model-len</code> equal to
   <code>tuning.sequence_length</code>, or deliberately select another context
   window policy.

<code>agentenv-qwen8b.json</code> and
<code>README-agentenv-launch.md</code> are reproducibility artifacts from the
H100 agentic-workload experiments. They contain machine-specific paths and a
five-second idle threshold, so they are not public quick-start configs.
