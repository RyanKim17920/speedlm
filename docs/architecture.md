# Architecture and request lifecycle

SpeedLM is a serving wrapper plus an opt-in idle-tuning controller. The verifier
model remains the authority for every emitted token.

## Serving path

~~~text
OpenAI-compatible client
          |
          v
SpeedLM streaming gateway
  |       |          |
  |       |          +--> exact exchange ledger
  |       +-------------> normalized bounded traces
  v
loopback vLLM child
  |
  +--> active speculative draft
~~~

The public port belongs to the gateway. It supervises vLLM on a private
loopback port, forwards streaming bytes with backpressure, and records an
exchange only after the request and response are paired.

## Idle-tuning lifecycle

| State | What happens | New request |
| --- | --- | --- |
| Serving | vLLM handles traffic normally | Admitted immediately |
| Quiescing | Admission closes and already admitted work drains | Waits; the cycle is preempted |
| Sleeping | vLLM releases GPU memory | Waits; serving recovery begins |
| Extracting / training | Separate processes prepare signals and train a candidate | Waits; workers are terminated and the incumbent is restored |
| Benchmarking | Stock and candidate drafts replay a frozen held-out suite | Waits; replay is cancelled and the incumbent is restored |
| Promoting | The passed candidate becomes the durable active pointer | Waits briefly, then uses the verified candidate |
| Rolling back / waking | The incumbent is restored and health-checked | Waits until readiness |

Admission is atomic with the activity watermark: a racing request is either
accepted before quiescence and prevents sleep, or waits outside the engine. It
does not reach an engine while that engine is asleep.

Current caveat: the admission wait has no SpeedLM-side request deadline or
queue bound. A client can time out while SpeedLM is cancelling a stage or
restarting vLLM. Restart and candidate activation calls are also not
interruptible in the middle of the child launch.

## Training and promotion

Each cycle:

1. Leases a bounded training window from completed traces.
2. Holds out whole sessions so nested multi-turn records cannot leak into both
   training and evaluation.
3. Extracts verifier signals and warm-starts from the current or public EAGLE-3
   draft.
4. Runs a validation-loss cost filter.
5. Replays the frozen held-out suite against the incumbent and candidate.
6. Promotes atomically only when the accepted-length criterion and throughput
   regression floor pass.

Training never mutates the live verifier or the active draft in place.
Candidate artifacts are immutable. A failed or preempted cycle restores the
durable incumbent before normal admission resumes.

Speculative decoding preserves the verifier's target distribution: draft tokens
are proposals that the verifier accepts or rejects. Independent vLLM runs can
still differ byte-for-byte because batching and GPU kernels are not guaranteed
to be bitwise deterministic.

## Persistent state

<code>SPEEDLM_HOME</code> defaults to <code>~/.speedlm</code>. The durable
active-artifact pointer, cycle status, traces, raw exchanges, and gate decisions
live beneath it. Status and gain reports read persisted evidence; they do not
invent a gain before a completed decision exists.

## Known release limitations

- Live validation currently targets H100 and one pinned CUDA/runtime stack.
- GPU dependencies and Speculators are installed out of band.
- Admission waiting has no server-side bound or backpressure limit.
- Compounding warm-start behavior has not yet been exercised over a long
  unattended promotion chain.
- There is no general post-promotion rollback policy for traffic-distribution
  drift.

See [the GPU E2E harness](e2e-harness.md) and
[benchmark evidence](benchmark-evidence.md) for the verification boundary.
