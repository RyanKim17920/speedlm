"""Render vLLM-shaped Prometheus text exposition from simulated counters.

The series emitted here are exactly the ones
:mod:`speedlm.gate.metrics` parses -- :data:`~speedlm.gate.metrics.COUNTER_NAMES`
and :data:`~speedlm.gate.metrics.GAUGE_NAMES` -- plus enough surrounding noise
(``_count``/``_bucket`` siblings, ``_by_reason`` breakdowns, an unrelated
histogram) that the parser is exercised on a body it has to *filter*, not on a
hand-picked six lines.  The label shape (``engine``/``model_name``) is copied
from ``tests/data/vllm_metrics_before.prom``, which is a verbatim capture.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

#: Model label used by the simulated exposition.  Any string works; a
#: filesystem-looking snapshot path is what a real deployment emits.
DEFAULT_MODEL_LABEL = (
    "/data/models/models--sim--Verifier-8B/snapshots/"
    "0f7b1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b"
)


@dataclass(frozen=True, slots=True)
class CounterState:
    """One engine's monotonic counters at a point in time.

    Every field is cumulative-since-engine-start, which is what makes a drop
    across two scrapes a *reset* rather than a negative rate -- see
    :class:`speedlm.gate.metrics.CounterResetError`.
    """

    generated_tokens: float = 0.0
    prompt_tokens: float = 0.0
    decode_time_seconds: float = 0.0
    drafted_tokens: float = 0.0
    accepted_tokens: float = 0.0
    num_drafts: float = 0.0
    #: Point-in-time gauges, not counters: these may legitimately go down.
    num_requests_running: float = 0.0
    num_requests_waiting: float = 0.0
    num_requests_swapped: float = 0.0
    #: Requests completed, used only to drive the ``_count`` siblings so the
    #: body stays internally plausible.
    finished_requests: float = 0.0

    def advanced(
        self,
        *,
        prompt_tokens: int,
        generated_tokens: int,
        decode_seconds: float,
        drafted: int,
        accepted: int,
        drafts: int,
    ) -> CounterState:
        """Return the state after one served request.

        Only ever moves counters forward.  A simulated restart is expressed by
        constructing a fresh :class:`CounterState`, never by subtracting here.
        """
        return replace(
            self,
            generated_tokens=self.generated_tokens + generated_tokens,
            prompt_tokens=self.prompt_tokens + prompt_tokens,
            decode_time_seconds=self.decode_time_seconds + decode_seconds,
            drafted_tokens=self.drafted_tokens + drafted,
            accepted_tokens=self.accepted_tokens + accepted,
            num_drafts=self.num_drafts + drafts,
            finished_requests=self.finished_requests + 1,
        )


def render_exposition(
    state: CounterState,
    *,
    model_label: str = DEFAULT_MODEL_LABEL,
    engine: str = "0",
    include_spec_decode: bool = True,
) -> str:
    """Render *state* as a vLLM ``/metrics`` body.

    ``include_spec_decode=False`` drops the three ``spec_decode`` series, which
    is what a non-speculative engine exposes.  The gate must then report
    acceptance as *unavailable* rather than as a measured 0% -- see
    :attr:`speedlm.gate.metrics.MetricsDelta.acceptance_available`.
    """
    labels = f'{{engine="{engine}",model_name="{model_label}"}}'
    lines: list[str] = []

    def emit(name: str, value: float, *, kind: str, help_text: str) -> None:
        lines.append(f"# HELP {name} {help_text}")
        lines.append(f"# TYPE {name} {kind}")
        lines.append(f"{name}{labels} {value}")

    if include_spec_decode:
        emit(
            "vllm:spec_decode_num_drafts_total",
            state.num_drafts,
            kind="counter",
            help_text="Number of spec decoding drafts.",
        )
        emit(
            "vllm:spec_decode_num_draft_tokens_total",
            state.drafted_tokens,
            kind="counter",
            help_text="Number of draft tokens.",
        )
        emit(
            "vllm:spec_decode_num_accepted_tokens_total",
            state.accepted_tokens,
            kind="counter",
            help_text="Number of accepted tokens.",
        )

    emit(
        "vllm:num_requests_running",
        state.num_requests_running,
        kind="gauge",
        help_text="Number of requests in model execution batches.",
    )
    emit(
        "vllm:num_requests_waiting",
        state.num_requests_waiting,
        kind="gauge",
        help_text="Number of requests waiting to be processed.",
    )
    # Real vLLM publishes a labelled breakdown whose members sum to the gauge
    # above.  The parser must not double-count them into the same field.
    lines.append(
        "# HELP vllm:num_requests_waiting_by_reason Number of waiting requests by reason."
    )
    lines.append("# TYPE vllm:num_requests_waiting_by_reason gauge")
    lines.append(
        f'vllm:num_requests_waiting_by_reason{{engine="{engine}",'
        f'model_name="{model_label}",reason="capacity"}} {state.num_requests_waiting}'
    )
    lines.append(
        f'vllm:num_requests_waiting_by_reason{{engine="{engine}",'
        f'model_name="{model_label}",reason="deferred"}} 0.0'
    )
    emit(
        "vllm:num_requests_swapped",
        state.num_requests_swapped,
        kind="gauge",
        help_text="Number of requests swapped to CPU.",
    )
    emit(
        "vllm:prompt_tokens_total",
        state.prompt_tokens,
        kind="counter",
        help_text="Number of prefill tokens processed.",
    )
    emit(
        "vllm:generation_tokens_total",
        state.generated_tokens,
        kind="counter",
        help_text="Number of generation tokens processed.",
    )

    # The decode-time histogram.  Only ``_sum`` is parsed, so the siblings are
    # here precisely to prove the parser ignores them: a body where ``_count``
    # or a ``_bucket`` leaked into ``decode_time_seconds`` would produce a
    # wildly wrong tok/s and this is the shape that would catch it.
    lines.append(
        "# HELP vllm:request_decode_time_seconds Decode time per request."
    )
    lines.append("# TYPE vllm:request_decode_time_seconds histogram")
    for bound in ("0.1", "1.0", "10.0", "+Inf"):
        lines.append(
            f'vllm:request_decode_time_seconds_bucket{{engine="{engine}",'
            f'model_name="{model_label}",le="{bound}"}} {state.finished_requests}'
        )
    lines.append(
        f"vllm:request_decode_time_seconds_count{labels} {state.finished_requests}"
    )
    lines.append(
        f"vllm:request_decode_time_seconds_sum{labels} {state.decode_time_seconds}"
    )

    # An unrelated histogram the gate has no opinion about, so that "parses a
    # real body" means more than "parses a body containing only what it wants".
    lines.append("# HELP vllm:time_to_first_token_seconds TTFT.")
    lines.append("# TYPE vllm:time_to_first_token_seconds histogram")
    lines.append(
        f"vllm:time_to_first_token_seconds_count{labels} {state.finished_requests}"
    )
    lines.append(f"vllm:time_to_first_token_seconds_sum{labels} 0.5")

    return "\n".join(lines) + "\n"
