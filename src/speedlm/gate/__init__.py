"""Promotion-gate module for SpeedLM.

Provides benchmarking, metrics parsing, and promotion-decision logic
for candidate speculative draft heads.
"""
from speedlm.gate.decide import Decision, decide_promotion
from speedlm.gate.metrics import (
    CounterResetError,
    MetricsSnapshot,
    compute_delta,
    parse_metrics,
)
from speedlm.gate.replay import ReplayResult, replay_suite
from speedlm.gate.runner import (
    BenchmarkGateRunner,
    DraftEndpoint,
    GateRunner,
    HttpReplayExecutor,
    MetricsSource,
    ReplayExecutor,
    TraceSource,
)
from speedlm.gate.suite import BenchmarkSuite, build_suite

__all__ = [
    "BenchmarkGateRunner",
    "BenchmarkSuite",
    "CounterResetError",
    "Decision",
    "DraftEndpoint",
    "GateRunner",
    "HttpReplayExecutor",
    "MetricsSource",
    "ReplayExecutor",
    "MetricsSnapshot",
    "ReplayResult",
    "TraceSource",
    "build_suite",
    "compute_delta",
    "decide_promotion",
    "parse_metrics",
    "replay_suite",
]
