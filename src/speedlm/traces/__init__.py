"""CPU-only trace spine for SpeedLM.

Captures proxy traces, stores them in a rolling buffer, and normalizes
external OpenAI-format JSONL into the internal contract.
"""

from speedlm.traces.normalize import (
    NormalizeError,
    NormalizeResult,
    Rejection,
    normalize_file,
    normalize_record,
)
from speedlm.traces.store import (
    TraceError,
    TraceRecord,
    TraceStats,
    TraceStore,
)

__all__ = [
    "NormalizeError",
    "NormalizeResult",
    "Rejection",
    "TraceError",
    "TraceRecord",
    "TraceStore",
    "TraceStats",
    "normalize_file",
    "normalize_record",
]