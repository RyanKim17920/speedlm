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
from speedlm.traces.redact import RedactionPolicy, RedactionReport, Redactor
from speedlm.traces.store import (
    RedactionSummary,
    TraceError,
    TraceRecord,
    TraceStats,
    TraceStore,
    format_redaction_summary,
    summarize_redactions,
)

__all__ = [
    "NormalizeError",
    "NormalizeResult",
    "Rejection",
    "RedactionPolicy",
    "RedactionReport",
    "RedactionSummary",
    "Redactor",
    "TraceError",
    "TraceRecord",
    "TraceStore",
    "TraceStats",
    "format_redaction_summary",
    "normalize_file",
    "normalize_record",
    "summarize_redactions",
]
