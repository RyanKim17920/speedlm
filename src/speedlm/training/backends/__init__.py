"""Built-in speculative-training backends."""

from speedlm.training.backends.eagle3 import Eagle3Backend, Eagle3Config
from speedlm.training.backends.mtp import MTPBackend, MTPBackendUnavailable

__all__ = [
    "Eagle3Backend",
    "Eagle3Config",
    "MTPBackend",
    "MTPBackendUnavailable",
]
