"""Deterministic, leakage-safe train/benchmark trace snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from speedlm.gate.suite import BenchmarkSuite, persist_suite
from speedlm.traces.store import TraceStore
from speedlm.tuner.eagle3 import Eagle3Error, TraceSnapshot
from speedlm.tuner.idle import TuningPreempted


class HeldOutTraceSnapshotLeaser:
    """Freeze one trace watermark into disjoint train and benchmark datasets."""

    def __init__(
        self,
        source: TraceStore,
        *,
        held_out_fraction: float = 0.2,
        scratch_quota_bytes: int,
    ) -> None:
        if not 0 < held_out_fraction < 1:
            raise ValueError("held_out_fraction must be in (0, 1)")
        self._source = source
        self._held_out_fraction = held_out_fraction
        self._scratch_quota_bytes = scratch_quota_bytes
        self._lock = threading.Lock()
        self._training_context_hashes: frozenset[str] = frozenset()
        self._suite_dir: Path | None = None

    @property
    def training_context_hashes(self) -> frozenset[str]:
        with self._lock:
            return self._training_context_hashes

    @property
    def suite_dir(self) -> Path:
        with self._lock:
            if self._suite_dir is None:
                raise Eagle3Error("held-out suite has not been frozen yet")
            return self._suite_dir

    def lease_snapshot(
        self,
        destination: Path,
        *,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> TraceSnapshot:
        started = time.monotonic()
        records = tuple(self._source.iter_records())
        self._checkpoint(started, timeout_seconds, should_abort)
        if len(records) < 2:
            raise Eagle3Error("at least two trace records are required for a held-out split")

        suite = BenchmarkSuite.build(
            records,
            held_out_fraction=self._held_out_fraction,
        )
        held_out = {context.context_hash for context in suite.contexts}
        training = tuple(
            record
            for record in records
            if BenchmarkSuite._record_hash(record) not in held_out
        )
        if not training:
            raise Eagle3Error(
                "held-out split left no training records; collect more unique contexts"
            )
        training_hashes = frozenset(
            BenchmarkSuite._record_hash(record) for record in training
        )
        if training_hashes & held_out:
            raise Eagle3Error("internal train/benchmark split leakage")

        destination.mkdir(parents=True, exist_ok=False)
        target = destination / self._source.path.name
        digest = hashlib.sha256()
        written = 0
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            with os.fdopen(fd, "wb") as output:
                for record in training:
                    self._checkpoint(started, timeout_seconds, should_abort)
                    line = (
                        json.dumps(
                            record.to_dict(),
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                        + b"\n"
                    )
                    written += len(line)
                    if written > self._scratch_quota_bytes:
                        raise Eagle3Error(
                            "training trace snapshot exceeds the scratch quota"
                        )
                    output.write(line)
                    digest.update(line)
                output.flush()
                os.fsync(output.fileno())
            suite_dir = destination.parent / "held-out"
            persist_suite(suite, suite_dir)
            with self._lock:
                self._training_context_hashes = training_hashes
                self._suite_dir = suite_dir
            return TraceSnapshot(target, digest.hexdigest())
        except BaseException:
            target.unlink(missing_ok=True)
            with suppress(OSError):
                destination.rmdir()
            raise

    @staticmethod
    def _checkpoint(
        started: float,
        timeout_seconds: float,
        should_abort: Callable[[], bool],
    ) -> None:
        if should_abort():
            raise TuningPreempted("incoming request preempted trace split")
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError("trace split timed out")
