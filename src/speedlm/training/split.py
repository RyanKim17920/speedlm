"""Deterministic, leakage-safe train/benchmark trace snapshots."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

from speedlm.gate.suite import BenchmarkSuite, persist_suite
from speedlm.traces.store import TraceRecord, TraceStore
from speedlm.tuner.eagle3 import Eagle3Error, TraceSnapshot
from speedlm.tuner.idle import TuningPreempted

logger = logging.getLogger(__name__)


class HeldOutTraceSnapshotLeaser:
    """Freeze one trace watermark into disjoint train and benchmark datasets.

    Selection is *incremental by window*, not by full rescan.  When
    ``training_window_records`` is set the leaser addresses only the newest
    ``training_window_records`` entries of the buffer through the store's
    record cursor, so a cycle costs one window rather than one corpus and does
    not re-serialize traces every previous cycle already trained on.

    Why a sliding window and not "only the records that are new since the last
    cycle": EAGLE-3 training warm-starts from the *currently active* draft, so
    each cycle is an incremental optimizer step, not a fresh fit.  A new-only
    dataset would routinely be one or two rows -- the tuner arms on a single
    fresh trace -- which is below the two-unique-context floor the held-out
    split requires, and a single high-variance step on an unrepresentative
    batch is exactly how a warm-started head is destroyed.  A bounded window
    keeps the per-cycle training distribution a stable, representative sample
    of recent traffic while still bounding the work, and it degrades to the
    old behaviour when the corpus is smaller than the window.  Retention is
    append-with-eviction: old records leave the window from the tail, they are
    not resampled.

    The window makes one new hazard possible that a monotonically growing
    corpus did not have.  ``BenchmarkSuite.build`` reserves a seeded top-*k*
    prefix of the ranked context hashes, and *k* shrinks when the window drops
    records, so a context benchmarked in an earlier cycle could fall back into
    the training set of a later one -- and the candidate warm-starts from the
    draft that earlier cycle promoted.  The leaser therefore keeps a sticky
    ledger of every context hash it has ever held out and refuses to train on
    any of them, so "benchmarked once, never trained on" holds across cycles
    and not merely within one.
    """

    def __init__(
        self,
        source: TraceStore,
        *,
        held_out_fraction: float = 0.2,
        scratch_quota_bytes: int,
        training_window_records: int | None = None,
    ) -> None:
        if not 0 < held_out_fraction < 1:
            raise ValueError("held_out_fraction must be in (0, 1)")
        if training_window_records is not None and (
            isinstance(training_window_records, bool)
            or not isinstance(training_window_records, int)
            or training_window_records < 2
        ):
            raise ValueError("training_window_records must be an int >= 2 or None")
        self._source = source
        self._held_out_fraction = held_out_fraction
        self._scratch_quota_bytes = scratch_quota_bytes
        self._training_window_records = training_window_records
        self._lock = threading.Lock()
        self._training_context_hashes: frozenset[str] = frozenset()
        self._benchmarked_context_hashes: frozenset[str] = frozenset()
        self._suite_dir: Path | None = None

    @property
    def training_context_hashes(self) -> frozenset[str]:
        with self._lock:
            return self._training_context_hashes

    @property
    def benchmarked_context_hashes(self) -> frozenset[str]:
        """Every context ever reserved for benchmarking, across all cycles."""
        with self._lock:
            return self._benchmarked_context_hashes

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
        records, buffered = self._select_window()
        self._checkpoint(started, timeout_seconds, should_abort)
        if len(records) < 2:
            raise Eagle3Error("at least two trace records are required for a held-out split")

        suite = BenchmarkSuite.build(
            records,
            held_out_fraction=self._held_out_fraction,
        )
        held_out = {context.context_hash for context in suite.contexts}
        # Excluded from training, not merely this cycle's reservation: a
        # context benchmarked by any earlier cycle already scored the draft
        # this cycle warm-starts from.
        excluded = held_out | self.benchmarked_context_hashes
        training = tuple(
            record
            for record in records
            if BenchmarkSuite._record_hash(record) not in excluded
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
            # One line per lease, not per record: without it the window is
            # unobservable by construction -- a reader cannot tell whether a
            # cycle trained on the whole corpus or on a bound window, nor
            # whether the configured window bound anything at all.
            logger.info(
                "leased %d training record(s) and %d held-out context(s) from a "
                "%d-record window over %d buffered record(s) (window=%s, bound=%s)",
                len(training),
                len(suite.contexts),
                len(records),
                buffered,
                self._training_window_records,
                bool(
                    self._training_window_records is not None
                    and buffered > self._training_window_records
                ),
            )
            with self._lock:
                self._training_context_hashes = training_hashes
                self._benchmarked_context_hashes |= held_out
                self._suite_dir = suite_dir
            return TraceSnapshot(target, digest.hexdigest())
        except BaseException:
            target.unlink(missing_ok=True)
            with suppress(OSError):
                destination.rmdir()
            raise

    def _select_window(self) -> tuple[tuple[TraceRecord, ...], int]:
        """Return the newest ``training_window_records`` records and the total.

        The buffered total is returned alongside so a lease can report how
        much of the corpus it actually addressed; without both numbers the
        window is indistinguishable from a full scan in the run artifacts.

        With no window configured this is the historical full scan, kept so
        that callers with a small, bounded corpus pay nothing for the cursor.
        With a window the store is asked for a record offset first, and only
        the tail is deserialized -- the cost of a cycle stops scaling with the
        size of the trace corpus.

        The offset is a pure function of ``(count, window)``, so two leases of
        an unchanged buffer select byte-identical records; determinism of the
        snapshot and of the suite derived from it is preserved.
        """
        window = self._training_window_records
        if window is None:
            records = tuple(self._source.iter_records())
            return records, len(records)
        buffered = self._source.count_records()
        start = max(0, buffered - window)
        return tuple(self._source.iter_records(start=start)), buffered

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
