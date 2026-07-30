from __future__ import annotations

import contextlib
import hashlib
import json
import os
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO

from speedlm.gateway.worker import await_worker

_SENSITIVE_HEADERS = {
    b"api-key",
    b"authorization",
    b"cookie",
    b"proxy-authorization",
    b"set-cookie",
    b"x-api-key",
}


class ExchangeWriteError(RuntimeError):
    """Raised when strict raw exchange persistence can no longer continue."""


class ExchangeLedger:
    """Protocol-neutral, append-only storage for proxied HTTP exchanges.

    Bodies are deliberately stored as bytes rather than decoded protocol
    objects. This makes the ledger independent of model family, endpoint,
    streaming format, content encoding, and harness implementation.
    """

    def __init__(self, root: Path, *, writer_threads: int = 4) -> None:
        if writer_threads <= 0:
            raise ValueError("writer_threads must be positive")
        self.root = root
        self._executor = ThreadPoolExecutor(
            max_workers=writer_threads,
            thread_name_prefix="speedlm-exchange-writer",
        )
        self._closed = False

    def start(
        self,
        *,
        method: str,
        path: str,
        query: bytes,
        request_headers: Sequence[tuple[bytes, bytes]],
        started_at: float,
    ) -> ExchangeRecorder:
        return ExchangeRecorder(
            self.root,
            executor=self._executor,
            method=method,
            path=path,
            query=query,
            request_headers=request_headers,
            started_at=started_at,
        )

    async def astart(
        self,
        *,
        method: str,
        path: str,
        query: bytes,
        request_headers: Sequence[tuple[bytes, bytes]],
        started_at: float,
    ) -> ExchangeRecorder:
        """Create the durable initial record without blocking the event loop."""
        if self._closed:
            raise RuntimeError("exchange ledger is closed")
        return await await_worker(
            self._executor.submit(
                partial(
                    self.start,
                    method=method,
                    path=path,
                    query=query,
                    request_headers=request_headers,
                    started_at=started_at,
                )
            )
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Relay shutdown drains every submitted recorder operation first, so
        # this is only a worker join and does not perform storage I/O.
        self._executor.shutdown(wait=True)

    def recover_incomplete(self) -> int:
        """Seal crash-left ``recording`` records as exact incomplete prefixes."""
        recovered = 0
        if not self.root.is_dir():
            return recovered
        for directory in sorted(self.root.iterdir()):
            manifest_path = directory / "manifest.json"
            if not manifest_path.is_file():
                continue
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not isinstance(manifest, dict) or manifest.get("state") != "recording":
                    continue
                for stream_name in ("request", "response"):
                    stream = manifest.get(stream_name)
                    if not isinstance(stream, dict):
                        stream = {}
                        manifest[stream_name] = stream
                    body_path = directory / f"{stream_name}.body"
                    size, digest = _file_size_and_sha256(body_path)
                    stream["body_file"] = f"{stream_name}.body"
                    stream["bytes"] = size
                    stream["sha256"] = digest
                    stream["complete"] = False
                manifest["state"] = "incomplete"
                manifest["completed_at"] = time.time()
                manifest["failure_reason"] = "recovered_after_restart"
                _atomic_private_json(manifest_path, manifest)
                recovered += 1
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                continue
        return recovered

    async def arecover_incomplete(self) -> int:
        if self._closed:
            raise RuntimeError("exchange ledger is closed")
        return await await_worker(
            self._executor.submit(self.recover_incomplete)
        )

    def iter_manifests(self) -> Iterator[dict[str, Any]]:
        if not self.root.is_dir():
            return
        for directory in sorted(self.root.iterdir()):
            manifest = directory / "manifest.json"
            if not manifest.is_file():
                continue
            value = json.loads(manifest.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                yield value


class ExchangeRecorder:
    """One durable raw request/response pair.

    A ``recording`` manifest is committed before proxying begins. If the
    process crashes, that manifest and any body prefix already written remain
    discoverable. Normal completion fsyncs both bodies before atomically
    replacing the manifest with ``complete``.
    """

    def __init__(
        self,
        root: Path,
        *,
        executor: ThreadPoolExecutor,
        method: str,
        path: str,
        query: bytes,
        request_headers: Sequence[tuple[bytes, bytes]],
        started_at: float,
    ) -> None:
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        with contextlib.suppress(OSError):
            root.chmod(0o700)
        self.id = uuid.uuid4().hex
        self.directory = root / self.id
        self.directory.mkdir(mode=0o700)
        self._executor = executor
        self._lock = threading.RLock()
        self._method = method
        self._path = path
        self._started_at = started_at
        self._request_headers = _safe_headers(request_headers)
        self._response_headers: list[list[str]] = []
        self._response_status: int | None = None
        self._request_size = 0
        self._response_size = 0
        self._request_hash = hashlib.sha256()
        self._response_hash = hashlib.sha256()
        self._request_complete = False
        self._response_complete = False
        self._finalized = False
        self._storage_errors: list[str] = []
        self._request_file = _private_binary_file(self.directory / "request.body")
        self._response_file = _private_binary_file(self.directory / "response.body")
        _write_private_bytes(self.directory / "query.bin", query)
        self._write_manifest(state="recording", failure_reason=None)
        _fsync_directory(root)

    def feed_request(self, chunk: bytes) -> None:
        with self._lock:
            self._feed(
                chunk,
                stream_name="request",
                destination=self._request_file,
                digest=self._request_hash,
            )

    async def afeed_request(self, chunk: bytes) -> None:
        await self._run(self.feed_request, chunk)

    def finish_request(self) -> None:
        with self._lock:
            self._request_complete = True

    async def afinish_request(self) -> None:
        await self._run(self.finish_request)

    def set_response(
        self,
        *,
        status: int,
        headers: Sequence[tuple[bytes, bytes]],
    ) -> None:
        with self._lock:
            self._response_status = status
            self._response_headers = _safe_headers(headers)

    async def aset_response(
        self,
        *,
        status: int,
        headers: Sequence[tuple[bytes, bytes]],
    ) -> None:
        await self._run(self.set_response, status=status, headers=headers)

    def feed_response(self, chunk: bytes) -> None:
        with self._lock:
            self._feed(
                chunk,
                stream_name="response",
                destination=self._response_file,
                digest=self._response_hash,
            )

    async def afeed_response(self, chunk: bytes) -> None:
        await self._run(self.feed_response, chunk)

    def complete(self) -> None:
        with self._lock:
            self._response_complete = True
            self._finalize(state="complete", failure_reason=None)

    async def acomplete(self) -> None:
        await self._run(self.complete)

    def abort(self, reason: str) -> None:
        with self._lock:
            self._finalize(state="incomplete", failure_reason=reason)

    async def aabort(self, reason: str) -> None:
        await self._run(self.abort, reason)

    async def _run(self, function: Any, *args: Any, **kwargs: Any) -> Any:
        return await await_worker(
            self._executor.submit(partial(function, *args, **kwargs))
        )

    def _feed(
        self,
        chunk: bytes,
        *,
        stream_name: str,
        destination: BinaryIO | None,
        digest: Any,
    ) -> None:
        if not chunk or self._finalized:
            return
        if stream_name == "request":
            self._request_size += len(chunk)
        else:
            self._response_size += len(chunk)
        digest.update(chunk)
        if destination is None:
            return
        try:
            remaining = memoryview(chunk)
            while remaining:
                written = destination.write(remaining)
                if written is None or written <= 0:
                    raise OSError("short write to raw exchange body")
                remaining = remaining[written:]
        except OSError as exc:
            self._storage_errors.append(
                f"{stream_name} body write failed: {type(exc).__name__}: {exc}"
            )
            self._close_file(stream_name)
            raise ExchangeWriteError(self._storage_errors[-1]) from exc

    def _close_file(self, stream_name: str) -> None:
        attribute = f"_{stream_name}_file"
        file = getattr(self, attribute)
        if file is None:
            return
        with contextlib.suppress(OSError):
            file.close()
        setattr(self, attribute, None)

    def _sync_and_close(self, stream_name: str) -> None:
        attribute = f"_{stream_name}_file"
        file = getattr(self, attribute)
        if file is None:
            return
        try:
            file.flush()
            os.fsync(file.fileno())
        except OSError as exc:
            self._storage_errors.append(
                f"{stream_name} body sync failed: {type(exc).__name__}: {exc}"
            )
        finally:
            with contextlib.suppress(OSError):
                file.close()
            setattr(self, attribute, None)

    def _finalize(self, *, state: str, failure_reason: str | None) -> None:
        if self._finalized:
            return
        self._finalized = True
        self._sync_and_close("request")
        self._sync_and_close("response")
        if self._storage_errors:
            state = "storage_error"
            failure_reason = "; ".join(self._storage_errors)
        # The body files remain recoverable by directory and exchange ID even
        # when the final manifest replacement itself fails.
        try:
            self._write_manifest(state=state, failure_reason=failure_reason)
        except OSError as exc:
            raise ExchangeWriteError(
                f"final exchange manifest write failed: {type(exc).__name__}: {exc}"
            ) from exc

    def _write_manifest(self, *, state: str, failure_reason: str | None) -> None:
        manifest = {
            "schema_version": 1,
            "exchange_id": self.id,
            "state": state,
            "method": self._method,
            "path": self._path,
            "query_file": "query.bin",
            "started_at": self._started_at,
            "completed_at": (
                time.time() if state != "recording" else None
            ),
            "request": {
                "headers": self._request_headers,
                "body_file": "request.body",
                "bytes": self._request_size,
                "sha256": self._request_hash.hexdigest(),
                "complete": self._request_complete,
            },
            "response": {
                "status": self._response_status,
                "headers": self._response_headers,
                "body_file": "response.body",
                "bytes": self._response_size,
                "sha256": self._response_hash.hexdigest(),
                "complete": self._response_complete,
            },
            "failure_reason": failure_reason,
        }
        _atomic_private_json(self.directory / "manifest.json", manifest)


def _safe_headers(
    headers: Sequence[tuple[bytes, bytes]],
) -> list[list[str]]:
    result: list[list[str]] = []
    for raw_name, raw_value in headers:
        name = raw_name.lower()
        value = b"<redacted>" if name in _SENSITIVE_HEADERS else raw_value
        result.append(
            [
                raw_name.decode("latin-1"),
                value.decode("latin-1"),
            ]
        )
    return result


def _private_binary_file(path: Path) -> BinaryIO:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    return os.fdopen(descriptor, "wb", buffering=0)


def _write_private_bytes(path: Path, value: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    with os.fdopen(descriptor, "wb", buffering=0) as file:
        file.write(value)
        file.flush()
        os.fsync(file.fileno())


def _atomic_private_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (
        json.dumps(value, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", buffering=0) as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def _file_size_and_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
