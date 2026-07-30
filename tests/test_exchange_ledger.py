from __future__ import annotations

import asyncio
import hashlib
import json
import stat
import threading
from pathlib import Path

import pytest

from speedlm.gateway.exchange import ExchangeLedger


def test_complete_exchange_preserves_exact_bytes_and_hashes(tmp_path: Path) -> None:
    ledger = ExchangeLedger(tmp_path / "exchanges")
    request_body = b'{"prompt":"hello"}\n'
    response_body = b"data: first\n\ndata: second\n\n"

    recorder = ledger.start(
        method="POST",
        path="/custom/generate",
        query=b"mode=raw",
        request_headers=[
            (b"content-type", b"application/json"),
            (b"authorization", b"Bearer secret"),
        ],
        started_at=123.0,
    )
    recorder.feed_request(request_body[:5])
    recorder.feed_request(request_body[5:])
    recorder.finish_request()
    recorder.set_response(
        status=201,
        headers=[(b"content-type", b"text/event-stream")],
    )
    recorder.feed_response(response_body[:7])
    recorder.feed_response(response_body[7:])
    recorder.complete()

    manifests = list(ledger.iter_manifests())
    assert len(manifests) == 1
    manifest = manifests[0]
    directory = ledger.root / manifest["exchange_id"]
    assert manifest["state"] == "complete"
    assert manifest["method"] == "POST"
    assert manifest["path"] == "/custom/generate"
    assert manifest["request"]["bytes"] == len(request_body)
    assert manifest["request"]["sha256"] == hashlib.sha256(request_body).hexdigest()
    assert manifest["request"]["complete"] is True
    assert manifest["response"]["status"] == 201
    assert manifest["response"]["bytes"] == len(response_body)
    assert manifest["response"]["sha256"] == hashlib.sha256(response_body).hexdigest()
    assert manifest["response"]["complete"] is True
    assert manifest["request"]["headers"] == [
        ["content-type", "application/json"],
        ["authorization", "<redacted>"],
    ]
    assert (directory / "query.bin").read_bytes() == b"mode=raw"
    assert (directory / "request.body").read_bytes() == request_body
    assert (directory / "response.body").read_bytes() == response_body
    for filename in ("manifest.json", "query.bin", "request.body", "response.body"):
        mode = stat.S_IMODE((directory / filename).stat().st_mode)
        assert mode == 0o600


def test_incomplete_exchange_retains_written_prefix(tmp_path: Path) -> None:
    ledger = ExchangeLedger(tmp_path / "exchanges")
    recorder = ledger.start(
        method="POST",
        path="/v1/responses",
        query=b"",
        request_headers=[],
        started_at=456.0,
    )
    recorder.feed_request(b"request")
    recorder.finish_request()
    recorder.set_response(status=200, headers=[])
    recorder.feed_response(b"partial-response")
    recorder.abort("client_disconnected")

    manifest = next(ledger.iter_manifests())
    directory = ledger.root / manifest["exchange_id"]
    assert manifest["state"] == "incomplete"
    assert manifest["failure_reason"] == "client_disconnected"
    assert manifest["request"]["complete"] is True
    assert manifest["response"]["complete"] is False
    assert (directory / "response.body").read_bytes() == b"partial-response"


def test_recording_manifest_is_recoverable_before_completion(tmp_path: Path) -> None:
    ledger = ExchangeLedger(tmp_path / "exchanges")
    recorder = ledger.start(
        method="POST",
        path="/future/protocol",
        query=b"",
        request_headers=[],
        started_at=789.0,
    )

    manifest_path = recorder.directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["state"] == "recording"
    assert manifest["exchange_id"] == recorder.id
    recorder.abort("test_cleanup")


def test_recovery_seals_exact_crash_prefix(tmp_path: Path) -> None:
    ledger = ExchangeLedger(tmp_path / "exchanges")
    recorder = ledger.start(
        method="POST",
        path="/inference/v1/generate",
        query=b"",
        request_headers=[],
        started_at=790.0,
    )
    recorder.feed_request(b"complete request")
    recorder.finish_request()
    recorder.set_response(status=200, headers=[])
    recorder.feed_response(b"response prefix")
    recorder.abort("simulated_process_exit")

    manifest_path = recorder.directory / "manifest.json"
    stale = json.loads(manifest_path.read_text(encoding="utf-8"))
    stale["state"] = "recording"
    stale["request"]["bytes"] = 0
    stale["request"]["sha256"] = ""
    stale["response"]["bytes"] = 0
    stale["response"]["sha256"] = ""
    manifest_path.write_text(json.dumps(stale), encoding="utf-8")

    assert ledger.recover_incomplete() == 1

    recovered = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert recovered["state"] == "incomplete"
    assert recovered["failure_reason"] == "recovered_after_restart"
    assert recovered["request"]["bytes"] == len(b"complete request")
    assert recovered["request"]["sha256"] == hashlib.sha256(
        b"complete request"
    ).hexdigest()
    assert recovered["response"]["bytes"] == len(b"response prefix")
    assert recovered["response"]["sha256"] == hashlib.sha256(
        b"response prefix"
    ).hexdigest()
    assert recovered["request"]["complete"] is False
    assert recovered["response"]["complete"] is False


def test_blocked_writer_does_not_block_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        ledger = ExchangeLedger(tmp_path / "exchanges", writer_threads=1)
        recorder = await ledger.astart(
            method="POST",
            path="/future/protocol",
            query=b"",
            request_headers=[],
            started_at=800.0,
        )
        writer_started = threading.Event()
        release_writer = threading.Event()
        original_feed = recorder.feed_request

        def blocked_feed(chunk: bytes) -> None:
            writer_started.set()
            assert release_writer.wait(timeout=2.0)
            original_feed(chunk)

        monkeypatch.setattr(recorder, "feed_request", blocked_feed)
        write_task = asyncio.create_task(recorder.afeed_request(b"payload"))
        while not writer_started.is_set():
            await asyncio.sleep(0.001)

        heartbeat_ticks = 0
        for _ in range(10):
            await asyncio.sleep(0.001)
            heartbeat_ticks += 1

        assert heartbeat_ticks == 10
        assert not write_task.done()
        release_writer.set()
        await write_task
        await recorder.afinish_request()
        await recorder.aabort("test_cleanup")
        await ledger.aclose()

    asyncio.run(scenario())
