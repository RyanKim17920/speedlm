from __future__ import annotations

import asyncio

import httpx

from speedlm.gateway.activity import ActivityTracker
from speedlm.gateway.app import create_app
from speedlm.gateway.control import AdmissionGate
from speedlm.tuner.idle import IdleDetector


class _OneShotStream(httpx.AsyncByteStream):
    async def __aiter__(self):
        yield b'{"path":"/v1/models"}'


def test_request_during_tuning_preempts_then_waits_transparently() -> None:
    asyncio.run(_exercise_transparent_admission())


async def _exercise_transparent_admission() -> None:
    async def upstream(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=_OneShotStream(),
        )

    tracker = ActivityTracker()
    gate = AdmissionGate(tracker)
    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(upstream),
    )
    app = create_app(
        "http://127.0.0.1:8000",
        activity=tracker,
        admission=gate,
        upstream_client=upstream_client,
    )
    detector = IdleDetector(tracker, threshold_seconds=0.001)
    await asyncio.sleep(0.002)
    guard = detector.arm()
    gate.stop_admitting()

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.test",
    ) as client:
        request = asyncio.create_task(client.get("/v1/models"))
        await asyncio.sleep(0.03)
        assert guard.is_preempted
        assert not request.done()

        gate.start_admitting()
        response = await asyncio.wait_for(request, timeout=1.0)

    assert response.status_code == 200
    assert response.json() == {"path": "/v1/models"}
    assert tracker.in_flight == 0
    await upstream_client.aclose()


def test_terminal_close_wakes_queued_request_with_service_unavailable() -> None:
    asyncio.run(_exercise_terminal_close())


async def _exercise_terminal_close() -> None:
    tracker = ActivityTracker()
    gate = AdmissionGate(tracker)
    gate.stop_admitting()
    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"unexpected": True})
        ),
    )
    app = create_app(
        "http://127.0.0.1:8000",
        activity=tracker,
        admission=gate,
        upstream_client=upstream_client,
    )

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://gateway.test",
    ) as client:
        request = asyncio.create_task(client.get("/v1/models"))
        await asyncio.sleep(0.03)
        assert not request.done()

        gate.close()
        response = await asyncio.wait_for(request, timeout=1.0)

    assert response.status_code == 503
    assert response.text == "Gateway shutting down"
    assert tracker.in_flight == 0
    gate.start_admitting()
    assert not gate.is_admitting
    await upstream_client.aclose()
