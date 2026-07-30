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
