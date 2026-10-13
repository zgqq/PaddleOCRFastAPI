"""Synthetic ASGI proofs: blocking OCR/URL work must not run on the event loop."""
import asyncio
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import httpx
import pytest

from main import create_app


@pytest.mark.parametrize("blocked_phase", ["file_inference", "url_fetch", "url_inference"])
def test_blocked_ocr_work_leaves_asgi_health_responsive(monkeypatch, blocked_phase):
    started, release = Event(), Event()
    evidence = {"released_before_return": False, "worker_thread": None}
    image = Path("screenshots/Swagger.png").read_bytes()

    def block():
        evidence["worker_thread"] = get_ident()
        started.set()
        evidence["released_before_return"] = release.wait(timeout=3)

    class Backend:
        def predict(self, image, *, detect=True, classify=True):
            if blocked_phase.endswith("inference"):
                block()
            return [{"rec_polys": [], "rec_texts": [], "rec_scores": []}]

    def get_image(url, *, timeout=None):
        # Accept the historical missing budget so the negative control reaches
        # its actual blocking operation instead of failing in the stub.
        evidence["url_timeout"] = timeout
        if blocked_phase == "url_fetch":
            block()
        return SimpleNamespace(content=image)

    monkeypatch.setattr("routers.ocr.requests.get", get_image)

    async def exercise():
        event_loop_thread = get_ident()
        app = create_app(Backend())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://example.test") as client:
            if blocked_phase == "file_inference":
                request = asyncio.create_task(client.post("/ocr/predict-by-file", files={"file": ("sample.png", image, "image/png")}))
            else:
                request = asyncio.create_task(client.get("/ocr/predict-by-url", params={"imageUrl": "https://example.test/sample.png"}))
            try:
                assert await asyncio.to_thread(started.wait, 4)
                health = await asyncio.wait_for(client.get("/health"), timeout=2)
                assert health.status_code == 200
            finally:
                release.set()
            response = await asyncio.wait_for(request, timeout=4)
            assert response.status_code == 200
            assert response.json()["data"] == [None]
        assert evidence["worker_thread"] != event_loop_thread
        assert evidence["released_before_return"] is True
        if blocked_phase.startswith("url_"):
            assert evidence["url_timeout"] == (5, 30)

    asyncio.run(exercise())
