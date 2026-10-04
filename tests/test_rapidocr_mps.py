import threading
import time
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backends.RapidOCRBackend import MPSUnavailableError, RapidOCRBackend
from main import create_app
from routers.ocr import get_ocr_backend
from utils.OCRResultAdapter import to_legacy_result


class FakeEngine:
    def __init__(self, devices=("mps", "mps", "mps")):
        for attr, device in zip(("text_det", "text_cls", "text_rec"), devices):
            setattr(self, attr, SimpleNamespace(session=SimpleNamespace(
                device=SimpleNamespace(type=device)
            )))
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return SimpleNamespace(boxes=[[[1, 2], [3, 2], [3, 4], [1, 4]]],
                               txts=("hello",), scores=(0.9,))


def test_selects_explicit_backend_without_importing_torch_or_changing_default(monkeypatch):
    get_ocr_backend.cache_clear()
    monkeypatch.delenv("OCR_BACKEND", raising=False)
    assert type(get_ocr_backend()).__name__ == "PaddleOCRBackend"
    get_ocr_backend.cache_clear()
    monkeypatch.setenv("OCR_BACKEND", "rapidocr-mps")
    assert isinstance(get_ocr_backend(), RapidOCRBackend)
    get_ocr_backend.cache_clear()
    monkeypatch.setenv("OCR_BACKEND", "typo")
    with pytest.raises(ValueError, match="Unsupported OCR_BACKEND"):
        get_ocr_backend()
    get_ocr_backend.cache_clear()


def test_mps_only_checks_all_sessions_and_preserves_flags():
    engine = FakeEngine()
    backend = RapidOCRBackend(engine_factory=lambda: engine, mps_available=lambda: True)
    assert backend.ensure_ready() == {
        "status": "ready", "backend": "rapidocr-mps", "device": "mps",
        "model_devices": {"det": "mps", "cls": "mps", "rec": "mps"},
        "max_concurrency": 1,
    }
    assert to_legacy_result(backend.predict("image", detect=False, classify=False)) == [
        [[[[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]], ["hello", 0.9]]]
    ]
    assert engine.calls == [("image", {"use_det": False, "use_cls": False, "use_rec": True})]


@pytest.mark.parametrize("devices", [("cpu", "mps", "mps"), ("mps", "cpu", "mps"),
                                     ("mps", "mps", "cpu")])
def test_cpu_session_fails_closed(devices):
    backend = RapidOCRBackend(engine_factory=lambda: FakeEngine(devices),
                              mps_available=lambda: True)
    with pytest.raises(MPSUnavailableError, match="did not load on MPS"):
        backend.ensure_ready()


def test_unavailable_mps_returns_503_on_ready_and_inference():
    backend = RapidOCRBackend(
        engine_factory=lambda: pytest.fail("engine must not be created"),
        mps_available=lambda: False,
    )
    client = TestClient(create_app(backend))
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/ready")
    assert response.status_code == 503
    assert "CPU fallback is disabled" in response.json()["detail"]
    response = client.get("/ocr/predict-by-path", params={"image_path": "fixture"})
    assert response.status_code == 503


def test_engine_init_failure_and_lost_mps_fail_closed():
    def fail_factory():
        raise RuntimeError("internal model path should not be exposed")

    backend = RapidOCRBackend(engine_factory=fail_factory, mps_available=lambda: True)
    response = TestClient(create_app(backend)).get("/ready")
    assert response.status_code == 503
    assert response.json() == {"detail": "RapidOCR MPS initialization failed"}
    available = [True]
    backend = RapidOCRBackend(engine_factory=FakeEngine,
                              mps_available=lambda: available[0])
    backend.ensure_ready()
    available[0] = False
    with pytest.raises(MPSUnavailableError, match="CPU fallback is disabled"):
        backend.predict("fixture")


def test_engine_is_loaded_once_and_requests_are_serialized():
    state = {"loads": 0, "active": 0, "max_active": 0}
    lock = threading.Lock()

    class SlowEngine(FakeEngine):
        def __call__(self, image, **kwargs):
            with lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.02)
            with lock:
                state["active"] -= 1
            return super().__call__(image, **kwargs)

    def factory():
        state["loads"] += 1
        return SlowEngine()

    backend = RapidOCRBackend(engine_factory=factory, mps_available=lambda: True)
    threads = [threading.Thread(target=backend.predict, args=(str(i),)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert state == {"loads": 1, "active": 0, "max_active": 1}
