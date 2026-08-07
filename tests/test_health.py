from fastapi.testclient import TestClient

from main import create_app


class ReadyBackend:
    def __init__(self):
        self.calls = 0

    def ensure_ready(self):
        self.calls += 1
        return {
            "status": "ready",
            "backend": "rapidocr",
            "device": "mps",
            "mps_available": True,
            "loaded": True,
            "max_concurrency": 1,
        }


class UnavailableBackend:
    def ensure_ready(self):
        raise RuntimeError("PyTorch MPS is unavailable; CPU fallback is disabled")


def test_health_is_lightweight_liveness():
    backend = ReadyBackend()
    response = TestClient(create_app(backend)).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert backend.calls == 0


def test_ready_loads_the_mps_engine_and_reports_status():
    backend = ReadyBackend()
    response = TestClient(create_app(backend)).get("/ready")

    assert response.status_code == 200
    assert response.json()["device"] == "mps"
    assert response.json()["loaded"] is True
    assert backend.calls == 1


def test_ready_fails_closed_when_mps_cannot_start():
    response = TestClient(create_app(UnavailableBackend())).get("/ready")

    assert response.status_code == 503
    assert "CPU fallback is disabled" in response.json()["detail"]
