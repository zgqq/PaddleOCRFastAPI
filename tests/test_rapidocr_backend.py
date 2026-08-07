import threading
import time

import pytest

from backends.RapidOCRBackend import RapidOCRBackend


class FakeEngine:
    def __init__(self):
        self.calls = []

    def __call__(self, image, **kwargs):
        self.calls.append((image, kwargs))
        return {"result": "ok"}


def test_rapidocr_backend_uses_torch_mps_and_preserves_request_flags():
    captured = {}
    engine = FakeEngine()

    def engine_factory(*, params):
        captured.update(params)
        return engine

    backend = RapidOCRBackend(
        engine_factory=engine_factory,
        mps_available=lambda: True,
    )

    assert backend.predict("image", detect=False, classify=False) == {"result": "ok"}
    assert captured["Det.engine_type"].value == "torch"
    assert captured["Cls.engine_type"].value == "torch"
    assert captured["Rec.engine_type"].value == "torch"
    assert captured["EngineConfig.torch.use_mps"] is True
    assert engine.calls == [
        (
            "image",
            {"use_det": False, "use_cls": False, "use_rec": True},
        )
    ]


def test_rapidocr_backend_fails_closed_when_mps_is_unavailable():
    backend = RapidOCRBackend(
        engine_factory=lambda **kwargs: pytest.fail("engine must not be built"),
        mps_available=lambda: False,
    )

    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        backend.ensure_ready()


def test_rapidocr_backend_lazy_loads_one_engine_and_serializes_inference():
    state = {"factory_calls": 0, "active": 0, "max_active": 0}
    state_lock = threading.Lock()

    class SlowEngine:
        def __call__(self, image, **kwargs):
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.02)
            with state_lock:
                state["active"] -= 1
            return image

    def engine_factory(*, params):
        state["factory_calls"] += 1
        return SlowEngine()

    backend = RapidOCRBackend(
        engine_factory=engine_factory,
        mps_available=lambda: True,
    )
    threads = [threading.Thread(target=backend.predict, args=(index,)) for index in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert state["factory_calls"] == 1
    assert state["max_active"] == 1
