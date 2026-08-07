import sys
from types import SimpleNamespace

from backends.PaddleOCRBackend import PaddleOCRBackend


def test_backend_lazily_loads_ppocr_v6_cpu_and_maps_orientation(monkeypatch):
    created = []
    calls = []

    class FakePaddleOCR:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def predict(self, image, **kwargs):
            calls.append((image, kwargs))
            return ["detected"]

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=FakePaddleOCR, TextRecognition=object),
    )
    backend = PaddleOCRBackend()

    assert created == []
    assert backend.predict("image", detect=True, classify=False) == ["detected"]
    assert created == [
        {
            "text_detection_model_name": "PP-OCRv6_small_det",
            "text_recognition_model_name": "PP-OCRv6_small_rec",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": True,
            "text_rec_score_thresh": 0.5,
            "device": "cpu",
        }
    ]
    assert calls == [("image", {"use_textline_orientation": False})]


def test_backend_uses_text_recognition_when_detection_is_disabled(monkeypatch):
    created = []
    calls = []

    class FakeTextRecognition:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def predict(self, image):
            calls.append(image)
            return ["recognized"]

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        SimpleNamespace(PaddleOCR=object, TextRecognition=FakeTextRecognition),
    )
    backend = PaddleOCRBackend()

    assert backend.predict("image", detect=False, classify=True) == ["recognized"]
    assert created == [{"model_name": "PP-OCRv6_small_rec", "device": "cpu"}]
    assert calls == ["image"]
