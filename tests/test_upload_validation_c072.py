# -*- coding: utf-8 -*-
"""
C072: upload validation contract tests.
- filename allowlist is case-normalized and includes .jpg/.jpeg/.png
- decode failure (valid extension but invalid content) returns 400 not 500
"""
from fastapi.testclient import TestClient
from main import create_app


class FakeOCRBackend:
    def predict(self, image, *, detect=True, classify=True):
        # Should not be called for invalid decode cases; for valid cases image must be decoded ndarray
        assert image is not None, "decode should have succeeded before predict"
        return [
            {
                "rec_polys": [[[1, 2], [3, 2], [3, 4], [1, 4]]],
                "rec_texts": ["ok"],
                "rec_scores": [0.99],
            }
        ]


def _valid_image_bytes():
    with open("screenshots/Swagger.png", "rb") as f:
        return f.read()


def test_file_accepts_jpg_lower():
    client = TestClient(create_app(FakeOCRBackend()))
    resp = client.post("/ocr/predict-by-file", files={"file": ("a.jpg", _valid_image_bytes(), "image/jpeg")})
    assert resp.status_code == 200


def test_file_accepts_jpg_upper():
    client = TestClient(create_app(FakeOCRBackend()))
    resp = client.post("/ocr/predict-by-file", files={"file": ("a.JPG", _valid_image_bytes(), "image/jpeg")})
    assert resp.status_code == 200


def test_file_accepts_jpeg_lower():
    client = TestClient(create_app(FakeOCRBackend()))
    resp = client.post("/ocr/predict-by-file", files={"file": ("a.jpeg", _valid_image_bytes(), "image/jpeg")})
    assert resp.status_code == 200


def test_file_accepts_jpeg_upper():
    client = TestClient(create_app(FakeOCRBackend()))
    resp = client.post("/ocr/predict-by-file", files={"file": ("a.JPEG", _valid_image_bytes(), "image/jpeg")})
    assert resp.status_code == 200


def test_file_accepts_png_lower():
    client = TestClient(create_app(FakeOCRBackend()))
    resp = client.post("/ocr/predict-by-file", files={"file": ("a.png", _valid_image_bytes(), "image/png")})
    assert resp.status_code == 200


def test_file_accepts_png_upper():
    client = TestClient(create_app(FakeOCRBackend()))
    resp = client.post("/ocr/predict-by-file", files={"file": ("a.PNG", _valid_image_bytes(), "image/png")})
    assert resp.status_code == 200


def test_file_accepts_mixed_case_jpg():
    client = TestClient(create_app(FakeOCRBackend()))
    resp = client.post("/ocr/predict-by-file", files={"file": ("a.JpG", _valid_image_bytes(), "image/jpeg")})
    assert resp.status_code == 200


def test_file_accepts_mixed_case_jpeg():
    client = TestClient(create_app(FakeOCRBackend()))
    resp = client.post("/ocr/predict-by-file", files={"file": ("a.JpEg", _valid_image_bytes(), "image/jpeg")})
    assert resp.status_code == 200


def test_file_rejects_txt_extension():
    client = TestClient(create_app(FakeOCRBackend()))
    resp = client.post("/ocr/predict-by-file", files={"file": ("a.txt", _valid_image_bytes(), "text/plain")})
    assert resp.status_code == 400
    assert ".jpg" in resp.json()["detail"]


def test_file_rejects_no_extension():
    client = TestClient(create_app(FakeOCRBackend()))
    resp = client.post("/ocr/predict-by-file", files={"file": ("afile", _valid_image_bytes(), "image/png")})
    assert resp.status_code == 400


def test_file_invalid_decoded_content_returns_400_not_500():
    """
    Valid extension but content is not a decodable image -> should be 400 via decode validation,
    not extension check and not 500 from backend.
    """
    class MustNotBeCalledBackend:
        def predict(self, image, *, detect=True, classify=True):
            raise AssertionError("predict must not be called for undecodable content")

    client = TestClient(create_app(MustNotBeCalledBackend()))
    fake_bytes = b"this is not an image"
    for filename in ["bad.jpg", "bad.JPG", "bad.jpeg", "bad.JPEG", "bad.png", "bad.PNG"]:
        resp = client.post("/ocr/predict-by-file", files={"file": (filename, fake_bytes, "image/jpeg")})
        assert resp.status_code == 400, f"expected 400 for {filename} with invalid content"
        assert "解码失败" in resp.json()["detail"]


def test_file_empty_content_with_valid_extension_returns_400():
    class MustNotBeCalledBackend:
        def predict(self, image, *, detect=True, classify=True):
            raise AssertionError("predict must not be called for empty content")

    client = TestClient(create_app(MustNotBeCalledBackend()))
    resp = client.post("/ocr/predict-by-file", files={"file": ("empty.jpg", b"", "image/jpeg")})
    assert resp.status_code == 400
    assert "解码失败" in resp.json()["detail"]


def test_url_invalid_decoded_content_returns_400(monkeypatch):
    """predict-by-url with JPEG/PNG magic but undecodable payload should also be 400, not 500"""
    class MustNotBeCalledBackend:
        def predict(self, image, *, detect=True, classify=True):
            raise AssertionError("predict must not be called for undecodable url content")

    class FakeResponse:
        # JPEG magic but following bytes are corrupted / not decodable
        content = b"\xff\xd8\xff" + b"not-an-image"

    monkeypatch.setattr("routers.ocr.requests.get", lambda url: FakeResponse())
    client = TestClient(create_app(MustNotBeCalledBackend()))
    resp = client.get("/ocr/predict-by-url", params={"imageUrl": "https://example.test/bad.jpg"})
    assert resp.status_code == 400
    assert "解码失败" in resp.json()["detail"]


def test_url_non_image_magic_returns_400(monkeypatch):
    class FakeResponse:
        content = b"plain text not image"

    monkeypatch.setattr("routers.ocr.requests.get", lambda url: FakeResponse())
    client = TestClient(create_app(FakeOCRBackend()))
    resp = client.get("/ocr/predict-by-url", params={"imageUrl": "https://example.test/bad.txt"})
    assert resp.status_code == 400
