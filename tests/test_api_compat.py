from fastapi.testclient import TestClient

from main import create_app


class FakeOCRBackend:
    def predict(self, image, *, detect=True, classify=True):
        assert image is not None
        assert detect is True
        assert classify is True
        return [
            {
                "rec_polys": [[[33, 67], [240, 68], [239, 91], [32, 90]]],
                "rec_texts": ["http://localhost:8000/openapijson"],
                "rec_scores": [0.9633703231811523],
            }
        ]


def expected_legacy_data():
    return [
        [
            [
                [[33.0, 67.0], [240.0, 68.0], [239.0, 91.0], [32.0, 90.0]],
                ["http://localhost:8000/openapijson", 0.9633703231811523],
            ]
        ]
    ]


def test_file_endpoint_preserves_legacy_response_contract():
    client = TestClient(create_app(FakeOCRBackend()))

    with open("screenshots/Swagger.png", "rb") as image:
        response = client.post(
            "/ocr/predict-by-file",
            files={"file": ("Swagger.png", image, "image/png")},
        )

    assert response.status_code == 200
    assert response.json() == {
        "resultcode": 200,
        "message": "Swagger.png",
        "data": expected_legacy_data(),
    }


def test_base64_endpoint_preserves_legacy_response_contract():
    import base64

    client = TestClient(create_app(FakeOCRBackend()))
    with open("screenshots/Swagger.png", "rb") as image:
        encoded = base64.b64encode(image.read()).decode()

    response = client.post(
        "/ocr/predict-by-base64",
        json={"base64_str": encoded, "ocr_det": True, "ocr_cls": True},
    )

    assert response.status_code == 200
    assert response.json() == {
        "resultcode": 200,
        "message": "Success",
        "data": expected_legacy_data(),
    }


def test_path_endpoint_preserves_legacy_response_contract():
    client = TestClient(create_app(FakeOCRBackend()))

    response = client.get(
        "/ocr/predict-by-path",
        params={"image_path": "screenshots/Swagger.png"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "resultcode": 200,
        "message": "Success",
        "data": expected_legacy_data(),
    }


def test_url_endpoint_preserves_legacy_response_contract(monkeypatch):
    class ImageResponse:
        content = open("screenshots/Swagger.png", "rb").read()

    monkeypatch.setattr("routers.ocr.requests.get", lambda url: ImageResponse())
    client = TestClient(create_app(FakeOCRBackend()))

    response = client.get(
        "/ocr/predict-by-url",
        params={"imageUrl": "https://example.test/Swagger.png"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "resultcode": 200,
        "message": "Success",
        "data": expected_legacy_data(),
    }
