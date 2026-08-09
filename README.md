# PaddleOCRFastAPI

FastAPI OCR service for the `macos-ppocr-v6` branch. This branch targets Apple Silicon macOS / CPU and uses PaddlePaddle 3.3.0, PaddleOCR 3.7.0, and PP-OCRv6 small.

For the current API contract, installation steps, compatibility limits, and tests, see [README_CN.md](README_CN.md).

## Quick start

```shell
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

Swagger UI: <http://localhost:8000/docs>

## Compatibility notes

- The four legacy `/ocr/predict-*` endpoints and outer JSON shape are preserved.
- PP-OCRv6 uses a unified multilingual model; the legacy `OCR_LANGUAGE` switch is not supported by this branch.
- Direct macOS execution has been verified. Docker remains an unverified compatibility path.

## Tests

```shell
python -m pytest tests -q
```

## License

MIT. See [LICENSE](LICENSE).
