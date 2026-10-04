# PaddleOCRFastAPI

FastAPI OCR service based on PP-OCRv6 small. PaddleOCR on CPU remains the default; `OCR_BACKEND=rapidocr-mps` selects an optional Apple Silicon PyTorch MPS runtime. This repository remains a **non-deployable development experiment**; no production promotion is implied.

For the current API contract, installation steps, compatibility limits, and tests, see [README_CN.md](README_CN.md). See [MPS rebuild evidence](docs/mps-rebuild-evidence.md) for the historical-branch disposition and isolated target acceptance.

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
- The default PaddleOCR/CPU install is unchanged. For the optional macOS arm64/Python 3.12 backend, install `requirements-mps.txt` in a separate virtualenv, set `OCR_BACKEND=rapidocr-mps`, and run `uvicorn main:app`.
- `/health` is a liveness check; `/ready` loads the selected backend and reports its actual device. RapidOCR verifies all three Torch model sessions are on MPS and fails with 503 instead of falling back to CPU.
- Run `python -m scripts.accept_mps` from the repository root for an isolated loopback HTTP smoke on Apple Silicon with the existing `screenshots/Swagger.png` fixture (no production listener).
- PP-OCRv6 uses a unified multilingual model; the legacy `OCR_LANGUAGE` switch is not supported.
- Docker and a fresh RapidOCR/MPS dependency install have not been verified.

## Tests

```shell
python -m pytest tests -q
```

## License

MIT. See [LICENSE](LICENSE).
