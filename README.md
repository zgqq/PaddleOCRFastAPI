# RapidOCR MPS FastAPI

[中文说明](README_CN.md)

This branch runs **RapidOCR 3.9.2 with PyTorch 2.8.0 MPS** on Apple Silicon macOS while preserving the legacy PaddleOCRFastAPI HTTP contract.

## Requirements

- Apple Silicon macOS
- Python 3.12
- PyTorch MPS available

CPU fallback is intentionally disabled. `/ready` returns HTTP 503 when MPS or model loading is unavailable.

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 18330 --workers 1
```

Health surfaces:

```bash
curl -fsS http://127.0.0.1:18330/health
curl -fsS http://127.0.0.1:18330/ready
```

Legacy OCR endpoints:

- `GET /ocr/predict-by-path`
- `POST /ocr/predict-by-base64`
- `POST /ocr/predict-by-file`
- `GET /ocr/predict-by-url`

The engine is a lazy singleton and inference is serialized at process concurrency 1. Do not add Uvicorn workers without new memory and correctness acceptance evidence.

```bash
python -m pytest tests -q
python -m compileall -q backends models routers utils main.py
```

Production launchd deployment and recovery scripts are maintained by `oh-my-server/macos/ocr-services/`. Clients use `http://ocr.prod.ai-infra.home.arpa`; direct host addresses are deployment and diagnostic surfaces only.

See [README_CN.md](README_CN.md) for compatibility limits, benchmark evidence, deployment gates, and rollback behavior.
