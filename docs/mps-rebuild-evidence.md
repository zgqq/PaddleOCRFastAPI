# RapidOCR/MPS candidate evidence

Baseline `macos-ppocr-v6` at `e8ea746`; historical branch `macos-rapidocr-mps` at `8e83d1c` used only as a comparison, not merged. Its complete 13-path commit inventory and disposition:

| Historical paths | Disposition on current baseline |
|---|---|
| `README.md`, `README_CN.md` | Rewritten around the unchanged PaddleCPU default and opt-in MPS; old deployment claims rejected. |
| `backends/RapidOCRBackend.py` | Reimplemented with lazy loading, serialization, fail-closed MPS and actual three-session device verification. |
| `main.py`, `routers/ocr.py` | Added explicit backend selection and scoped liveness/readiness; retained current route contracts and upload validation. |
| `project.manifest.yaml` | Retained `kind: experiment`, `lifecycle_status: development`, and non-deployable semantics; historical `kind: service`, `app_id: rapidocr-mps`, `deployment` and module claims rejected. |
| `requirements.in`, `requirements.txt` | Retained current PaddleCPU pins; optional RapidOCR pins isolated in `requirements-mps.txt`. |
| `tests/test_backend_factory.py`, `tests/test_health.py`, `tests/test_rapidocr_backend.py`, `tests/test_ocr_result_adapter.py` | Rebuilt relevant selection, readiness, fail-closed, concurrency and result-shape coverage against current API; no old tests transplanted. |
| `utils/OCRResultAdapter.py` | Extended current Paddle result adapter with RapidOCR detection/recognition-only/empty handling and cardinality checks. |

Current default is PaddleOCR 3.7.0/PP-OCRv6 small CPU. Explicit `OCR_BACKEND=rapidocr-mps` uses RapidOCR 3.9.2 and torch 2.8.0 in an independent environment. The target machine is Resource Manager `home.machine.claw-gateway-macmini`; authorized SSH was used for read-only inspection and a temporary, loopback-only candidate process. Candidate source was copied to a user-cache scratch directory and deleted afterward; no production source, package, or service was changed. Its existing MPS virtualenv and cached models were reused. A freshly provisioned MPS environment, Docker path, production release and routing have **not** been accepted.

## Evidence matrix

| Requirement | Boundary | Result |
|---|---|---|
| Current CPU contract | Linux isolated test venv; `python -m pytest tests -q` | 34 tests pass, including all four legacy HTTP routes; no Paddle model download. |
| Build/syntax | `python -m compileall -q backends models routers utils main.py scripts` | Pass. |
| MPS unavailable / CPU fallback | HTTP `GET /ready` and OCR path with unavailable MPS, model session CPU, or engine init failure | 503 fail-closed (unit/API tests); no remote MPS outage induced. |
| Real target MPS device | macOS arm64 candidate process, existing RapidOCR/PyTorch runtime | `/ready` 200 with det/cls/rec sessions all `mps`; model loader logs also report MPS. |
| Real fixture inference | Isolated `127.0.0.1` HTTP `POST /ocr/predict-by-base64`; `screenshots/Swagger.png`, SHA256 `9fef56ac8b7fc34931a1e0315e5639687e8286f0799ced758479a282d52145cf` | HTTP 200, 14 lines, first text `Paddle OCR API`, 1013.8ms detection; cropped first line recognition-only HTTP 200, `Paddle OCR API` at confidence 0.96982, 103.5ms. |
| Invalid input | Same isolated HTTP service, malformed image Base64 | HTTP 400 `图片解码失败`, 0.9ms; no OCR inference. |
| Isolation/cleanup | Temporary remote cache dir and loopback listener | Runner shuts server down and shell trap removes isolated source. No production switch/restart/install. |

Latency numbers are one observed run, not a benchmark. The actual model sessions are MPS; this does not prove every individual PyTorch kernel executes without a framework-level fallback. The target run reused the already-installed runtime, not a fresh install of the optional requirements.
