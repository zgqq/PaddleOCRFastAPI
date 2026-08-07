# RapidOCR MPS FastAPI

在 Apple Silicon macOS 上运行的 OCR HTTP 服务。当前分支使用 **RapidOCR 3.9.2 + PyTorch 2.8.0 MPS**，并保持原 PaddleOCRFastAPI 的四个接口和主要返回结构。

## 分支职责

- `master` / `gpu*`：旧 PaddleOCR 实现，包含 Windows CUDA 部署路径。
- `macos-ppocr-v6`：macOS PaddlePaddle CPU 兼容基线。
- `macos-rapidocr-mps`：Apple Silicon 生产候选，RapidOCR Torch MPS；本分支。

不要在本分支静默回退到 CPU。`/ready` 在 PyTorch MPS 不可用或模型加载失败时返回 HTTP 503。

## 运行环境

- Apple Silicon Mac
- macOS
- Python 3.12
- RapidOCR 3.9.2
- PyTorch 2.8.0，`torch.backends.mps.is_available() == True`

推荐通过 `mise` 或 `uv`使用独立 Python 3.12，不使用 macOS 系统 Python 3.9。

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m uvicorn main:app --host 0.0.0.0 --port 18330 --workers 1
```

首次访问`/ready`会构建单例OCR引擎并加载模型。推理由进程内锁串行化，初始最大并发为1；不要通过增加Uvicorn workers绕过该限制。

## 健康检查

```bash
curl -fsS http://127.0.0.1:18330/health
curl -fsS http://127.0.0.1:18330/ready
```

- `/health`：轻量liveness，不触发模型加载。
- `/ready`：验证MPS可用并加载RapidOCR引擎；成功后返回`device=mps`和`loaded=true`。

## OCR接口

- `GET /ocr/predict-by-path`
- `POST /ocr/predict-by-base64`
- `POST /ocr/predict-by-file`
- `GET /ocr/predict-by-url`

完整检测返回旧格式的四点框、文字和置信度。`ocr_det=false`保留旧recognition-only形状；空白检测页返回`data: [null]`。

接口层保持兼容，但不同OCR引擎的行序、框坐标和置信度不是数值等价保证。切流必须使用真实业务图片验收。

## 测试

```bash
python -m pytest tests -q
python -m compileall -q backends models routers utils main.py
```

单元测试通过fake engine验证，不下载模型。生产验收还必须在Apple Silicon目标机完成：`/ready`、真实图片OCR、重复推理、内存/swap以及与embedding/reranker共存测试。

## 已验证的M4基准

在192.168.11.222的Apple M4上，三类样本的RapidOCR MPS中位延迟约为0.218–0.428秒，比同模型RapidOCR CPU快约1.8–1.9倍；仍比旧Windows CUDA服务慢约3.25–5.96倍。MPS进程RSS约1.38GB、峰值footprint约2.47GB。以上为benchmark，不替代生产长稳验收。

## 部署和回滚

launchd安装、升级、健康检查及恢复脚本由`oh-my-server/macos/ocr-services/`维护。客户端只使用：

```text
http://ocr.prod.ai-infra.home.arpa
```

192.168.11.99:8000旧PaddleOCR CUDA服务在迁移窗口内仅作为HAProxy回滚目标，不应继续写入客户端代码。
