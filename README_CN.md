# PaddleOCRFastAPI

基于 FastAPI 的轻量 OCR HTTP 服务。默认仍为 **Apple Silicon macOS / CPU** 的 PaddleOCR PP-OCRv6 small。可显式选择 RapidOCR 3.9.2 + PyTorch 2.8.0 的 MPS 后端；本项目仍是**不可部署的开发实验**，不代表生产发布：

- PaddlePaddle 3.3.0
- PaddleOCR 3.7.0
- PP-OCRv6 small（检测与识别）

## 兼容接口

保留旧服务的四个接口和外层 JSON 契约：

- `GET /ocr/predict-by-path`
- `POST /ocr/predict-by-base64`
- `POST /ocr/predict-by-file`
- `GET /ocr/predict-by-url`

返回结构：

```json
{
  "resultcode": 200,
  "message": "Success",
  "data": [
    [
      [
        [[33.0, 67.0], [240.0, 68.0], [239.0, 91.0], [32.0, 90.0]],
        ["识别文本", 0.98]
      ]
    ]
  ]
}
```

`ocr_det=false`时保留旧识别-only结构：

```json
{
  "data": [[[
    "识别文本",
    0.98
  ]]]
}
```

空检测页保留旧兼容结构：`{"resultcode": 200, "message": "Success", "data": [null]}`（上传接口的 `message` 仍为文件名）。它不是错误，也不是 `[]`。

## 执行与 URL 等待边界

file/url 与 path/base64 接口均使用同步 FastAPI path operation，在框架线程池运行文件读取、网络、解码和推理，不占用 ASGI 事件循环等待。RapidOCR MPS 的 backend `max_concurrency=1` 保持不变：事件循环可服务其他请求，不代表 GPU 推理并行。

URL 请求设置 5 秒连接 timeout、30 秒 socket 读取无数据 timeout；这是 Requests 的 connect/read-inactivity 预算，不是整个下载或推理的墙钟总 deadline，也不表示客户端断开会取消正在运行的同步推理。

## 兼容边界

- HTTP路径、请求参数、外层字段、文字框/文本/置信度结构兼容旧版。
- `ocr_cls` 参数仍被接受，但 CPU `ocr_det=false` 分支直接调用 `TextRecognition.predict(image)`，不消费该分类选项；RapidOCR MPS 会转发 `use_cls`。不能把 recognition-only 的参数接受解释为 CPU 分类已执行。
- PP-OCRv4升级为PP-OCRv6后，检测框、文本、置信度和行顺序可能变化，不保证响应逐字节一致。
- 本实现仅切换后端，不改变HTTP路径及外层响应结构；MPS后端的识别结果、分数、行序可能不同。
- `OCR_BACKEND` 默认 `paddle`；只有显式设置 `rapidocr-mps` 才会引入 RapidOCR/PyTorch，未知值直接报错。不同依赖请使用隔离虚拟环境，不要将两套依赖混装进生产运行时。
- `/health` 只反映进程存活；`/ready` 加载所选后端。MPS后端校验det/cls/rec三个模型实际运行设备均为mps，MPS不可用或模型退回CPU时返回503，不会自动回退。
- `predict-by-path`读取的是**服务所在Mac的本地路径**。旧WSL路径不会自动映射到Mac。
- PP-OCRv6 small是统一多语言模型，不再使用旧版`OCR_LANGUAGE`切换模型。
- 默认CPU模式此前已在Mac直接运行验证；本候选额外验证了Mac上的全新MPS虚拟环境安装及本地Docker的CPU边界。详情见[readiness证据](docs/mps-readiness-evidence.md)。Docker不提供Apple Metal/MPS。

## 安装

建议使用独立Python虚拟环境：

```shell
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

已在Apple M4、macOS arm64、Python 3.9环境验证依赖安装和真实推理。

## 运行

```shell
.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
```

首次OCR请求会下载并加载PP-OCRv6 small模型，后续请求复用同一模型实例。

可选MPS模式：仅在Apple Silicon macOS/Python 3.12的**独立**虚拟环境安装`requirements-mps.txt`，显式设置`OCR_BACKEND=rapidocr-mps`后运行同一个`uvicorn main:app`。不设置时仍运行默认CPU模式。全新安装已在隔离scratch验收，模型从ModelScope下载到该虚拟环境内。启动后可请求`/ready`判断模型设备。未知后端值不会自动降级。为确保PyTorch内核也不会静默退回CPU，运行时设置`PYTORCH_ENABLE_MPS_FALLBACK=0`；应用检测到`=1`会拒绝就绪/推理。模型设备校验和关闭该开关不等于逐个内核的性能归因。

在目标Mac mini的隔离候选副本与独立虚拟环境中执行`PYTORCH_ENABLE_MPS_FALLBACK=0 OCR_BACKEND=rapidocr-mps .venv/bin/python -m scripts.accept_mps`：仅在127.0.0.1临时端口启动，固定`screenshots/Swagger.png`（SHA256见证据）测试检测/识别、真实裁剪识别-only、混合并发和坏图片400，并关闭临时服务；重复执行验证进程重启。**不要以生产实例的/ready结果代替候选验收**。

历史候选（见[MPS rebuild evidence](docs/mps-rebuild-evidence.md)）在目标Mac mini的隔离副本中复用既有虚拟环境完成真实HTTP推理：macOS arm64、RapidOCR 3.9.2/PyTorch 2.8.0，det/cls/rec实际设备均为mps；固定fixture SHA256 `9fef56ac8b7fc34931a1e0315e5639687e8286f0799ced758479a282d52145cf`，检测14行（包含`Paddle OCR API`），热推理约1014ms；第一行裁剪识别-only返回`Paddle OCR API`、约104ms；坏Base64图片HTTP 400、约1ms。该历史测试未验证新安装或Docker；本轮全新安装、Docker及清理证据单独见[readiness证据](docs/mps-readiness-evidence.md)。延迟只是单次观测，不是性能保证。

Swagger：<http://localhost:8000/docs>

## 测试

测试不需要下载Paddle模型：

```shell
python -m pytest tests -q
```

测试覆盖：

- 四个HTTP接口的旧响应契约；
- PaddleOCR 3.x Result到旧数据结构的转换；
- `ocr_det=false`结构；
- PP-OCRv6 small CPU模型选择和懒加载；
- `ocr_cls`到文字行方向分类参数的映射。

## License

MIT。详见 [LICENSE](LICENSE)。
