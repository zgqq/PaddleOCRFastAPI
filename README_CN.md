# PaddleOCRFastAPI

基于 FastAPI 的轻量 OCR HTTP 服务。本分支面向 **Apple Silicon macOS / CPU**，使用：

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

## 兼容边界

- HTTP路径、请求参数、外层字段、文字框/文本/置信度结构兼容旧版。
- PP-OCRv4升级为PP-OCRv6后，检测框、文本、置信度和行顺序可能变化，不保证响应逐字节一致。
- `predict-by-path`读取的是**服务所在Mac的本地路径**。旧WSL路径不会自动映射到Mac。
- PP-OCRv6 small是统一多语言模型，不再使用旧版`OCR_LANGUAGE`切换模型。
- 本分支的macOS直接运行方式已验证；Docker尚未验证。

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
