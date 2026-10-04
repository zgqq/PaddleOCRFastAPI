"""Run one isolated, real MPS HTTP smoke on an Apple Silicon target.

Requires existing requirements-mps.txt runtime; does not install, deploy, or
connect to a production listener. OCR_BACKEND must be explicitly selected.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import threading
import time
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import uvicorn
import cv2
import numpy as np
from main import create_app


def main():
    if os.environ.get("OCR_BACKEND") != "rapidocr-mps":
        raise SystemExit("Set OCR_BACKEND=rapidocr-mps for isolated acceptance")
    fixture = Path("screenshots/Swagger.png").read_bytes()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(5)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1",
                                           log_level="error", access_log=False))
    worker = threading.Thread(target=server.run, kwargs={"sockets": [sock]})
    worker.start()
    opener = build_opener(ProxyHandler({}))

    def request(path, payload=None):
        req = Request(f"http://127.0.0.1:{port}{path}",
                      data=json.dumps(payload).encode() if payload is not None else None,
                      headers={"Content-Type": "application/json"})
        begin = time.perf_counter()
        try:
            with opener.open(req, timeout=150) as response:
                return response.status, json.load(response), round(
                    (time.perf_counter() - begin) * 1000, 1)
        except HTTPError as exc:
            return exc.code, json.load(exc), round(
                (time.perf_counter() - begin) * 1000, 1)

    try:
        deadline = time.monotonic() + 15
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        if not server.started:
            raise RuntimeError("isolated HTTP server did not start")
        ready_code, ready, ready_ms = request("/ready")
        assert ready_code == 200 and ready["model_devices"] == {
            "det": "mps", "cls": "mps", "rec": "mps"
        }, (ready_code, ready)
        encoded = base64.b64encode(fixture).decode("ascii")
        code, result, inference_ms = request("/ocr/predict-by-base64", {
            "base64_str": encoded, "ocr_det": True, "ocr_cls": True,
        })
        assert code == 200 and result["resultcode"] == 200, (code, result)
        lines = result["data"][0]
        assert len(lines) == 14 and all(len(line) == 2 for line in lines), result
        assert lines[0][1][0] == "Paddle OCR API", result
        image = cv2.imdecode(np.frombuffer(fixture, dtype=np.uint8), cv2.IMREAD_COLOR)
        polygon = lines[0][0]
        x0 = max(0, int(min(point[0] for point in polygon)) - 4)
        y0 = max(0, int(min(point[1] for point in polygon)) - 4)
        x1 = min(image.shape[1], int(max(point[0] for point in polygon)) + 5)
        y1 = min(image.shape[0], int(max(point[1] for point in polygon)) + 5)
        assert x1 > x0 and y1 > y0
        success, crop = cv2.imencode(".png", image[y0:y1, x0:x1])
        assert success
        rec_code, rec, rec_ms = request("/ocr/predict-by-base64", {
            "base64_str": base64.b64encode(crop.tobytes()).decode("ascii"),
            "ocr_det": False, "ocr_cls": False,
        })
        assert rec_code == 200 and rec["data"][0][0][0] == "Paddle OCR API", (rec_code, rec)
        crop_encoded = base64.b64encode(crop.tobytes()).decode("ascii")
        # Mix mutable RapidOCR request modes at the real HTTP boundary.
        with ThreadPoolExecutor(max_workers=4) as pool:
            concurrent = list(pool.map(lambda flags: request("/ocr/predict-by-base64", {
                "base64_str": encoded if flags else crop_encoded,
                "ocr_det": flags, "ocr_cls": flags,
            }), (True, False, True, False)))
        for is_full, (parallel_code, parallel_result, _) in zip(
                (True, False, True, False), concurrent):
            assert parallel_code == 200, (parallel_code, parallel_result)
            parallel_lines = parallel_result["data"][0]
            if is_full:
                assert len(parallel_lines) == 14 and parallel_lines[0][1][0] == "Paddle OCR API"
            else:
                assert parallel_lines[0][0] == "Paddle OCR API", parallel_lines
        invalid_code, invalid, invalid_ms = request("/ocr/predict-by-base64", {
            "base64_str": "not-valid-image", "ocr_det": True, "ocr_cls": True,
        })
        assert invalid_code == 400 and "图片解码失败" in invalid["detail"], (invalid_code, invalid)
        print(json.dumps({
            "arch": platform.machine(), "os": platform.system(),
            "fixture": "screenshots/Swagger.png", "fixture_sha256": hashlib.sha256(fixture).hexdigest(),
            "ready": ready, "ready_ms": ready_ms,
            "detection_http": code, "detection_ms": inference_ms,
            "lines": len(lines), "sample_texts": [line[1][0] for line in lines[:5]],
            "recognition_http": rec_code, "recognition_ms": rec_ms,
            "recognition_sample": rec["data"][0][:2],
            "concurrent_http": [item[0] for item in concurrent],
            "concurrent_ms": [item[2] for item in concurrent],
            "invalid_image_http": invalid_code, "invalid_image_ms": invalid_ms,
        }, ensure_ascii=False))
    finally:
        server.should_exit = True
        worker.join(timeout=10)
        sock.close()
        if worker.is_alive():
            raise RuntimeError("isolated HTTP server did not stop")


if __name__ == "__main__":
    main()
