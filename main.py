# -*- coding: utf-8 -*-

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import ocr


def create_app(ocr_backend=None):
    app = FastAPI(title="Paddle OCR API",
                  description="基于 Paddle OCR 和 FastAPI 的自用接口")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )

    if ocr_backend is not None:
        app.dependency_overrides[ocr.get_ocr_backend] = lambda: ocr_backend
    app.include_router(ocr.router)
    return app


app = create_app()
