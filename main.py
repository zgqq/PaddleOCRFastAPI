# -*- coding: utf-8 -*-

from fastapi import Depends, FastAPI, HTTPException, status
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

    @app.get("/health", tags=["Health"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", tags=["Health"])
    def ready(backend=Depends(ocr.get_ocr_backend)):
        try:
            return backend.ensure_ready()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=str(exc),
            ) from exc

    app.include_router(ocr.router)
    return app


app = create_app()
