from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from rapidocr import EngineType, RapidOCR


EngineFactory = Callable[..., Any]
MPSAvailability = Callable[[], bool]


def _torch_mps_available() -> bool:
    import torch

    return bool(
        torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


class RapidOCRBackend:
    """Lazy singleton RapidOCR engine that fails closed without Apple MPS."""

    def __init__(
        self,
        *,
        engine_factory: EngineFactory = RapidOCR,
        mps_available: MPSAvailability = _torch_mps_available,
    ) -> None:
        self._engine_factory = engine_factory
        self._mps_available = mps_available
        self._engine: Any | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._load_lock:
            if self._engine is not None:
                return self._engine
            if not self._mps_available():
                raise RuntimeError("PyTorch MPS is unavailable; CPU fallback is disabled")
            params = {
                "Det.engine_type": EngineType.TORCH,
                "Cls.engine_type": EngineType.TORCH,
                "Rec.engine_type": EngineType.TORCH,
                "EngineConfig.torch.use_mps": True,
            }
            self._engine = self._engine_factory(params=params)
            return self._engine

    def ensure_ready(self) -> dict[str, object]:
        self._load_engine()
        return self.status()

    def status(self) -> dict[str, object]:
        return {
            "status": "ready" if self._engine is not None else "not_ready",
            "backend": "rapidocr",
            "device": "mps",
            "mps_available": self._mps_available(),
            "loaded": self._engine is not None,
            "max_concurrency": 1,
        }

    def predict(self, image: Any, *, detect: bool = True, classify: bool = True) -> Any:
        engine = self._load_engine()
        with self._inference_lock:
            return engine(
                image,
                use_det=detect,
                use_cls=classify,
                use_rec=True,
            )
