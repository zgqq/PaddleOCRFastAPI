"""Explicit, fail-closed RapidOCR 3.9.2 / PyTorch MPS backend.

The session.device checks are intentionally version-specific: a ready result must
prove that all three model sessions actually loaded on MPS, not just that MPS is
available or that use_mps was requested. Pin RapidOCR to 3.9.2 when using this.
"""

import os
import threading


class MPSUnavailableError(RuntimeError):
    """The requested accelerator is absent or a model did not adopt it."""


def _mps_available():
    import platform

    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.backends.mps.is_built() and torch.backends.mps.is_available()


def _make_engine():
    from rapidocr import EngineType, RapidOCR

    return RapidOCR(params={
        "Det.engine_type": EngineType.TORCH,
        "Cls.engine_type": EngineType.TORCH,
        "Rec.engine_type": EngineType.TORCH,
        "EngineConfig.torch.use_mps": True,
    })


class RapidOCRBackend:
    def __init__(self, *, engine_factory=None, mps_available=None):
        self._engine_factory = engine_factory or _make_engine
        self._mps_available = mps_available or _mps_available
        self._engine = None
        self._load_lock = threading.Lock()
        # RapidOCR mutates its request flags during __call__; serialize requests.
        self._inference_lock = threading.Lock()

    @staticmethod
    def _check_devices(engine):
        try:
            devices = {
                name: getattr(engine, attr).session.device.type
                for name, attr in (
                    ("det", "text_det"), ("cls", "text_cls"), ("rec", "text_rec")
                )
            }
        except (AttributeError, KeyError, TypeError) as exc:
            raise MPSUnavailableError("RapidOCR model device cannot be verified") from exc
        if any(device != "mps" for device in devices.values()):
            raise MPSUnavailableError("RapidOCR model did not load on MPS")
        return devices

    def _load_engine(self):
        # PyTorch accepts this switch at import time; reject it even when all
        # model parameters report MPS, since unsupported kernels could use CPU.
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise MPSUnavailableError("PyTorch MPS CPU kernel fallback must be disabled")
        if not self._mps_available():
            raise MPSUnavailableError("PyTorch MPS is unavailable; CPU fallback is disabled")
        if self._engine is None:
            with self._load_lock:
                if self._engine is None:
                    try:
                        engine = self._engine_factory()
                    except Exception as exc:
                        raise MPSUnavailableError("RapidOCR MPS initialization failed") from exc
                    self._check_devices(engine)
                    self._engine = engine
        return self._engine

    def ensure_ready(self):
        engine = self._load_engine()
        return {
            "status": "ready",
            "backend": "rapidocr-mps",
            "device": "mps",
            "model_devices": self._check_devices(engine),
            "max_concurrency": 1,
        }

    def predict(self, image, *, detect=True, classify=True):
        engine = self._load_engine()
        with self._inference_lock:
            # Some models can be moved after construction; fail before each call.
            self._check_devices(engine)
            return engine(image, use_det=detect, use_cls=classify, use_rec=True)
