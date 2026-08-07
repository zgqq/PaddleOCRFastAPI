from backends.RapidOCRBackend import RapidOCRBackend
from routers.ocr import get_ocr_backend


def test_default_backend_is_rapidocr_mps():
    get_ocr_backend.cache_clear()
    assert isinstance(get_ocr_backend(), RapidOCRBackend)
