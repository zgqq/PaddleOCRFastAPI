# -*- coding: utf-8 -*-

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from models.OCRModel import Base64PostModel
from models.RestfulModel import RestfulModel
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray
from utils.OCRResultAdapter import to_legacy_result
import requests

router = APIRouter(prefix="/ocr", tags=["OCR"])

# 上传校验契约：文件名后缀 allowlist 仅作初步过滤（大小写不敏感，支持 .jpg/.jpeg/.png），
# 最终以 bytes_to_ndarray 解码结果为准；解码失败返回 400，避免不可信扩展名/MIME 导致 500。
ALLOWED_FILE_EXTENSIONS = (".jpg", ".jpeg", ".png")
ALLOWED_FILE_EXTENSIONS_LABEL = ".jpg/.jpeg 或 .png"

@lru_cache
def get_ocr_backend():
    from backends.PaddleOCRBackend import PaddleOCRBackend

    return PaddleOCRBackend()


@router.get('/predict-by-path', response_model=RestfulModel, summary="识别本地图片")
def predict_by_path(
        image_path: str,
        ocr_det: bool = True,
        ocr_cls: bool = True,
        ocr_backend=Depends(get_ocr_backend)):
    result = ocr_backend.predict(
        image_path,
        detect=ocr_det,
        classify=ocr_cls,
    )
    restfulModel = RestfulModel(
        resultcode=200,
        message="Success",
        data=to_legacy_result(result))
    return restfulModel


@router.post('/predict-by-base64', response_model=RestfulModel, summary="识别 Base64 数据", description="Base64 解码后以实际图像解码为准，解码失败返回 400。")
def predict_by_base64(
        base64model: Base64PostModel,
        ocr_backend=Depends(get_ocr_backend)):
    try:
        img = base64_to_ndarray(base64model.base64_str)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图片解码失败，请上传有效的 Base64 编码的 .jpg/.jpeg 或 .png 图片"
        )
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图片解码失败，请上传有效的 Base64 编码的 .jpg/.jpeg 或 .png 图片"
        )
    result = ocr_backend.predict(
        img,
        detect=base64model.ocr_det,
        classify=base64model.ocr_cls,
    )
    restfulModel = RestfulModel(
        resultcode=200,
        message="Success",
        data=to_legacy_result(result))
    return restfulModel


@router.post('/predict-by-file', response_model=RestfulModel, summary="识别上传文件", description="仅接受 .jpg/.jpeg/.png（大小写不敏感）；以实际解码为准，解码失败返回 400。")
async def predict_by_file(
        file: UploadFile,
        ocr_det: bool = True,
        ocr_cls: bool = True,
        ocr_backend=Depends(get_ocr_backend)):
    filename = file.filename or ""
    # 校验 1：文件名后缀 allowlist（大小写不敏感，含 .jpeg）
    if not filename.lower().endswith(ALLOWED_FILE_EXTENSIONS):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请上传 {ALLOWED_FILE_EXTENSIONS_LABEL} 格式图片"
        )
    file_bytes = await file.read()
    # 校验 2：实际内容解码校验，不信任扩展名/MIME；解码失败返回受控 400 而非 500
    img = bytes_to_ndarray(file_bytes)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图片解码失败，请上传有效的 .jpg/.jpeg 或 .png 图片"
        )
    result = ocr_backend.predict(img, detect=ocr_det, classify=ocr_cls)
    restfulModel: RestfulModel = RestfulModel(
        resultcode=200,
        message=filename,
        data=to_legacy_result(result)
    )
    return restfulModel


@router.get('/predict-by-url', response_model=RestfulModel, summary="识别图片 URL", description="仅接受 JPEG/PNG 内容（以魔数及解码为准），解码失败返回 400。")
async def predict_by_url(
        imageUrl: str,
        ocr_det: bool = True,
        ocr_cls: bool = True,
        ocr_backend=Depends(get_ocr_backend)):
    response = requests.get(imageUrl)
    image_bytes = response.content
    # 仅处理 JPEG/PNG：先以魔数快速过滤，最终以解码结果为准
    if not (image_bytes.startswith(b"\xff\xd8\xff") or image_bytes.startswith(b"\x89PNG\r\n\x1a\n")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"请上传 {ALLOWED_FILE_EXTENSIONS_LABEL} 格式图片"
        )
    img = bytes_to_ndarray(image_bytes)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="图片解码失败，请上传有效的 .jpg/.jpeg 或 .png 图片"
        )
    result = ocr_backend.predict(img, detect=ocr_det, classify=ocr_cls)
    restfulModel: RestfulModel = RestfulModel(
        resultcode=200,
        message="Success",
        data=to_legacy_result(result)
    )
    return restfulModel
