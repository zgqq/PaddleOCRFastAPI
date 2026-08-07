# -*- coding: utf-8 -*-

from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from models.OCRModel import Base64PostModel
from models.RestfulModel import RestfulModel
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray
from utils.OCRResultAdapter import to_legacy_result
import requests

router = APIRouter(prefix="/ocr", tags=["OCR"])

@lru_cache
def get_ocr_backend():
    from backends.RapidOCRBackend import RapidOCRBackend

    return RapidOCRBackend()


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


@router.post('/predict-by-base64', response_model=RestfulModel, summary="识别 Base64 数据")
def predict_by_base64(
        base64model: Base64PostModel,
        ocr_backend=Depends(get_ocr_backend)):
    img = base64_to_ndarray(base64model.base64_str)
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


@router.post('/predict-by-file', response_model=RestfulModel, summary="识别上传文件")
async def predict_by_file(
        file: UploadFile,
        ocr_det: bool = True,
        ocr_cls: bool = True,
        ocr_backend=Depends(get_ocr_backend)):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png")):  # 只处理常见格式图片
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
        file_data = file.file
        file_bytes = file_data.read()
        img = bytes_to_ndarray(file_bytes)
        result = ocr_backend.predict(img, detect=ocr_det, classify=ocr_cls)
        restfulModel.data = to_legacy_result(result)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片"
        )
    return restfulModel


@router.get('/predict-by-url', response_model=RestfulModel, summary="识别图片 URL")
async def predict_by_url(
        imageUrl: str,
        ocr_det: bool = True,
        ocr_cls: bool = True,
        ocr_backend=Depends(get_ocr_backend)):
    restfulModel: RestfulModel = RestfulModel()
    response = requests.get(imageUrl)
    image_bytes = response.content
    if image_bytes.startswith(b"\xff\xd8\xff") or image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):  # 只处理常见格式图片 (jpg / png)
        restfulModel.resultcode = 200
        img = bytes_to_ndarray(image_bytes)
        result = ocr_backend.predict(img, detect=ocr_det, classify=ocr_cls)
        restfulModel.data = to_legacy_result(result)
        restfulModel.message = "Success"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片"
        )
    return restfulModel
