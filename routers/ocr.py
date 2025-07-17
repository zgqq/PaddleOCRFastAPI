# -*- coding: utf-8 -*-

from fastapi import APIRouter, HTTPException, UploadFile, status
from models.OCRModel import *
from models.RestfulModel import *
from paddleocr import PaddleOCR
from utils.ImageHelper import base64_to_ndarray, bytes_to_ndarray
import requests
import os
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

OCR_LANGUAGE = os.environ.get("OCR_LANGUAGE", "ch")

router = APIRouter(prefix="/ocr", tags=["OCR"])

#ocr = PaddleOCR(use_gpu=True, use_angle_cls=True, lang=OCR_LANGUAGE)
ocr = PaddleOCR(
     #use_angle_cls=True,
     lang=OCR_LANGUAGE,
     # 指定 Server 版检测与识别模型
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="PP-OCRv5_server_rec",
    # 视场景启用文本行方向分类（类似旧版 cls）
    use_textline_orientation=True
    )


@router.get('/predict-by-path', response_model=RestfulModel, summary="识别本地图片")
def predict_by_path(image_path: str, ocr_det: bool = True, ocr_cls: bool = True):
    result = ocr.ocr(image_path, cls=ocr_cls, det=ocr_det)
    restfulModel = RestfulModel(
        resultcode=200, message="Success", data=result, cls=OCRModel)
    return restfulModel


@router.post('/predict-by-base64', response_model=RestfulModel, summary="识别 Base64 数据")
def predict_by_base64(base64model: Base64PostModel):
    img = base64_to_ndarray(base64model.base64_str)
    #result = ocr.ocr(img=img, det=base64model.ocr_det, cls=base64model.ocr_cls)
    result = ocr.ocr(img=img)

    #restfulModel = RestfulModel(
    #    resultcode=200, message="Success", data=result, cls=OCRModel)
    #return restfulModel

    print(result)
    
    content = jsonable_encoder([r.res for r in result])
    return JSONResponse(content=content)



@router.post('/predict-by-file', response_model=RestfulModel, summary="识别上传文件")
async def predict_by_file(file: UploadFile, ocr_det: bool = True, ocr_cls: bool = True):
    restfulModel: RestfulModel = RestfulModel()
    if file.filename.endswith((".jpg", ".png")):  # 只处理常见格式图片
        restfulModel.resultcode = 200
        restfulModel.message = file.filename
        file_data = file.file
        file_bytes = file_data.read()
        img = bytes_to_ndarray(file_bytes)
        result = ocr.ocr(img=img, cls=ocr_cls, det=ocr_det)
        restfulModel.data = result
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片"
        )
    return restfulModel


@router.get('/predict-by-url', response_model=RestfulModel, summary="识别图片 URL")
async def predict_by_url(imageUrl: str, ocr_det: bool = True, ocr_cls: bool = True):
    restfulModel: RestfulModel = RestfulModel()
    response = requests.get(imageUrl)
    image_bytes = response.content
    if image_bytes.startswith(b"\xff\xd8\xff") or image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):  # 只处理常见格式图片 (jpg / png)
        restfulModel.resultcode = 200
        img = bytes_to_ndarray(image_bytes)
        result = ocr.ocr(img=img, cls=ocr_cls, det=ocr_det)
        restfulModel.data = result
        restfulModel.message = "Success"
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="请上传 .jpg 或 .png 格式图片"
        )
    return restfulModel
