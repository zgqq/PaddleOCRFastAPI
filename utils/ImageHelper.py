# -*- coding: utf-8 -*-

import base64

import cv2
import numpy as np


def base64_to_ndarray(b64_data: str):
    """base64转numpy数组，解码失败返回 None（由调用方映射为 400）。"""
    try:
        image_bytes = base64.b64decode(b64_data)
    except Exception:
        return None
    if not image_bytes:
        return None
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    try:
        image_np2 = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    except Exception:
        return None
    return image_np2


def bytes_to_ndarray(img_bytes: bytes):
    """字节转numpy数组，解码失败返回 None（由调用方映射为 400）。"""
    if not img_bytes:
        return None
    image_array = np.frombuffer(img_bytes, dtype=np.uint8)
    try:
        image_np2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    except Exception:
        return None
    return image_np2
