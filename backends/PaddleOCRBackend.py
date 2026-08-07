# -*- coding: utf-8 -*-


class PaddleOCRBackend:
    def __init__(self):
        self._ocr = None
        self._recognizer = None

    def _get_ocr(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv6_small_det",
                text_recognition_model_name="PP-OCRv6_small_rec",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=True,
                text_rec_score_thresh=0.5,
                device="cpu",
            )
        return self._ocr

    def _get_recognizer(self):
        if self._recognizer is None:
            from paddleocr import TextRecognition

            self._recognizer = TextRecognition(
                model_name="PP-OCRv6_small_rec",
                device="cpu",
            )
        return self._recognizer

    def predict(self, image, *, detect=True, classify=True):
        if not detect:
            return self._get_recognizer().predict(image)
        return self._get_ocr().predict(
            image,
            use_textline_orientation=classify,
        )
