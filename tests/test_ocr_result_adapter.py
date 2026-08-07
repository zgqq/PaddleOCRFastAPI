from utils.OCRResultAdapter import to_legacy_result


class PaddleResult:
    def __init__(self, result):
        self.json = {"res": result}


def test_ppocr_v6_result_converts_to_legacy_page_shape():
    raw_results = [
        {
            "rec_polys": [
                [[33, 67], [240, 68], [239, 91], [32, 90]],
                [[36, 106], [298, 106], [298, 127], [36, 127]],
            ],
            "rec_texts": [
                "http://localhost:8000/openapijson",
                "基于 Paddle OCR 和 FastAPI 的自用接口",
            ],
            "rec_scores": [0.9633703231811523, 0.972227931022644],
        }
    ]

    assert to_legacy_result(raw_results) == [
        [
            [
                [[33.0, 67.0], [240.0, 68.0], [239.0, 91.0], [32.0, 90.0]],
                ["http://localhost:8000/openapijson", 0.9633703231811523],
            ],
            [
                [[36.0, 106.0], [298.0, 106.0], [298.0, 127.0], [36.0, 127.0]],
                ["基于 Paddle OCR 和 FastAPI 的自用接口", 0.972227931022644],
            ],
        ]
    ]


def test_ppocr_result_object_is_unwrapped_before_conversion():
    raw_results = [
        PaddleResult(
            {
                "rec_polys": [[[1, 2], [3, 2], [3, 4], [1, 4]]],
                "rec_texts": ["兼容"],
                "rec_scores": [0.9],
            }
        )
    ]

    assert to_legacy_result(raw_results) == [
        [[[[1.0, 2.0], [3.0, 2.0], [3.0, 4.0], [1.0, 4.0]], ["兼容", 0.9]]]
    ]


def test_text_recognition_result_preserves_legacy_det_false_shape():
    raw_results = [PaddleResult({"rec_text": "兼容", "rec_score": 0.9})]

    assert to_legacy_result(raw_results) == [[["兼容", 0.9]]]


def test_empty_detection_preserves_legacy_null_page():
    raw_results = [
        PaddleResult({"rec_polys": [], "rec_texts": [], "rec_scores": []})
    ]

    assert to_legacy_result(raw_results) == [None]
