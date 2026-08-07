# -*- coding: utf-8 -*-


def _result_data(result):
    if isinstance(result, dict):
        return result.get("res", result)
    return result.json["res"]


def _rapid_result_to_legacy(result):
    texts = tuple(result.txts or ())
    scores = tuple(result.scores or ())
    if not texts:
        return None
    if result.boxes is None:
        return [[text, float(score)] for text, score in zip(texts, scores)]

    lines = []
    for polygon, text, score in zip(result.boxes, texts, scores):
        coordinates = [[float(x), float(y)] for x, y in polygon]
        lines.append([coordinates, [text, float(score)]])
    return lines


def to_legacy_result(raw_results):
    if all(hasattr(raw_results, field) for field in ("boxes", "txts", "scores")):
        return [_rapid_result_to_legacy(raw_results)]

    pages = []
    for result in raw_results:
        result = _result_data(result)
        if "rec_text" in result:
            pages.append([[result["rec_text"], float(result["rec_score"])]])
            continue
        if not result["rec_texts"]:
            pages.append(None)
            continue
        lines = []
        for polygon, text, score in zip(
            result["rec_polys"], result["rec_texts"], result["rec_scores"]
        ):
            coordinates = [
                [float(x), float(y)]
                for x, y in polygon
            ]
            lines.append([coordinates, [text, float(score)]])
        pages.append(lines)
    return pages
