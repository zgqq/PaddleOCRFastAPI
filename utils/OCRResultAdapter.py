# -*- coding: utf-8 -*-


def _result_data(result):
    if isinstance(result, dict):
        return result.get("res", result)
    return result.json["res"]


def to_legacy_result(raw_results):
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
