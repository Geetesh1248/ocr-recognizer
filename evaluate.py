"""Offline automatic scoring against existing labels, separate from runtime OCR."""
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pyclipper
from common import read_json, normal_text, distance, manifest, image_read, ROOT


def text_score(result, labels):
    if result["until"] < 3:
        raise ValueError("Text scoring needs increment 3 or later")
    actual = {p["number"]: p for p in result["pages"]}
    expected = {p["number"]: p for p in labels["pages"]}
    if set(actual) != set(expected) or len(actual) != len(result["pages"]) or len(expected) != len(labels["pages"]):
        raise ValueError("Every result page needs exactly one label")
    edits, chars, word_edits, words, details = 0, 0, 0, 0, []
    for number, page in actual.items():
        reference, predicted = normal_text(expected[number]["text"]), normal_text(page["text"])
        ce, we = distance(reference, predicted), distance(reference.split(), predicted.split())
        edits += ce
        chars += len(reference)
        word_edits += we
        words += len(reference.split())
        details.append({"page": number, "character_edits": ce, "reference_characters": len(reference), "word_edits": we})
    return {"cer": edits/max(1, chars), "wer": word_edits/max(1, words), "pages": details,
            "execution_status": result["status"], "scope": "automatic text; table/order accuracy needs corresponding labels"}


def polygon_iou(a, b):
    paths = [np.round(np.asarray(p)*100).astype(np.int64).tolist() for p in [a, b]]
    clipper = pyclipper.Pyclipper()
    clipper.AddPath(paths[0], pyclipper.PT_SUBJECT, True)
    clipper.AddPath(paths[1], pyclipper.PT_CLIP, True)
    intersection = abs(sum(pyclipper.Area(p) for p in clipper.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)))
    area = sum(abs(pyclipper.Area(p)) for p in paths)
    return intersection/max(1., area-intersection)


def match_polygons(predictions, labels, threshold=.5):
    pairs = sorted([(polygon_iou(pred, truth), i, j) for i, pred in enumerate(predictions) for j, truth in enumerate(labels)], reverse=True)
    used_predictions, used_labels = set(), set()
    for overlap, i, j in pairs:
        if overlap >= threshold and i not in used_predictions and j not in used_labels:
            used_predictions.add(i)
            used_labels.add(j)
    return len(used_labels), len(predictions)-len(used_predictions), len(labels)-len(used_labels)


def detection_score(checkpoint, manifest_path, settings, threshold=.5):
    from increment_02_detection.inference import Detector
    model = Detector(checkpoint, settings)
    true_positive = false_positive = false_negative = 0
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        for index, record in enumerate(manifest(manifest_path)):
            level = record.get('annotation_level', 'region')
            if level != 'region' and model.annotation_level != 'region' and level != model.annotation_level:
                raise ValueError(f'Cannot evaluate {model.annotation_level} detector against {level} labels')
            folder = Path(temporary) / str(index)
            folder.mkdir()
            predicted = model(image_read(record["image"]), folder)
            ignored = record.get("ignore", [False]*len(record["polygons"]))
            if len(ignored) != len(record["polygons"]):
                raise ValueError("Detection label/ignore counts differ")
            truths = [p for p, flag in zip(record["polygons"], ignored) if not flag]
            ignored_regions = [p for p, flag in zip(record["polygons"], ignored) if flag]
            polygons = [p["polygon"] for p in predicted if not any(polygon_iou(p["polygon"], ignored_polygon) >= threshold for ignored_polygon in ignored_regions)]
            tp, fp, fn = match_polygons(polygons, truths, threshold)
            true_positive += tp
            false_positive += fp
            false_negative += fn
    precision = true_positive/max(1, true_positive+false_positive)
    recall = true_positive/max(1, true_positive+false_negative)
    return {"precision": precision, "recall": recall, "f1": 2*precision*recall/max(1e-9, precision+recall),
            "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative,
            "polygon_iou_threshold": threshold, "annotation_level": model.annotation_level,
            "matching": "greedy highest-IoU one-to-one"}
