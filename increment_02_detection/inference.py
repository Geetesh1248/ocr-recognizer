from pathlib import Path
import cv2
import numpy as np
import torch
from checkpoints import load
from common import ROOT, image_write
from geometry import quad, rectangle, crop_quad, box_iou
from .model import TextDetector
from .data import letterbox, offset_polygon


def polygons_from_map(probability, scales, content_size, settings):
    probability = probability[:content_size[0], :content_size[1]]
    binary = (probability >= settings["pixel_threshold"]).astype(np.uint8)*255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < settings["min_area"]:
            continue
        mask = np.zeros(probability.shape, np.uint8)
        cv2.drawContours(mask, [contour], -1, 1, thickness=-1)
        score = float(probability[mask.astype(bool)].mean())
        if score < settings["region_threshold"]:
            continue
        perimeter = cv2.arcLength(contour, True)
        expanded = offset_polygon(contour.reshape(-1, 2), area*settings["expand_ratio"]/max(perimeter, 1))
        if not expanded:
            continue
        p = cv2.boxPoints(cv2.minAreaRect(max(expanded, key=cv2.contourArea)))
        p[:, 0] /= scales[0]
        p[:, 1] /= scales[1]
        try:
            p = quad(p)
        except ValueError:
            continue
        candidates.append({"polygon": p.tolist(), "bbox": rectangle(p), "detection_score": score})
    kept = []
    for candidate in sorted(candidates, key=lambda c: -c["detection_score"]):
        if not any(box_iou(candidate["bbox"], k["bbox"]) > settings["duplicate_iou"] for k in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda c: (c["bbox"][1], c["bbox"][0]))


class Detector:
    def __init__(self, checkpoint, settings):
        payload = load(checkpoint, "detector")
        self.annotation_level = payload.get('annotation_level', 'region')
        expected = settings.get('annotation_level')
        if expected and expected != self.annotation_level:
            raise ValueError(f'Expected {expected} detector, got {self.annotation_level} checkpoint')
        self.model = TextDetector()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        self.size, self.settings = int(payload["image_size"]), settings
        self.line_orientation = None
        if settings.get("line_orientation_checkpoint"):
            from .line_orientation import LineOrientation
            self.line_orientation = LineOrientation(ROOT / settings["line_orientation_checkpoint"])

    @torch.inference_mode()
    def __call__(self, image, folder):
        tensor, scales, shape = letterbox(image, self.size)
        probability = self.model(tensor[None]).sigmoid()[0, 0].numpy()
        image_write(folder / "probability_map.png", (probability*255).astype(np.uint8))
        entries = polygons_from_map(probability, scales, shape, self.settings)
        if len(entries) > self.settings["max_regions"]:
            raise ValueError("Too many detected regions; refuse silent truncation")
        height, width = image.shape[:2]
        total_crop_pixels = 0
        for index, entry in enumerate(entries):
            p = np.asarray(entry["polygon"], np.float32)
            p[:, 0] = p[:, 0].clip(0, width-1)
            p[:, 1] = p[:, 1].clip(0, height-1)
            crop, rotation = crop_quad(image, p)
            orientation_details = {"line_rotation": 0, "line_orientation_score": None}
            if self.line_orientation:
                angle, score = self.line_orientation(crop)
                orientation_details["line_orientation_score"] = score
                if angle == 180 and score >= self.settings["line_orientation_min_score"]:
                    crop = cv2.rotate(crop, cv2.ROTATE_180)
                    orientation_details["line_rotation"] = 180
            total_crop_pixels += crop.shape[0]*crop.shape[1]
            if total_crop_pixels > self.settings["max_crop_pixels"]:
                raise ValueError("Total crop pixel budget exceeded")
            path = folder / "crops" / f"line_{index:04d}.png"
            image_write(path, crop)
            entry.update(id=index, polygon=p.tolist(), bbox=rectangle(p), crop=str(path.resolve()), annotation_level=self.annotation_level, tall_rotation_ccw=rotation, **orientation_details)
        return entries
