import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from common import ROOT, image_write
from increment_01_preparation.orientation import train, Orientation
from increment_02_detection.line_orientation import LineOrientation


class OrientationTrainingTests(unittest.TestCase):
    def test_page_and_crop_classifiers_train_and_load(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            folder = Path(temporary)
            for index, label in enumerate([0, 180]):
                image = np.full((40, 120, 3), 255, np.uint8)
                cv2.putText(image, "AB" if index == 0 else "BA", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 1)
                if index:
                    image = cv2.rotate(image, cv2.ROTATE_180)
                image_write(folder / f"{index}.png", image)
                record = {"image": f"{index}.png", "document_id": str(index), "correction_clockwise": label}
                (folder / ("train.jsonl" if index == 0 else "val.jsonl")).write_text(json.dumps(record)+"\n", encoding="utf-8")
            for kind, loader in [("orientation", Orientation), ("line_orientation", LineOrientation)]:
                train(folder / "train.jsonl", folder / "val.jsonl", folder / f"{kind}.pt", epochs=1, batch_size=1, kind=kind)
                model = loader(folder / f"{kind}.pt")
                angle, confidence = model(image)
                self.assertIn(angle, {0, 180} if kind == "line_orientation" else {0, 90, 180, 270})
                self.assertGreaterEqual(confidence, 0)
                self.assertLessEqual(confidence, 1)
