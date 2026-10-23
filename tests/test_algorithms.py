import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
from common import ROOT, image_write, read_json, check_splits
from geometry import crop_quad
from increment_01_preparation.preparation import prepare, pages, deskew_angle
from increment_04_layout.layout import reading_order, blocks
from increment_05_tables.tables import extract, html
from evaluate import polygon_iou, match_polygons, text_score


class AlgorithmTests(unittest.TestCase):
    def test_real_perspective_crop(self):
        image = np.full((100, 100, 3), (10, 20, 30), np.uint8)
        crop, angle = crop_quad(image, [[10, 10], [90, 10], [90, 30], [10, 30]])
        self.assertEqual(crop.shape[:2], (20, 80))
        self.assertEqual(crop[10, 40].tolist(), [10, 20, 30])
        self.assertEqual(angle, 0)

    def test_tall_line_rotation_is_not_page_rotation(self):
        image = np.full((100, 100, 3), 255, np.uint8)
        points = [[10, 10], [30, 10], [30, 90], [10, 90]]
        crop, angle = crop_quad(image, points)
        self.assertEqual(crop.shape[:2], (20, 80))
        self.assertEqual(angle, 90)
        page, angle = crop_quad(image, points, rotate_tall=False)
        self.assertEqual(page.shape[:2], (80, 20))
        self.assertEqual(angle, 0)

    def test_bad_polygon_rejected(self):
        with self.assertRaises(ValueError):
            crop_quad(np.ones((20, 20, 3), np.uint8), [[0, 0], [1, 1], [2, 2], [3, 3]])

    def test_blank_page_deskew_zero(self):
        self.assertEqual(deskew_angle(np.full((100, 100, 3), 255, np.uint8)), 0)

    def test_preparation_no_learned_dependency(self):
        settings = read_json(ROOT / "config.json")["preparation"]
        settings.update(deskew=False, contrast=False, denoise=False, perspective=False)
        result, metadata = prepare(np.full((100, 200, 3), 255, np.uint8), settings, rotation=90)
        self.assertEqual(result.shape[:2], (200, 100))
        self.assertEqual(metadata["coordinate_frame"], "prepared_image_pixels")

    def test_pdf_pages_and_transparency(self):
        from PIL import Image
        Image.init()
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            root = Path(temp)
            image = Image.new("RGB", (100, 100), "white")
            image.save(root / "input.pdf", save_all=True, append_images=[image.copy()])
            selected = list(pages(root / "input.pdf", root / "pages", read_json(ROOT / "config.json"), [2]))
            self.assertEqual(selected[0][0], 2)
            Image.new("RGBA", (20, 30), (0, 0, 0, 0)).save(root / "alpha.png")
            selected = list(pages(root / "alpha.png", root / "alpha", read_json(ROOT / "config.json")))
            self.assertEqual(selected[0][1][0, 0].tolist(), [255, 255, 255])

    def test_column_reading_order(self):
        lines = [{"id": 0, "bbox": [10, 10, 100, 20], "text": "L1"},
                 {"id": 1, "bbox": [200, 10, 300, 20], "text": "R1"},
                 {"id": 2, "bbox": [10, 25, 100, 35], "text": "L2"},
                 {"id": 3, "bbox": [200, 25, 300, 35], "text": "R2"}]
        self.assertEqual([l["text"] for l in reading_order(lines)], ["L1", "L2", "R1", "R2"])
        output = blocks(lines)
        self.assertEqual(sum(len(b["line_ids"]) for b in output), 4)

    def test_bordered_table_and_text_assignment(self):
        image = np.full((200, 300, 3), 255, np.uint8)
        for x in [30, 150, 270]:
            cv2.line(image, (x, 30), (x, 170), (0, 0, 0), 2)
        for y in [30, 100, 170]:
            cv2.line(image, (30, y), (270, y), (0, 0, 0), 2)
        table = extract(image, [{"id": 0, "bbox": [40, 45, 100, 70], "text": "A"}])[0]
        self.assertEqual((table["rows"], table["columns"]), (2, 2))
        self.assertEqual(table["cells"][0]["text"], "A")
        self.assertEqual(table["assigned_line_ids"], [0])

    def test_table_merged_header(self):
        image = np.full((200, 300, 3), 255, np.uint8)
        for x in [30, 270]:
            cv2.line(image, (x, 30), (x, 170), (0, 0, 0), 2)
        cv2.line(image, (150, 100), (150, 170), (0, 0, 0), 2)
        for y in [30, 100, 170]:
            cv2.line(image, (30, y), (270, y), (0, 0, 0), 2)
        table = extract(image, [])[0]
        self.assertEqual(table["cells"][0]["column_span"], 2)
        self.assertEqual(len(table["cells"]), 3)

    def test_spanning_text_is_not_silently_assigned(self):
        image = np.full((150, 300, 3), 255, np.uint8)
        for x in [30, 150, 270]:
            cv2.line(image, (x, 30), (x, 120), (0, 0, 0), 2)
        for y in [30, 120]:
            cv2.line(image, (30, y), (270, y), (0, 0, 0), 2)
        table = extract(image, [{"id": 8, "bbox": [40, 40, 260, 80], "text": "A B"}])[0]
        self.assertEqual(table["assigned_line_ids"], [])
        self.assertEqual(table["unassigned_line_ids"], [8])

    def test_table_html_escapes_text(self):
        output = html({"rows": 1, "columns": 1, "cells": [{"row": 0, "column": 0, "row_span": 1, "column_span": 1, "text": "<script>"}]})
        self.assertIn("&lt;script&gt;", output)

    def test_polygon_metric_and_text_metric(self):
        p = [[0, 0], [10, 0], [10, 10], [0, 10]]
        self.assertAlmostEqual(polygon_iou(p, p), 1)
        self.assertEqual(match_polygons([p, p], [p]), (1, 1, 0))
        report = text_score({"until": 3, "status": "completed", "pages": [{"number": 1, "text": "Hallo"}]}, {"pages": [{"number": 1, "text": "Hello"}]})
        self.assertEqual(report["cer"], .2)
