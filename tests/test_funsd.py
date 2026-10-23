import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from common import ROOT, image_write, read_json, manifest
from geometry import crop_quad
from increment_01_preparation.preparation import prepare
from prepare_funsd import transform_polygon, convert


class FunsdTests(unittest.TestCase):
    def test_rotation_tracks_label_and_ink(self):
        image = np.full((100, 180, 3), 255, np.uint8)
        cv2.rectangle(image, (20, 30), (80, 50), (0, 0, 0), -1)
        settings = read_json(ROOT / 'config.json')['preparation']
        settings.update(deskew=False, perspective=False)
        prepared, details = prepare(image, settings, rotation=90)
        p, visible = transform_polygon([[20,30],[80,30],[80,50],[20,50]],
            details['input_to_prepared_matrix'], prepared.shape[1], prepared.shape[0])
        ys, xs = np.where(prepared[:,:,0] < 30)
        self.assertAlmostEqual(min(x[0] for x in p), float(xs.min()), places=3)
        self.assertAlmostEqual(max(x[1] for x in p), float(ys.max()), places=3)
        self.assertAlmostEqual(visible, 1.)

    def test_perspective_matrix_matches_actual_crop(self):
        image = np.full((150, 220, 3), 255, np.uint8)
        p = np.array([[20,20],[190,10],[200,120],[30,130]], np.float32)
        crop, _, matrix = crop_quad(image, p, rotate_tall=False, return_transform=True)
        expected = np.array([[0,0],[crop.shape[1]-1,0],[crop.shape[1]-1,crop.shape[0]-1],[0,crop.shape[0]-1]])
        actual = cv2.perspectiveTransform(p[None], matrix)[0]
        np.testing.assert_allclose(actual, expected, atol=1e-4)

    def test_prepare_composes_steps_in_order(self):
        image = np.full((180, 260, 3), 255, np.uint8)
        cv2.rectangle(image, (30,20), (230,160), (0,0,0), 3)
        settings = read_json(ROOT / 'config.json')['preparation']
        settings.update(perspective=True, deskew=True, denoise=True, contrast=True)
        prepared, details = prepare(image, settings, rotation=90)
        combined = np.eye(3)
        for step in details['steps']:
            if 'matrix' in step:
                m = np.asarray(step['matrix'])
                combined = (np.vstack([m,[0,0,1]]) if m.shape == (2,3) else m) @ combined
        np.testing.assert_allclose(combined, details['input_to_prepared_matrix'])
        self.assertEqual(details['output_size'], [prepared.shape[1], prepared.shape[0]])
        self.assertTrue(any(s['operation'] == 'flat_page_perspective' and s['corners'] for s in details['steps']))

    def test_clipping_and_invisible_regions(self):
        p, fraction = transform_polygon([[-10,10],[10,10],[10,30],[-10,30]], np.eye(3), 100, 100)
        self.assertAlmostEqual(fraction, .5)
        self.assertTrue(all(x >= 0 for x,y in p))
        self.assertIsNone(transform_polygon([[110,10],[120,10],[120,30],[110,30]], np.eye(3), 100, 100)[0])

    def test_end_to_end_splits_and_labels(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            raw, output = Path(temporary)/'dataset', Path(temporary)/'prepared'
            for split, count in [('training_data',3),('testing_data',1)]:
                for i in range(count):
                    stem = f'{split}_{i}'
                    image = np.full((60+i, 120, 3), 250-i-(10 if split == 'testing_data' else 0), np.uint8)
                    image_write(raw/split/'images'/f'{stem}.png', image)
                    annotation = raw/split/'annotations'/f'{stem}.json'
                    annotation.parent.mkdir(parents=True, exist_ok=True)
                    annotation.write_text(json.dumps({'form':[{'words':[{'text':'hello','box':[10,10,50,30]}]}]}))
            config = read_json(ROOT/'config.json')
            config['preparation'].update(deskew=False, perspective=False)
            result = convert(raw, output, config, rotation=90)
            self.assertEqual(result['splits']['train']['pages'], 2)
            self.assertEqual(result['splits']['val']['pages'], 1)
            self.assertEqual(len(manifest(output/'detection_test.jsonl')), 1)
            self.assertEqual(result['annotation_level'], 'word')
            with self.assertRaises(ValueError):
                convert(raw, output, config)
