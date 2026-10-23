import tempfile
import unittest
from pathlib import Path
import json
import cv2
import numpy as np
import torch
from common import ROOT, image_write, read_json, check_splits
from checkpoints import load
from increment_02_detection.model import TextDetector
from increment_02_detection.data import masks, letterbox
from increment_02_detection.losses import detection_loss
from increment_02_detection.inference import polygons_from_map
from increment_03_recognition.model import TextRecognizer
from increment_03_recognition.data import Vocabulary, minimum_ctc_steps
from increment_03_recognition.decoder import greedy_ctc

torch.set_num_threads(2)


class ModelTests(unittest.TestCase):
    def test_detector_has_real_trainable_gradients(self):
        model = TextDetector()
        images = torch.randn(1, 3, 64, 64)
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, 10:20, 10:50] = 1
        logits = model(images)
        self.assertEqual(logits.shape, target.shape)
        loss = detection_loss(logits, target, torch.ones_like(target))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0)

    def test_shrunk_mask_and_padding_are_correct(self):
        image = np.full((32, 128, 3), 255, np.uint8)
        _, scales, shape = letterbox(image, 64)
        target, valid = masks([[[10, 5], [110, 5], [110, 25], [10, 25]]], [False], 64, scales, shape)
        self.assertGreater(float(target.sum()), 0)
        self.assertLess(float(target.sum()), 50*10)
        self.assertEqual(float(valid[:, 16:].sum()), 0)

    def test_heatmap_postprocessing_generates_a_polygon(self):
        probability = np.zeros((64, 64), np.float32)
        probability[10:20, 5:50] = .9
        settings = read_json(ROOT / "config.json")["detector"]
        output = polygons_from_map(probability, (1, 1), (64, 64), settings)
        self.assertEqual(len(output), 1)
        self.assertAlmostEqual(output[0]["detection_score"], .9, places=5)

    def test_ctc_repeated_letters_and_blanks(self):
        ids = [1, 1, 0, 2, 3, 0, 3, 4]
        probabilities = np.zeros((len(ids), 5))
        for index, token in enumerate(ids):
            probabilities[index, token] = 1
        text, confidence = greedy_ctc(probabilities, "helo")
        self.assertEqual(text, "hello")
        self.assertEqual(confidence, 1)
        self.assertEqual(minimum_ctc_steps([1, 1, 2]), 4)

    def test_recognizer_has_real_ctc_gradients(self):
        model = TextRecognizer(3)
        output = model(torch.randn(2, 1, 32, 64), torch.tensor([16, 12]))
        self.assertEqual(output.shape, (16, 2, 3))
        loss = torch.nn.CTCLoss(blank=0)(output.log_softmax(-1), torch.tensor([1, 2, 2]), torch.tensor([16, 12]), torch.tensor([2, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.characters.weight.grad.abs().sum()), 0)

    def test_vocabulary_rejects_unlabelled_characters(self):
        with self.assertRaises(ValueError):
            Vocabulary("AB").encode("AC")

    def test_unicode_labels_and_validation_text_use_same_normalization(self):
        from increment_03_recognition.data import RecognitionDataset
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "crop.png"
            image_write(path, np.full((32, 80, 3), 255, np.uint8))
            sample = RecognitionDataset([{"image": str(path), "text": "e\u0301"}], Vocabulary("é"))[0]
            self.assertEqual(sample[1].tolist(), [1])
            self.assertEqual(sample[3], "é")

    def test_missing_checkpoint_is_explicit(self):
        with self.assertRaises(FileNotFoundError):
            load(ROOT / "nonexistent_model.pt", "detector")

    def test_document_split_leakage_rejected(self):
        with self.assertRaises(ValueError):
            check_splits([{"document_id": "same"}], [{"document_id": "same"}])

    def test_tiny_training_saves_our_loadable_checkpoints(self):
        from increment_02_detection.train import train as train_detector
        from increment_03_recognition.train import train as train_recognizer
        from increment_02_detection.inference import Detector
        from increment_03_recognition.inference import Recognizer
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp)
            for index, text in enumerate(["AB", "BA"]):
                image = np.full((32, 128, 3), 255-index, np.uint8)
                cv2.putText(image, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 0), 1)
                image_write(folder / f"{index}.png", image)
                record = {"document_id": str(index), "image": f"{index}.png", "text": text,
                          "polygons": [[[5, 3], [100, 3], [100, 28], [5, 28]]], "annotation_level": "word"}
                (folder / ("train.jsonl" if index == 0 else "val.jsonl")).write_text(json.dumps(record)+"\n", encoding="utf-8")
            train_detector(folder / "train.jsonl", folder / "val.jsonl", folder / "detector.pt", epochs=1, size=64, batch_size=1)
            train_recognizer(folder / "train.jsonl", folder / "val.jsonl", folder / "recognizer.pt", "AB", epochs=1, batch_size=1, max_width=128)
            detector = Detector(folder / "detector.pt", read_json(ROOT / "config.json")["detector"])
            recognizer = Recognizer(folder / "recognizer.pt")
            self.assertEqual(detector.size, 64)
            self.assertEqual(recognizer.alphabet, "AB")
            # Execute the actual trained-network path through all five increments.
            from pipeline import run
            config = read_json(ROOT / "config.json")
            config["detector"].update(checkpoint=str(folder / "detector.pt"), pixel_threshold=.1, region_threshold=.0)
            config["recognizer"]["checkpoint"] = str(folder / "recognizer.pt")
            config["preparation"]["deskew"] = False
            result = run(folder / "0.png", folder / "pipeline", 5, config)
            self.assertEqual(result["status"], "completed")
            self.assertTrue((folder / "pipeline/page_0001/increment_05.json").exists())
            self.assertEqual(set(result["model_sha256"]), {"detector", "recognizer"})

    def test_preparation_pipeline_runs_without_learned_weights(self):
        from pipeline import run
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder = Path(temp)
            image_write(folder / "input.png", np.full((100, 100, 3), 255, np.uint8))
            result = run(folder / "input.png", folder / "out", 1, read_json(ROOT / "config.json"))
            self.assertEqual(result["status"], "completed")
            self.assertTrue((folder / "out/page_0001/prepared.png").exists())
            snapshot = read_json(folder / "out/page_0001/increment_01.json")
            self.assertEqual(snapshot["status"], "completed")
            self.assertEqual(snapshot["completed_increment"], 1)

    def test_orientation_network_has_classification_gradients(self):
        from increment_01_preparation.orientation import OrientationCNN
        model = OrientationCNN()
        output = model(torch.randn(2, 1, 128, 128))
        self.assertEqual(output.shape, (2, 4))
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 1]))
        loss.backward()
        self.assertGreater(float(model.classes.weight.grad.abs().sum()), 0)

    def test_impossible_ctc_label_is_rejected(self):
        from increment_03_recognition.data import RecognitionDataset
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            path = Path(temp) / "crop.png"
            image_write(path, np.full((32, 8, 3), 255, np.uint8))
            dataset = RecognitionDataset([{"image": str(path), "text": "AAAA"}], Vocabulary("A"))
            with self.assertRaises(ValueError):
                dataset[0]
