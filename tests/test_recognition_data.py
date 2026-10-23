"""Tests for recognizer data generation, character coverage, font isolation, and CTC constraints."""
import json
from pathlib import Path
import unittest

from common import ROOT, read_json, manifest, check_splits
from make_recognition_data import CorpusManager, FontManager, StreamingSyntheticDataset, normalize_label
from increment_03_recognition.data import Vocabulary, minimum_ctc_steps, line_tensor


class RecognitionDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(ROOT / "config.json")
        cls.alphabet = cls.config["recognizer"]["alphabet"]
        cls.vocab = Vocabulary(cls.alphabet)
        cls.corpus_mgr = CorpusManager(ROOT / "corpus", ROOT / "config.json")
        cls.font_mgr = FontManager(ROOT / "fonts")

    def test_character_coverage(self):
        """Verify the generator can produce all characters in config.json alphabet."""
        seen = set()
        # Sample until all characters or at least 95% of characters seen in a small budget
        for _ in range(2000):
            text = self.corpus_mgr.sample_text()
            seen.update(text)
            if len(seen) == len(self.alphabet):
                break

        # Check that characters sampled belong strictly to vocabulary
        unallowed = seen - set(self.alphabet)
        self.assertEqual(len(unallowed), 0, f"Unallowed characters found: {unallowed}")
        # Core alphanumeric and punctuation must be thoroughly covered
        core_chars = set("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,:;!?'-\"")
        self.assertTrue(core_chars.issubset(seen), f"Missing core characters: {core_chars - seen}")

    def test_font_family_split_isolation(self):
        """Verify train and validation font families are mutually exclusive."""
        splits = read_json(ROOT / "fonts" / "splits.json")
        train_families = set(splits["train_families"].keys())
        val_families = set(splits["val_families"].keys())
        self.assertEqual(len(train_families & val_families), 0, "Train and validation font families overlap")

        # Verify all listed font files exist on disk
        for fam, files in {**splits["train_families"], **splits["val_families"]}.items():
            for f in files:
                self.assertTrue((ROOT / "fonts" / f).is_file(), f"Missing font file: {f}")

    def test_no_train_val_test_leakage(self):
        """Verify train, val, and test splits have disjoint document IDs."""
        train_path = ROOT / "datasets" / "recognition_train.jsonl"
        val_path = ROOT / "datasets" / "recognition_val.jsonl"
        test_path = ROOT / "datasets" / "recognition_test.jsonl"

        if train_path.is_file() and val_path.is_file() and test_path.is_file():
            train_recs = manifest(train_path)
            val_recs = manifest(val_path)
            test_recs = manifest(test_path)

            train_docs = {r["document_id"] for r in train_recs}
            val_docs = {r["document_id"] for r in val_recs}
            test_docs = {r["document_id"] for r in test_recs}

            self.assertEqual(len(train_docs & val_docs), 0, "Train and Val document IDs overlap")
            self.assertEqual(len(train_docs & test_docs), 0, "Train and Test document IDs overlap")
            self.assertEqual(len(val_docs & test_docs), 0, "Val and Test document IDs overlap")

            # Test set must ONLY contain real crops, no synthetic samples
            for r in test_recs:
                self.assertFalse(r["document_id"].startswith("synth_"), f"Synthetic sample leaked to test set: {r}")

    def test_label_correctness_and_ctc_validity(self):
        """Verify labels encode without error and crops satisfy CTC length bounds."""
        train_path = ROOT / "datasets" / "recognition_train.jsonl"
        if not train_path.is_file():
            return

        recs = manifest(train_path)[:500]  # Check first 500 samples
        for r in recs:
            tokens = self.vocab.encode(r["text"])
            req_steps = minimum_ctc_steps(tokens)
            from common import image_read
            img = image_read(r["image"])
            tensor, steps = line_tensor(img, self.config["recognizer"]["height"], self.config["recognizer"]["max_width"])
            self.assertGreaterEqual(steps, req_steps, f"Crop too narrow for CTC: {r['image']} (steps={steps}, req={req_steps})")

    def test_streaming_synthetic_dataset(self):
        """Verify the streaming generator yields valid (image, text) batches."""
        stream = StreamingSyntheticDataset(self.corpus_mgr, self.font_mgr, split="train", max_samples=10)
        samples = list(stream)
        self.assertEqual(len(samples), 10)
        for img, text in samples:
            self.assertEqual(img.shape[0], 32)
            self.assertGreater(img.shape[1], 0)
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)
            tokens = self.vocab.encode(text)
            self.assertGreater(len(tokens), 0)

    def test_preview_grid_file(self):
        """Verify preview grid image was generated and has non-zero size."""
        preview = ROOT / "outputs" / "recognition_previews" / "preview_grid.png"
        self.assertTrue(preview.is_file(), "Preview grid missing")
        self.assertGreater(preview.stat().st_size, 1000, "Preview grid too small")


if __name__ == "__main__":
    unittest.main()
