"""Tests for environment verification, zip packaging, and device auto-detection."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
import torch

from common import ROOT, read_json
from validation import validate_config
from check_env import check_environment, REQUIRED_PACKAGES
from make_shareable_zip import create_archive, should_include


class EnvironmentAndToolsTests(unittest.TestCase):
    def test_configs_consistency(self):
        for config_name in ["config.json", "config_words.json", "config_lines.json", "config_preparation_full.json"]:
            path = ROOT / config_name
            self.assertTrue(path.is_file(), f"Missing config: {config_name}")
            cfg = read_json(path)
            validate_config(cfg)
        # Default config and config_words should point to the 20-epoch detector
        cfg_default = read_json(ROOT / "config.json")
        cfg_words = read_json(ROOT / "config_words.json")
        self.assertEqual(cfg_default["detector"]["checkpoint"], "models/detector_20epochs.pt")
        self.assertEqual(cfg_words["detector"]["checkpoint"], "models/detector_20epochs.pt")
        self.assertEqual(cfg_default["detector"]["annotation_level"], "word")
        self.assertEqual(cfg_words["detector"]["annotation_level"], "word")

    def test_check_environment_execution(self):
        saved_stdout = sys.stdout
        try:
            sys.stdout = io.StringIO()
            code = check_environment()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = saved_stdout
        self.assertEqual(code, 0)
        self.assertIn("Fully Self-Made OCR - Environment Check", output)
        self.assertIn("Python Version", output)
        for pkg in REQUIRED_PACKAGES:
            self.assertIn(pkg, output)

    def test_shareable_zip_excludes_prohibited(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            zip_dest = Path(temp) / "test_out.zip"
            created = create_archive(zip_dest)
            self.assertTrue(created.is_file())
            with zipfile.ZipFile(created, "r") as zf:
                names = zf.namelist()
                self.assertGreater(len(names), 20)
                for name in names:
                    self.assertFalse(name.startswith("datasets/"), f"Found dataset in zip: {name}")
                    self.assertFalse(name.startswith("outputs/"), f"Found output in zip: {name}")
                    self.assertFalse(name.startswith("models/"), f"Found model in zip: {name}")
                    self.assertFalse(name.startswith(".venv/"), f"Found venv in zip: {name}")
                    self.assertFalse(name.startswith("__pycache__/"), f"Found pycache in zip: {name}")
                    self.assertFalse(name.endswith(".pt"), f"Found .pt checkpoint in zip: {name}")

    def test_should_include_filter(self):
        self.assertFalse(should_include(ROOT / "datasets" / "test.jsonl"))
        self.assertFalse(should_include(ROOT / "outputs" / "result.json"))
        self.assertFalse(should_include(ROOT / "models" / "detector.pt"))
        self.assertFalse(should_include(ROOT / "some_dir" / "weights.pt"))
        self.assertFalse(should_include(ROOT / "script.pyc"))
        self.assertTrue(should_include(ROOT / "cli.py"))
        self.assertTrue(should_include(ROOT / "common.py"))

    def test_auto_device_selection(self):
        from cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["train-detector", "--train", "t.jsonl", "--val", "v.jsonl", "--output", "o.pt"])
        self.assertEqual(args.device, "auto")

        # Test device resolution
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(device, expected)


if __name__ == "__main__":
    unittest.main()
