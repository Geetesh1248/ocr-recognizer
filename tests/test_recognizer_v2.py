"""Unit tests for upgraded TextRecognizerV2, beam search decoding, and evaluation tools."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

from common import ROOT, read_json
from increment_03_recognition.model import TextRecognizerV1, TextRecognizerV2, build_recognizer
from increment_03_recognition.decoder import greedy_ctc, beam_search_ctc, CharNgramLM


class RecognizerV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = read_json(ROOT / "config.json")
        cls.alphabet = cls.config["recognizer"]["alphabet"]
        cls.num_classes = len(cls.alphabet) + 1

    def test_model_v2_shape(self):
        """Verify TextRecognizerV2 forward pass output shape is (T, B, classes)."""
        model = TextRecognizerV2(self.num_classes)
        images = torch.randn(2, 1, 32, 128)
        lengths = torch.tensor([32, 24])
        logits = model(images, lengths)
        # 4x horizontal downsampling: width 128 -> 32 time steps
        self.assertEqual(logits.shape, (32, 2, self.num_classes))

    def test_model_v2_gradients(self):
        """Verify CTC loss backward pass produces real non-zero gradients in V2."""
        model = TextRecognizerV2(self.num_classes)
        images = torch.randn(2, 1, 32, 64)
        lengths = torch.tensor([16, 12])
        targets = torch.tensor([1, 2, 3, 4, 5])
        target_lengths = torch.tensor([3, 2])

        logits = model(images, lengths)
        criterion = nn.CTCLoss(blank=0, zero_infinity=True)
        loss = criterion(logits.log_softmax(-1), targets, lengths, target_lengths)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        # Linear classifier weights have non-zero gradients
        self.assertGreater(float(model.characters.weight.grad.abs().sum()), 0)
        # Conv stem weights have non-zero gradients
        self.assertGreater(float(model.cnn[0].weight.grad.abs().sum()), 0)

    def test_model_v2_param_count(self):
        """Verify parameter count satisfies constraint: under 10M parameters."""
        model = TextRecognizerV2(self.num_classes)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLess(params, 10_000_000, f"Too many parameters: {params:,} (limit 10M)")
        self.assertGreater(params, 2_000_000, f"Too few parameters: {params:,}")

    def test_build_recognizer_versions(self):
        """Verify build_recognizer instantiates correct model version."""
        v1 = build_recognizer(self.num_classes, version=1)
        self.assertIsInstance(v1, TextRecognizerV1)
        v2 = build_recognizer(self.num_classes, version=2)
        self.assertIsInstance(v2, TextRecognizerV2)
        with self.assertRaises(ValueError):
            build_recognizer(self.num_classes, version=99)

    def test_beam_search_ctc_decoding(self):
        """Verify beam search correctly decodes high-probability CTC sequence."""
        # Target: "cat" -> indices 3, 1, 20
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        target_str = "cat"
        target_tokens = [alphabet.index(c) + 1 for c in target_str]
        classes = len(alphabet) + 1

        # 6 time steps: [c, c, blank, a, blank, t]
        steps = 6
        probs = np.full((steps, classes), 0.01 / (classes - 1), dtype=np.float32)
        seq = [target_tokens[0], target_tokens[0], 0, target_tokens[1], 0, target_tokens[2]]
        for t, tok in enumerate(seq):
            probs[t, :] = 0.01 / (classes - 1)
            probs[t, tok] = 0.99

        # Greedy
        g_text, g_conf = greedy_ctc(probs, alphabet)
        self.assertEqual(g_text, "cat")

        # Beam search
        b_text, b_conf = beam_search_ctc(probs, alphabet, beam_width=5)
        self.assertEqual(b_text, "cat")
        self.assertGreater(b_conf, 0.5)

    def test_char_ngram_lm(self):
        """Verify CharNgramLM trains on text and scores valid contexts higher."""
        lm = CharNgramLM(self.alphabet, n=3)
        lm.train_text("TOTAL AMOUNT DUE: $120.00\nDATE OF INVOICE: 01/15/2024\n")
        
        # 'T' followed by 'O' should have higher probability than 'T' followed by 'Z'
        score_to = lm.score("T", "O")
        score_tz = lm.score("T", "Z")
        self.assertGreater(score_to, score_tz)

    def test_evaluate_script_test_guard(self):
        """Verify evaluate_recognizer.py blocks test split without --final-test flag."""
        # Find any checkpoint or create a dummy one
        dummy_ckpt = ROOT / "models" / "detector_20epochs.pt"
        if dummy_ckpt.is_file():
            cmd = [sys.executable, str(ROOT / "evaluate_recognizer.py"),
                   "--checkpoint", str(dummy_ckpt), "--split", "test"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 1)
            self.assertIn("guard", res.stdout.lower() + res.stderr.lower())

    def test_init_from_vs_resume_and_pruning(self):
        """Verify --init-from vs --resume behavior, mutual exclusivity, and checkpoint pruning."""
        import tempfile
        import shutil
        from checkpoints import save
        from train_recognizer_gpu import prune_epoch_checkpoints

        temp_dir = Path(tempfile.mkdtemp(prefix="test_init_resume_"))
        try:
            model = build_recognizer(self.num_classes, version=2)
            # Create dummy checkpoint with epoch 10 and validation_cer 0.1234
            ckpt_path = temp_dir / "recognizer_stageA_best.pt"
            payload = {
                "format_version": 1,
                "kind": "recognizer",
                "epochs": 10,
                "stage": "A",
                "model_version": 2,
                "alphabet": self.alphabet,
                "height": 32,
                "max_width": 1024,
                "validation_cer": 0.1234,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer_state": {"state": {}, "param_groups": [{"lr": 0.0001}]},
                "seed": 42
            }
            save(ckpt_path, payload)

            # 1. Verify mutual exclusivity in train_recognizer_gpu.py
            cmd_both = [
                sys.executable, str(ROOT / "train_recognizer_gpu.py"),
                "--stage", "B",
                "--resume", str(ckpt_path),
                "--init-from", str(ckpt_path),
                "--output-dir", str(temp_dir)
            ]
            res_both = subprocess.run(cmd_both, capture_output=True, text=True)
            self.assertNotEqual(res_both.returncode, 0)
            self.assertIn("Cannot specify both --resume and --init-from", res_both.stderr + res_both.stdout)

            # 2. Verify --init-from resets epoch to 1, best_cer to inf, and uses requested lr
            cmd_init = [
                sys.executable, str(ROOT / "train_recognizer_gpu.py"),
                "--stage", "B",
                "--init-from", str(ckpt_path),
                "--lr", "0.0005",
                "--epochs", "1",
                "--max-steps", "1",
                "--batch-size", "4",
                "--synth-count", "10",
                "--output-dir", str(temp_dir / "init_test")
            ]
            res_init = subprocess.run(cmd_init, capture_output=True, text=True)
            self.assertEqual(res_init.returncode, 0, f"Init failed: {res_init.stderr}")
            self.assertIn("[INIT-FROM] Initializing model weights ONLY from checkpoint", res_init.stdout)
            self.assertIn("Epoch counter: reset to 1", res_init.stdout)
            self.assertIn("Starting Best CER:", res_init.stdout)
            self.assertIn("Requested LR: 5.000000e-04", res_init.stdout)
            self.assertIn("Actual Optimizer LR: 5.000000e-04", res_init.stdout)

            # 3. Verify --resume restores epoch 11 and previous best CER
            cmd_resume = [
                sys.executable, str(ROOT / "train_recognizer_gpu.py"),
                "--stage", "A",
                "--resume", str(ckpt_path),
                "--epochs", "1",
                "--max-steps", "1",
                "--batch-size", "4",
                "--synth-count", "10",
                "--output-dir", str(temp_dir / "resume_test")
            ]
            res_resume = subprocess.run(cmd_resume, capture_output=True, text=True)
            self.assertEqual(res_resume.returncode, 0, f"Resume failed: {res_resume.stderr}")
            self.assertIn("Resuming at epoch 11, previous best CER: 0.1234", res_resume.stdout)

            # 4. Verify checkpoint pruning: create epochs 1..4 + best
            for ep in range(1, 5):
                p = temp_dir / f"recognizer_stageB_epoch{ep:03d}.pt"
                save(p, payload)
            best_p = temp_dir / "recognizer_stageB_best.pt"
            save(best_p, payload)

            prune_epoch_checkpoints(temp_dir, "B", keep_last=2)
            remaining_epochs = sorted([p.name for p in temp_dir.glob("recognizer_stageB_epoch*.pt")])
            self.assertEqual(remaining_epochs, ["recognizer_stageB_epoch003.pt", "recognizer_stageB_epoch004.pt"])
            self.assertTrue(best_p.is_file(), "Best checkpoint must never be pruned")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_resume_lr_and_patience_exactness(self):
        """Verify that resuming restores exact LR (diff < 1e-9), scheduler state, best CER, and patience counter."""
        import tempfile
        import shutil
        import re

        temp_dir = Path(tempfile.mkdtemp(prefix="test_lr_exactness_"))
        try:
            dir_uninterrupted = temp_dir / "uninterrupted"
            dir_resumed = temp_dir / "resumed"

            # 1. Uninterrupted 3-epoch run
            cmd1 = [
                sys.executable, str(ROOT / "train_recognizer_gpu.py"),
                "--stage", "A",
                "--epochs", "3",
                "--lr", "0.001",
                "--batch-size", "5",
                "--synth-count", "10",
                "--max-steps", "1",
                "--output-dir", str(dir_uninterrupted)
            ]
            res1 = subprocess.run(cmd1, capture_output=True, text=True)
            self.assertEqual(res1.returncode, 0, f"Uninterrupted run failed: {res1.stderr}")

            # Extract uninterrupted LR at start of epoch 3
            m1 = re.search(r"Epoch:\s+3\s+\|.*?Actual Optimizer LR:\s+([\d\.e\-\+]+)", res1.stdout)
            self.assertIsNotNone(m1, f"Epoch 3 LR not found in uninterrupted output:\n{res1.stdout}")
            lr_uninterrupted = float(m1.group(1))

            # Inspect checkpoint saved after epoch 2
            ckpt_ep2_path = dir_uninterrupted / "recognizer_stageA_epoch002.pt"
            self.assertTrue(ckpt_ep2_path.is_file(), "Missing epoch 2 checkpoint")
            ckpt_ep2 = torch.load(ckpt_ep2_path, map_location="cpu")
            expected_best_cer = ckpt_ep2["best_cer"]
            expected_patience = ckpt_ep2["epochs_without_improvement"]

            # 2. Resumed run starting from epoch 2 checkpoint for 1 epoch (epoch 3)
            cmd2 = [
                sys.executable, str(ROOT / "train_recognizer_gpu.py"),
                "--stage", "A",
                "--epochs", "1",
                "--resume", str(ckpt_ep2_path),
                "--batch-size", "5",
                "--synth-count", "10",
                "--max-steps", "1",
                "--output-dir", str(dir_resumed)
            ]
            res2 = subprocess.run(cmd2, capture_output=True, text=True)
            self.assertEqual(res2.returncode, 0, f"Resumed run failed: {res2.stderr}")

            m2 = re.search(r"Epoch:\s+3\s+\|.*?Actual Optimizer LR:\s+([\d\.e\-\+]+)", res2.stdout)
            self.assertIsNotNone(m2, f"Epoch 3 LR not found in resumed output:\n{res2.stdout}")
            lr_resumed = float(m2.group(1))

            diff = abs(lr_uninterrupted - lr_resumed)
            self.assertLess(diff, 1e-9, f"LR mismatch: {lr_uninterrupted} vs {lr_resumed} (diff {diff})")
            self.assertIn(f"previous best CER: {expected_best_cer:.4f}", res2.stdout)
            self.assertIn(f"patience counter: {expected_patience}", res2.stdout)

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
