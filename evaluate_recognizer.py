"""Comprehensive Text Recognizer Evaluation Script.

Evaluates trained checkpoints on synthetic validation (unseen fonts),
real FUNSD validation, and real FUNSD test splits.

Features:
- Guard: test split evaluation requires explicit --final-test flag.
- Separate breakdowns for:
    - Synthetic unseen-fonts val
    - Real FUNSD ground-truth word crops
    - Real FUNSD ground-truth line crops
    - Real FUNSD detector-predicted crops (det_ crops)
- Greedy CTC and Beam Search (with character n-gram LM) comparison
- Top 20 character confusion matrix
- Error breakdown by crop length and source

Usage:
  python evaluate_recognizer.py --checkpoint models/recognizer.pt --split val
  python evaluate_recognizer.py --checkpoint models/recognizer.pt --split test --final-test
  python evaluate_recognizer.py --checkpoint models/recognizer.pt --split val --beam-width 10 --lm-weight 0.5
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common import ROOT, read_json, save_json, distance, image_read
from checkpoints import load
from increment_03_recognition.model import build_recognizer
from increment_03_recognition.data import Vocabulary, line_tensor, minimum_ctc_steps, collate
from increment_03_recognition.decoder import greedy_ctc, beam_search_ctc, CharNgramLM


class EvalDataset(Dataset):
    """Dataset for evaluation with text and metadata preserved."""

    def __init__(self, records: List[dict], vocab: Vocabulary, height: int = 32, max_width: int = 1024):
        self.vocab = vocab
        self.height = height
        self.max_width = max_width
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        label = rec["text"]
        tokens = self.vocab.encode(label)

        # Resolve image path
        img_path = Path(rec["image"])
        if not img_path.is_file():
            img_path = ROOT / "datasets" / rec["image"]
        if not img_path.is_file():
            img_path = ROOT / rec["image"]

        img = image_read(img_path)
        tensor, steps = line_tensor(img, self.height, self.max_width)
        req_steps = minimum_ctc_steps(tokens)
        if steps < req_steps:
            pad_needed = (req_steps - steps + 1) * 4
            padded = torch.ones(1, self.height, tensor.shape[-1] + pad_needed)
            padded[:, :, :tensor.shape[-1]] = tensor
            tensor = padded
            steps = tensor.shape[-1] // 4

        return tensor, torch.tensor(tokens, dtype=torch.long), steps, label


def evaluate_records(
    model: torch.nn.Module,
    records: List[dict],
    vocab: Vocabulary,
    alphabet: str,
    device: str,
    height: int = 32,
    max_width: int = 1024,
    batch_size: int = 32,
    beam_width: int = 0,
    lm_weight: float = 0.0,
    lm: CharNgramLM = None
) -> dict:
    """Run full evaluation on records and compute granular metrics and error analysis."""
    if not records:
        return {"count": 0, "cer": 0.0, "wer": 0.0}

    dataset = EvalDataset(records, vocab, height=height, max_width=max_width)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    model.eval()
    all_preds_greedy = []
    all_preds_beam = []
    all_gts = []
    all_images = []

    confusions = defaultdict(int)

    with torch.no_grad():
        for images, targets, lengths, target_lengths, labels in loader:
            images = images.to(device)
            logits = model(images, lengths)
            probs = logits.softmax(-1).cpu().numpy()

            for i, gt_text in enumerate(labels):
                t_steps = int(lengths[i])
                slice_probs = probs[:t_steps, i]

                # 1. Greedy decoding
                pred_greedy, _ = greedy_ctc(slice_probs, alphabet)
                all_preds_greedy.append(pred_greedy)
                all_gts.append(gt_text)

                # 2. Beam search (if requested)
                if beam_width > 1:
                    pred_beam, _ = beam_search_ctc(slice_probs, alphabet, beam_width=beam_width, lm_weight=lm_weight, lm=lm)
                    all_preds_beam.append(pred_beam)

                # Record character confusions
                # Align gt and greedy pred with simple edit operations
                pred_chars = list(pred_greedy)
                gt_chars = list(gt_text)
                for g_ch in gt_chars:
                    if g_ch not in pred_chars:
                        # Missing/substituted
                        confusions[(g_ch, "<deleted/sub>")] += 1

    # Metrics
    total_chars = sum(len(g) for g in all_gts)
    total_words = sum(len(g.split()) for g in all_gts)

    char_errs_greedy = sum(distance(g, p) for g, p in zip(all_gts, all_preds_greedy))
    word_errs_greedy = sum(distance(g.split(), p.split()) for g, p in zip(all_gts, all_preds_greedy))

    results = {
        "count": len(all_gts),
        "total_characters": total_chars,
        "total_words": total_words,
        "cer_greedy": round(char_errs_greedy / max(1, total_chars), 5),
        "wer_greedy": round(word_errs_greedy / max(1, total_words), 5)
    }

    if beam_width > 1:
        char_errs_beam = sum(distance(g, p) for g, p in zip(all_gts, all_preds_beam))
        word_errs_beam = sum(distance(g.split(), p.split()) for g, p in zip(all_gts, all_preds_beam))
        results["cer_beam"] = round(char_errs_beam / max(1, total_chars), 5)
        results["wer_beam"] = round(word_errs_beam / max(1, total_words), 5)

    # By length buckets
    length_buckets = {"1-5": [], "6-10": [], "11-20": [], "21+": []}
    for gt, pred in zip(all_gts, all_preds_greedy):
        L = len(gt)
        if L <= 5:
            b = "1-5"
        elif L <= 10:
            b = "6-10"
        elif L <= 20:
            b = "11-20"
        else:
            b = "21+"
        length_buckets[b].append((gt, pred))

    results["cer_by_length"] = {}
    for b_name, b_pairs in length_buckets.items():
        if b_pairs:
            b_chars = sum(len(g) for g, _ in b_pairs)
            b_errs = sum(distance(g, p) for g, p in b_pairs)
            results["cer_by_length"][b_name] = {
                "count": len(b_pairs),
                "cer": round(b_errs / max(1, b_chars), 5)
            }

    # Top confusions
    sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)[:20]
    results["top_confusions"] = [{"char": k[0], "error_type": k[1], "count": v} for k, v in sorted_confusions]

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate text recognizer.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to recognizer checkpoint (.pt)")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val", help="Evaluation split (val or test)")
    parser.add_argument("--final-test", action="store_true", help="Explicit guard flag required to run on test split")
    parser.add_argument("--beam-width", type=int, default=0, help="CTC beam search width (0 = greedy only)")
    parser.add_argument("--lm-weight", type=float, default=0.0, help="Weight for character n-gram LM")
    parser.add_argument("--lm-corpus-dir", type=str, default="corpus", help="Directory containing LM training text")
    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples for smoke test")
    parser.add_argument("--output", type=str, default=None, help="Path to save JSON evaluation report")
    args = parser.parse_args()

    # 1. HARD GUARD FOR TEST SPLIT
    if args.split == "test" and not args.final_test:
        print("\n" + "!" * 70)
        print("ERROR: Test set evaluation guard blocked execution.")
        print("Rule 2 / Constraint: The real test split may only be run ONCE at the end of each phase.")
        print("To confirm this is the final evaluation run, re-run with: --final-test")
        print("!" * 70 + "\n")
        sys.exit(1)

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    # 2. Device & Load Checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluation device: {device}")
    ckpt = load(ckpt_path, "recognizer")

    alphabet = ckpt["alphabet"]
    vocab = Vocabulary(alphabet)
    height = ckpt["height"]
    max_width = ckpt.get("max_width", 1024)
    model_version = ckpt.get("model_version", 2)

    model = build_recognizer(len(alphabet) + 1, version=model_version).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3. Optional Language Model
    lm = None
    if args.beam_width > 1 and args.lm_weight > 0.0:
        corpus_path = ROOT / args.lm_corpus_dir
        print(f"Loading character n-gram LM from: {corpus_path}")
        lm = CharNgramLM.from_corpus(corpus_path, alphabet, n=4)

    # 4. Load Manifest
    manifest_file = ROOT / "datasets" / f"recognition_{args.split}.jsonl"
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_file}")

    all_records = [json.loads(l) for l in manifest_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.max_samples:
        all_records = all_records[:args.max_samples]

    print(f"\nLoaded {len(all_records):,} records from {manifest_file.name}")
    print(f"Decoding mode: {'Beam Search (width=' + str(args.beam_width) + ', lm_weight=' + str(args.lm_weight) + ')' if args.beam_width > 1 else 'Greedy CTC'}")

    report = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "model_version": model_version,
        "epochs": ckpt.get("epochs", ckpt.get("epoch", 1)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    if args.split == "val":
        # Group records by category
        synth_unseen = [r for r in all_records if r.get("document_id", "").startswith("synth_")]
        real_all = [r for r in all_records if not r.get("document_id", "").startswith("synth_")]
        real_word = [r for r in real_all if "word_" in r["image"]]
        real_line = [r for r in real_all if "line_" in r["image"]]
        real_det = [r for r in real_all if "det_" in r["image"]]

        print("\n" + "=" * 80)
        print("EVALUATION RESULTS: VALIDATION SPLIT")
        print("=" * 80)

        # (a) Synthetic Val (Unseen Fonts)
        print(f"\n[Subset A] Synthetic Val (Unseen Fonts) - {len(synth_unseen):,} samples:")
        res_synth = evaluate_records(model, synth_unseen, vocab, alphabet, device, height, max_width,
                                     args.batch_size, args.beam_width, args.lm_weight, lm)
        print(f"  Greedy CER: {res_synth.get('cer_greedy', 0.0):.4f} | Greedy WER: {res_synth.get('wer_greedy', 0.0):.4f}")
        if args.beam_width > 1:
            print(f"  Beam   CER: {res_synth.get('cer_beam', 0.0):.4f} | Beam   WER: {res_synth.get('wer_beam', 0.0):.4f}")

        # (b) Real FUNSD Val (All)
        print(f"\n[Subset B] Real FUNSD Val (Combined) - {len(real_all):,} samples:")
        res_real = evaluate_records(model, real_all, vocab, alphabet, device, height, max_width,
                                    args.batch_size, args.beam_width, args.lm_weight, lm)
        print(f"  Greedy CER: {res_real.get('cer_greedy', 0.0):.4f} | Greedy WER: {res_real.get('wer_greedy', 0.0):.4f}")
        if args.beam_width > 1:
            print(f"  Beam   CER: {res_real.get('cer_beam', 0.0):.4f} | Beam   WER: {res_real.get('wer_beam', 0.0):.4f}")

        # (b-det) Real FUNSD Detector Crops
        print(f"\n[Subset B-det] Real-Val Detector Crops (det_*) - {len(real_det):,} samples:")
        res_det = evaluate_records(model, real_det, vocab, alphabet, device, height, max_width,
                                   args.batch_size, args.beam_width, args.lm_weight, lm)
        print(f"  Greedy CER: {res_det.get('cer_greedy', 0.0):.4f} | Greedy WER: {res_det.get('wer_greedy', 0.0):.4f}")

        # (b-word) Real FUNSD Word Crops
        print(f"\n[Subset B-word] Real-Val GT Word Crops - {len(real_word):,} samples:")
        res_word = evaluate_records(model, real_word, vocab, alphabet, device, height, max_width,
                                    args.batch_size, args.beam_width, args.lm_weight, lm)
        print(f"  Greedy CER: {res_word.get('cer_greedy', 0.0):.4f} | Greedy WER: {res_word.get('wer_greedy', 0.0):.4f}")

        # (b-line) Real FUNSD Line Crops
        print(f"\n[Subset B-line] Real-Val GT Line Crops - {len(real_line):,} samples:")
        res_line = evaluate_records(model, real_line, vocab, alphabet, device, height, max_width,
                                    args.batch_size, args.beam_width, args.lm_weight, lm)
        print(f"  Greedy CER: {res_line.get('cer_greedy', 0.0):.4f} | Greedy WER: {res_line.get('wer_greedy', 0.0):.4f}")

        report["synthetic_unseen_val"] = res_synth
        report["real_funsd_val"] = res_real
        report["real_val_det_crops"] = res_det
        report["real_val_word_crops"] = res_word
        report["real_val_line_crops"] = res_line

        print("\n" + "-" * 80)
        print("CER by Crop Length (Real Val):")
        for l_bucket, l_data in res_real.get("cer_by_length", {}).items():
            print(f"  Length {l_bucket:<6}: {l_data['count']:>5} samples | CER = {l_data['cer']:.4f}")

        print("\nTop Character Confusions (Real Val):")
        for conf in res_real.get("top_confusions", [])[:10]:
            print(f"  '{conf['char']}' -> {conf['error_type']}: {conf['count']} times")

    else:
        # Final test split
        print("\n" + "=" * 80)
        print("FINAL EVALUATION: TEST SPLIT (GUARD PASSED)")
        print("=" * 80)
        res_test = evaluate_records(model, all_records, vocab, alphabet, device, height, max_width,
                                    args.batch_size, args.beam_width, args.lm_weight, lm)
        print(f"Test Set Total Samples: {len(all_records):,}")
        print(f"  Greedy CER: {res_test.get('cer_greedy', 0.0):.4f} | Greedy WER: {res_test.get('wer_greedy', 0.0):.4f}")
        if args.beam_width > 1:
            print(f"  Beam   CER: {res_test.get('cer_beam', 0.0):.4f} | Beam   WER: {res_test.get('wer_beam', 0.0):.4f}")
        report["real_funsd_test"] = res_test

    if args.output:
        save_json(args.output, report)
        print(f"\nReport saved to: {args.output}")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
