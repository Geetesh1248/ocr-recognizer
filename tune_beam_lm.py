"""Hyperparameter tuning grid for CTC Beam Search + Character n-gram Language Model.

Evaluates combinations of beam_width in [5, 10] and lm_weight in [0.1, 0.2, 0.4]
by caching forward-pass probabilities once to avoid redundant neural network passes.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import ROOT, distance, save_json
from checkpoints import load
from increment_03_recognition.model import build_recognizer
from increment_03_recognition.data import Vocabulary, collate
from increment_03_recognition.decoder import greedy_ctc, beam_search_ctc, CharNgramLM
from evaluate_recognizer import EvalDataset


def main():
    parser = argparse.ArgumentParser(description="Tune Beam Search and LM hyperparameters.")
    parser.add_argument("--checkpoint", type=str, default="models/recognizer_stageC_best.pt")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of real val samples for grid search")
    parser.add_argument("--output", type=str, default="outputs/beam_lm_tuning_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint {args.checkpoint} on {device}...")
    ckpt = load(args.checkpoint, "recognizer")
    alphabet = ckpt["alphabet"]
    vocab = Vocabulary(alphabet)
    height = ckpt["height"]
    max_width = ckpt.get("max_width", 1024)
    model_version = ckpt.get("model_version", 2)

    model = build_recognizer(len(alphabet) + 1, version=model_version).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    corpus_path = ROOT / "corpus"
    print(f"Building Character 4-gram LM from {corpus_path}...")
    t_lm0 = time.time()
    lm = CharNgramLM.from_corpus(corpus_path, alphabet, n=4)
    print(f"LM built in {time.time() - t_lm0:.2f}s ({len(lm.ngram_counts)} n-grams)")

    manifest_path = ROOT / "datasets" / f"recognition_{args.split}.jsonl"
    all_recs = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    real_recs = [r for r in all_recs if not r.get("document_id", "").startswith("synth_")]

    if args.sample_size and len(real_recs) > args.sample_size:
        # Uniform stratified sampling across records
        step = len(real_recs) / args.sample_size
        eval_recs = [real_recs[int(i * step)] for i in range(args.sample_size)]
    else:
        eval_recs = real_recs

    print(f"\nEvaluating on {len(eval_recs):,} real validation crops...")
    dataset = EvalDataset(eval_recs, vocab, height=height, max_width=max_width)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    print("Computing neural network forward-pass probabilities once...")
    t_fwd0 = time.time()
    all_probs = []
    all_gts = []

    with torch.no_grad():
        for images, targets, lengths, target_lengths, labels in loader:
            images = images.to(device)
            logits = model(images, lengths)
            probs = logits.softmax(-1).cpu().numpy()
            for i, gt in enumerate(labels):
                t_steps = int(lengths[i])
                all_probs.append(probs[:t_steps, i])
                all_gts.append(gt)

    fwd_time = time.time() - t_fwd0
    print(f"Forward pass completed in {fwd_time:.2f}s ({len(all_probs)/fwd_time:.1f} samples/sec).")

    total_chars = sum(len(g) for g in all_gts)
    total_words = sum(len(g.split()) for g in all_gts)

    # 1. Greedy baseline
    greedy_preds = [greedy_ctc(p, alphabet)[0] for p in all_probs]
    greedy_cer = sum(distance(g, p) for g, p in zip(all_gts, greedy_preds)) / max(1, total_chars)
    greedy_wer = sum(distance(g.split(), p.split()) for g, p in zip(all_gts, greedy_preds)) / max(1, total_words)

    print("\n" + "=" * 80)
    print("BEAM SEARCH + CHAR-LM HYPERPARAMETER TUNING GRID")
    print("=" * 80)
    print(f"Greedy Baseline: CER = {greedy_cer:.4f} ({greedy_cer*100:.2f}%) | WER = {greedy_wer:.4f} ({greedy_wer*100:.2f}%)")
    print("-" * 80)
    print(f"{'Beam Width':>10} | {'LM Weight':>10} | {'CER':>10} | {'WER':>10} | {'CER Delta':>12} | {'Time':>8}")
    print("-" * 80)

    grid_results = []
    widths = [5, 10]
    weights = [0.1, 0.2, 0.4]

    best_config = None
    best_score = greedy_cer

    for w in widths:
        for lm_w in weights:
            t0 = time.time()
            beam_preds = [beam_search_ctc(p, alphabet, beam_width=w, lm_weight=lm_w, lm=lm)[0] for p in all_probs]
            elapsed = time.time() - t0

            b_cer = sum(distance(g, p) for g, p in zip(all_gts, beam_preds)) / max(1, total_chars)
            b_wer = sum(distance(g.split(), p.split()) for g, p in zip(all_gts, beam_preds)) / max(1, total_words)
            delta = b_cer - greedy_cer

            grid_results.append({
                "beam_width": w,
                "lm_weight": lm_w,
                "cer": round(b_cer, 5),
                "wer": round(b_wer, 5),
                "cer_delta": round(delta, 5),
                "time_seconds": round(elapsed, 2)
            })

            delta_str = f"{delta:+.4f}"
            print(f"{w:>10d} | {lm_w:>10.2f} | {b_cer:>10.4f} | {b_wer:>10.4f} | {delta_str:>12} | {elapsed:>7.1f}s")

            if b_cer < best_score:
                best_score = b_cer
                best_config = (w, lm_w)

    print("=" * 80)
    if best_config:
        print(f"Optimal Beam Search Configuration: width={best_config[0]}, lm_weight={best_config[1]} (CER: {best_score:.4f})")
    else:
        print(f"Greedy CTC remains optimal or tied (CER: {greedy_cer:.4f}). Choosing default beam=5, lm_weight=0.2.")
        best_config = (5, 0.2)

    output_data = {
        "checkpoint": args.checkpoint,
        "sample_size": len(eval_recs),
        "greedy_cer": round(greedy_cer, 5),
        "greedy_wer": round(greedy_wer, 5),
        "best_beam_width": best_config[0],
        "best_lm_weight": best_config[1],
        "grid": grid_results
    }
    save_json(args.output, output_data)
    print(f"Tuning results saved to: {args.output}\n")
    return best_config


if __name__ == "__main__":
    main()
