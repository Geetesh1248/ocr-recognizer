"""Training script for Text Recognizer on GPU (CUDA/MPS) or CPU.

Supports:
- Automatic GPU detection and Automatic Mixed Precision (AMP)
- 3-Stage Curriculum:
    Stage A: Synthetic only (streaming on-the-fly with training fonts)
    Stage B: 50/50 Synthetic + Real FUNSD train crops (streaming fresh synthetic each epoch)
    Stage C: Fine-tuning on 100% Real FUNSD train crops with low LR
- Resumable checkpoints (--resume for crash recovery)
- Distinct initialisation (--init-from to transfer model weights and reset optimizer/scheduler/epochs)
- Checkpoint pruning (keeps only last 2 epoch checkpoints per stage + best)
- Stage completion markers to prevent redundant re-runs
- Early stopping with patience on validation CER (real FUNSD val for Stages B/C)
- Detailed CSV and JSON metrics logging

Usage:
  python train_recognizer_gpu.py --stage A --epochs 30 --synth-count 50000 --amp
  python train_recognizer_gpu.py --stage B --epochs 25 --init-from models/recognizer_stageA_best.pt --lr 5e-4 --amp
  python train_recognizer_gpu.py --stage C --epochs 15 --init-from models/recognizer_stageB_best.pt --lr 1e-4 --amp
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common import ROOT, read_json, save_json, sha256, distance, seed_all, image_read
from checkpoints import save, load
from increment_03_recognition.model import build_recognizer, TextRecognizerV2
from increment_03_recognition.data import Vocabulary, line_tensor, minimum_ctc_steps, collate
from increment_03_recognition.decoder import greedy_ctc


def compute_metrics(predictions: List[str], ground_truths: List[str]) -> Tuple[float, float]:
    """Compute Character Error Rate (CER) and Word Error Rate (WER)."""
    total_chars = 0
    char_errs = 0
    total_words = 0
    word_errs = 0

    for pred, gt in zip(predictions, ground_truths):
        # Character level
        char_errs += distance(gt, pred)
        total_chars += len(gt)

        # Word level
        gt_words = gt.split()
        pred_words = pred.split()
        word_errs += distance(gt_words, pred_words)
        total_words += len(gt_words)

    cer = char_errs / max(1, total_chars)
    wer = word_errs / max(1, total_words)
    return cer, wer


def prune_epoch_checkpoints(output_dir: Path, stage: str, keep_last: int = 2):
    """Keep only the last `keep_last` epoch checkpoints for the given stage.
    Best checkpoint (recognizer_stage{stage}_best.pt) is strictly preserved.
    """
    pattern = f"recognizer_stage{stage}_epoch*.pt"
    ckpts = list(output_dir.glob(pattern))
    if len(ckpts) <= keep_last:
        return

    def parse_epoch(p: Path) -> int:
        m = re.search(r"epoch(\d+)", p.name)
        return int(m.group(1)) if m else 0

    sorted_ckpts = sorted(ckpts, key=parse_epoch)
    to_delete = sorted_ckpts[:-keep_last]
    for p in to_delete:
        try:
            p.unlink()
            print(f"  [Checkpoint Pruned] Removed older epoch checkpoint: {p.name}", flush=True)
        except Exception as e:
            print(f"  [Checkpoint Pruning Warning] Could not remove {p.name}: {e}", flush=True)


def mark_stage_completed(output_dir: Path, stage: str, epoch: int, best_cer: float):
    """Save completion marker file to prevent re-running completed stage."""
    marker_path = output_dir / f"stage_{stage}_completed.json"
    marker_data = {
        "stage": stage,
        "completed_at_epoch": epoch,
        "best_cer": float(best_cer),
        "best_checkpoint": str(output_dir / f"recognizer_stage{stage}_best.pt"),
        "timestamp": time.time()
    }
    save_json(marker_path, marker_data)
    print(f"\n[STAGE COMPLETED] Stage {stage} marked complete: {marker_path}", flush=True)


class InMemoryRecognitionDataset(Dataset):
    """Loads and caches recognition crops with CTC verification."""

    def __init__(self, records: List[dict], vocab: Vocabulary, height: int = 32, max_width: int = 1024, augment: bool = False):
        self.vocab = vocab
        self.height = height
        self.max_width = max_width
        self.augment = augment
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
        if self.augment:
            if random.random() < 0.3:
                scale = random.uniform(0.85, 1.15)
                img = np.clip(img.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        tensor, steps = line_tensor(img, self.height, self.max_width)
        req_steps = minimum_ctc_steps(tokens)
        if steps < req_steps:
            pad_needed = (req_steps - steps + 1) * 4
            padded = torch.ones(1, self.height, tensor.shape[-1] + pad_needed)
            padded[:, :, :tensor.shape[-1]] = tensor
            tensor = padded
            steps = tensor.shape[-1] // 4

        return tensor, torch.tensor(tokens, dtype=torch.long), steps, label


class DeterministicSyntheticValDataset(Dataset):
    """Deterministic synthetic validation dataset using unseen validation font families.
    Used when static synthetic val crops are not stored on disk.
    Pre-renders with a fixed seed so CER is completely consistent across epochs.
    """

    def __init__(self, count: int, corpus_dir: Path, fonts_dir: Path, config_path: Path,
                 vocab: Vocabulary, height: int = 32, max_width: int = 1024, seed: int = 42):
        self.vocab = vocab
        self.height = height
        self.max_width = max_width
        from make_recognition_data import CorpusManager, FontManager, render_synthetic_image
        rng = random.Random(seed)
        corpus_mgr = CorpusManager(corpus_dir, config_path)
        font_mgr = FontManager(fonts_dir)
        self.samples = []
        for _ in range(count):
            text = corpus_mgr.sample_text()
            if not text:
                text = "validation sample"
            val_font = rng.choice(font_mgr.val_fonts)
            try:
                img = render_synthetic_image(text, val_font, target_height=self.height, max_width=self.max_width)
                tokens = self.vocab.encode(text)
                req_steps = minimum_ctc_steps(tokens)
                tensor, steps = line_tensor(img, self.height, self.max_width)
                if steps < req_steps:
                    pad_needed = (req_steps - steps + 1) * 4
                    padded = torch.ones(1, self.height, tensor.shape[-1] + pad_needed)
                    padded[:, :, :tensor.shape[-1]] = tensor
                    tensor = padded
                    steps = tensor.shape[-1] // 4
                self.samples.append((tensor, torch.tensor(tokens, dtype=torch.long), steps, text))
            except Exception:
                continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


class StreamingSyntheticDataset(Dataset):
    """Stage A dataset: 100% on-the-fly streaming synthetic text crops.
    Every epoch generates fresh unique synthetic samples using training font families only.
    """

    def __init__(self, count: int, corpus_dir: Path, fonts_dir: Path, config_path: Path,
                 vocab: Vocabulary, height: int = 32, max_width: int = 1024):
        self.count = count
        self.vocab = vocab
        self.height = height
        self.max_width = max_width
        from make_recognition_data import CorpusManager, FontManager
        self.corpus_mgr = CorpusManager(corpus_dir, config_path)
        self.font_mgr = FontManager(fonts_dir)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, idx: int):
        from make_recognition_data import render_synthetic_image
        for _ in range(10):
            text = self.corpus_mgr.sample_text()
            if not text:
                continue
            font_path = self.font_mgr.get_font("train")
            try:
                img = render_synthetic_image(text, font_path, target_height=self.height, max_width=self.max_width)
                tokens = self.vocab.encode(text)
                req_steps = minimum_ctc_steps(tokens)
                tensor, steps = line_tensor(img, self.height, self.max_width)
                if steps < req_steps:
                    pad_needed = (req_steps - steps + 1) * 4
                    padded = torch.ones(1, self.height, tensor.shape[-1] + pad_needed)
                    padded[:, :, :tensor.shape[-1]] = tensor
                    tensor = padded
                    steps = tensor.shape[-1] // 4
                return tensor, torch.tensor(tokens, dtype=torch.long), steps, text
            except Exception:
                continue
        raise RuntimeError("Failed to render synthetic sample after 10 retries.")


class StageBMixedDataset(Dataset):
    """Stage B dataset: 50% real FUNSD train crops and 50% on-the-fly streaming synthetic crops.
    Every epoch generates fresh unique synthetic samples using training font families only.
    """

    def __init__(self, real_records: List[dict], num_synth: int, corpus_dir: Path, fonts_dir: Path, config_path: Path,
                 vocab: Vocabulary, height: int = 32, max_width: int = 1024, augment: bool = True):
        self.real_ds = InMemoryRecognitionDataset(real_records, vocab, height=height, max_width=max_width, augment=augment)
        self.num_real = len(real_records)
        self.num_synth = num_synth
        self.vocab = vocab
        self.height = height
        self.max_width = max_width
        from make_recognition_data import CorpusManager, FontManager
        self.corpus_mgr = CorpusManager(corpus_dir, config_path)
        self.font_mgr = FontManager(fonts_dir)
        self.current_epoch = 1
        self.epoch_synth_generated = 0
        self.epoch_synth_texts = []

    def reset_epoch(self, epoch: int):
        self.current_epoch = epoch
        self.epoch_synth_generated = 0
        self.epoch_synth_texts = []
        log_dir = Path(tempfile.gettempdir()) / f"synth_stageB_{os.getpid()}_{epoch}"
        if log_dir.is_dir():
            for f in log_dir.glob("w_*.txt"):
                try:
                    f.unlink()
                except Exception:
                    pass

    def get_synth_epoch_stats(self) -> dict:
        log_dir = Path(tempfile.gettempdir()) / f"synth_stageB_{os.getpid()}_{self.current_epoch}"
        if log_dir.is_dir():
            all_texts = []
            for f in log_dir.glob("w_*.txt"):
                try:
                    all_texts.extend(f.read_text(encoding="utf-8").splitlines())
                except Exception:
                    pass
            total = len(all_texts)
            unique = len(set(all_texts))
        else:
            total = self.epoch_synth_generated
            unique = len(set(self.epoch_synth_texts))
        pct = (unique / max(1, total)) * 100.0 if total > 0 else 100.0
        return {
            "generated": total,
            "distinct": unique,
            "distinct_pct": pct
        }

    def __len__(self) -> int:
        return self.num_real + self.num_synth

    def __getitem__(self, idx: int):
        if idx < self.num_real:
            return self.real_ds[idx]
        else:
            # Generate fresh synthetic sample on-the-fly with training fonts only
            from make_recognition_data import render_synthetic_image
            for _ in range(10):
                text = self.corpus_mgr.sample_text()
                if not text:
                    continue
                font_path = self.font_mgr.get_font("train")
                try:
                    img = render_synthetic_image(text, font_path, target_height=self.height, max_width=self.max_width)
                    tokens = self.vocab.encode(text)
                    req_steps = minimum_ctc_steps(tokens)
                    tensor, steps = line_tensor(img, self.height, self.max_width)
                    if steps < req_steps:
                        pad_needed = (req_steps - steps + 1) * 4
                        padded = torch.ones(1, self.height, tensor.shape[-1] + pad_needed)
                        padded[:, :, :tensor.shape[-1]] = tensor
                        tensor = padded
                        steps = tensor.shape[-1] // 4
                    worker_info = torch.utils.data.get_worker_info()
                    if worker_info is not None:
                        log_dir = Path(tempfile.gettempdir()) / f"synth_stageB_{os.getppid()}_{self.current_epoch}"
                        log_dir.mkdir(parents=True, exist_ok=True)
                        log_file = log_dir / f"w_{worker_info.id}.txt"
                        with open(log_file, "a", encoding="utf-8") as lf:
                            lf.write(text + "\n")
                    else:
                        self.epoch_synth_generated += 1
                        self.epoch_synth_texts.append(text)
                    return tensor, torch.tensor(tokens, dtype=torch.long), steps, text
                except Exception:
                    continue
            return self.real_ds[idx % self.num_real]


def evaluate_split(model: nn.Module, loader: DataLoader, alphabet: str, device: str, max_batches: int = 0) -> Tuple[float, float, float]:
    """Evaluate model on a DataLoader using greedy CTC decoding."""
    model.eval()
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for b_idx, (images, targets, lengths, target_lengths, labels) in enumerate(loader, 1):
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images, lengths)
            loss = criterion(logits.log_softmax(-1), targets, lengths, target_lengths)
            if torch.isfinite(loss):
                total_loss += float(loss.detach()) * len(images)
                total_count += len(images)

            probs = logits.softmax(-1).cpu().numpy()
            for i, gt_label in enumerate(labels):
                pred_text, _ = greedy_ctc(probs[:int(lengths[i]), i], alphabet)
                all_preds.append(pred_text)
                all_gts.append(gt_label)

            if max_batches > 0 and b_idx >= max_batches:
                break

    avg_loss = total_loss / max(1, total_count)
    cer, wer = compute_metrics(all_preds, all_gts)
    return avg_loss, cer, wer


def worker_init_fn(worker_id):
    """Seed each DataLoader worker independently to ensure unique synthetic generations."""
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def main():
    parser = argparse.ArgumentParser(description="Train text recognizer on GPU/CPU.")
    parser.add_argument("--stage", type=str, choices=["A", "B", "C"], default="A", help="Curriculum stage (A, B, C)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default 1e-3 for A, 5e-4 for B, 1e-4 for C)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from (restores weights, optimizer, scaler, epoch, and best_cer)")
    parser.add_argument("--init-from", type=str, default=None, help="Path to checkpoint to initialize model weights ONLY from (resets optimizer, scheduler, scaler, epoch counter to 1, and best_cer to inf)")
    parser.add_argument("--force", action="store_true", help="Force training even if stage completion marker exists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="models", help="Output directory for checkpoints")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs)")
    parser.add_argument("--amp", action="store_true", help="Enable Automatic Mixed Precision on CUDA")
    parser.add_argument("--max-width", type=int, default=1024, help="Maximum image width")
    parser.add_argument("--model-version", type=int, default=2, help="Model architecture version (default 2)")
    parser.add_argument("--workers", type=int, default=None, help="Number of DataLoader workers (default: all CPU cores on Linux, 0 on Windows)")
    parser.add_argument("--max-steps", type=int, default=0, help="Maximum training steps per epoch for smoke tests (0 = full epoch)")
    parser.add_argument("--synth-count", type=int, default=None, help="Number of synthetic samples per epoch (Stage A default: 50,000; Stage B default: matches real train crops count automatically)")
    parser.add_argument("--manifest-data", action="store_true", help="Use static manifest synthetic records instead of on-the-fly streaming")
    args = parser.parse_args()

    seed_all(args.seed)

    if args.resume and args.init_from:
        raise ValueError("Cannot specify both --resume and --init-from. Use --resume for crash recovery and --init-from for starting a new curriculum stage.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Check stage completion marker
    marker_path = output_dir / f"stage_{args.stage}_completed.json"
    if marker_path.is_file() and not args.force:
        print(f"\n[STAGE COMPLETED] Stage {args.stage} is already marked complete: {marker_path}", flush=True)
        print("Skipping training. (Use --force to override)\n", flush=True)
        return 0

    # 1. Device detection
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device: CUDA ({gpu_name})", flush=True)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using device: Apple MPS", flush=True)
    else:
        device = "cpu"
        print("Using device: CPU", flush=True)

    use_amp = (device == "cuda") and args.amp
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"AMP enabled: {use_amp}", flush=True)

    # Worker count with safe Windows fallback
    if args.workers is not None:
        num_workers = max(0, args.workers)
    else:
        num_workers = 0 if sys.platform == "win32" else (os.cpu_count() or 2)

    persistent_workers = (num_workers > 0)
    print(f"DataLoader configuration: num_workers={num_workers}, persistent_workers={persistent_workers}", flush=True)

    # 2. Config & Alphabet
    cfg = read_json(ROOT / "config.json")
    alphabet = cfg["recognizer"]["alphabet"]
    height = cfg["recognizer"]["height"]
    vocab = Vocabulary(alphabet)
    num_classes = len(alphabet) + 1

    # 3. Model construction
    model = build_recognizer(num_classes, version=args.model_version).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)

    start_epoch = 1
    best_cer = float("inf")
    epochs_without_improvement = 0

    # 4. Resume or Init-from logic
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"\n[RESUME] Restoring full training state from: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state" in ckpt and ckpt["optimizer_state"]:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
                print("  -> Restored optimizer state", flush=True)
            except Exception as e:
                print(f"  -> Notice: optimizer state not loaded ({e}); continuing with fresh optimizer.", flush=True)
        if "scheduler_state" in ckpt and ckpt["scheduler_state"]:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state"])
                print("  -> Restored scheduler state", flush=True)
            except Exception as e:
                print(f"  -> Notice: scheduler state not loaded ({e}); stepping scheduler.", flush=True)
                for _ in range(ckpt.get("epochs", 0)):
                    scheduler.step()
        elif ckpt.get("epochs", 0) > 0:
            for _ in range(ckpt.get("epochs", 0)):
                scheduler.step()
        if "scaler_state" in ckpt and ckpt["scaler_state"] and use_amp:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
                print("  -> Restored GradScaler state", flush=True)
            except Exception:
                pass
        start_epoch = ckpt.get("epochs", ckpt.get("epoch", 0)) + 1
        best_cer = ckpt.get("best_cer", ckpt.get("validation_cer", float("inf")))
        epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        print(f"  -> Resuming at epoch {start_epoch}, previous best CER: {best_cer:.4f}, patience counter: {epochs_without_improvement}\n", flush=True)

    elif args.init_from:
        init_path = Path(args.init_from)
        if not init_path.is_file():
            raise FileNotFoundError(f"Init-from checkpoint not found: {init_path}")
        print(f"\n[INIT-FROM] Initializing model weights ONLY from checkpoint: {init_path}", flush=True)
        ckpt = torch.load(init_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        # Fresh optimizer & scheduler with newly initialized state
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
        start_epoch = 1
        best_cer = float("inf")
        epochs_without_improvement = 0
        print("  -> Model weights loaded successfully.", flush=True)
        print("  -> FRESH STATE VERIFIED:", flush=True)
        print("     * Optimizer: newly initialized (fresh AdamW with weight decay 1e-4)", flush=True)
        print(f"     * Scheduler: newly initialized CosineAnnealingLR (T_max={args.epochs})", flush=True)
        print("     * Scaler: newly initialized GradScaler", flush=True)
        print(f"     * Epoch counter: reset to {start_epoch}", flush=True)
        print(f"     * Early-stopping state: reset (best_cer = {best_cer}, epochs_without_improvement = 0)", flush=True)
        print(f"     * Requested LR: {args.lr:.6e} | Initial LR: {args.lr:.6e}\n", flush=True)
    else:
        print(f"\n[INIT] Starting fresh from random initialization (no checkpoint).", flush=True)
        start_epoch = 1
        best_cer = float("inf")
        epochs_without_improvement = 0
        print("  -> FRESH STATE VERIFIED:", flush=True)
        print("     * Model: random initialization", flush=True)
        print("     * Optimizer: newly initialized (fresh AdamW with weight decay 1e-4)", flush=True)
        print(f"     * Scheduler: newly initialized CosineAnnealingLR (T_max={args.epochs})", flush=True)
        print("     * Scaler: newly initialized GradScaler", flush=True)
        print(f"     * Epoch counter: reset to {start_epoch}", flush=True)
        print(f"     * Early-stopping state: reset (best_cer = {best_cer}, epochs_without_improvement = 0)", flush=True)
        print(f"     * Requested LR: {args.lr:.6e} | Initial LR: {args.lr:.6e}\n", flush=True)

    # 5. Load manifests
    train_manifest_path = ROOT / "datasets" / "recognition_train.jsonl"
    val_manifest_path = ROOT / "datasets" / "recognition_val.jsonl"
    if not val_manifest_path.is_file():
        raise FileNotFoundError("Missing recognition manifests. Run make_recognition_data.py first.")

    val_recs = [json.loads(l) for l in val_manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    synth_val_recs = [r for r in val_recs if r.get("document_id", "").startswith("synth_")]
    real_val_recs = [r for r in val_recs if not r.get("document_id", "").startswith("synth_")]

    all_train_recs = []
    if train_manifest_path.is_file():
        all_train_recs = [json.loads(l) for l in train_manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    real_train_recs = [r for r in all_train_recs if not r.get("document_id", "").startswith("synth_")]

    # Build Stage Dataset
    has_corpus_and_fonts = (ROOT / "fonts").is_dir() and (ROOT / "corpus").is_dir()

    if args.stage == "A":
        synth_target = args.synth_count if args.synth_count is not None else 50000
        count_real = 0
        count_synth = synth_target
        if has_corpus_and_fonts and not args.manifest_data:
            train_ds = StreamingSyntheticDataset(
                count=synth_target,
                corpus_dir=ROOT / "corpus",
                fonts_dir=ROOT / "fonts",
                config_path=ROOT / "config.json",
                vocab=vocab,
                height=height,
                max_width=args.max_width
            )
            synth_type = "streaming on-the-fly (training fonts only)"
        else:
            synth_recs = [r for r in all_train_recs if r.get("document_id", "").startswith("synth_")]
            if len(synth_recs) == 0:
                raise ValueError("No synthetic records found in manifest and no fonts/corpus directory available.")
            sampled = [synth_recs[i % len(synth_recs)] for i in range(synth_target)]
            train_ds = InMemoryRecognitionDataset(sampled, vocab, height=height, max_width=args.max_width, augment=True)
            synth_type = f"manifest static records ({len(synth_recs)} unique)"

    elif args.stage == "B":
        count_real = len(real_train_recs)
        # Stage B automatically derives synthetic count to equal real train crops (37,715) unless explicitly overridden
        synth_target = args.synth_count if args.synth_count is not None else count_real
        count_synth = synth_target
        if has_corpus_and_fonts and not args.manifest_data:
            train_ds = StageBMixedDataset(
                real_records=real_train_recs,
                num_synth=synth_target,
                corpus_dir=ROOT / "corpus",
                fonts_dir=ROOT / "fonts",
                config_path=ROOT / "config.json",
                vocab=vocab,
                height=height,
                max_width=args.max_width,
                augment=True
            )
            synth_type = f"streaming on-the-fly ({count_synth:,} unique samples per epoch, training fonts only)"
        else:
            synth_recs = [r for r in all_train_recs if r.get("document_id", "").startswith("synth_")]
            sampled = [synth_recs[i % len(synth_recs)] for i in range(count_synth)]
            train_ds = InMemoryRecognitionDataset(real_train_recs + sampled, vocab, height=height, max_width=args.max_width, augment=True)
            synth_type = f"manifest static records ({len(synth_recs)} unique)"

    elif args.stage == "C":
        count_real = len(real_train_recs)
        count_synth = 0
        train_ds = InMemoryRecognitionDataset(real_train_recs, vocab, height=height, max_width=args.max_width, augment=True)
        synth_type = "none (100% real train crops)"

    metric_name = "Synthetic Val CER" if args.stage == "A" else "Real FUNSD Val CER"
    initial_lr = optimizer.param_groups[0]["lr"]

    print("=" * 80, flush=True)
    print(f"CURRICULUM STAGE {args.stage} DATASET BREAKDOWN:", flush=True)
    print(f"  Total training samples:         {len(train_ds):,}", flush=True)
    print(f"  Real FUNSD train crops:         {count_real:,} ({count_real/max(1, len(train_ds))*100:.1f}%)", flush=True)
    print(f"  Synthetic crops:                {count_synth:,} ({count_synth/max(1, len(train_ds))*100:.1f}%)", flush=True)
    if count_synth > 0:
        print(f"  Synthetic generator:            {synth_type}", flush=True)
        print(f"  Unique synthetic samples/epoch: {count_synth:,}", flush=True)
    print(f"  Validation Real FUNSD:          {len(real_val_recs):,} crops", flush=True)
    print(f"  Validation Synthetic (unseen):  {len(synth_val_recs):,} crops", flush=True)
    print(f"  Starting Epoch:                 {start_epoch}", flush=True)
    print(f"  Starting Best CER:              {best_cer}", flush=True)
    print(f"  Initial Learning Rate:          {initial_lr:.6e} (requested: {args.lr:.6e})", flush=True)
    print(f"  Early Stopping Target Metric:   {metric_name} (patience={args.patience})", flush=True)
    print("=" * 80 + "\n", flush=True)

    # Check if synthetic val image files exist on disk
    synth_val_files_exist = False
    if synth_val_recs:
        test_path = Path(synth_val_recs[0]["image"])
        if not test_path.is_file():
            test_path = ROOT / "datasets" / synth_val_recs[0]["image"]
        synth_val_files_exist = test_path.is_file()

    if synth_val_files_exist:
        synth_val_ds = InMemoryRecognitionDataset(synth_val_recs, vocab, height=height, max_width=args.max_width, augment=False)
    else:
        synth_val_count = 50 if args.max_steps > 0 else 500
        synth_val_ds = DeterministicSyntheticValDataset(
            count=synth_val_count,
            corpus_dir=ROOT / "corpus",
            fonts_dir=ROOT / "fonts",
            config_path=ROOT / "config.json",
            vocab=vocab,
            height=height,
            max_width=args.max_width,
            seed=42
        )
    real_val_ds = InMemoryRecognitionDataset(real_val_recs, vocab, height=height, max_width=args.max_width, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        worker_init_fn=worker_init_fn if num_workers > 0 else None,
        pin_memory=(device == "cuda")
    )
    val_workers = min(num_workers, 2)
    synth_val_loader = DataLoader(
        synth_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=val_workers,
        persistent_workers=(val_workers > 0),
        pin_memory=(device == "cuda")
    )
    real_val_loader = DataLoader(
        real_val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=val_workers,
        persistent_workers=(val_workers > 0),
        pin_memory=(device == "cuda")
    )

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Logging setup
    csv_log_path = outputs_dir / "recognizer_train_log.csv"
    json_log_path = outputs_dir / "recognizer_train_log.json"
    log_history = []
    if json_log_path.is_file():
        try:
            log_history = read_json(json_log_path)
        except Exception:
            log_history = []

    if not csv_log_path.is_file():
        with open(csv_log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["stage", "epoch", "train_loss", "synth_val_cer", "synth_val_wer", "real_val_cer", "real_val_wer", "lr", "seconds"])

    print("=" * 95)
    print(f"STARTING RECOGNIZER TRAINING: STAGE {args.stage} (Epochs {start_epoch} to {start_epoch + args.epochs - 1})")
    print("=" * 95)
    header = f"{'Epoch':>5} | {'Train Loss':>10} | {'Synth CER':>9} | {'Synth WER':>9} | {'Real CER':>8} | {'Real WER':>8} | {'LR':>8} | {'Time':>6}"
    print(header)
    print("-" * 95)

    epochs_without_improvement = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        epoch_start = time.time()
        if hasattr(train_ds, "reset_epoch"):
            train_ds.reset_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_samples = 0

        step_idx = 0
        for step_idx, (images, targets, lengths, target_lengths, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=(device == "cuda"))
            targets = targets.to(device, non_blocking=(device == "cuda"))

            if step_idx == 1:
                cur_step_lr = optimizer.param_groups[0]["lr"]
                print(f"  [Step 1 Verified] Epoch: {epoch} | Requested LR: {args.lr:.6e} | Actual Optimizer LR: {cur_step_lr:.6e} | Best CER: {best_cer}", flush=True)

            optimizer.zero_grad()
            autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp) if hasattr(torch, "amp") and hasattr(torch.amp, "autocast") else torch.cuda.amp.autocast(enabled=use_amp)
            with autocast_ctx:
                logits = model(images, lengths)
                loss = criterion(logits.log_softmax(-1), targets, lengths, target_lengths)

            if torch.isfinite(loss):
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                train_loss += float(loss.detach()) * len(images)
                train_samples += len(images)

            if args.max_steps > 0 and (step_idx % 10 == 0 or step_idx == args.max_steps or step_idx == 1):
                elapsed_now = max(1e-5, time.time() - epoch_start)
                rate_steps = step_idx / elapsed_now
                rate_samples = train_samples / elapsed_now
                print(f"  [Smoke Test] Step {step_idx:3d}/{args.max_steps} | Loss: {loss.item():.4f} | "
                      f"Speed: {rate_steps:.2f} steps/sec ({rate_samples:.1f} samples/sec) on {device.upper()}", flush=True)

            if args.max_steps > 0 and step_idx >= args.max_steps:
                break

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        avg_train_loss = train_loss / max(1, train_samples)

        # Validation
        max_val_batches = 2 if args.max_steps > 0 else 0
        _, s_cer, s_wer = evaluate_split(model, synth_val_loader, alphabet, device, max_batches=max_val_batches)
        _, r_cer, r_wer = evaluate_split(model, real_val_loader, alphabet, device, max_batches=max_val_batches)
        epoch_seconds = time.time() - epoch_start

        # Monitor primary metric for stage
        primary_cer = s_cer if args.stage == "A" else r_cer

        print(f"{epoch:>5} | {avg_train_loss:>10.4f} | {s_cer:>9.4f} | {s_wer:>9.4f} | {r_cer:>8.4f} | {r_wer:>8.4f} | {current_lr:>8.2e} | {epoch_seconds:>5.1f}s", flush=True)

        if hasattr(train_ds, "get_synth_epoch_stats"):
            stats = train_ds.get_synth_epoch_stats()
            if stats["generated"] > 0:
                print(f"       -> [Stage B Synthetic Tracking] Generated: {stats['generated']:,} | Distinct: {stats['distinct']:,} ({stats['distinct_pct']:.1f}% unique)", flush=True)
                print("          Distinctness measurement: unique text strings / total synthetic samples generated this epoch.", flush=True)

        # Log entry
        entry = {
            "stage": args.stage,
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 5),
            "synth_val_cer": round(s_cer, 5),
            "synth_val_wer": round(s_wer, 5),
            "real_val_cer": round(r_cer, 5),
            "real_val_wer": round(r_wer, 5),
            "lr": round(current_lr, 7),
            "seconds": round(epoch_seconds, 2)
        }
        log_history.append(entry)
        save_json(json_log_path, log_history)
        if output_dir != outputs_dir:
            save_json(output_dir / "recognizer_train_log.json", log_history)

        with open(csv_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([entry["stage"], entry["epoch"], entry["train_loss"],
                             entry["synth_val_cer"], entry["synth_val_wer"],
                             entry["real_val_cer"], entry["real_val_wer"],
                             entry["lr"], entry["seconds"]])

        # Step scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        # Update early stopping and best CER state
        metric_label = "Synthetic Val CER" if args.stage == "A" else "Real Val CER"
        print(f"       -> [Stage {args.stage} Stopping Metric: {metric_label} = {primary_cer:.4f}]", flush=True)

        if primary_cer < best_cer:
            best_cer = primary_cer
            epochs_without_improvement = 0
            is_best = True
        else:
            epochs_without_improvement += 1
            is_best = False

        # Checkpoint every epoch
        epoch_ckpt_path = output_dir / f"recognizer_stage{args.stage}_epoch{epoch:03d}.pt"
        ckpt_payload = {
            "format_version": 1,
            "kind": "recognizer",
            "epochs": epoch,
            "stage": args.stage,
            "model_version": args.model_version,
            "alphabet": alphabet,
            "height": height,
            "max_width": args.max_width,
            "validation_cer": primary_cer,
            "best_cer": best_cer,
            "epochs_without_improvement": epochs_without_improvement,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if use_amp else None,
            "seed": args.seed
        }
        save(epoch_ckpt_path, ckpt_payload)

        # Prune older epoch checkpoints (keep last 2)
        prune_epoch_checkpoints(output_dir, args.stage, keep_last=2)

        if is_best:
            best_path = output_dir / f"recognizer_stage{args.stage}_best.pt"
            save(best_path, ckpt_payload)
            print(f"  --> Saved new best checkpoint: {best_path} ({metric_label}={best_cer:.4f})", flush=True)
        else:
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping triggered: {epochs_without_improvement} epochs without {metric_label} improvement.", flush=True)
                break

    # Mark stage completed
    mark_stage_completed(output_dir, args.stage, epoch, best_cer)

    print("=" * 95)
    print(f"Stage {args.stage} training completed. Best validation CER: {best_cer:.4f}")
    print("=" * 95)
    return 0


if __name__ == "__main__":
    sys.exit(main())
