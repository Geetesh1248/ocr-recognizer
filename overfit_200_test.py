"""Step 1: Overfit test on 200 samples locally to verify the recognizer architecture,
CTC loss, gradient propagation, and decoding logic before GPU scaling.
"""
import json
import time
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import torch
from common import ROOT, read_json, image_read, distance, seed_all
from increment_03_recognition.model import TextRecognizerV2
from increment_03_recognition.data import Vocabulary, line_tensor, minimum_ctc_steps, collate
from increment_03_recognition.decoder import greedy_ctc

torch.set_num_threads(8)
seed_all(42)

cfg = read_json(ROOT / "config.json")
alphabet = cfg["recognizer"]["alphabet"]
vocab = Vocabulary(alphabet)

print("=" * 60, flush=True)
print("PHASE 3 - STEP 1: OVERFIT TEST ON 200 REAL SAMPLES", flush=True)
print("=" * 60, flush=True)

# Select 200 real word crops
records = [json.loads(l) for l in open(ROOT / "datasets" / "recognition_train.jsonl", encoding="utf-8") if l.strip()]
real_records = [r for r in records if not r["document_id"].startswith("synth_") and "word_" in r["image"]]

samples = []
for r in real_records:
    p = ROOT / "datasets" / r["image"]
    if not p.is_file():
        continue
    try:
        tokens = vocab.encode(r["text"])
        img = image_read(p)
        tensor, steps = line_tensor(img, 32, 112)
        if 2 <= len(tokens) <= 8 and steps >= minimum_ctc_steps(tokens) and steps <= 28:
            samples.append((tensor, torch.tensor(tokens, dtype=torch.long), steps, r["text"]))
            if len(samples) == 200:
                break
    except Exception:
        pass

print(f"Loaded and verified {len(samples)} real word crops into RAM.", flush=True)
assert len(samples) == 200, f"Expected 200 samples, got {len(samples)}"

# Sort slightly by length to minimize padding overhead in batches
samples.sort(key=lambda s: s[2])

class InMemoryDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        return self.data[i]

dataset = InMemoryDataset(samples)
batch_size = 25
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
eval_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

model = TextRecognizerV2(len(alphabet) + 1)
optimizer = torch.optim.AdamW(model.parameters(), lr=7e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=5e-4)
criterion = torch.nn.CTCLoss(blank=0, zero_infinity=True)

print(f"\n{'Epoch':>5} | {'Train Loss':>11} | {'CER':>8} | {'Elapsed':>8} | Sample Predictions", flush=True)
print("-" * 75, flush=True)

t0 = time.time()
for epoch in range(1, 31):
    model.train()
    total_loss = 0.0
    for images, targets, lengths, target_lengths, labels in loader:
        logits = model(images, lengths)
        loss = criterion(logits.log_softmax(-1), targets, lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(images)
    scheduler.step()
    avg_loss = total_loss / len(dataset)

    # Compute CER
    model.eval()
    char_errors = 0
    total_chars = 0
    sample_preds = []
    with torch.no_grad():
        for images, targets, lengths, target_lengths, labels in eval_loader:
            logits = model(images, lengths)
            probs = logits.softmax(-1).cpu().numpy()
            for i, label in enumerate(labels):
                pred, _ = greedy_ctc(probs[:int(lengths[i]), i], alphabet)
                char_errors += distance(label, pred)
                total_chars += len(label)
                if len(sample_preds) < 3:
                    sample_preds.append(f"'{label}'->'{pred}'")

    cer = char_errors / max(1, total_chars)
    elapsed = time.time() - t0
    sample_str = ", ".join(sample_preds)
    print(f"{epoch:>5} | {avg_loss:>11.4f} | {cer:>8.4f} | {elapsed:>7.1f}s | {sample_str}", flush=True)

    if cer < 0.05:
        print(f"\nSUCCESS: Overfit target achieved! CER={cer:.4f} < 0.05 at epoch {epoch}.", flush=True)
        break
