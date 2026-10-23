from pathlib import Path
import torch
from torch.utils.data import DataLoader
from common import manifest, check_splits, seed_all, save_json, sha256
from checkpoints import save
from .model import TextDetector
from .data import DetectionDataset
from .losses import detection_loss


def train(train_path, val_path, output, epochs=20, size=512, batch_size=2, learning_rate=.001, seed=42, device=None):
    if epochs < 1 or size < 32 or size % 8 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("Invalid detector training settings")
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    records, validation = manifest(train_path), manifest(val_path)
    check_splits(records, validation)
    levels = {r.get('annotation_level', 'region') for r in records+validation}
    if len(levels) != 1:
        raise ValueError('Training and validation must use the same annotation level')
    annotation_level = levels.pop()
    seed_all(seed)
    model = TextDetector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loaders = [DataLoader(DetectionDataset(records, size, True), batch_size=batch_size, shuffle=True),
               DataLoader(DetectionDataset(validation, size), batch_size=batch_size)]
    best, history = float("inf"), []
    for epoch in range(1, epochs+1):
        totals = []
        for training, loader in zip([True, False], loaders):
            model.train(training)
            total, count = 0., 0
            for images, targets, valid in loader:
                images, targets, valid = images.to(device), targets.to(device), valid.to(device)
                with torch.set_grad_enabled(training):
                    loss = detection_loss(model(images), targets, valid)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite detector loss")
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    optimizer.step()
                total += float(loss.detach())*len(images)
                count += len(images)
            totals.append(total/count)
        history.append({"epoch": epoch, "train_loss": totals[0], "validation_loss": totals[1]})
        print(history[-1], flush=True)
        if totals[1] < best:
            best = totals[1]
            save(output, {"format_version": 1, "kind": "detector", "epochs": epoch,
                          "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "image_size": size, "validation_loss": best, "seed": seed,
                          "annotation_level": annotation_level,
                          "train_manifest_sha256": sha256(train_path), "val_manifest_sha256": sha256(val_path)})
        save_json(str(output)+".history.json", history)
    return history
