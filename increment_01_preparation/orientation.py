"""Optional four-class orientation network, trained from labelled angles."""
import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from common import image_read, manifest, check_splits, seed_all, save_json, sha256
from checkpoints import load, save


class OrientationCNN(nn.Module):
    def __init__(self, classes=4):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)))
        self.classes = nn.Linear(64*4*4, classes)

    def forward(self, images):
        return self.classes(self.features(images).flatten(1))


def tensor(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(128/gray.shape[0], 128/gray.shape[1])
    resized = cv2.resize(gray, (max(1, round(gray.shape[1]*scale)), max(1, round(gray.shape[0]*scale))))
    canvas = np.full((128, 128), 255, np.uint8)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    return torch.from_numpy(canvas.copy()).float()[None]/127.5-1


class Orientation:
    def __init__(self, checkpoint):
        payload = load(checkpoint, "orientation")
        self.model = OrientationCNN()
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, image):
        scores = self.model(tensor(image)[None]).softmax(-1)[0]
        index = int(scores.argmax())
        return index*90, float(scores[index])


class OrientationData(Dataset):
    def __init__(self, records, line_mode=False):
        self.records, self.line_mode = records, line_mode

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        angle = record["correction_clockwise"]
        allowed = {0, 180} if self.line_mode else {0, 90, 180, 270}
        if not isinstance(angle, int) or isinstance(angle, bool) or angle not in allowed:
            raise ValueError(f"Invalid clockwise correction label; expected {sorted(allowed)}")
        return tensor(image_read(record["image"])), angle//(180 if self.line_mode else 90)


def train(train_path, val_path, output, epochs=15, batch_size=8, seed=42, device=None, kind="orientation"):
    if epochs < 1 or batch_size < 1:
        raise ValueError("Invalid orientation training settings")
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}", flush=True)
    records, validation = manifest(train_path), manifest(val_path)
    check_splits(records, validation)
    seed_all(seed)
    if kind not in {"orientation", "line_orientation"}:
        raise ValueError("Unknown orientation checkpoint kind")
    line_mode = kind == "line_orientation"
    model = OrientationCNN(2 if line_mode else 4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    loaders = [DataLoader(OrientationData(records, line_mode), batch_size=batch_size, shuffle=True), DataLoader(OrientationData(validation, line_mode), batch_size=batch_size)]
    best, history = -1., []
    for epoch in range(1, epochs+1):
        counts = []
        for training, loader in zip([True, False], loaders):
            model.train(training)
            correct, count = 0, 0
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                with torch.set_grad_enabled(training):
                    logits = model(images)
                    loss = nn.functional.cross_entropy(logits, labels)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite orientation loss")
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                correct += int((logits.argmax(-1) == labels).sum())
                count += len(images)
            counts.append(correct/count)
        history.append({"epoch": epoch, "train_accuracy": counts[0], "validation_accuracy": counts[1]})
        print(history[-1], flush=True)
        if counts[1] > best:
            best = counts[1]
            save(output, {"format_version": 1, "kind": kind, "epochs": epoch,
                          "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "seed": seed,
                          "validation_accuracy": best, "train_manifest_sha256": sha256(train_path), "val_manifest_sha256": sha256(val_path)})
        save_json(str(output)+".history.json", history)
    return history
