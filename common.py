"""Shared file/data utilities. They contain no OCR implementation or external calls."""
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
import unicodedata
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,:;!?'-\"()/\\[]{}+=%&@#_$*<>|"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024*1024), b""):
            digest.update(part)
    return digest.hexdigest()


def image_read(path):
    # imdecode/fromfile supports spaces and Unicode paths on Windows.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return image


def image_write(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix or ".png", image)
    if not success:
        raise ValueError(f"Cannot encode image: {path}")
    with open(path, "wb") as f:
        f.write(encoded.tobytes())


def manifest(path):
    path = Path(path).resolve()
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item.get("document_id"), str) or not item["document_id"]:
            raise ValueError(f"{path}:{number}: document_id required")
        image = (path.parent / item["image"]).resolve()
        if not image.is_file():
            raise ValueError(f"{path}:{number}: missing image {image}")
        item["image"] = str(image)
        records.append(item)
    if not records:
        raise ValueError("Dataset manifest is empty")
    return records


def check_splits(train, validation):
    if {r["document_id"] for r in train} & {r["document_id"] for r in validation}:
        raise ValueError("Training and validation document IDs overlap")
    if {sha256(r["image"]) for r in train} & {sha256(r["image"]) for r in validation}:
        raise ValueError("Identical image bytes occur in training and validation")


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]


def normal_text(text):
    return " ".join(unicodedata.normalize("NFC", text).split())
