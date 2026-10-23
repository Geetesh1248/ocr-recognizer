"""Load only explicitly trained checkpoints of our own model types."""
from pathlib import Path
import os
import tempfile
import torch


def save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path, kind):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing trained {kind} checkpoint: {path}. Train this model first; there is no external OCR fallback.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("kind") != kind or checkpoint.get("format_version") != 1 or checkpoint.get("epochs", 0) < 1:
        raise ValueError(f"Expected a trained {kind} checkpoint in format 1")
    return checkpoint
