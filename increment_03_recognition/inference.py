import torch
from checkpoints import load
from common import image_read
from .model import TextRecognizer
from .data import line_tensor
from .decoder import greedy_ctc


class Recognizer:
    def __init__(self, checkpoint):
        payload = load(checkpoint, "recognizer")
        self.alphabet = payload["alphabet"]
        self.height, self.max_width = payload["height"], payload["max_width"]
        version = payload.get("model_version")
        if version is None:
            # Detect by state_dict keys: V1 has cnn.0.weight with 32 out_channels, V2 has 64
            sd = payload.get("state_dict", {})
            if "cnn.0.weight" in sd and sd["cnn.0.weight"].shape[0] == 32:
                version = 1
            else:
                version = 2
        from .model import build_recognizer
        self.model = build_recognizer(len(self.alphabet)+1, version=version)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, regions):
        lines = []
        for region in regions:
            item = dict(region)
            try:
                image, steps = line_tensor(image_read(region["crop"]), self.height, self.max_width)
                probabilities = self.model(image[None], torch.tensor([steps])).softmax(-1)[:, 0].numpy()
                item["text"], item["confidence"] = greedy_ctc(probabilities, self.alphabet)
                item["status"] = "completed" if item["text"] else "empty"
            except Exception as error:
                item.update(text="", confidence=None, status="failed", error=f"{type(error).__name__}: {error}")
            lines.append(item)
        return lines
