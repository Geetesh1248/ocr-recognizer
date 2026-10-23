"""Our two-class crop-orientation model, separately trained from page orientation."""
import torch
from checkpoints import load
from increment_01_preparation.orientation import OrientationCNN, tensor


class LineOrientation:
    def __init__(self, checkpoint):
        payload = load(checkpoint, "line_orientation")
        self.model = OrientationCNN(classes=2)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()

    @torch.inference_mode()
    def __call__(self, image):
        scores = self.model(tensor(image)[None]).softmax(-1)[0]
        index = int(scores.argmax())
        return index*180, float(scores[index])
